"""
EG-GM-PUNet v3 Stage A.0 — Per-Scale Stochastic Intervention Diagnosis.

目的
----
不重新训练、不修改模型参数，只对已训练 v3 Stage A checkpoint 的 prior sampling
进行尺度级干预，回答：coarse / middle / fine 中哪一级 stochasticity 导致
fidelity 下降或 diversity 过度扩张，以及不同尺度之间是否存在交互放大。

核心原则
--------
1. Prediction 始终 prior-only；posterior 完全不参与预测。
2. 所有模型权重保持不变。
3. 某尺度关闭 stochasticity 时，不将 latent 置零，也不跳过该尺度；
   而是使用当前条件 GMM 的 mixture expectation E[z_g]。
4. 某尺度开启 stochasticity 时，从当前条件 GMM 精确采样。
5. 8 个 2^3 factorial interventions 使用同一批 common random numbers，
   尽量降低不同 intervention 之间由 Monte-Carlo 随机性带来的比较噪声。
6. component 只解释为 latent-density basis，不解释为 semantic mode / annotator。

默认 8 个 intervention
---------------------
000 mean_all              : coarse mean, middle mean, fine mean
100 sample_coarse         : coarse sample only
010 sample_middle         : middle sample only
001 sample_fine           : fine sample only
110 sample_coarse_middle  : coarse + middle sample
101 sample_coarse_fine    : coarse + fine sample（补全 2^3 factorial）
011 sample_middle_fine    : middle + fine sample
111 sample_all            : coarse + middle + fine sample

输出
----
- intervention_per_case_metrics.csv
- intervention_per_patient_metrics.csv
- factorial_effects_per_case.csv
- conditional_full_context_effects.csv
- stagea0_intervention_summary.json

主要判据
--------
优先看 num_samples=16：
- sample_diversity_distance：哪个尺度/交互显著增加 dispersion；
- cross_dice：哪个尺度/交互损害 average fidelity；
- ged2：哪个尺度/交互使 distribution matching 恶化；
- best_sample_dice_to_raters：额外 stochasticity 是否至少带来 useful coverage。

注意
----
sample_all 使用 common-random-number 手工 exact sampler，因此其 Monte-Carlo
实现与正式 evaluator 的 torch.multinomial/random stream 不完全相同；总体均值应接近，
但不要求逐病例/逐样本与原 summary.json 完全一致。Stage A.0 的核心是同一随机 tape
下 interventions 之间的相对差异。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from itertools import islice, product
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "evaluate" else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.lidc_dataset_full import build_lidc_full_dataloader
from models.EG_GM_PUNet_v3 import EGGMPUNetV3
from models.Hierarchical_Multi_scale_Gaussian_Mixture_prob_unet.hierarchical_multiscale_gmm_parts import (
    mixture_expected_mean,
)
from utils.gmm_experiment_utils import prepare_device, set_seed
from utils.gmm_prior_evaluation import (
    evaluate_single_case,
    normalize_sample_counts,
    summarize_rows,
    write_csv,
)


GROUPS = ("coarse", "middle", "fine")
INTERVENTIONS: dict[str, tuple[int, int, int]] = {
    "mean_all": (0, 0, 0),
    "sample_coarse": (1, 0, 0),
    "sample_middle": (0, 1, 0),
    "sample_fine": (0, 0, 1),
    "sample_coarse_middle": (1, 1, 0),
    "sample_coarse_fine": (1, 0, 1),
    "sample_middle_fine": (0, 1, 1),
    "sample_all": (1, 1, 1),
}
BITS_TO_NAME = {bits: name for name, bits in INTERVENTIONS.items()}
METRICS = (
    "ged2",
    "cross_dice",
    "best_sample_dice_to_raters",
    "sample_diversity_distance",
    "mean_probability_mask_dice",
)


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ValueError("YAML 必须为 dict。")
    return value


def build_model(cfg: Mapping[str, Any]) -> EGGMPUNetV3:
    m = cfg["model"]
    return EGGMPUNetV3(
        in_channels=int(m["in_channels"]),
        num_classes=int(m["num_classes"]),
        num_components={str(k): int(v) for k, v in m["num_components"].items()},
        latent_dims={str(k): int(v) for k, v in m["latent_dims"].items()},
        num_channels=tuple(int(v) for v in m["num_channels"]),
        group_feature_indices={str(k): int(v) for k, v in m["group_feature_indices"].items()},
        group_decoder_indices={str(k): int(v) for k, v in m["group_decoder_indices"].items()},
        num_convs_per_block=int(m["num_convs_per_block"]),
        head_hidden_dim=int(m["head_hidden_dim"]),
        fusion_1x1_convs=int(m["fusion_1x1_convs"]),
        segmentation_head_1x1_convs=int(m["segmentation_head_1x1_convs"]),
        min_log_sigma=m.get("min_log_sigma"),
        max_log_sigma=m.get("max_log_sigma"),
        mixture_temperature=float(m["mixture_temperature"]),
        evidence_projection_channels=int(m["evidence_projection_channels"]),
        evidence_fusion_channels=int(m["evidence_fusion_channels"]),
        feedback_config=m["feedback"],
    )


def case_ids_from_batch(value, batch_size: int, fallback: int, split: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if torch.is_tensor(value):
        x = value.detach().cpu().tolist()
        if not isinstance(x, list):
            x = [x]
        return [str(v) for v in x]
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [f"{split}_{fallback+i:06d}" for i in range(batch_size)]


def values_from_batch(value, batch_size: int, default: str = "") -> list[str]:
    if isinstance(value, str):
        return [value] * batch_size
    if torch.is_tensor(value):
        x = value.detach().cpu().tolist()
        if not isinstance(x, list):
            x = [x]
        return [str(v) for v in x]
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [default] * batch_size


def patient_from_case(case_id: str) -> str:
    m = re.search(r"(LIDC-IDRI-\d{4})", str(case_id))
    return m.group(1) if m else str(case_id).split("_")[0]


def annotation_diversity_iou(masks: torch.Tensor, eps: float = 1e-7) -> float:
    x = masks.detach().float()
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    vals = []
    for i in range(x.shape[0]):
        a = x[i].reshape(-1)
        for j in range(i + 1, x.shape[0]):
            b = x[j].reshape(-1)
            inter = (a * b).sum()
            union = ((a + b) > 0).float().sum()
            iou = torch.tensor(1.0, device=x.device) if union <= eps else inter / union.clamp_min(eps)
            vals.append(1.0 - iou)
    return float(torch.stack(vals).mean().item()) if vals else 0.0


def _make_random_tape(
    *,
    num_samples: int,
    batch_size: int,
    latent_dims: Mapping[str, int],
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, dict[str, torch.Tensor]]:
    """为同一 batch 的所有 interventions 生成 common random numbers。"""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    tape: dict[str, dict[str, torch.Tensor]] = {}
    for group in GROUPS:
        d = int(latent_dims[group])
        u = torch.rand((num_samples, batch_size), generator=gen, dtype=torch.float32)
        eps = torch.randn((num_samples, batch_size, d), generator=gen, dtype=torch.float32)
        tape[group] = {
            "u": u.to(device=device, dtype=dtype),
            "eps": eps.to(device=device, dtype=dtype),
        }
    return tape


def _sample_exact_with_tape(distribution, u: torch.Tensor, eps: torch.Tensor):
    """用指定 uniform + Gaussian noise 从当前条件 GMM 精确采样。"""
    probs = distribution.mixture_probs
    if u.ndim != 1 or u.shape[0] != probs.shape[0]:
        raise ValueError("u 必须为 [B]。")
    if eps.ndim != 2 or eps.shape[0] != probs.shape[0] or eps.shape[1] != distribution.mu.shape[2]:
        raise ValueError("eps 必须为 [B,D]。")

    # inverse-CDF categorical sample；u in [0,1)。
    cdf = probs.cumsum(dim=1)
    idx = (u.unsqueeze(1) > cdf).sum(dim=1).clamp_max(probs.shape[1] - 1)

    b, _, d = distribution.mu.shape
    gather = idx.view(b, 1, 1).expand(b, 1, d)
    mu = distribution.mu.gather(1, gather).squeeze(1)
    sigma = distribution.sigma.gather(1, gather).squeeze(1)
    z = mu + sigma * eps
    return z, idx


@torch.no_grad()
def intervention_logits(
    *,
    model: EGGMPUNetV3,
    state,
    bits: tuple[int, int, int],
    random_tape: Mapping[str, Mapping[str, torch.Tensor]],
    num_samples: int,
) -> torch.Tensor:
    """返回 [S,B,C,H,W] probabilistic logits。"""
    active = {group: bool(bits[i]) for i, group in enumerate(GROUPS)}
    outputs = []

    for s in range(int(num_samples)):
        previous_z: dict[str, torch.Tensor] = {}
        indices: dict[str, torch.Tensor] = {}
        distributions = {}

        for group in GROUPS:
            p = model.probabilistic_model.prior.distribution(
                features=state.prior_context_features,
                group_name=group,
                previous_z=previous_z,
            )
            distributions[group] = p

            if active[group]:
                z, idx = _sample_exact_with_tape(
                    p,
                    random_tape[group]["u"][s],
                    random_tape[group]["eps"][s],
                )
            else:
                z = mixture_expected_mean(p)
                idx = torch.full(
                    (z.shape[0],), -1,
                    device=z.device,
                    dtype=torch.long,
                )

            # 无论当前尺度是 sample 还是 expectation，后续尺度都条件于这个实际 z。
            previous_z[group] = z
            indices[group] = idx

        decode = model.probabilistic_model.decode_from_groups(
            state=state,
            z_groups=previous_z,
            component_indices=indices,
            distributions=distributions,
        )
        outputs.append(decode.logits)

    return torch.stack(outputs, dim=0)


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # union of keys, stable first-seen order
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_patient(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["patient_id"]), str(row["intervention"]), int(row["num_samples"]))
        groups.setdefault(key, []).append(row)

    out = []
    for (patient, intervention, count), items in sorted(groups.items()):
        result: dict[str, Any] = {
            "patient_id": patient,
            "intervention": intervention,
            "num_samples": count,
            "num_cases": len(items),
            "sample_coarse": int(items[0]["sample_coarse"]),
            "sample_middle": int(items[0]["sample_middle"]),
            "sample_fine": int(items[0]["sample_fine"]),
        }
        for metric in METRICS + ("annotation_diversity_distance",):
            vals = []
            for item in items:
                try:
                    v = float(item[metric])
                    if math.isfinite(v):
                        vals.append(v)
                except Exception:
                    pass
            if vals:
                result[metric] = float(np.mean(vals))
        out.append(result)
    return out


def _metric_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, int, tuple[int, int, int]], dict[str, Any]]:
    out = {}
    for row in rows:
        bits = (
            int(row["sample_coarse"]),
            int(row["sample_middle"]),
            int(row["sample_fine"]),
        )
        out[(str(row["case_id"]), int(row["num_samples"]), bits)] = row
    return out


def _factorial_effects(rows: list[dict[str, Any]], sample_counts: list[int]):
    """逐病例计算 2^3 factorial effects 与 full-context removal effects。"""
    lookup = _metric_lookup(rows)
    case_ids = sorted({str(r["case_id"]) for r in rows})
    patient_by_case = {str(r["case_id"]): str(r["patient_id"]) for r in rows}

    factorial_rows = []
    conditional_rows = []

    for case_id in case_ids:
        for count in sample_counts:
            # 只有 8 个 cell 全齐才计算。
            cells = {bits: lookup.get((case_id, int(count), bits)) for bits in product((0, 1), repeat=3)}
            if any(v is None for v in cells.values()):
                continue

            frow: dict[str, Any] = {
                "case_id": case_id,
                "patient_id": patient_by_case[case_id],
                "num_samples": int(count),
            }
            crow: dict[str, Any] = dict(frow)

            for metric in METRICS:
                y = {bits: float(cells[bits][metric]) for bits in cells}

                # averaged factorial main effects: E[y|g=1]-E[y|g=0]
                c_main = np.mean([y[(1, m, f)] - y[(0, m, f)] for m, f in product((0, 1), repeat=2)])
                m_main = np.mean([y[(c, 1, f)] - y[(c, 0, f)] for c, f in product((0, 1), repeat=2)])
                f_main = np.mean([y[(c, m, 1)] - y[(c, m, 0)] for c, m in product((0, 1), repeat=2)])

                cm = np.mean([
                    y[(1, 1, f)] - y[(1, 0, f)] - y[(0, 1, f)] + y[(0, 0, f)]
                    for f in (0, 1)
                ])
                cf = np.mean([
                    y[(1, m, 1)] - y[(1, m, 0)] - y[(0, m, 1)] + y[(0, m, 0)]
                    for m in (0, 1)
                ])
                mf = np.mean([
                    y[(c, 1, 1)] - y[(c, 1, 0)] - y[(c, 0, 1)] + y[(c, 0, 0)]
                    for c in (0, 1)
                ])
                three = (
                    y[(1, 1, 1)] - y[(1, 1, 0)] - y[(1, 0, 1)] - y[(0, 1, 1)]
                    + y[(1, 0, 0)] + y[(0, 1, 0)] + y[(0, 0, 1)] - y[(0, 0, 0)]
                )

                frow[f"{metric}_main_coarse"] = float(c_main)
                frow[f"{metric}_main_middle"] = float(m_main)
                frow[f"{metric}_main_fine"] = float(f_main)
                frow[f"{metric}_interaction_coarse_middle"] = float(cm)
                frow[f"{metric}_interaction_coarse_fine"] = float(cf)
                frow[f"{metric}_interaction_middle_fine"] = float(mf)
                frow[f"{metric}_interaction_three_way"] = float(three)

                # Full-context effect of adding each stochastic scale:
                # full(111) - same context with that scale made deterministic.
                crow[f"{metric}_add_coarse_given_middle_fine"] = float(y[(1, 1, 1)] - y[(0, 1, 1)])
                crow[f"{metric}_add_middle_given_coarse_fine"] = float(y[(1, 1, 1)] - y[(1, 0, 1)])
                crow[f"{metric}_add_fine_given_coarse_middle"] = float(y[(1, 1, 1)] - y[(1, 1, 0)])

            factorial_rows.append(frow)
            conditional_rows.append(crow)

    return factorial_rows, conditional_rows


def _summarize_effect_rows(rows: list[dict[str, Any]], sample_count: int) -> dict[str, Any]:
    selected = [r for r in rows if int(r["num_samples"]) == int(sample_count)]
    if not selected:
        return {}
    summary = {}
    keys = [k for k in selected[0] if k not in {"case_id", "patient_id", "num_samples"}]
    for key in keys:
        vals = np.asarray([float(r[key]) for r in selected], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            summary[key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "median": float(np.median(vals)),
                "q25": float(np.quantile(vals, 0.25)),
                "q75": float(np.quantile(vals, 0.75)),
            }
    return summary


def _group_intervention_summary(rows: list[dict[str, Any]], sample_counts: list[int]) -> dict[str, Any]:
    result = {}
    for name, bits in INTERVENTIONS.items():
        subset = [r for r in rows if str(r["intervention"]) == name]
        result[name] = {
            "bits": {"coarse": bits[0], "middle": bits[1], "fine": bits[2]},
            "metrics": summarize_rows(subset, sample_counts),
        }
    return result


def _diagnostic_ranking(conditional_summary: dict[str, Any]) -> dict[str, Any]:
    """仅做描述性排序，不自动宣判病因。"""
    if not conditional_summary:
        return {}

    def mean(key: str) -> float:
        return float(conditional_summary.get(key, {}).get("mean", float("nan")))

    diversity_effects = {
        "coarse": mean("sample_diversity_distance_add_coarse_given_middle_fine"),
        "middle": mean("sample_diversity_distance_add_middle_given_coarse_fine"),
        "fine": mean("sample_diversity_distance_add_fine_given_coarse_middle"),
    }
    ged_effects = {
        "coarse": mean("ged2_add_coarse_given_middle_fine"),
        "middle": mean("ged2_add_middle_given_coarse_fine"),
        "fine": mean("ged2_add_fine_given_coarse_middle"),
    }
    cross_effects = {
        "coarse": mean("cross_dice_add_coarse_given_middle_fine"),
        "middle": mean("cross_dice_add_middle_given_coarse_fine"),
        "fine": mean("cross_dice_add_fine_given_coarse_middle"),
    }

    finite_div = {k: v for k, v in diversity_effects.items() if math.isfinite(v)}
    finite_ged = {k: v for k, v in ged_effects.items() if math.isfinite(v)}
    finite_cross = {k: v for k, v in cross_effects.items() if math.isfinite(v)}

    return {
        "interpretation_note": (
            "Effects are full-context descriptive effects: full stochastic 111 minus the same path with one scale deterministic. "
            "Positive diversity effect = that scale adds dispersion; positive GED2 effect = worse GED2; negative CrossDice effect = worse fidelity. "
            "Interactions may be large, so do not interpret any one scale in isolation without factorial interactions."
        ),
        "full_context_effects": {
            "diversity": diversity_effects,
            "ged2": ged_effects,
            "cross_dice": cross_effects,
        },
        "largest_positive_diversity_effect": max(finite_div, key=finite_div.get) if finite_div else None,
        "largest_ged2_worsening_effect": max(finite_ged, key=finite_ged.get) if finite_ged else None,
        "largest_crossdice_harm_effect": min(finite_cross, key=finite_cross.get) if finite_cross else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/eg_gm_punet_v3_hierarchical_ms_gmm_lidc.yaml",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--sample-counts", type=int, nargs="+", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="只跑原计划 7 路径，跳过 coarse+fine；不推荐用于正式 factorial 分析。",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_yaml(config_path)

    seed = int(cfg.get("validation_seed", cfg["seed"]))
    set_seed(seed)
    device = prepare_device(cfg.get("device", "cuda"))

    save_dir = Path(cfg["save_dir"])
    if not save_dir.is_absolute():
        save_dir = PROJECT_ROOT / save_dir

    checkpoint = Path(args.checkpoint) if args.checkpoint else save_dir / "best_val_ged4_eg_gm_punet_v3.pth"
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    obj = torch.load(checkpoint, map_location=device)
    checkpoint_epoch = int(obj.get("epoch", 0))

    model = build_model(cfg).to(device)
    model.load_state_dict(obj["model_state_dict"], strict=True)
    model.eval()

    ecfg = cfg["evaluation"]
    num_samples = int(args.num_samples or ecfg["final_prior_samples"])

    raw_sample_counts = list(args.sample_counts or ecfg["sample_counts"])

    if args.smoke_test:
        num_samples = min(num_samples, 4)
        raw_sample_counts = [
            int(n) for n in raw_sample_counts
            if int(n) <= num_samples
        ]

    sample_counts = normalize_sample_counts(
        raw_sample_counts,
        num_samples,
    )

    interventions = dict(INTERVENTIONS)
    if args.primary_only:
        interventions.pop("sample_coarse_fine", None)

    output_dir = (
        save_dir
        / "diagnostics"
        / "stage_a0_scale_intervention"
        / args.split
        / f"samples_{num_samples}"
        / checkpoint.stem
    )
    if args.smoke_test:
        output_dir = output_dir / "smoke_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = build_lidc_full_dataloader(
        root=cfg["data_root"],
        split=args.split,
        batch_size=1 if args.smoke_test else int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"]["num_workers"]),
        shuffle=False,
        random_rater=False,
        rater_index=int(cfg["validation"]["rater_index"]),
        return_consensus=True,
        return_meta=True,
    )

    rows: list[dict[str, Any]] = []
    fallback = 0
    max_batches = 2 if args.smoke_test else None
    iterator = loader if max_batches is None else islice(loader, max_batches)

    progress = tqdm(iterator, desc="Stage A.0 interventions", ncols=180)
    for batch_index, batch in enumerate(progress):
        images = batch["image"].to(device=device, dtype=torch.float32)
        all_seg = batch["masks"].to(device=device, dtype=torch.float32)

        case_ids = case_ids_from_batch(batch.get("case_id"), images.shape[0], fallback, args.split)
        patient_ids = values_from_batch(batch.get("patient_id"), images.shape[0], "")
        patient_ids = [p if p else patient_from_case(c) for p, c in zip(patient_ids, case_ids)]

        # Stage A.0 prediction is strictly prior-only. No segmentation/posterior is required.
        with torch.no_grad():
            state = model(image=images, segmentation=None, is_training=False)

            tape = _make_random_tape(
                num_samples=num_samples,
                batch_size=images.shape[0],
                latent_dims=model.probabilistic_model.latent_dims,
                seed=seed + 100003 * batch_index,
                device=images.device,
                dtype=images.dtype,
            )

            logits_by_intervention = {}
            for name, bits in interventions.items():
                logits_by_intervention[name] = intervention_logits(
                    model=model,
                    state=state,
                    bits=bits,
                    random_tape=tape,
                    num_samples=num_samples,
                )

        for name, bits in interventions.items():
            p_logits = logits_by_intervention[name]
            p_masks = p_logits.argmax(dim=2).float()
            p_fg = torch.softmax(p_logits, dim=2)[:, :, 1]

            for bi in range(images.shape[0]):
                case_metrics = evaluate_single_case(
                    sample_masks=p_masks[:, bi],
                    foreground_probabilities=p_fg[:, bi],
                    rater_masks=all_seg[bi],
                    sample_counts=sample_counts,
                )
                ann_div_iou = annotation_diversity_iou(all_seg[bi])
                for row in case_metrics:
                    row.update({
                        "case_id": case_ids[bi],
                        "patient_id": patient_ids[bi],
                        "split": args.split,
                        "intervention": name,
                        "sample_coarse": int(bits[0]),
                        "sample_middle": int(bits[1]),
                        "sample_fine": int(bits[2]),
                        "annotation_diversity_iou": ann_div_iou,
                    })
                rows.extend(case_metrics)

        fallback += images.shape[0]

    _write_rows(rows, output_dir / "intervention_per_case_metrics.csv")
    patient_rows = _aggregate_patient(rows)
    _write_rows(patient_rows, output_dir / "intervention_per_patient_metrics.csv")

    if len(interventions) == 8:
        factorial_rows, conditional_rows = _factorial_effects(rows, sample_counts)
        _write_rows(factorial_rows, output_dir / "factorial_effects_per_case.csv")
        _write_rows(conditional_rows, output_dir / "conditional_full_context_effects.csv")
    else:
        factorial_rows, conditional_rows = [], []

    intervention_summary = _group_intervention_summary(rows, sample_counts)
    primary_count = max(sample_counts)
    factorial_summary = _summarize_effect_rows(factorial_rows, primary_count)
    conditional_summary = _summarize_effect_rows(conditional_rows, primary_count)

    summary = {
        "stage": "EG-GM-PUNet v3 Stage A.0 — Per-Scale Stochastic Intervention Diagnosis",
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "split": args.split,
        "num_cases": len({r["case_id"] for r in rows}),
        "num_prior_samples": num_samples,
        "sample_counts": sample_counts,
        "common_random_numbers": True,
        "deterministic_off_policy": "conditional mixture expectation E[z_g]",
        "prediction_is_prior_only": True,
        "posterior_used": False,
        "interventions": {
            name: {
                "sample_coarse": bits[0],
                "sample_middle": bits[1],
                "sample_fine": bits[2],
            }
            for name, bits in interventions.items()
        },
        "intervention_summary": intervention_summary,
        "factorial_primary_sample_count": primary_count,
        "factorial_effects_at_primary_count": factorial_summary,
        "conditional_full_context_effects_at_primary_count": conditional_summary,
        "diagnostic_ranking_at_primary_count": _diagnostic_ranking(conditional_summary),
        "interpretation_guardrails": [
            "This is an intervention diagnosis on a fixed trained checkpoint, not a retrained ablation.",
            "Positive diversity change is not automatically beneficial; compare with CrossDice/GED2 and annotation diversity.",
            "GMM basis count has no semantic mode/annotator interpretation.",
            "Large factorial interactions mean a scale should not be blamed in isolation.",
            "sample_all uses controlled common-random-number exact sampling and need not exactly reproduce the previous evaluator random stream.",
        ],
    }
    (output_dir / "stagea0_intervention_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 110)
    print("Stage A.0 Per-Scale Stochastic Intervention Diagnosis")
    print("=" * 110)
    print(f"Checkpoint epoch : {checkpoint_epoch}")
    print(f"Cases            : {summary['num_cases']}")
    print(f"Samples          : {num_samples}")
    print(f"Interventions    : {len(interventions)}")
    print(f"Primary count    : {primary_count}")
    print("-" * 110)

    for name in interventions:
        block = intervention_summary[name]["metrics"].get(str(primary_count), intervention_summary[name]["metrics"].get(primary_count, {}))
        # summarize_rows may use int or string keys depending on project version.
        if not block:
            continue
        def _mean(metric):
            v = block.get(metric, {})
            return v.get("mean") if isinstance(v, dict) else None
        print(
            f"{name:24s} "
            f"GED2={_mean('ged2')!s:>10}  "
            f"CrossDice={_mean('cross_dice')!s:>10}  "
            f"BestDice={_mean('best_sample_dice_to_raters')!s:>10}  "
            f"Diversity={_mean('sample_diversity_distance')!s:>10}"
        )

    if conditional_summary:
        print("-" * 110)
        print("[FULL-CONTEXT ADD-ONE-SCALE EFFECTS | full(111) - without-scale]")
        for group, key_suffix in (
            ("coarse", "add_coarse_given_middle_fine"),
            ("middle", "add_middle_given_coarse_fine"),
            ("fine", "add_fine_given_coarse_middle"),
        ):
            def em(metric):
                key = f"{metric}_{key_suffix}"
                return conditional_summary.get(key, {}).get("mean")
            print(
                f"{group:8s}: "
                f"dGED2={em('ged2')!s:>10}  "
                f"dCrossDice={em('cross_dice')!s:>10}  "
                f"dBestDice={em('best_sample_dice_to_raters')!s:>10}  "
                f"dDiversity={em('sample_diversity_distance')!s:>10}"
            )

    print("-" * 110)
    print(f"[DONE] {output_dir / 'intervention_per_case_metrics.csv'}")
    print(f"[DONE] {output_dir / 'intervention_per_patient_metrics.csv'}")
    if factorial_rows:
        print(f"[DONE] {output_dir / 'factorial_effects_per_case.csv'}")
        print(f"[DONE] {output_dir / 'conditional_full_context_effects.csv'}")
    print(f"[DONE] {output_dir / 'stagea0_intervention_summary.json'}")
    print("=" * 110)


if __name__ == "__main__":
    main()
