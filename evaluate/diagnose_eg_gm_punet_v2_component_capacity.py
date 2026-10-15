"""
EG-GM-PUNet v2 Stage 0：Component Collapse / Latent Capacity 诊断脚本。

这个文件是干什么的：
1. 不训练新模型，直接读取当前第一版 EG-GM-PUNet 的 best GED checkpoint；
2. 在 test 集逐病例测量 prior/posterior component probability、熵、K_eff；
3. 计算 4 位医生 annotation diversity；
4. 计算 posterior-probe 与 prior-probe 的 evidence uncertainty；
5. 分别 decode K 个 prior component mean，检查“component 被占用”与
   “component 输出是否真的不同”是不是两件事；
6. 输出每病例 CSV 和 quartile/correlation summary，用来判断 K_eff≈2 到底是
   合理的 intrinsic mode number，还是高歧义病例也无法使用更多 component 的真正 collapse。

默认读取：
    configs/eg_gm_punet_lidc.yaml
    result/eg_gm_punet_k5_full/best_val_ged4_eg_gm_punet.pth

Smoke test：
    python evaluate/diagnose_eg_gm_punet_v2_component_capacity.py --smoke-test

正式诊断：
    python evaluate/diagnose_eg_gm_punet_v2_component_capacity.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from itertools import combinations, islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.lidc_dataset_full import build_lidc_full_dataloader
from models.EG_GM_PUNet import EGGMPUNet
from models.Multi_scale_Gaussian_Mixture_prob_unet.multiscale_grouped_gmm_model import (
    MultiScaleGroupedGMMState,
)
from utils.gmm_experiment_utils import prepare_device, set_seed


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ValueError("YAML 必须为字典。")
    return value


def build_legacy_model(config: dict[str, Any]) -> EGGMPUNet:
    item = config["model"]
    return EGGMPUNet(
        in_channels=int(item["in_channels"]),
        num_classes=int(item["num_classes"]),
        num_components=int(item["num_components"]),
        latent_dims={str(k): int(v) for k, v in item["latent_dims"].items()},
        num_channels=tuple(int(v) for v in item["num_channels"]),
        group_feature_indices={str(k): int(v) for k, v in item["group_feature_indices"].items()},
        group_decoder_indices={str(k): int(v) for k, v in item["group_decoder_indices"].items()},
        num_convs_per_block=int(item["num_convs_per_block"]),
        fusion_1x1_convs=int(item["fusion_1x1_convs"]),
        segmentation_head_1x1_convs=int(item["segmentation_head_1x1_convs"]),
        min_log_sigma=item.get("min_log_sigma"),
        max_log_sigma=item.get("max_log_sigma"),
        evidence_projection_channels=int(item["evidence_projection_channels"]),
        evidence_fusion_channels=int(item["evidence_fusion_channels"]),
        feedback_config=item.get("feedback"),
    )


def ids_from_batch(value: Any, batch_size: int, prefix: str, offset: int) -> list[str]:
    if value is None:
        return [f"{prefix}_{offset+i:06d}" for i in range(batch_size)]
    if isinstance(value, str):
        return [value]
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)] * batch_size


def binary_iou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    inter = (a * b).sum()
    union = ((a + b) > 0).float().sum()
    if float(union.item()) <= eps:
        return a.new_tensor(1.0)
    return inter / (union + eps)


def binary_dice(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    denom = a.sum() + b.sum()
    if float(denom.item()) <= eps:
        return a.new_tensor(1.0)
    return (2.0 * (a * b).sum() + eps) / (denom + eps)


def annotation_diversity(masks: torch.Tensor) -> float:
    """4 位医生之间平均 1-IoU。"""
    values = []
    for i, j in combinations(range(masks.shape[0]), 2):
        values.append(1.0 - float(binary_iou(masks[i], masks[j]).item()))
    return float(np.mean(values)) if values else 0.0


def distribution_stats(probs: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    safe = probs.clamp_min(eps)
    entropy = -(safe * safe.log()).sum(dim=1)
    norm = entropy / math.log(float(probs.shape[1]))
    keff = torch.exp(entropy)
    return entropy, norm, keff


def composite_evidence_score(evidence) -> torch.Tensor:
    summary = evidence.summaries["fused"]
    return (
        0.60 * summary.ambiguity_weighted_vacuity
        + 0.20 * summary.mean_predictive_entropy
        + 0.20 * summary.mean_mutual_information
    )


def component_mean_outputs(model: EGGMPUNet, state: MultiScaleGroupedGMMState) -> tuple[torch.Tensor, torch.Tensor]:
    """对每个 prior component 的 Gaussian mean 解码，输出 mask 和前景概率。"""
    pi = state.prior.mixture_probs
    batch, k = pi.shape
    probs = []
    masks = []
    for component in range(k):
        z_groups = {
            name: state.prior.group_mu[name][:, component]
            for name in model.group_names
        }
        indices = torch.full((batch,), component, device=pi.device, dtype=torch.long)
        decode = model.probabilistic_model.decode_from_groups(
            state=state,
            z_groups=z_groups,
            component_indices=indices,
        )
        fg = torch.softmax(decode.logits, dim=1)[:, 1]
        probs.append(fg)
        masks.append((fg >= 0.5).float())
    return torch.stack(masks, dim=1), torch.stack(probs, dim=1)


def pairwise_component_stats(masks: torch.Tensor) -> tuple[float, float]:
    dice_values = []
    iou_distances = []
    for i, j in combinations(range(masks.shape[0]), 2):
        dice_values.append(float(binary_dice(masks[i], masks[j]).item()))
        iou_distances.append(1.0 - float(binary_iou(masks[i], masks[j]).item()))
    return float(np.mean(dice_values)), float(np.mean(iou_distances))


def weighted_component_variance(component_probs: torch.Tensor, pi: torch.Tensor) -> float:
    mean = (pi[:, None, None] * component_probs).sum(dim=0)
    var = (pi[:, None, None] * (component_probs - mean[None]).square()).sum(dim=0)
    return float((4.0 * var).mean().item())


def rankdata(values: np.ndarray) -> np.ndarray:
    """不依赖 scipy 的简单平均秩实现，用于 Spearman correlation。"""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def correlation(x: list[float], y: list[float], spearman: bool = False) -> float | None:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    if spearman:
        a, b = rankdata(a), rankdata(b)
    return float(np.corrcoef(a, b)[0, 1])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ann = np.asarray([float(r["annotation_diversity"]) for r in rows])
    prior_k = np.asarray([float(r["prior_k_eff"]) for r in rows])
    post_k = np.asarray([float(r["posterior_k_eff"]) for r in rows])
    comp_div = np.asarray([float(r["component_pairwise_iou_distance"]) for r in rows])
    prior_u = np.asarray([float(r["prior_probe_uncertainty"]) for r in rows])

    quantiles = np.quantile(ann, [0.25, 0.5, 0.75])
    quartile_rows = []
    bounds = [-np.inf, *quantiles.tolist(), np.inf]
    for q in range(4):
        selector = (ann > bounds[q]) & (ann <= bounds[q + 1])
        if q == 0:
            selector = ann <= bounds[q + 1]
        quartile_rows.append({
            "quartile": q + 1,
            "num_cases": int(selector.sum()),
            "annotation_diversity_mean": float(ann[selector].mean()),
            "prior_k_eff_mean": float(prior_k[selector].mean()),
            "posterior_k_eff_mean": float(post_k[selector].mean()),
            "component_pairwise_iou_distance_mean": float(comp_div[selector].mean()),
            "prior_probe_uncertainty_mean": float(prior_u[selector].mean()),
        })

    return {
        "num_cases": len(rows),
        "annotation_diversity_mean": float(ann.mean()),
        "prior_k_eff_mean": float(prior_k.mean()),
        "posterior_k_eff_mean": float(post_k.mean()),
        "corr_annotation_diversity_prior_k_eff_pearson": correlation(ann.tolist(), prior_k.tolist()),
        "corr_annotation_diversity_prior_k_eff_spearman": correlation(ann.tolist(), prior_k.tolist(), True),
        "corr_annotation_diversity_posterior_k_eff_pearson": correlation(ann.tolist(), post_k.tolist()),
        "corr_annotation_diversity_component_diversity_pearson": correlation(ann.tolist(), comp_div.tolist()),
        "corr_annotation_diversity_prior_probe_uncertainty_pearson": correlation(ann.tolist(), prior_u.tolist()),
        "quartiles": quartile_rows,
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eg_gm_punet_lidc.yaml")
    parser.add_argument(
        "--checkpoint",
        default="result/eg_gm_punet_k5_full/best_val_ged4_eg_gm_punet.pth",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    config = load_yaml(config_path)
    set_seed(int(config["validation_seed"]))
    device = prepare_device(config.get("device", "cuda"))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("config", config)
    model = build_legacy_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    loader = build_lidc_full_dataloader(
        root=config["data_root"],
        split=args.split,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        shuffle=False,
        random_rater=False,
        rater_index=int(config["validation"]["rater_index"]),
    )

    output_dir = (
        PROJECT_ROOT
        / "result"
        / "eg_gm_punet_v2_diagnostics"
        / "component_capacity"
        / args.split
    )
    if args.smoke_test:
        output_dir = output_dir / "smoke_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    fallback = 0
    iterable = islice(loader, 2) if args.smoke_test else loader
    total = min(len(loader), 2) if args.smoke_test else len(loader)

    print("=" * 104)
    print("EG-GM-PUNet v2 Stage 0: component/capacity diagnosis")
    print("=" * 104)
    print(f"[INFO] Checkpoint : {checkpoint_path}")
    print(f"[INFO] Epoch      : {checkpoint.get('epoch')}")
    print(f"[INFO] Split      : {args.split}")
    print(f"[INFO] Cases      : {len(loader.dataset)}")
    print(f"[INFO] Output     : {output_dir}")

    for batch in tqdm(iterable, total=total, desc="v2 diagnose", ncols=140):
        image = batch["image"].to(device=device, dtype=torch.float32)
        selected = batch["mask"].to(device=device, dtype=torch.float32)
        all_masks = batch["masks"].to(device=device, dtype=torch.float32)
        b = image.shape[0]
        case_ids = ids_from_batch(batch.get("case_id"), b, args.split, fallback)
        patient_ids = ids_from_batch(batch.get("patient_id"), b, "patient", fallback)

        # 使用一个固定 rater 构建 posterior，目的是诊断训练时 posterior routing。
        state = model(image=image, segmentation=selected, is_training=True)
        p_ent, p_norm, p_keff = distribution_stats(state.prior.mixture_probs)
        q_ent, q_norm, q_keff = distribution_stats(state.posterior.mixture_probs)

        posterior_probe = model.feedback_probe(state)
        posterior_u = composite_evidence_score(posterior_probe)
        prior_only_state = MultiScaleGroupedGMMState(
            image_encoder_features=state.image_encoder_features,
            prior=state.prior,
            posterior=None,
        )
        prior_probe = model.feedback_probe(prior_only_state)
        prior_u = composite_evidence_score(prior_probe)

        component_masks, component_probs = component_mean_outputs(model, prior_only_state)

        for i in range(b):
            ann_div = annotation_diversity(all_masks[i])
            pair_dice, pair_iou_dist = pairwise_component_stats(component_masks[i])
            comp_var = weighted_component_variance(
                component_probs[i], state.prior.mixture_probs[i]
            )
            row: dict[str, Any] = {
                "case_id": case_ids[i],
                "patient_id": patient_ids[i],
                "annotation_diversity": ann_div,
                "prior_entropy": float(p_ent[i].item()),
                "prior_normalized_entropy": float(p_norm[i].item()),
                "prior_k_eff": float(p_keff[i].item()),
                "posterior_entropy": float(q_ent[i].item()),
                "posterior_normalized_entropy": float(q_norm[i].item()),
                "posterior_k_eff": float(q_keff[i].item()),
                "prior_probe_uncertainty": float(prior_u[i].item()),
                "posterior_probe_uncertainty": float(posterior_u[i].item()),
                "component_pairwise_dice": pair_dice,
                "component_pairwise_iou_distance": pair_iou_dist,
                "component_weighted_variance": comp_var,
            }
            for k in range(model.num_components):
                row[f"prior_component_{k}_prob"] = float(state.prior.mixture_probs[i, k].item())
                row[f"posterior_component_{k}_prob"] = float(state.posterior.mixture_probs[i, k].item())
            rows.append(row)
        fallback += b

    if not rows:
        raise RuntimeError("没有生成任何诊断行。")

    csv_path = output_dir / "component_capacity_per_case.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary.update({
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
    })
    summary_path = output_dir / "component_capacity_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n[SUMMARY]")
    print(f"  prior K_eff mean  : {summary['prior_k_eff_mean']:.4f}")
    print(f"  posterior K_eff   : {summary['posterior_k_eff_mean']:.4f}")
    print(
        "  Corr(D_ann,Keff) : "
        f"Pearson={summary['corr_annotation_diversity_prior_k_eff_pearson']}, "
        f"Spearman={summary['corr_annotation_diversity_prior_k_eff_spearman']}"
    )
    for item in summary["quartiles"]:
        print(
            f"  Q{item['quartile']}: n={item['num_cases']}, "
            f"D_ann={item['annotation_diversity_mean']:.4f}, "
            f"prior_Keff={item['prior_k_eff_mean']:.4f}, "
            f"componentDiv={item['component_pairwise_iou_distance_mean']:.4f}"
        )
    print(f"[DONE] CSV     : {csv_path}")
    print(f"[DONE] Summary : {summary_path}")


if __name__ == "__main__":
    main()
