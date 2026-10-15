"""
EG-GM-PUNet v3 Stage A.0b — Latent Amplitude & Decoder Sensitivity Diagnosis
================================================================================

目的
----
Stage A.0 已确认：当前 v3 的 predictive stochasticity 几乎全部由 coarse latent
承担，而 middle / fine 对最终分割随机性贡献接近 0。

本脚本不训练、不修改 checkpoint，只做 prior-only 固定模型诊断，区分两种原因：

A. Latent-density under-utilization
   middle/fine GMM 本身方差很小，因而采样几乎不改变 z。

B. Decoder under-sensitivity
   middle/fine GMM 有明显 latent variation，但 decoder 对这些 latent 的响应很弱。

诊断分两部分：
1) GMM latent amplitude audit
   对每个尺度分解总方差：
       Var(z) = E_k[sigma_k^2] + E_k[(mu_k - E[z])^2]
              = within-basis variance + between-basis variance

2) Standardized decoder sensitivity
   在 deterministic hierarchical prior mean path 上，对每个 latent dimension 做 ±1σ 扰动。
   同时计算两种响应：

   Direct response:
       只改变当前 z_g，其他 z 保持 baseline expectation，隔离 decoder injection coupling。

   Propagated response:
       改变 z_g 后，重新计算所有 downstream conditional GMM expectations，测量完整
       hierarchical path 的功能响应。

   对每个 latent dimension d：
       z_g^+ = E[z_g] + sigma_total,g,d
       z_g^- = E[z_g] - sigma_total,g,d

   输出：
       - foreground probability L1 half-difference
       - foreground probability RMS half-difference
       - logit-margin L1 half-difference
       - binary segmentation IoU distance
       - per-latent-unit local response（除以 sigma step）

重要 guardrails
----------------
- prediction 只使用 prior；posterior 不参与任何测试预测；
- GMM basis 仅表示 latent-density basis，不解释为 annotator / semantic mode；
- ±1σ 是固定 checkpoint 上的 intervention，不是训练 ablation；
- direct 与 propagated response 必须分开解释。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from itertools import islice
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.lidc_dataset_full import build_lidc_full_dataloader
from models.EG_GM_PUNet_v3 import EGGMPUNetV3
from models.Hierarchical_Multi_scale_Gaussian_Mixture_prob_unet.hierarchical_multiscale_gmm_parts import (
    ScaleGaussianMixtureOutput,
    effective_component_count,
    mixture_expected_mean,
    normalized_mixture_entropy,
)
from utils.gmm_experiment_utils import prepare_device, set_seed


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

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


def case_ids_from_batch(value: Any, batch_size: int, fallback: int, split: str) -> list[str]:
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


def values_from_batch(value: Any, batch_size: int, default: str = "") -> list[str]:
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
    r = int(x.shape[0])
    vals = []
    for i in range(r):
        a = x[i].reshape(-1)
        for j in range(i + 1, r):
            b = x[j].reshape(-1)
            inter = (a * b).sum()
            union = ((a + b) > 0).float().sum()
            iou = (
                torch.tensor(1.0, device=x.device)
                if union <= eps
                else inter / union.clamp_min(eps)
            )
            vals.append(1.0 - iou)
    return float(torch.stack(vals).mean().item())


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = []
    for row in rows:
        try:
            v = float(row[key])
            if math.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    return np.asarray(vals, dtype=np.float64)


def describe(values: np.ndarray) -> dict[str, float | int | None]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "q25": None, "q75": None}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
    }


def rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def correlation(x: np.ndarray, y: np.ndarray, spearman: bool = False) -> float | None:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    if spearman:
        a, b = rankdata(a), rankdata(b)
    return float(np.corrcoef(a, b)[0, 1])


def aggregate_patient(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["patient_id"]), []).append(row)
    output = []
    for patient_id, items in sorted(groups.items()):
        result: dict[str, Any] = {"patient_id": patient_id, "num_cases": len(items)}
        keys = set().union(*(item.keys() for item in items))
        for key in keys:
            if key in {"case_id", "patient_id", "scale", "sensitivity_mode"}:
                continue
            vals = []
            for item in items:
                try:
                    v = float(item[key])
                    if math.isfinite(v):
                        vals.append(v)
                except Exception:
                    pass
            if vals:
                result[key] = float(np.mean(vals))
        output.append(result)
    return output


# -----------------------------------------------------------------------------
# GMM amplitude math
# -----------------------------------------------------------------------------

def gmm_variance_decomposition(
    distribution: ScaleGaussianMixtureOutput,
) -> dict[str, torch.Tensor]:
    """Return per-case, per-latent-dimension variance decomposition.

    Shapes:
        mean              [B,D]
        within_var        [B,D]
        between_var       [B,D]
        total_var         [B,D]
        total_std         [B,D]
    """
    pi = distribution.mixture_probs
    mu = distribution.mu
    sigma2 = distribution.sigma.pow(2)

    mean = (pi.unsqueeze(-1) * mu).sum(dim=1)
    within_var = (pi.unsqueeze(-1) * sigma2).sum(dim=1)
    between_var = (
        pi.unsqueeze(-1) * (mu - mean.unsqueeze(1)).pow(2)
    ).sum(dim=1)
    total_var = (within_var + between_var).clamp_min(0.0)
    total_std = torch.sqrt(total_var.clamp_min(1e-12))

    return {
        "mean": mean,
        "within_var": within_var,
        "between_var": between_var,
        "total_var": total_var,
        "total_std": total_std,
    }


def build_deterministic_prior_hierarchy(model: EGGMPUNetV3, state):
    previous_z: dict[str, torch.Tensor] = {}
    distributions: dict[str, ScaleGaussianMixtureOutput] = {}
    amplitude: dict[str, dict[str, torch.Tensor]] = {}

    for name in model.group_names:
        dist = model.prior.distribution(
            features=state.prior_context_features,
            group_name=name,
            previous_z=previous_z,
        )
        amp = gmm_variance_decomposition(dist)
        z = amp["mean"]
        distributions[name] = dist
        amplitude[name] = amp
        previous_z[name] = z

    return previous_z, distributions, amplitude


# -----------------------------------------------------------------------------
# Decoder response math
# -----------------------------------------------------------------------------

def foreground_probability(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, 1]


def logit_margin(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] != 2:
        raise ValueError("Stage A.0b 当前针对 binary LIDC num_classes=2。")
    return logits[:, 1] - logits[:, 0]


def binary_iou_distance_from_logits(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    a = logits_a.argmax(dim=1).float()
    b = logits_b.argmax(dim=1).float()
    inter = (a * b).flatten(start_dim=1).sum(dim=1)
    union = ((a + b) > 0).float().flatten(start_dim=1).sum(dim=1)
    iou = torch.where(
        union <= eps,
        torch.ones_like(union),
        inter / union.clamp_min(eps),
    )
    return 1.0 - iou


def response_metrics(
    logits_plus: torch.Tensor,
    logits_minus: torch.Tensor,
    step: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Central ±step response for one latent dimension.

    step: [B], positive 1-sigma step of the perturbed dimension.
    """
    p_plus = foreground_probability(logits_plus)
    p_minus = foreground_probability(logits_minus)
    dp = p_plus - p_minus

    margin_plus = logit_margin(logits_plus)
    margin_minus = logit_margin(logits_minus)
    dm = margin_plus - margin_minus

    prob_l1_half = 0.5 * dp.abs().flatten(start_dim=1).mean(dim=1)
    prob_rms_half = 0.5 * torch.sqrt(
        dp.pow(2).flatten(start_dim=1).mean(dim=1).clamp_min(0.0)
    )
    margin_l1_half = 0.5 * dm.abs().flatten(start_dim=1).mean(dim=1)
    binary_iou_distance = binary_iou_distance_from_logits(logits_plus, logits_minus)

    safe_step = step.abs().clamp_min(1e-6)
    prob_l1_per_latent_unit = prob_l1_half / safe_step
    margin_l1_per_latent_unit = margin_l1_half / safe_step

    return {
        "prob_l1_half": prob_l1_half,
        "prob_rms_half": prob_rms_half,
        "margin_l1_half": margin_l1_half,
        "binary_iou_distance": binary_iou_distance,
        "prob_l1_per_latent_unit": prob_l1_per_latent_unit,
        "margin_l1_per_latent_unit": margin_l1_per_latent_unit,
    }


def decode_direct_perturbation(
    model: EGGMPUNetV3,
    state,
    baseline_z: Mapping[str, torch.Tensor],
    scale: str,
    dim: int,
    delta: torch.Tensor,
):
    """Only change z_scale; all other scale latents remain baseline means."""
    z_groups = {name: z.clone() for name, z in baseline_z.items()}
    z_groups[scale][:, dim] = z_groups[scale][:, dim] + delta
    return model.probabilistic_model.decode_from_groups(
        state=state,
        z_groups=z_groups,
    )


def decode_propagated_perturbation(
    model: EGGMPUNetV3,
    state,
    target_scale: str,
    target_dim: int,
    sign: float,
    baseline_std: torch.Tensor,
):
    """Perturb target z by ±1σ and recompute downstream conditional expectations.

    Earlier scales use deterministic expectations. At target scale we add the supplied
    baseline 1-sigma step. All later scales recompute their GMM under the perturbed
    previous_z and then take E[z].
    """
    previous_z: dict[str, torch.Tensor] = {}
    for name in model.group_names:
        dist = model.prior.distribution(
            features=state.prior_context_features,
            group_name=name,
            previous_z=previous_z,
        )
        z = mixture_expected_mean(dist)
        if name == target_scale:
            z = z.clone()
            z[:, target_dim] = z[:, target_dim] + float(sign) * baseline_std
        previous_z[name] = z

    return model.probabilistic_model.decode_from_groups(
        state=state,
        z_groups=previous_z,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/eg_gm_punet_v3_hierarchical_ms_gmm_lidc.yaml",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sigma-multiplier", type=float, default=1.0)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.sigma_multiplier <= 0:
        raise ValueError("--sigma-multiplier 必须 > 0。")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_yaml(config_path)

    set_seed(int(cfg.get("validation_seed", cfg["seed"])))
    device = prepare_device(cfg.get("device", "cuda"))

    save_dir = Path(cfg["save_dir"])
    checkpoint = Path(
        args.checkpoint or save_dir / "best_val_ged4_eg_gm_punet_v3.pth"
    )
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    obj = torch.load(checkpoint, map_location=device)
    checkpoint_epoch = int(obj.get("epoch", 0))

    model = build_model(cfg).to(device)
    model.load_state_dict(obj["model_state_dict"], strict=True)
    model.eval()

    output_dir = (
        save_dir
        / "diagnostics"
        / "stage_a0b_latent_decoder_sensitivity"
        / args.split
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

    amplitude_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    combined_case_rows: list[dict[str, Any]] = []

    fallback = 0
    max_batches = 2 if args.smoke_test else None
    iterator = loader if max_batches is None else islice(loader, max_batches)

    progress = tqdm(iterator, desc="v3 Stage A.0b", ncols=180)

    with torch.no_grad():
        for batch in progress:
            images = batch["image"].to(device=device, dtype=torch.float32)
            all_seg = batch["masks"].to(device=device, dtype=torch.float32)

            case_ids = case_ids_from_batch(
                batch.get("case_id"), images.shape[0], fallback, args.split
            )
            patient_ids = values_from_batch(batch.get("patient_id"), images.shape[0], "")
            patient_ids = [
                p if p else patient_from_case(c)
                for p, c in zip(patient_ids, case_ids)
            ]

            # Prior-only state. No posterior is constructed or used.
            state = model(image=images, segmentation=None, is_training=False)

            baseline_z, baseline_dist, amplitude = build_deterministic_prior_hierarchy(
                model, state
            )
            baseline_decode = model.probabilistic_model.decode_from_groups(
                state=state,
                z_groups=baseline_z,
                distributions=baseline_dist,
            )
            baseline_prob = foreground_probability(baseline_decode.logits)

            # Per-batch cache for case-level combined output.
            combined = []
            for bi in range(images.shape[0]):
                combined.append({
                    "case_id": case_ids[bi],
                    "patient_id": patient_ids[bi],
                    "annotation_diversity_iou": annotation_diversity_iou(all_seg[bi]),
                    "baseline_foreground_probability_mean": float(
                        baseline_prob[bi].mean().item()
                    ),
                })

            for scale in model.group_names:
                dist = baseline_dist[scale]
                amp = amplitude[scale]
                pi = dist.mixture_probs
                total_var = amp["total_var"]
                total_std = amp["total_std"]
                within_var = amp["within_var"]
                between_var = amp["between_var"]

                entropy_norm = normalized_mixture_entropy(dist)
                keff = effective_component_count(dist)

                # amplitude output per case
                for bi in range(images.shape[0]):
                    within_mean = float(within_var[bi].mean().item())
                    between_mean = float(between_var[bi].mean().item())
                    total_mean = float(total_var[bi].mean().item())
                    std_rms = float(torch.sqrt(total_var[bi].mean()).item())
                    std_mean = float(total_std[bi].mean().item())
                    total_sum = float(total_var[bi].sum().item())
                    between_fraction = between_mean / max(total_mean, 1e-12)

                    row = {
                        "case_id": case_ids[bi],
                        "patient_id": patient_ids[bi],
                        "scale": scale,
                        "annotation_diversity_iou": combined[bi]["annotation_diversity_iou"],
                        "latent_dim": int(total_var.shape[1]),
                        "within_var_mean_dim": within_mean,
                        "between_var_mean_dim": between_mean,
                        "total_var_mean_dim": total_mean,
                        "total_var_sum": total_sum,
                        "total_std_rms": std_rms,
                        "total_std_mean_dim": std_mean,
                        "between_variance_fraction": between_fraction,
                        "normalized_mixture_entropy": float(entropy_norm[bi].item()),
                        "effective_basis_count": float(keff[bi].item()),
                    }
                    for d in range(total_var.shape[1]):
                        row[f"within_var_dim_{d}"] = float(within_var[bi, d].item())
                        row[f"between_var_dim_{d}"] = float(between_var[bi, d].item())
                        row[f"total_var_dim_{d}"] = float(total_var[bi, d].item())
                        row[f"total_std_dim_{d}"] = float(total_std[bi, d].item())
                    amplitude_rows.append(row)

                    prefix = f"{scale}_"
                    combined[bi][prefix + "within_var_mean_dim"] = within_mean
                    combined[bi][prefix + "between_var_mean_dim"] = between_mean
                    combined[bi][prefix + "total_var_mean_dim"] = total_mean
                    combined[bi][prefix + "total_std_rms"] = std_rms
                    combined[bi][prefix + "between_variance_fraction"] = between_fraction
                    combined[bi][prefix + "effective_basis_count"] = float(keff[bi].item())

                # Sensitivity is computed per latent dimension, then averaged per case/scale.
                per_mode_metrics: dict[str, list[dict[str, torch.Tensor]]] = {
                    "direct": [],
                    "propagated": [],
                }

                for dim in range(int(total_std.shape[1])):
                    step = total_std[:, dim] * float(args.sigma_multiplier)

                    # Direct decoder coupling: downstream z is held at baseline expectation.
                    direct_plus = decode_direct_perturbation(
                        model, state, baseline_z, scale, dim, +step
                    )
                    direct_minus = decode_direct_perturbation(
                        model, state, baseline_z, scale, dim, -step
                    )
                    direct_metrics = response_metrics(
                        direct_plus.logits,
                        direct_minus.logits,
                        step,
                    )
                    per_mode_metrics["direct"].append(direct_metrics)

                    # Full hierarchical propagation.
                    propagated_plus = decode_propagated_perturbation(
                        model,
                        state,
                        target_scale=scale,
                        target_dim=dim,
                        sign=+1.0,
                        baseline_std=step,
                    )
                    propagated_minus = decode_propagated_perturbation(
                        model,
                        state,
                        target_scale=scale,
                        target_dim=dim,
                        sign=-1.0,
                        baseline_std=step,
                    )
                    propagated_metrics = response_metrics(
                        propagated_plus.logits,
                        propagated_minus.logits,
                        step,
                    )
                    per_mode_metrics["propagated"].append(propagated_metrics)

                for mode in ("direct", "propagated"):
                    metrics_by_dim = per_mode_metrics[mode]
                    metric_names = tuple(metrics_by_dim[0].keys())
                    stacked = {
                        metric: torch.stack(
                            [m[metric] for m in metrics_by_dim], dim=1
                        )
                        for metric in metric_names
                    }  # metric -> [B,D]

                    for bi in range(images.shape[0]):
                        row = {
                            "case_id": case_ids[bi],
                            "patient_id": patient_ids[bi],
                            "scale": scale,
                            "sensitivity_mode": mode,
                            "annotation_diversity_iou": combined[bi]["annotation_diversity_iou"],
                            "sigma_multiplier": float(args.sigma_multiplier),
                            "latent_dim": int(total_std.shape[1]),
                            "total_std_rms": float(torch.sqrt(total_var[bi].mean()).item()),
                        }
                        for metric, tensor in stacked.items():
                            row[metric + "_mean_dim"] = float(tensor[bi].mean().item())
                            row[metric + "_max_dim"] = float(tensor[bi].max().item())
                            for d in range(tensor.shape[1]):
                                row[f"{metric}_dim_{d}"] = float(tensor[bi, d].item())
                        sensitivity_rows.append(row)

                        prefix = f"{scale}_{mode}_"
                        for metric, tensor in stacked.items():
                            combined[bi][prefix + metric + "_mean_dim"] = float(
                                tensor[bi].mean().item()
                            )

                # Extra derived ratio: propagated / direct standardized response.
                direct_prob = torch.stack(
                    [m["prob_l1_half"] for m in per_mode_metrics["direct"]], dim=1
                ).mean(dim=1)
                propagated_prob = torch.stack(
                    [m["prob_l1_half"] for m in per_mode_metrics["propagated"]], dim=1
                ).mean(dim=1)
                ratio = propagated_prob / direct_prob.clamp_min(1e-12)
                for bi in range(images.shape[0]):
                    combined[bi][f"{scale}_propagation_to_direct_prob_response_ratio"] = float(
                        ratio[bi].item()
                    )

            combined_case_rows.extend(combined)
            fallback += images.shape[0]

    # ------------------------------------------------------------------
    # Save per-case and patient outputs
    # ------------------------------------------------------------------
    write_csv(amplitude_rows, output_dir / "latent_amplitude_per_case.csv")
    write_csv(sensitivity_rows, output_dir / "decoder_sensitivity_per_case.csv")
    write_csv(combined_case_rows, output_dir / "stagea0b_analysis_per_case.csv")
    write_csv(aggregate_patient(combined_case_rows), output_dir / "stagea0b_analysis_per_patient.csv")

    # ------------------------------------------------------------------
    # Summary by scale and mode
    # ------------------------------------------------------------------
    summary: dict[str, Any] = {
        "stage": "EG-GM-PUNet v3 Stage A.0b — Latent Amplitude and Decoder Sensitivity Diagnosis",
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "split": args.split,
        "num_cases": len(combined_case_rows),
        "prior_only": True,
        "posterior_used": False,
        "sigma_multiplier": float(args.sigma_multiplier),
        "amplitude_definition": "Var(z)=E_pi[sigma^2]+E_pi[(mu-E[z])^2]",
        "direct_sensitivity": "perturb z_g by ±1σ; all other z held at deterministic baseline expectation",
        "propagated_sensitivity": "perturb z_g by ±1σ; recompute downstream conditional GMM expectations",
        "scale_summary": {},
        "correlations_with_annotation_diversity": {},
        "diagnostic_ratios": {},
        "interpretation_guardrails": [
            "This is a fixed-checkpoint prior-only intervention diagnosis, not a retrained ablation.",
            "GMM basis count has no annotator or semantic-mode interpretation.",
            "total_std_rms measures latent amplitude; prob_l1_per_latent_unit approximates decoder response per latent unit.",
            "Direct sensitivity isolates decoder injection coupling; propagated sensitivity also includes downstream conditional-head effects.",
            "A large 1-sigma output response can come from large latent variance, high decoder sensitivity, or both; interpret amplitude and per-unit response jointly.",
        ],
    }

    for scale in model.group_names:
        amp_rows = [r for r in amplitude_rows if r["scale"] == scale]
        summary["scale_summary"][scale] = {
            "amplitude": {
                key: describe(finite_values(amp_rows, key))
                for key in (
                    "within_var_mean_dim",
                    "between_var_mean_dim",
                    "total_var_mean_dim",
                    "total_std_rms",
                    "total_std_mean_dim",
                    "between_variance_fraction",
                    "normalized_mixture_entropy",
                    "effective_basis_count",
                )
            },
            "sensitivity": {},
        }

        for mode in ("direct", "propagated"):
            rows = [
                r for r in sensitivity_rows
                if r["scale"] == scale and r["sensitivity_mode"] == mode
            ]
            summary["scale_summary"][scale]["sensitivity"][mode] = {
                key: describe(finite_values(rows, key))
                for key in (
                    "prob_l1_half_mean_dim",
                    "prob_rms_half_mean_dim",
                    "margin_l1_half_mean_dim",
                    "binary_iou_distance_mean_dim",
                    "prob_l1_per_latent_unit_mean_dim",
                    "margin_l1_per_latent_unit_mean_dim",
                )
            }

        # Correlation with annotation diversity, per case.
        ann = finite_values(amp_rows, "annotation_diversity_iou")
        for metric in ("total_std_rms", "total_var_mean_dim", "between_variance_fraction"):
            vals = finite_values(amp_rows, metric)
            if len(vals) == len(ann):
                summary["correlations_with_annotation_diversity"][f"{scale}_{metric}"] = {
                    "pearson": correlation(ann, vals, spearman=False),
                    "spearman": correlation(ann, vals, spearman=True),
                }

        for mode in ("direct", "propagated"):
            rows = [
                r for r in sensitivity_rows
                if r["scale"] == scale and r["sensitivity_mode"] == mode
            ]
            ann_s = finite_values(rows, "annotation_diversity_iou")
            for metric in (
                "prob_l1_half_mean_dim",
                "prob_l1_per_latent_unit_mean_dim",
                "binary_iou_distance_mean_dim",
            ):
                vals = finite_values(rows, metric)
                if len(vals) == len(ann_s):
                    summary["correlations_with_annotation_diversity"][f"{scale}_{mode}_{metric}"] = {
                        "pearson": correlation(ann_s, vals, spearman=False),
                        "spearman": correlation(ann_s, vals, spearman=True),
                    }

    # Scale ratios: useful for quick diagnosis.
    def mean_summary(scale: str, section: str, mode: str | None, metric: str) -> float | None:
        try:
            if section == "amplitude":
                return summary["scale_summary"][scale]["amplitude"][metric]["mean"]
            return summary["scale_summary"][scale]["sensitivity"][mode][metric]["mean"]
        except Exception:
            return None

    for numerator in model.group_names:
        for denominator in model.group_names:
            if numerator == denominator:
                continue
            a = mean_summary(numerator, "amplitude", None, "total_std_rms")
            b = mean_summary(denominator, "amplitude", None, "total_std_rms")
            if a is not None and b is not None and abs(b) > 1e-12:
                summary["diagnostic_ratios"][f"latent_std_{numerator}_over_{denominator}"] = float(a / b)

            for mode in ("direct", "propagated"):
                a = mean_summary(numerator, "sensitivity", mode, "prob_l1_per_latent_unit_mean_dim")
                b = mean_summary(denominator, "sensitivity", mode, "prob_l1_per_latent_unit_mean_dim")
                if a is not None and b is not None and abs(b) > 1e-12:
                    summary["diagnostic_ratios"][
                        f"per_unit_prob_sensitivity_{mode}_{numerator}_over_{denominator}"
                    ] = float(a / b)

                a = mean_summary(numerator, "sensitivity", mode, "prob_l1_half_mean_dim")
                b = mean_summary(denominator, "sensitivity", mode, "prob_l1_half_mean_dim")
                if a is not None and b is not None and abs(b) > 1e-12:
                    summary["diagnostic_ratios"][
                        f"one_sigma_prob_response_{mode}_{numerator}_over_{denominator}"
                    ] = float(a / b)

    with open(output_dir / "stagea0b_latent_decoder_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("Stage A.0b — Latent Amplitude & Decoder Sensitivity Diagnosis")
    print("=" * 100)
    print(f"Checkpoint epoch : {checkpoint_epoch}")
    print(f"Cases            : {len(combined_case_rows)}")
    print(f"Sigma multiplier : {args.sigma_multiplier:.3f}")
    print()
    print("[SCALE SUMMARY]")
    for scale in model.group_names:
        amp = summary["scale_summary"][scale]["amplitude"]
        direct = summary["scale_summary"][scale]["sensitivity"]["direct"]
        prop = summary["scale_summary"][scale]["sensitivity"]["propagated"]
        print(
            f"  {scale:6s} | std_rms={amp['total_std_rms']['mean']:.6g} "
            f"withinVar={amp['within_var_mean_dim']['mean']:.6g} "
            f"betweenVar={amp['between_var_mean_dim']['mean']:.6g} "
            f"| direct 1σ Δp={direct['prob_l1_half_mean_dim']['mean']:.6g} "
            f"perUnit={direct['prob_l1_per_latent_unit_mean_dim']['mean']:.6g} "
            f"| propagated 1σ Δp={prop['prob_l1_half_mean_dim']['mean']:.6g} "
            f"perUnit={prop['prob_l1_per_latent_unit_mean_dim']['mean']:.6g}"
        )
    print()
    print(f"[DONE] {output_dir / 'latent_amplitude_per_case.csv'}")
    print(f"[DONE] {output_dir / 'decoder_sensitivity_per_case.csv'}")
    print(f"[DONE] {output_dir / 'stagea0b_analysis_per_case.csv'}")
    print(f"[DONE] {output_dir / 'stagea0b_analysis_per_patient.csv'}")
    print(f"[DONE] {output_dir / 'stagea0b_latent_decoder_summary.json'}")
    print("=" * 100)


if __name__ == "__main__":
    main()