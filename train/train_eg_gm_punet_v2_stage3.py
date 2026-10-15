"""
EG-GM-PUNet v2 Stage 3：Conditional Component Vitality 训练入口。

本文件复用已经验证过的 train/train_eg_gm_punet.py 主训练循环，仅替换：
1. 模型为 EGGMPUNetV2DemandCapacity；
2. 在线日志增加 U^amb / capacity / mismatch / beta / tau；
3. validation 继续保存 demand-vs-annotation Pearson/Spearman/Q1-Q4。

Stage 2 不启用：
- Component Vitality；
- Ambiguity Alignment。

Smoke：
    python train/train_eg_gm_punet_v2_stage2.py ^
      --config configs/eg_gm_punet_v2_conditional_vitality_lidc.yaml ^
      --smoke-test

正式训练：
    python train/train_eg_gm_punet_v2_stage2.py ^
      --config configs/eg_gm_punet_v2_conditional_vitality_lidc.yaml
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LEGACY_TRAIN = PROJECT_ROOT / "train" / "train_eg_gm_punet.py"
if not LEGACY_TRAIN.exists():
    raise FileNotFoundError(
        "Stage 2 复用第一版已验证训练循环，但未找到："
        f"{LEGACY_TRAIN}"
    )

spec = importlib.util.spec_from_file_location(
    "_eg_gm_punet_stage3_legacy_train",
    LEGACY_TRAIN,
)
if spec is None or spec.loader is None:
    raise ImportError(f"无法导入：{LEGACY_TRAIN}")

legacy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)

from models.EG_GM_PUNet_v2 import EGGMPUNetV2ConditionalVitality


class ConfiguredStage3Model(EGGMPUNetV2ConditionalVitality):
    """把训练 feedback switches 固定到 Stage 2 model。"""

    def __init__(
        self,
        *args,
        feedback_enabled: bool,
        categorical_feedback: bool,
        group_feedback: bool,
        temperature_feedback: bool,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.training_feedback_enabled = bool(feedback_enabled)
        self.training_categorical_feedback = bool(categorical_feedback)
        self.training_group_feedback = bool(group_feedback)
        self.training_temperature_feedback = bool(temperature_feedback)

    def elbo(self, *args, **kwargs):
        kwargs.setdefault("feedback_enabled", self.training_feedback_enabled)
        kwargs.setdefault(
            "categorical_feedback", self.training_categorical_feedback
        )
        kwargs.setdefault("group_feedback", self.training_group_feedback)
        kwargs.setdefault(
            "temperature_feedback", self.training_temperature_feedback
        )
        return super().elbo(*args, **kwargs)


def build_stage3_model(config: dict[str, Any]) -> ConfiguredStage3Model:
    item = config["model"]
    training = config["training"]

    return ConfiguredStage3Model(
        in_channels=int(item["in_channels"]),
        num_classes=int(item["num_classes"]),
        num_components=int(item["num_components"]),
        latent_dims={str(k): int(v) for k, v in item["latent_dims"].items()},
        num_channels=tuple(int(v) for v in item["num_channels"]),
        group_feature_indices={
            str(k): int(v)
            for k, v in item["group_feature_indices"].items()
        },
        group_decoder_indices={
            str(k): int(v)
            for k, v in item["group_decoder_indices"].items()
        },
        num_convs_per_block=int(item["num_convs_per_block"]),
        fusion_1x1_convs=int(item["fusion_1x1_convs"]),
        segmentation_head_1x1_convs=int(item["segmentation_head_1x1_convs"]),
        min_log_sigma=item.get("min_log_sigma"),
        max_log_sigma=item.get("max_log_sigma"),
        evidence_projection_channels=int(item["evidence_projection_channels"]),
        evidence_fusion_channels=int(item["evidence_fusion_channels"]),
        feedback_config=item["feedback"],
        vitality_config=item["vitality"],
        feedback_enabled=bool(training["feedback_enabled"]),
        categorical_feedback=bool(training["categorical_feedback"]),
        group_feedback=bool(training["group_feedback"]),
        temperature_feedback=bool(training["temperature_feedback"]),
    )


def build_stage3_best_records() -> dict[str, dict[str, Any]]:
    return {
        "val_joint_loss": {
            "value": float("inf"),
            "epoch": None,
            "checkpoint": "best_val_joint_loss_eg_gm_punet_v2_stage3.pth",
        },
        "val_probabilistic_dice": {
            "value": float("-inf"),
            "epoch": None,
            "checkpoint": "best_val_probabilistic_dice_eg_gm_punet_v2_stage3.pth",
        },
        "val_prior_ged4": {
            "value": float("inf"),
            "epoch": None,
            "checkpoint": "best_val_ged4_eg_gm_punet_v2_stage3.pth",
        },
        "val_evidence_dice": {
            "value": float("-inf"),
            "epoch": None,
            "checkpoint": "best_val_evidence_dice_eg_gm_punet_v2_stage3.pth",
        },
        "val_evidence_ece": {
            "value": float("inf"),
            "epoch": None,
            "checkpoint": "best_val_evidence_ece_eg_gm_punet_v2_stage3.pth",
        },
    }


# =============================================================================
# 在线诊断辅助：D_ann 与相关性
# =============================================================================

def _rankdata(values: np.ndarray) -> np.ndarray:
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


def _corr(x: list[float], y: list[float], spearman: bool = False) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    if spearman:
        a, b = _rankdata(a), _rankdata(b)
    return float(np.corrcoef(a, b)[0, 1])


def _annotation_diversity(all_masks: torch.Tensor, eps: float = 1e-7):
    masks = all_masks.detach().float()
    if masks.ndim == 5 and masks.shape[2] == 1:
        masks = masks[:, :, 0]
    batch, raters = masks.shape[:2]
    distances = []
    for a in range(raters):
        ma = masks[:, a].reshape(batch, -1)
        for b in range(a + 1, raters):
            mb = masks[:, b].reshape(batch, -1)
            inter = (ma * mb).sum(dim=1)
            union = ((ma + mb) > 0).float().sum(dim=1)
            iou = torch.where(
                union <= eps,
                torch.ones_like(union),
                inter / (union + eps),
            )
            distances.append(1.0 - iou)
    return torch.stack(distances, dim=1).mean(dim=1)


# =============================================================================
# Patch validated accumulator
# =============================================================================

_PREV_NEW = legacy.base.new_accumulator
_PREV_UPDATE = legacy.base.update_accumulator
_PREV_FINISH = legacy.base.finish_accumulator
_PREV_RUN_EPOCH = legacy.base.run_epoch


def stage3_new_accumulator(model, ece_bins: int, ranking_bins: int):
    acc = _PREV_NEW(model=model, ece_bins=ece_bins, ranking_bins=ranking_bins)

    names = [
        "raw_demand",
        "ambiguity_demand",
        "categorical_capacity",
        "categorical_error",
    ]
    for name in names:
        acc[f"stage3_{name}_sum"] = 0.0

    for group in model.group_names:
        for name in ("capacity", "error", "raw_kl"):
            acc[f"stage3_{group}_{name}_sum"] = 0.0

    acc["stage3_case_demand"] = []
    acc["stage3_case_annotation"] = []
    acc["stage3_case_cat_capacity"] = []
    acc["stage3_case_cat_error"] = []

    acc["stage3_cat_deficit_sum"] = 0.0
    acc["stage3_cat_surplus_sum"] = 0.0
    for group in model.group_names:
        acc[f"stage3_{group}_deficit_sum"] = 0.0
        acc[f"stage3_{group}_surplus_sum"] = 0.0

    acc["stage3_vitality_raw_loss_sum"] = 0.0
    acc["stage3_vitality_weighted_loss_sum"] = 0.0
    acc["stage3_vitality_gate_fraction_sum"] = 0.0
    acc["stage3_vitality_below_prior_sum"] = 0.0
    acc["stage3_vitality_below_posterior_sum"] = 0.0
    acc["stage3_vitality_prior_occupancy_sum"] = torch.zeros(model.num_components, dtype=torch.float64)
    acc["stage3_vitality_posterior_occupancy_sum"] = torch.zeros(model.num_components, dtype=torch.float64)
    acc["stage3_case_k_eff"] = []
    acc["stage3_case_active_k"] = []
    return acc


def stage3_update_accumulator(
    accumulator,
    output,
    state,
    segmentation,
    all_segmentations,
    objective_loss,
    regularization_loss,
    model,
):
    _PREV_UPDATE(
        accumulator=accumulator,
        output=output,
        state=state,
        segmentation=segmentation,
        all_segmentations=all_segmentations,
        objective_loss=objective_loss,
        regularization_loss=regularization_loss,
        model=model,
    )

    c = output.adaptive_controls
    batch = int(segmentation.shape[0])

    accumulator["stage3_raw_demand_sum"] += float(c.raw_demand.sum().item())
    accumulator["stage3_ambiguity_demand_sum"] += float(
        c.ambiguity_demand.sum().item()
    )
    accumulator["stage3_categorical_capacity_sum"] += float(
        c.categorical_capacity.sum().item()
    )
    accumulator["stage3_categorical_error_sum"] += float(
        c.categorical_error.sum().item()
    )

    for group in model.group_names:
        accumulator[f"stage3_{group}_capacity_sum"] += float(
            c.group_capacity[group].sum().item()
        )
        accumulator[f"stage3_{group}_error_sum"] += float(
            c.group_error[group].sum().item()
        )
        accumulator[f"stage3_{group}_raw_kl_sum"] += float(
            c.raw_group_kl[group].sum().item()
        )

    ann = _annotation_diversity(all_segmentations)
    accumulator["stage3_case_demand"].extend(
        c.raw_demand.detach().cpu().reshape(-1).tolist()
    )
    accumulator["stage3_case_annotation"].extend(
        ann.detach().cpu().reshape(-1).tolist()
    )
    accumulator["stage3_case_cat_capacity"].extend(
        c.categorical_capacity.detach().cpu().reshape(-1).tolist()
    )
    accumulator["stage3_case_cat_error"].extend(
        c.categorical_error.detach().cpu().reshape(-1).tolist()
    )

    accumulator["stage3_cat_deficit_sum"] += float(
        c.categorical_deficit.sum().item()
    )
    accumulator["stage3_cat_surplus_sum"] += float(
        c.categorical_surplus.sum().item()
    )

    for group in model.group_names:
        accumulator[f"stage3_{group}_deficit_sum"] += float(
            c.group_deficit[group].sum().item()
        )
        accumulator[f"stage3_{group}_surplus_sum"] += float(
            c.group_surplus[group].sum().item()
        )

    v = output.vitality
    batch_size = int(segmentation.shape[0])
    accumulator["stage3_vitality_raw_loss_sum"] += float(v.raw_loss.detach().item()) * batch_size
    accumulator["stage3_vitality_weighted_loss_sum"] += float(v.weighted_loss.detach().item()) * batch_size
    accumulator["stage3_vitality_gate_fraction_sum"] += float(v.gate_fraction.detach().item()) * batch_size
    accumulator["stage3_vitality_below_prior_sum"] += float(v.prior_below_floor_count.detach().item()) * batch_size
    accumulator["stage3_vitality_below_posterior_sum"] += float(v.posterior_below_floor_count.detach().item()) * batch_size
    accumulator["stage3_vitality_prior_occupancy_sum"] += v.prior_weighted_occupancy.detach().cpu().double() * batch_size
    accumulator["stage3_vitality_posterior_occupancy_sum"] += v.posterior_weighted_occupancy.detach().cpu().double() * batch_size

    prior_probs = state.prior.mixture_probs.detach()
    safe = prior_probs.clamp_min(1e-8)
    k_eff = torch.exp(-(safe * safe.log()).sum(dim=1))
    active = (prior_probs >= float(model.component_vitality.config.active_threshold)).sum(dim=1)
    accumulator["stage3_case_k_eff"].extend(k_eff.cpu().tolist())
    accumulator["stage3_case_active_k"].extend(active.cpu().tolist())


def stage3_finish_accumulator(accumulator, model):
    result = _PREV_FINISH(accumulator=accumulator, model=model)
    count = max(int(accumulator["samples"]), 1)

    for name in (
        "raw_demand",
        "ambiguity_demand",
        "categorical_capacity",
        "categorical_error",
    ):
        result[f"stage3_{name}"] = accumulator[f"stage3_{name}_sum"] / count

    for group in model.group_names:
        for name in ("capacity", "error", "raw_kl"):
            result[f"stage3_{group}_{name}"] = (
                accumulator[f"stage3_{group}_{name}_sum"] / count
            )

    ann = accumulator["stage3_case_annotation"]
    demand = accumulator["stage3_case_demand"]
    result["demand_annotation_pearson"] = _corr(ann, demand, False)
    result["demand_annotation_spearman"] = _corr(ann, demand, True)
    result["demand_annotation_num_cases"] = float(len(demand))

    ann_arr = np.asarray(ann, dtype=float)
    demand_arr = np.asarray(demand, dtype=float)
    cap_arr = np.asarray(accumulator["stage3_case_cat_capacity"], dtype=float)
    err_arr = np.asarray(accumulator["stage3_case_cat_error"], dtype=float)

    if len(ann_arr) >= 4:
        q25, q50, q75 = np.quantile(ann_arr, [0.25, 0.5, 0.75])
        selectors = [
            ann_arr <= q25,
            (ann_arr > q25) & (ann_arr <= q50),
            (ann_arr > q50) & (ann_arr <= q75),
            ann_arr > q75,
        ]
        for q, selector in enumerate(selectors, 1):
            result[f"demand_q{q}_mean"] = float(demand_arr[selector].mean())
            result[f"cat_capacity_q{q}_mean"] = float(cap_arr[selector].mean())
            result[f"cat_error_q{q}_mean"] = float(err_arr[selector].mean())
            result[f"annotation_q{q}_mean"] = float(ann_arr[selector].mean())
            result[f"demand_q{q}_count"] = float(selector.sum())

    result["stage3_categorical_deficit"] = (
        accumulator["stage3_cat_deficit_sum"] / count
    )
    result["stage3_categorical_surplus"] = (
        accumulator["stage3_cat_surplus_sum"] / count
    )

    for group in model.group_names:
        result[f"stage3_{group}_deficit"] = (
            accumulator[f"stage3_{group}_deficit_sum"] / count
        )
        result[f"stage3_{group}_surplus"] = (
            accumulator[f"stage3_{group}_surplus_sum"] / count
        )

    result["stage3_vitality_raw_loss"] = accumulator["stage3_vitality_raw_loss_sum"] / count
    result["stage3_vitality_weighted_loss"] = accumulator["stage3_vitality_weighted_loss_sum"] / count
    result["stage3_vitality_gate_fraction"] = accumulator["stage3_vitality_gate_fraction_sum"] / count
    result["stage3_vitality_below_prior"] = accumulator["stage3_vitality_below_prior_sum"] / count
    result["stage3_vitality_below_posterior"] = accumulator["stage3_vitality_below_posterior_sum"] / count
    result["stage3_vitality_prior_occupancy"] = (accumulator["stage3_vitality_prior_occupancy_sum"] / count).tolist()
    result["stage3_vitality_posterior_occupancy"] = (accumulator["stage3_vitality_posterior_occupancy_sum"] / count).tolist()
    result["stage3_mean_per_case_k_eff"] = float(np.mean(accumulator["stage3_case_k_eff"]))
    result["stage3_mean_active_k"] = float(np.mean(accumulator["stage3_case_active_k"]))

    return result


def stage3_run_epoch(*args, **kwargs):
    metrics, step = _PREV_RUN_EPOCH(*args, **kwargs)
    training = bool(kwargs.get("training", False))
    epoch = int(kwargs.get("epoch", 0))

    if training:
        print(
            "  [Stage3 Conditional Vitality] "
            f"epoch={epoch}, U={metrics['stage3_ambiguity_demand']:.3f}, "
            f"Ccat={metrics['stage3_categorical_capacity']:.3f}, "
            f"Ecat={metrics['stage3_categorical_error']:.3f}, "
            f"betaC={metrics['adaptive_beta_categorical']:.3f}, "
            f"tau={metrics['adaptive_temperature']:.3f}, "
            f"Vraw={metrics['stage3_vitality_raw_loss']:.4f}, "
            f"Vw={metrics['stage3_vitality_weighted_loss']:.5f}, "
            f"gate={metrics['stage3_vitality_gate_fraction']:.3f}, "
            f"belowP/Q={metrics['stage3_vitality_below_prior']:.2f}/"
            f"{metrics['stage3_vitality_below_posterior']:.2f}, "
            f"Keff={metrics['stage3_mean_per_case_k_eff']:.3f}, "
            f"Kactive={metrics['stage3_mean_active_k']:.3f}"
        )
    else:
        print(
            "  [Val Demand-Annotation] "
            f"epoch={epoch}, "
            f"Pearson={metrics['demand_annotation_pearson']:.4f}, "
            f"Spearman={metrics['demand_annotation_spearman']:.4f}, "
            f"cases={int(metrics['demand_annotation_num_cases'])}"
        )
        if "demand_q1_mean" in metrics:
            parts = []
            for q in range(1, 5):
                parts.append(
                    f"Q{q}:D={metrics[f'demand_q{q}_mean']:.4f},"
                    f"Ccat={metrics[f'cat_capacity_q{q}_mean']:.3f},"
                    f"E={metrics[f'cat_error_q{q}_mean']:.3f}"
                )
            print("  [Val Demand-Capacity Quartiles] " + " | ".join(parts))

    return metrics, step


legacy.base.new_accumulator = stage3_new_accumulator
legacy.base.update_accumulator = stage3_update_accumulator
legacy.base.finish_accumulator = stage3_finish_accumulator
legacy.base.run_epoch = stage3_run_epoch
legacy.build_model = build_stage3_model
legacy.build_best_records = build_stage3_best_records


def _prepare_smoke_config(config_path: Path) -> Path:
    """
    Smoke 只用于验证 Demand–Capacity 数学链，因此强制第一轮就启用 full ramp。
    正式 YAML 不修改。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["model"]["feedback"]["start_epoch"] = 1
    config["model"]["feedback"]["ramp_epochs"] = 1
    config["model"]["vitality"]["start_epoch"] = 1
    config["model"]["vitality"]["ramp_epochs"] = 1
    config["initialization"]["probabilistic_feedback_start_epoch"] = 1
    config["initialization"]["probabilistic_freeze_probabilistic_epochs"] = 0
    config["training"]["freeze_probabilistic_epochs"] = 0

    path = (
        PROJECT_ROOT
        / "result"
        / "eg_gm_punet_v2_conditional_vitality_k5_full"
        / "stage3_smoke_runtime_config.yaml"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        default="configs/eg_gm_punet_v2_conditional_vitality_lidc.yaml",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", default=None)
    known, _ = parser.parse_known_args()

    config_path = Path(known.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if known.smoke_test:
        config_path = _prepare_smoke_config(config_path)

    argv = [sys.argv[0], "--config", str(config_path)]
    if known.smoke_test:
        argv.append("--smoke-test")
    if known.resume:
        argv.extend(["--resume", str(known.resume)])
    sys.argv = argv

    print("=" * 108)
    print("EG-GM-PUNet v2 Stage 3: Conditional Component Vitality")
    print("=" * 108)
    print("[INFO] Demand   : prior fine foreground expected-data entropy")
    print("[INFO] C_cat    : normalized prior mixture entropy")
    print("[INFO] C_groups : EMA-relative posterior-prior group KL information")
    print("[INFO] Control  : bounded asymmetric correction; beta>=1, tau<=1")
    print("[INFO] Hard boundary: beta >= 1.0, tau <= 1.0")
    print("[INFO] Stage 3 Component Vitality: ON (conditional floor=0.03)")
    print("[INFO] Stage 4 Ambiguity Alignment: OFF")
    print("[INFO] Vitality gate: categorical deficit > 0.05; floor protects survival, not uniformity")
    if known.smoke_test:
        print("[INFO] Smoke override: feedback start=1, ramp=1, freeze=0")

    legacy.main()


if __name__ == "__main__":
    main()
