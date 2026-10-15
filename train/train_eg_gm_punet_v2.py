"""
EG-GM-PUNet v2 Stage 1 training entry.

This wrapper reuses the already validated train/train_eg_gm_punet.py loop and
changes only the model/controller used for Stage 1.

Smoke test:
    python train/train_eg_gm_punet_v2.py ^
      --config configs/eg_gm_punet_v2_relative_feedback_lidc.yaml ^
      --smoke-test

Formal training:
    python train/train_eg_gm_punet_v2.py ^
      --config configs/eg_gm_punet_v2_relative_feedback_lidc.yaml
"""
from __future__ import annotations

import argparse
import importlib.util
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
        "Stage 1 intentionally reuses the validated first-version training "
        f"loop, but it was not found: {LEGACY_TRAIN}"
    )

spec = importlib.util.spec_from_file_location(
    "_eg_gm_punet_stage1_legacy_train",
    LEGACY_TRAIN,
)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot import: {LEGACY_TRAIN}")
legacy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)

from models.EG_GM_PUNet_v2 import EGGMPUNetV2Relative


class ConfiguredStage1Model(EGGMPUNetV2Relative):
    """Store training feedback switches on the Stage 1 model."""

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


def build_stage1_model(config: dict[str, Any]) -> ConfiguredStage1Model:
    item = config["model"]
    training = config["training"]
    return ConfiguredStage1Model(
        in_channels=int(item["in_channels"]),
        num_classes=int(item["num_classes"]),
        num_components=int(item["num_components"]),
        latent_dims={
            str(name): int(value) for name, value in item["latent_dims"].items()
        },
        num_channels=tuple(int(value) for value in item["num_channels"]),
        group_feature_indices={
            str(name): int(value)
            for name, value in item["group_feature_indices"].items()
        },
        group_decoder_indices={
            str(name): int(value)
            for name, value in item["group_decoder_indices"].items()
        },
        num_convs_per_block=int(item["num_convs_per_block"]),
        fusion_1x1_convs=int(item["fusion_1x1_convs"]),
        segmentation_head_1x1_convs=int(item["segmentation_head_1x1_convs"]),
        min_log_sigma=item.get("min_log_sigma"),
        max_log_sigma=item.get("max_log_sigma"),
        evidence_projection_channels=int(item["evidence_projection_channels"]),
        evidence_fusion_channels=int(item["evidence_fusion_channels"]),
        feedback_config=item["feedback"],
        feedback_enabled=bool(training["feedback_enabled"]),
        categorical_feedback=bool(training["categorical_feedback"]),
        group_feedback=bool(training["group_feedback"]),
        temperature_feedback=bool(training["temperature_feedback"]),
    )


def build_stage1_best_records() -> dict[str, dict[str, Any]]:
    return {
        "val_joint_loss": {
            "value": float("inf"),
            "epoch": None,
            "checkpoint": "best_val_joint_loss_eg_gm_punet_v2_stage1.pth",
        },
        "val_probabilistic_dice": {
            "value": float("-inf"),
            "epoch": None,
            "checkpoint": "best_val_probabilistic_dice_eg_gm_punet_v2_stage1.pth",
        },
        "val_prior_ged4": {
            "value": float("inf"),
            "epoch": None,
            "checkpoint": "best_val_ged4_eg_gm_punet_v2_stage1.pth",
        },
        "val_evidence_dice": {
            "value": float("-inf"),
            "epoch": None,
            "checkpoint": "best_val_evidence_dice_eg_gm_punet_v2_stage1.pth",
        },
        "val_evidence_ece": {
            "value": float("inf"),
            "epoch": None,
            "checkpoint": "best_val_evidence_ece_eg_gm_punet_v2_stage1.pth",
        },
    }



# ---------------------------------------------------------------------------
# Stage 1 validation ambiguity-demand diagnostics
# ---------------------------------------------------------------------------

def _rankdata(values: np.ndarray) -> np.ndarray:
    """平均秩实现，用于 Spearman，支持并列值。"""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        next_index = index + 1
        while (
            next_index < len(values)
            and values[order[next_index]] == values[order[index]]
        ):
            next_index += 1
        rank = 0.5 * (index + next_index - 1) + 1.0
        ranks[order[index:next_index]] = rank
        index = next_index
    return ranks


def _correlation(
    first: list[float],
    second: list[float],
    *,
    spearman: bool = False,
) -> float:
    """计算 Pearson / Spearman；样本不足或常数列返回 NaN。"""
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)

    valid = np.isfinite(first_array) & np.isfinite(second_array)
    first_array = first_array[valid]
    second_array = second_array[valid]

    if (
        first_array.size < 3
        or np.std(first_array) < 1e-12
        or np.std(second_array) < 1e-12
    ):
        return float("nan")

    if spearman:
        first_array = _rankdata(first_array)
        second_array = _rankdata(second_array)

    return float(np.corrcoef(first_array, second_array)[0, 1])


def _annotation_diversity_per_case(
    all_segmentations: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    计算 batch 内每个病例的四医生 annotation diversity：
        D_ann = 所有医生两两 (1 - IoU) 的平均值。

    只用于 validation 在线诊断；
    不参与 loss、不反向传播、不进入 controller。
    """
    masks = all_segmentations.detach().float()

    if masks.ndim == 5 and masks.shape[2] == 1:
        masks = masks[:, :, 0]

    if masks.ndim != 4:
        raise ValueError(
            "all_segmentations 必须为 [B,R,H,W] 或 [B,R,1,H,W]，"
            f"实际为 {tuple(masks.shape)}"
        )

    batch_size, num_raters = masks.shape[:2]

    if num_raters < 2:
        return torch.zeros(
            batch_size,
            device=masks.device,
            dtype=masks.dtype,
        )

    pairwise_distances = []

    for first in range(num_raters):
        first_mask = masks[:, first].reshape(batch_size, -1)

        for second in range(first + 1, num_raters):
            second_mask = masks[:, second].reshape(batch_size, -1)

            intersection = (first_mask * second_mask).sum(dim=1)
            union = ((first_mask + second_mask) > 0).float().sum(dim=1)

            # 两个标注都为空时按 IoU=1，即 diversity=0。
            iou = torch.where(
                union <= eps,
                torch.ones_like(union),
                intersection / (union + eps),
            )

            pairwise_distances.append(1.0 - iou)

    return torch.stack(pairwise_distances, dim=1).mean(dim=1)


def _demand_annotation_diagnostics(
    demands: list[float],
    annotation_diversities: list[float],
) -> dict[str, float]:
    """
    计算：
    1. raw ambiguity demand 与 annotation diversity 的 Pearson；
    2. Spearman；
    3. 按 annotation diversity 四分位划分后的 Q1-Q4 mean demand。
    """
    result = {
        "demand_annotation_pearson": float("nan"),
        "demand_annotation_spearman": float("nan"),
        "demand_annotation_num_cases": 0.0,
    }

    for q in range(1, 5):
        result[f"demand_q{q}_mean"] = float("nan")
        result[f"annotation_q{q}_mean"] = float("nan")
        result[f"demand_q{q}_count"] = 0.0

    demand_array = np.asarray(demands, dtype=np.float64)
    ann_array = np.asarray(annotation_diversities, dtype=np.float64)

    valid = np.isfinite(demand_array) & np.isfinite(ann_array)
    demand_array = demand_array[valid]
    ann_array = ann_array[valid]

    result["demand_annotation_num_cases"] = float(demand_array.size)

    if demand_array.size < 3:
        return result

    result["demand_annotation_pearson"] = _correlation(
        ann_array.tolist(),
        demand_array.tolist(),
        spearman=False,
    )
    result["demand_annotation_spearman"] = _correlation(
        ann_array.tolist(),
        demand_array.tolist(),
        spearman=True,
    )

    q25, q50, q75 = np.quantile(ann_array, [0.25, 0.50, 0.75])

    selectors = [
        ann_array <= q25,
        (ann_array > q25) & (ann_array <= q50),
        (ann_array > q50) & (ann_array <= q75),
        ann_array > q75,
    ]

    for q, selector in enumerate(selectors, start=1):
        count = int(selector.sum())
        result[f"demand_q{q}_count"] = float(count)

        if count > 0:
            result[f"demand_q{q}_mean"] = float(
                demand_array[selector].mean()
            )
            result[f"annotation_q{q}_mean"] = float(
                ann_array[selector].mean()
            )

    return result


# ---------------------------------------------------------------------------
# Add Stage-1-specific diagnostics while retaining every old metric.
# ---------------------------------------------------------------------------
_PREV_NEW = legacy.base.new_accumulator
_PREV_UPDATE = legacy.base.update_accumulator
_PREV_FINISH = legacy.base.finish_accumulator
_PREV_RUN_EPOCH = legacy.base.run_epoch


def stage1_new_accumulator(model, ece_bins: int, ranking_bins: int):
    acc = _PREV_NEW(model=model, ece_bins=ece_bins, ranking_bins=ranking_bins)
    acc["stage1_raw_demand_sum"] = 0.0
    acc["stage1_z_sum"] = 0.0
    acc["stage1_z_min"] = float("inf")
    acc["stage1_z_max"] = float("-inf")
    acc["stage1_relative_sum"] = 0.0
    acc["stage1_relative_min"] = float("inf")
    acc["stage1_relative_max"] = float("-inf")
    acc["stage1_running_mean_sum"] = 0.0
    acc["stage1_running_std_sum"] = 0.0

    # 保存逐病例 raw ambiguity demand 与四医生 annotation diversity。
    # 这里只用于 epoch 结束后的 validation diagnosis。
    acc["stage1_case_demands"] = []
    acc["stage1_case_annotation_diversities"] = []
    return acc


def stage1_update_accumulator(
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
    controls = output.adaptive_controls
    raw = controls.raw_demand.detach().float()
    z = controls.demand_z_score.detach().float()
    rel = controls.relative_demand.detach().float()
    batch_size = int(segmentation.shape[0])

    accumulator["stage1_raw_demand_sum"] += float(raw.sum().item())
    accumulator["stage1_z_sum"] += float(z.sum().item())
    accumulator["stage1_z_min"] = min(
        accumulator["stage1_z_min"], float(z.min().item())
    )
    accumulator["stage1_z_max"] = max(
        accumulator["stage1_z_max"], float(z.max().item())
    )
    accumulator["stage1_relative_sum"] += float(rel.sum().item())
    accumulator["stage1_relative_min"] = min(
        accumulator["stage1_relative_min"], float(rel.min().item())
    )
    accumulator["stage1_relative_max"] = max(
        accumulator["stage1_relative_max"], float(rel.max().item())
    )
    accumulator["stage1_running_mean_sum"] += (
        float(controls.running_mean.item()) * batch_size
    )
    accumulator["stage1_running_std_sum"] += (
        float(controls.running_std.item()) * batch_size
    )

    accumulator["stage1_case_demands"].extend(
        raw.detach().cpu().reshape(-1).tolist()
    )

    annotation_diversity = _annotation_diversity_per_case(
        all_segmentations
    )
    accumulator["stage1_case_annotation_diversities"].extend(
        annotation_diversity.detach().cpu().reshape(-1).tolist()
    )


def stage1_finish_accumulator(accumulator, model):
    result = _PREV_FINISH(accumulator=accumulator, model=model)
    count = max(int(accumulator["samples"]), 1)
    result["stage1_raw_demand"] = accumulator["stage1_raw_demand_sum"] / count
    result["stage1_z"] = accumulator["stage1_z_sum"] / count
    result["stage1_z_min"] = accumulator["stage1_z_min"]
    result["stage1_z_max"] = accumulator["stage1_z_max"]
    result["stage1_relative_demand"] = accumulator["stage1_relative_sum"] / count
    result["stage1_relative_demand_min"] = accumulator["stage1_relative_min"]
    result["stage1_relative_demand_max"] = accumulator["stage1_relative_max"]
    result["stage1_running_mean"] = accumulator["stage1_running_mean_sum"] / count
    result["stage1_running_std"] = accumulator["stage1_running_std_sum"] / count

    result.update(
        _demand_annotation_diagnostics(
            accumulator["stage1_case_demands"],
            accumulator["stage1_case_annotation_diversities"],
        )
    )

    return result


def stage1_run_epoch(*args, **kwargs):
    metrics, step = _PREV_RUN_EPOCH(*args, **kwargs)
    training = bool(kwargs.get("training", False))
    epoch = int(kwargs.get("epoch", 0))
    if training:
        print(
            "  [Stage1 Relative Demand] "
            f"epoch={epoch}, "
            f"raw={metrics['stage1_raw_demand']:.4f}, "
            f"EMA={metrics['stage1_running_mean']:.4f}±"
            f"{metrics['stage1_running_std']:.4f}, "
            f"z={metrics['stage1_z']:.3f} "
            f"[{metrics['stage1_z_min']:.3f},{metrics['stage1_z_max']:.3f}], "
            f"Urel={metrics['stage1_relative_demand']:.3f} "
            f"[{metrics['stage1_relative_demand_min']:.3f},"
            f"{metrics['stage1_relative_demand_max']:.3f}], "
            f"betaC=[{metrics['adaptive_beta_categorical_min']:.3f},"
            f"{metrics['adaptive_beta_categorical_max']:.3f}], "
            f"tau=[{metrics['adaptive_temperature_min']:.3f},"
            f"{metrics['adaptive_temperature_max']:.3f}]"
        )
    else:
        print(
            "  [Val Demand-Annotation] "
            f"epoch={epoch}, "
            f"Pearson={metrics['demand_annotation_pearson']:.4f}, "
            f"Spearman={metrics['demand_annotation_spearman']:.4f}, "
            f"cases={int(metrics['demand_annotation_num_cases'])}"
        )

        quartile_parts = []
        for q in range(1, 5):
            quartile_parts.append(
                f"Q{q}={metrics[f'demand_q{q}_mean']:.4f} "
                f"(D_ann={metrics[f'annotation_q{q}_mean']:.4f}, "
                f"n={int(metrics[f'demand_q{q}_count'])})"
            )

        print(
            "  [Val Demand Quartiles] "
            + ", ".join(quartile_parts)
        )

    return metrics, step


legacy.base.new_accumulator = stage1_new_accumulator
legacy.base.update_accumulator = stage1_update_accumulator
legacy.base.finish_accumulator = stage1_finish_accumulator
legacy.base.run_epoch = stage1_run_epoch
legacy.build_model = build_stage1_model
legacy.build_best_records = build_stage1_best_records


def _prepare_smoke_config(config_path: Path) -> Path:
    """Enable feedback immediately only for the two-batch smoke test."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping.")

    config["model"]["feedback"]["start_epoch"] = 1
    config["model"]["feedback"]["ramp_epochs"] = 1
    config["initialization"]["probabilistic_feedback_start_epoch"] = 1
    config["initialization"]["probabilistic_freeze_probabilistic_epochs"] = 0
    config["training"]["freeze_probabilistic_epochs"] = 0

    runtime_path = (
        PROJECT_ROOT
        / "result"
        / "eg_gm_punet_v2_relative_feedback_k5_full"
        / "stage1_smoke_runtime_config.yaml"
    )
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    with open(runtime_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    return runtime_path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        default="configs/eg_gm_punet_v2_relative_feedback_lidc.yaml",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", default=None)
    known, _ = parser.parse_known_args()

    config_path = Path(known.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if known.smoke_test:
        config_path = _prepare_smoke_config(config_path)

    # Rebuild argv for the validated legacy main().
    argv = [sys.argv[0], "--config", str(config_path)]
    if known.smoke_test:
        argv.append("--smoke-test")
    if known.resume:
        argv.extend(["--resume", str(known.resume)])
    sys.argv = argv

    print("=" * 100)
    print("EG-GM-PUNet v2 Stage 1: Relative Ambiguity-Demand Feedback")
    print("=" * 100)
    print("[INFO] Demand: prior / fine / foreground-weighted expected-data entropy")
    print("[INFO] Relative normalization: EMA z-score -> sigmoid -> neutral 0.5")
    print("[INFO] Stage 2/3/4 mechanisms: disabled by design")
    if known.smoke_test:
        print("[INFO] Smoke override: feedback start=1, ramp=1, probability freeze=0")

    legacy.main()


if __name__ == "__main__":
    main()