"""
EG-GM-PUNet v2 Stage 1：正式测试集评估脚本。

用途
----
Stage 1（Relative Ambiguity-Demand Feedback）训练完成后，用与第一版 Full Feedback /
No-Feedback 完全一致的测试口径进行评估：

1. test split；
2. prior samples = 16；
3. sample counts = [1, 4, 8, 16]；
4. 计算 GED²、Cross Dice、Best-sample Dice、sample diversity 等；
5. 计算 Evidential Dice / ECE / error AUROC / disagreement AUROC；
6. 统计 GMM component 使用情况；
7. 额外评估 Stage 1 机制：
   - raw ambiguity demand 与 annotation diversity 的 Pearson / Spearman；
   - patient-level Pearson / Spearman；
   - Q1-Q4 ambiguity demand；
   - relative demand / beta / tau 的病例级分布；
8. 保存逐病例 CSV、summary.json 与诊断图。

重要
----
- 测试阶段不更新 EMA running statistics，因为 model.eval()；
- 医生标注仅用于离线评估，不进入模型 feedback；
- 概率分支指标使用与此前 EG-GM-PUNet 相同的 utils.gmm_prior_evaluation；
- Stage 1 的预测本身仍由 prior sampling 得到，controller 只用于机制诊断。

放置位置建议：
    E:/workspace/EG-GM-PUNet/evaluate/evaluate_eg_gm_punet_v2_stage1.py

Smoke test：
    python evaluate/evaluate_eg_gm_punet_v2_stage1.py --smoke-test

正式评估：
    python evaluate/evaluate_eg_gm_punet_v2_stage1.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from itertools import combinations, islice
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.lidc_dataset_full import build_lidc_full_dataloader
from models.EG_GM_PUNet_v2 import EGGMPUNetV2Relative
from models.Evidential_Multi_scale_Gaussian_Mixture_prob_unet.evidential_multiscale_gmm_parts import (
    multi_rater_soft_target,
)
from models.Multi_scale_Gaussian_Mixture_prob_unet.multiscale_grouped_gmm_parts import (
    sample_grouped_exact,
)
from utils.evidential_metrics import EvidentialMetricAccumulator
from utils.gmm_experiment_utils import prepare_device, set_seed
from utils.gmm_prior_evaluation import (
    evaluate_single_case,
    normalize_sample_counts,
    safe_filename,
    summarize_rows,
    write_csv,
)


# =============================================================================
# 基础工具
# =============================================================================

def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"YAML 顶层必须为字典：{path}")
    return value


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def ids_from_batch(
    value: Any,
    batch_size: int,
    prefix: str,
    offset: int,
) -> list[str]:
    if value is None:
        return [f"{prefix}_{offset + i:06d}" for i in range(batch_size)]
    if isinstance(value, str):
        return [value] * batch_size
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)] * batch_size


def patient_id_from_case_id(case_id: str) -> str:
    match = re.search(r"(LIDC-IDRI-\d{4})", str(case_id))
    if match:
        return match.group(1)
    return str(case_id).split("_")[0]


def build_stage1_model(config: dict[str, Any]) -> EGGMPUNetV2Relative:
    item = config["model"]
    return EGGMPUNetV2Relative(
        in_channels=int(item["in_channels"]),
        num_classes=int(item["num_classes"]),
        num_components=int(item["num_components"]),
        latent_dims={
            str(name): int(value)
            for name, value in item["latent_dims"].items()
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
        feedback_config=item.get("feedback"),
    )


# =============================================================================
# Annotation ambiguity 与相关性
# =============================================================================

def binary_iou(mask_a: torch.Tensor, mask_b: torch.Tensor, eps: float = 1e-7) -> float:
    a = mask_a.float().reshape(-1)
    b = mask_b.float().reshape(-1)
    intersection = (a * b).sum()
    union = ((a + b) > 0).float().sum()
    if float(union.item()) <= eps:
        return 1.0
    return float((intersection / (union + eps)).item())


def annotation_diversity(masks: torch.Tensor) -> float:
    """四医生 pairwise (1-IoU) 的平均值。"""
    value = masks
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    distances = [
        1.0 - binary_iou(value[first], value[second])
        for first, second in combinations(range(value.shape[0]), 2)
    ]
    return float(np.mean(distances)) if distances else 0.0


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
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


def correlation(
    first: Iterable[float],
    second: Iterable[float],
    *,
    spearman: bool = False,
) -> float | None:
    x = np.asarray(list(first), dtype=float)
    y = np.asarray(list(second), dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    if spearman:
        x, y = rankdata(x), rankdata(y)
    return float(np.corrcoef(x, y)[0, 1])


def aggregate_patient_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["patient_id"]), []).append(row)

    outputs: list[dict[str, Any]] = []
    numeric_keys = (
        "annotation_diversity",
        "raw_demand",
        "demand_z_score",
        "relative_demand",
        "beta_categorical",
        "gumbel_temperature",
        "prior_k_eff",
        "active_component_count",
    )
    for patient_id, items in grouped.items():
        out: dict[str, Any] = {
            "patient_id": patient_id,
            "num_slices": len(items),
        }
        for key in numeric_keys:
            out[key] = float(np.mean([float(item[key]) for item in items]))
        outputs.append(out)
    return outputs


def quartile_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ann = np.asarray([float(row["annotation_diversity"]) for row in rows])
    q25, q50, q75 = np.quantile(ann, [0.25, 0.50, 0.75])
    selectors = [
        ann <= q25,
        (ann > q25) & (ann <= q50),
        (ann > q50) & (ann <= q75),
        ann > q75,
    ]

    outputs = []
    for index, selector in enumerate(selectors, start=1):
        selected = [rows[i] for i in np.where(selector)[0]]
        outputs.append(
            {
                "quartile": index,
                "num_cases": len(selected),
                "annotation_diversity_mean": float(
                    np.mean([float(row["annotation_diversity"]) for row in selected])
                ),
                "raw_demand_mean": float(
                    np.mean([float(row["raw_demand"]) for row in selected])
                ),
                "relative_demand_mean": float(
                    np.mean([float(row["relative_demand"]) for row in selected])
                ),
                "beta_categorical_mean": float(
                    np.mean([float(row["beta_categorical"]) for row in selected])
                ),
                "gumbel_temperature_mean": float(
                    np.mean([float(row["gumbel_temperature"]) for row in selected])
                ),
                "prior_k_eff_mean": float(
                    np.mean([float(row["prior_k_eff"]) for row in selected])
                ),
                "active_component_count_mean": float(
                    np.mean([float(row["active_component_count"]) for row in selected])
                ),
            }
        )
    return outputs


def summarize_demand(rows: list[dict[str, Any]]) -> dict[str, Any]:
    patient_rows = aggregate_patient_rows(rows)

    ann = [float(row["annotation_diversity"]) for row in rows]
    demand = [float(row["raw_demand"]) for row in rows]
    patient_ann = [float(row["annotation_diversity"]) for row in patient_rows]
    patient_demand = [float(row["raw_demand"]) for row in patient_rows]

    return {
        "num_cases": len(rows),
        "num_patients": len(patient_rows),
        "slice_level": {
            "pearson": correlation(ann, demand),
            "spearman": correlation(ann, demand, spearman=True),
        },
        "patient_level": {
            "pearson": correlation(patient_ann, patient_demand),
            "spearman": correlation(patient_ann, patient_demand, spearman=True),
        },
        "raw_demand": {
            "mean": float(np.mean(demand)),
            "std": float(np.std(demand)),
        },
        "relative_demand": {
            "mean": float(np.mean([float(row["relative_demand"]) for row in rows])),
            "std": float(np.std([float(row["relative_demand"]) for row in rows])),
            "min": float(np.min([float(row["relative_demand"]) for row in rows])),
            "max": float(np.max([float(row["relative_demand"]) for row in rows])),
        },
        "beta_categorical": {
            "mean": float(np.mean([float(row["beta_categorical"]) for row in rows])),
            "min": float(np.min([float(row["beta_categorical"]) for row in rows])),
            "max": float(np.max([float(row["beta_categorical"]) for row in rows])),
        },
        "gumbel_temperature": {
            "mean": float(np.mean([float(row["gumbel_temperature"]) for row in rows])),
            "min": float(np.min([float(row["gumbel_temperature"]) for row in rows])),
            "max": float(np.max([float(row["gumbel_temperature"]) for row in rows])),
        },
        "quartiles": quartile_summary(rows),
        "patient_rows": patient_rows,
    }


# =============================================================================
# 画图
# =============================================================================

def plot_sample_count_metrics(
    probabilistic_summary: dict[str, Any],
    sample_counts: list[int],
    output_dir: Path,
) -> None:
    definitions = [
        ("ged2", "GED²", "GED² (lower is better)"),
        ("cross_dice", "Cross Dice", "Cross Dice (higher is better)"),
        (
            "best_sample_dice_to_raters",
            "Best-sample Dice",
            "Best-sample Dice (higher is better)",
        ),
        (
            "sample_diversity_distance",
            "Sample diversity",
            "Sample diversity distance",
        ),
    ]
    for key, title, ylabel in definitions:
        xs, ys = [], []
        for count in sample_counts:
            metric = probabilistic_summary[str(count)].get(key, {})
            if isinstance(metric, dict) and "mean" in metric:
                xs.append(count)
                ys.append(float(metric["mean"]))
        if not ys:
            continue
        plt.figure(figsize=(7.5, 5.0))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Number of prior samples")
        plt.ylabel(ylabel)
        plt.title(f"Stage 1: {title} vs prior samples")
        plt.xticks(xs)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / f"{key}_vs_samples.png", dpi=180, bbox_inches="tight")
        plt.close()


def plot_demand_quartiles(summary: dict[str, Any], output_dir: Path) -> None:
    quartiles = summary["quartiles"]
    x = [item["quartile"] for item in quartiles]
    raw = [item["raw_demand_mean"] for item in quartiles]
    beta = [item["beta_categorical_mean"] for item in quartiles]
    tau = [item["gumbel_temperature_mean"] for item in quartiles]

    plt.figure(figsize=(7.5, 5.0))
    plt.plot(x, raw, marker="o")
    plt.xlabel("Annotation-diversity quartile")
    plt.ylabel("Raw ambiguity demand")
    plt.title("Stage 1 demand across annotation-diversity quartiles")
    plt.xticks(x, [f"Q{i}" for i in x])
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "stage1_demand_quartiles.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.5, 5.0))
    plt.plot(x, beta, marker="o", label="Categorical beta")
    plt.plot(x, tau, marker="o", label="Gumbel temperature")
    plt.xlabel("Annotation-diversity quartile")
    plt.ylabel("Controller value")
    plt.title("Stage 1 controller response across ambiguity quartiles")
    plt.xticks(x, [f"Q{i}" for i in x])
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "stage1_controller_quartiles.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_training_demand_history(result_root: Path, output_dir: Path) -> None:
    history_path = result_root / "history.csv"
    if not history_path.exists():
        print(f"[WARN] 未找到 history.csv，跳过 Stage 1 在线 demand 曲线：{history_path}")
        return

    with open(history_path, "r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    def series(key: str) -> tuple[list[float], list[float]]:
        xs, ys = [], []
        for i, row in enumerate(rows):
            value = finite_float(row.get(key))
            if value is not None:
                xs.append(float(row.get("epoch", i + 1)))
                ys.append(value)
        return xs, ys

    x, y = series("val_demand_annotation_spearman")
    if y:
        plt.figure(figsize=(8.0, 5.0))
        plt.plot(x, y)
        plt.axhline(0.5, linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Spearman")
        plt.title("Validation ambiguity-demand Spearman during Stage 1")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(
            output_dir / "training_val_demand_spearman.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close()

    plt.figure(figsize=(8.0, 5.0))
    found = False
    for q in range(1, 5):
        xq, yq = series(f"val_demand_q{q}_mean")
        if yq:
            plt.plot(xq, yq, label=f"Q{q}")
            found = True
    if found:
        plt.xlabel("Epoch")
        plt.ylabel("Validation raw ambiguity demand")
        plt.title("Validation Q1-Q4 ambiguity demand during Stage 1")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            output_dir / "training_val_demand_quartiles.png",
            dpi=180,
            bbox_inches="tight",
        )
    plt.close()


# =============================================================================
# 正式评估
# =============================================================================

@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate EG-GM-PUNet v2 Stage 1"
    )
    parser.add_argument(
        "--config",
        default="configs/eg_gm_punet_v2_relative_feedback_lidc.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "result/eg_gm_punet_v2_relative_feedback_k5_full/"
            "best_val_ged4_eg_gm_punet_v2_stage1.pth"
        ),
    )
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument(
        "--sample-counts",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16],
    )
    parser.add_argument("--save-examples", type=int, default=12)
    parser.add_argument("--active-threshold", type=float, default=0.02)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Stage 1 checkpoint 不存在。训练完成后确认路径：\n"
            f"{checkpoint_path}"
        )

    config = load_yaml(config_path)
    set_seed(int(config["validation_seed"]))
    device = prepare_device(config.get("device", "cuda"))

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", config)
    model = build_stage1_model(checkpoint_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    validation_config = config["validation"]
    training_config = config["training"]

    loader = build_lidc_full_dataloader(
        root=config["data_root"],
        split=args.split,
        batch_size=int(training_config["batch_size"]),
        num_workers=int(training_config["num_workers"]),
        shuffle=False,
        random_rater=False,
        rater_index=int(validation_config["rater_index"]),
    )

    sample_counts = normalize_sample_counts(
        list(args.sample_counts),
        int(args.num_samples),
    )

    output_dir = (
        checkpoint_path.parent
        / "evaluation"
        / args.split
        / f"samples_{args.num_samples}"
        / checkpoint_path.stem
    )
    if args.smoke_test:
        output_dir = output_dir / "smoke_test"
    figures_dir = output_dir / "figures"
    examples_dir = output_dir / "sample_arrays"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    evidence_metrics = EvidentialMetricAccumulator(
        num_bins=int(validation_config["ece_bins"]),
        ranking_bins=int(validation_config["ranking_bins"]),
    )

    probabilistic_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    demand_rows: list[dict[str, Any]] = []

    component_probability_sum = torch.zeros(
        model.num_components, dtype=torch.float64
    )
    component_sample_counts = torch.zeros(
        model.num_components, dtype=torch.long
    )

    checkpoint_epoch = int(checkpoint.get("epoch") or config["training"]["epochs"])
    fallback = 0
    total_cases = 0
    saved = 0

    iterable = islice(loader, 2) if args.smoke_test else loader
    total_batches = min(2, len(loader)) if args.smoke_test else len(loader)

    print("=" * 108)
    print("EG-GM-PUNet v2 Stage 1 — Formal Evaluation")
    print("=" * 108)
    print(f"[INFO] Config            : {config_path}")
    print(f"[INFO] Checkpoint        : {checkpoint_path}")
    print(f"[INFO] Checkpoint epoch  : {checkpoint_epoch}")
    print(f"[INFO] Split             : {args.split}")
    print(f"[INFO] Dataset cases     : {len(loader.dataset)}")
    print(f"[INFO] Prior samples     : {args.num_samples}")
    print(f"[INFO] Sample counts     : {sample_counts}")
    print(f"[INFO] Output            : {output_dir}")
    print(
        "[INFO] EMA reference     : "
        f"mean={float(model.feedback_controller.normalizer.running_mean.item()):.6f}, "
        f"std={float(model.feedback_controller.normalizer.running_var.clamp_min(0).sqrt().item()):.6f}, "
        f"updates={int(model.feedback_controller.normalizer.num_updates.item())}"
    )

    progress = tqdm(
        iterable,
        total=total_batches,
        desc=f"Stage1 evaluate {args.split}",
        ncols=150,
    )

    for batch in progress:
        images = batch["image"].to(device=device, dtype=torch.float32)
        selected_segmentation = batch["mask"].to(device=device, dtype=torch.float32)
        all_segmentations = batch["masks"].to(device=device, dtype=torch.float32)

        batch_size = int(images.shape[0])
        case_ids = ids_from_batch(
            batch.get("case_id"), batch_size, args.split, fallback
        )
        patient_ids = ids_from_batch(
            batch.get("patient_id"), batch_size, "patient", fallback
        )
        if all(value.startswith("patient_") for value in patient_ids):
            patient_ids = [patient_id_from_case_id(case_id) for case_id in case_ids]

        state = model(
            image=images,
            segmentation=None,
            is_training=False,
        )

        # ---------------------------------------------------------------------
        # Stage 1 deterministic demand probe。
        # model.eval() 下 normalizer 不更新 EMA。
        # ---------------------------------------------------------------------
        feedback_evidence = model.feedback_probe(state)
        controls = model.feedback_controller(
            evidence=feedback_evidence,
            base_beta_categorical=float(training_config["beta_categorical"]),
            base_beta_groups={
                str(name): float(value)
                for name, value in training_config["beta_groups"].items()
            },
            base_gumbel_temperature=float(training_config["gumbel_temperature"]),
            current_epoch=checkpoint_epoch,
            enabled=True,
            categorical_feedback=True,
            group_feedback=True,
            temperature_feedback=True,
        )

        mixture_probs = state.prior.mixture_probs
        safe_probs = mixture_probs.clamp_min(1e-8)
        entropy = -(safe_probs * safe_probs.log()).sum(dim=1)
        k_eff = torch.exp(entropy)
        active_k = (mixture_probs >= float(args.active_threshold)).sum(dim=1)

        for batch_index in range(batch_size):
            demand_rows.append(
                {
                    "case_id": case_ids[batch_index],
                    "patient_id": patient_ids[batch_index],
                    "annotation_diversity": annotation_diversity(
                        all_segmentations[batch_index]
                    ),
                    "raw_demand": float(
                        controls.raw_demand[batch_index].item()
                    ),
                    "demand_z_score": float(
                        controls.demand_z_score[batch_index].item()
                    ),
                    "relative_demand": float(
                        controls.relative_demand[batch_index].item()
                    ),
                    "beta_categorical": float(
                        controls.beta_categorical[batch_index].item()
                    ),
                    "beta_coarse": float(
                        controls.beta_groups["coarse"][batch_index].item()
                    ),
                    "beta_middle": float(
                        controls.beta_groups["middle"][batch_index].item()
                    ),
                    "beta_fine": float(
                        controls.beta_groups["fine"][batch_index].item()
                    ),
                    "gumbel_temperature": float(
                        controls.gumbel_temperature[batch_index].item()
                    ),
                    "feedback_ramp_at_checkpoint": float(
                        controls.feedback_ramp.item()
                    ),
                    "prior_k_eff": float(k_eff[batch_index].item()),
                    "active_component_count": int(active_k[batch_index].item()),
                    "prior_top2_probability_mass": float(
                        torch.sort(
                            mixture_probs[batch_index], descending=True
                        ).values[:2].sum().item()
                    ),
                }
            )

        # ---------------------------------------------------------------------
        # 与此前 Full / No-Feedback 完全一致的 exact prior sampling。
        # ---------------------------------------------------------------------
        exact_samples = sample_grouped_exact(
            state.prior,
            num_samples=int(args.num_samples),
        )
        component_probability_sum += (
            mixture_probs.sum(dim=0).detach().cpu().double()
        )
        component_sample_counts += torch.bincount(
            exact_samples.component_indices.detach().cpu().flatten(),
            minlength=model.num_components,
        )
        total_cases += batch_size

        probabilistic_logits: list[torch.Tensor] = []
        evidence_probabilities: list[torch.Tensor] = []
        evidence_vacuity: list[torch.Tensor] = []
        evidence_entropy: list[torch.Tensor] = []

        for sample_index in range(int(args.num_samples)):
            z_groups = {
                name: exact_samples.z_groups[name][sample_index]
                for name in model.group_names
            }
            decode = model.probabilistic_model.decode_from_groups(
                state=state,
                z_groups=z_groups,
                component_indices=exact_samples.component_indices[sample_index],
            )
            evidence = model.evidence_from_decode(decode)

            probabilistic_logits.append(decode.logits)
            evidence_probabilities.append(evidence.fused.probabilities)
            evidence_vacuity.append(evidence.fused.vacuity)
            evidence_entropy.append(evidence.fused.predictive_entropy)

            soft_target = multi_rater_soft_target(
                selected_segmentation,
                all_segmentations,
                num_classes=model.num_classes,
            ).to(
                device=device,
                dtype=evidence.fused.probabilities.dtype,
            )
            evidence_metrics.update(
                evidence_output=evidence.fused,
                evidence_summary=evidence.summaries["fused"],
                segmentation=selected_segmentation,
                all_segmentations=all_segmentations,
                soft_target=soft_target,
            )

        probabilistic_logits_tensor = torch.stack(probabilistic_logits, dim=0)
        evidence_probabilities_tensor = torch.stack(evidence_probabilities, dim=0)
        evidence_vacuity_tensor = torch.stack(evidence_vacuity, dim=0)
        evidence_entropy_tensor = torch.stack(evidence_entropy, dim=0)

        probabilistic_masks = probabilistic_logits_tensor.argmax(dim=2).float()
        probabilistic_foreground = torch.softmax(
            probabilistic_logits_tensor, dim=2
        )[:, :, 1]

        evidence_masks = evidence_probabilities_tensor.argmax(dim=2).float()
        evidence_foreground = evidence_probabilities_tensor[:, :, 1]

        for batch_index in range(batch_size):
            case_id = case_ids[batch_index]
            patient_id = patient_ids[batch_index]

            p_rows = evaluate_single_case(
                sample_masks=probabilistic_masks[:, batch_index],
                foreground_probabilities=probabilistic_foreground[:, batch_index],
                rater_masks=all_segmentations[batch_index],
                sample_counts=sample_counts,
            )
            for row in p_rows:
                row["case_id"] = case_id
                row["patient_id"] = patient_id
                row["prediction_branch"] = "probabilistic"
                row["split"] = args.split
            probabilistic_rows.extend(p_rows)

            e_rows = evaluate_single_case(
                sample_masks=evidence_masks[:, batch_index],
                foreground_probabilities=evidence_foreground[:, batch_index],
                rater_masks=all_segmentations[batch_index],
                sample_counts=sample_counts,
            )
            for row in e_rows:
                row["case_id"] = case_id
                row["patient_id"] = patient_id
                row["prediction_branch"] = "evidence"
                row["split"] = args.split
            evidence_rows.extend(e_rows)

            if saved < int(args.save_examples):
                examples_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    examples_dir / f"{safe_filename(case_id)}.npz",
                    case_id=np.asarray(case_id),
                    patient_id=np.asarray(patient_id),
                    image=images[batch_index].detach().cpu().numpy(),
                    rater_masks=all_segmentations[batch_index]
                    .detach().cpu().numpy().astype(np.uint8),
                    probabilistic_masks=probabilistic_masks[:, batch_index]
                    .detach().cpu().numpy().astype(np.uint8),
                    probabilistic_probabilities=probabilistic_foreground[:, batch_index]
                    .detach().cpu().numpy(),
                    evidence_probabilities=evidence_probabilities_tensor[:, batch_index]
                    .detach().cpu().numpy(),
                    evidence_vacuity=evidence_vacuity_tensor[:, batch_index]
                    .detach().cpu().numpy(),
                    evidence_predictive_entropy=evidence_entropy_tensor[:, batch_index]
                    .detach().cpu().numpy(),
                    component_indices=exact_samples.component_indices[:, batch_index]
                    .detach().cpu().numpy(),
                    prior_component_probabilities=mixture_probs[batch_index]
                    .detach().cpu().numpy(),
                    raw_demand=np.asarray(
                        float(controls.raw_demand[batch_index].item())
                    ),
                    relative_demand=np.asarray(
                        float(controls.relative_demand[batch_index].item())
                    ),
                )
                saved += 1

        fallback += batch_size

        latest = sample_counts[-1]
        values = [
            float(row["ged2"])
            for row in probabilistic_rows
            if int(row["num_samples"]) == latest
        ]
        if values:
            progress.set_postfix({f"GED{latest}": f"{np.mean(values):.4f}"})

    probabilistic_summary = summarize_rows(probabilistic_rows, sample_counts)
    evidence_summary = summarize_rows(evidence_rows, sample_counts)
    demand_summary = summarize_demand(demand_rows)
    calibration = evidence_metrics.compute()

    write_csv(
        probabilistic_rows,
        output_dir / "probabilistic_per_case_metrics.csv",
    )
    write_csv(
        evidence_rows,
        output_dir / "evidence_per_case_metrics.csv",
    )
    write_csv(
        demand_rows,
        output_dir / "stage1_demand_per_case.csv",
    )
    write_csv(
        demand_summary["patient_rows"],
        output_dir / "stage1_demand_per_patient.csv",
    )

    mean_component_probs = (
        component_probability_sum / max(total_cases, 1)
    )
    safe_mean = mean_component_probs.clamp_min(1e-8)
    component_summary = {
        "mean_prior_component_probabilities": mean_component_probs.tolist(),
        "effective_component_count_from_mean_probability": float(
            torch.exp(-(safe_mean * safe_mean.log()).sum()).item()
        ),
        "sampled_component_counts": component_sample_counts.tolist(),
        "sampled_component_frequencies": (
            component_sample_counts.float()
            / component_sample_counts.sum().clamp_min(1)
        ).tolist(),
        "mean_per_case_k_eff": float(
            np.mean([float(row["prior_k_eff"]) for row in demand_rows])
        ),
        "mean_active_component_count": float(
            np.mean([float(row["active_component_count"]) for row in demand_rows])
        ),
        "active_threshold": float(args.active_threshold),
    }

    # patient_rows 已另存 CSV，避免 summary.json 过大。
    demand_summary_for_json = dict(demand_summary)
    demand_summary_for_json.pop("patient_rows", None)

    summary = {
        "model": "EG-GM-PUNet v2 Stage 1 Relative Ambiguity-Demand Feedback",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "config": str(config_path.resolve()),
        "split": args.split,
        "num_cases": total_cases,
        "num_prior_samples": int(args.num_samples),
        "sample_counts": sample_counts,
        "smoke_test": bool(args.smoke_test),
        "probabilistic_summary": probabilistic_summary,
        "evidence_summary": evidence_summary,
        "evidence_calibration": calibration,
        "component_summary": component_summary,
        "stage1_demand_summary": demand_summary_for_json,
        "stage2_demand_capacity_matching": False,
        "stage3_component_vitality": False,
        "stage4_ambiguity_alignment": False,
    }

    save_json(output_dir / "summary.json", summary)

    plot_sample_count_metrics(
        probabilistic_summary,
        sample_counts,
        figures_dir,
    )
    plot_demand_quartiles(demand_summary, figures_dir)
    plot_training_demand_history(checkpoint_path.parent, figures_dir)

    print("\n" + "=" * 108)
    print("Stage 1 Probabilistic Evaluation")
    print("=" * 108)
    print(
        f"{'Samples':>8}  {'GED²↓':>10}  "
        f"{'CrossDice↑':>12}  {'BestDice↑':>11}  {'Diversity':>10}"
    )
    print("-" * 68)
    for count in sample_counts:
        item = probabilistic_summary[str(count)]
        print(
            f"{count:>8d}  "
            f"{item['ged2']['mean']:>10.5f}  "
            f"{item['cross_dice']['mean']:>12.5f}  "
            f"{item['best_sample_dice_to_raters']['mean']:>11.5f}  "
            f"{item['sample_diversity_distance']['mean']:>10.5f}"
        )

    print("\n[STAGE 1 DEMAND DIAGNOSTICS]")
    print(
        "  Slice Pearson / Spearman  : "
        f"{demand_summary['slice_level']['pearson']} / "
        f"{demand_summary['slice_level']['spearman']}"
    )
    print(
        "  Patient Pearson / Spearman: "
        f"{demand_summary['patient_level']['pearson']} / "
        f"{demand_summary['patient_level']['spearman']}"
    )
    for item in demand_summary["quartiles"]:
        print(
            f"  Q{item['quartile']}: "
            f"D_ann={item['annotation_diversity_mean']:.4f}, "
            f"demand={item['raw_demand_mean']:.4f}, "
            f"Urel={item['relative_demand_mean']:.4f}, "
            f"betaC={item['beta_categorical_mean']:.4f}, "
            f"tau={item['gumbel_temperature_mean']:.4f}, "
            f"Keff={item['prior_k_eff_mean']:.4f}"
        )

    print("\n[COMPONENT]")
    print(
        f"  mean per-case K_eff : "
        f"{component_summary['mean_per_case_k_eff']:.6f}"
    )
    print(
        f"  mean active K       : "
        f"{component_summary['mean_active_component_count']:.6f}"
    )

    print("\n[EVIDENCE CALIBRATION]")
    for key in (
        "hard_dice_all_raters",
        "soft_dice_multi_rater",
        "ece",
        "error_detection_auroc",
        "rater_disagreement_auroc",
        "vacuity_disagreement_correlation",
        "ambiguity_weighted_vacuity",
    ):
        if key in calibration:
            print(f"  {key:40s}: {calibration[key]}")

    print("\n" + "=" * 108)
    print(f"[DONE] summary               : {output_dir / 'summary.json'}")
    print(f"[DONE] probabilistic per-case: {output_dir / 'probabilistic_per_case_metrics.csv'}")
    print(f"[DONE] evidence per-case     : {output_dir / 'evidence_per_case_metrics.csv'}")
    print(f"[DONE] Stage1 demand per-case: {output_dir / 'stage1_demand_per_case.csv'}")
    print(f"[DONE] Stage1 demand patient : {output_dir / 'stage1_demand_per_patient.csv'}")
    print(f"[DONE] figures               : {figures_dir}")
    print("=" * 108)


if __name__ == "__main__":
    main()
