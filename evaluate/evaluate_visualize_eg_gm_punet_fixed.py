"""
EG-GM-PUNet 最终测试与可视化（一键运行版）

放置位置：
    E:/workspace/EG-GM-PUNet/evaluate/evaluate_visualize_eg_gm_punet.py

使用方法：
    在 PyCharm 中打开本文件，直接点击 Run。

功能：
1. 直接构建并评估最终 EGGMPUNet，不调用其他评估脚本入口。
2. 默认评估 best_val_ged4_eg_gm_punet.pth 的 test 集结果。
3. 保存 GED²、Cross Dice、Best-sample Dice、证据校准指标和逐病例 CSV。
4. 绘制训练曲线、证据反哺控制曲线、component 使用情况。
5. 绘制典型病例：CT、4 位医生标注、多样本分割、证据概率、
   vacuity 和 predictive entropy。
6. 生成 report.html，并自动用浏览器打开。
"""

from __future__ import annotations

import csv
import html
import json
import math
import sys
import webbrowser
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm


# =============================================================================
# 只需要检查这里的路径和参数
# =============================================================================

PROJECT_ROOT = Path(r"E:\workspace\EG-GM-PUNet")

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "eg_gm_punet_no_feedback_lidc.yaml"
)

# 优先评估概率分布性能最佳的 checkpoint。
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "result"
    / "eg_gm_punet_no_feedback_k5_full"
    / "best_val_ged4_eg_gm_punet.pth"
)

SPLIT = "test"
NUM_SAMPLES = 16
SAMPLE_COUNTS = [1, 4, 8, 16]
SAVE_EXAMPLES = 12

# False：如果已有 summary.json，则不重复执行完整测试，只重新画图。
# True ：无论是否已有结果，都重新跑一次 test 集评估。
FORCE_REEVALUATE = True

# 每个病例大图中最多展示多少个概率分割样本。
MAX_PLOTTED_SAMPLES = 8

# 结束后自动在浏览器中打开 report.html。
AUTO_OPEN_REPORT = True

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "result"
    / "eg_gm_punet_no_feedback_k5_full"
    / "evaluation"
    / SPLIT
    / f"samples_{NUM_SAMPLES}"
    / CHECKPOINT_PATH.stem
)



# =============================================================================
# 路径检查与项目导入
# =============================================================================

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description}不存在：\n{path}"
        )


require_file(CONFIG_PATH, "配置文件")
require_file(CHECKPOINT_PATH, "EG-GM-PUNet checkpoint")
from models.EG_GM_PUNet import EGGMPUNet
from dataset.lidc_dataset_full import build_lidc_full_dataloader
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
# 通用读取工具
# =============================================================================

def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"YAML 不是字典：{path}")
    return value


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            value,
            file,
            indent=2,
            ensure_ascii=False,
        )


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def first_numeric(
    row: dict[str, Any],
    candidates: Iterable[str],
) -> float | None:
    for key in candidates:
        if key not in row:
            continue
        result = finite_float(row[key])
        if result is not None:
            return result
    return None


def safe_name(value: str) -> str:
    return "".join(
        character
        if character.isalnum() or character in ("-", "_", ".")
        else "_"
        for character in value
    )


# =============================================================================
# EG-GM-PUNet 直接测试评估
# =============================================================================

def build_final_model(
    config: dict[str, Any],
) -> EGGMPUNet:
    item = config["model"]

    return EGGMPUNet(
        in_channels=int(item["in_channels"]),
        num_classes=int(item["num_classes"]),
        num_components=int(item["num_components"]),
        latent_dims={
            str(name): int(value)
            for name, value in item["latent_dims"].items()
        },
        num_channels=tuple(
            int(value)
            for value in item["num_channels"]
        ),
        group_feature_indices={
            str(name): int(value)
            for name, value
            in item["group_feature_indices"].items()
        },
        group_decoder_indices={
            str(name): int(value)
            for name, value
            in item["group_decoder_indices"].items()
        },
        num_convs_per_block=int(
            item["num_convs_per_block"]
        ),
        fusion_1x1_convs=int(
            item["fusion_1x1_convs"]
        ),
        segmentation_head_1x1_convs=int(
            item["segmentation_head_1x1_convs"]
        ),
        min_log_sigma=item.get("min_log_sigma"),
        max_log_sigma=item.get("max_log_sigma"),
        evidence_projection_channels=int(
            item["evidence_projection_channels"]
        ),
        evidence_fusion_channels=int(
            item["evidence_fusion_channels"]
        ),
        feedback_config=item.get("feedback"),
    )


def case_ids_from_batch(
    value: Any,
    batch_size: int,
    fallback: int,
    split: str,
) -> list[str]:
    if value is None:
        return [
            f"{split}_{fallback + index:06d}"
            for index in range(batch_size)
        ]
    if isinstance(value, str):
        return [value]
    if torch.is_tensor(value):
        values = value.detach().cpu().tolist()
        if not isinstance(values, list):
            values = [values]
        return [str(item) for item in values]
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def add_branch_metadata(
    rows: list[dict[str, Any]],
    branch: str,
    case_id: str,
    split: str,
) -> None:
    for row in rows:
        row["prediction_branch"] = branch
        row["case_id"] = case_id
        row["split"] = split


def print_branch_summary(
    title: str,
    summary: dict[str, Any],
    counts: list[int],
) -> None:
    print("\n" + title)
    print(
        f"{'Samples':>8}  {'GED²↓':>10}  "
        f"{'CrossDice↑':>12}  {'BestDice↑':>11}  "
        f"{'Diversity':>10}"
    )
    print("-" * 62)

    for count in counts:
        item = summary[str(count)]
        print(
            f"{count:>8d}  "
            f"{item['ged2']['mean']:>10.4f}  "
            f"{item['cross_dice']['mean']:>12.4f}  "
            f"{item['best_sample_dice_to_raters']['mean']:>11.4f}  "
            f"{item['sample_diversity_distance']['mean']:>10.4f}"
        )


@torch.no_grad()
def run_test_evaluation() -> None:
    config = load_yaml(CONFIG_PATH)
    set_seed(int(config["validation_seed"]))
    device = prepare_device(
        config.get("device", "cuda")
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )
    checkpoint_config = checkpoint.get(
        "config",
        config,
    )

    model = build_final_model(
        checkpoint_config
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.eval()

    training_config = config["training"]
    validation_config = config["validation"]

    batch_size = int(
        training_config["batch_size"]
    )
    num_workers = int(
        training_config["num_workers"]
    )

    loader = build_lidc_full_dataloader(
        root=config["data_root"],
        split=SPLIT,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        random_rater=False,
        rater_index=int(
            validation_config["rater_index"]
        ),
    )

    sample_counts = normalize_sample_counts(
        SAMPLE_COUNTS,
        NUM_SAMPLES,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    example_directory = (
        OUTPUT_DIRECTORY / "sample_arrays"
    )

    probabilistic_rows: list[
        dict[str, Any]
    ] = []
    evidence_rows: list[
        dict[str, Any]
    ] = []

    evidence_metrics = EvidentialMetricAccumulator(
        num_bins=int(
            validation_config["ece_bins"]
        ),
        ranking_bins=int(
            validation_config["ranking_bins"]
        ),
    )

    component_counts = torch.zeros(
        model.num_components,
        dtype=torch.long,
    )
    probability_sum = torch.zeros(
        model.num_components,
        dtype=torch.float64,
    )

    total_cases = 0
    fallback = 0
    saved = 0

    print("=" * 96)
    print("开始执行 EG-GM-PUNet 最终测试集评估")
    print("=" * 96)
    print(f"[INFO] Config     : {CONFIG_PATH}")
    print(f"[INFO] Checkpoint : {CHECKPOINT_PATH}")
    print(f"[INFO] Epoch      : {checkpoint.get('epoch')}")
    print(f"[INFO] Device     : {device}")
    print(f"[INFO] Split      : {SPLIT}")
    print(f"[INFO] Cases      : {len(loader.dataset)}")
    print(f"[INFO] Samples    : {NUM_SAMPLES}")
    print(f"[INFO] Counts     : {sample_counts}")
    print(f"[INFO] Output     : {OUTPUT_DIRECTORY}")

    progress = tqdm(
        loader,
        desc=f"EG-GM-PUNet evaluate {SPLIT}",
        ncols=150,
    )

    for batch in progress:
        images = batch["image"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        segmentation = batch["mask"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        all_segmentations = batch["masks"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        current_case_ids = case_ids_from_batch(
            batch.get("case_id"),
            images.shape[0],
            fallback,
            SPLIT,
        )

        state = model(
            image=images,
            segmentation=None,
            is_training=False,
        )

        exact_samples = sample_grouped_exact(
            state.prior,
            num_samples=NUM_SAMPLES,
        )

        probability_sum += (
            state.prior.mixture_probs
            .sum(dim=0)
            .cpu()
            .double()
        )
        total_cases += int(images.shape[0])

        component_counts += torch.bincount(
            exact_samples
            .component_indices
            .cpu()
            .flatten(),
            minlength=model.num_components,
        )

        probabilistic_logits: list[
            torch.Tensor
        ] = []
        evidence_probabilities: list[
            torch.Tensor
        ] = []
        evidence_vacuity: list[
            torch.Tensor
        ] = []
        evidence_predictive_entropy: list[
            torch.Tensor
        ] = []

        for sample_index in range(
            NUM_SAMPLES
        ):
            z_groups = {
                name: (
                    exact_samples
                    .z_groups[name][sample_index]
                )
                for name in model.group_names
            }

            decode = (
                model
                .probabilistic_model
                .decode_from_groups(
                    state=state,
                    z_groups=z_groups,
                    component_indices=(
                        exact_samples
                        .component_indices[
                            sample_index
                        ]
                    ),
                )
            )

            evidence = model.evidence_from_decode(
                decode
            )

            probabilistic_logits.append(
                decode.logits
            )
            evidence_probabilities.append(
                evidence.fused.probabilities
            )
            evidence_vacuity.append(
                evidence.fused.vacuity
            )
            evidence_predictive_entropy.append(
                evidence.fused.predictive_entropy
            )

            soft_target = multi_rater_soft_target(
                segmentation,
                all_segmentations,
                num_classes=model.num_classes,
            ).to(
                device=device,
                dtype=(
                    evidence
                    .fused
                    .probabilities
                    .dtype
                ),
            )

            evidence_metrics.update(
                evidence_output=(
                    evidence.fused
                ),
                evidence_summary=(
                    evidence.summaries[
                        "fused"
                    ]
                ),
                segmentation=segmentation,
                all_segmentations=(
                    all_segmentations
                ),
                soft_target=soft_target,
            )

        probabilistic_logits_tensor = (
            torch.stack(
                probabilistic_logits,
                dim=0,
            )
        )
        evidence_probabilities_tensor = (
            torch.stack(
                evidence_probabilities,
                dim=0,
            )
        )
        evidence_vacuity_tensor = (
            torch.stack(
                evidence_vacuity,
                dim=0,
            )
        )
        evidence_entropy_tensor = (
            torch.stack(
                evidence_predictive_entropy,
                dim=0,
            )
        )

        probabilistic_masks = (
            probabilistic_logits_tensor
            .argmax(dim=2)
            .float()
        )
        probabilistic_foreground = (
            torch.softmax(
                probabilistic_logits_tensor,
                dim=2,
            )[:, :, 1]
        )

        evidence_masks = (
            evidence_probabilities_tensor
            .argmax(dim=2)
            .float()
        )
        evidence_foreground = (
            evidence_probabilities_tensor[
                :, :, 1
            ]
        )

        for batch_index in range(
            images.shape[0]
        ):
            case_id = current_case_ids[
                batch_index
            ]

            current_probability_rows = (
                evaluate_single_case(
                    sample_masks=(
                        probabilistic_masks[
                            :, batch_index
                        ]
                    ),
                    foreground_probabilities=(
                        probabilistic_foreground[
                            :, batch_index
                        ]
                    ),
                    rater_masks=(
                        all_segmentations[
                            batch_index
                        ]
                    ),
                    sample_counts=sample_counts,
                )
            )
            add_branch_metadata(
                current_probability_rows,
                "probabilistic",
                case_id,
                SPLIT,
            )
            probabilistic_rows.extend(
                current_probability_rows
            )

            current_evidence_rows = (
                evaluate_single_case(
                    sample_masks=(
                        evidence_masks[
                            :, batch_index
                        ]
                    ),
                    foreground_probabilities=(
                        evidence_foreground[
                            :, batch_index
                        ]
                    ),
                    rater_masks=(
                        all_segmentations[
                            batch_index
                        ]
                    ),
                    sample_counts=sample_counts,
                )
            )
            add_branch_metadata(
                current_evidence_rows,
                "evidence",
                case_id,
                SPLIT,
            )
            evidence_rows.extend(
                current_evidence_rows
            )

            if saved < SAVE_EXAMPLES:
                example_directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                np.savez_compressed(
                    example_directory
                    / (
                        f"{safe_filename(case_id)}"
                        ".npz"
                    ),
                    case_id=np.asarray(case_id),
                    image=(
                        images[batch_index]
                        .cpu()
                        .numpy()
                    ),
                    rater_masks=(
                        all_segmentations[
                            batch_index
                        ]
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    ),
                    probabilistic_masks=(
                        probabilistic_masks[
                            :, batch_index
                        ]
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    ),
                    probabilistic_probabilities=(
                        probabilistic_foreground[
                            :, batch_index
                        ]
                        .cpu()
                        .numpy()
                    ),
                    evidence_masks=(
                        evidence_masks[
                            :, batch_index
                        ]
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    ),
                    evidence_probabilities=(
                        evidence_probabilities_tensor[
                            :, batch_index
                        ]
                        .cpu()
                        .numpy()
                    ),
                    evidence_vacuity=(
                        evidence_vacuity_tensor[
                            :, batch_index
                        ]
                        .cpu()
                        .numpy()
                    ),
                    evidence_predictive_entropy=(
                        evidence_entropy_tensor[
                            :, batch_index
                        ]
                        .cpu()
                        .numpy()
                    ),
                    component_indices=(
                        exact_samples
                        .component_indices[
                            :, batch_index
                        ]
                        .cpu()
                        .numpy()
                    ),
                    prior_component_probabilities=(
                        state.prior
                        .mixture_probs[
                            batch_index
                        ]
                        .cpu()
                        .numpy()
                    ),
                )
                saved += 1

        fallback += int(images.shape[0])

        latest = sample_counts[-1]
        latest_values = [
            row["ged2"]
            for row in probabilistic_rows
            if row["num_samples"] == latest
        ]
        if latest_values:
            progress.set_postfix(
                {
                    f"pGED{latest}": (
                        f"{np.mean(latest_values):.4f}"
                    )
                }
            )

    probabilistic_summary = summarize_rows(
        probabilistic_rows,
        sample_counts,
    )
    evidence_summary = summarize_rows(
        evidence_rows,
        sample_counts,
    )

    write_csv(
        probabilistic_rows,
        OUTPUT_DIRECTORY
        / "probabilistic_per_case_metrics.csv",
    )
    write_csv(
        evidence_rows,
        OUTPUT_DIRECTORY
        / "evidence_per_case_metrics.csv",
    )

    mean_probabilities = (
        probability_sum
        / max(total_cases, 1)
    )
    safe_probabilities = (
        mean_probabilities.clamp_min(1e-8)
    )

    component_summary = {
        "mean_prior_component_probabilities": (
            mean_probabilities.tolist()
        ),
        "effective_component_count": float(
            torch.exp(
                -(
                    safe_probabilities
                    * safe_probabilities.log()
                ).sum()
            ).item()
        ),
        "sampled_component_counts": (
            component_counts.tolist()
        ),
        "sampled_component_frequencies": (
            component_counts.float()
            / component_counts.sum().clamp_min(1)
        ).tolist(),
    }

    result = {
        "model": "MS-GMM + MS-DEH (No Feedback)",
        "checkpoint": str(
            CHECKPOINT_PATH.resolve()
        ),
        "checkpoint_epoch": checkpoint.get(
            "epoch"
        ),
        "split": SPLIT,
        "num_cases": total_cases,
        "num_prior_samples": NUM_SAMPLES,
        "sample_counts": sample_counts,
        "probabilistic_summary": (
            probabilistic_summary
        ),
        "evidence_summary": (
            evidence_summary
        ),
        "evidence_calibration": (
            evidence_metrics.compute()
        ),
        "component_summary": (
            component_summary
        ),
        "evidence_feedback_used": False,
        "visualization_entry": str(
            Path(__file__).resolve()
        ),
    }

    save_json(
        OUTPUT_DIRECTORY / "summary.json",
        result,
    )

    print_branch_summary(
        "Probabilistic branch",
        probabilistic_summary,
        sample_counts,
    )
    print_branch_summary(
        "Evidential branch",
        evidence_summary,
        sample_counts,
    )

    calibration = result[
        "evidence_calibration"
    ]
    print("\nEvidence calibration")
    for label, key in (
        (
            "hard Dice to all raters",
            "hard_dice_all_raters",
        ),
        (
            "soft Dice multi-rater",
            "soft_dice_multi_rater",
        ),
        ("ECE", "ece"),
        (
            "error-detection AUROC",
            "error_detection_auroc",
        ),
        (
            "disagreement AUROC",
            "rater_disagreement_auroc",
        ),
        (
            "vacuity/disagreement r",
            "vacuity_disagreement_correlation",
        ),
        (
            "ambiguity vacuity",
            "ambiguity_weighted_vacuity",
        ),
    ):
        value = finite_float(
            calibration.get(key)
        )
        rendered = (
            "N/A"
            if value is None
            else f"{value:.4f}"
        )
        print(
            f"{label:28s}: {rendered}"
        )

    print(
        "\n[INFO] Component summary:",
        component_summary,
    )
    print(
        f"[INFO] Results: "
        f"{OUTPUT_DIRECTORY}"
    )
    print(
        "[DONE] EG-GM-PUNet final "
        "evaluation finished."
    )


# =============================================================================
# 训练历史曲线
# =============================================================================

def read_history() -> list[dict[str, Any]]:
    result_root = CHECKPOINT_PATH.parent
    json_path = result_root / "history.json"
    csv_path = result_root / "history.csv"

    if json_path.exists():
        value = load_json(json_path)
        if isinstance(value, list):
            return [
                item
                for item in value
                if isinstance(item, dict)
            ]

    if csv_path.exists():
        with open(
            csv_path,
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            return list(csv.DictReader(file))

    print(
        "[WARN] 未找到 history.json/history.csv，"
        "将跳过训练曲线。"
    )
    return []


def plot_series(
    *,
    history: list[dict[str, Any]],
    definitions: list[
        tuple[str, list[str]]
    ],
    title: str,
    y_label: str,
    output_path: Path,
    horizontal_zero: bool = False,
    mark_feedback: bool = True,
) -> bool:
    if not history:
        return False

    epochs: list[float] = []
    for index, row in enumerate(history):
        epoch = finite_float(row.get("epoch"))
        epochs.append(
            float(index + 1)
            if epoch is None
            else epoch
        )

    found = False
    plt.figure(figsize=(10, 6))

    for label, candidates in definitions:
        values = [
            first_numeric(row, candidates)
            for row in history
        ]

        x_valid = [
            epochs[index]
            for index, value in enumerate(values)
            if value is not None
        ]
        y_valid = [
            float(value)
            for value in values
            if value is not None
        ]

        if y_valid:
            plt.plot(
                x_valid,
                y_valid,
                linewidth=1.7,
                label=label,
            )
            found = True

    if not found:
        plt.close()
        return False

    config = load_yaml(CONFIG_PATH)
    feedback_config = (
        config.get("model", {})
        .get("feedback", {})
    )

    # 概率 checkpoint 初始化时，正式训练日志中是第 21 epoch 开始反馈。
    start_epoch = int(
        feedback_config.get("start_epoch", 21)
    )
    ramp_epochs = int(
        feedback_config.get("ramp_epochs", 20)
    )
    full_epoch = (
        start_epoch + ramp_epochs
        if ramp_epochs > 0
        else start_epoch
    )

    feedback_enabled = bool(
        config.get("training", {}).get("feedback_enabled", False)
        and feedback_config.get("enabled", False)
    )

    if mark_feedback and feedback_enabled:
        plt.axvline(
            start_epoch,
            linestyle="--",
            linewidth=1.0,
            label="Feedback start",
        )
        plt.axvline(
            full_epoch,
            linestyle=":",
            linewidth=1.0,
            label="Full feedback",
        )

    if horizontal_zero:
        plt.axhline(
            0.0,
            linestyle="-.",
            linewidth=1.0,
        )

    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()
    return True


def generate_training_figures() -> list[Path]:
    history = read_history()
    figure_directory = (
        OUTPUT_DIRECTORY / "figures"
    )
    figure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    specifications = [
        {
            "filename": "01_joint_loss.png",
            "title": "MS-GMM + MS-DEH (No Feedback) joint loss",
            "y_label": "Loss",
            "definitions": [
                (
                    "Train",
                    [
                        "train_joint_loss",
                        "train_objective_loss",
                    ],
                ),
                (
                    "Validation",
                    [
                        "val_joint_loss",
                        "val_objective_loss",
                    ],
                ),
            ],
        },
        {
            "filename": "02_probabilistic_dice.png",
            "title": "Probabilistic segmentation Dice",
            "y_label": "Dice",
            "definitions": [
                (
                    "Train probabilistic Dice",
                    ["train_probabilistic_dice"],
                ),
                (
                    "Validation probabilistic Dice",
                    ["val_probabilistic_dice"],
                ),
            ],
        },
        {
            "filename": "03_evidence_dice.png",
            "title": "Evidential segmentation Dice",
            "y_label": "Dice",
            "definitions": [
                (
                    "Train evidence Dice",
                    [
                        "train_evidence_hard_dice_all_raters",
                        "train_evidence_soft_dice_multi_rater",
                    ],
                ),
                (
                    "Validation evidence Dice",
                    [
                        "val_evidence_hard_dice_all_raters",
                        "val_evidence_soft_dice_multi_rater",
                    ],
                ),
            ],
        },
        {
            "filename": "04_validation_ged2.png",
            "title": "Validation GED²",
            "y_label": "GED² (lower is better)",
            "definitions": [
                (
                    "Validation GED²",
                    ["val_prior_ged2"],
                )
            ],
        },
        {
            "filename": "05_validation_ece.png",
            "title": "Validation evidence ECE",
            "y_label": "ECE (lower is better)",
            "definitions": [
                (
                    "Validation ECE",
                    ["val_evidence_ece"],
                )
            ],
        },
        {
            "filename": "06_feedback_ramp.png",
            "title": "Evidence-feedback ramp",
            "y_label": "Ramp coefficient",
            "definitions": [
                (
                    "Train feedback ramp",
                    ["train_feedback_ramp"],
                ),
                (
                    "Validation feedback ramp",
                    ["val_feedback_ramp"],
                ),
            ],
        },
        {
            "filename": "07_adaptive_betas.png",
            "title": "Evidence-guided adaptive KL weights",
            "y_label": "Adaptive beta",
            "definitions": [
                (
                    "Categorical beta",
                    ["train_adaptive_beta_categorical"],
                ),
                (
                    "Coarse beta",
                    ["train_adaptive_beta_coarse"],
                ),
                (
                    "Middle beta",
                    ["train_adaptive_beta_middle"],
                ),
                (
                    "Fine beta",
                    ["train_adaptive_beta_fine"],
                ),
            ],
        },
        {
            "filename": "08_adaptive_temperature.png",
            "title": "Adaptive Gumbel-Softmax temperature",
            "y_label": "Temperature",
            "definitions": [
                (
                    "Train temperature",
                    ["train_adaptive_temperature"],
                ),
                (
                    "Validation temperature",
                    ["val_adaptive_temperature"],
                ),
            ],
        },
        {
            "filename": "09_feedback_uncertainty.png",
            "title": "Multi-scale feedback uncertainty",
            "y_label": "Uncertainty score",
            "definitions": [
                (
                    "Fused/categorical",
                    [
                        "train_feedback_uncertainty_categorical"
                    ],
                ),
                (
                    "Coarse",
                    ["train_feedback_uncertainty_coarse"],
                ),
                (
                    "Middle",
                    ["train_feedback_uncertainty_middle"],
                ),
                (
                    "Fine",
                    ["train_feedback_uncertainty_fine"],
                ),
            ],
        },
        {
            "filename": "10_feedback_correlations.png",
            "title": "Uncertainty-control correlations",
            "y_label": "Pearson correlation",
            "horizontal_zero": True,
            "definitions": [
                (
                    "Uncertainty vs categorical beta",
                    [
                        "train_corr_uncertainty_beta_categorical"
                    ],
                ),
                (
                    "Uncertainty vs coarse beta",
                    [
                        "train_corr_uncertainty_beta_coarse"
                    ],
                ),
                (
                    "Uncertainty vs middle beta",
                    [
                        "train_corr_uncertainty_beta_middle"
                    ],
                ),
                (
                    "Uncertainty vs fine beta",
                    [
                        "train_corr_uncertainty_beta_fine"
                    ],
                ),
                (
                    "Uncertainty vs temperature",
                    [
                        "train_corr_uncertainty_temperature"
                    ],
                ),
            ],
        },
    ]

    outputs: list[Path] = []

    for item in specifications:
        output_path = (
            figure_directory
            / str(item["filename"])
        )
        generated = plot_series(
            history=history,
            definitions=item["definitions"],
            title=str(item["title"]),
            y_label=str(item["y_label"]),
            output_path=output_path,
            horizontal_zero=bool(
                item.get("horizontal_zero", False)
            ),
        )
        if generated:
            outputs.append(output_path)

    return outputs


# =============================================================================
# 测试集 summary 可视化
# =============================================================================

def metric_mean(
    summary_item: dict[str, Any],
    key: str,
) -> float | None:
    metric = summary_item.get(key)
    if not isinstance(metric, dict):
        return None
    return finite_float(metric.get("mean"))


def generate_component_figure(
    summary: dict[str, Any],
) -> Path | None:
    component = summary.get("component_summary")
    if not isinstance(component, dict):
        return None

    prior_probabilities = component.get(
        "mean_prior_component_probabilities"
    )
    sampled_frequencies = component.get(
        "sampled_component_frequencies"
    )

    if not isinstance(prior_probabilities, list):
        return None

    count = len(prior_probabilities)
    positions = np.arange(count)
    width = 0.36

    plt.figure(figsize=(9, 5.5))
    plt.bar(
        positions - width / 2,
        prior_probabilities,
        width=width,
        label="Mean prior probability",
    )

    if (
        isinstance(sampled_frequencies, list)
        and len(sampled_frequencies) == count
    ):
        plt.bar(
            positions + width / 2,
            sampled_frequencies,
            width=width,
            label="Sampled frequency",
        )

    effective = finite_float(
        component.get("effective_component_count")
    )
    title = "Mixture-component utilization"
    if effective is not None:
        title += (
            f"\nEffective component count = "
            f"{effective:.3f}"
        )

    plt.xticks(
        positions,
        [
            f"Component {index}"
            for index in range(count)
        ],
    )
    plt.ylabel("Probability / frequency")
    plt.title(title)
    plt.ylim(bottom=0.0)
    plt.grid(
        axis="y",
        alpha=0.25,
    )
    plt.legend()
    plt.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "figures"
        / "11_component_usage.png"
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()
    return output_path


def generate_sample_count_figures(
    summary: dict[str, Any],
) -> list[Path]:
    probability_summary = summary.get(
        "probabilistic_summary"
    )
    sample_counts = summary.get("sample_counts")

    if (
        not isinstance(probability_summary, dict)
        or not isinstance(sample_counts, list)
    ):
        return []

    definitions = [
        (
            "12_ged2_vs_samples.png",
            "GED² versus number of prior samples",
            "GED² (lower is better)",
            "ged2",
        ),
        (
            "13_cross_dice_vs_samples.png",
            "Cross Dice versus number of prior samples",
            "Cross Dice (higher is better)",
            "cross_dice",
        ),
        (
            "14_best_dice_vs_samples.png",
            "Best-sample Dice versus number of prior samples",
            "Best-sample Dice (higher is better)",
            "best_sample_dice_to_raters",
        ),
        (
            "15_diversity_vs_samples.png",
            "Sample diversity versus number of prior samples",
            "Sample diversity distance",
            "sample_diversity_distance",
        ),
    ]

    outputs: list[Path] = []

    for filename, title, y_label, metric_key in definitions:
        x_values: list[int] = []
        y_values: list[float] = []

        for sample_count in sample_counts:
            item = probability_summary.get(
                str(sample_count)
            )
            if not isinstance(item, dict):
                continue

            value = metric_mean(
                item,
                metric_key,
            )
            if value is not None:
                x_values.append(int(sample_count))
                y_values.append(value)

        if not y_values:
            continue

        plt.figure(figsize=(8, 5.5))
        plt.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=1.8,
        )
        plt.xlabel("Number of prior samples")
        plt.ylabel(y_label)
        plt.title(title)
        plt.xticks(x_values)
        plt.grid(alpha=0.25)
        plt.tight_layout()

        output_path = (
            OUTPUT_DIRECTORY
            / "figures"
            / filename
        )
        plt.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",
        )
        plt.close()
        outputs.append(output_path)

    return outputs


# =============================================================================
# 典型病例可视化
# =============================================================================

def squeeze_2d(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(
            f"需要二维图像，实际 shape={array.shape}"
        )
    return array


def normalize_ct(value: np.ndarray) -> np.ndarray:
    image = squeeze_2d(value).astype(np.float32)

    low = float(np.percentile(image, 1))
    high = float(np.percentile(image, 99))

    if high <= low:
        low = float(image.min())
        high = float(image.max())

    if high <= low:
        return np.zeros_like(image)

    return np.clip(
        (image - low) / (high - low),
        0.0,
        1.0,
    )


def foreground_probability(
    probabilities: np.ndarray,
) -> np.ndarray:
    value = np.asarray(probabilities)

    # 预期 [S, 2, H, W]
    if value.ndim == 4 and value.shape[1] >= 2:
        return value[:, 1]

    # 容错：[S, H, W]
    if value.ndim == 3:
        return value

    raise ValueError(
        "无法识别 evidence_probabilities shape："
        f"{value.shape}"
    )


def mean_single_channel(
    values: np.ndarray,
) -> np.ndarray:
    array = np.asarray(values)

    if array.ndim == 4:
        # [S, 1, H, W]
        array = array[:, 0]

    if array.ndim != 3:
        raise ValueError(
            f"预期 [S,H,W] 或 [S,1,H,W]，实际 {array.shape}"
        )

    return array.mean(axis=0)


def add_image_axis(
    figure: plt.Figure,
    grid,
    row: int,
    column: int,
    image: np.ndarray,
    title: str,
    *,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    add_colorbar: bool = False,
) -> None:
    axis = figure.add_subplot(
        grid[row, column]
    )
    shown = axis.imshow(
        image,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_title(
        title,
        fontsize=9,
    )
    axis.axis("off")

    if add_colorbar:
        figure.colorbar(
            shown,
            ax=axis,
            fraction=0.046,
            pad=0.04,
        )


def create_case_figure(
    npz_path: Path,
) -> Path:
    with np.load(
        npz_path,
        allow_pickle=False,
    ) as data:
        image = normalize_ct(data["image"])
        rater_masks = np.asarray(
            data["rater_masks"]
        )
        probabilistic_masks = np.asarray(
            data["probabilistic_masks"]
        )
        evidence_masks = np.asarray(
            data["evidence_masks"]
        )
        evidence_probabilities = np.asarray(
            data["evidence_probabilities"]
        )
        evidence_vacuity = np.asarray(
            data["evidence_vacuity"]
        )
        evidence_entropy = np.asarray(
            data["evidence_predictive_entropy"]
        )
        component_indices = np.asarray(
            data["component_indices"]
        ).reshape(-1)

    sample_limit = min(
        MAX_PLOTTED_SAMPLES,
        probabilistic_masks.shape[0],
    )

    columns = max(8, sample_limit)
    figure = plt.figure(
        figsize=(2.8 * columns, 10.5)
    )
    grid = figure.add_gridspec(
        3,
        columns,
    )

    # 第一行：CT 和 4 位医生标注。
    add_image_axis(
        figure,
        grid,
        0,
        0,
        image,
        "CT image",
    )

    for rater_index in range(
        min(4, rater_masks.shape[0])
    ):
        overlay = image.copy()
        axis = figure.add_subplot(
            grid[0, rater_index + 1]
        )
        axis.imshow(
            overlay,
            cmap="gray",
        )
        axis.imshow(
            squeeze_2d(
                rater_masks[rater_index]
            ),
            alpha=0.45,
            vmin=0.0,
            vmax=1.0,
        )
        axis.set_title(
            f"Rater {rater_index + 1}",
            fontsize=9,
        )
        axis.axis("off")

    rater_probability = (
        rater_masks.astype(np.float32)
        .mean(axis=0)
    )
    rater_disagreement = (
        rater_masks.astype(np.float32)
        .var(axis=0)
    )

    add_image_axis(
        figure,
        grid,
        0,
        5,
        rater_probability,
        "Rater probability",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        add_colorbar=True,
    )
    add_image_axis(
        figure,
        grid,
        0,
        6,
        rater_disagreement,
        "Rater disagreement",
        cmap="magma",
        vmin=0.0,
        add_colorbar=True,
    )

    # 第一行剩余位置留空。
    for column in range(7, columns):
        axis = figure.add_subplot(
            grid[0, column]
        )
        axis.axis("off")

    # 第二行：概率分支多样本。
    for sample_index in range(sample_limit):
        axis = figure.add_subplot(
            grid[1, sample_index]
        )
        axis.imshow(
            image,
            cmap="gray",
        )
        axis.imshow(
            squeeze_2d(
                probabilistic_masks[
                    sample_index
                ]
            ),
            alpha=0.45,
            vmin=0.0,
            vmax=1.0,
        )

        component = (
            int(component_indices[sample_index])
            if sample_index < len(component_indices)
            else -1
        )
        axis.set_title(
            f"Prior sample {sample_index + 1}\n"
            f"component {component}",
            fontsize=9,
        )
        axis.axis("off")

    for column in range(
        sample_limit,
        columns,
    ):
        axis = figure.add_subplot(
            grid[1, column]
        )
        axis.axis("off")

    # 第三行：分布和证据图。
    probability_masks_float = (
        probabilistic_masks.astype(np.float32)
    )
    mean_probability_mask = (
        probability_masks_float.mean(axis=0)
    )
    sample_variance = (
        probability_masks_float.var(axis=0)
    )

    evidence_foreground = (
        foreground_probability(
            evidence_probabilities
        )
    )
    mean_evidence_probability = (
        evidence_foreground.mean(axis=0)
    )
    mean_evidence_mask = (
        evidence_masks.astype(np.float32)
        .mean(axis=0)
    )
    mean_vacuity = mean_single_channel(
        evidence_vacuity
    )
    mean_entropy = mean_single_channel(
        evidence_entropy
    )

    bottom_items = [
        (
            mean_probability_mask,
            "Mean probabilistic mask",
            "viridis",
            0.0,
            1.0,
        ),
        (
            sample_variance,
            "Sample variance",
            "magma",
            0.0,
            None,
        ),
        (
            mean_evidence_mask,
            "Mean evidence mask",
            "viridis",
            0.0,
            1.0,
        ),
        (
            mean_evidence_probability,
            "Evidence foreground probability",
            "viridis",
            0.0,
            1.0,
        ),
        (
            mean_vacuity,
            "Mean vacuity",
            "magma",
            0.0,
            1.0,
        ),
        (
            mean_entropy,
            "Mean predictive entropy",
            "magma",
            0.0,
            1.0,
        ),
    ]

    for column, (
        value,
        title,
        cmap,
        vmin,
        vmax,
    ) in enumerate(bottom_items):
        add_image_axis(
            figure,
            grid,
            2,
            column,
            squeeze_2d(value),
            title,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=True,
        )

    for column in range(
        len(bottom_items),
        columns,
    ):
        axis = figure.add_subplot(
            grid[2, column]
        )
        axis.axis("off")

    case_id = npz_path.stem
    figure.suptitle(
        f"MS-GMM + MS-DEH (No Feedback): {case_id}",
        fontsize=15,
    )
    figure.tight_layout()

    output_directory = (
        OUTPUT_DIRECTORY / "cases"
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path = (
        output_directory
        / f"{safe_name(case_id)}.png"
    )
    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path


def generate_case_figures() -> list[Path]:
    sample_directory = (
        OUTPUT_DIRECTORY / "sample_arrays"
    )

    if not sample_directory.exists():
        print(
            "[WARN] 未找到 sample_arrays，"
            "无法绘制病例图。"
        )
        return []

    outputs: list[Path] = []

    for npz_path in sorted(
        sample_directory.glob("*.npz")
    ):
        try:
            output = create_case_figure(
                npz_path
            )
            outputs.append(output)
        except Exception as error:
            print(
                f"[WARN] 病例 {npz_path.name} "
                f"绘图失败：{error}"
            )

    return outputs


# =============================================================================
# HTML 报告
# =============================================================================

def summary_metric(
    summary: dict[str, Any],
    sample_count: int,
    metric_name: str,
) -> float | None:
    branch = summary.get(
        "probabilistic_summary"
    )
    if not isinstance(branch, dict):
        return None

    item = branch.get(str(sample_count))
    if not isinstance(item, dict):
        return None

    return metric_mean(
        item,
        metric_name,
    )


def calibration_metric(
    summary: dict[str, Any],
    candidates: list[str],
) -> float | None:
    calibration = summary.get(
        "evidence_calibration"
    )
    if not isinstance(calibration, dict):
        return None

    for key in candidates:
        value = finite_float(
            calibration.get(key)
        )
        if value is not None:
            return value
    return None


def relative_url(path: Path) -> str:
    return (
        path.relative_to(OUTPUT_DIRECTORY)
        .as_posix()
    )


def metric_card_html(
    label: str,
    value: float | None,
    direction: str = "",
) -> str:
    rendered = (
        "N/A"
        if value is None
        else f"{value:.4f}"
    )
    return f"""
    <div class="metric">
      <div class="metric-label">
        {html.escape(label)} {html.escape(direction)}
      </div>
      <div class="metric-value">{rendered}</div>
    </div>
    """


def write_html_report(
    summary: dict[str, Any],
    figure_paths: list[Path],
    case_paths: list[Path],
) -> Path:
    maximum_sample_count = max(
        [
            int(value)
            for value in summary.get(
                "sample_counts",
                [NUM_SAMPLES],
            )
        ],
        default=NUM_SAMPLES,
    )

    component = summary.get(
        "component_summary",
        {},
    )
    effective_components = (
        finite_float(
            component.get(
                "effective_component_count"
            )
        )
        if isinstance(component, dict)
        else None
    )

    cards = [
        metric_card_html(
            f"GED²@{maximum_sample_count}",
            summary_metric(
                summary,
                maximum_sample_count,
                "ged2",
            ),
            "↓",
        ),
        metric_card_html(
            f"Cross Dice@{maximum_sample_count}",
            summary_metric(
                summary,
                maximum_sample_count,
                "cross_dice",
            ),
            "↑",
        ),
        metric_card_html(
            f"Best Dice@{maximum_sample_count}",
            summary_metric(
                summary,
                maximum_sample_count,
                "best_sample_dice_to_raters",
            ),
            "↑",
        ),
        metric_card_html(
            "Evidence Dice",
            calibration_metric(
                summary,
                [
                    "hard_dice_all_raters",
                    "soft_dice_multi_rater",
                ],
            ),
            "↑",
        ),
        metric_card_html(
            "Evidence ECE",
            calibration_metric(
                summary,
                ["ece"],
            ),
            "↓",
        ),
        metric_card_html(
            "Error-detection AUROC",
            calibration_metric(
                summary,
                [
                    "error_detection_auroc",
                    "error_auroc",
                ],
            ),
            "↑",
        ),
        metric_card_html(
            "Disagreement AUROC",
            calibration_metric(
                summary,
                [
                    "rater_disagreement_auroc",
                    "disagreement_auroc",
                ],
            ),
            "↑",
        ),
        metric_card_html(
            "Effective components",
            effective_components,
            "↑",
        ),
    ]

    figure_sections = "\n".join(
        f"""
        <section class="image-panel">
          <img
            src="{html.escape(relative_url(path))}"
            alt="{html.escape(path.stem)}"
          >
        </section>
        """
        for path in figure_paths
        if path.exists()
    )

    case_sections = "\n".join(
        f"""
        <section class="image-panel">
          <img
            src="{html.escape(relative_url(path))}"
            alt="{html.escape(path.stem)}"
          >
        </section>
        """
        for path in case_paths
        if path.exists()
    )

    checkpoint_text = html.escape(
        str(summary.get("checkpoint", CHECKPOINT_PATH))
    )

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta
  name="viewport"
  content="width=device-width, initial-scale=1"
>
<title>MS-GMM + MS-DEH (No Feedback) 结果</title>
<style>
  :root {{
    color-scheme: light dark;
    --background: #f5f7fa;
    --panel: #ffffff;
    --text: #1f2937;
    --muted: #6b7280;
    --border: #d1d5db;
    --accent: #315efb;
  }}

  @media (prefers-color-scheme: dark) {{
    :root {{
      --background: #111827;
      --panel: #1f2937;
      --text: #f3f4f6;
      --muted: #9ca3af;
      --border: #374151;
      --accent: #8aa4ff;
    }}
  }}

  * {{
    box-sizing: border-box;
  }}

  body {{
    margin: 0;
    background: var(--background);
    color: var(--text);
    font-family:
      Arial,
      "Microsoft YaHei",
      sans-serif;
  }}

  main {{
    width: min(1400px, calc(100% - 32px));
    margin: 24px auto 60px;
  }}

  header,
  .metric,
  .image-panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
  }}

  header {{
    padding: 22px;
    margin-bottom: 18px;
  }}

  h1 {{
    margin: 0 0 10px;
    font-size: 28px;
  }}

  h2 {{
    margin: 30px 0 14px;
  }}

  p {{
    color: var(--muted);
    overflow-wrap: anywhere;
  }}

  .metrics {{
    display: grid;
    grid-template-columns:
      repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px;
  }}

  .metric {{
    padding: 16px;
  }}

  .metric-label {{
    color: var(--muted);
    font-size: 14px;
  }}

  .metric-value {{
    margin-top: 8px;
    color: var(--accent);
    font-size: 26px;
    font-weight: 700;
  }}

  .image-panel {{
    padding: 12px;
    margin-bottom: 16px;
  }}

  img {{
    display: block;
    width: 100%;
    height: auto;
    border-radius: 8px;
  }}

  code {{
    padding: 2px 5px;
    border-radius: 4px;
    background:
      color-mix(
        in srgb,
        var(--panel),
        var(--accent) 10%
      );
  }}
</style>
</head>
<body>
<main>
  <header>
    <h1>EG-GM-PUNet 最终结果</h1>
    <p>
      Split：
      <code>{html.escape(str(summary.get("split", SPLIT)))}</code>
      &nbsp; 病例数：
      <code>{html.escape(str(summary.get("num_cases", "N/A")))}</code>
      &nbsp; Checkpoint epoch：
      <code>{html.escape(str(summary.get("checkpoint_epoch", "N/A")))}</code>
    </p>
    <p>Checkpoint：{checkpoint_text}</p>
  </header>

  <h2>核心测试指标</h2>
  <div class="metrics">
    {''.join(cards)}
  </div>

  <h2>训练、反馈与测试统计图</h2>
  {figure_sections or '<p>没有可用统计图。</p>'}

  <h2>典型病例可视化</h2>
  {case_sections or '<p>没有可用病例图。</p>'}
</main>
</body>
</html>
"""

    report_path = (
        OUTPUT_DIRECTORY / "report.html"
    )
    report_path.write_text(
        document,
        encoding="utf-8",
    )
    return report_path


# =============================================================================
# 控制台摘要
# =============================================================================

def print_summary(
    summary: dict[str, Any],
) -> None:
    sample_counts = summary.get(
        "sample_counts",
        [],
    )
    probability_summary = summary.get(
        "probabilistic_summary",
        {},
    )

    print("\n" + "=" * 88)
    print("EG-GM-PUNet probabilistic branch")
    print("=" * 88)
    print(
        f"{'Samples':>8}  "
        f"{'GED²↓':>10}  "
        f"{'CrossDice↑':>12}  "
        f"{'BestDice↑':>11}  "
        f"{'Diversity':>10}"
    )
    print("-" * 65)

    if (
        isinstance(sample_counts, list)
        and isinstance(probability_summary, dict)
    ):
        for count in sample_counts:
            item = probability_summary.get(
                str(count),
                {},
            )
            if not isinstance(item, dict):
                continue

            ged = metric_mean(item, "ged2")
            cross = metric_mean(
                item,
                "cross_dice",
            )
            best = metric_mean(
                item,
                "best_sample_dice_to_raters",
            )
            diversity = metric_mean(
                item,
                "sample_diversity_distance",
            )

            def render(value: float | None) -> str:
                return (
                    "N/A"
                    if value is None
                    else f"{value:.4f}"
                )

            print(
                f"{int(count):>8d}  "
                f"{render(ged):>10}  "
                f"{render(cross):>12}  "
                f"{render(best):>11}  "
                f"{render(diversity):>10}"
            )

    calibration = summary.get(
        "evidence_calibration",
        {},
    )
    print("\nEvidence calibration")
    print("-" * 65)

    if isinstance(calibration, dict):
        for key, value in calibration.items():
            numeric = finite_float(value)
            if numeric is not None:
                print(
                    f"{key:42s}: {numeric:.6f}"
                )


# =============================================================================
# 主入口
# =============================================================================

def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUTPUT_DIRECTORY / "summary.json"
    )

    if (
        FORCE_REEVALUATE
        or not summary_path.exists()
    ):
        run_test_evaluation()
    else:
        print(
            "[INFO] 已存在测试 summary.json，"
            "跳过重复测试，直接生成可视化。"
        )
        print(
            f"[INFO] 如需重新测试，请把 "
            f"FORCE_REEVALUATE 改为 True。"
        )

    require_file(
        summary_path,
        "测试 summary.json",
    )
    summary = load_json(summary_path)

    # 保证旧 summary 的元数据也被修正。
    summary["model"] = "MS-GMM + MS-DEH (No Feedback)"
    summary["evidence_feedback_used"] = False
    save_json(summary_path, summary)

    figure_paths = generate_training_figures()

    component_figure = (
        generate_component_figure(summary)
    )
    if component_figure is not None:
        figure_paths.append(
            component_figure
        )

    figure_paths.extend(
        generate_sample_count_figures(summary)
    )

    case_paths = generate_case_figures()

    report_path = write_html_report(
        summary=summary,
        figure_paths=figure_paths,
        case_paths=case_paths,
    )

    print_summary(summary)

    print("\n" + "=" * 96)
    print("EG-GM-PUNet 最终测试与可视化完成")
    print("=" * 96)
    print(f"[INFO] 输出目录：{OUTPUT_DIRECTORY}")
    print(f"[INFO] 综合报告：{report_path}")
    print(f"[INFO] 定量汇总：{summary_path}")
    print(
        "[INFO] 概率分支逐病例 CSV："
        f"{OUTPUT_DIRECTORY / 'probabilistic_per_case_metrics.csv'}"
    )
    print(
        "[INFO] 证据分支逐病例 CSV："
        f"{OUTPUT_DIRECTORY / 'evidence_per_case_metrics.csv'}"
    )
    print(
        f"[INFO] 训练和统计图："
        f"{OUTPUT_DIRECTORY / 'figures'}"
    )
    print(
        f"[INFO] 典型病例图："
        f"{OUTPUT_DIRECTORY / 'cases'}"
    )
    print("=" * 96)

    if AUTO_OPEN_REPORT:
        try:
            webbrowser.open(
                report_path.resolve().as_uri()
            )
        except Exception as error:
            print(
                "[WARN] 无法自动打开浏览器："
                f"{error}"
            )
            print(
                f"请手动双击：{report_path}"
            )


if __name__ == "__main__":
    main()
