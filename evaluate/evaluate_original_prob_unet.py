"""Evaluate the Original Probabilistic U-Net with prior samples.

This script loads the best checkpoint and evaluates p(z|x) on the
validation or test split. It follows the original Probabilistic U-Net
evaluation principle by computing the squared generalized energy
distance (GED^2) with foreground IoU distance:

    d(A, B) = 1 - IoU(A, B)

    GED^2 = 2 E[d(S, Y)] - E[d(S, S')] - E[d(Y, Y')]

where S and S' are prior samples and Y and Y' are annotations.

Run from the project root, for example:

    python evaluate/evaluate_original_prob_unet.py \
        --config configs/original_prob_unet_lidc.yaml \
        --checkpoint result/v1_original_prob_unet/best_original_prob_unet.pth \
        --split val \
        --num-samples 16

To evaluate 50 samples and report 10/30/50-sample results:

    python evaluate/evaluate_original_prob_unet.py \
        --config configs/original_prob_unet_lidc.yaml \
        --checkpoint result/v1_original_prob_unet/best_original_prob_unet.pth \
        --split test \
        --num-samples 50 \
        --sample-counts 10 30 50
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from tqdm import tqdm


# ============================================================
# Project imports
#
# Current file:
#   EG-GM-PUNet/evaluate/evaluate_original_prob_unet.py
#
# parents[1]:
#   EG-GM-PUNet/
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from dataset.lidc_dataset import build_lidc_dataloader
from models.original_prob_unet.original_prob_unet_model import (
    OriginalProbUNet,
)


METRIC_NAMES = (
    "ged2",
    "cross_iou",
    "cross_dice",
    "best_sample_dice_to_raters",
    "mean_probability_mask_dice",
    "sample_diversity_distance",
    "annotation_diversity_distance",
    "sample_pairwise_iou",
    "annotation_pairwise_iou",
    "foreground_probability_variance",
    "empty_sample_fraction",
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "The YAML configuration must contain a dictionary."
        )

    return config


def set_seed(seed: int) -> None:
    """Set random seeds for deterministic evaluation."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prepare_device(device_name: str) -> torch.device:
    """Select the requested device when available."""

    if (
        device_name.lower() == "cuda"
        and torch.cuda.is_available()
    ):
        return torch.device("cuda")

    return torch.device("cpu")


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def normalize_sample_counts(
    sample_counts: list[int] | None,
    num_samples: int,
) -> list[int]:
    """Validate and normalize requested sample-count prefixes."""

    if num_samples < 1:
        raise ValueError(
            f"num_samples must be >= 1, got {num_samples}."
        )

    if sample_counts is None or len(sample_counts) == 0:
        sample_counts = [
            count
            for count in (1, 4, 8, 16)
            if count <= num_samples
        ]

        if num_samples not in sample_counts:
            sample_counts.append(num_samples)

    normalized = sorted(set(int(value) for value in sample_counts))

    for value in normalized:
        if value < 1:
            raise ValueError(
                "Every sample count must be at least one."
            )

        if value > num_samples:
            raise ValueError(
                f"sample count {value} exceeds "
                f"num_samples={num_samples}."
            )

    return normalized


def convert_case_ids(
    case_id_value: Any,
    batch_size: int,
    fallback_start_index: int,
    split: str,
) -> list[str]:
    """Convert a collated case-id value into a list of strings."""

    if case_id_value is None:
        return [
            f"{split}_{fallback_start_index + index:06d}"
            for index in range(batch_size)
        ]

    if isinstance(case_id_value, str):
        if batch_size != 1:
            raise ValueError(
                "A scalar case_id was returned for a batch "
                f"of size {batch_size}."
            )

        return [case_id_value]

    if torch.is_tensor(case_id_value):
        values = case_id_value.detach().cpu().tolist()

        if not isinstance(values, list):
            values = [values]

        return [str(value) for value in values]

    if isinstance(case_id_value, np.ndarray):
        return [
            str(value)
            for value in case_id_value.tolist()
        ]

    if isinstance(case_id_value, (list, tuple)):
        return [str(value) for value in case_id_value]

    if batch_size != 1:
        raise ValueError(
            "Unsupported case_id format for a multi-sample batch: "
            f"{type(case_id_value)}."
        )

    return [str(case_id_value)]


def flatten_binary_masks(
    masks: torch.Tensor,
) -> torch.Tensor:
    """Convert [N,H,W] binary masks to [N,P] float tensors."""

    if masks.ndim != 3:
        raise ValueError(
            "Expected masks with shape [N,H,W], "
            f"got {tuple(masks.shape)}."
        )

    return (
        (masks > 0.5)
        .to(dtype=torch.float32)
        .flatten(start_dim=1)
    )


def pairwise_iou(
    first_masks: torch.Tensor,
    second_masks: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute every foreground IoU between two mask sets.

    Args:
        first_masks:  [N,H,W]
        second_masks: [M,H,W]

    Returns:
        IoU matrix with shape [N,M].

    For two empty masks, IoU is defined as one because the masks
    agree perfectly. This gives distance zero for an empty-empty pair.
    """

    first = flatten_binary_masks(first_masks)
    second = flatten_binary_masks(second_masks)

    intersection = first @ second.transpose(0, 1)

    first_sum = first.sum(dim=1, keepdim=True)
    second_sum = second.sum(dim=1).unsqueeze(0)

    union = first_sum + second_sum - intersection

    iou = intersection / union.clamp_min(eps)

    return torch.where(
        union > 0,
        iou,
        torch.ones_like(iou),
    )


def pairwise_dice(
    first_masks: torch.Tensor,
    second_masks: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute every foreground Dice score between two mask sets."""

    first = flatten_binary_masks(first_masks)
    second = flatten_binary_masks(second_masks)

    intersection = first @ second.transpose(0, 1)

    first_sum = first.sum(dim=1, keepdim=True)
    second_sum = second.sum(dim=1).unsqueeze(0)

    denominator = first_sum + second_sum

    dice = (
        2.0 * intersection
        / denominator.clamp_min(eps)
    )

    return torch.where(
        denominator > 0,
        dice,
        torch.ones_like(dice),
    )


def evaluate_single_case(
    sample_masks: torch.Tensor,
    foreground_probabilities: torch.Tensor,
    rater_masks: torch.Tensor,
    sample_counts: Iterable[int],
) -> list[dict[str, float]]:
    """Evaluate one image for multiple prior-sample prefixes.

    Args:
        sample_masks:
            Prior segmentation samples [S,H,W].

        foreground_probabilities:
            Foreground probabilities [S,H,W].

        rater_masks:
            Multi-rater labels [R,H,W].

        sample_counts:
            Prefix lengths to evaluate, e.g. [1,4,8,16].
    """

    if sample_masks.ndim != 3:
        raise ValueError(
            "sample_masks must have shape [S,H,W]."
        )

    if foreground_probabilities.shape != sample_masks.shape:
        raise ValueError(
            "foreground_probabilities and sample_masks "
            "must have identical shapes."
        )

    if rater_masks.ndim != 3:
        raise ValueError(
            "rater_masks must have shape [R,H,W]."
        )

    rater_masks = (rater_masks > 0.5).float()

    annotation_iou = pairwise_iou(
        rater_masks,
        rater_masks,
    )

    annotation_distance = 1.0 - annotation_iou
    annotation_diversity = annotation_distance.mean()
    annotation_pairwise_iou = annotation_iou.mean()

    rows: list[dict[str, float]] = []

    for count in sample_counts:
        current_samples = sample_masks[:count]
        current_probabilities = (
            foreground_probabilities[:count]
        )

        sample_to_rater_iou = pairwise_iou(
            current_samples,
            rater_masks,
        )

        sample_to_rater_dice = pairwise_dice(
            current_samples,
            rater_masks,
        )

        sample_iou = pairwise_iou(
            current_samples,
            current_samples,
        )

        cross_distance = 1.0 - sample_to_rater_iou
        sample_distance = 1.0 - sample_iou

        # Same Monte Carlo structure as the original
        # Probabilistic U-Net evaluation:
        #
        # GED^2 = 2 E[d(S,Y)] - E[d(S,S')] - E[d(Y,Y')]
        ged2 = (
            2.0 * cross_distance.mean()
            - sample_distance.mean()
            - annotation_distance.mean()
        )

        # For each rater, select its closest prior sample.
        best_sample_dice = (
            sample_to_rater_dice
            .max(dim=0)
            .values
            .mean()
        )

        mean_probability_mask = (
            current_probabilities.mean(dim=0)
            >= 0.5
        ).float()

        mean_probability_mask_dice = pairwise_dice(
            mean_probability_mask.unsqueeze(0),
            rater_masks,
        ).mean()

        foreground_variance = (
            current_probabilities
            .var(dim=0, unbiased=False)
            .mean()
        )

        sample_foreground_pixels = (
            current_samples
            .flatten(start_dim=1)
            .sum(dim=1)
        )

        empty_sample_fraction = (
            sample_foreground_pixels == 0
        ).float().mean()

        rows.append(
            {
                "num_samples": int(count),
                "ged2": float(ged2.item()),
                "cross_iou": float(
                    sample_to_rater_iou.mean().item()
                ),
                "cross_dice": float(
                    sample_to_rater_dice.mean().item()
                ),
                "best_sample_dice_to_raters": float(
                    best_sample_dice.item()
                ),
                "mean_probability_mask_dice": float(
                    mean_probability_mask_dice.item()
                ),
                "sample_diversity_distance": float(
                    sample_distance.mean().item()
                ),
                "annotation_diversity_distance": float(
                    annotation_diversity.item()
                ),
                "sample_pairwise_iou": float(
                    sample_iou.mean().item()
                ),
                "annotation_pairwise_iou": float(
                    annotation_pairwise_iou.item()
                ),
                "foreground_probability_variance": float(
                    foreground_variance.item()
                ),
                "empty_sample_fraction": float(
                    empty_sample_fraction.item()
                ),
            }
        )

    return rows


def safe_filename(value: str) -> str:
    """Convert an identifier into a Windows-safe file name."""

    cleaned = re.sub(
        r'[^A-Za-z0-9._-]+',
        "_",
        value,
    )

    return cleaned.strip("._") or "sample"


def save_case_arrays(
    output_directory: Path,
    case_id: str,
    image: torch.Tensor,
    rater_masks: torch.Tensor,
    sample_masks: torch.Tensor,
    foreground_probabilities: torch.Tensor,
) -> None:
    """Save one evaluated case as a compressed NumPy archive."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path = (
        output_directory
        / f"{safe_filename(case_id)}.npz"
    )

    np.savez_compressed(
        file_path,
        image=(
            image.detach().cpu().numpy().astype(np.float32)
        ),
        rater_masks=(
            rater_masks
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        ),
        prior_sample_masks=(
            sample_masks
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        ),
        foreground_probabilities=(
            foreground_probabilities
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        ),
    )


def summarize_rows(
    rows: list[dict[str, Any]],
    sample_counts: list[int],
) -> dict[str, Any]:
    """Aggregate per-case metrics into mean and population std."""

    summary: dict[str, Any] = {}

    for count in sample_counts:
        selected_rows = [
            row
            for row in rows
            if int(row["num_samples"]) == count
        ]

        if not selected_rows:
            raise RuntimeError(
                f"No evaluation rows found for {count} samples."
            )

        count_summary: dict[str, Any] = {
            "num_cases": len(selected_rows),
        }

        for metric_name in METRIC_NAMES:
            values = np.asarray(
                [
                    float(row[metric_name])
                    for row in selected_rows
                ],
                dtype=np.float64,
            )

            count_summary[metric_name] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            }

        summary[str(count)] = count_summary

    return summary


def write_rows_to_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write per-case metrics to CSV."""

    if not rows:
        raise RuntimeError(
            "Cannot write an empty metrics table."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows[0].keys())

    with open(
        output_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    summary: dict[str, Any],
    sample_counts: list[int],
) -> None:
    """Print the main prior-sampling metrics."""

    print("\n" + "=" * 104)
    print("Prior-sampling evaluation summary")
    print("=" * 104)

    header = (
        f"{'Samples':>8}  "
        f"{'GED^2↓':>12}  "
        f"{'Cross IoU↑':>12}  "
        f"{'Cross Dice↑':>12}  "
        f"{'Best Dice↑':>12}  "
        f"{'Diversity↑':>12}"
    )

    print(header)
    print("-" * len(header))

    for count in sample_counts:
        item = summary[str(count)]

        print(
            f"{count:>8d}  "
            f"{item['ged2']['mean']:>12.4f}  "
            f"{item['cross_iou']['mean']:>12.4f}  "
            f"{item['cross_dice']['mean']:>12.4f}  "
            f"{item['best_sample_dice_to_raters']['mean']:>12.4f}  "
            f"{item['sample_diversity_distance']['mean']:>12.4f}"
        )

    print("=" * 104)


@torch.no_grad()
def evaluate(
    model: OriginalProbUNet,
    loader,
    device: torch.device,
    split: str,
    num_samples: int,
    sample_counts: list[int],
    save_examples: int,
    example_directory: Path,
) -> list[dict[str, Any]]:
    """Evaluate prior samples over a complete data split."""

    model.eval()

    rows: list[dict[str, Any]] = []

    fallback_index = 0
    saved_examples = 0

    progress_bar = tqdm(
        loader,
        desc=f"Evaluate {split}",
        ncols=130,
    )

    for batch in progress_bar:
        if "image" not in batch:
            raise KeyError(
                "The dataset batch does not contain 'image'."
            )

        if "masks" not in batch:
            raise KeyError(
                "The dataset batch does not contain 'masks'. "
                "Multi-rater masks are required for GED."
            )

        images = batch["image"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        all_rater_masks = batch["masks"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        if all_rater_masks.ndim != 4:
            raise ValueError(
                "Expected batch['masks'] with shape [B,R,H,W], "
                f"got {tuple(all_rater_masks.shape)}."
            )

        batch_size = images.shape[0]

        case_ids = convert_case_ids(
            case_id_value=batch.get("case_id"),
            batch_size=batch_size,
            fallback_start_index=fallback_index,
            split=split,
        )

        if len(case_ids) != batch_size:
            raise ValueError(
                "The number of case IDs does not match "
                f"the batch size: {len(case_ids)} vs {batch_size}."
            )

        inference_state = model(
            image=images,
            segmentation=None,
            is_training=False,
        )

        if inference_state.posterior is not None:
            raise RuntimeError(
                "Posterior was unexpectedly created "
                "during prior-only evaluation."
            )

        sample_logits = model.sample(
            state=inference_state,
            num_samples=num_samples,
        )

        # [S,B,2,H,W] -> [S,B,H,W]
        foreground_probabilities = torch.softmax(
            sample_logits,
            dim=2,
        )[:, :, 1]

        sample_masks = sample_logits.argmax(
            dim=2
        ).to(dtype=torch.float32)

        for batch_index in range(batch_size):
            current_case_id = case_ids[batch_index]

            case_rows = evaluate_single_case(
                sample_masks=sample_masks[
                    :, batch_index
                ],
                foreground_probabilities=(
                    foreground_probabilities[
                        :, batch_index
                    ]
                ),
                rater_masks=all_rater_masks[
                    batch_index
                ],
                sample_counts=sample_counts,
            )

            for row in case_rows:
                row.update(
                    {
                        "case_id": current_case_id,
                        "split": split,
                    }
                )

            rows.extend(case_rows)

            if saved_examples < save_examples:
                save_case_arrays(
                    output_directory=example_directory,
                    case_id=current_case_id,
                    image=images[batch_index],
                    rater_masks=all_rater_masks[
                        batch_index
                    ],
                    sample_masks=sample_masks[
                        :, batch_index
                    ],
                    foreground_probabilities=(
                        foreground_probabilities[
                            :, batch_index
                        ]
                    ),
                )

                saved_examples += 1

        fallback_index += batch_size

        latest_count = sample_counts[-1]
        latest_rows = [
            row
            for row in rows
            if int(row["num_samples"])
            == latest_count
        ]

        if latest_rows:
            running_ged = np.mean(
                [
                    row["ged2"]
                    for row in latest_rows
                ]
            )

            progress_bar.set_postfix(
                {
                    f"GED2@{latest_count}": (
                        f"{running_ged:.4f}"
                    )
                }
            )

    if not rows:
        raise RuntimeError(
            "The evaluation loader produced no samples."
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Original Probabilistic U-Net "
            "using prior samples."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=(
            "configs/original_prob_unet_lidc.yaml"
        ),
        help="Path to the experiment YAML configuration.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "result/v1_original_prob_unet/"
            "best_original_prob_unet.pth"
        ),
        help="Path to the best trained checkpoint.",
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=("val", "test"),
        default="val",
        help="Dataset split to evaluate.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help=(
            "Total number of prior samples generated "
            "for every image."
        ),
    )

    parser.add_argument(
        "--sample-counts",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Sample prefixes to report. "
            "Default: 1 4 8 16, filtered by num-samples."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Evaluation batch size. By default, use "
            "training.batch_size from the YAML file."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "DataLoader workers. By default, use "
            "training.num_workers from the YAML file."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Evaluation sampling seed. By default, "
            "use the YAML experiment seed."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Evaluation device. By default, use "
            "the device in the YAML file."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Optional output directory. If omitted, use "
            "<save_dir>/evaluation/<split>/samples_<N>."
        ),
    )

    parser.add_argument(
        "--save-examples",
        type=int,
        default=5,
        help=(
            "Number of cases saved as compressed NPZ files. "
            "Set to zero to disable."
        ),
    )

    args = parser.parse_args()

    config = load_yaml(args.config)

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    training_config = config["training"]

    evaluation_seed = int(
        args.seed
        if args.seed is not None
        else config["seed"]
    )

    set_seed(evaluation_seed)

    device_name = (
        args.device
        if args.device is not None
        else config.get("device", "cuda")
    )

    device = prepare_device(device_name)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain model_state_dict."
        )

    # Prefer the model configuration stored in the checkpoint,
    # because it is the exact architecture used for training.
    checkpoint_config = checkpoint.get("config")

    if (
        isinstance(checkpoint_config, dict)
        and "model" in checkpoint_config
    ):
        model_config = checkpoint_config["model"]
    else:
        model_config = config["model"]

    model = OriginalProbUNet(
        in_channels=int(
            model_config["in_channels"]
        ),
        num_classes=int(
            model_config["num_classes"]
        ),
        latent_dim=int(
            model_config["latent_dim"]
        ),
        num_channels=tuple(
            int(channel)
            for channel in model_config[
                "num_channels"
            ]
        ),
        num_convs_per_block=int(
            model_config["num_convs_per_block"]
        ),
        num_1x1_convs=int(
            model_config["num_1x1_convs"]
        ),
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else training_config["batch_size"]
    )

    num_workers = int(
        args.num_workers
        if args.num_workers is not None
        else training_config["num_workers"]
    )

    sample_counts = normalize_sample_counts(
        sample_counts=args.sample_counts,
        num_samples=args.num_samples,
    )

    loader = build_lidc_dataloader(
        root=config["data_root"],
        split=args.split,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        random_rater=False,
        rater_index=0,
    )

    if args.output_dir is None:
        output_directory = (
            Path(config["save_dir"])
            / "evaluation"
            / args.split
            / f"samples_{args.num_samples}"
        )
    else:
        output_directory = Path(args.output_dir)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    example_directory = (
        output_directory / "sample_arrays"
    )

    print("=" * 76)
    print(
        "Original Probabilistic U-Net prior evaluation"
    )
    print("=" * 76)
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Split: {args.split}")
    print(f"[INFO] Cases: {len(loader.dataset)}")
    print(f"[INFO] Batch size: {batch_size}")
    print(
        f"[INFO] Prior samples per case: "
        f"{args.num_samples}"
    )
    print(
        f"[INFO] Report sample counts: "
        f"{sample_counts}"
    )
    print(
        f"[INFO] Evaluation seed: "
        f"{evaluation_seed}"
    )
    print(
        f"[INFO] Trainable parameters: "
        f"{count_trainable_parameters(model):,}"
    )
    print(
        f"[INFO] Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )
    print(
        f"[INFO] Checkpoint best val loss: "
        f"{checkpoint.get('best_val_loss', 'unknown')}"
    )

    rows = evaluate(
        model=model,
        loader=loader,
        device=device,
        split=args.split,
        num_samples=args.num_samples,
        sample_counts=sample_counts,
        save_examples=max(
            int(args.save_examples),
            0,
        ),
        example_directory=example_directory,
    )

    summary = summarize_rows(
        rows=rows,
        sample_counts=sample_counts,
    )

    per_case_path = (
        output_directory
        / "per_case_metrics.csv"
    )

    summary_path = (
        output_directory
        / "summary.json"
    )

    write_rows_to_csv(
        rows=rows,
        output_path=per_case_path,
    )

    result_document = {
        "model": (
            "Original Probabilistic U-Net reproduction"
        ),
        "checkpoint": str(
            checkpoint_path.resolve()
        ),
        "checkpoint_epoch": checkpoint.get(
            "epoch"
        ),
        "checkpoint_best_val_loss": checkpoint.get(
            "best_val_loss"
        ),
        "split": args.split,
        "num_cases": len(loader.dataset),
        "num_raters": 4,
        "num_prior_samples": args.num_samples,
        "sample_counts": sample_counts,
        "evaluation_seed": evaluation_seed,
        "metric_definition": {
            "distance": "1 - foreground IoU",
            "ged2": (
                "2*mean[d(S,Y)] - mean[d(S,S')] "
                "- mean[d(Y,Y')]"
            ),
            "pair_sampling": (
                "All Cartesian pairs, including diagonal "
                "self-pairs, matching the public reference "
                "evaluation structure."
            ),
            "empty_empty_iou": 1.0,
            "std": "population standard deviation across cases",
        },
        "summary": summary,
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result_document,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print_summary(
        summary=summary,
        sample_counts=sample_counts,
    )

    print(f"[INFO] Per-case CSV: {per_case_path}")
    print(f"[INFO] Summary JSON: {summary_path}")

    if args.save_examples > 0:
        print(
            f"[INFO] Saved sample arrays: "
            f"{example_directory}"
        )

    print(
        "[DONE] Prior-sampling evaluation finished."
    )


if __name__ == "__main__":
    main()