"""Compare Global GMM and Multi-Scale Grouped GMM experiments.

Run from the project root:

    python tools/compare_gmm_results.py

Optional custom paths:

    python tools/compare_gmm_results.py ^
        --global-dir result/v2_global_gmm_prob_unet_k5 ^
        --multiscale-dir result/v2b_multiscale_grouped_gmm_k5
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GLOBAL_DIR = (
    PROJECT_ROOT
    / "result"
    / "v2_global_gmm_prob_unet_k5"
)

DEFAULT_MULTISCALE_DIR = (
    PROJECT_ROOT
    / "result"
    / "v2b_multiscale_grouped_gmm_k5"
)


def load_json(path: Path) -> Any:
    """Load a JSON file and raise a clear error when missing."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path.resolve()}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def safe_float(
    value: Any,
    default: float = float("nan"),
) -> float:
    """Convert a value to float without crashing."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_float(
    value: Any,
    digits: int = 4,
) -> str:
    """Format a number for terminal output."""

    number = safe_float(value)

    if math.isnan(number):
        return "N/A"

    return f"{number:.{digits}f}"


def format_epoch_metric(
    record: dict[str, Any] | None,
    digits: int = 4,
) -> str:
    """Format a best-record value together with its epoch."""

    if not record:
        return "N/A"

    value = format_float(
        record.get("value"),
        digits=digits,
    )

    epoch = record.get("epoch")

    if epoch is None:
        return value

    return f"{value} (ep {epoch})"


def nested_get(
    data: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    """Safely access nested dictionary keys."""

    current: Any = data

    for key in keys:
        if not isinstance(current, dict):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def metric_mean(
    evaluation: dict[str, Any],
    sample_count: int,
    metric_name: str,
) -> float:
    """Read the mean of one evaluation metric."""

    return safe_float(
        nested_get(
            evaluation,
            "summary",
            str(sample_count),
            metric_name,
            "mean",
        )
    )


def print_table(
    headers: list[str],
    rows: list[list[Any]],
    right_align_columns: set[int] | None = None,
) -> None:
    """Print a dependency-free aligned terminal table."""

    if right_align_columns is None:
        right_align_columns = set()

    string_rows = [
        [str(value) for value in row]
        for row in rows
    ]

    widths = []

    for column_index, header in enumerate(headers):
        column_values = [
            row[column_index]
            for row in string_rows
        ]

        widths.append(
            max(
                len(str(header)),
                max(
                    (
                        len(value)
                        for value in column_values
                    ),
                    default=0,
                ),
            )
        )

    def format_row(values: list[str]) -> str:
        formatted = []

        for index, value in enumerate(values):
            if index in right_align_columns:
                formatted.append(
                    value.rjust(widths[index])
                )
            else:
                formatted.append(
                    value.ljust(widths[index])
                )

        return " | ".join(formatted)

    print(
        format_row(
            [str(header) for header in headers]
        )
    )

    print(
        "-+-".join(
            "-" * width
            for width in widths
        )
    )

    for row in string_rows:
        print(format_row(row))


def load_experiment(
    root: Path,
) -> dict[str, Any]:
    """Load the main training and evaluation outputs."""

    return {
        "root": root,
        "history": load_json(
            root / "history.json"
        ),
        "best_records": load_json(
            root / "best_records.json"
        ),
        "training_summary": load_json(
            root / "training_summary.json"
        ),
        "run_manifest": load_json(
            root / "run_manifest.json"
        ),
        "evaluation": load_json(
            root
            / "evaluation"
            / "test"
            / "samples_16"
            / "summary.json"
        ),
    }


def final_history_record(
    experiment: dict[str, Any],
) -> dict[str, Any]:
    """Return the final epoch history entry."""

    history = experiment["history"]

    if not isinstance(history, list) or not history:
        raise RuntimeError(
            f"Empty history in {experiment['root']}"
        )

    return history[-1]


def common_sample_counts(
    global_evaluation: dict[str, Any],
    multiscale_evaluation: dict[str, Any],
) -> list[int]:
    """Get sample counts present in both evaluations."""

    global_counts = {
        int(value)
        for value in global_evaluation[
            "summary"
        ].keys()
    }

    multiscale_counts = {
        int(value)
        for value in multiscale_evaluation[
            "summary"
        ].keys()
    }

    counts = sorted(
        global_counts.intersection(
            multiscale_counts
        )
    )

    if not counts:
        raise RuntimeError(
            "The two evaluation files have no common "
            "sample counts."
        )

    return counts


def relative_improvement_lower_is_better(
    baseline: float,
    new_value: float,
) -> float:
    """Calculate relative improvement for a lower-is-better metric."""

    if (
        math.isnan(baseline)
        or math.isnan(new_value)
        or abs(baseline) < 1e-12
    ):
        return float("nan")

    return (
        (baseline - new_value)
        / abs(baseline)
        * 100.0
    )


def winner_lower(
    first: float,
    second: float,
) -> str:
    """Return the winner for a lower-is-better metric."""

    if math.isnan(first) or math.isnan(second):
        return "N/A"

    if abs(first - second) < 1e-8:
        return "Tie"

    return (
        "Multi-scale"
        if second < first
        else "Global"
    )


def winner_higher(
    first: float,
    second: float,
) -> str:
    """Return the winner for a higher-is-better metric."""

    if math.isnan(first) or math.isnan(second):
        return "N/A"

    if abs(first - second) < 1e-8:
        return "Tie"

    return (
        "Multi-scale"
        if second > first
        else "Global"
    )


def get_component_summary(
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Read component statistics from evaluation output."""

    component_summary = evaluation.get(
        "component_summary",
        {}
    )

    return (
        component_summary
        if isinstance(component_summary, dict)
        else {}
    )


def format_probability_list(
    values: Any,
    digits: int = 3,
) -> str:
    """Format mixture probabilities compactly."""

    if not isinstance(values, list):
        return "N/A"

    return "[" + ", ".join(
        format_float(value, digits)
        for value in values
    ) + "]"


def component_collapse_warning(
    component_summary: dict[str, Any],
    expected_components: int,
) -> str:
    """Produce a simple component-collapse diagnostic."""

    probabilities = component_summary.get(
        "mean_prior_component_probabilities",
        []
    )

    effective_count = safe_float(
        component_summary.get(
            "prior_effective_component_count"
        )
    )

    warnings = []

    if (
        not math.isnan(effective_count)
        and effective_count
        < 0.5 * expected_components
    ):
        warnings.append(
            f"low K_eff={effective_count:.2f}"
        )

    if isinstance(probabilities, list):
        minimum_probability = min(
            (
                safe_float(value)
                for value in probabilities
            ),
            default=float("nan"),
        )

        maximum_probability = max(
            (
                safe_float(value)
                for value in probabilities
            ),
            default=float("nan"),
        )

        if (
            not math.isnan(minimum_probability)
            and minimum_probability < 0.02
        ):
            warnings.append(
                f"inactive component p_min="
                f"{minimum_probability:.3f}"
            )

        if (
            not math.isnan(maximum_probability)
            and maximum_probability > 0.90
        ):
            warnings.append(
                f"dominant component p_max="
                f"{maximum_probability:.3f}"
            )

    if warnings:
        return "WARNING: " + "; ".join(warnings)

    return "No obvious collapse"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Global GMM and Multi-Scale "
            "Grouped GMM experiments."
        )
    )

    parser.add_argument(
        "--global-dir",
        type=Path,
        default=DEFAULT_GLOBAL_DIR,
    )

    parser.add_argument(
        "--multiscale-dir",
        type=Path,
        default=DEFAULT_MULTISCALE_DIR,
    )

    args = parser.parse_args()

    global_experiment = load_experiment(
        args.global_dir
    )

    multiscale_experiment = load_experiment(
        args.multiscale_dir
    )

    global_last = final_history_record(
        global_experiment
    )

    multiscale_last = final_history_record(
        multiscale_experiment
    )

    global_evaluation = (
        global_experiment["evaluation"]
    )

    multiscale_evaluation = (
        multiscale_experiment["evaluation"]
    )

    sample_counts = common_sample_counts(
        global_evaluation,
        multiscale_evaluation,
    )

    print("\n" + "=" * 100)
    print(
        "GLOBAL GMM vs MULTI-SCALE GROUPED GMM"
    )
    print("=" * 100)

    print(
        f"Global directory     : "
        f"{args.global_dir.resolve()}"
    )

    print(
        f"Multi-scale directory: "
        f"{args.multiscale_dir.resolve()}"
    )

    # ========================================================
    # Best validation checkpoints
    # ========================================================
    print("\n[1] Best validation records")

    global_best = global_experiment[
        "best_records"
    ]

    multiscale_best = multiscale_experiment[
        "best_records"
    ]

    best_rows = [
        [
            "Val loss (lower)",
            format_epoch_metric(
                global_best.get("val_loss")
            ),
            format_epoch_metric(
                multiscale_best.get("val_loss")
            ),
            winner_lower(
                safe_float(
                    nested_get(
                        global_best,
                        "val_loss",
                        "value",
                    )
                ),
                safe_float(
                    nested_get(
                        multiscale_best,
                        "val_loss",
                        "value",
                    )
                ),
            ),
        ],
        [
            "Val Dice (higher)",
            format_epoch_metric(
                global_best.get("val_dice")
            ),
            format_epoch_metric(
                multiscale_best.get("val_dice")
            ),
            winner_higher(
                safe_float(
                    nested_get(
                        global_best,
                        "val_dice",
                        "value",
                    )
                ),
                safe_float(
                    nested_get(
                        multiscale_best,
                        "val_dice",
                        "value",
                    )
                ),
            ),
        ],
        [
            "Val GED@4 (lower)",
            format_epoch_metric(
                global_best.get("val_ged2")
            ),
            format_epoch_metric(
                multiscale_best.get(
                    "val_ged2"
                )
            ),
            winner_lower(
                safe_float(
                    nested_get(
                        global_best,
                        "val_ged2",
                        "value",
                    )
                ),
                safe_float(
                    nested_get(
                        multiscale_best,
                        "val_ged2",
                        "value",
                    )
                ),
            ),
        ],
    ]

    print_table(
        headers=[
            "Metric",
            "Global GMM",
            "Multi-scale GMM",
            "Better",
        ],
        rows=best_rows,
        right_align_columns={1, 2},
    )

    # ========================================================
    # Last epoch diagnostics
    # ========================================================
    print("\n[2] Last epoch diagnostics")

    final_rows = [
        [
            "Epoch",
            global_last.get("epoch", "N/A"),
            multiscale_last.get(
                "epoch",
                "N/A",
            ),
        ],
        [
            "Train CE mean",
            format_float(
                global_last.get(
                    "train_reconstruction_loss_mean"
                )
            ),
            format_float(
                multiscale_last.get(
                    "train_reconstruction_loss_mean"
                )
            ),
        ],
        [
            "Train Dice",
            format_float(
                global_last.get("train_dice")
            ),
            format_float(
                multiscale_last.get("train_dice")
            ),
        ],
        [
            "Val CE mean",
            format_float(
                global_last.get(
                    "val_reconstruction_loss_mean"
                )
            ),
            format_float(
                multiscale_last.get(
                    "val_reconstruction_loss_mean"
                )
            ),
        ],
        [
            "Val Dice",
            format_float(
                global_last.get("val_dice")
            ),
            format_float(
                multiscale_last.get("val_dice")
            ),
        ],
        [
            "Val GED@4",
            format_float(
                global_last.get(
                    "val_prior_ged2"
                )
            ),
            format_float(
                multiscale_last.get(
                    "val_prior_ged2"
                )
            ),
        ],
        [
            "Val prior K_eff",
            format_float(
                global_last.get(
                    "val_prior_effective_components"
                ),
                digits=3,
            ),
            format_float(
                multiscale_last.get(
                    "val_prior_effective_components"
                ),
                digits=3,
            ),
        ],
    ]

    print_table(
        headers=[
            "Metric",
            "Global GMM",
            "Multi-scale GMM",
        ],
        rows=final_rows,
        right_align_columns={1, 2},
    )

    # ========================================================
    # Test performance
    # ========================================================
    print("\n[3] Test prior-sampling performance")

    performance_rows = []

    for count in sample_counts:
        global_ged = metric_mean(
            global_evaluation,
            count,
            "ged2",
        )

        multiscale_ged = metric_mean(
            multiscale_evaluation,
            count,
            "ged2",
        )

        global_cross_dice = metric_mean(
            global_evaluation,
            count,
            "cross_dice",
        )

        multiscale_cross_dice = metric_mean(
            multiscale_evaluation,
            count,
            "cross_dice",
        )

        global_best_dice = metric_mean(
            global_evaluation,
            count,
            "best_sample_dice_to_raters",
        )

        multiscale_best_dice = metric_mean(
            multiscale_evaluation,
            count,
            "best_sample_dice_to_raters",
        )

        performance_rows.append(
            [
                count,
                format_float(global_ged),
                format_float(multiscale_ged),
                format_float(
                    multiscale_ged - global_ged
                ),
                winner_lower(
                    global_ged,
                    multiscale_ged,
                ),
                format_float(
                    global_cross_dice
                ),
                format_float(
                    multiscale_cross_dice
                ),
                format_float(
                    global_best_dice
                ),
                format_float(
                    multiscale_best_dice
                ),
            ]
        )

    print_table(
        headers=[
            "Samples",
            "G GED2",
            "M GED2",
            "Delta M-G",
            "GED winner",
            "G CrossDice",
            "M CrossDice",
            "G BestDice",
            "M BestDice",
        ],
        rows=performance_rows,
        right_align_columns={
            0, 1, 2, 3, 5, 6, 7, 8
        },
    )

    # ========================================================
    # Diversity comparison
    # ========================================================
    print("\n[4] Diversity matching")

    diversity_rows = []

    for count in sample_counts:
        global_sample_diversity = metric_mean(
            global_evaluation,
            count,
            "sample_diversity_distance",
        )

        multiscale_sample_diversity = metric_mean(
            multiscale_evaluation,
            count,
            "sample_diversity_distance",
        )

        global_annotation_diversity = metric_mean(
            global_evaluation,
            count,
            "annotation_diversity_distance",
        )

        multiscale_annotation_diversity = metric_mean(
            multiscale_evaluation,
            count,
            "annotation_diversity_distance",
        )

        annotation_diversity = (
            global_annotation_diversity
            + multiscale_annotation_diversity
        ) / 2.0

        global_gap = abs(
            global_sample_diversity
            - annotation_diversity
        )

        multiscale_gap = abs(
            multiscale_sample_diversity
            - annotation_diversity
        )

        diversity_rows.append(
            [
                count,
                format_float(
                    annotation_diversity
                ),
                format_float(
                    global_sample_diversity
                ),
                format_float(
                    multiscale_sample_diversity
                ),
                format_float(global_gap),
                format_float(multiscale_gap),
                winner_lower(
                    global_gap,
                    multiscale_gap,
                ),
            ]
        )

    print_table(
        headers=[
            "Samples",
            "GT diversity",
            "G diversity",
            "M diversity",
            "G gap",
            "M gap",
            "Closer to GT",
        ],
        rows=diversity_rows,
        right_align_columns={
            0, 1, 2, 3, 4, 5
        },
    )

    # ========================================================
    # Component use
    # ========================================================
    print("\n[5] Mixture-component usage")

    global_components = get_component_summary(
        global_evaluation
    )

    multiscale_components = (
        get_component_summary(
            multiscale_evaluation
        )
    )

    component_rows = [
        [
            "Effective components",
            format_float(
                global_components.get(
                    "prior_effective_component_count"
                ),
                digits=3,
            ),
            format_float(
                multiscale_components.get(
                    "prior_effective_component_count"
                ),
                digits=3,
            ),
        ],
        [
            "Mean prior probabilities",
            format_probability_list(
                global_components.get(
                    "mean_prior_component_probabilities"
                )
            ),
            format_probability_list(
                multiscale_components.get(
                    "mean_prior_component_probabilities"
                )
            ),
        ],
        [
            "Sample frequencies",
            format_probability_list(
                global_components.get(
                    "sampled_component_frequencies"
                )
            ),
            format_probability_list(
                multiscale_components.get(
                    "sampled_component_frequencies"
                )
            ),
        ],
        [
            "Collapse diagnostic",
            component_collapse_warning(
                global_components,
                expected_components=5,
            ),
            component_collapse_warning(
                multiscale_components,
                expected_components=5,
            ),
        ],
    ]

    print_table(
        headers=[
            "Statistic",
            "Global GMM",
            "Multi-scale GMM",
        ],
        rows=component_rows,
    )

    # ========================================================
    # Multi-scale group KL
    # ========================================================
    print("\n[6] Multi-scale latent-group KL at last epoch")

    group_kl_rows = [
        [
            "Train categorical KL",
            format_float(
                multiscale_last.get(
                    "train_kl_categorical"
                )
            ),
        ],
        [
            "Train coarse KL",
            format_float(
                multiscale_last.get(
                    "train_kl_coarse"
                )
            ),
        ],
        [
            "Train middle KL",
            format_float(
                multiscale_last.get(
                    "train_kl_middle"
                )
            ),
        ],
        [
            "Train fine KL",
            format_float(
                multiscale_last.get(
                    "train_kl_fine"
                )
            ),
        ],
        [
            "Val categorical KL",
            format_float(
                multiscale_last.get(
                    "val_kl_categorical"
                )
            ),
        ],
        [
            "Val coarse KL",
            format_float(
                multiscale_last.get(
                    "val_kl_coarse"
                )
            ),
        ],
        [
            "Val middle KL",
            format_float(
                multiscale_last.get(
                    "val_kl_middle"
                )
            ),
        ],
        [
            "Val fine KL",
            format_float(
                multiscale_last.get(
                    "val_kl_fine"
                )
            ),
        ],
    ]

    print_table(
        headers=[
            "Multi-scale KL term",
            "Value",
        ],
        rows=group_kl_rows,
        right_align_columns={1},
    )

    # ========================================================
    # Simple conclusion
    # ========================================================
    final_count = max(sample_counts)

    global_final_ged = metric_mean(
        global_evaluation,
        final_count,
        "ged2",
    )

    multiscale_final_ged = metric_mean(
        multiscale_evaluation,
        final_count,
        "ged2",
    )

    global_final_best_dice = metric_mean(
        global_evaluation,
        final_count,
        "best_sample_dice_to_raters",
    )

    multiscale_final_best_dice = metric_mean(
        multiscale_evaluation,
        final_count,
        "best_sample_dice_to_raters",
    )

    improvement = (
        relative_improvement_lower_is_better(
            global_final_ged,
            multiscale_final_ged,
        )
    )

    print("\n[7] Quick conclusion")

    print(
        f"GED2@{final_count}: "
        f"Global={global_final_ged:.4f}, "
        f"Multi-scale={multiscale_final_ged:.4f}"
    )

    if not math.isnan(improvement):
        if improvement > 0:
            print(
                "Multi-scale relative GED improvement: "
                f"{improvement:.2f}%"
            )
        else:
            print(
                "Multi-scale relative GED change: "
                f"{improvement:.2f}% "
                "(negative means worse)"
            )

    print(
        f"Best Dice@{final_count}: "
        f"Global={global_final_best_dice:.4f}, "
        f"Multi-scale="
        f"{multiscale_final_best_dice:.4f}"
    )

    print(
        "GED winner: "
        f"{winner_lower(global_final_ged, multiscale_final_ged)}"
    )

    print(
        "Best-Dice winner: "
        f"{winner_higher(global_final_best_dice, multiscale_final_best_dice)}"
    )

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()