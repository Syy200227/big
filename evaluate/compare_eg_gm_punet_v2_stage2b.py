
"""
Stage 2b vs Stage 2a / Stage1 / Fixed EG-ALC / No-Feedback。

固定比较：
- GED² @1/4/8/16
- CrossDice @1/4/8/16
- BestDice @1/4/8/16
- SampleDiversity @1/4/8/16
- slice-level / patient-level
- paired bootstrap 95% CI
- Wilcoxon signed-rank
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS = {
    "ged2": "lower",
    "cross_dice": "higher",
    "best_sample_dice_to_raters": "higher",
    "sample_diversity_distance": "context",
}


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"CSV 不存在：{path}")
    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def patient_id(case_id: str):
    match = re.search(
        r"(LIDC-IDRI-\d{4})",
        str(case_id),
    )
    return (
        match.group(1)
        if match
        else str(case_id).split("_")[0]
    )


def index_rows(rows):
    return {
        (str(row["case_id"]), int(row["num_samples"])): row
        for row in rows
    }


def bootstrap_ci(values, n_boot, seed):
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):
        indices = rng.integers(
            0,
            len(values),
            size=len(values),
        )
        means[i] = values[indices].mean()

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def wilcoxon_p(candidate, reference):
    if wilcoxon is None:
        return None

    difference = candidate - reference
    if np.allclose(difference, 0.0):
        return 1.0

    try:
        return float(
            wilcoxon(
                candidate,
                reference,
                zero_method="wilcox",
                alternative="two-sided",
            ).pvalue
        )
    except ValueError:
        return None


def paired(reference, candidate, count, metric):
    keys = sorted(
        key
        for key in set(reference).intersection(candidate)
        if key[1] == count
    )

    cases = [key[0] for key in keys]
    ref = np.asarray(
        [float(reference[key][metric]) for key in keys],
        dtype=np.float64,
    )
    new = np.asarray(
        [float(candidate[key][metric]) for key in keys],
        dtype=np.float64,
    )

    return cases, ref, new


def aggregate_patient(cases, ref, new):
    groups = {}

    for index, case in enumerate(cases):
        groups.setdefault(
            patient_id(case),
            [],
        ).append(index)

    ref_values = []
    new_values = []

    for _, indices in sorted(groups.items()):
        ref_values.append(
            float(ref[indices].mean())
        )
        new_values.append(
            float(new[indices].mean())
        )

    return (
        np.asarray(ref_values),
        np.asarray(new_values),
    )


def compare(
    *,
    reference_name,
    reference,
    candidate,
    count,
    metric,
    level,
    n_boot,
    seed,
):
    cases, ref, new = paired(
        reference,
        candidate,
        count,
        metric,
    )

    if level == "patient":
        ref, new = aggregate_patient(
            cases,
            ref,
            new,
        )

    if METRICS[metric] == "lower":
        improvement = ref - new
    elif METRICS[metric] == "higher":
        improvement = new - ref
    else:
        improvement = new - ref

    ci_low, ci_high = bootstrap_ci(
        improvement,
        n_boot,
        seed,
    )

    return {
        "reference": reference_name,
        "candidate": "Stage 2b Bounded Demand-Capacity",
        "level": level,
        "num_pairs": int(len(ref)),
        "num_samples": int(count),
        "metric": metric,
        "direction": METRICS[metric],
        "reference_mean": float(ref.mean()),
        "candidate_mean": float(new.mean()),
        "improvement_mean": float(improvement.mean()),
        "improvement_ci95_low": ci_low,
        "improvement_ci95_high": ci_high,
        "wilcoxon_p": wilcoxon_p(
            new,
            ref,
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage2b",
        default=(
            "result/eg_gm_punet_v2_bounded_demand_capacity_k5_full/"
            "evaluation/test/samples_16/"
            "best_val_ged4_eg_gm_punet_v2_stage2b/"
            "probabilistic_per_case_metrics.csv"
        ),
    )
    parser.add_argument(
        "--stage2a",
        default=(
            "result/eg_gm_punet_v2_demand_capacity_k5_full/"
            "evaluation/test/samples_16/"
            "best_val_ged4_eg_gm_punet_v2_stage2/"
            "probabilistic_per_case_metrics.csv"
        ),
    )
    parser.add_argument(
        "--stage1",
        default=(
            "result/eg_gm_punet_v2_relative_feedback_k5_full/"
            "evaluation/test/samples_16/"
            "best_val_ged4_eg_gm_punet_v2_stage1/"
            "probabilistic_per_case_metrics.csv"
        ),
    )
    parser.add_argument(
        "--fixed",
        default=(
            "result/eg_gm_punet_k5_full/"
            "evaluation/test/samples_16/"
            "best_val_ged4_eg_gm_punet/"
            "probabilistic_per_case_metrics.csv"
        ),
    )
    parser.add_argument(
        "--no-feedback",
        dest="no_feedback",
        default=(
            "result/eg_gm_punet_no_feedback_k5_full/"
            "evaluation/test/samples_16/"
            "best_val_ged4_eg_gm_punet/"
            "no_feedback_probabilistic_per_case_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "result/eg_gm_punet_v2_bounded_demand_capacity_k5_full/"
            "stage2b_comparison"
        ),
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    args = parser.parse_args()

    def resolve(value):
        path = Path(value)
        return (
            path
            if path.is_absolute()
            else PROJECT_ROOT / path
        )

    candidate = index_rows(
        read_csv(resolve(args.stage2b))
    )

    references = {
        "Stage 2a Symmetric Demand-Capacity":
            index_rows(read_csv(resolve(args.stage2a))),
        "Stage 1 Relative Demand":
            index_rows(read_csv(resolve(args.stage1))),
        "Fixed EG-ALC":
            index_rows(read_csv(resolve(args.fixed))),
        "No Feedback":
            index_rows(read_csv(resolve(args.no_feedback))),
    }

    rows = []

    for reference_name, reference in references.items():
        for count in (1, 4, 8, 16):
            for metric in METRICS:
                for level in ("slice", "patient"):
                    rows.append(
                        compare(
                            reference_name=reference_name,
                            reference=reference,
                            candidate=candidate,
                            count=count,
                            metric=metric,
                            level=level,
                            n_boot=int(args.bootstrap),
                            seed=int(args.seed) + count,
                        )
                    )

    output = resolve(args.output)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output / "comparison_summary.csv",
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    (
        output / "comparison_summary.json"
    ).write_text(
        json.dumps(
            {
                "results": rows,
                "scipy_wilcoxon_available":
                    wilcoxon is not None,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 126)
    print(
        "Stage 2b paired comparison — patient level"
    )
    print("=" * 126)

    for row in rows:
        if (
            row["level"] == "patient"
            and row["metric"] in (
                "ged2",
                "cross_dice",
                "best_sample_dice_to_raters",
            )
        ):
            p = row["wilcoxon_p"]
            p_text = (
                "N/A"
                if p is None
                else f"{p:.3e}"
            )

            print(
                f"{row['reference']:<36} "
                f"N={row['num_samples']:<2d} "
                f"{row['metric']:<28} "
                f"base={row['reference_mean']:.5f} "
                f"stage2b={row['candidate_mean']:.5f} "
                f"improve={row['improvement_mean']:+.5f} "
                f"CI=[{row['improvement_ci95_low']:+.5f},"
                f"{row['improvement_ci95_high']:+.5f}] "
                f"p={p_text}"
            )

    print("=" * 126)
    print(
        f"[DONE] {output / 'comparison_summary.csv'}"
    )
    print(
        f"[DONE] {output / 'comparison_summary.json'}"
    )


if __name__ == "__main__":
    main()
