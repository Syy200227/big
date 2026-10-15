"""
Stage 2 与 Stage 1 / Fixed EG-ALC / No-Feedback 的 paired comparison。

核心问题：
1. Stage 2 是否恢复 Stage 1 丢失的 fidelity？
2. Stage 2 是否保留/改善 diversity？
3. Stage 2 是否在 GED²@1/4/8/16 上优于 Stage 1，并尽量超过 Fixed EG-ALC？

默认读取各自 probabilistic_per_case_metrics.csv。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

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
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def patient_id(case_id: str):
    m = re.search(r"(LIDC-IDRI-\d{4})", str(case_id))
    return m.group(1) if m else str(case_id).split("_")[0]


def index_rows(rows):
    return {
        (str(r["case_id"]), int(r["num_samples"])): r
        for r in rows
    }


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(values), len(values))
        means[i] = values[idx].mean()
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def wtest(a, b):
    if wilcoxon is None:
        return None
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(a, b, zero_method="wilcox").pvalue)


def paired(index_ref, index_new, count, metric):
    keys = sorted(
        k for k in set(index_ref).intersection(index_new)
        if k[1] == count
    )
    cases = [k[0] for k in keys]
    ref = np.asarray([float(index_ref[k][metric]) for k in keys])
    new = np.asarray([float(index_new[k][metric]) for k in keys])
    return cases, ref, new


def aggregate_patient(cases, ref, new):
    groups = {}
    for i, c in enumerate(cases):
        groups.setdefault(patient_id(c), []).append(i)
    r, n = [], []
    for _, idx in sorted(groups.items()):
        r.append(ref[idx].mean())
        n.append(new[idx].mean())
    return np.asarray(r), np.asarray(n)


def compare(
    ref_name,
    ref_index,
    new_index,
    count,
    metric,
    level,
    n_boot,
    seed,
):
    cases, ref, new = paired(ref_index, new_index, count, metric)
    if level == "patient":
        ref, new = aggregate_patient(cases, ref, new)

    if METRICS[metric] == "lower":
        improve = ref - new
    elif METRICS[metric] == "higher":
        improve = new - ref
    else:
        improve = new - ref

    lo, hi = bootstrap_ci(improve, n_boot, seed)
    return {
        "reference": ref_name,
        "candidate": "Stage 2 Demand-Capacity",
        "level": level,
        "num_pairs": len(ref),
        "num_samples": count,
        "metric": metric,
        "direction": METRICS[metric],
        "reference_mean": float(ref.mean()),
        "candidate_mean": float(new.mean()),
        "improvement_mean": float(improve.mean()),
        "improvement_ci95_low": lo,
        "improvement_ci95_high": hi,
        "wilcoxon_p": wtest(new, ref),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage2",
        default=(
            "result/eg_gm_punet_v2_demand_capacity_k5_full/evaluation/test/"
            "samples_16/best_val_ged4_eg_gm_punet_v2_stage2/"
            "probabilistic_per_case_metrics.csv"
        ),
    )
    parser.add_argument(
        "--stage1",
        default=(
            "result/eg_gm_punet_v2_relative_feedback_k5_full/evaluation/test/"
            "samples_16/best_val_ged4_eg_gm_punet_v2_stage1/"
            "probabilistic_per_case_metrics.csv"
        ),
    )
    parser.add_argument(
        "--fixed",
        default=(
            "result/eg_gm_punet_k5_full/evaluation/test/"
            "samples_16/best_val_ged4_eg_gm_punet/"
            "probabilistic_per_case_metrics.csv"
        ),
    )
    parser.add_argument(
        "--no-feedback",
        dest="no_feedback",
        default=(
            "result/eg_gm_punet_no_feedback_k5_full/evaluation/test/"
            "samples_16/best_val_ged4_eg_gm_punet/"
            "no_feedback_probabilistic_per_case_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default="result/eg_gm_punet_v2_demand_capacity_k5_full/stage2_comparison",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    def resolve(v):
        p = Path(v)
        return p if p.is_absolute() else PROJECT_ROOT / p

    stage2 = index_rows(read_csv(resolve(args.stage2)))
    refs = {
        "Stage 1 Relative Demand": index_rows(read_csv(resolve(args.stage1))),
        "Fixed EG-ALC": index_rows(read_csv(resolve(args.fixed))),
        "No Feedback": index_rows(read_csv(resolve(args.no_feedback))),
    }

    rows = []
    for ref_name, ref in refs.items():
        for count in (1, 4, 8, 16):
            for metric in METRICS:
                for level in ("slice", "patient"):
                    rows.append(
                        compare(
                            ref_name, ref, stage2,
                            count, metric, level,
                            args.bootstrap,
                            args.seed + count,
                        )
                    )

    out = resolve(args.output)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "comparison_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (out / "comparison_summary.json").write_text(
        json.dumps(
            {
                "results": rows,
                "scipy_wilcoxon_available": wilcoxon is not None,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 122)
    print("Stage 2 paired comparison — patient level")
    print("=" * 122)
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
            p_text = "N/A" if p is None else f"{p:.3e}"
            print(
                f"{row['reference']:<24} "
                f"N={row['num_samples']:<2d} "
                f"{row['metric']:<28} "
                f"base={row['reference_mean']:.5f} "
                f"stage2={row['candidate_mean']:.5f} "
                f"improve={row['improvement_mean']:+.5f} "
                f"CI=[{row['improvement_ci95_low']:+.5f},"
                f"{row['improvement_ci95_high']:+.5f}] "
                f"p={p_text}"
            )
    print("=" * 122)


if __name__ == "__main__":
    main()
