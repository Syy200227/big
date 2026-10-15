
"""
EG-GM-PUNet A.2 Final Paired Statistical Analysis
==================================================

Publication-oriented paired statistics for:
1) A.2 epoch12 best-GED vs Stage2b        [PRIMARY]
2) A.2 epoch24 minimum-diversity-gap vs Stage2b
3) A.2 epoch12 vs A.2 epoch24

Required metrics:
- GED² @1/4/8/16
- CrossDice @1/4/8/16
- BestDice @1/4/8/16
- Diversity @1/4/8/16

Required statistics:
- slice-level paired bootstrap 95% CI
- patient-level aggregation
- paired Wilcoxon signed-rank test

Additional publication-safe diagnostic:
- Diversity calibration error:
      abs(sample_diversity - annotation_diversity)
  because raw diversity is NOT a monotonic "higher-is-better" metric.

Positive `improvement_mean` convention:
- GED²: lower candidate is better
- CrossDice: higher candidate is better
- BestDice: higher candidate is better
- Diversity calibration error: lower candidate is better
- Raw Diversity: candidate - reference only; no better/worse interpretation

No model inference is performed. This script reads existing
probabilistic_per_case_metrics.csv files only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


SAMPLE_COUNTS = (1, 4, 8, 16)

METRICS = {
    "ged2": {
        "display": "GED²",
        "direction": "lower",
    },
    "cross_dice": {
        "display": "CrossDice",
        "direction": "higher",
    },
    "best_sample_dice_to_raters": {
        "display": "BestDice",
        "direction": "higher",
    },
    "sample_diversity_distance": {
        "display": "Diversity",
        "direction": "context",
    },
    "diversity_calibration_error": {
        "display": "|Diversity-AnnDiv|",
        "direction": "lower",
    },
}


def resolve(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def finite_float(value) -> float:
    try:
        x = float(value)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def patient_from_case(case_id: str) -> str:
    m = re.search(r"(LIDC-IDRI-\d{4})", str(case_id))
    if m:
        return m.group(1)
    # fallback: keep a stable identifier rather than merge unrelated cases
    return str(case_id).split("_")[0]


def add_derived_metrics(row: dict[str, str]) -> dict[str, str | float]:
    out = dict(row)
    d = finite_float(row.get("sample_diversity_distance"))
    a = finite_float(row.get("annotation_diversity_distance"))
    out["diversity_calibration_error"] = (
        abs(d - a)
        if math.isfinite(d) and math.isfinite(a)
        else float("nan")
    )
    return out


def index_rows(rows: list[dict[str, str]]):
    index = {}
    duplicate = []
    for raw in rows:
        row = add_derived_metrics(raw)
        if "case_id" not in row:
            raise KeyError("CSV lacks case_id.")
        if "num_samples" not in row:
            raise KeyError("CSV lacks num_samples.")
        key = (str(row["case_id"]), int(float(row["num_samples"])))
        if key in index:
            duplicate.append(key)
        index[key] = row
    if duplicate:
        raise RuntimeError(
            "Duplicate (case_id,num_samples) rows detected. "
            f"Examples: {duplicate[:5]}"
        )
    return index


def paired_slice(
    reference_rows,
    candidate_rows,
    count: int,
    metric: str,
):
    ref_i = index_rows(reference_rows)
    cand_i = index_rows(candidate_rows)

    keys = sorted(
        {k for k in ref_i if k[1] == count}
        & {k for k in cand_i if k[1] == count}
    )
    if not keys:
        raise RuntimeError(
            f"No paired cases at sample_count={count}."
        )

    ref = np.asarray(
        [finite_float(ref_i[k].get(metric)) for k in keys],
        dtype=np.float64,
    )
    cand = np.asarray(
        [finite_float(cand_i[k].get(metric)) for k in keys],
        dtype=np.float64,
    )

    good = np.isfinite(ref) & np.isfinite(cand)
    keys = [k for k, ok in zip(keys, good) if ok]
    return keys, ref[good], cand[good]


def paired_patient(
    reference_rows,
    candidate_rows,
    count: int,
    metric: str,
):
    keys, ref, cand = paired_slice(
        reference_rows,
        candidate_rows,
        count,
        metric,
    )

    grouped: dict[str, list[list[float]]] = {}
    for key, rv, cv in zip(keys, ref, cand):
        case_id = key[0]

        # Prefer patient_id already saved by evaluator if present.
        # We need row lookup only to read metadata.
        # Fallback is regex LIDC-IDRI-XXXX from case_id.
        pid = patient_from_case(case_id)

        grouped.setdefault(pid, [[], []])
        grouped[pid][0].append(float(rv))
        grouped[pid][1].append(float(cv))

    pids = sorted(grouped)
    ref_p = np.asarray(
        [np.mean(grouped[p][0]) for p in pids],
        dtype=np.float64,
    )
    cand_p = np.asarray(
        [np.mean(grouped[p][1]) for p in pids],
        dtype=np.float64,
    )
    return pids, ref_p, cand_p


def normalized_improvement(
    reference: np.ndarray,
    candidate: np.ndarray,
    direction: str,
) -> np.ndarray:
    if direction == "lower":
        return reference - candidate
    if direction == "higher":
        return candidate - reference
    # Raw diversity is contextual.
    return candidate - reference


def paired_bootstrap_ci(
    values: np.ndarray,
    *,
    seed: int = 2026,
    n_boot: int = 20000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    means = []
    remaining = int(n_boot)

    # Chunk to keep RAM low.
    while remaining > 0:
        current = min(1000, remaining)
        idx = rng.integers(
            0,
            len(values),
            size=(current, len(values)),
        )
        means.append(values[idx].mean(axis=1))
        remaining -= current

    boot = np.concatenate(means)
    return (
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def wilcoxon_pvalue(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return float("nan")
    if np.allclose(diff, 0.0):
        return 1.0
    if wilcoxon is None:
        return float("nan")
    try:
        # zero_method='wilcox' removes exact zero differences.
        return float(
            wilcoxon(
                diff,
                alternative="two-sided",
                zero_method="wilcox",
                correction=False,
                method="auto",
            ).pvalue
        )
    except Exception:
        return float("nan")


def rank_biserial_from_diff(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation.

    Positive means candidate is better under the normalized improvement
    convention, except raw Diversity where it simply means candidate higher.
    """
    d = np.asarray(diff, dtype=np.float64)
    d = d[np.isfinite(d)]
    d = d[np.abs(d) > 0]
    if len(d) == 0:
        return 0.0

    absd = np.abs(d)
    order = np.argsort(absd, kind="mergesort")
    ranks = np.empty(len(d), dtype=np.float64)

    # average ranks for ties
    sorted_abs = absd[order]
    start = 0
    while start < len(d):
        end = start + 1
        while end < len(d) and sorted_abs[end] == sorted_abs[start]:
            end += 1
        avg_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = avg_rank
        start = end

    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    denom = w_pos + w_neg
    return float((w_pos - w_neg) / denom) if denom > 0 else 0.0


def describe_pair(
    *,
    comparison: str,
    reference_name: str,
    candidate_name: str,
    level: str,
    sample_count: int,
    metric: str,
    direction: str,
    reference: np.ndarray,
    candidate: np.ndarray,
    bootstrap_seed: int,
    n_boot: int,
):
    improvement = normalized_improvement(
        reference,
        candidate,
        direction,
    )
    ci_low, ci_high = paired_bootstrap_ci(
        improvement,
        seed=bootstrap_seed,
        n_boot=n_boot,
    )
    p = wilcoxon_pvalue(improvement)

    return {
        "comparison": comparison,
        "reference": reference_name,
        "candidate": candidate_name,
        "level": level,
        "num_pairs": int(len(improvement)),
        "num_samples": int(sample_count),
        "metric": metric,
        "metric_display": METRICS[metric]["display"],
        "direction": direction,
        "reference_mean": float(np.mean(reference)),
        "reference_std": float(np.std(reference, ddof=0)),
        "candidate_mean": float(np.mean(candidate)),
        "candidate_std": float(np.std(candidate, ddof=0)),
        "raw_candidate_minus_reference": float(
            np.mean(candidate - reference)
        ),
        "improvement_mean": float(np.mean(improvement)),
        "improvement_median": float(np.median(improvement)),
        "improvement_ci95_low": ci_low,
        "improvement_ci95_high": ci_high,
        "bootstrap_n": int(n_boot),
        "wilcoxon_p": p,
        "rank_biserial": rank_biserial_from_diff(improvement),
        "ci_excludes_zero": bool(
            math.isfinite(ci_low)
            and math.isfinite(ci_high)
            and (ci_low > 0 or ci_high < 0)
        ),
    }


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values.

    Added as an auxiliary field. Raw Wilcoxon p-values remain available.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    out = np.full(len(p), np.nan, dtype=np.float64)
    good = np.flatnonzero(np.isfinite(p))
    if len(good) == 0:
        return out.tolist()

    order = good[np.argsort(p[good])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        adjusted = (m - rank) * p[idx]
        running = max(running, adjusted)
        out[idx] = min(running, 1.0)
    return out.tolist()


def make_comparison(
    *,
    reference_rows,
    candidate_rows,
    reference_name: str,
    candidate_name: str,
    bootstrap_seed: int,
    n_boot: int,
):
    comparison = f"{candidate_name} vs {reference_name}"
    rows = []

    for count in SAMPLE_COUNTS:
        for metric, spec in METRICS.items():
            _, sr, sc = paired_slice(
                reference_rows,
                candidate_rows,
                count,
                metric,
            )
            rows.append(
                describe_pair(
                    comparison=comparison,
                    reference_name=reference_name,
                    candidate_name=candidate_name,
                    level="slice",
                    sample_count=count,
                    metric=metric,
                    direction=spec["direction"],
                    reference=sr,
                    candidate=sc,
                    bootstrap_seed=bootstrap_seed + count,
                    n_boot=n_boot,
                )
            )

            _, pr, pc = paired_patient(
                reference_rows,
                candidate_rows,
                count,
                metric,
            )
            rows.append(
                describe_pair(
                    comparison=comparison,
                    reference_name=reference_name,
                    candidate_name=candidate_name,
                    level="patient",
                    sample_count=count,
                    metric=metric,
                    direction=spec["direction"],
                    reference=pr,
                    candidate=pc,
                    bootstrap_seed=bootstrap_seed + 1000 + count,
                    n_boot=n_boot,
                )
            )

    # Holm correction within this comparison over all reported tests.
    adjusted = holm_adjust([r["wilcoxon_p"] for r in rows])
    for row, value in zip(rows, adjusted):
        row["wilcoxon_p_holm_all_tests"] = value

    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def markdown_paper_table(rows, comparison, level="patient") -> str:
    subset = [
        r for r in rows
        if r["comparison"] == comparison
        and r["level"] == level
        and r["metric"] in {
            "ged2",
            "cross_dice",
            "best_sample_dice_to_raters",
            "sample_diversity_distance",
        }
    ]

    lines = [
        f"# {comparison} — {level}-level",
        "",
        "| Samples | Metric | Reference | Candidate | Δ / Improvement | 95% CI | Wilcoxon p |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in subset:
        # For raw Diversity report candidate-reference rather than call it improvement.
        delta = (
            r["raw_candidate_minus_reference"]
            if r["metric"] == "sample_diversity_distance"
            else r["improvement_mean"]
        )
        lines.append(
            "| {n} | {m} | {ref:.5f} | {cand:.5f} | {d:+.5f} | [{lo:+.5f}, {hi:+.5f}] | {p:.3e} |".format(
                n=r["num_samples"],
                m=r["metric_display"],
                ref=r["reference_mean"],
                cand=r["candidate_mean"],
                d=delta,
                lo=r["improvement_ci95_low"],
                hi=r["improvement_ci95_high"],
                p=r["wilcoxon_p"],
            )
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--a2-ged",
        required=True,
        help="A.2 epoch12 probabilistic_per_case_metrics.csv",
    )
    parser.add_argument(
        "--stage2b",
        required=True,
        help="Stage2b probabilistic_per_case_metrics.csv",
    )
    parser.add_argument(
        "--a2-gap",
        default=None,
        help="Optional A.2 epoch24 probabilistic_per_case_metrics.csv",
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    a2_ged_path = resolve(root, args.a2_ged)
    stage2b_path = resolve(root, args.stage2b)
    a2_gap_path = (
        resolve(root, args.a2_gap)
        if args.a2_gap
        else None
    )
    output = resolve(root, args.output)
    output.mkdir(parents=True, exist_ok=True)

    a2_ged = read_csv(a2_ged_path)
    stage2b = read_csv(stage2b_path)
    a2_gap = read_csv(a2_gap_path) if a2_gap_path else None

    all_rows = []

    # PRIMARY comparison
    all_rows.extend(
        make_comparison(
            reference_rows=stage2b,
            candidate_rows=a2_ged,
            reference_name="Stage2b",
            candidate_name="A2_epoch12_bestGED",
            bootstrap_seed=args.seed,
            n_boot=args.bootstrap_n,
        )
    )

    if a2_gap is not None:
        all_rows.extend(
            make_comparison(
                reference_rows=stage2b,
                candidate_rows=a2_gap,
                reference_name="Stage2b",
                candidate_name="A2_epoch24_bestDivGap",
                bootstrap_seed=args.seed + 10000,
                n_boot=args.bootstrap_n,
            )
        )
        all_rows.extend(
            make_comparison(
                reference_rows=a2_gap,
                candidate_rows=a2_ged,
                reference_name="A2_epoch24_bestDivGap",
                candidate_name="A2_epoch12_bestGED",
                bootstrap_seed=args.seed + 20000,
                n_boot=args.bootstrap_n,
            )
        )

    write_csv(
        output / "paired_statistics_full.csv",
        all_rows,
    )

    primary = "A2_epoch12_bestGED vs Stage2b"
    primary_rows = [
        r for r in all_rows
        if r["comparison"] == primary
    ]
    write_csv(
        output / "paired_statistics_A2_epoch12_vs_Stage2b.csv",
        primary_rows,
    )

    # compact publication table
    compact = []
    for r in primary_rows:
        if r["metric"] not in {
            "ged2",
            "cross_dice",
            "best_sample_dice_to_raters",
            "sample_diversity_distance",
        }:
            continue
        compact.append({
            "level": r["level"],
            "num_samples": r["num_samples"],
            "metric": r["metric_display"],
            "stage2b_mean": r["reference_mean"],
            "a2_epoch12_mean": r["candidate_mean"],
            "candidate_minus_stage2b": r["raw_candidate_minus_reference"],
            "normalized_improvement": r["improvement_mean"],
            "ci95_low": r["improvement_ci95_low"],
            "ci95_high": r["improvement_ci95_high"],
            "wilcoxon_p": r["wilcoxon_p"],
            "wilcoxon_p_holm": r["wilcoxon_p_holm_all_tests"],
            "rank_biserial": r["rank_biserial"],
        })
    write_csv(
        output / "paper_table_A2_epoch12_vs_Stage2b.csv",
        compact,
    )

    md = []
    md.append(
        markdown_paper_table(
            all_rows,
            primary,
            level="slice",
        )
    )
    md.append("")
    md.append(
        markdown_paper_table(
            all_rows,
            primary,
            level="patient",
        )
    )
    (output / "paper_table_A2_epoch12_vs_Stage2b.md").write_text(
        "\n".join(md),
        encoding="utf-8",
    )

    summary = {
        "primary_comparison": primary,
        "input_files": {
            "a2_epoch12": str(a2_ged_path),
            "stage2b": str(stage2b_path),
            "a2_epoch24": str(a2_gap_path) if a2_gap_path else None,
        },
        "bootstrap": {
            "paired": True,
            "n": int(args.bootstrap_n),
            "confidence_interval": 0.95,
            "slice_level": True,
            "patient_level_after_patient_aggregation": True,
        },
        "wilcoxon": {
            "paired_signed_rank": True,
            "scipy_available": wilcoxon is not None,
            "raw_p_reported": True,
            "holm_adjusted_p_reported": True,
        },
        "patient_aggregation": (
            "Mean all slices belonging to the same LIDC-IDRI patient "
            "for each metric/sample-count, then perform paired statistics "
            "over patients."
        ),
        "positive_improvement_definition": {
            "GED²": "Stage2b - A2; positive means A2 lower/better",
            "CrossDice": "A2 - Stage2b; positive means A2 higher/better",
            "BestDice": "A2 - Stage2b; positive means A2 higher/better",
            "Diversity": "raw A2 - Stage2b; contextual, NOT better/worse",
            "|Diversity-AnnDiv|": "Stage2b error - A2 error; positive means A2 better calibrated",
        },
        "sample_counts": list(SAMPLE_COUNTS),
        "results": all_rows,
    }

    (output / "paired_statistics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 100)
    print("A.2 paired statistics finished")
    print("=" * 100)
    print(f"Primary: {primary}")
    print(f"Output : {output}")
    print(f"Wilcoxon/scipy available: {wilcoxon is not None}")
    print(f"Bootstrap iterations: {args.bootstrap_n}")
    print()
    print((output / "paper_table_A2_epoch12_vs_Stage2b.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
