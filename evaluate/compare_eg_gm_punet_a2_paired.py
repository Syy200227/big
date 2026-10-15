
"""
Paired comparison for A.2 full-test candidates.

Inputs are probabilistic_per_case_metrics.csv files produced by the same
gmm_prior_evaluation utility, therefore @1/@4/@8/@16 are directly comparable.

Produces:
- paired_comparison_summary.csv
- paired_comparison_summary.json

Statistics:
- slice level
- patient level
- paired bootstrap 95% CI
- paired Wilcoxon when scipy is available
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


METRICS = {
    "ged2": "lower",
    "cross_dice": "higher",
    "best_sample_dice_to_raters": "higher",
    "sample_diversity_distance": "context",
}


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def f(row, key):
    try:
        x = float(row[key])
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def patient_from_case(case_id):
    m = re.search(r"(LIDC-IDRI-\\d{4})", str(case_id))
    return m.group(1) if m else str(case_id).split("_")[0]


def index_rows(rows):
    out = {}
    for row in rows:
        out[(str(row["case_id"]), int(float(row["num_samples"])))] = row
    return out


def paired_slice(ref_rows, cand_rows, count, metric):
    a, b = index_rows(ref_rows), index_rows(cand_rows)
    keys = sorted(
        {k for k in a if k[1] == count}
        & {k for k in b if k[1] == count}
    )
    ref = np.asarray([f(a[k], metric) for k in keys], dtype=float)
    cand = np.asarray([f(b[k], metric) for k in keys], dtype=float)
    good = np.isfinite(ref) & np.isfinite(cand)
    return ref[good], cand[good]


def paired_patient(ref_rows, cand_rows, count, metric):
    a, b = index_rows(ref_rows), index_rows(cand_rows)
    keys = sorted(
        {k for k in a if k[1] == count}
        & {k for k in b if k[1] == count}
    )
    groups = {}
    for k in keys:
        rv, cv = f(a[k], metric), f(b[k], metric)
        if not (math.isfinite(rv) and math.isfinite(cv)):
            continue
        pid = (
            a[k].get("patient_id")
            or b[k].get("patient_id")
            or patient_from_case(k[0])
        )
        groups.setdefault(str(pid), [[], []])
        groups[str(pid)][0].append(rv)
        groups[str(pid)][1].append(cv)
    r, c = [], []
    for _, (rv, cv) in sorted(groups.items()):
        r.append(float(np.mean(rv)))
        c.append(float(np.mean(cv)))
    return np.asarray(r), np.asarray(c)


def improvement(reference, candidate, direction):
    if direction == "lower":
        return reference - candidate
    if direction == "higher":
        return candidate - reference
    return candidate - reference


def bootstrap_ci(values, seed=2026, n_boot=10000):
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = []
    remaining = n_boot
    while remaining > 0:
        n = min(1000, remaining)
        idx = rng.integers(0, len(x), size=(n, len(x)))
        means.append(x[idx].mean(axis=1))
        remaining -= n
    boot = np.concatenate(means)
    return [
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    ]


def stat_row(ref_name, cand_name, level, count, metric, direction, ref, cand):
    imp = improvement(ref, cand, direction)
    p = 1.0
    if wilcoxon is not None and len(imp) > 0:
        if np.allclose(imp, 0):
            p = 1.0
        else:
            try:
                p = float(wilcoxon(imp).pvalue)
            except Exception:
                p = float("nan")
    ci = bootstrap_ci(imp)
    return {
        "reference": ref_name,
        "candidate": cand_name,
        "level": level,
        "num_pairs": int(len(imp)),
        "num_samples": int(count),
        "metric": metric,
        "direction": direction,
        "reference_mean": float(np.mean(ref)) if len(ref) else None,
        "candidate_mean": float(np.mean(cand)) if len(cand) else None,
        "improvement_mean": float(np.mean(imp)) if len(imp) else None,
        "improvement_ci95_low": ci[0],
        "improvement_ci95_high": ci[1],
        "wilcoxon_p": p,
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--candidate-name", default="A2 candidate")
    p.add_argument("--reference", required=True)
    p.add_argument("--reference-name", default="reference")
    p.add_argument("--output", required=True)
    p.add_argument("--sample-counts", nargs="+", type=int, default=[1, 4, 8, 16])
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    def resolve(x):
        q = Path(x)
        return q if q.is_absolute() else root / q

    cand = read_csv(resolve(args.candidate))
    ref = read_csv(resolve(args.reference))

    rows = []
    for count in args.sample_counts:
        for metric, direction in METRICS.items():
            sr, sc = paired_slice(ref, cand, count, metric)
            rows.append(stat_row(
                args.reference_name, args.candidate_name,
                "slice", count, metric, direction, sr, sc
            ))
            pr, pc = paired_patient(ref, cand, count, metric)
            rows.append(stat_row(
                args.reference_name, args.candidate_name,
                "patient", count, metric, direction, pr, pc
            ))

    out = resolve(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "paired_comparison_summary.csv", rows)
    summary = {
        "reference": args.reference_name,
        "candidate": args.candidate_name,
        "results": rows,
        "scipy_wilcoxon_available": wilcoxon is not None,
        "interpretation": {
            "positive_improvement_mean": "candidate is better for GED/CrossDice/BestDice; diversity is contextual",
            "component_count_is_not_success_criterion": True,
        },
    }
    (out / "paired_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
