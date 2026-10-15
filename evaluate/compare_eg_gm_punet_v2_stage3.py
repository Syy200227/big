"""Stage 3 performance paired comparison."""
from __future__ import annotations
import argparse, csv, json, re
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

def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def patient_id(case):
    m = re.search(r"(LIDC-IDRI-\d{4})", str(case))
    return m.group(1) if m else str(case).split("_")[0]

def index_rows(rows):
    return {(str(r["case_id"]), int(r["num_samples"])): r for r in rows}

def bootstrap_ci(values, n, seed):
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        ix = rng.integers(0, len(values), len(values))
        means[i] = values[ix].mean()
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))

def wilcoxon_p(a, b):
    if wilcoxon is None:
        return None
    if np.allclose(a - b, 0):
        return 1.0
    try:
        return float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    except ValueError:
        return None

def paired_values(ref, new, count, metric):
    keys = sorted(k for k in set(ref).intersection(new) if k[1] == count)
    cases = [k[0] for k in keys]
    a = np.asarray([float(ref[k][metric]) for k in keys])
    b = np.asarray([float(new[k][metric]) for k in keys])
    return cases, a, b

def aggregate_patient(cases, a, b):
    groups = {}
    for i, c in enumerate(cases):
        groups.setdefault(patient_id(c), []).append(i)
    return (
        np.asarray([a[ix].mean() for _, ix in sorted(groups.items())]),
        np.asarray([b[ix].mean() for _, ix in sorted(groups.items())]),
    )

def compare_one(ref_name, ref, new, count, metric, level, nboot, seed):
    cases, a, b = paired_values(ref, new, count, metric)
    if level == "patient":
        a, b = aggregate_patient(cases, a, b)
    if METRICS[metric] == "lower":
        improvement = a - b
    elif METRICS[metric] == "higher":
        improvement = b - a
    else:
        improvement = b - a
    lo, hi = bootstrap_ci(improvement, nboot, seed)
    return {
        "reference": ref_name,
        "candidate": "Stage 3 Conditional Vitality",
        "level": level,
        "num_pairs": len(a),
        "num_samples": count,
        "metric": metric,
        "direction": METRICS[metric],
        "reference_mean": float(a.mean()),
        "candidate_mean": float(b.mean()),
        "improvement_mean": float(improvement.mean()),
        "improvement_ci95_low": lo,
        "improvement_ci95_high": hi,
        "wilcoxon_p": wilcoxon_p(b, a),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage3", default="result/eg_gm_punet_v2_conditional_vitality_k5_full/evaluation/test/samples_16/best_val_ged4_eg_gm_punet_v2_stage3/probabilistic_per_case_metrics.csv")
    p.add_argument("--stage2b", default="result/eg_gm_punet_v2_bounded_demand_capacity_k5_full/evaluation/test/samples_16/best_val_ged4_eg_gm_punet_v2_stage2b/probabilistic_per_case_metrics.csv")
    p.add_argument("--stage2a", default="result/eg_gm_punet_v2_demand_capacity_k5_full/evaluation/test/samples_16/best_val_ged4_eg_gm_punet_v2_stage2/probabilistic_per_case_metrics.csv")
    p.add_argument("--stage1", default="result/eg_gm_punet_v2_relative_feedback_k5_full/evaluation/test/samples_16/best_val_ged4_eg_gm_punet_v2_stage1/probabilistic_per_case_metrics.csv")
    p.add_argument("--fixed", default="result/eg_gm_punet_k5_full/evaluation/test/samples_16/best_val_ged4_eg_gm_punet/probabilistic_per_case_metrics.csv")
    p.add_argument("--no-feedback", dest="no_feedback", default="result/eg_gm_punet_no_feedback_k5_full/evaluation/test/samples_16/best_val_ged4_eg_gm_punet/no_feedback_probabilistic_per_case_metrics.csv")
    p.add_argument("--output", default="result/eg_gm_punet_v2_conditional_vitality_k5_full/stage3_comparison")
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    def resolve(v):
        q = Path(v)
        return q if q.is_absolute() else PROJECT_ROOT / q

    new = index_rows(read_csv(resolve(args.stage3)))
    refs = {
        "Stage 2b Bounded Demand-Capacity": index_rows(read_csv(resolve(args.stage2b))),
        "Stage 2a Symmetric Demand-Capacity": index_rows(read_csv(resolve(args.stage2a))),
        "Stage 1 Relative Demand": index_rows(read_csv(resolve(args.stage1))),
        "Fixed EG-ALC": index_rows(read_csv(resolve(args.fixed))),
        "No Feedback": index_rows(read_csv(resolve(args.no_feedback))),
    }
    rows = []
    for ref_name, ref in refs.items():
        for count in (1, 4, 8, 16):
            for metric in METRICS:
                for level in ("slice", "patient"):
                    rows.append(compare_one(ref_name, ref, new, count, metric, level, args.bootstrap, args.seed + count))

    out = resolve(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "comparison_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    (out / "comparison_summary.json").write_text(json.dumps({"results": rows, "scipy_wilcoxon_available": wilcoxon is not None}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 126)
    print("Stage 3 paired comparison — patient level")
    print("=" * 126)
    for r in rows:
        if r["level"] == "patient" and r["metric"] in ("ged2", "cross_dice", "best_sample_dice_to_raters"):
            ptext = "N/A" if r["wilcoxon_p"] is None else f'{r["wilcoxon_p"]:.3e}'
            print(f'{r["reference"]:<36} N={r["num_samples"]:<2d} {r["metric"]:<28} base={r["reference_mean"]:.5f} stage3={r["candidate_mean"]:.5f} improve={r["improvement_mean"]:+.5f} CI=[{r["improvement_ci95_low"]:+.5f},{r["improvement_ci95_high"]:+.5f}] p={ptext}')
    print("=" * 126)

if __name__ == "__main__":
    main()
