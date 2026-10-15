"""Stage3 vs Stage2b component-utilization comparison."""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
import numpy as np
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def rankdata(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1
        i = j
    return ranks

def corr(x, y, spearman=False):
    x = np.asarray(x, float); y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y); x = x[valid]; y = y[valid]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    if spearman:
        x, y = rankdata(x), rankdata(y)
    return float(np.corrcoef(x, y)[0, 1])

def bootstrap_ci(v, n, seed):
    rng = np.random.default_rng(seed); means = np.empty(n)
    for i in range(n):
        ix = rng.integers(0, len(v), len(v)); means[i] = v[ix].mean()
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage3", default="result/eg_gm_punet_v2_conditional_vitality_k5_full/evaluation/test/samples_16/best_val_ged4_eg_gm_punet_v2_stage3/stage3_analysis_table.csv")
    p.add_argument("--stage2b", default="result/eg_gm_punet_v2_bounded_demand_capacity_k5_full/evaluation/test/samples_16/best_val_ged4_eg_gm_punet_v2_stage2b/stage2b_analysis_table.csv")
    p.add_argument("--output", default="result/eg_gm_punet_v2_conditional_vitality_k5_full/stage3_component_comparison.json")
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    def resolve(v):
        q = Path(v); return q if q.is_absolute() else PROJECT_ROOT / q

    s3 = {r["case_id"]: r for r in read_csv(resolve(args.stage3))}
    s2 = {r["case_id"]: r for r in read_csv(resolve(args.stage2b))}
    ids = sorted(set(s3).intersection(s2))
    result = {"num_cases": len(ids), "metrics": {}, "stage3": {}}

    for key in ("prior_k_eff", "active_component_count", "sample_diversity_16", "ged2_16", "cross_dice_16"):
        x = np.asarray([float(s2[i][key]) for i in ids])
        y = np.asarray([float(s3[i][key]) for i in ids])
        d = y - x
        lo, hi = bootstrap_ci(d, args.bootstrap, args.seed)
        if wilcoxon is None or np.allclose(d, 0):
            pvalue = None if wilcoxon is None else 1.0
        else:
            pvalue = float(wilcoxon(y, x, zero_method="wilcox").pvalue)
        result["metrics"][key] = {
            "stage2b_mean": float(x.mean()),
            "stage3_mean": float(y.mean()),
            "stage3_minus_stage2b": float(d.mean()),
            "ci95": [lo, hi],
            "wilcoxon_p": pvalue,
        }

    ann = [float(s3[i]["annotation_diversity"]) for i in ids]
    keff = [float(s3[i]["prior_k_eff"]) for i in ids]
    gate = [float(s3[i]["vitality_gate"]) for i in ids]
    active = np.asarray([int(float(s3[i]["active_component_count"])) for i in ids])
    gated = [i for i in ids if float(s3[i]["vitality_gate"]) > 0]
    result["stage3"] = {
        "annotation_vs_k_eff_pearson": corr(ann, keff, False),
        "annotation_vs_k_eff_spearman": corr(ann, keff, True),
        "annotation_vs_gate_spearman": corr(ann, gate, True),
        "gated_fraction": len(gated) / max(len(ids), 1),
        "active_count_histogram": {str(k): int((active == k).sum()) for k in range(1, 6)},
        "gated_mean_k_eff": float(np.mean([float(s3[i]["prior_k_eff"]) for i in gated])) if gated else None,
        "gated_mean_active_k": float(np.mean([float(s3[i]["active_component_count"]) for i in gated])) if gated else None,
    }
    out = resolve(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
