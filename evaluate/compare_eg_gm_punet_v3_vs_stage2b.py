"""
Compare EG-GM-PUNet v3 Stage A against Stage 2b.

Primary scientific comparison:
    Stage2b shared-k bounded demand-capacity
        vs
    v3 hierarchical scale-decoupled GMM + continuous bounded EG-ALC

Metrics:
    GED² / CrossDice / BestDice / SampleDiversity @ 1,4,8,16
Levels:
    slice + patient
Statistics:
    paired bootstrap 95% CI + paired Wilcoxon (if scipy available)

这个比较的重点不是“active component 数”，而是：
1. 是否维持或改善 Stage2b 的 fidelity-diversity balance；
2. 去掉 shared semantic k 后是否仍能获得有效 probabilistic coverage。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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
    with open(path,"r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def f(row,key):
    try:
        x=float(row[key])
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def patient_from_case(case_id):
    text=str(case_id)
    marker="LIDC-IDRI-"
    pos=text.find(marker)
    if pos>=0:
        return text[pos:pos+14]
    return text.split("_")[0]


def index_rows(rows):
    out={}
    for row in rows:
        case=str(row["case_id"])
        count=int(float(row["num_samples"]))
        out[(case,count)]=row
    return out


def paired_slice(ref_rows,cand_rows,count,metric):
    a=index_rows(ref_rows); b=index_rows(cand_rows)
    keys=sorted(
        set(k for k in a if k[1]==count)
        & set(k for k in b if k[1]==count)
    )
    ref=np.asarray([f(a[k],metric) for k in keys],dtype=float)
    cand=np.asarray([f(b[k],metric) for k in keys],dtype=float)
    good=np.isfinite(ref)&np.isfinite(cand)
    return ref[good],cand[good]


def paired_patient(ref_rows,cand_rows,count,metric):
    ref,cand=paired_slice(ref_rows,cand_rows,count,metric)
    # Need case ids again to group; build directly from indexes.
    a=index_rows(ref_rows); b=index_rows(cand_rows)
    keys=sorted(
        set(k for k in a if k[1]==count)
        & set(k for k in b if k[1]==count)
    )
    groups={}
    for k in keys:
        rv=f(a[k],metric); cv=f(b[k],metric)
        if not (math.isfinite(rv) and math.isfinite(cv)):
            continue
        pid=(
            a[k].get("patient_id")
            or b[k].get("patient_id")
            or patient_from_case(k[0])
        )
        groups.setdefault(str(pid),[[],[]])
        groups[str(pid)][0].append(rv)
        groups[str(pid)][1].append(cv)
    r=[]; c=[]
    for pid,(rv,cv) in sorted(groups.items()):
        r.append(float(np.mean(rv)))
        c.append(float(np.mean(cv)))
    return np.asarray(r),np.asarray(c)


def improvement(reference,candidate,direction):
    if direction=="lower":
        return reference-candidate
    if direction=="higher":
        return candidate-reference
    return candidate-reference


def bootstrap_ci(values,seed=2026,n_boot=10000):
    x=np.asarray(values,dtype=float)
    if len(x)==0:
        return [None,None]
    rng=np.random.default_rng(seed)
    # chunk to avoid huge temporary memory
    means=[]
    remaining=n_boot
    while remaining>0:
        n=min(1000,remaining)
        idx=rng.integers(0,len(x),size=(n,len(x)))
        means.append(x[idx].mean(axis=1))
        remaining-=n
    boot=np.concatenate(means)
    return [
        float(np.quantile(boot,0.025)),
        float(np.quantile(boot,0.975)),
    ]


def stat_row(
    ref_name,cand_name,level,count,metric,direction,ref,cand
):
    imp=improvement(ref,cand,direction)
    p=1.0
    if wilcoxon is not None and len(imp)>0:
        if np.allclose(imp,0):
            p=1.0
        else:
            try:
                p=float(wilcoxon(imp).pvalue)
            except Exception:
                p=float("nan")
    ci=bootstrap_ci(imp)
    return {
        "reference":ref_name,
        "candidate":cand_name,
        "level":level,
        "num_pairs":int(len(imp)),
        "num_samples":int(count),
        "metric":metric,
        "direction":direction,
        "reference_mean":float(np.mean(ref)) if len(ref) else None,
        "candidate_mean":float(np.mean(cand)) if len(cand) else None,
        "improvement_mean":float(np.mean(imp)) if len(imp) else None,
        "improvement_ci95_low":ci[0],
        "improvement_ci95_high":ci[1],
        "wilcoxon_p":p,
    }


def write_csv(path,rows):
    if not rows:
        return
    with open(path,"w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument(
        "--candidate",
        default=(
            "result/eg_gm_punet_v3_hierarchical_ms_gmm_k5_full/"
            "evaluation/test/samples_16/"
            "best_val_ged4_eg_gm_punet_v3/"
            "probabilistic_per_case_metrics.csv"
        ),
    )
    p.add_argument(
        "--stage2b",
        default=(
            "result/eg_gm_punet_v2_bounded_demand_capacity_k5_full/"
            "evaluation/test/samples_16/"
            "best_val_ged4_eg_gm_punet_v2_stage2b/"
            "probabilistic_per_case_metrics.csv"
        ),
    )
    p.add_argument(
        "--output",
        default=(
            "result/eg_gm_punet_v3_hierarchical_ms_gmm_k5_full/"
            "comparison_vs_stage2b"
        ),
    )
    p.add_argument("--sample-counts",nargs="+",type=int,default=[1,4,8,16])
    args=p.parse_args()

    root=Path(__file__).resolve().parents[1]
    def resolve(x):
        q=Path(x)
        return q if q.is_absolute() else root/q

    candidate=read_csv(resolve(args.candidate))
    reference=read_csv(resolve(args.stage2b))

    rows=[]
    for count in args.sample_counts:
        for metric,direction in METRICS.items():
            sr,sc=paired_slice(reference,candidate,count,metric)
            rows.append(stat_row(
                "Stage 2b Bounded Demand-Capacity",
                "v3 Hierarchical Scale-Decoupled MS-GMM",
                "slice",count,metric,direction,sr,sc
            ))
            pr,pc=paired_patient(reference,candidate,count,metric)
            rows.append(stat_row(
                "Stage 2b Bounded Demand-Capacity",
                "v3 Hierarchical Scale-Decoupled MS-GMM",
                "patient",count,metric,direction,pr,pc
            ))

    out=resolve(args.output)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"comparison_summary.csv",rows)
    summary={
        "candidate":"v3 Hierarchical Scale-Decoupled MS-GMM",
        "reference":"Stage 2b Bounded Demand-Capacity",
        "results":rows,
        "scipy_wilcoxon_available":wilcoxon is not None,
        "interpretation":{
            "primary_goal":(
                "Remove shared semantic k without losing Stage2b "
                "fidelity-diversity balance."
            ),
            "component_count_is_not_success_criterion":True,
        },
    }
    (out/"comparison_summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
