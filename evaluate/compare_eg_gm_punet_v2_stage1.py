"""
Stage 1 与 Fixed EG-ALC / No-Feedback 的公平配对统计比较。

前提：
三组模型都已经使用同一 test split、同一 validation_seed、同一 prior sampling
设置（16 samples，统计 [1,4,8,16]）完成评估，并生成
probabilistic_per_case_metrics.csv。

默认比较：
1. Stage 1 Relative Ambiguity-Demand
2. 第一版 Full / Fixed EG-ALC
3. No-Feedback

输出：
- slice-level paired statistics；
- patient-level paired statistics；
- Wilcoxon signed-rank p-value；
- paired bootstrap 95% CI；
- comparison_summary.csv；
- comparison_summary.json。

放置位置建议：
    E:/workspace/EG-GM-PUNet/evaluate/compare_eg_gm_punet_v2_stage1.py

运行：
    python evaluate/compare_eg_gm_punet_v2_stage1.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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
    "annotation_diversity_distance": "context",
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"评估 CSV 不存在：{path}")
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def patient_id(case_id: str) -> str:
    match = re.search(r"(LIDC-IDRI-\d{4})", str(case_id))
    return match.group(1) if match else str(case_id).split("_")[0]


def index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    output = {}
    for row in rows:
        key = (str(row["case_id"]), int(row["num_samples"]))
        if key in output:
            raise ValueError(f"重复的 case/sample key：{key}")
        output[key] = row
    return output


def paired_bootstrap_ci(
    differences: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(differences)
    if n == 0:
        return float("nan"), float("nan")

    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        means[i] = differences[indices].mean()

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def safe_wilcoxon(first: np.ndarray, second: np.ndarray) -> float | None:
    if wilcoxon is None:
        return None
    differences = first - second
    if np.allclose(differences, 0.0):
        return 1.0
    try:
        return float(
            wilcoxon(
                first,
                second,
                alternative="two-sided",
                zero_method="wilcox",
            ).pvalue
        )
    except ValueError:
        return None


def paired_values(
    reference: dict[tuple[str, int], dict[str, Any]],
    candidate: dict[tuple[str, int], dict[str, Any]],
    sample_count: int,
    metric: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    common = sorted(
        key
        for key in set(reference).intersection(candidate)
        if key[1] == sample_count
    )
    case_ids = [key[0] for key in common]
    ref = np.asarray(
        [float(reference[key][metric]) for key in common],
        dtype=np.float64,
    )
    cand = np.asarray(
        [float(candidate[key][metric]) for key in common],
        dtype=np.float64,
    )
    return case_ids, ref, cand


def aggregate_by_patient(
    case_ids: list[str],
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for i, case_id in enumerate(case_ids):
        grouped.setdefault(patient_id(case_id), []).append(i)

    ref_values, cand_values = [], []
    for _, indices in sorted(grouped.items()):
        ref_values.append(float(reference[indices].mean()))
        cand_values.append(float(candidate[indices].mean()))
    return (
        np.asarray(ref_values, dtype=np.float64),
        np.asarray(cand_values, dtype=np.float64),
    )


def compare_one(
    *,
    reference_name: str,
    candidate_name: str,
    reference_index: dict[tuple[str, int], dict[str, Any]],
    candidate_index: dict[tuple[str, int], dict[str, Any]],
    sample_count: int,
    metric: str,
    level: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    case_ids, reference, candidate = paired_values(
        reference_index,
        candidate_index,
        sample_count,
        metric,
    )

    if level == "patient":
        reference, candidate = aggregate_by_patient(
            case_ids,
            reference,
            candidate,
        )

    # delta 始终定义为 candidate - reference。
    differences = candidate - reference

    if METRICS[metric] == "lower":
        improvement = reference - candidate
    elif METRICS[metric] == "higher":
        improvement = candidate - reference
    else:
        improvement = differences

    ci_low, ci_high = paired_bootstrap_ci(
        improvement,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    return {
        "reference": reference_name,
        "candidate": candidate_name,
        "level": level,
        "num_pairs": int(len(reference)),
        "num_samples": int(sample_count),
        "metric": metric,
        "direction": METRICS[metric],
        "reference_mean": float(reference.mean()),
        "candidate_mean": float(candidate.mean()),
        "candidate_minus_reference": float(differences.mean()),
        "improvement_mean": float(improvement.mean()),
        "improvement_ci95_low": ci_low,
        "improvement_ci95_high": ci_high,
        "wilcoxon_p": safe_wilcoxon(candidate, reference),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Stage 1 with Fixed EG-ALC and No-Feedback"
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
        default="result/eg_gm_punet_v2_relative_feedback_k5_full/stage1_comparison",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    stage1_path = resolve(args.stage1)
    fixed_path = resolve(args.fixed)
    no_feedback_path = resolve(args.no_feedback)
    output_dir = resolve(args.output)

    stage1 = index_rows(read_csv(stage1_path))
    fixed = index_rows(read_csv(fixed_path))
    no_feedback = index_rows(read_csv(no_feedback_path))

    comparisons = [
        ("Fixed EG-ALC", fixed),
        ("No Feedback", no_feedback),
    ]

    results: list[dict[str, Any]] = []

    for reference_name, reference_index in comparisons:
        for sample_count in (1, 4, 8, 16):
            for metric in METRICS:
                for level in ("slice", "patient"):
                    results.append(
                        compare_one(
                            reference_name=reference_name,
                            candidate_name="Stage 1 Relative Demand",
                            reference_index=reference_index,
                            candidate_index=stage1,
                            sample_count=sample_count,
                            metric=metric,
                            level=level,
                            n_bootstrap=int(args.bootstrap),
                            seed=int(args.seed) + sample_count,
                        )
                    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(results, output_dir / "comparison_summary.csv")

    summary = {
        "stage1_csv": str(stage1_path),
        "fixed_csv": str(fixed_path),
        "no_feedback_csv": str(no_feedback_path),
        "bootstrap_iterations": int(args.bootstrap),
        "results": results,
        "scipy_wilcoxon_available": wilcoxon is not None,
    }
    with open(
        output_dir / "comparison_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("=" * 116)
    print("Stage 1 paired comparison — patient level")
    print("=" * 116)
    print(
        f"{'Reference':<18} {'N':>3} {'Metric':<18} "
        f"{'Base':>10} {'Stage1':>10} {'Improve':>10} {'95% CI':>23} {'p':>11}"
    )
    print("-" * 116)

    for row in results:
        if row["level"] != "patient":
            continue
        if int(row["num_samples"]) not in (1, 16):
            continue
        if row["metric"] not in (
            "ged2",
            "cross_dice",
            "best_sample_dice_to_raters",
        ):
            continue

        p = row["wilcoxon_p"]
        p_text = "N/A" if p is None else f"{p:.3e}"
        ci_text = (
            f"[{row['improvement_ci95_low']:.5f},"
            f"{row['improvement_ci95_high']:.5f}]"
        )
        print(
            f"{row['reference']:<18} "
            f"{row['num_samples']:>3d} "
            f"{row['metric']:<18} "
            f"{row['reference_mean']:>10.5f} "
            f"{row['candidate_mean']:>10.5f} "
            f"{row['improvement_mean']:>10.5f} "
            f"{ci_text:>23} "
            f"{p_text:>11}"
        )

    print("=" * 116)
    print(f"[DONE] {output_dir / 'comparison_summary.csv'}")
    print(f"[DONE] {output_dir / 'comparison_summary.json'}")
    if wilcoxon is None:
        print("[WARN] 当前环境没有 scipy，因此 Wilcoxon p-value 为 N/A。")
        print("       可安装：pip install scipy")
    print("=" * 116)


if __name__ == "__main__":
    main()
