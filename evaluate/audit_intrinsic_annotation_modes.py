"""
Stage 3.5 — Intrinsic Annotation Mode Audit
==========================================

目的
----
在继续修改 categorical GMM 之前，先回答：

    “当前 LIDC subset 的 4 位医生标注，本身到底需要多少个离散原型来解释？”

这一步不训练模型，不改变任何参数。

为什么要先做它
--------------
4 位 annotators != 4 个 intrinsic modes。
Probabilistic U-Net / PHiSeg 等经典 ambiguous segmentation 工作均把多标注视为
p(y|x) 的有限观测，而不是把每位 annotator 当成一个离散类别。因此不能因为 K=5
就要求模型一定激活 5 个 component。

本脚本使用精确 k-medoids（R=4 时直接枚举所有 medoid 子集），以
    d(mask_a, mask_b) = 1 - IoU(mask_a, mask_b)
作为形状距离。

对每个病例计算：
    E_1, E_2, E_3, E_4
其中 E_M 是用 M 个真实 annotation medoids 解释 4 个 rater masks 时的最小平均距离。

同时保存：
    explained_M = 1 - E_M / E_1
    M80 / M90 / M95 = 分别解释 80/90/95% annotation variation 所需的最小原型数

注意：
    M80/M90/M95 是 sensitivity diagnostics，不应被单独解释成“真实模式数真值”。

如果 case artifact 中还保存了：
    generated_samples
    component_indices
    component_probabilities
则进一步计算：
1. 每个已采样 categorical component 的 empirical sample medoid；
2. 2 个/多个离散 component prototype 对真实 4 个 annotations 的覆盖误差；
3. 全部 16 个 samples 对 annotations 的覆盖误差；
4. 二者差异，用于判断连续 z 是否已经承担了主要剩余 variability。

输出
----
intrinsic_annotation_mode_per_case.csv
intrinsic_annotation_mode_per_patient.csv
intrinsic_annotation_mode_quartiles.csv
intrinsic_annotation_mode_summary.json
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.stats import spearmanr, pearsonr
except Exception:
    spearmanr = None
    pearsonr = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-8


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def patient_id_from_case(case_id: str) -> str:
    match = re.search(r"(LIDC-IDRI-\d{4})", str(case_id))
    if match:
        return match.group(1)
    return str(case_id).split("_")[0]


def resolve_artifact_path(index_path: Path, row: dict[str, str]) -> Path:
    """
    优先使用 index 内绝对路径。
    如果结果目录移动过，则 fallback 到：
        <index_dir>/case_artifacts/<case_id>.npz
    """
    raw = str(row.get("artifact_path", "")).strip()
    if raw:
        p = Path(raw)
        if p.exists():
            return p

    case_id = str(row["case_id"])
    fallback = index_path.parent / "case_artifacts" / f"{case_id}.npz"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"找不到 case artifact：{case_id}\n"
        f"index path={raw}\n"
        f"fallback={fallback}"
    )


def as_binary_masks(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value)
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 3:
        raise ValueError(f"rater_masks 必须为 [R,H,W]，当前 {x.shape}")
    return (x > 0.5).astype(np.uint8)


def iou_distance(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)

    inter = int(np.logical_and(aa, bb).sum())
    union = int(np.logical_or(aa, bb).sum())

    if union == 0:
        return 0.0
    return 1.0 - (inter / union)


def cross_distance_matrix(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    output = np.zeros((len(left), len(right)), dtype=np.float64)
    for i in range(len(left)):
        for j in range(len(right)):
            output[i, j] = iou_distance(left[i], right[j])
    return output


def pairwise_distance_matrix(masks: np.ndarray) -> np.ndarray:
    n = int(masks.shape[0])
    output = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = iou_distance(masks[i], masks[j])
            output[i, j] = d
            output[j, i] = d
    return output


def mean_upper_triangle(distance: np.ndarray) -> float:
    n = distance.shape[0]
    if n < 2:
        return 0.0
    values = [
        distance[i, j]
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return float(np.mean(values)) if values else 0.0


def exact_medoid_solution(
    distance: np.ndarray,
    num_medoids: int,
) -> tuple[float, tuple[int, ...], np.ndarray]:
    """
    R=4，因此直接枚举全部 medoid combinations，得到精确最优解。
    """
    n = int(distance.shape[0])

    if num_medoids < 1 or num_medoids > n:
        raise ValueError("num_medoids 必须在 [1,R]。")

    best_error = float("inf")
    best_medoids: tuple[int, ...] | None = None
    best_assignment: np.ndarray | None = None

    for medoids in itertools.combinations(range(n), num_medoids):
        medoid_array = np.asarray(medoids, dtype=int)
        local = distance[:, medoid_array]
        nearest = np.argmin(local, axis=1)
        min_distances = local[np.arange(n), nearest]
        error = float(np.mean(min_distances))

        if error < best_error - 1e-12:
            best_error = error
            best_medoids = tuple(int(v) for v in medoids)
            best_assignment = medoid_array[nearest]

    assert best_medoids is not None
    assert best_assignment is not None
    return best_error, best_medoids, best_assignment


def explained_fraction(e1: float, em: float) -> float:
    if e1 <= EPS:
        return 1.0
    return float(np.clip(1.0 - em / e1, 0.0, 1.0))


def minimum_m_for_explained(
    errors: dict[int, float],
    threshold: float,
) -> int:
    e1 = float(errors[1])
    if e1 <= EPS:
        return 1

    for m in sorted(errors):
        if explained_fraction(e1, errors[m]) >= threshold:
            return int(m)
    return max(errors)


def empirical_component_prototypes(
    samples: np.ndarray,
    indices: np.ndarray,
) -> tuple[list[int], np.ndarray, dict[int, int]]:
    """
    对每个实际被采样到的 categorical component，从属于该 component 的生成样本中
    选一个 medoid sample 作为 empirical component prototype。

    这不是严格的 z=mu component mean segmentation，而是无需改模型即可得到的
    “该离散 component 在当前 16 个 prior samples 中的代表性输出”。
    """
    samples = as_binary_masks(samples)
    indices = np.asarray(indices).reshape(-1).astype(int)

    if len(samples) != len(indices):
        raise ValueError("generated_samples 与 component_indices 数量不一致。")

    component_ids: list[int] = []
    prototypes: list[np.ndarray] = []
    counts: dict[int, int] = {}

    for component in sorted(set(indices.tolist())):
        subset = samples[indices == component]
        if len(subset) == 0:
            continue

        d = pairwise_distance_matrix(subset)
        medoid = int(np.argmin(d.mean(axis=1)))

        component_ids.append(int(component))
        prototypes.append(subset[medoid])
        counts[int(component)] = int(len(subset))

    if not prototypes:
        return [], np.empty((0,) + samples.shape[1:], dtype=np.uint8), {}

    return component_ids, np.stack(prototypes, axis=0), counts


def mean_annotation_coverage(
    annotations: np.ndarray,
    prototypes: np.ndarray,
) -> float:
    """
    每张真实 annotation 到最近 prototype 的平均 IoU distance。
    越低越好。
    """
    if len(prototypes) == 0:
        return float("nan")
    d = cross_distance_matrix(annotations, prototypes)
    return float(np.min(d, axis=1).mean())


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def corr(x: np.ndarray, y: np.ndarray, kind: str) -> float | None:
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]

    if len(x) < 3 or np.std(x) <= EPS or np.std(y) <= EPS:
        return None

    if kind == "spearman" and spearmanr is not None:
        return float(spearmanr(x, y).statistic)
    if kind == "pearson" and pearsonr is not None:
        return float(pearsonr(x, y).statistic)

    # scipy 不可用时：
    if kind == "pearson":
        return float(np.corrcoef(x, y)[0, 1])

    # 简单 average-rank fallback
    def rankdata(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty(len(v), dtype=float)
        ranks[order] = np.arange(len(v), dtype=float)

        unique, inverse, counts = np.unique(v, return_inverse=True, return_counts=True)
        for idx, count in enumerate(counts):
            if count > 1:
                loc = np.where(inverse == idx)[0]
                ranks[loc] = ranks[loc].mean()
        return ranks

    return float(
        np.corrcoef(
            rankdata(x),
            rankdata(y),
        )[0, 1]
    )


def mean_dict(rows: list[dict[str, Any]], columns: list[str]) -> dict[str, float]:
    output: dict[str, float] = {}
    for name in columns:
        values = np.asarray(
            [safe_float(row.get(name)) for row in rows],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        output[name] = float(values.mean()) if len(values) else float("nan")
    return output


def patient_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["patient_id"]), []).append(row)

    numeric = [
        "annotation_diversity",
        "positive_rater_count",
        "presence_disagreement",
        "E1",
        "E2",
        "E3",
        "E4",
        "explained_2",
        "explained_3",
        "M80",
        "M90",
        "M95",
        "delta_2_over_E1",
        "delta_3_over_E1",
        "delta_4_over_E1",
        "component_prototype_coverage",
        "all_sample_coverage",
        "component_coverage_gap_vs_oracle_E2",
        "sample_coverage_gain_over_components",
        "probability_active_component_count",
        "sampled_component_count",
    ]

    output: list[dict[str, Any]] = []
    for patient, items in sorted(groups.items()):
        row: dict[str, Any] = {
            "patient_id": patient,
            "num_slices": len(items),
        }
        row.update(mean_dict(items, numeric))
        output.append(row)
    return output


def quartile_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = np.asarray(
        [safe_float(row["annotation_diversity"]) for row in rows],
        dtype=np.float64,
    )

    boundaries = np.quantile(values, [0.25, 0.50, 0.75])

    output = []
    for q in range(4):
        if q == 0:
            indices = np.where(values <= boundaries[0])[0]
        elif q == 1:
            indices = np.where(
                (values > boundaries[0]) & (values <= boundaries[1])
            )[0]
        elif q == 2:
            indices = np.where(
                (values > boundaries[1]) & (values <= boundaries[2])
            )[0]
        else:
            indices = np.where(values > boundaries[2])[0]

        subset = [rows[int(i)] for i in indices]

        row: dict[str, Any] = {
            "quartile": f"Q{q + 1}",
            "num_cases": len(subset),
        }
        row.update(
            mean_dict(
                subset,
                [
                    "annotation_diversity",
                    "presence_disagreement",
                    "E1",
                    "E2",
                    "E3",
                    "explained_2",
                    "explained_3",
                    "M80",
                    "M90",
                    "M95",
                    "component_prototype_coverage",
                    "all_sample_coverage",
                    "probability_active_component_count",
                    "sampled_component_count",
                ],
            )
        )

        for m in (1, 2, 3, 4):
            row[f"fraction_M90_eq_{m}"] = float(
                np.mean([int(item["M90"]) == m for item in subset])
            )

        output.append(row)

    return output


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arrays = {
        key: np.asarray(
            [safe_float(row.get(key)) for row in rows],
            dtype=np.float64,
        )
        for key in [
            "annotation_diversity",
            "E1",
            "E2",
            "E3",
            "E4",
            "explained_2",
            "explained_3",
            "M80",
            "M90",
            "M95",
            "presence_disagreement",
            "component_prototype_coverage",
            "all_sample_coverage",
            "component_coverage_gap_vs_oracle_E2",
            "sample_coverage_gain_over_components",
            "probability_active_component_count",
            "sampled_component_count",
        ]
    }

    summary: dict[str, Any] = {
        "stage": "Stage 3.5 Intrinsic Annotation Mode Audit",
        "num_cases": len(rows),
        "num_patients": len(set(row["patient_id"] for row in rows)),
        "distance": "1-IoU",
        "rater_count": int(round(np.mean([
            int(row["num_raters"]) for row in rows
        ]))),
        "means": {},
        "effective_prototype_count_sensitivity": {},
        "presence_disagreement": {},
        "correlations": {},
        "interpretation_guardrail": (
            "M80/M90/M95 are sensitivity diagnostics, not ground-truth mode labels."
        ),
    }

    for key, values in arrays.items():
        good = values[np.isfinite(values)]
        if len(good):
            summary["means"][key] = float(good.mean())

    for threshold_name in ("M80", "M90", "M95"):
        values = arrays[threshold_name]
        values = values[np.isfinite(values)].astype(int)

        hist = {
            str(m): int(np.sum(values == m))
            for m in (1, 2, 3, 4)
        }
        frac_le2 = float(np.mean(values <= 2)) if len(values) else None
        frac_ge3 = float(np.mean(values >= 3)) if len(values) else None

        summary["effective_prototype_count_sensitivity"][threshold_name] = {
            "histogram": hist,
            "fraction_le_2": frac_le2,
            "fraction_ge_3": frac_ge3,
        }

    presence = arrays["presence_disagreement"]
    presence = presence[np.isfinite(presence)]
    summary["presence_disagreement"] = {
        "fraction_cases": float(presence.mean()) if len(presence) else None,
    }

    x = arrays["annotation_diversity"]
    for y_name in (
        "explained_2",
        "M90",
        "component_prototype_coverage",
        "sample_coverage_gain_over_components",
        "probability_active_component_count",
    ):
        y = arrays[y_name]
        summary["correlations"][f"annotation_diversity_vs_{y_name}"] = {
            "pearson": corr(x, y, "pearson"),
            "spearman": corr(x, y, "spearman"),
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--artifact-index",
        default=(
            "result/eg_gm_punet_v2_conditional_vitality_k5_full/"
            "evaluation/test/samples_16/"
            "best_val_ged4_eg_gm_punet_v2_stage3/"
            "case_artifact_index.csv"
        ),
        help="Stage3/Stage2b evaluator 保存的 case_artifact_index.csv",
    )
    parser.add_argument(
        "--output",
        default=(
            "result/eg_gm_punet_v2_conditional_vitality_k5_full/"
            "stage35_intrinsic_annotation_mode_audit"
        ),
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="仅用于 smoke/debug。",
    )

    args = parser.parse_args()

    def resolve(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p

    index_path = resolve(args.artifact_index)
    output_dir = resolve(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_rows = read_csv(index_path)
    if args.max_cases is not None:
        index_rows = index_rows[: int(args.max_cases)]

    rows: list[dict[str, Any]] = []

    for case_number, index_row in enumerate(index_rows, start=1):
        artifact_path = resolve_artifact_path(index_path, index_row)

        with np.load(artifact_path, allow_pickle=True) as item:
            if "rater_masks" not in item:
                raise KeyError(
                    f"{artifact_path} 缺少 rater_masks，不能做 annotation mode audit。"
                )

            annotations = as_binary_masks(item["rater_masks"])
            num_raters = int(annotations.shape[0])

            d = pairwise_distance_matrix(annotations)
            ann_div = mean_upper_triangle(d)

            errors: dict[int, float] = {}
            medoids: dict[int, tuple[int, ...]] = {}

            for m in range(1, num_raters + 1):
                error, selected, _ = exact_medoid_solution(d, m)
                errors[m] = float(error)
                medoids[m] = selected

            e1 = errors[1]

            explained = {
                m: explained_fraction(e1, errors[m])
                for m in errors
            }

            positive_rater_count = int(
                np.sum(annotations.reshape(num_raters, -1).sum(axis=1) > 0)
            )
            presence_disagreement = int(
                0 < positive_rater_count < num_raters
            )

            row: dict[str, Any] = {
                "case_id": str(index_row["case_id"]),
                "patient_id": str(
                    index_row.get("patient_id")
                    or patient_id_from_case(index_row["case_id"])
                ),
                "num_raters": num_raters,
                "positive_rater_count": positive_rater_count,
                "presence_disagreement": presence_disagreement,
                "annotation_diversity": ann_div,

                "E1": errors.get(1, float("nan")),
                "E2": errors.get(2, float("nan")),
                "E3": errors.get(3, float("nan")),
                "E4": errors.get(4, float("nan")),

                "explained_1": explained.get(1, float("nan")),
                "explained_2": explained.get(2, float("nan")),
                "explained_3": explained.get(3, float("nan")),
                "explained_4": explained.get(4, float("nan")),

                "M80": minimum_m_for_explained(errors, 0.80),
                "M90": minimum_m_for_explained(errors, 0.90),
                "M95": minimum_m_for_explained(errors, 0.95),

                "delta_2_over_E1": (
                    (errors[1] - errors[2]) / e1
                    if e1 > EPS and 2 in errors else 0.0
                ),
                "delta_3_over_E1": (
                    (errors[2] - errors[3]) / e1
                    if e1 > EPS and 3 in errors else 0.0
                ),
                "delta_4_over_E1": (
                    (errors[3] - errors[4]) / e1
                    if e1 > EPS and 4 in errors else 0.0
                ),

                "medoids_M1": "|".join(map(str, medoids.get(1, ()))),
                "medoids_M2": "|".join(map(str, medoids.get(2, ()))),
                "medoids_M3": "|".join(map(str, medoids.get(3, ()))),
                "medoids_M4": "|".join(map(str, medoids.get(4, ()))),
            }

            # --------------------------------------------------------------
            # 可选：模型输出的离散 component coverage
            # --------------------------------------------------------------
            row.update({
                "probability_active_component_count": float("nan"),
                "sampled_component_count": float("nan"),
                "sampled_component_ids": "",
                "component_prototype_coverage": float("nan"),
                "all_sample_coverage": float("nan"),
                "component_coverage_gap_vs_oracle_E2": float("nan"),
                "sample_coverage_gain_over_components": float("nan"),
            })

            if (
                "generated_samples" in item
                and "component_indices" in item
            ):
                generated = as_binary_masks(item["generated_samples"])
                indices = np.asarray(item["component_indices"]).reshape(-1)

                ids, prototypes, counts = empirical_component_prototypes(
                    generated,
                    indices,
                )

                component_coverage = mean_annotation_coverage(
                    annotations,
                    prototypes,
                )
                all_sample_coverage = mean_annotation_coverage(
                    annotations,
                    generated,
                )

                row["sampled_component_count"] = len(ids)
                row["sampled_component_ids"] = "|".join(map(str, ids))
                row["component_prototype_coverage"] = component_coverage
                row["all_sample_coverage"] = all_sample_coverage
                row["component_coverage_gap_vs_oracle_E2"] = (
                    component_coverage - errors[2]
                    if 2 in errors else float("nan")
                )
                row["sample_coverage_gain_over_components"] = (
                    component_coverage - all_sample_coverage
                )

                for component, count in counts.items():
                    row[f"sample_count_component_{component}"] = count

            if "component_probabilities" in item:
                probabilities = np.asarray(
                    item["component_probabilities"],
                    dtype=np.float64,
                ).reshape(-1)

                row["probability_active_component_count"] = int(
                    np.sum(probabilities >= float(args.active_threshold))
                )

                for k, probability in enumerate(probabilities):
                    row[f"component_probability_{k}"] = float(probability)

            rows.append(row)

        if case_number % 100 == 0 or case_number == len(index_rows):
            print(
                f"[Stage3.5] processed {case_number}/{len(index_rows)} cases"
            )

    # 不同病例可能因为 K 不同而出现不同动态列，统一 field names。
    all_fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                all_fields.append(key)
                seen.add(key)
    rows = [
        {key: row.get(key, "") for key in all_fields}
        for row in rows
    ]

    per_case_path = output_dir / "intrinsic_annotation_mode_per_case.csv"
    write_csv(per_case_path, rows)

    patient_rows = patient_summary(rows)
    write_csv(
        output_dir / "intrinsic_annotation_mode_per_patient.csv",
        patient_rows,
    )

    quartiles = quartile_summary(rows)
    write_csv(
        output_dir / "intrinsic_annotation_mode_quartiles.csv",
        quartiles,
    )

    summary = build_summary(rows)
    summary["artifact_index"] = str(index_path)
    summary["active_component_threshold"] = float(args.active_threshold)

    (
        output_dir / "intrinsic_annotation_mode_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 简洁控制台报告
    print("=" * 100)
    print("Stage 3.5 — Intrinsic Annotation Mode Audit")
    print("=" * 100)
    print(
        f"Cases/Patients : {summary['num_cases']}/"
        f"{summary['num_patients']}"
    )
    print(
        f"Mean D_ann     : "
        f"{summary['means'].get('annotation_diversity', float('nan')):.4f}"
    )
    print(
        f"Mean explained by M=2 : "
        f"{summary['means'].get('explained_2', float('nan')):.4f}"
    )

    for name in ("M80", "M90", "M95"):
        item = summary["effective_prototype_count_sensitivity"][name]
        print(
            f"{name}: hist={item['histogram']} | "
            f"P(M<=2)={item['fraction_le_2']:.3f} | "
            f"P(M>=3)={item['fraction_ge_3']:.3f}"
        )

    print(
        "Presence disagreement fraction: "
        f"{summary['presence_disagreement']['fraction_cases']:.4f}"
    )

    if np.isfinite(
        summary["means"].get("component_prototype_coverage", float("nan"))
    ):
        print(
            "Component-prototype coverage: "
            f"{summary['means']['component_prototype_coverage']:.4f}"
        )
        print(
            "All-sample coverage       : "
            f"{summary['means']['all_sample_coverage']:.4f}"
        )
        print(
            "Continuous-z coverage gain: "
            f"{summary['means']['sample_coverage_gain_over_components']:.4f}"
        )

    print("=" * 100)
    print(f"[DONE] {per_case_path}")
    print(
        f"[DONE] "
        f"{output_dir / 'intrinsic_annotation_mode_per_patient.csv'}"
    )
    print(
        f"[DONE] "
        f"{output_dir / 'intrinsic_annotation_mode_quartiles.csv'}"
    )
    print(
        f"[DONE] "
        f"{output_dir / 'intrinsic_annotation_mode_summary.json'}"
    )


if __name__ == "__main__":
    main()
