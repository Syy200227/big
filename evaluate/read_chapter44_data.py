"""
读取论文第 4.4 节所需的已有结果数据（不训练、不推理）。

目标章节
--------
4.4 Uncertainty and Evidence-Guided Regulation
  4.4.1 Reliability of Evidential Uncertainty
  4.4.2 Ambiguity Demand and Adaptive Regulation

默认读取最终 LIDC-IDRI 主模型：
EG-GM-PUNet A2 / best validation GED checkpoint / test / 16 prior samples

运行后会生成：
E:\workspace\EG-GM-PUNet\paper_chapter44_export\
    44_source_metadata.json
    44_reliability_summary.csv
    44_demand_correlation_summary.csv
    44_regulation_means.csv
    44_mechanism_case_plot.csv
    44_mechanism_patient_plot.csv
    44_mechanism_long_plot.csv
    44_regulation_correlations.csv
    44_ambiguity_quartiles.csv
    44_consistency_check.csv
    source_summary.json
    source_evidence_per_case_metrics.csv
    source_v3_mechanism_per_case.csv
    source_v3_mechanism_per_patient.csv
    source_v3_analysis_table.csv
    source_case_artifact_index.csv   (若存在)
    chapter4_4_data_package.zip

注意：
- 本脚本只读取现有 CSV/JSON，不加载 checkpoint。
- 不会重新训练，也不会重新跑 evaluation。
- 第 4.4 节采用最终主模型 test@16，而不是 4.3 的 epoch-20 validation 消融模型。
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================================
# 只需确认这一行与你电脑一致
# ======================================================================================
PROJECT_ROOT = Path(r"E:\workspace\EG-GM-PUNet")

# 最优先读取：最终 A2 主模型、test、16 samples、best validation GED checkpoint
PREFERRED_RESULT_DIR = (
    PROJECT_ROOT
    / "result"
    / "eg_gm_punet_v3_a2_direct_diversity_match_k5"
    / "evaluation"
    / "test"
    / "samples_16"
    / "best_val_ged4_eg_gm_punet_v3_a2"
)

EXPORT_DIR = PROJECT_ROOT / "paper_chapter44_export"


# ======================================================================================
# 工具函数
# ======================================================================================

def finite_float(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def find_result_dir() -> Path:
    if (PREFERRED_RESULT_DIR / "summary.json").exists():
        return PREFERRED_RESULT_DIR

    base = (
        PROJECT_ROOT
        / "result"
        / "eg_gm_punet_v3_a2_direct_diversity_match_k5"
        / "evaluation"
        / "test"
    )
    candidates = []
    if base.exists():
        for p in base.rglob("summary.json"):
            parent = p.parent
            score = 0
            s = str(parent).lower().replace("\\", "/")
            if "samples_16" in s:
                score += 100
            if "best_val_ged4_eg_gm_punet_v3_a2" in s:
                score += 100
            if "smoke" in s:
                score -= 1000
            candidates.append((score, parent))

    if not candidates:
        raise FileNotFoundError(
            "没有找到最终 A2 evaluation 目录。\n"
            f"优先查找路径：\n{PREFERRED_RESULT_DIR}\n\n"
            "请确认结果目录仍位于 result\\eg_gm_punet_v3_a2_direct_diversity_match_k5 下。"
        )

    candidates.sort(key=lambda x: (x[0], x[1].stat().st_mtime), reverse=True)
    picked = candidates[0][1]
    print("[WARN] 优先目录不存在，自动选择：", picked)
    return picked


def require_file(root: Path, name: str) -> Path:
    p = root / name
    if not p.exists():
        raise FileNotFoundError(f"缺少必需文件：{p}")
    return p


def corr_pair(df: pd.DataFrame, x: str, y: str) -> tuple[float | None, float | None, int]:
    if x not in df.columns or y not in df.columns:
        return None, None, 0
    z = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(z) < 3:
        return None, None, len(z)
    return (
        float(z[x].corr(z[y], method="pearson")),
        float(z[x].corr(z[y], method="spearman")),
        int(len(z)),
    )


def mean_if_present(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    s = s[np.isfinite(s)]
    return float(s.mean()) if len(s) else None


# ======================================================================================
# 主程序
# ======================================================================================

def main():
    result_dir = find_result_dir()
    print("=" * 110)
    print("CHAPTER 4.4 DATA READER")
    print("=" * 110)
    print("Source:", result_dir)

    summary_path = require_file(result_dir, "summary.json")
    evidence_path = require_file(result_dir, "evidence_per_case_metrics.csv")
    mech_case_path = require_file(result_dir, "v3_mechanism_per_case.csv")
    mech_patient_path = require_file(result_dir, "v3_mechanism_per_patient.csv")

    analysis_path = result_dir / "v3_analysis_table.csv"
    artifact_index_path = result_dir / "case_artifact_index.csv"
    probabilistic_path = result_dir / "probabilistic_per_case_metrics.csv"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    evidence_df = pd.read_csv(evidence_path)
    mech_case = pd.read_csv(mech_case_path)
    mech_patient = pd.read_csv(mech_patient_path)

    if len(mech_case) == 0:
        raise RuntimeError("v3_mechanism_per_case.csv 为空。")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------------
    # 0. 来源元数据：保证以后写论文时知道表/图来自哪个 checkpoint
    # ------------------------------------------------------------------------------
    metadata = {
        "source_result_dir": str(result_dir),
        "stage": summary.get("stage"),
        "checkpoint": summary.get("checkpoint"),
        "checkpoint_epoch": summary.get("checkpoint_epoch"),
        "split": summary.get("split"),
        "num_cases": summary.get("num_cases"),
        "num_prior_samples": summary.get("num_prior_samples"),
        "sample_counts": summary.get("sample_counts"),
        "architecture": summary.get("architecture"),
        "chapter_usage": {
            "4.4.1": "Reliability of Evidential Uncertainty",
            "4.4.2": "Ambiguity Demand and Adaptive Regulation",
        },
    }
    (EXPORT_DIR / "44_source_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------------------
    # 1. 4.4.1 Reliability of Evidential Uncertainty
    # ------------------------------------------------------------------------------
    cal = summary.get("evidence_calibration", {})
    reliability_items = [
        ("Hard Dice (selected annotation)", "hard_dice_selected"),
        ("Hard Dice (all raters)", "hard_dice_all_raters"),
        ("Soft Dice (multi-rater)", "soft_dice_multi_rater"),
        ("Soft NLL (multi-rater)", "soft_nll_multi_rater"),
        ("Brier (multi-rater)", "brier_multi_rater"),
        ("ECE", "ece"),
        ("Error-detection AUROC", "error_detection_auroc"),
        ("Rater-disagreement AUROC", "rater_disagreement_auroc"),
        ("Mean vacuity", "mean_vacuity"),
        ("Foreground-weighted vacuity", "foreground_weighted_vacuity"),
        ("Ambiguity-weighted vacuity", "ambiguity_weighted_vacuity"),
        ("Predictive entropy", "predictive_entropy"),
        ("Expected-data entropy", "expected_data_entropy"),
        ("Mutual information", "mutual_information"),
        ("Vacuity-disagreement correlation", "vacuity_disagreement_correlation"),
        ("Correct-pixel vacuity", "correct_pixel_vacuity"),
        ("Incorrect-pixel vacuity", "incorrect_pixel_vacuity"),
        ("Incorrect-pixel fraction", "incorrect_pixel_fraction"),
    ]

    reliability_rows = []
    for display, key in reliability_items:
        reliability_rows.append({
            "metric": display,
            "json_key": key,
            "value": finite_float(cal.get(key)),
        })
    reliability_df = pd.DataFrame(reliability_rows)
    reliability_df.to_csv(
        EXPORT_DIR / "44_reliability_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------------------
    # 2. 4.4.2 Demand validity：slice + patient correlation
    #    这里采用 evaluator 原始定义：annotation_diversity_iou vs raw_ambiguity_demand
    # ------------------------------------------------------------------------------
    corr_json = summary.get("demand_annotation_iou", {})
    demand_corr_df = pd.DataFrame([
        {
            "level": "slice",
            "pearson_r": finite_float(corr_json.get("slice_pearson")),
            "spearman_rho": finite_float(corr_json.get("slice_spearman")),
            "n": int(summary.get("num_cases") or len(mech_case)),
        },
        {
            "level": "patient",
            "pearson_r": finite_float(corr_json.get("patient_pearson")),
            "spearman_rho": finite_float(corr_json.get("patient_spearman")),
            "n": int(len(mech_patient)),
        },
    ])
    demand_corr_df.to_csv(
        EXPORT_DIR / "44_demand_correlation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------------------
    # 3. EG-ALC 三尺度机制均值
    # ------------------------------------------------------------------------------
    scales = ["coarse", "middle", "fine"]
    reg_rows = []
    for g in scales:
        reg_rows.append({
            "scale": g,
            "ambiguity_demand_mean": finite_float(
                safe_get(summary, "mechanism_means", "ambiguity_demand")
            ),
            "raw_ambiguity_demand_mean": finite_float(
                safe_get(summary, "mechanism_means", "raw_ambiguity_demand")
            ),
            "capacity_mean": finite_float(
                safe_get(summary, "mechanism_means", f"{g}_capacity")
            ),
            "demand_minus_capacity_mean": finite_float(
                safe_get(summary, "mechanism_means", f"{g}_error")
            ),
            "deficit_mean": finite_float(
                safe_get(summary, "mechanism_means", f"{g}_deficit")
            ),
            "surplus_mean": finite_float(
                safe_get(summary, "mechanism_means", f"{g}_surplus")
            ),
            "beta_mean": finite_float(
                safe_get(summary, "mechanism_means", f"beta_{g}")
            ),
            "raw_joint_kl_mean": finite_float(
                safe_get(summary, "mechanism_means", f"{g}_raw_joint_kl")
            ),
            "raw_categorical_kl_mean": finite_float(
                safe_get(summary, "mechanism_means", f"{g}_raw_categorical_kl")
            ),
            "raw_gaussian_kl_mean": finite_float(
                safe_get(summary, "mechanism_means", f"{g}_raw_gaussian_kl")
            ),
            "basis_keff_mean": finite_float(
                safe_get(summary, "mechanism_means", f"{g}_basis_keff")
            ),
            "active_basis_count_mean": finite_float(
                safe_get(summary, "mechanism_means", f"{g}_active_basis_count")
            ),
        })
    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(
        EXPORT_DIR / "44_regulation_means.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------------------
    # 4. 画 D_ann vs U 的 slice-level / patient-level 数据
    # ------------------------------------------------------------------------------
    case_cols = [
        "case_id", "patient_id",
        "annotation_diversity_iou",
        "raw_ambiguity_demand",
        "ambiguity_demand",
        "feedback_ramp",
        "coarse_capacity", "middle_capacity", "fine_capacity",
        "coarse_error", "middle_error", "fine_error",
        "coarse_deficit", "middle_deficit", "fine_deficit",
        "coarse_surplus", "middle_surplus", "fine_surplus",
        "beta_coarse", "beta_middle", "beta_fine",
        "coarse_raw_joint_kl", "middle_raw_joint_kl", "fine_raw_joint_kl",
        "coarse_basis_keff", "middle_basis_keff", "fine_basis_keff",
        "coarse_active_basis_count",
        "middle_active_basis_count",
        "fine_active_basis_count",
    ]
    existing_case_cols = [c for c in case_cols if c in mech_case.columns]
    mech_case[existing_case_cols].to_csv(
        EXPORT_DIR / "44_mechanism_case_plot.csv",
        index=False,
        encoding="utf-8-sig",
    )

    patient_cols = [
        "patient_id", "num_cases",
        "annotation_diversity_iou",
        "raw_ambiguity_demand",
        "ambiguity_demand",
        "coarse_capacity", "middle_capacity", "fine_capacity",
        "coarse_error", "middle_error", "fine_error",
        "beta_coarse", "beta_middle", "beta_fine",
    ]
    existing_patient_cols = [c for c in patient_cols if c in mech_patient.columns]
    mech_patient[existing_patient_cols].to_csv(
        EXPORT_DIR / "44_mechanism_patient_plot.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------------------
    # 5. 三尺度 long-format：后面直接拿去画 U-C -> beta 图
    # ------------------------------------------------------------------------------
    long_rows = []
    for _, r in mech_case.iterrows():
        base = {
            "case_id": r.get("case_id"),
            "patient_id": r.get("patient_id"),
            "annotation_diversity_iou": finite_float(r.get("annotation_diversity_iou")),
            "raw_ambiguity_demand": finite_float(r.get("raw_ambiguity_demand")),
            "ambiguity_demand": finite_float(r.get("ambiguity_demand")),
        }
        for g in scales:
            row = dict(base)
            row.update({
                "scale": g,
                "capacity": finite_float(r.get(f"{g}_capacity")),
                "demand_minus_capacity": finite_float(r.get(f"{g}_error")),
                "deficit": finite_float(r.get(f"{g}_deficit")),
                "surplus": finite_float(r.get(f"{g}_surplus")),
                "beta": finite_float(r.get(f"beta_{g}")),
                "raw_joint_kl": finite_float(r.get(f"{g}_raw_joint_kl")),
                "basis_keff": finite_float(r.get(f"{g}_basis_keff")),
                "active_basis_count": finite_float(r.get(f"{g}_active_basis_count")),
            })
            long_rows.append(row)

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(
        EXPORT_DIR / "44_mechanism_long_plot.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------------------
    # 6. 机制相关性：直接帮助判断 Fig. 4.4.2 怎么画
    # ------------------------------------------------------------------------------
    corr_rows = []

    # Demand vs annotation disagreement
    for level, df in [("slice", mech_case), ("patient", mech_patient)]:
        p, s, n = corr_pair(
            df, "annotation_diversity_iou", "raw_ambiguity_demand"
        )
        corr_rows.append({
            "relationship": "annotation_diversity_iou vs raw_ambiguity_demand",
            "level_or_scale": level,
            "pearson_r": p,
            "spearman_rho": s,
            "n": n,
        })

    # Demand-capacity mismatch vs beta
    for g in scales:
        p, s, n = corr_pair(mech_case, f"{g}_error", f"beta_{g}")
        corr_rows.append({
            "relationship": "demand_minus_capacity vs beta",
            "level_or_scale": g,
            "pearson_r": p,
            "spearman_rho": s,
            "n": n,
        })

        p2, s2, n2 = corr_pair(
            mech_case, "raw_ambiguity_demand", f"beta_{g}"
        )
        corr_rows.append({
            "relationship": "raw_ambiguity_demand vs beta",
            "level_or_scale": g,
            "pearson_r": p2,
            "spearman_rho": s2,
            "n": n2,
        })

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(
        EXPORT_DIR / "44_regulation_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------------------
    # 7. 按真实 annotation disagreement 四分位数汇总，用于论文机制解释
    # ------------------------------------------------------------------------------
    qdf = mech_case.copy()
    ann_col = "annotation_diversity_iou"
    if ann_col in qdf.columns:
        ann_numeric = pd.to_numeric(qdf[ann_col], errors="coerce")
        good = ann_numeric.notna()
        qdf = qdf.loc[good].copy()
        qdf[ann_col] = ann_numeric.loc[good]

        try:
            qdf["ambiguity_quartile"] = pd.qcut(
                qdf[ann_col],
                q=4,
                labels=["Q1-low", "Q2", "Q3", "Q4-high"],
                duplicates="drop",
            )
            agg_cols = [
                "annotation_diversity_iou",
                "raw_ambiguity_demand",
                "ambiguity_demand",
                "coarse_capacity", "middle_capacity", "fine_capacity",
                "coarse_error", "middle_error", "fine_error",
                "beta_coarse", "beta_middle", "beta_fine",
            ]
            agg_cols = [c for c in agg_cols if c in qdf.columns]
            quart = (
                qdf.groupby("ambiguity_quartile", observed=True)[agg_cols]
                .mean(numeric_only=True)
                .reset_index()
            )
            quart.insert(
                1,
                "num_cases",
                qdf.groupby("ambiguity_quartile", observed=True).size().values,
            )
            quart.to_csv(
                EXPORT_DIR / "44_ambiguity_quartiles.csv",
                index=False,
                encoding="utf-8-sig",
            )
        except Exception as e:
            print("[WARN] 四分位数汇总失败：", e)

    # ------------------------------------------------------------------------------
    # 8. 一致性检查：
    #    mechanism 中的 annotation_diversity_iou 与 probabilistic evaluator 中的
    #    annotation_diversity_distance 是否一致。若不一致，先保留两个定义，不私自混用。
    # ------------------------------------------------------------------------------
    checks = []

    checks.append({
        "check": "num_mechanism_cases",
        "value": len(mech_case),
        "note": "Expected 769 for the final LIDC test split.",
    })
    checks.append({
        "check": "num_patients",
        "value": len(mech_patient),
        "note": "Patient-aggregated mechanism rows.",
    })
    checks.append({
        "check": "mechanism_annotation_diversity_mean",
        "value": mean_if_present(mech_case, "annotation_diversity_iou"),
        "note": "Used by demand-correlation analysis in the official A2 evaluator.",
    })

    if probabilistic_path.exists():
        prob = pd.read_csv(probabilistic_path)
        if "num_samples" in prob.columns:
            p16 = prob[pd.to_numeric(prob["num_samples"], errors="coerce") == 16].copy()
        else:
            p16 = prob.copy()

        checks.append({
            "check": "probabilistic_annotation_diversity_mean",
            "value": mean_if_present(p16, "annotation_diversity_distance"),
            "note": "Output-space annotation diversity from probabilistic evaluator.",
        })

        # If both case-wise definitions exist, compare them after de-duplicating per case.
        if (
            "case_id" in p16.columns
            and "annotation_diversity_distance" in p16.columns
            and "case_id" in mech_case.columns
            and "annotation_diversity_iou" in mech_case.columns
        ):
            psmall = (
                p16[["case_id", "annotation_diversity_distance"]]
                .drop_duplicates("case_id")
            )
            msmall = (
                mech_case[["case_id", "annotation_diversity_iou"]]
                .drop_duplicates("case_id")
            )
            merged = msmall.merge(psmall, on="case_id", how="inner")
            a = pd.to_numeric(merged["annotation_diversity_iou"], errors="coerce")
            b = pd.to_numeric(merged["annotation_diversity_distance"], errors="coerce")
            delta = (a - b).abs()
            checks.append({
                "check": "casewise_annotation_diversity_mean_abs_difference",
                "value": float(delta.mean()) if delta.notna().any() else None,
                "note": (
                    "If non-negligible, DO NOT silently treat the two columns as identical; "
                    "send this check to ChatGPT and we will inspect the metric definitions."
                ),
            })

    check_df = pd.DataFrame(checks)
    check_df.to_csv(
        EXPORT_DIR / "44_consistency_check.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------------------
    # 9. 把原始源文件复制一份，方便直接上传给我核对
    # ------------------------------------------------------------------------------
    copy_map = {
        summary_path: "source_summary.json",
        evidence_path: "source_evidence_per_case_metrics.csv",
        mech_case_path: "source_v3_mechanism_per_case.csv",
        mech_patient_path: "source_v3_mechanism_per_patient.csv",
    }
    if analysis_path.exists():
        copy_map[analysis_path] = "source_v3_analysis_table.csv"
    if artifact_index_path.exists():
        copy_map[artifact_index_path] = "source_case_artifact_index.csv"
    if probabilistic_path.exists():
        copy_map[probabilistic_path] = "source_probabilistic_per_case_metrics.csv"

    for src, dst_name in copy_map.items():
        shutil.copy2(src, EXPORT_DIR / dst_name)

    # ------------------------------------------------------------------------------
    # 10. 控制台快速预览
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("4.4.1 RELIABILITY OF EVIDENTIAL UNCERTAINTY")
    print("=" * 110)
    focus_keys = [
        "hard_dice_all_raters",
        "soft_dice_multi_rater",
        "soft_nll_multi_rater",
        "brier_multi_rater",
        "ece",
        "error_detection_auroc",
        "rater_disagreement_auroc",
        "vacuity_disagreement_correlation",
        "correct_pixel_vacuity",
        "incorrect_pixel_vacuity",
    ]
    for k in focus_keys:
        print(f"{k:40s}: {cal.get(k)}")

    print("\n" + "=" * 110)
    print("4.4.2 AMBIGUITY DEMAND VALIDITY")
    print("=" * 110)
    print(demand_corr_df.to_string(index=False))

    print("\n" + "=" * 110)
    print("4.4.2 ADAPTIVE REGULATION MEANS")
    print("=" * 110)
    print(reg_df.to_string(index=False))

    print("\n" + "=" * 110)
    print("CONSISTENCY CHECK")
    print("=" * 110)
    print(check_df.to_string(index=False))

    # ------------------------------------------------------------------------------
    # 11. zip：你运行完后，把这个 zip 直接发给 ChatGPT 即可
    # ------------------------------------------------------------------------------
    zip_base = PROJECT_ROOT / "chapter4_4_data_package"
    zip_path = Path(
        shutil.make_archive(
            str(zip_base),
            "zip",
            root_dir=EXPORT_DIR,
        )
    )

    print("\n" + "=" * 110)
    print("[DONE]")
    print("Export folder :", EXPORT_DIR)
    print("Upload this   :", zip_path)
    print("=" * 110)


if __name__ == "__main__":
    main()
