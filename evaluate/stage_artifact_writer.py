"""
Stage 2/3/4 共用实验数据保存规范。

从 Stage 2 开始固定保存：
1. GED² @ 1/4/8/16
2. CrossDice @ 1/4/8/16
3. BestDice @ 1/4/8/16
4. SampleDiversity @ 1/4/8/16
5. U^amb, D_ann, K_eff, K_active
6. beta_c, beta_coarse/middle/fine, tau
7. 每病例原始 Fig. 数据：
   - image
   - 4 rater masks
   - 16 generated samples
   - component indices
   - component probabilities
   - ambiguity map
   - vacuity map
   - expected-data entropy map

本文件只负责“保存协议”，不负责模型推理。
Stage 3/4 直接复用，保证不同阶段输出列名/NPZ key 一致。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "EG_GM_PUNET_STAGE_ARTIFACTS_V1"


def safe_case_filename(case_id: str) -> str:
    value = str(case_id)
    for character in '<>:"/\\|?* ':
        value = value.replace(character, "_")
    return value


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_case_artifact(
    *,
    directory: Path,
    case_id: str,
    patient_id: str,
    arrays: dict[str, Any],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{safe_case_filename(case_id)}.npz"

    payload = {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "case_id": np.asarray(str(case_id)),
        "patient_id": np.asarray(str(patient_id)),
    }
    payload.update(arrays)

    np.savez_compressed(path, **payload)
    return path


def build_wide_case_rows(
    *,
    metric_rows: list[dict[str, Any]],
    mechanism_rows: list[dict[str, Any]],
    sample_counts: list[int],
) -> list[dict[str, Any]]:
    """
    把 long-format sampling metrics 转成后续画表最方便的一病例一行格式。

    固定列：
        ged2_1/4/8/16
        cross_dice_1/4/8/16
        best_dice_1/4/8/16
        sample_diversity_1/4/8/16
    """
    metric_index = {
        (str(row["case_id"]), int(row["num_samples"])): row
        for row in metric_rows
    }

    outputs: list[dict[str, Any]] = []
    for mechanism in mechanism_rows:
        case_id = str(mechanism["case_id"])
        row = dict(mechanism)

        for count in sample_counts:
            metric = metric_index[(case_id, int(count))]
            row[f"ged2_{count}"] = float(metric["ged2"])
            row[f"cross_dice_{count}"] = float(metric["cross_dice"])
            row[f"best_dice_{count}"] = float(
                metric["best_sample_dice_to_raters"]
            )
            row[f"sample_diversity_{count}"] = float(
                metric["sample_diversity_distance"]
            )

        outputs.append(row)

    return outputs


def write_schema_manifest(
    *,
    path: Path,
    stage_name: str,
    sample_counts: list[int],
    artifact_keys: list[str],
) -> None:
    value = {
        "schema_version": SCHEMA_VERSION,
        "stage_name": stage_name,
        "sample_counts": [int(v) for v in sample_counts],
        "required_scalar_metrics": [
            "GED2@1/4/8/16",
            "CrossDice@1/4/8/16",
            "BestDice@1/4/8/16",
            "SampleDiversity@1/4/8/16",
            "U_amb",
            "D_ann",
            "K_eff",
            "K_active",
            "beta_categorical",
            "beta_coarse",
            "beta_middle",
            "beta_fine",
            "tau",
        ],
        "per_case_npz_keys": artifact_keys,
        "notes": (
            "Stage 2/3/4 share this schema. "
            "Prediction samples are always exact prior samples; "
            "posterior-derived capacities are diagnostic/training-control only."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
