"""
比较两个 Stage 3.5 Intrinsic Annotation Mode Audit。

典型用途：
    allpos_full vs anypos_full

重点比较：
- presence disagreement
- D_ann
- explained_2
- M80 / M90 / M95
- high-ambiguity Q4 中 M90>=3 的比例

注意：
如果两个数据集不是同一批 cases，本脚本只做 descriptive comparison；
若 case_id 有重叠，会另外报告 matched-case difference。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path: Path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def val(row, key):
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def summarize(rows):
    keys = [
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
    ]

    output = {
        "num_cases": len(rows),
        "means": {},
    }

    for key in keys:
        x = np.asarray([val(row, key) for row in rows], dtype=float)
        x = x[np.isfinite(x)]
        output["means"][key] = float(x.mean()) if len(x) else None

    for key in ("M80", "M90", "M95"):
        x = np.asarray([val(row, key) for row in rows], dtype=float)
        x = x[np.isfinite(x)].astype(int)
        output[key] = {
            "histogram": {
                str(m): int(np.sum(x == m))
                for m in (1, 2, 3, 4)
            },
            "fraction_le_2": float(np.mean(x <= 2)) if len(x) else None,
            "fraction_ge_3": float(np.mean(x >= 3)) if len(x) else None,
        }

    # annotation diversity quartile Q4
    d = np.asarray(
        [val(row, "annotation_diversity") for row in rows],
        dtype=float,
    )
    threshold = float(np.nanquantile(d, 0.75))
    q4 = [
        row
        for row in rows
        if val(row, "annotation_diversity") > threshold
    ]
    m90 = np.asarray([val(row, "M90") for row in q4], dtype=float)
    m90 = m90[np.isfinite(m90)].astype(int)

    output["Q4"] = {
        "num_cases": len(q4),
        "annotation_diversity_threshold": threshold,
        "mean_M90": float(m90.mean()) if len(m90) else None,
        "fraction_M90_ge_3": float(np.mean(m90 >= 3)) if len(m90) else None,
    }

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    a = read_csv(Path(args.a))
    b = read_csv(Path(args.b))

    result = {
        args.name_a: summarize(a),
        args.name_b: summarize(b),
    }

    # matched case comparison
    ai = {row["case_id"]: row for row in a}
    bi = {row["case_id"]: row for row in b}
    common = sorted(set(ai).intersection(bi))

    matched = {
        "num_cases": len(common),
        "mean_differences_B_minus_A": {},
    }

    for key in (
        "annotation_diversity",
        "explained_2",
        "M80",
        "M90",
        "M95",
    ):
        diff = np.asarray(
            [val(bi[c], key) - val(ai[c], key) for c in common],
            dtype=float,
        )
        diff = diff[np.isfinite(diff)]
        matched["mean_differences_B_minus_A"][key] = (
            float(diff.mean()) if len(diff) else None
        )

    result["matched_cases"] = matched

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
