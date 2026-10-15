"""
把 processed dataset 中的 4-rater masks 导出成 Stage 3.5 可读取的 annotation artifacts。

用途
----
当前 Stage3 evaluator 的 case_artifacts 已经能直接审计 allpos_full。
如果以后想比较：
    allpos_full vs anypos_full
而 anypos 没有对应模型 evaluation artifacts，可用本脚本只导出 annotations，
无需训练模型。

支持扫描：
    *.pt / *.pth / *.npz

对于 torch 文件，期待对象为 dict，至少包含：
    masks
并尽量包含：
    case_id
    patient_id

输出：
    annotation_artifacts/<case_id>.npz
    annotation_artifact_index.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


def patient_id_from_case(case_id: str) -> str:
    match = re.search(r"(LIDC-IDRI-\d{4})", str(case_id))
    return match.group(1) if match else str(case_id).split("_")[0]


def to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_sample(path: Path):
    suffix = path.suffix.lower()

    if suffix == ".npz":
        item = np.load(path, allow_pickle=True)
        return {key: item[key] for key in item.files}

    if suffix in {".pt", ".pth"}:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "读取 .pt/.pth 需要当前环境安装 PyTorch。"
            ) from exc

        value = torch.load(path, map_location="cpu")

        if isinstance(value, dict):
            return value

        raise TypeError(
            f"{path} 不是 dict。请根据你的 processed dataset 格式 "
            "在 load_sample() 中补充解析。"
        )

    raise ValueError(f"不支持文件：{path}")


def scalar_text(value, default: str) -> str:
    if value is None:
        return default

    if isinstance(value, str):
        return value

    if hasattr(value, "item"):
        try:
            return str(value.item())
        except Exception:
            pass

    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    split_root = root / args.split
    search_root = split_root if split_root.exists() else root

    files = []
    for pattern in ("*.pt", "*.pth", "*.npz"):
        files.extend(search_root.rglob(pattern))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(
            f"{search_root} 下没有找到 .pt/.pth/.npz 样本。"
        )

    output = Path(args.output)
    artifact_dir = output / "annotation_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for index, path in enumerate(files, start=1):
        item = load_sample(path)

        if "masks" not in item:
            continue

        masks = to_numpy(item["masks"])
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim != 3:
            raise ValueError(
                f"{path}: masks 应为 [R,H,W]，实际 {masks.shape}"
            )

        case_id = scalar_text(
            item.get("case_id"),
            path.stem,
        )
        patient_id = scalar_text(
            item.get("patient_id"),
            patient_id_from_case(case_id),
        )

        artifact = artifact_dir / f"{case_id}.npz"
        np.savez_compressed(
            artifact,
            case_id=np.asarray(case_id),
            patient_id=np.asarray(patient_id),
            rater_masks=(masks > 0.5).astype(np.uint8),
        )

        rows.append({
            "case_id": case_id,
            "patient_id": patient_id,
            "artifact_path": str(artifact.resolve()),
        })

        if index % 500 == 0:
            print(f"[EXPORT] scanned {index}/{len(files)}")

    index_path = output / "annotation_artifact_index.csv"
    with open(index_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case_id", "patient_id", "artifact_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Exported {len(rows)} cases")
    print(f"[DONE] {index_path}")


if __name__ == "__main__":
    main()
