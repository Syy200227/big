"""
EG-GM-PUNet v2 Stage 2b 正式评估 + 全量原始 Fig. 数据归档。

默认：
- split=test
- exact prior samples=16
- metrics@1/4/8/16
- 每个病例都保存 NPZ，不再只保存示例病例。

特别说明
--------
测试阶段为了记录 Stage 2 的 coarse/middle/fine capacity 与 beta：
会使用 config.validation.rater_index 构建一个 diagnostic posterior。

但是：
1. 16 个 segmentation samples 始终只从 prior exact sampling 产生；
2. 医生 mask 不参与 prior prediction；
3. diagnostic posterior 仅用于记录训练 controller 所定义的 group capacity；
4. D_ann 仅作为离线评价 target。

因此不会把 posterior 信息泄漏进最终 probabilistic prediction。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from itertools import combinations, islice
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.lidc_dataset_full import build_lidc_full_dataloader
from models.EG_GM_PUNet_v2 import EGGMPUNetV2BoundedDemandCapacity
from models.EG_GM_PUNet_v2.ambiguity_demand import ambiguity_demand_map
from models.Evidential_Multi_scale_Gaussian_Mixture_prob_unet.evidential_multiscale_gmm_parts import (
    multi_rater_soft_target,
)
from models.Multi_scale_Gaussian_Mixture_prob_unet.multiscale_grouped_gmm_parts import (
    sample_grouped_exact,
)
from utils.evidential_metrics import EvidentialMetricAccumulator
from utils.gmm_experiment_utils import prepare_device, set_seed
from utils.gmm_prior_evaluation import (
    evaluate_single_case,
    normalize_sample_counts,
    summarize_rows,
)

from evaluate.stage_artifact_writer import (
    build_wide_case_rows,
    save_case_artifact,
    write_csv,
    write_schema_manifest,
)


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ValueError("YAML 顶层必须是 mapping。")
    return value


def ids_from_batch(value, batch_size: int, prefix: str, offset: int) -> list[str]:
    if value is None:
        return [f"{prefix}_{offset+i:06d}" for i in range(batch_size)]
    if isinstance(value, str):
        return [value] * batch_size
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)] * batch_size


def patient_from_case(case_id: str) -> str:
    match = re.search(r"(LIDC-IDRI-\d{4})", str(case_id))
    return match.group(1) if match else str(case_id).split("_")[0]


def binary_iou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    inter = (a * b).sum()
    union = ((a + b) > 0).float().sum()
    if float(union.item()) <= eps:
        return 1.0
    return float((inter / (union + eps)).item())


def annotation_diversity(masks: torch.Tensor) -> float:
    value = masks
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    distances = [
        1.0 - binary_iou(value[a], value[b])
        for a, b in combinations(range(value.shape[0]), 2)
    ]
    return float(np.mean(distances)) if distances else 0.0


def build_model(config: dict[str, Any]) -> EGGMPUNetV2BoundedDemandCapacity:
    item = config["model"]
    return EGGMPUNetV2BoundedDemandCapacity(
        in_channels=int(item["in_channels"]),
        num_classes=int(item["num_classes"]),
        num_components=int(item["num_components"]),
        latent_dims={str(k): int(v) for k, v in item["latent_dims"].items()},
        num_channels=tuple(int(v) for v in item["num_channels"]),
        group_feature_indices={
            str(k): int(v)
            for k, v in item["group_feature_indices"].items()
        },
        group_decoder_indices={
            str(k): int(v)
            for k, v in item["group_decoder_indices"].items()
        },
        num_convs_per_block=int(item["num_convs_per_block"]),
        fusion_1x1_convs=int(item["fusion_1x1_convs"]),
        segmentation_head_1x1_convs=int(item["segmentation_head_1x1_convs"]),
        min_log_sigma=item.get("min_log_sigma"),
        max_log_sigma=item.get("max_log_sigma"),
        evidence_projection_channels=int(item["evidence_projection_channels"]),
        evidence_fusion_channels=int(item["evidence_fusion_channels"]),
        feedback_config=item["feedback"],
    )


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def corr(x, y, spearman=False):
    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    if spearman:
        a, b = rankdata(a), rankdata(b)
    return float(np.corrcoef(a, b)[0, 1])


def patient_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["patient_id"]), []).append(row)

    keys = [
        "annotation_diversity",
        "raw_ambiguity_demand",
        "ambiguity_demand",
        "categorical_capacity",
        "categorical_error",
        "categorical_deficit",
        "categorical_surplus",
        "prior_k_eff",
        "active_component_count",
        "coarse_deficit",
        "middle_deficit",
        "fine_deficit",
        "coarse_surplus",
        "middle_surplus",
        "fine_surplus",
        "beta_categorical",
        "beta_coarse",
        "beta_middle",
        "beta_fine",
        "gumbel_temperature",
    ]

    outputs = []
    for patient_id, items in grouped.items():
        row = {"patient_id": patient_id, "num_slices": len(items)}
        for key in keys:
            row[key] = float(np.mean([float(v[key]) for v in items]))
        outputs.append(row)
    return outputs


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/eg_gm_punet_v2_bounded_demand_capacity_lidc.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "result/eg_gm_punet_v2_bounded_demand_capacity_k5_full/"
            "best_val_ged4_eg_gm_punet_v2_stage2b.pth"
        ),
    )
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument(
        "--sample-counts",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16],
    )
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    config_path = resolve(args.config)
    checkpoint_path = resolve(args.checkpoint)

    config = load_yaml(config_path)
    set_seed(int(config["validation_seed"]))
    device = prepare_device(config.get("device", "cuda"))

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("config", config)
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    training = config["training"]
    validation = config["validation"]
    evaluation = config.get("evaluation", {})

    loader = build_lidc_full_dataloader(
        root=config["data_root"],
        split=args.split,
        batch_size=int(training["batch_size"]),
        num_workers=int(training["num_workers"]),
        shuffle=False,
        random_rater=False,
        rater_index=int(validation["rater_index"]),
    )

    sample_counts = normalize_sample_counts(
        list(args.sample_counts),
        int(args.num_samples),
    )

    output_dir = (
        checkpoint_path.parent
        / "evaluation"
        / args.split
        / f"samples_{args.num_samples}"
        / checkpoint_path.stem
    )
    if args.smoke_test:
        output_dir = output_dir / "smoke_test"

    case_artifact_dir = output_dir / "case_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence_metrics = EvidentialMetricAccumulator(
        num_bins=int(validation["ece_bins"]),
        ranking_bins=int(validation["ranking_bins"]),
    )

    metric_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    artifact_index: list[dict[str, Any]] = []

    total_cases = 0
    fallback = 0
    checkpoint_epoch = int(
        checkpoint.get("epoch") or training["epochs"]
    )

    active_threshold = float(
        evaluation.get("active_component_threshold", 0.02)
    )

    iterable = islice(loader, 2) if args.smoke_test else loader
    total_batches = min(2, len(loader)) if args.smoke_test else len(loader)

    print("=" * 108)
    print("EG-GM-PUNet v2 Stage 2b — Evaluation + Full Artifact Archive")
    print("=" * 108)
    print(f"[INFO] Checkpoint : {checkpoint_path}")
    print(f"[INFO] Epoch      : {checkpoint_epoch}")
    print(f"[INFO] Split      : {args.split}")
    print(f"[INFO] Samples    : {args.num_samples}")
    print(f"[INFO] Counts     : {sample_counts}")
    print(f"[INFO] Save all per-case artifacts: True")
    print(f"[INFO] Output     : {output_dir}")

    progress = tqdm(
        iterable,
        total=total_batches,
        desc="Stage2 evaluate",
        ncols=150,
    )

    for batch in progress:
        images = batch["image"].to(device=device, dtype=torch.float32)
        selected = batch["mask"].to(device=device, dtype=torch.float32)
        all_masks = batch["masks"].to(device=device, dtype=torch.float32)

        batch_size = images.shape[0]
        case_ids = ids_from_batch(
            batch.get("case_id"), batch_size, args.split, fallback
        )
        patient_ids = ids_from_batch(
            batch.get("patient_id"), batch_size, "patient", fallback
        )
        patient_ids = [
            patient_from_case(c) if p.startswith("patient_") else p
            for c, p in zip(case_ids, patient_ids)
        ]

        # Diagnostic posterior is used ONLY for group capacity/controller logging.
        state = model(
            image=images,
            segmentation=selected,
            is_training=True,
        )

        probe = model.feedback_probe(state)
        controls = model.feedback_controller(
            evidence=probe,
            state=state,
            base_beta_categorical=float(training["beta_categorical"]),
            base_beta_groups={
                str(k): float(v)
                for k, v in training["beta_groups"].items()
            },
            base_gumbel_temperature=float(training["gumbel_temperature"]),
            current_epoch=checkpoint_epoch,
            enabled=True,
            categorical_feedback=True,
            group_feedback=True,
            temperature_feedback=True,
        )

        prior_probs = state.prior.mixture_probs
        safe = prior_probs.clamp_min(1e-8)
        prior_entropy = -(safe * safe.log()).sum(dim=1)
        prior_k_eff = torch.exp(prior_entropy)
        active_k = (prior_probs >= active_threshold).sum(dim=1)

        exact = sample_grouped_exact(
            state.prior,
            num_samples=int(args.num_samples),
        )

        probabilistic_logits = []
        sample_evidence = []

        for sample_index in range(int(args.num_samples)):
            z_groups = {
                name: exact.z_groups[name][sample_index]
                for name in model.group_names
            }
            decode = model.probabilistic_model.decode_from_groups(
                state=state,
                z_groups=z_groups,
                component_indices=exact.component_indices[sample_index],
            )
            evidence = model.evidence_from_decode(decode)

            probabilistic_logits.append(decode.logits)
            sample_evidence.append(evidence)

            soft_target = multi_rater_soft_target(
                selected,
                all_masks,
                num_classes=model.num_classes,
            ).to(
                device=device,
                dtype=evidence.fused.probabilities.dtype,
            )
            evidence_metrics.update(
                evidence_output=evidence.fused,
                evidence_summary=evidence.summaries["fused"],
                segmentation=selected,
                all_segmentations=all_masks,
                soft_target=soft_target,
            )

        logits = torch.stack(probabilistic_logits, dim=0)
        generated_masks = logits.argmax(dim=2).float()
        generated_fg = torch.softmax(logits, dim=2)[:, :, 1]

        # Deterministic prior evidence maps are the canonical Fig. source.
        fine = probe.scales["fine"]
        fused = probe.fused
        demand_map = ambiguity_demand_map(probe, scale="fine")

        for b in range(batch_size):
            case_id = case_ids[b]
            patient_id = patient_ids[b]
            d_ann = annotation_diversity(all_masks[b])

            case_metric_rows = evaluate_single_case(
                sample_masks=generated_masks[:, b],
                foreground_probabilities=generated_fg[:, b],
                rater_masks=all_masks[b],
                sample_counts=sample_counts,
            )
            for row in case_metric_rows:
                row["case_id"] = case_id
                row["patient_id"] = patient_id
                row["prediction_branch"] = "probabilistic"
                row["split"] = args.split
            metric_rows.extend(case_metric_rows)

            # Evidence sampling metrics retained for consistency.
            e_probs = torch.stack(
                [e.fused.probabilities[b] for e in sample_evidence],
                dim=0,
            )
            e_masks = e_probs.argmax(dim=1).float()
            e_fg = e_probs[:, 1]
            case_e_rows = evaluate_single_case(
                sample_masks=e_masks,
                foreground_probabilities=e_fg,
                rater_masks=all_masks[b],
                sample_counts=sample_counts,
            )
            for row in case_e_rows:
                row["case_id"] = case_id
                row["patient_id"] = patient_id
                row["prediction_branch"] = "evidence"
                row["split"] = args.split
            evidence_rows.extend(case_e_rows)

            mechanism = {
                "case_id": case_id,
                "patient_id": patient_id,
                "annotation_diversity": d_ann,
                "raw_ambiguity_demand": float(
                    controls.raw_demand[b].item()
                ),
                "ambiguity_demand": float(
                    controls.ambiguity_demand[b].item()
                ),
                "categorical_capacity": float(
                    controls.categorical_capacity[b].item()
                ),
                "coarse_capacity": float(
                    controls.group_capacity["coarse"][b].item()
                ),
                "middle_capacity": float(
                    controls.group_capacity["middle"][b].item()
                ),
                "fine_capacity": float(
                    controls.group_capacity["fine"][b].item()
                ),
                "categorical_error": float(
                    controls.categorical_error[b].item()
                ),
                "categorical_deficit": float(
                    controls.categorical_deficit[b].item()
                ),
                "categorical_surplus": float(
                    controls.categorical_surplus[b].item()
                ),
                "coarse_error": float(
                    controls.group_error["coarse"][b].item()
                ),
                "coarse_deficit": float(
                    controls.group_deficit["coarse"][b].item()
                ),
                "coarse_surplus": float(
                    controls.group_surplus["coarse"][b].item()
                ),
                "middle_error": float(
                    controls.group_error["middle"][b].item()
                ),
                "middle_deficit": float(
                    controls.group_deficit["middle"][b].item()
                ),
                "middle_surplus": float(
                    controls.group_surplus["middle"][b].item()
                ),
                "fine_error": float(
                    controls.group_error["fine"][b].item()
                ),
                "fine_deficit": float(
                    controls.group_deficit["fine"][b].item()
                ),
                "fine_surplus": float(
                    controls.group_surplus["fine"][b].item()
                ),
                "coarse_raw_kl": float(
                    controls.raw_group_kl["coarse"][b].item()
                ),
                "middle_raw_kl": float(
                    controls.raw_group_kl["middle"][b].item()
                ),
                "fine_raw_kl": float(
                    controls.raw_group_kl["fine"][b].item()
                ),
                "prior_k_eff": float(prior_k_eff[b].item()),
                "active_component_count": int(active_k[b].item()),
                "beta_categorical": float(
                    controls.beta_categorical[b].item()
                ),
                "beta_coarse": float(
                    controls.beta_groups["coarse"][b].item()
                ),
                "beta_middle": float(
                    controls.beta_groups["middle"][b].item()
                ),
                "beta_fine": float(
                    controls.beta_groups["fine"][b].item()
                ),
                "gumbel_temperature": float(
                    controls.gumbel_temperature[b].item()
                ),
            }
            mechanism_rows.append(mechanism)

            artifact_path = save_case_artifact(
                directory=case_artifact_dir,
                case_id=case_id,
                patient_id=patient_id,
                arrays={
                    # 后续 Fig. 必需的 CT image 也一并保存。
                    "image": images[b].detach().cpu().numpy().astype(np.float16),
                    "rater_masks": all_masks[b]
                    .detach().cpu().numpy().astype(np.uint8),
                    "generated_samples": generated_masks[:, b]
                    .detach().cpu().numpy().astype(np.uint8),
                    "component_indices": exact.component_indices[:, b]
                    .detach().cpu().numpy().astype(np.int16),
                    "component_probabilities": prior_probs[b]
                    .detach().cpu().numpy().astype(np.float32),

                    # 固定的空间 Fig. 原始数据。
                    "ambiguity_map": demand_map[b, 0]
                    .detach().cpu().numpy().astype(np.float16),
                    "vacuity_map": fine.vacuity[b, 0]
                    .detach().cpu().numpy().astype(np.float16),
                    "expected_data_entropy_map": fine.expected_data_entropy[b, 0]
                    .detach().cpu().numpy().astype(np.float16),

                    # 额外保留 fused maps，后面做尺度对比时无需重跑。
                    "fused_vacuity_map": fused.vacuity[b, 0]
                    .detach().cpu().numpy().astype(np.float16),
                    "fused_expected_data_entropy_map": fused.expected_data_entropy[b, 0]
                    .detach().cpu().numpy().astype(np.float16),
                    "fine_foreground_probability": fine.probabilities[b, 1]
                    .detach().cpu().numpy().astype(np.float16),

                    # 机制标量也写入 NPZ，单病例画图无需再 join CSV。
                    "annotation_diversity": np.asarray(d_ann, dtype=np.float32),
                    "raw_ambiguity_demand": np.asarray(
                        mechanism["raw_ambiguity_demand"], dtype=np.float32
                    ),
                    "ambiguity_demand": np.asarray(
                        mechanism["ambiguity_demand"], dtype=np.float32
                    ),
                    "categorical_capacity": np.asarray(
                        mechanism["categorical_capacity"], dtype=np.float32
                    ),
                    "categorical_deficit": np.asarray(
                        mechanism["categorical_deficit"], dtype=np.float32
                    ),
                    "categorical_surplus": np.asarray(
                        mechanism["categorical_surplus"], dtype=np.float32
                    ),
                    "coarse_capacity": np.asarray(
                        mechanism["coarse_capacity"], dtype=np.float32
                    ),
                    "middle_capacity": np.asarray(
                        mechanism["middle_capacity"], dtype=np.float32
                    ),
                    "fine_capacity": np.asarray(
                        mechanism["fine_capacity"], dtype=np.float32
                    ),
                    "coarse_deficit": np.asarray(
                        mechanism["coarse_deficit"], dtype=np.float32
                    ),
                    "middle_deficit": np.asarray(
                        mechanism["middle_deficit"], dtype=np.float32
                    ),
                    "fine_deficit": np.asarray(
                        mechanism["fine_deficit"], dtype=np.float32
                    ),
                    "coarse_surplus": np.asarray(
                        mechanism["coarse_surplus"], dtype=np.float32
                    ),
                    "middle_surplus": np.asarray(
                        mechanism["middle_surplus"], dtype=np.float32
                    ),
                    "fine_surplus": np.asarray(
                        mechanism["fine_surplus"], dtype=np.float32
                    ),
                    "beta_categorical": np.asarray(
                        mechanism["beta_categorical"], dtype=np.float32
                    ),
                    "beta_coarse": np.asarray(
                        mechanism["beta_coarse"], dtype=np.float32
                    ),
                    "beta_middle": np.asarray(
                        mechanism["beta_middle"], dtype=np.float32
                    ),
                    "beta_fine": np.asarray(
                        mechanism["beta_fine"], dtype=np.float32
                    ),
                    "gumbel_temperature": np.asarray(
                        mechanism["gumbel_temperature"], dtype=np.float32
                    ),
                },
            )

            artifact_index.append(
                {
                    "case_id": case_id,
                    "patient_id": patient_id,
                    "artifact_path": str(artifact_path),
                }
            )

        fallback += batch_size
        total_cases += batch_size

    probabilistic_summary = summarize_rows(metric_rows, sample_counts)
    evidence_summary = summarize_rows(evidence_rows, sample_counts)
    calibration = evidence_metrics.compute()

    patient_rows = patient_summary(mechanism_rows)
    wide_rows = build_wide_case_rows(
        metric_rows=metric_rows,
        mechanism_rows=mechanism_rows,
        sample_counts=sample_counts,
    )

    write_csv(metric_rows, output_dir / "probabilistic_per_case_metrics.csv")
    write_csv(evidence_rows, output_dir / "evidence_per_case_metrics.csv")
    write_csv(mechanism_rows, output_dir / "stage2b_mechanism_per_case.csv")
    write_csv(patient_rows, output_dir / "stage2b_mechanism_per_patient.csv")
    write_csv(wide_rows, output_dir / "stage2b_analysis_table.csv")
    write_csv(artifact_index, output_dir / "case_artifact_index.csv")

    ann = [float(r["annotation_diversity"]) for r in mechanism_rows]
    raw_u = [float(r["raw_ambiguity_demand"]) for r in mechanism_rows]

    patient_ann = [float(r["annotation_diversity"]) for r in patient_rows]
    patient_raw_u = [float(r["raw_ambiguity_demand"]) for r in patient_rows]

    summary = {
        "stage": "Stage 2b Conservative Bounded Demand-Capacity Matching",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "split": args.split,
        "num_cases": total_cases,
        "num_prior_samples": int(args.num_samples),
        "sample_counts": sample_counts,
        "probabilistic_summary": probabilistic_summary,
        "evidence_summary": evidence_summary,
        "evidence_calibration": calibration,
        "demand_annotation": {
            "slice_pearson": corr(ann, raw_u, False),
            "slice_spearman": corr(ann, raw_u, True),
            "patient_pearson": corr(patient_ann, patient_raw_u, False),
            "patient_spearman": corr(patient_ann, patient_raw_u, True),
        },
        "mechanism_means": {
            key: float(np.mean([float(r[key]) for r in mechanism_rows]))
            for key in (
                "raw_ambiguity_demand",
                "ambiguity_demand",
                "categorical_capacity",
                "coarse_capacity",
                "middle_capacity",
                "fine_capacity",
                "categorical_error",
                "categorical_deficit",
                "categorical_surplus",
                "coarse_error",
                "middle_error",
                "fine_error",
                "coarse_deficit",
                "middle_deficit",
                "fine_deficit",
                "coarse_surplus",
                "middle_surplus",
                "fine_surplus",
                "prior_k_eff",
                "active_component_count",
                "beta_categorical",
                "beta_coarse",
                "beta_middle",
                "beta_fine",
                "gumbel_temperature",
            )
        },
        "stage2b_bounded_neutral_controller": True,
        "stage3_component_vitality": False,
        "stage4_ambiguity_alignment": False,
        "artifact_schema": "EG_GM_PUNET_STAGE_ARTIFACTS_V1",
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_schema_manifest(
        path=output_dir / "artifact_schema.json",
        stage_name="Stage 2b Conservative Bounded Demand-Capacity Matching",
        sample_counts=sample_counts,
        artifact_keys=[
            "image",
            "rater_masks",
            "generated_samples",
            "component_indices",
            "component_probabilities",
            "ambiguity_map",
            "vacuity_map",
            "expected_data_entropy_map",
            "fused_vacuity_map",
            "fused_expected_data_entropy_map",
            "fine_foreground_probability",
            "annotation_diversity",
            "raw_ambiguity_demand",
            "ambiguity_demand",
            "categorical_capacity",
            "coarse_capacity",
            "middle_capacity",
            "fine_capacity",
            "categorical_deficit",
            "categorical_surplus",
            "coarse_deficit",
            "middle_deficit",
            "fine_deficit",
            "coarse_surplus",
            "middle_surplus",
            "fine_surplus",
            "beta_categorical",
            "beta_coarse",
            "beta_middle",
            "beta_fine",
            "gumbel_temperature",
        ],
    )

    print("\n" + "=" * 108)
    print("Stage 2b Probabilistic Evaluation")
    print("=" * 108)
    print(
        f"{'Samples':>8} {'GED²↓':>10} {'CrossDice↑':>12} "
        f"{'BestDice↑':>11} {'Diversity':>10}"
    )
    print("-" * 68)
    for count in sample_counts:
        item = probabilistic_summary[str(count)]
        print(
            f"{count:>8d} "
            f"{item['ged2']['mean']:>10.5f} "
            f"{item['cross_dice']['mean']:>12.5f} "
            f"{item['best_sample_dice_to_raters']['mean']:>11.5f} "
            f"{item['sample_diversity_distance']['mean']:>10.5f}"
        )

    print("\n[DEMAND-CAPACITY]")
    print(
        "  Slice Pearson/Spearman  : "
        f"{summary['demand_annotation']['slice_pearson']} / "
        f"{summary['demand_annotation']['slice_spearman']}"
    )
    print(
        "  Patient Pearson/Spearman: "
        f"{summary['demand_annotation']['patient_pearson']} / "
        f"{summary['demand_annotation']['patient_spearman']}"
    )
    for key, value in summary["mechanism_means"].items():
        print(f"  {key:30s}: {value:.6f}")

    print("\n[ARTIFACTS]")
    print(f"  每病例 NPZ        : {case_artifact_dir}")
    print(f"  宽表               : {output_dir / 'stage2b_analysis_table.csv'}")
    print(f"  artifact index     : {output_dir / 'case_artifact_index.csv'}")
    print(f"  schema             : {output_dir / 'artifact_schema.json'}")
    print("=" * 108)


if __name__ == "__main__":
    main()
