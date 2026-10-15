"""
EG-GM-PUNet v2 定量评估脚本。

这个文件是干什么的：
1. 读取 v2 best_val_ged4 checkpoint；
2. 在 test 集用与第一版完全一致的 prior sample counts=[1,4,8,16]；
3. 计算 probabilistic branch 和 evidence branch 的 GED²、Cross Dice、Best Dice、
   diversity 等逐病例指标；
4. 计算 evidential calibration 指标和 component utilization；
5. 输出 summary.json、probabilistic_per_case_metrics.csv、
   evidence_per_case_metrics.csv，后续可直接与 Full/No-Feedback 做 paired analysis。

运行示例：
    python evaluate/evaluate_eg_gm_punet_v2.py ^
      --config configs/eg_gm_punet_v2_demand_capacity_lidc.yaml ^
      --checkpoint result/eg_gm_punet_v2_demand_capacity_k5_full/best_val_ged4_eg_gm_punet_v2.pth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.lidc_dataset_full import build_lidc_full_dataloader
from models.EG_GM_PUNet_v2 import EGGMPUNetV2
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
    write_csv,
)


def build_model(config: dict[str, Any]) -> EGGMPUNetV2:
    item = config["model"]
    return EGGMPUNetV2(
        in_channels=int(item["in_channels"]),
        num_classes=int(item["num_classes"]),
        num_components=int(item["num_components"]),
        latent_dims={str(k): int(v) for k, v in item["latent_dims"].items()},
        num_channels=tuple(int(v) for v in item["num_channels"]),
        group_feature_indices={str(k): int(v) for k, v in item["group_feature_indices"].items()},
        group_decoder_indices={str(k): int(v) for k, v in item["group_decoder_indices"].items()},
        num_convs_per_block=int(item["num_convs_per_block"]),
        fusion_1x1_convs=int(item["fusion_1x1_convs"]),
        segmentation_head_1x1_convs=int(item["segmentation_head_1x1_convs"]),
        min_log_sigma=item.get("min_log_sigma"),
        max_log_sigma=item.get("max_log_sigma"),
        evidence_projection_channels=int(item["evidence_projection_channels"]),
        evidence_fusion_channels=int(item["evidence_fusion_channels"]),
        feedback_v2_config=item["feedback_v2"],
    )


def add_meta(rows, branch: str, case_id: str, split: str) -> None:
    for row in rows:
        row["prediction_branch"] = branch
        row["case_id"] = case_id
        row["split"] = split


def case_ids(value, batch: int, offset: int, split: str) -> list[str]:
    if value is None:
        return [f"{split}_{offset+i:06d}" for i in range(batch)]
    if isinstance(value, str):
        return [value]
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)] * batch


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/eg_gm_punet_v2_demand_capacity_lidc.yaml",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=16)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    import yaml
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    set_seed(int(config["validation_seed"]))
    device = prepare_device(config.get("device", "cuda"))

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", config)
    model = build_model(checkpoint_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    loader = build_lidc_full_dataloader(
        root=config["data_root"],
        split=args.split,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        shuffle=False,
        random_rater=False,
        rater_index=int(config["validation"]["rater_index"]),
    )
    counts = normalize_sample_counts(
        config.get("evaluation", {}).get("sample_counts", [1,4,8,16]),
        args.num_samples,
    )
    output_dir = (
        checkpoint_path.parent
        / "evaluation"
        / args.split
        / f"samples_{args.num_samples}"
        / checkpoint_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    p_rows, e_rows = [], []
    evidence_metrics = EvidentialMetricAccumulator(
        num_bins=int(config["validation"]["ece_bins"]),
        ranking_bins=int(config["validation"]["ranking_bins"]),
    )
    component_counts = torch.zeros(model.num_components, dtype=torch.long)
    probability_sum = torch.zeros(model.num_components, dtype=torch.float64)
    total_cases = 0
    offset = 0

    print("=" * 100)
    print("EG-GM-PUNet v2 quantitative evaluation")
    print("=" * 100)
    print(f"[INFO] Checkpoint : {checkpoint_path}")
    print(f"[INFO] Epoch      : {checkpoint.get('epoch')}")
    print(f"[INFO] Cases      : {len(loader.dataset)}")
    print(f"[INFO] Samples    : {args.num_samples}")
    print(f"[INFO] Counts     : {counts}")

    for batch in tqdm(loader, desc="v2 evaluate", ncols=140):
        image = batch["image"].to(device=device, dtype=torch.float32)
        selected = batch["mask"].to(device=device, dtype=torch.float32)
        all_masks = batch["masks"].to(device=device, dtype=torch.float32)
        ids = case_ids(batch.get("case_id"), image.shape[0], offset, args.split)

        state = model(image=image, segmentation=None, is_training=False)
        exact = sample_grouped_exact(state.prior, num_samples=args.num_samples)
        probability_sum += state.prior.mixture_probs.sum(dim=0).cpu().double()
        component_counts += torch.bincount(
            exact.component_indices.cpu().flatten(), minlength=model.num_components
        )
        total_cases += image.shape[0]

        p_logits = []
        e_probs = []
        for s in range(args.num_samples):
            z_groups = {name: exact.z_groups[name][s] for name in model.group_names}
            decode = model.probabilistic_model.decode_from_groups(
                state=state,
                z_groups=z_groups,
                component_indices=exact.component_indices[s],
            )
            evidence = model.evidence_from_decode(decode)
            p_logits.append(decode.logits)
            e_probs.append(evidence.fused.probabilities)

            soft_target = multi_rater_soft_target(
                selected, all_masks, num_classes=model.num_classes
            ).to(evidence.fused.probabilities)
            evidence_metrics.update(
                evidence_output=evidence.fused,
                evidence_summary=evidence.summaries["fused"],
                segmentation=selected,
                all_segmentations=all_masks,
                soft_target=soft_target,
            )

        p_logits = torch.stack(p_logits, dim=0)
        e_probs = torch.stack(e_probs, dim=0)
        p_masks = p_logits.argmax(dim=2).float()
        p_fg = torch.softmax(p_logits, dim=2)[:, :, 1]
        e_masks = e_probs.argmax(dim=2).float()
        e_fg = e_probs[:, :, 1]

        for i, cid in enumerate(ids):
            rows = evaluate_single_case(
                sample_masks=p_masks[:, i],
                foreground_probabilities=p_fg[:, i],
                rater_masks=all_masks[i],
                sample_counts=counts,
            )
            add_meta(rows, "probabilistic", cid, args.split)
            p_rows.extend(rows)

            rows = evaluate_single_case(
                sample_masks=e_masks[:, i],
                foreground_probabilities=e_fg[:, i],
                rater_masks=all_masks[i],
                sample_counts=counts,
            )
            add_meta(rows, "evidence", cid, args.split)
            e_rows.extend(rows)
        offset += image.shape[0]

    p_summary = summarize_rows(p_rows, counts)
    e_summary = summarize_rows(e_rows, counts)
    write_csv(p_rows, output_dir / "probabilistic_per_case_metrics.csv")
    write_csv(e_rows, output_dir / "evidence_per_case_metrics.csv")

    mean_pi = probability_sum / max(total_cases, 1)
    safe = mean_pi.clamp_min(1e-8)
    component_summary = {
        "mean_prior_component_probabilities": mean_pi.tolist(),
        "effective_component_count": float(torch.exp(-(safe * safe.log()).sum()).item()),
        "sampled_component_counts": component_counts.tolist(),
        "sampled_component_frequencies": (
            component_counts.float() / component_counts.sum().clamp_min(1)
        ).tolist(),
    }

    result = {
        "model": "EG-GM-PUNet-v2",
        "feedback_mode": checkpoint_config["model"]["feedback_v2"]["mode"],
        "vitality_enabled": checkpoint_config["model"]["feedback_v2"]["vitality"]["enabled"],
        "ambiguity_alignment_enabled": checkpoint_config["model"]["feedback_v2"]["ambiguity_alignment"]["enabled"],
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "num_cases": total_cases,
        "num_prior_samples": args.num_samples,
        "sample_counts": counts,
        "probabilistic_summary": p_summary,
        "evidence_summary": e_summary,
        "evidence_calibration": evidence_metrics.compute(),
        "component_summary": component_summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nProbabilistic branch")
    for count in counts:
        item = p_summary[str(count)]
        print(
            f"  S={count:2d}: GED2={item['ged2']['mean']:.4f}, "
            f"CrossDice={item['cross_dice']['mean']:.4f}, "
            f"BestDice={item['best_sample_dice_to_raters']['mean']:.4f}, "
            f"Diversity={item['sample_diversity_distance']['mean']:.4f}"
        )
    print(f"[INFO] K_eff  : {component_summary['effective_component_count']:.4f}")
    print(f"[DONE] Output : {output_dir}")


if __name__ == "__main__":
    main()
