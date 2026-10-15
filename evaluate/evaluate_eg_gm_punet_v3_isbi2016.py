
"""
Final ISBI 2016 evaluator for EG-GM-PUNet v3.

Evaluates an existing checkpoint only; NO training.

Unified metrics at @1/@4/@8/@16/@32:
- GED²
- CrossDice (for R=1 this is also m-Dice)
- BestDice (Dice*)
- Diversity
- CrossIoU
- BestIoU / single-GT Hungarian IoU
- mean probability mask Dice
- foreground probability variance
- empty sample fraction

Evidence:
- hard/soft Dice
- NLL
- Brier
- ECE
- error-detection AUROC
- vacuity, predictive entropy, expected-data entropy, MI

Mechanism:
- KL, K_eff, latent std, capacity, beta for c/m/f

Prediction is prior-only exact GMM sampling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.isbi2016_dataset import build_isbi2016_dataloader
from models.EG_GM_PUNet_v3_A1a import EGGMPUNetV3A1a
from models.Hierarchical_Multi_scale_Gaussian_Mixture_prob_unet_A1a import effective_component_count
from utils.evidential_metrics import EvidentialMetricAccumulator
from utils.gmm_experiment_utils import (
    binary_scores_from_logits,
    prepare_device,
    prior_distribution_metrics,
    set_seed,
)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg):
    m = cfg["model"]
    return EGGMPUNetV3A1a(
        in_channels=int(m["in_channels"]),
        num_classes=int(m["num_classes"]),
        num_components={str(k): int(v) for k, v in m["num_components"].items()},
        latent_dims={str(k): int(v) for k, v in m["latent_dims"].items()},
        num_channels=tuple(int(v) for v in m["num_channels"]),
        group_feature_indices={str(k): int(v) for k, v in m["group_feature_indices"].items()},
        group_decoder_indices={str(k): int(v) for k, v in m["group_decoder_indices"].items()},
        num_convs_per_block=int(m["num_convs_per_block"]),
        head_hidden_dim=int(m["head_hidden_dim"]),
        fusion_1x1_convs=int(m["fusion_1x1_convs"]),
        segmentation_head_1x1_convs=int(m["segmentation_head_1x1_convs"]),
        min_log_sigma=m.get("min_log_sigma"),
        max_log_sigma=m.get("max_log_sigma"),
        mixture_temperature=float(m["mixture_temperature"]),
        posterior_delta_logit_scale=float(m["posterior_residual"]["delta_logit_scale"]),
        posterior_delta_mu_scale=float(m["posterior_residual"]["delta_mu_scale"]),
        posterior_delta_log_sigma_scale=float(m["posterior_residual"]["delta_log_sigma_scale"]),
        scale_balance_ema_decay=float(m["scale_balance"]["ema_decay"]),
        scale_balance_gamma=float(m["scale_balance"]["gamma"]),
        scale_balance_alpha_min=float(m["scale_balance"]["alpha_min"]),
        scale_balance_alpha_max=float(m["scale_balance"]["alpha_max"]),
        evidence_projection_channels=int(m["evidence_projection_channels"]),
        evidence_fusion_channels=int(m["evidence_fusion_channels"]),
        feedback_config=m["feedback"],
    )


def mixture_total_std_rms(distribution):
    pi = distribution.mixture_probs.unsqueeze(-1)
    mean = (pi * distribution.mu).sum(dim=1, keepdim=True)
    within = (pi * distribution.sigma.pow(2)).sum(dim=1)
    between = (pi * (distribution.mu - mean).pow(2)).sum(dim=1)
    return torch.sqrt((within + between).mean(dim=1).clamp_min(1e-12))


def safe_mean(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def summary_stat(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def pairwise_diversity_binary(pred_binary):
    # pred_binary [S,H,W], mean unordered pairwise 1-IoU
    s = pred_binary.shape[0]
    if s < 2:
        return 0.0

    vals = []
    for i in range(s):
        a = pred_binary[i]
        for j in range(i + 1, s):
            b = pred_binary[j]
            inter = (a & b).sum().item()
            union = (a | b).sum().item()
            iou = 1.0 if union == 0 else inter / union
            vals.append(1.0 - iou)
    return float(np.mean(vals))


def case_extra_metrics(logits, gt):
    # logits [S,1? or 2,H,W], gt [1,H,W]
    probs = torch.softmax(logits, dim=1)[:, 1]
    pred = probs >= 0.5
    target = gt[0] >= 0.5

    dice = []
    iou = []
    for p in pred:
        inter = (p & target).sum().float()
        denom = p.sum().float() + target.sum().float()
        union = (p | target).sum().float()

        d = torch.where(
            denom <= 0,
            torch.ones_like(denom),
            2 * inter / denom.clamp_min(1e-7),
        )
        j = torch.where(
            union <= 0,
            torch.ones_like(union),
            inter / union.clamp_min(1e-7),
        )
        dice.append(float(d.item()))
        iou.append(float(j.item()))

    mean_prob = probs.mean(dim=0) >= 0.5
    inter = (mean_prob & target).sum().float()
    denom = mean_prob.sum().float() + target.sum().float()
    mean_prob_dice = float(
        torch.where(
            denom <= 0,
            torch.ones_like(denom),
            2 * inter / denom.clamp_min(1e-7),
        ).item()
    )

    return {
        "m_dice": float(np.mean(dice)),         # paper's m-Dice for N=1
        "dice_star": float(np.max(dice)),       # best of samples
        "hm_iou_single_gt": float(np.max(iou)), # Hungarian optimum for 1 GT
        "cross_iou": float(np.mean(iou)),
        "best_iou": float(np.max(iou)),
        "mean_probability_mask_dice": mean_prob_dice,
        "sample_diversity_distance_manual": pairwise_diversity_binary(pred.cpu()),
        "foreground_probability_variance": float(
            probs.var(dim=0, unbiased=False).mean().item()
        ),
        "empty_sample_fraction": float(
            (pred.flatten(1).sum(dim=1) == 0).float().mean().item()
        ),
    }


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eg_gm_punet_v3_isbi2016_150.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--sample-counts", nargs="+", type=int, default=[1,4,8,16,32])
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(config_path)

    set_seed(int(cfg["evaluation_seed"]))
    device = prepare_device(cfg.get("device", "cuda"))

    loader = build_isbi2016_dataloader(
        root=cfg["data_root"],
        split=args.split,
        batch_size=1,
        num_workers=int(cfg["evaluation"]["num_workers"]),
        shuffle=False,
    )

    model = build_model(cfg).to(device)
    cp = Path(args.checkpoint)
    if not cp.is_absolute():
        cp = PROJECT_ROOT / cp
    obj = torch.load(cp, map_location=device)
    model.load_state_dict(obj["model_state_dict"], strict=True)
    model.eval()

    out_dir = (
        Path(cfg["save_dir"]) / "evaluation" / args.split
        / f"samples_{args.num_samples}" / cp.stem
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = sorted(set(int(v) for v in args.sample_counts))
    if max(counts) > int(args.num_samples):
        raise ValueError("sample-count cannot exceed --num-samples")

    rows = []
    evidence_acc = EvidentialMetricAccumulator(
        num_bins=int(cfg["validation"]["ece_bins"]),
        ranking_bins=int(cfg["validation"]["ranking_bins"]),
    )
    mechanism_rows = []

    iterator = loader
    if args.smoke_test:
        from itertools import islice
        iterator = islice(loader, 3)

    with torch.no_grad():
        for batch in tqdm(iterator, desc="ISBI final evaluation", ncols=160):
            image = batch["image"].to(device=device, dtype=torch.float32)
            gt = batch["mask"].to(device=device, dtype=torch.float32)
            all_masks = batch["masks"].to(device=device, dtype=torch.float32)
            case_id = batch["case_id"][0]

            # Posterior state only for evidence/mechanism diagnostics.
            post_state = model(
                image=image,
                segmentation=gt,
                is_training=True,
            )
            output = model.elbo(
                state=post_state,
                segmentation=gt,
                all_segmentations=all_masks,
                base_beta_groups={str(k): float(v) for k, v in cfg["training"]["base_beta_groups"].items()},
                evidence_weight=float(cfg["training"]["evidence_weight"]),
                current_epoch=int(obj.get("epoch", cfg["training"]["epochs"])),
                total_epochs=int(cfg["training"]["epochs"]),
                evidential_annealing_epochs=int(cfg["training"]["evidential_annealing_epochs"]),
                evidence_dice_weight=float(cfg["training"]["evidence_dice_weight"]),
                evidence_kl_weight=float(cfg["training"]["evidence_kl_weight"]),
                evidence_calibration_weight=float(cfg["training"]["evidence_calibration_weight"]),
                evidence_auxiliary_scale_weight=float(cfg["training"]["evidence_auxiliary_scale_weight"]),
                feedback_enabled=bool(cfg["training"]["feedback_enabled"]),
                update_feedback_stats=False,
                scale_balance_enabled=False,
                update_scale_balance_stats=False,
            )

            evidence_acc.update(
                evidence_output=output.evidence.fused,
                evidence_summary=output.evidence.summaries["fused"],
                segmentation=gt,
                all_segmentations=all_masks,
                soft_target=output.evidential_loss.soft_target,
            )

            # Prior-only exact sampling for all reported segmentation metrics.
            prior_state = model(
                image=image,
                segmentation=None,
                is_training=False,
            )
            sample_logits = model.probabilistic_model.sample(
                prior_state,
                num_samples=int(args.num_samples),
                return_component_indices=False,
            )[:, 0]  # [S,C,H,W], batch=1

            for n in counts:
                subset = sample_logits[:n].unsqueeze(1)  # [S,B=1,C,H,W]
                formal = prior_distribution_metrics(subset, all_masks)
                extra = case_extra_metrics(sample_logits[:n], gt[0])

                row = {
                    "case_id": case_id,
                    "num_samples": int(n),
                    "ged2": float(formal["ged2"][0].item()),
                    "cross_dice": float(formal["cross_dice"][0].item()),
                    "best_sample_dice_to_raters": float(formal["best_dice"][0].item()),
                    "sample_diversity_distance": float(formal["sample_diversity"][0].item()),
                    "annotation_diversity_distance": float(formal["annotation_diversity"][0].item()),
                    **extra,
                }
                rows.append(row)

            p = output.probabilistic
            c = output.adaptive_controls
            mr = {
                "case_id": case_id,
                "raw_ambiguity_demand": float(c.raw_demand[0].item()),
                "ambiguity_demand": float(c.ambiguity_demand[0].item()),
            }

            for name in model.group_names:
                pr = p.prior_distributions[name]
                qr = p.posterior_distributions[name]
                mr[f"{name}_kl"] = float(p.kl_group_per_sample[name][0].item())
                mr[f"{name}_beta"] = float(c.beta_groups[name][0].item())
                mr[f"{name}_capacity"] = float(c.group_capacity[name][0].item())
                mr[f"{name}_prior_keff"] = float(effective_component_count(pr)[0].item())
                mr[f"{name}_posterior_keff"] = float(effective_component_count(qr)[0].item())
                mr[f"{name}_prior_std_rms"] = float(mixture_total_std_rms(pr)[0].item())
                mr[f"{name}_posterior_std_rms"] = float(mixture_total_std_rms(qr)[0].item())

            mechanism_rows.append(mr)

    write_csv(out_dir / "probabilistic_per_case_metrics.csv", rows)
    write_csv(out_dir / "mechanism_per_case.csv", mechanism_rows)

    summary = {}
    for n in counts:
        subset = [r for r in rows if int(r["num_samples"]) == int(n)]
        summary[str(n)] = {}
        for metric in (
            "ged2", "cross_dice", "best_sample_dice_to_raters",
            "sample_diversity_distance", "m_dice", "dice_star",
            "hm_iou_single_gt", "cross_iou", "best_iou",
            "mean_probability_mask_dice",
            "foreground_probability_variance", "empty_sample_fraction",
        ):
            summary[str(n)][metric] = summary_stat([r[metric] for r in subset])

    result = {
        "dataset": "ISBI 2016 / ISIC 2016 Task 1",
        "split": args.split,
        "single_annotation": True,
        "checkpoint": str(cp),
        "checkpoint_epoch": int(obj.get("epoch", -1)),
        "num_prior_samples": int(args.num_samples),
        "sample_counts": counts,
        "prediction": "prior-only exact GMM sampling",
        "probabilistic_summary": summary,
        "evidence_calibration": evidence_acc.compute(),
        "mechanism_means": {
            k: safe_mean([r[k] for r in mechanism_rows])
            for k in mechanism_rows[0]
            if k != "case_id"
        } if mechanism_rows else {},
        "metric_notes": {
            "m_dice": "For one ground-truth annotation, mean Dice across prior samples.",
            "dice_star": "Best Dice among sampled segmentations.",
            "hm_iou_single_gt": "Optimal one-to-one IoU with one GT; reduces to best sampled IoU.",
            "diversity": "Prediction sample diversity only; no inter-rater diversity exists for R=1.",
        },
    }

    (out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
