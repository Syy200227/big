
"""
EG-GM-PUNet v3 final architecture on ISBI 2016 — FROM SCRATCH
===============================================================

Core model remains:
    MS-GMM + MS-DEH + EG-ALC

Important single-annotation rule:
    DMC is NOT a fourth module and is disabled here because R=1.
    No annotation is duplicated to fabricate disagreement.

Training stages:
1) --smoke-test:
   2 train batches + 2 val batches, 1 epoch.
2) --quick-screen:
   short isolated run from RANDOM INITIALIZATION.
   It never becomes the formal checkpoint and never resumes into formal training.
3) formal:
   independent RANDOM INITIALIZATION, full 150 epochs.

Recorded every epoch:
- joint/probabilistic/reconstruction/KL losses
- posterior segmentation Dice/IoU
- prior GED² / CrossDice / BestDice / Diversity
- coarse/middle/fine KL decomposition
- coarse/middle/fine prior/posterior K_eff
- latent std RMS
- scale-balance alpha
- EG-ALC demand/capacity/error/deficit/surplus/beta
- MS-DEH Dice/NLL/Brier/ECE/error-detection AUROC/vacuity/entropy/MI
- rater-disagreement quantities may be NaN or degenerate because ISBI has R=1

Rich validation every 10 epochs:
- exact prior sampling @1/@4/@8/@16
Final epoch additionally:
- @32
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from contextlib import nullcontext
from datetime import datetime, timezone, timedelta
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim import Adam
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.isbi2016_dataset import build_isbi2016_dataloader
from models.EG_GM_PUNet_v3_A1a import EGGMPUNetV3A1a
from models.Hierarchical_Multi_scale_Gaussian_Mixture_prob_unet_A1a import (
    effective_component_count,
)
from utils.evidential_metrics import EvidentialMetricAccumulator
from utils.gmm_experiment_utils import (
    binary_scores_from_logits,
    build_step_boundaries,
    calculate_l2_regularization,
    piecewise_learning_rate,
    prepare_device,
    prior_distribution_metrics,
    set_seed,
)


BEIJING = timezone(timedelta(hours=8))


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a dict")
    return cfg


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


def set_learning_rates(optimizer, probability_lr, evidence_lr):
    for group in optimizer.param_groups:
        if group["group_name"] == "probability":
            group["lr"] = float(probability_lr)
        elif group["group_name"] == "evidence":
            group["lr"] = float(evidence_lr)


def new_accumulator(model, ece_bins, ranking_bins):
    acc = {
        "samples": 0,
        "objective_loss": 0.0,
        "joint_loss": 0.0,
        "probabilistic_loss": 0.0,
        "weighted_evidence_loss": 0.0,
        "regularization_loss": 0.0,
        "reconstruction": 0.0,
        "weighted_kl": 0.0,
        "unweighted_kl": 0.0,
        "prob_dice": 0.0,
        "prob_iou": 0.0,
        "raw_demand": 0.0,
        "relative_demand": 0.0,
        "prior_ged2": 0.0,
        "prior_cross_dice": 0.0,
        "prior_best_dice": 0.0,
        "prior_sample_diversity": 0.0,
        "annotation_diversity_metric": 0.0,
        "prior_metric_samples": 0,
        "evidence_metrics": EvidentialMetricAccumulator(
            num_bins=int(ece_bins),
            ranking_bins=int(ranking_bins),
        ),
    }
    for name in model.group_names:
        for key in (
            "kl", "kl_cat", "kl_gauss",
            "capacity", "error", "deficit", "surplus",
            "beta", "raw_capacity_kl",
            "prior_keff", "posterior_keff",
            "alpha", "prior_std_rms", "posterior_std_rms",
            "kl_rate",
        ):
            acc[f"{name}_{key}"] = 0.0
    return acc


def update_accumulator(
    acc, *, output, segmentation, all_segmentations,
    objective_loss, regularization_loss, model
):
    b = int(segmentation.shape[0])
    p = output.probabilistic
    c = output.adaptive_controls

    scores = binary_scores_from_logits(p.decode.logits.detach(), segmentation)

    acc["samples"] += b
    acc["objective_loss"] += float(objective_loss.detach().item()) * b
    acc["joint_loss"] += float(output.loss.detach().item()) * b
    acc["probabilistic_loss"] += float(output.probabilistic_loss.detach().item()) * b
    acc["weighted_evidence_loss"] += float(output.weighted_evidential_loss.detach().item()) * b
    acc["regularization_loss"] += float(regularization_loss.detach().item()) * b
    acc["reconstruction"] += float(p.reconstruction_loss_sum.detach().item()) * b
    acc["weighted_kl"] += float(p.weighted_kl_mean.detach().item()) * b
    acc["unweighted_kl"] += float(p.kl_total_unweighted_mean.detach().item()) * b
    acc["prob_dice"] += float(scores["dice"].sum().item())
    acc["prob_iou"] += float(scores["iou"].sum().item())
    acc["raw_demand"] += float(c.raw_demand.detach().sum().item())
    acc["relative_demand"] += float(c.ambiguity_demand.detach().sum().item())

    for name in model.group_names:
        q = p.posterior_distributions[name]
        pr = p.prior_distributions[name]

        acc[f"{name}_kl"] += float(p.kl_group_per_sample[name].detach().sum().item())
        acc[f"{name}_kl_cat"] += float(p.kl_categorical_mean[name].detach().item()) * b
        acc[f"{name}_kl_gauss"] += float(p.kl_gaussian_mean[name].detach().item()) * b
        acc[f"{name}_capacity"] += float(c.group_capacity[name].detach().sum().item())
        acc[f"{name}_error"] += float(c.group_error[name].detach().sum().item())
        acc[f"{name}_deficit"] += float(c.group_deficit[name].detach().sum().item())
        acc[f"{name}_surplus"] += float(c.group_surplus[name].detach().sum().item())
        acc[f"{name}_beta"] += float(c.beta_groups[name].detach().sum().item())
        acc[f"{name}_raw_capacity_kl"] += float(c.raw_group_joint_kl[name].detach().sum().item())
        acc[f"{name}_prior_keff"] += float(effective_component_count(pr).detach().sum().item())
        acc[f"{name}_posterior_keff"] += float(effective_component_count(q).detach().sum().item())
        acc[f"{name}_alpha"] += float(p.scale_balance_alpha[name].detach().item()) * b
        acc[f"{name}_prior_std_rms"] += float(mixture_total_std_rms(pr).detach().sum().item())
        acc[f"{name}_posterior_std_rms"] += float(mixture_total_std_rms(q).detach().sum().item())
        acc[f"{name}_kl_rate"] += float(p.dimension_normalized_kl_mean[name].detach().item()) * b

    acc["evidence_metrics"].update(
        evidence_output=output.evidence.fused,
        evidence_summary=output.evidence.summaries["fused"],
        segmentation=segmentation,
        all_segmentations=all_segmentations,
        soft_target=output.evidential_loss.soft_target,
    )


def finish_accumulator(acc, model):
    n = max(int(acc["samples"]), 1)
    result = {}

    for key in (
        "objective_loss", "joint_loss", "probabilistic_loss",
        "weighted_evidence_loss", "regularization_loss",
        "reconstruction", "weighted_kl", "unweighted_kl",
        "prob_dice", "prob_iou", "raw_demand", "relative_demand",
    ):
        result[key] = acc[key] / n

    for name in model.group_names:
        for key in (
            "kl", "kl_cat", "kl_gauss",
            "capacity", "error", "deficit", "surplus",
            "beta", "raw_capacity_kl",
            "prior_keff", "posterior_keff",
            "alpha", "prior_std_rms", "posterior_std_rms",
            "kl_rate",
        ):
            result[f"{name}_{key}"] = acc[f"{name}_{key}"] / n

    result.update({
        f"evidence_{k}": v
        for k, v in acc["evidence_metrics"].compute().items()
    })

    m = int(acc["prior_metric_samples"])
    if m > 0:
        for key in (
            "prior_ged2", "prior_cross_dice",
            "prior_best_dice", "prior_sample_diversity",
            "annotation_diversity_metric",
        ):
            result[key] = acc[key] / m

    # Single-annotation dataset: no inter-rater diversity target exists.
    result["annotation_count"] = 1
    result["dmc_enabled"] = False
    return result


def iter_loader(loader, max_batches):
    if max_batches is None:
        return loader, len(loader)
    count = min(int(max_batches), len(loader))
    return islice(loader, count), count


def run_epoch(
    *,
    model, loader, device, cfg, epoch, training,
    optimizer, global_step, boundaries, max_batches=None,
):
    tcfg = cfg["training"]
    vcfg = cfg["validation"]

    model.train(training)
    acc = new_accumulator(
        model,
        ece_bins=vcfg["ece_bins"],
        ranking_bins=vcfg["ranking_bins"],
    )

    iterable, total = iter_loader(loader, max_batches)
    progress = tqdm(
        iterable,
        total=total,
        desc=f"ISBI {'Train' if training else 'Val'} {epoch}",
        ncols=205,
    )

    last_plr = 0.0
    last_elr = 0.0

    if training:
        optimizer.zero_grad(set_to_none=True)

    accumulation = int(tcfg.get("gradient_accumulation_steps", 1))

    for batch_idx, batch in enumerate(progress, start=1):
        if training:
            plr = piecewise_learning_rate(
                global_step,
                boundaries,
                [float(v) for v in tcfg["probabilistic_learning_rate_values"]],
            )
            elr = piecewise_learning_rate(
                global_step,
                boundaries,
                [float(v) for v in tcfg["evidence_learning_rate_values"]],
            )
            set_learning_rates(optimizer, plr, elr)
            last_plr, last_elr = plr, elr

        images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        segmentation = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)
        all_segmentations = batch["masks"].to(device=device, dtype=torch.float32, non_blocking=True)

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            state = model(
                image=images,
                segmentation=segmentation,
                is_training=True,  # posterior diagnostics available on validation
            )

            output = model.elbo(
                state=state,
                segmentation=segmentation,
                all_segmentations=all_segmentations,
                base_beta_groups={str(k): float(v) for k, v in tcfg["base_beta_groups"].items()},
                evidence_weight=float(tcfg["evidence_weight"]),
                current_epoch=int(epoch),
                total_epochs=int(tcfg["epochs"]),
                evidential_annealing_epochs=int(tcfg["evidential_annealing_epochs"]),
                evidence_dice_weight=float(tcfg["evidence_dice_weight"]),
                evidence_kl_weight=float(tcfg["evidence_kl_weight"]),
                evidence_calibration_weight=float(tcfg["evidence_calibration_weight"]),
                evidence_auxiliary_scale_weight=float(tcfg["evidence_auxiliary_scale_weight"]),
                feedback_enabled=bool(tcfg["feedback_enabled"]),
                update_feedback_stats=training,
                scale_balance_enabled=(
                    bool(tcfg["scale_balance_enabled"])
                    and int(epoch) <= int(tcfg["scale_balance_active_epochs"])
                ),
                update_scale_balance_stats=training,
            )

            if training:
                regularization_loss = (
                    float(tcfg["regularization_weight"])
                    * calculate_l2_regularization(model)
                )
            else:
                regularization_loss = torch.zeros((), device=device)

            objective = output.loss + regularization_loss

        if training:
            if not torch.isfinite(objective):
                raise RuntimeError(
                    f"NaN/Inf objective at epoch={epoch}, batch={batch_idx}"
                )

            (objective / accumulation).backward()

            do_step = (
                batch_idx % accumulation == 0
                or batch_idx == total
            )
            if do_step:
                clip = float(tcfg.get("gradient_clip_norm", 0.0))
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        update_accumulator(
            acc,
            output=output,
            segmentation=segmentation,
            all_segmentations=all_segmentations,
            objective_loss=objective,
            regularization_loss=regularization_loss,
            model=model,
        )

        if not training:
            sample_logits = model.probabilistic_model.sample(
                state,
                num_samples=int(vcfg["prior_samples_each_epoch"]),
                return_component_indices=False,
            )
            metrics = prior_distribution_metrics(sample_logits, all_segmentations)
            b = images.shape[0]
            acc["prior_metric_samples"] += b
            mapping = {
                "prior_ged2": "ged2",
                "prior_cross_dice": "cross_dice",
                "prior_best_dice": "best_dice",
                "prior_sample_diversity": "sample_diversity",
                "annotation_diversity_metric": "annotation_diversity",
            }
            for dst, src in mapping.items():
                acc[dst] += float(metrics[src].sum().item())

        completed = max(acc["samples"], 1)
        postfix = {
            "pDice": f"{acc['prob_dice']/completed:.4f}",
            "U": f"{acc['relative_demand']/completed:.3f}",
            "KLc": f"{acc['coarse_kl']/completed:.3f}",
            "KLm": f"{acc['middle_kl']/completed:.3f}",
            "KLf": f"{acc['fine_kl']/completed:.3f}",
        }
        if not training and acc["prior_metric_samples"] > 0:
            postfix["pCross"] = f"{acc['prior_cross_dice']/acc['prior_metric_samples']:.4f}"
            postfix["GED"] = f"{acc['prior_ged2']/acc['prior_metric_samples']:.4f}"
        progress.set_postfix(postfix)

    result = finish_accumulator(acc, model)
    result["probabilistic_learning_rate"] = last_plr
    result["evidence_learning_rate"] = last_elr
    return result, global_step


@torch.no_grad()
def rich_validation(model, loader, device, sample_counts):
    model.eval()
    max_samples = max(int(v) for v in sample_counts)

    totals = {
        int(n): {
            "n": 0,
            "ged2": 0.0,
            "cross_dice": 0.0,
            "best_dice": 0.0,
            "diversity": 0.0,
        }
        for n in sample_counts
    }

    for batch in tqdm(loader, desc=f"RichVal@{max_samples}", ncols=150):
        images = batch["image"].to(device=device, dtype=torch.float32)
        all_masks = batch["masks"].to(device=device, dtype=torch.float32)

        state = model(
            image=images,
            segmentation=None,
            is_training=False,
        )
        logits = model.probabilistic_model.sample(
            state,
            num_samples=max_samples,
            return_component_indices=False,
        )

        for n in sample_counts:
            metrics = prior_distribution_metrics(logits[:int(n)], all_masks)
            b = images.shape[0]
            totals[int(n)]["n"] += b
            totals[int(n)]["ged2"] += float(metrics["ged2"].sum().item())
            totals[int(n)]["cross_dice"] += float(metrics["cross_dice"].sum().item())
            totals[int(n)]["best_dice"] += float(metrics["best_dice"].sum().item())
            totals[int(n)]["diversity"] += float(metrics["sample_diversity"].sum().item())

    out = {}
    for n, d in totals.items():
        denom = max(d["n"], 1)
        out[str(n)] = {
            "num_cases": d["n"],
            "ged2": d["ged2"] / denom,
            "cross_dice": d["cross_dice"] / denom,
            "best_dice": d["best_dice"] / denom,
            "sample_diversity": d["diversity"] / denom,
            # With one ground-truth annotation, m-Dice equals mean sample-vs-GT Dice.
            "m_dice_single_annotation": d["cross_dice"] / denom,
        }
    return out


def checkpoint_state(model, optimizer, epoch, global_step, cfg, best_records, metrics):
    return {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
        "best_records": best_records,
        "metrics": metrics,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }


def save_checkpoint(path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_state(**kwargs), path)


def restore_checkpoint(path, model, optimizer, device):
    obj = torch.load(path, map_location=device)
    model.load_state_dict(obj["model_state_dict"], strict=True)
    optimizer.load_state_dict(obj["optimizer_state_dict"])
    return obj


def best_records_template():
    return {
        "val_prior_cross_dice": {
            "value": float("-inf"),
            "direction": "max",
            "checkpoint": "best_val_prior_cross_dice_isbi2016.pth",
            "epoch": None,
        },
        "val_prior_ged": {
            "value": float("inf"),
            "direction": "min",
            "checkpoint": "best_val_prior_ged_isbi2016.pth",
            "epoch": None,
        },
        "val_joint_loss": {
            "value": float("inf"),
            "direction": "min",
            "checkpoint": "best_val_joint_loss_isbi2016.pth",
            "epoch": None,
        },
        "val_probabilistic_dice": {
            "value": float("-inf"),
            "direction": "max",
            "checkpoint": "best_val_probabilistic_dice_isbi2016.pth",
            "epoch": None,
        },
        "val_evidence_dice": {
            "value": float("-inf"),
            "direction": "max",
            "checkpoint": "best_val_evidence_dice_isbi2016.pth",
            "epoch": None,
        },
        "val_error_detection_auroc": {
            "value": float("-inf"),
            "direction": "max",
            "checkpoint": "best_val_error_auroc_isbi2016.pth",
            "epoch": None,
        },
    }


def update_best(save_dir, best_records, model, optimizer, epoch, global_step, cfg, val):
    candidates = {
        "val_prior_cross_dice": float(val.get("prior_cross_dice", float("-inf"))),
        "val_prior_ged": float(val.get("prior_ged2", float("inf"))),
        "val_joint_loss": float(val["joint_loss"]),
        "val_probabilistic_dice": float(val["prob_dice"]),
        "val_evidence_dice": float(val.get("evidence_hard_dice_all_raters", float("-inf"))),
        "val_error_detection_auroc": float(val.get("evidence_error_detection_auroc", float("-inf"))),
    }

    for name, value in candidates.items():
        if not math.isfinite(value):
            continue
        rec = best_records[name]
        better = value < rec["value"] if rec["direction"] == "min" else value > rec["value"]
        if better:
            rec["value"] = value
            rec["epoch"] = int(epoch)
            save_checkpoint(
                Path(save_dir) / rec["checkpoint"],
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                cfg=cfg,
                best_records=best_records,
                metrics=val,
            )


def flatten_record(epoch, global_step, train, val):
    row = {"epoch": int(epoch), "global_step": int(global_step)}
    for prefix, metrics in (("train", train), ("val", val)):
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[f"{prefix}_{k}"] = v
    return row


def write_history(save_dir, history):
    save_dir = Path(save_dir)
    (save_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not history:
        return

    fields = []
    seen = set()
    for row in history:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with open(save_dir / "history.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(history)


def quick_screen_report(history):
    if not history:
        return {"status": "NO_DATA"}

    first = history[0]
    last = history[-1]

    checks = {
        "finite_train_loss": math.isfinite(float(last.get("train_objective_loss", float("nan")))),
        "finite_val_loss": math.isfinite(float(last.get("val_objective_loss", float("nan")))),
        "val_prior_cross_not_zero": float(last.get("val_prior_cross_dice", 0.0)) > 0.05,
        "val_prob_dice_not_zero": float(last.get("val_prob_dice", 0.0)) > 0.05,
        "coarse_kl_alive": float(last.get("val_coarse_kl", 0.0)) > 1e-5,
        "middle_kl_alive": float(last.get("val_middle_kl", 0.0)) > 1e-5,
        "fine_prior_std_finite": math.isfinite(float(last.get("val_fine_prior_std_rms", float("nan")))),
    }

    start_cross = float(first.get("val_prior_cross_dice", float("nan")))
    end_cross = float(last.get("val_prior_cross_dice", float("nan")))
    trend = end_cross - start_cross if math.isfinite(start_cross) and math.isfinite(end_cross) else float("nan")

    status = "CONTINUE_FORMAL" if all(checks.values()) else "CHECK_BEFORE_FORMAL"

    return {
        "status": status,
        "checks": checks,
        "first_val_prior_cross_dice": start_cross,
        "last_val_prior_cross_dice": end_cross,
        "cross_dice_change": trend,
        "note": (
            "Quick screen only checks numerical/training direction. "
            "It is not a formal model-selection result and is never resumed into the 150-epoch run."
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="configs/eg_gm_punet_v3_isbi2016_150.yaml",
    )
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--quick-screen", action="store_true")
    p.add_argument("--resume", default=None)
    args = p.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(config_path)

    mode = "formal"
    if args.smoke_test:
        mode = "smoke"
    elif args.quick_screen:
        mode = "quick"

    cfg = copy.deepcopy(cfg)

    if mode == "smoke":
        cfg["training"]["epochs"] = 1
        cfg["training"]["feedback_enabled"] = False
        cfg["model"]["feedback"]["enabled"] = False
    elif mode == "quick":
        cfg["training"]["epochs"] = int(cfg["screening"]["quick_screen_epochs"])
        cfg["training"]["feedback_enabled"] = False
        cfg["model"]["feedback"]["enabled"] = False

    set_seed(int(cfg["seed"]))
    device = prepare_device(cfg.get("device", "cuda"))

    base_save = Path(cfg["save_dir"])
    if mode == "smoke":
        save_dir = base_save / "_smoke_test"
    elif mode == "quick":
        save_dir = base_save / "_quick_screen"
    else:
        save_dir = base_save

    save_dir.mkdir(parents=True, exist_ok=True)

    aug = cfg["data_augmentation"]
    train_loader = build_isbi2016_dataloader(
        root=cfg["data_root"],
        split="train",
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"]["num_workers"]),
        shuffle=True,
        horizontal_flip_p=float(aug["horizontal_flip_p"]),
        vertical_flip_p=float(aug["vertical_flip_p"]),
    )
    val_loader = build_isbi2016_dataloader(
        root=cfg["data_root"],
        split="val",
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"]["num_workers"]),
        shuffle=False,
    )

    model = build_model(cfg).to(device)

    # CRITICAL: no LIDC checkpoint transfer.
    initialization_report = {
        "mode": "random_from_scratch",
        "loaded_checkpoint": None,
        "lidc_transfer": False,
        "in_channels": int(cfg["model"]["in_channels"]),
    }

    optimizer = Adam(
        [
            {
                "params": model.probabilistic_model.parameters(),
                "lr": float(cfg["training"]["probabilistic_learning_rate_values"][0]),
                "group_name": "probability",
            },
            {
                "params": model.evidence_head.parameters(),
                "lr": float(cfg["training"]["evidence_learning_rate_values"][0]),
                "group_name": "evidence",
            },
        ],
        weight_decay=0.0,
        betas=tuple(float(v) for v in cfg["training"]["adam_betas"]),
    )

    configured_epochs = int(cfg["training"]["epochs"])
    if mode == "formal":
        configured_epochs = 150
        cfg["training"]["epochs"] = 150

    # Global step counts optimizer updates, considering gradient accumulation.
    updates_per_epoch = math.ceil(
        len(train_loader) / int(cfg["training"]["gradient_accumulation_steps"])
    )
    total_updates = configured_epochs * updates_per_epoch
    boundaries = build_step_boundaries(
        total_updates,
        cfg["training"]["learning_rate_boundary_fractions"],
    )

    start_epoch = 1
    global_step = 0
    best_records = best_records_template()
    history = []
    rich_history = {}

    if args.resume:
        if mode != "formal":
            raise ValueError("--resume is only allowed for the formal run.")
        rp = Path(args.resume)
        if not rp.is_absolute():
            rp = PROJECT_ROOT / rp
        obj = restore_checkpoint(rp, model, optimizer, device)
        start_epoch = int(obj["epoch"]) + 1
        global_step = int(obj.get("global_step", 0))
        best_records = obj.get("best_records", best_records)
        hp = save_dir / "history.json"
        if hp.exists():
            history = json.loads(hp.read_text(encoding="utf-8"))
        rhp = save_dir / "rich_validation_history.json"
        if rhp.exists():
            rich_history = json.loads(rhp.read_text(encoding="utf-8"))

    manifest = {
        "experiment": cfg["experiment_name"],
        "mode": mode,
        "initialization": initialization_report,
        "dataset": {
            "name": "ISBI 2016 / ISIC 2016 Task 1",
            "annotation_count_per_image": 1,
            "train_cases": len(train_loader.dataset),
            "val_cases": len(val_loader.dataset),
            "test_not_used_during_training": True,
        },
        "architecture": {
            "top_level": "MS-GMM + MS-DEH + EG-ALC",
            "shared_semantic_k": False,
            "component_semantics": "latent density basis only",
            "prior_anchored_residual_posterior": True,
            "dmc_enabled": False,
            "dmc_reason": "single annotation R=1; no annotation-pair diversity target exists",
            "single_annotation_compatible": True,
        },
        "config": cfg,
    }
    (save_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if mode == "smoke":
        train_max_batches = int(cfg["screening"]["smoke_train_batches"])
        val_max_batches = int(cfg["screening"]["smoke_val_batches"])
    elif mode == "quick":
        train_max_batches = int(cfg["screening"]["quick_train_batches"])
        val_max_batches = int(cfg["screening"]["quick_val_batches"])
    else:
        train_max_batches = None
        val_max_batches = None

    epochs = int(cfg["training"]["epochs"])

    print("=" * 118)
    print("EG-GM-PUNet v3 on ISBI 2016 — FROM SCRATCH")
    print("MS-GMM + MS-DEH + EG-ALC | single annotation | DMC disabled")
    print("=" * 118)
    print(f"Mode       : {mode}")
    print(f"Device     : {device}")
    print(f"Train/Val  : {len(train_loader.dataset)}/{len(val_loader.dataset)}")
    print(f"Input      : RGB 3x256x256")
    print(f"Epochs     : {epochs}")
    print(f"Batch      : {cfg['training']['batch_size']} x accumulation {cfg['training']['gradient_accumulation_steps']}")
    print(f"Save dir   : {save_dir}")
    print(f"LIDC init  : NO")
    print(f"DMC        : OFF (R=1)")
    print("[START Beijing]", datetime.now(BEIJING).isoformat(timespec="seconds"))

    for epoch in range(start_epoch, epochs + 1):
        train_metrics, global_step = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            cfg=cfg,
            epoch=epoch,
            training=True,
            optimizer=optimizer,
            global_step=global_step,
            boundaries=boundaries,
            max_batches=train_max_batches,
        )

        # Fixed validation RNG for comparable stochastic metrics.
        cuda_devices = []
        if torch.cuda.is_available() and device.type == "cuda":
            cuda_devices = [
                device.index if device.index is not None
                else torch.cuda.current_device()
            ]

        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(int(cfg["validation_seed"]))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(cfg["validation_seed"]))

            val_metrics, _ = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                cfg=cfg,
                epoch=epoch,
                training=False,
                optimizer=None,
                global_step=global_step,
                boundaries=boundaries,
                max_batches=val_max_batches,
            )

        history.append(
            flatten_record(epoch, global_step, train_metrics, val_metrics)
        )
        write_history(save_dir, history)

        update_best(
            save_dir,
            best_records,
            model,
            optimizer,
            epoch,
            global_step,
            cfg,
            val_metrics,
        )

        # Save latest every epoch.
        save_checkpoint(
            save_dir / "latest_isbi2016.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            cfg=cfg,
            best_records=best_records,
            metrics=val_metrics,
        )

        if epoch % int(cfg["training"]["save_every_epochs"]) == 0 or epoch == epochs:
            save_checkpoint(
                save_dir / f"epoch_{epoch:03d}_isbi2016.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                cfg=cfg,
                best_records=best_records,
                metrics=val_metrics,
            )

        do_rich = (
            mode == "formal"
            and (
                epoch % int(cfg["validation"]["rich_validation_every_epochs"]) == 0
                or epoch == epochs
            )
        )

        if do_rich:
            sample_counts = list(cfg["validation"]["rich_sample_counts"])
            if epoch == epochs:
                sample_counts = sorted(set(sample_counts + [32]))

            with torch.random.fork_rng(devices=cuda_devices, enabled=True):
                torch.manual_seed(int(cfg["validation_seed"]) + 999)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(cfg["validation_seed"]) + 999)

                rich = rich_validation(
                    model,
                    val_loader,
                    device,
                    sample_counts,
                )

            rich_history[str(epoch)] = rich
            (save_dir / "rich_validation_history.json").write_text(
                json.dumps(rich_history, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"valCross={val_metrics.get('prior_cross_dice', float('nan')):.4f} "
            f"valGED={val_metrics.get('prior_ged2', float('nan')):.4f} "
            f"valPostDice={val_metrics.get('prob_dice', float('nan')):.4f} "
            f"ECE={val_metrics.get('evidence_ece', float('nan')):.5f} "
            f"ErrAUROC={val_metrics.get('evidence_error_detection_auroc', float('nan')):.4f}"
        )
        print(
            "  [GMM] "
            f"KL(c/m/f)=({val_metrics['coarse_kl']:.4f},"
            f"{val_metrics['middle_kl']:.4f},"
            f"{val_metrics['fine_kl']:.4f}) "
            f"Keff=({val_metrics['coarse_prior_keff']:.2f},"
            f"{val_metrics['middle_prior_keff']:.2f},"
            f"{val_metrics['fine_prior_keff']:.2f}) "
            f"std=({val_metrics['coarse_prior_std_rms']:.3f},"
            f"{val_metrics['middle_prior_std_rms']:.3f},"
            f"{val_metrics['fine_prior_std_rms']:.3f})"
        )
        print(
            "  [EG-ALC] "
            f"U={val_metrics['relative_demand']:.3f} "
            f"beta=({val_metrics['coarse_beta']:.3f},"
            f"{val_metrics['middle_beta']:.3f},"
            f"{val_metrics['fine_beta']:.3f})"
        )

    if mode == "formal":
        # Explicit named final checkpoint, even though epoch_150 already exists.
        save_checkpoint(
            save_dir / "final_epoch150_isbi2016.pth",
            model=model,
            optimizer=optimizer,
            epoch=epochs,
            global_step=global_step,
            cfg=cfg,
            best_records=best_records,
            metrics=history[-1] if history else {},
        )

    report = quick_screen_report(history) if mode in {"smoke", "quick"} else None
    if report is not None:
        (save_dir / "screening_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("[SCREENING]")
        print(json.dumps(report, indent=2, ensure_ascii=False))

    summary = {
        "experiment": cfg["experiment_name"],
        "mode": mode,
        "completed_epochs": epochs,
        "initialization": initialization_report,
        "best_records": best_records,
        "final_record": history[-1] if history else None,
        "rich_validation_epochs": sorted(int(k) for k in rich_history),
        "screening_report": report,
        "training_finished_beijing": datetime.now(BEIJING).isoformat(timespec="seconds"),
    }

    (save_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (save_dir / "best_records.json").write_text(
        json.dumps(best_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[END Beijing]", datetime.now(BEIJING).isoformat(timespec="seconds"))
    print("[DONE]")
    print(json.dumps(best_records, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
