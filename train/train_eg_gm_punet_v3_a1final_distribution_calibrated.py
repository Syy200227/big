
"""
EG-GM-PUNet v3 Stage A.1-Final — 150 Epoch Overnight Candidate
===============================================================

核心：
- 初始化：A.1a epoch9 best GED checkpoint
- MS-GMM：prior-anchored residual GMM + temporary scale KL balance
- Decoder：middle/fine centered stochastic residual bridge，端到端联合训练
- Training-only SSDC：
    coverage + fidelity + spread calibration + mean consistency
- MS-DEH：保留
- EG-ALC：epoch21 开始 bounded feedback，沿用 validated shared demand

推荐：
    --smoke-test 先跑 1 epoch / 2 batches
    正式训练直接不带其它参数，跑满 150 epoch。
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
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

from dataset.lidc_dataset_full import (
    build_lidc_full_dataloader as build_lidc_dataloader,
)
from models.EG_GM_PUNet_v3_A1final import (
    EGGMPUNetV3A1final,
    symmetric_set_distribution_loss,
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


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ValueError("config 必须为 dict。")
    return value


def build_model(config: dict[str, Any]) -> EGGMPUNetV3A1final:
    m = config["model"]
    b = m["scale_residual_bridge"]
    return EGGMPUNetV3A1final(
        in_channels=int(m["in_channels"]),
        num_classes=int(m["num_classes"]),
        num_components={str(k): int(v) for k, v in m["num_components"].items()},
        latent_dims={str(k): int(v) for k, v in m["latent_dims"].items()},
        num_channels=tuple(int(v) for v in m["num_channels"]),
        group_feature_indices={
            str(k): int(v) for k, v in m["group_feature_indices"].items()
        },
        group_decoder_indices={
            str(k): int(v) for k, v in m["group_decoder_indices"].items()
        },
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
        bridge_enabled_scales=tuple(str(x) for x in b["enabled_scales"]),
        bridge_hidden_ratio=float(b["hidden_ratio"]),
        bridge_residual_scale=float(b["residual_scale"]),
    )


def save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    epoch,
    global_step,
    config,
    best_records,
    metrics,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "best_records": best_records,
            "metrics": metrics,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None
                ),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        },
        path,
    )


def restore_checkpoint(path, *, model, optimizer, device):
    obj = torch.load(path, map_location=device)
    model.load_state_dict(obj["model_state_dict"], strict=True)
    optimizer.load_state_dict(obj["optimizer_state_dict"])
    return obj


def annotation_diversity_iou(all_masks: torch.Tensor, eps: float = 1e-7):
    masks = all_masks.detach().float()
    if masks.ndim == 5 and masks.shape[2] == 1:
        masks = masks[:, :, 0]
    b, r = masks.shape[:2]
    if r < 2:
        return torch.zeros((b,), device=masks.device)
    values = []
    for i in range(r):
        a = masks[:, i].reshape(b, -1)
        for j in range(i + 1, r):
            c = masks[:, j].reshape(b, -1)
            inter = (a * c).sum(dim=1)
            union = ((a + c) > 0).float().sum(dim=1)
            iou = torch.where(
                union <= eps,
                torch.ones_like(union),
                inter / union.clamp_min(eps),
            )
            values.append(1.0 - iou)
    return torch.stack(values, dim=1).mean(dim=1)


def rankdata(x: np.ndarray):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def corr(x, y, spearman=False):
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    if spearman:
        a, b = rankdata(a), rankdata(b)
    return float(np.corrcoef(a, b)[0, 1])


def mixture_total_std_rms(distribution):
    pi = distribution.mixture_probs.unsqueeze(-1)
    mean = (pi * distribution.mu).sum(dim=1, keepdim=True)
    within = (pi * distribution.sigma.pow(2)).sum(dim=1)
    between = (pi * (distribution.mu - mean).pow(2)).sum(dim=1)
    return torch.sqrt((within + between).mean(dim=1).clamp_min(1e-12))


def set_probability_stage(model, epoch: int, warmup_epochs: int):
    # 默认全部 trainable
    for p in model.probabilistic_model.parameters():
        p.requires_grad_(True)

    if epoch > warmup_epochs:
        return "all_probability_trainable"

    # 前 5 epoch 保护 encoder context；但 decoder / GMM / bridge 已可联合适配。
    freeze_modules = [
        model.probabilistic_model.image_encoder,
        model.probabilistic_model.prior.encoder,
        model.probabilistic_model.posterior.encoder,
    ]
    for module in freeze_modules:
        for p in module.parameters():
            p.requires_grad_(False)
    return "context_encoders_frozen"


def distribution_ramp(epoch: int, start_epoch: int, ramp_epochs: int) -> float:
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 1:
        return 1.0
    return float(min(max((epoch - start_epoch + 1) / float(ramp_epochs), 0.0), 1.0))


def set_learning_rates(optimizer, base_lr, bridge_lr, evidence_lr):
    for group in optimizer.param_groups:
        name = group["group_name"]
        if name == "probability_base":
            group["lr"] = float(base_lr)
        elif name == "bridge":
            group["lr"] = float(bridge_lr)
        elif name == "evidence":
            group["lr"] = float(evidence_lr)
        else:
            raise RuntimeError(f"Unknown optimizer group: {name}")


def build_optimizer(model, config):
    t = config["training"]
    bridge_ids = {id(p) for p in model.probabilistic_model.scale_residual_bridge.parameters()}
    base_params = [
        p for p in model.probabilistic_model.parameters()
        if id(p) not in bridge_ids
    ]
    bridge_params = list(model.probabilistic_model.scale_residual_bridge.parameters())

    return Adam(
        [
            {
                "params": base_params,
                "lr": float(t["probabilistic_learning_rate_values"][0]),
                "group_name": "probability_base",
            },
            {
                "params": bridge_params,
                "lr": float(t["bridge_learning_rate_values"][0]),
                "group_name": "bridge",
            },
            {
                "params": model.evidence_head.parameters(),
                "lr": float(t["evidence_learning_rate_values"][0]),
                "group_name": "evidence",
            },
        ],
        weight_decay=0.0,
    )


def new_accumulator(model, ece_bins, ranking_bins):
    acc = {
        "samples": 0,
        "objective_loss": 0.0,
        "joint_loss": 0.0,
        "probabilistic_loss": 0.0,
        "regularization_loss": 0.0,
        "reconstruction": 0.0,
        "weighted_kl": 0.0,
        "unweighted_kl": 0.0,
        "prob_dice": 0.0,
        "prob_iou": 0.0,
        "prior_ged2": 0.0,
        "prior_cross_dice": 0.0,
        "prior_best_dice": 0.0,
        "prior_sample_diversity": 0.0,
        "annotation_diversity_metric": 0.0,
        "prior_metric_samples": 0,
        "raw_demand": 0.0,
        "relative_demand": 0.0,
        "case_raw_demand": [],
        "case_dann_iou": [],
        "ssdc_updates": 0,
        "ssdc_weighted": 0.0,
        "ssdc_coverage": 0.0,
        "ssdc_fidelity": 0.0,
        "ssdc_spread": 0.0,
        "ssdc_mean": 0.0,
        "ssdc_pred_div": 0.0,
        "ssdc_target_div": 0.0,
        "ssdc_ramp": 0.0,
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
    objective_loss, regularization_loss, model,
):
    b = int(segmentation.shape[0])
    p = output.probabilistic
    c = output.adaptive_controls
    scores = binary_scores_from_logits(p.decode.logits.detach(), segmentation)

    acc["samples"] += b
    acc["objective_loss"] += float(objective_loss.detach().item()) * b
    acc["joint_loss"] += float(output.loss.detach().item()) * b
    acc["probabilistic_loss"] += float(output.probabilistic_loss.detach().item()) * b
    acc["regularization_loss"] += float(regularization_loss.detach().item()) * b
    acc["reconstruction"] += float(p.reconstruction_loss_sum.detach().item()) * b
    acc["weighted_kl"] += float(p.weighted_kl_mean.detach().item()) * b
    acc["unweighted_kl"] += float(p.kl_total_unweighted_mean.detach().item()) * b
    acc["prob_dice"] += float(scores["dice"].sum().item())
    acc["prob_iou"] += float(scores["iou"].sum().item())

    acc["raw_demand"] += float(c.raw_demand.detach().sum().item())
    acc["relative_demand"] += float(c.ambiguity_demand.detach().sum().item())
    dann = annotation_diversity_iou(all_segmentations)
    acc["case_raw_demand"].extend(c.raw_demand.detach().cpu().tolist())
    acc["case_dann_iou"].extend(dann.detach().cpu().tolist())

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
        acc[f"{name}_prior_keff"] += float(torch.exp(
            -(pr.mixture_probs.clamp_min(1e-12) * pr.mixture_probs.clamp_min(1e-12).log()).sum(dim=1)
        ).sum().item())
        acc[f"{name}_posterior_keff"] += float(torch.exp(
            -(q.mixture_probs.clamp_min(1e-12) * q.mixture_probs.clamp_min(1e-12).log()).sum(dim=1)
        ).sum().item())
        acc[f"{name}_alpha"] += float(p.scale_balance_alpha[name].detach().item()) * b
        acc[f"{name}_prior_std_rms"] += float(mixture_total_std_rms(pr).detach().sum().item())
        acc[f"{name}_posterior_std_rms"] += float(mixture_total_std_rms(q).detach().sum().item())
        acc[f"{name}_kl_rate"] += float(p.dimension_normalized_kl_mean[name].detach().item()) * b


def finish_accumulator(acc, model):
    n = max(int(acc["samples"]), 1)
    result = {}
    scalar_keys = [
        "objective_loss","joint_loss","probabilistic_loss","regularization_loss",
        "reconstruction","weighted_kl","unweighted_kl","prob_dice","prob_iou",
        "raw_demand","relative_demand",
    ]
    for k in scalar_keys:
        result[k] = acc[k] / n

    for name in model.group_names:
        for key in (
            "kl","kl_cat","kl_gauss","capacity","error","deficit","surplus","beta",
            "raw_capacity_kl","prior_keff","posterior_keff","alpha",
            "prior_std_rms","posterior_std_rms","kl_rate",
        ):
            result[f"{name}_{key}"] = acc[f"{name}_{key}"] / n

    result["demand_annotation_pearson"] = corr(
        acc["case_raw_demand"], acc["case_dann_iou"], False
    )
    result["demand_annotation_spearman"] = corr(
        acc["case_raw_demand"], acc["case_dann_iou"], True
    )

    m = int(acc["prior_metric_samples"])
    if m > 0:
        for key in (
            "prior_ged2","prior_cross_dice","prior_best_dice",
            "prior_sample_diversity","annotation_diversity_metric",
        ):
            result[key] = acc[key] / m

    u = max(int(acc["ssdc_updates"]), 1)
    result["ssdc_updates"] = int(acc["ssdc_updates"])
    for key in (
        "ssdc_weighted","ssdc_coverage","ssdc_fidelity","ssdc_spread",
        "ssdc_mean","ssdc_pred_div","ssdc_target_div","ssdc_ramp",
    ):
        result[key] = acc[key] / u if acc["ssdc_updates"] > 0 else 0.0

    return result


def iter_loader(loader, max_batches):
    if max_batches is None:
        return loader, len(loader)
    count = min(int(max_batches), len(loader))
    return islice(loader, count), count


def run_epoch(
    *,
    model, loader, device, config, epoch, training,
    optimizer, global_step, boundaries, max_batches=None,
):
    tcfg = config["training"]
    vcfg = config["validation"]
    dcfg = tcfg["distribution_calibration"]

    if training:
        model.train()
        stage = set_probability_stage(
            model,
            epoch=epoch,
            warmup_epochs=int(tcfg["context_encoder_warmup_epochs"]),
        )
    else:
        model.eval()
        stage = "validation"

    acc = new_accumulator(
        model,
        ece_bins=int(vcfg["ece_bins"]),
        ranking_bins=int(vcfg["ranking_bins"]),
    )
    iterable, total = iter_loader(loader, max_batches)
    progress = tqdm(
        iterable, total=total,
        desc=f"A1Final {'Train' if training else 'Val'} {epoch}",
        ncols=210,
    )

    last_base_lr = last_bridge_lr = last_evidence_lr = 0.0

    for batch_idx, batch in enumerate(progress):
        if training:
            base_lr = piecewise_learning_rate(
                global_step, boundaries,
                [float(v) for v in tcfg["probabilistic_learning_rate_values"]],
            )
            bridge_lr = piecewise_learning_rate(
                global_step, boundaries,
                [float(v) for v in tcfg["bridge_learning_rate_values"]],
            )
            evidence_lr = piecewise_learning_rate(
                global_step, boundaries,
                [float(v) for v in tcfg["evidence_learning_rate_values"]],
            )
            set_learning_rates(optimizer, base_lr, bridge_lr, evidence_lr)
            last_base_lr, last_bridge_lr, last_evidence_lr = (
                base_lr, bridge_lr, evidence_lr
            )

        images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        segmentation = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)
        all_segmentations = batch["masks"].to(device=device, dtype=torch.float32, non_blocking=True)

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            state = model(
                image=images,
                segmentation=segmentation,
                is_training=True,
            )

            output = model.elbo(
                state=state,
                segmentation=segmentation,
                all_segmentations=all_segmentations,
                base_beta_groups={
                    str(k): float(v) for k, v in tcfg["base_beta_groups"].items()
                },
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

            # --------------------------------------------------------------
            # Training-only SSDC：每 N 个 step 做一次，降低 overnight 开销。
            # --------------------------------------------------------------
            ssdc_weighted = torch.zeros((), device=device)
            do_ssdc = (
                training
                and bool(dcfg["enabled"])
                and int(epoch) >= int(dcfg["start_epoch"])
                and (global_step % int(dcfg["every_n_steps"]) == 0)
            )

            if do_ssdc:
                decodes, _ = model.probabilistic_model.sample_train_paths(
                    state=state,
                    num_samples=int(dcfg["num_prior_samples"]),
                )
                prior_logits = torch.stack([d.logits for d in decodes], dim=0)

                ssdc = symmetric_set_distribution_loss(
                    prior_logits,
                    all_segmentations,
                    softmin_temperature=float(dcfg["softmin_temperature"]),
                    spread_huber_beta=float(dcfg["spread_huber_beta"]),
                )

                ramp = distribution_ramp(
                    epoch,
                    int(dcfg["start_epoch"]),
                    int(dcfg["ramp_epochs"]),
                )
                pixels = float(segmentation.shape[-2] * segmentation.shape[-1])

                ssdc_weighted = ramp * pixels * (
                    float(dcfg["coverage_weight"]) * ssdc.coverage
                    + float(dcfg["fidelity_weight"]) * ssdc.fidelity
                    + float(dcfg["spread_weight"]) * ssdc.spread
                    + float(dcfg["mean_weight"]) * ssdc.mean_consistency
                )
                objective = objective + ssdc_weighted

        if training:
            if not torch.isfinite(objective):
                raise RuntimeError("objective contains NaN/Inf.")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()

            clip = float(tcfg.get("gradient_clip_norm", 0.0))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

            optimizer.step()
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

        if do_ssdc:
            acc["ssdc_updates"] += 1
            acc["ssdc_weighted"] += float(ssdc_weighted.detach().item())
            acc["ssdc_coverage"] += float(ssdc.coverage.detach().item())
            acc["ssdc_fidelity"] += float(ssdc.fidelity.detach().item())
            acc["ssdc_spread"] += float(ssdc.spread.detach().item())
            acc["ssdc_mean"] += float(ssdc.mean_consistency.detach().item())
            acc["ssdc_pred_div"] += float(ssdc.predicted_diversity.detach().item())
            acc["ssdc_target_div"] += float(ssdc.target_diversity.detach().item())
            acc["ssdc_ramp"] += float(ramp)

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
            "bC": f"{acc['coarse_beta']/completed:.3f}",
            "bM": f"{acc['middle_beta']/completed:.3f}",
            "bF": f"{acc['fine_beta']/completed:.3f}",
        }
        if training and acc["ssdc_updates"] > 0:
            postfix["Sdiv"] = f"{acc['ssdc_pred_div']/acc['ssdc_updates']:.3f}"
            postfix["Tdiv"] = f"{acc['ssdc_target_div']/acc['ssdc_updates']:.3f}"
        if not training and acc["prior_metric_samples"] > 0:
            postfix["GED4"] = f"{acc['prior_ged2']/acc['prior_metric_samples']:.4f}"
            postfix["Div4"] = f"{acc['prior_sample_diversity']/acc['prior_metric_samples']:.3f}"
        progress.set_postfix(postfix)

    result = finish_accumulator(acc, model)
    result["probability_stage"] = stage
    result["probabilistic_learning_rate"] = last_base_lr
    result["bridge_learning_rate"] = last_bridge_lr
    result["evidence_learning_rate"] = last_evidence_lr
    return result, global_step


def flatten_record(epoch, global_step, train_metrics, val_metrics):
    row = {"epoch": epoch, "global_step": global_step}
    for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[f"{prefix}_{k}"] = v
    return row


def write_history(save_dir, history):
    (save_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not history:
        return
    keys, seen = [], set()
    for row in history:
        for k in row:
            if k not in seen:
                keys.append(k); seen.add(k)
    with open(save_dir / "history.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)


def best_records_template():
    return {
        "val_joint_loss": {
            "value": float("inf"), "direction": "min",
            "checkpoint": "best_val_joint_loss_eg_gm_punet_v3.pth", "epoch": None,
        },
        "val_prior_ged4": {
            "value": float("inf"), "direction": "min",
            "checkpoint": "best_val_ged4_eg_gm_punet_v3.pth", "epoch": None,
        },
        "val_probabilistic_dice": {
            "value": float("-inf"), "direction": "max",
            "checkpoint": "best_val_probabilistic_dice_eg_gm_punet_v3.pth", "epoch": None,
        },
    }


def update_best_records(save_dir, best_records, *, model, optimizer, epoch, global_step, cfg, val_metrics):
    candidates = {
        "val_joint_loss": float(val_metrics["joint_loss"]),
        "val_prior_ged4": float(val_metrics.get("prior_ged2", float("inf"))),
        "val_probabilistic_dice": float(val_metrics["prob_dice"]),
    }
    for name, value in candidates.items():
        rec = best_records[name]
        better = value < rec["value"] if rec["direction"] == "min" else value > rec["value"]
        if better:
            rec["value"] = value
            rec["epoch"] = int(epoch)
            save_checkpoint(
                save_dir / rec["checkpoint"],
                model=model, optimizer=optimizer, epoch=epoch,
                global_step=global_step, config=cfg,
                best_records=best_records, metrics=val_metrics,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/eg_gm_punet_v3_a1final_lidc.yaml",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = load_config(cfg_path)

    if args.smoke_test:
        cfg = copy.deepcopy(cfg)
        cfg["training"]["epochs"] = 1
        cfg["training"]["context_encoder_warmup_epochs"] = 0
        cfg["training"]["distribution_calibration"]["start_epoch"] = 1
        cfg["training"]["distribution_calibration"]["ramp_epochs"] = 1
        # smoke 时 feedback 关闭，避免 controller warm-up 干扰代码检查
        cfg["training"]["feedback_enabled"] = False
        cfg["model"]["feedback"]["enabled"] = False

    set_seed(int(cfg["seed"]))
    device = prepare_device(cfg.get("device", "cuda"))

    save_dir = Path(cfg["save_dir"])
    if args.smoke_test:
        save_dir = save_dir / "smoke_test"
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_lidc_dataloader(
        root=cfg["data_root"], split="train",
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"]["num_workers"]),
        shuffle=True, random_rater=True,
    )
    val_loader = build_lidc_dataloader(
        root=cfg["data_root"], split="val",
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"]["num_workers"]),
        shuffle=False, random_rater=False,
        rater_index=int(cfg["validation"]["rater_index"]),
    )

    model = build_model(cfg).to(device)

    init_report = None
    if not args.resume:
        init = cfg["initialization"]
        if init.get("mode") != "a1a_best_ged":
            raise ValueError("A.1-Final 固定从 A.1a best GED 初始化。")
        init_report = model.load_a1a_checkpoint(
            init["checkpoint"], map_location=device
        )
        print("[INIT] A.1a best-GED -> A.1-Final:")
        print(json.dumps(init_report, indent=2, ensure_ascii=False))

        expected = init.get("expected_source_epoch")
        if expected is not None and int(init_report["source_epoch"]) != int(expected):
            raise RuntimeError(
                f"期望 A.1a epoch={expected}，实际={init_report['source_epoch']}。"
            )

        norms = model.probabilistic_model.scale_residual_bridge.output_projection_l2_norms()
        print("[INIT] zero-init bridge output projection norms:", norms)
        if any(abs(float(v)) > 1e-10 for v in norms.values()):
            raise RuntimeError("新 bridge 必须从 zero-output 初始化。")

    optimizer = build_optimizer(model, cfg)

    configured_epochs = int(cfg["training"]["epochs"])
    epochs = configured_epochs
    train_max_batches = 2 if args.smoke_test else None
    val_max_batches = 2 if args.smoke_test else None

    total_steps = configured_epochs * len(train_loader)
    boundaries = build_step_boundaries(
        total_steps,
        cfg["training"]["learning_rate_boundary_fractions"],
    )

    start_epoch = 1
    global_step = 0
    history = []
    best_records = best_records_template()

    resume = args.resume or cfg.get("resume")
    if resume:
        rp = Path(resume)
        if not rp.is_absolute():
            rp = PROJECT_ROOT / rp
        obj = restore_checkpoint(
            rp, model=model, optimizer=optimizer, device=device
        )
        start_epoch = int(obj["epoch"]) + 1
        global_step = int(obj.get("global_step", 0))
        best_records = obj.get("best_records", best_records)
        hp = save_dir / "history.json"
        if hp.exists():
            history = json.loads(hp.read_text(encoding="utf-8"))

    manifest = {
        "config": cfg,
        "initialization_report": init_report,
        "notes": {
            "top_level": "MS-GMM + MS-DEH + EG-ALC",
            "shared_semantic_k": False,
            "component_semantics": "latent density basis only",
            "prior_anchored_residual_posterior": True,
            "temporary_scale_kl_balance": True,
            "centered_middle_fine_bridge_joint_training": True,
            "training_only_ssdc": True,
            "scale_specific_demand": False,
            "validated_shared_demand": "fine_foreground_expected_data_entropy",
        },
    }
    (save_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 116)
    print("EG-GM-PUNet v3 Stage A.1-Final — Distribution-Calibrated Joint Training")
    print("MS-GMM + MS-DEH + bounded EG-ALC; 150-epoch overnight candidate")
    print("=" * 116)
    print(f"Device: {device}")
    print(f"Train/Val: {len(train_loader.dataset)}/{len(val_loader.dataset)}")
    print(f"Output: {save_dir}")
    print("[START Beijing]", datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"))

    for epoch in range(start_epoch, epochs + 1):
        train_metrics, global_step = run_epoch(
            model=model, loader=train_loader, device=device,
            config=cfg, epoch=epoch, training=True,
            optimizer=optimizer, global_step=global_step,
            boundaries=boundaries, max_batches=train_max_batches,
        )
        val_metrics, _ = run_epoch(
            model=model, loader=val_loader, device=device,
            config=cfg, epoch=epoch, training=False,
            optimizer=None, global_step=global_step,
            boundaries=boundaries, max_batches=val_max_batches,
        )

        history.append(flatten_record(
            epoch, global_step, train_metrics, val_metrics
        ))
        write_history(save_dir, history)

        update_best_records(
            save_dir, best_records,
            model=model, optimizer=optimizer,
            epoch=epoch, global_step=global_step,
            cfg=cfg, val_metrics=val_metrics,
        )

        save_checkpoint(
            save_dir / "latest_eg_gm_punet_v3.pth",
            model=model, optimizer=optimizer, epoch=epoch,
            global_step=global_step, config=cfg,
            best_records=best_records, metrics=val_metrics,
        )

        if epoch % int(cfg["training"]["save_every_epochs"]) == 0:
            save_checkpoint(
                save_dir / f"epoch_{epoch:03d}_eg_gm_punet_v3.pth",
                model=model, optimizer=optimizer, epoch=epoch,
                global_step=global_step, config=cfg,
                best_records=best_records, metrics=val_metrics,
            )

        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"train_pDice={train_metrics['prob_dice']:.4f} "
            f"val_pDice={val_metrics['prob_dice']:.4f} "
            f"val_GED4={val_metrics.get('prior_ged2', float('nan')):.4f} "
            f"val_Cross={val_metrics.get('prior_cross_dice', float('nan')):.4f} "
            f"val_Best={val_metrics.get('prior_best_dice', float('nan')):.4f} "
            f"val_Div4={val_metrics.get('prior_sample_diversity', float('nan')):.4f}"
        )
        print(
            "  [MECH] "
            f"KL=({val_metrics['coarse_kl']:.4g},"
            f"{val_metrics['middle_kl']:.4g},"
            f"{val_metrics['fine_kl']:.4g}) "
            f"std=({val_metrics['coarse_prior_std_rms']:.3f},"
            f"{val_metrics['middle_prior_std_rms']:.3f},"
            f"{val_metrics['fine_prior_std_rms']:.3f}) "
            f"beta=({val_metrics['coarse_beta']:.3f},"
            f"{val_metrics['middle_beta']:.3f},"
            f"{val_metrics['fine_beta']:.3f})"
        )
        if train_metrics.get("ssdc_updates", 0) > 0:
            print(
                "  [SSDC] "
                f"ramp={train_metrics['ssdc_ramp']:.2f} "
                f"predDiv={train_metrics['ssdc_pred_div']:.4f} "
                f"targetDiv={train_metrics['ssdc_target_div']:.4f} "
                f"coverage={train_metrics['ssdc_coverage']:.4f} "
                f"fidelity={train_metrics['ssdc_fidelity']:.4f} "
                f"spread={train_metrics['ssdc_spread']:.4f} "
                f"weighted={train_metrics['ssdc_weighted']:.2f}"
            )

    summary = {
        "experiment": cfg["experiment_name"],
        "completed_epochs": int(epochs),
        "best_records": best_records,
        "probability_evaluation_checkpoint": "best_val_ged4_eg_gm_punet_v3.pth",
        "final_epoch": history[-1] if history else None,
        "architecture": manifest["notes"],
    }
    (save_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (save_dir / "best_records.json").write_text(
        json.dumps(best_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[END Beijing]", datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"))
    print("[DONE] Best records:")
    print(json.dumps(best_records, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
