
"""
EG-GM-PUNet v3 Stage A.2 — Direct Diversity-Matched MS-GMM
===========================================================

目标
----
A.1a 已经修复了 coarse/middle/fine 的极端 KL / latent-std 失衡，
但输出 under-dispersed。
A.1-Final SSDC 失败的原因是 coverage / fidelity / mean 等中心化项
压过 spread calibration。

A.2 不再新增任何结构模块，只保留一个直接、对称、case-wise 的
Distribution Matching Calibration (DMC)：

    L_DMC = SmoothL1(D_pred, D_ann)

其中：
- D_ann：真实 annotator masks 的 mean pairwise (1 - IoU)
- D_pred：prior samples 的 differentiable pairwise soft-Jaccard discrepancy
          sum |p_i-p_j| / sum(p_i+p_j-p_i p_j)

这个定义有两个重要性质：
1. 当输入是 binary masks 时，严格等于 1-IoU；
2. 当两个概率图完全相同时，D_pred=0，
   不会像 naive soft-IoU/soft-Dice 那样把“相同的模糊概率图”误判为有 diversity。

三大论文模块保持：
    MS-GMM + MS-DEH + EG-ALC

初始化：
    A.1a epoch9 best GED checkpoint

正式训练：
    最多 30 epoch
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
import torch.nn.functional as F
import yaml
from torch.optim import Adam
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.lidc_dataset_full import (
    build_lidc_full_dataloader as build_lidc_dataloader,
)
from models.EG_GM_PUNet_v3_A1a import EGGMPUNetV3A1a
from models.Hierarchical_Multi_scale_Gaussian_Mixture_prob_unet_A1a import (
    effective_component_count,
)
from models.Hierarchical_Multi_scale_Gaussian_Mixture_prob_unet_A1a.hierarchical_multiscale_gmm_parts_a1a import (
    sample_posterior_st,
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


def build_model(config: dict[str, Any]) -> EGGMPUNetV3A1a:
    m = config["model"]
    return EGGMPUNetV3A1a(
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
    )


def load_a1a_checkpoint(model, path: str | Path, device) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    obj = torch.load(path, map_location=device)
    if "model_state_dict" not in obj:
        raise KeyError("A.1a checkpoint 缺少 model_state_dict。")

    result = model.load_state_dict(obj["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"A.1a strict load 异常: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    return {
        "checkpoint": str(path),
        "source_epoch": int(obj.get("epoch", -1)),
        "strict_load": True,
        "missing_keys": [],
        "unexpected_keys": [],
    }


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
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
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


def set_probability_stage(model, *, epoch: int, warmup_epochs: int) -> str:
    """A.2 前若干 epoch 仅保护 context encoders。

    与 A.1a 不同，这里 decoder / segmentation head 不冻结，
    因为 DMC 必须允许 prior distribution 与 decoder 联合适配。
    """
    for p in model.probabilistic_model.parameters():
        p.requires_grad_(True)

    if epoch > int(warmup_epochs):
        return "all_probability_trainable"

    modules_to_freeze = [
        model.probabilistic_model.image_encoder,
        model.probabilistic_model.prior.encoder,
        model.probabilistic_model.posterior.encoder,
    ]
    for module in modules_to_freeze:
        for p in module.parameters():
            p.requires_grad_(False)
    return "context_encoders_frozen"


def set_learning_rates(optimizer, probability_lr: float, evidence_lr: float):
    for group in optimizer.param_groups:
        name = group["group_name"]
        if name == "probability":
            group["lr"] = float(probability_lr)
        elif name == "evidence":
            group["lr"] = float(evidence_lr)
        else:
            raise RuntimeError(f"Unknown optimizer group: {name}")


def annotation_diversity_iou(
    all_masks: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """逐病例 mean pairwise (1-IoU)。"""
    masks = all_masks.detach().float()
    if masks.ndim == 5 and masks.shape[2] == 1:
        masks = masks[:, :, 0]
    if masks.ndim != 4:
        raise ValueError(
            f"masks 必须为 [B,R,H,W]/[B,R,1,H,W]，得到 {tuple(all_masks.shape)}"
        )

    b, r = masks.shape[:2]
    if r < 2:
        return torch.zeros((b,), device=masks.device, dtype=masks.dtype)

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


def differentiable_prior_logits(
    model: EGGMPUNetV3A1a,
    state,
    num_samples: int,
) -> torch.Tensor:
    """返回 [S,B,C,H,W] 的 differentiable prior samples。

    mixture basis 用 A.1a 固定 temperature 的 ST Gumbel；
    Gaussian 用 reparameterization。
    正式 evaluation 仍然使用 model.sample() 的 exact mixture sampling。
    """
    if int(num_samples) < 2:
        raise ValueError("DMC 至少需要 2 个 prior samples。")

    outputs = []
    pm = model.probabilistic_model

    for _ in range(int(num_samples)):
        previous_z = {}
        component_indices = {}
        distributions = {}

        for name in pm.group_names:
            p = pm.prior.distribution(
                features=state.prior_context_features,
                group_name=name,
                previous_z=previous_z,
            )
            sample = sample_posterior_st(
                p,
                temperature=pm.mixture_temperature,
            )
            previous_z[name] = sample.z
            component_indices[name] = sample.component_indices
            distributions[name] = p

        decode = pm.decode_from_groups(
            state=state,
            z_groups=previous_z,
            component_indices=component_indices,
            distributions=distributions,
        )
        outputs.append(decode.logits)

    return torch.stack(outputs, dim=0)


def pairwise_jaccard_discrepancy(
    values: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """逐病例 pairwise Jaccard discrepancy。

    values: [S,B,H,W]，可为概率或 binary mask。

    d(a,b) = sum |a-b| / sum(a+b-a*b)

    对 binary masks：
        |a-b| = union - intersection
    因此严格有：
        d(a,b) = 1 - IoU(a,b)

    对完全相同的 soft probability maps：
        numerator=0 -> d=0
    所以不会产生 naive soft-IoU 的伪 diversity。
    """
    if values.ndim != 4:
        raise ValueError("values 必须为 [S,B,H,W]。")
    s, b = values.shape[:2]
    if s < 2:
        return torch.zeros((b,), device=values.device, dtype=values.dtype)

    pairs = []
    for i in range(s):
        a = values[i].flatten(start_dim=1)
        for j in range(i + 1, s):
            c = values[j].flatten(start_dim=1)
            numerator = (a - c).abs().sum(dim=1)
            union = (a + c - a * c).sum(dim=1)
            dist = torch.where(
                union <= eps,
                torch.zeros_like(union),
                numerator / union.clamp_min(eps),
            )
            pairs.append(dist)
    return torch.stack(pairs, dim=1).mean(dim=1)


def direct_diversity_matching_loss(
    prior_logits: torch.Tensor,
    all_masks: torch.Tensor,
    *,
    huber_beta: float = 0.05,
    deadband: float = 0.01,
):
    """Case-wise DMC。

    只匹配 spread，不再添加 coverage / fidelity / mean 项。
    Fidelity 由原 ELBO reconstruction 保持。
    """
    if prior_logits.ndim != 5:
        raise ValueError("prior_logits 必须 [S,B,C,H,W]。")

    masks = all_masks.float()
    if masks.ndim == 5 and masks.shape[2] == 1:
        masks = masks[:, :, 0]
    if masks.ndim != 4:
        raise ValueError("all_masks 必须 [B,R,H,W]/[B,R,1,H,W]。")

    probs = torch.softmax(prior_logits, dim=2)[:, :, 1]  # [S,B,H,W]
    pred_div = pairwise_jaccard_discrepancy(probs)

    r = int(masks.shape[1])
    if r < 2:
        zero = pred_div.sum() * 0.0
        return {
            "loss": zero,
            "pred_div": pred_div.mean(),
            "target_div": zero.detach(),
            "gap": pred_div.mean(),
            "abs_gap": pred_div.abs().mean(),
            "under_fraction": zero.detach(),
            "valid_fraction": zero.detach(),
        }

    target_div = pairwise_jaccard_discrepancy(
        masks.permute(1, 0, 2, 3)
    ).detach()

    raw_gap = pred_div - target_div
    db = float(deadband)
    if db > 0:
        effective_gap = torch.sign(raw_gap) * torch.relu(raw_gap.abs() - db)
    else:
        effective_gap = raw_gap

    loss_case = F.smooth_l1_loss(
        effective_gap,
        torch.zeros_like(effective_gap),
        reduction="none",
        beta=float(huber_beta),
    )

    return {
        "loss": loss_case.mean(),
        "pred_div": pred_div.mean(),
        "target_div": target_div.mean(),
        "gap": raw_gap.mean(),
        "abs_gap": raw_gap.abs().mean(),
        "under_fraction": (raw_gap < -db).float().mean(),
        "valid_fraction": torch.ones((), device=prior_logits.device),
    }


def ramp_value(epoch: int, start_epoch: int, ramp_epochs: int) -> float:
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 1:
        return 1.0
    return float(
        min(
            max((epoch - start_epoch + 1) / float(ramp_epochs), 0.0),
            1.0,
        )
    )


def rankdata(x: np.ndarray) -> np.ndarray:
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


def corr(x, y, spearman=False) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    if spearman:
        a, b = rankdata(a), rankdata(b)
    return float(np.corrcoef(a, b)[0, 1])


def mixture_total_std_rms(distribution) -> torch.Tensor:
    pi = distribution.mixture_probs.unsqueeze(-1)
    mean = (pi * distribution.mu).sum(dim=1, keepdim=True)
    within = (pi * distribution.sigma.pow(2)).sum(dim=1)
    between = (pi * (distribution.mu - mean).pow(2)).sum(dim=1)
    return torch.sqrt((within + between).mean(dim=1).clamp_min(1e-12))


def new_accumulator(model, *, ece_bins: int, ranking_bins: int):
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
        "evidence_loss": 0.0,
        "evidence_metrics": EvidentialMetricAccumulator(
            num_bins=ece_bins,
            ranking_bins=ranking_bins,
        ),
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

        # DMC training diagnostics
        "dmc_updates": 0,
        "dmc_weighted": 0.0,
        "dmc_raw_loss": 0.0,
        "dmc_pred_div": 0.0,
        "dmc_target_div": 0.0,
        "dmc_gap": 0.0,
        "dmc_abs_gap": 0.0,
        "dmc_under_fraction": 0.0,
        "dmc_ramp": 0.0,
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
    acc,
    *,
    output,
    segmentation,
    all_segmentations,
    objective_loss,
    regularization_loss,
    model,
):
    b = int(segmentation.shape[0])
    p = output.probabilistic
    c = output.adaptive_controls
    scores = binary_scores_from_logits(
        p.decode.logits.detach(),
        segmentation,
    )

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
    acc["evidence_loss"] += float(output.evidential_loss.total_loss_mean.detach().item()) * b

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
    result = {
        key: acc[key] / n
        for key in (
            "objective_loss", "joint_loss",
            "probabilistic_loss", "weighted_evidence_loss",
            "regularization_loss", "reconstruction",
            "weighted_kl", "unweighted_kl",
            "prob_dice", "prob_iou",
            "evidence_loss",
            "raw_demand", "relative_demand",
        )
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
            result[f"{name}_{key}"] = acc[f"{name}_{key}"] / n

    result["demand_annotation_pearson"] = corr(
        acc["case_raw_demand"], acc["case_dann_iou"], False
    )
    result["demand_annotation_spearman"] = corr(
        acc["case_raw_demand"], acc["case_dann_iou"], True
    )

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
        result["prior_diversity_gap"] = (
            result["prior_sample_diversity"]
            - result["annotation_diversity_metric"]
        )
        result["prior_diversity_abs_gap"] = abs(
            result["prior_diversity_gap"]
        )

    u = int(acc["dmc_updates"])
    result["dmc_updates"] = u
    if u > 0:
        for key in (
            "dmc_weighted", "dmc_raw_loss",
            "dmc_pred_div", "dmc_target_div",
            "dmc_gap", "dmc_abs_gap",
            "dmc_under_fraction", "dmc_ramp",
        ):
            result[key] = acc[key] / u
    else:
        for key in (
            "dmc_weighted", "dmc_raw_loss",
            "dmc_pred_div", "dmc_target_div",
            "dmc_gap", "dmc_abs_gap",
            "dmc_under_fraction", "dmc_ramp",
        ):
            result[key] = 0.0

    return result


def iter_loader(loader, max_batches):
    if max_batches is None:
        return loader, len(loader)
    count = min(int(max_batches), len(loader))
    return islice(loader, count), count


def run_epoch(
    *,
    model,
    loader,
    device,
    config,
    epoch,
    training,
    optimizer,
    global_step,
    boundaries,
    max_batches=None,
):
    tcfg = config["training"]
    vcfg = config["validation"]
    dcfg = tcfg["direct_diversity_matching"]

    if training:
        model.train()
        probability_stage = set_probability_stage(
            model,
            epoch=epoch,
            warmup_epochs=int(tcfg["context_encoder_warmup_epochs"]),
        )
    else:
        model.eval()
        probability_stage = "validation"

    acc = new_accumulator(
        model,
        ece_bins=int(vcfg["ece_bins"]),
        ranking_bins=int(vcfg["ranking_bins"]),
    )

    iterable, total = iter_loader(loader, max_batches)
    progress = tqdm(
        iterable,
        total=total,
        desc=f"A2 {'Train' if training else 'Val'} {epoch}",
        ncols=210,
    )

    last_plr = 0.0
    last_elr = 0.0

    for batch in progress:
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

        images = batch["image"].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        segmentation = batch["mask"].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        all_segmentations = batch["masks"].to(
            device=device, dtype=torch.float32, non_blocking=True
        )

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
                    str(k): float(v)
                    for k, v in tcfg["base_beta_groups"].items()
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
                # A.1a 已经完成 temporary global balance；
                # A.2 不再继续人为 equalize KL。
                scale_balance_enabled=False,
                update_scale_balance_stats=False,
            )

            if training:
                regularization_loss = (
                    float(tcfg["regularization_weight"])
                    * calculate_l2_regularization(model)
                )
            else:
                regularization_loss = torch.zeros((), device=device)

            objective = output.loss + regularization_loss

            dmc = None
            weighted_dmc = torch.zeros((), device=device)

            do_dmc = (
                training
                and bool(dcfg["enabled"])
                and int(epoch) >= int(dcfg["start_epoch"])
                and global_step % int(dcfg["every_n_steps"]) == 0
            )

            if do_dmc:
                prior_logits = differentiable_prior_logits(
                    model,
                    state,
                    num_samples=int(dcfg["num_prior_samples"]),
                )
                dmc = direct_diversity_matching_loss(
                    prior_logits,
                    all_segmentations,
                    huber_beta=float(dcfg["huber_beta"]),
                    deadband=float(dcfg["deadband"]),
                )
                ramp = ramp_value(
                    epoch,
                    int(dcfg["start_epoch"]),
                    int(dcfg["ramp_epochs"]),
                )

                # 原 ELBO reconstruction 是 pixel-sum 量级，
                # 因此用 H*W 做尺度对齐，再乘一个小的 λ_DMC。
                pixels = float(segmentation.shape[-2] * segmentation.shape[-1])
                weighted_dmc = (
                    float(dcfg["weight"])
                    * pixels
                    * float(ramp)
                    * dmc["loss"]
                )
                objective = objective + weighted_dmc

        if training:
            if not torch.isfinite(objective):
                raise RuntimeError("objective contains NaN/Inf.")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()

            clip = float(tcfg.get("gradient_clip_norm", 0.0))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=clip,
                )

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

        if dmc is not None:
            acc["dmc_updates"] += 1
            acc["dmc_weighted"] += float(weighted_dmc.detach().item())
            acc["dmc_raw_loss"] += float(dmc["loss"].detach().item())
            acc["dmc_pred_div"] += float(dmc["pred_div"].detach().item())
            acc["dmc_target_div"] += float(dmc["target_div"].detach().item())
            acc["dmc_gap"] += float(dmc["gap"].detach().item())
            acc["dmc_abs_gap"] += float(dmc["abs_gap"].detach().item())
            acc["dmc_under_fraction"] += float(dmc["under_fraction"].detach().item())
            acc["dmc_ramp"] += float(ramp)

        if not training:
            sample_logits = model.probabilistic_model.sample(
                state,
                num_samples=int(vcfg["prior_samples_each_epoch"]),
                return_component_indices=False,
            )
            metrics = prior_distribution_metrics(
                sample_logits,
                all_segmentations,
            )
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
        if training and acc["dmc_updates"] > 0:
            postfix["Pdiv"] = f"{acc['dmc_pred_div']/acc['dmc_updates']:.3f}"
            postfix["Tdiv"] = f"{acc['dmc_target_div']/acc['dmc_updates']:.3f}"
        if not training and acc["prior_metric_samples"] > 0:
            postfix["GED4"] = f"{acc['prior_ged2']/acc['prior_metric_samples']:.4f}"
            postfix["Div4"] = f"{acc['prior_sample_diversity']/acc['prior_metric_samples']:.3f}"
        progress.set_postfix(postfix)

    result = finish_accumulator(acc, model)
    result["probability_stage"] = probability_stage
    result["probabilistic_learning_rate"] = last_plr
    result["evidence_learning_rate"] = last_elr
    return result, global_step


def flatten_record(epoch, global_step, train_metrics, val_metrics):
    row = {"epoch": epoch, "global_step": global_step}
    for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[f"{prefix}_{k}"] = v
    return row


def write_history(save_dir: Path, history):
    (save_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not history:
        return
    keys, seen = [], set()
    for row in history:
        for k in row:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with open(
        save_dir / "history.csv",
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)


def best_records_template():
    return {
        "val_prior_ged4": {
            "value": float("inf"),
            "direction": "min",
            "checkpoint": "best_val_ged4_eg_gm_punet_v3_a2.pth",
            "epoch": None,
        },
        "val_distribution_gap": {
            "value": float("inf"),
            "direction": "min",
            "checkpoint": "best_val_distribution_gap_eg_gm_punet_v3_a2.pth",
            "epoch": None,
        },
        "val_distribution_score": {
            "value": float("inf"),
            "direction": "min",
            "checkpoint": "best_val_distribution_score_eg_gm_punet_v3_a2.pth",
            "epoch": None,
        },
        "val_probabilistic_dice": {
            "value": float("-inf"),
            "direction": "max",
            "checkpoint": "best_val_probabilistic_dice_eg_gm_punet_v3_a2.pth",
            "epoch": None,
        },
    }


def update_best_records(
    save_dir,
    best_records,
    *,
    model,
    optimizer,
    epoch,
    global_step,
    cfg,
    val_metrics,
):
    ged = float(val_metrics.get("prior_ged2", float("inf")))
    gap = float(val_metrics.get("prior_diversity_abs_gap", float("inf")))

    # GED 为主，分布 gap 作为次级约束。
    # 0.25 的量级使 gap=0.10 只增加 0.025，不会喧宾夺主。
    distribution_score = ged + 0.25 * gap

    candidates = {
        "val_prior_ged4": ged,
        "val_distribution_gap": gap,
        "val_distribution_score": distribution_score,
        "val_probabilistic_dice": float(val_metrics["prob_dice"]),
    }

    for name, value in candidates.items():
        rec = best_records[name]
        better = (
            value < rec["value"]
            if rec["direction"] == "min"
            else value > rec["value"]
        )
        if better:
            rec["value"] = value
            rec["epoch"] = int(epoch)
            save_checkpoint(
                save_dir / rec["checkpoint"],
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                config=cfg,
                best_records=best_records,
                metrics=val_metrics,
            )

    return distribution_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/eg_gm_punet_v3_a2_direct_diversity_match_lidc.yaml",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(config_path)

    if args.smoke_test:
        cfg = copy.deepcopy(cfg)
        cfg["training"]["epochs"] = 1
        cfg["training"]["context_encoder_warmup_epochs"] = 0
        cfg["training"]["direct_diversity_matching"]["start_epoch"] = 1
        cfg["training"]["direct_diversity_matching"]["ramp_epochs"] = 1
        cfg["training"]["feedback_enabled"] = False
        cfg["model"]["feedback"]["enabled"] = False

    set_seed(int(cfg["seed"]))
    device = prepare_device(cfg.get("device", "cuda"))

    save_dir = Path(cfg["save_dir"])
    if args.smoke_test:
        save_dir = save_dir / "smoke_test"
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_lidc_dataloader(
        root=cfg["data_root"],
        split="train",
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"]["num_workers"]),
        shuffle=True,
        random_rater=True,
    )
    val_loader = build_lidc_dataloader(
        root=cfg["data_root"],
        split="val",
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"]["num_workers"]),
        shuffle=False,
        random_rater=False,
        rater_index=int(cfg["validation"]["rater_index"]),
    )

    model = build_model(cfg).to(device)

    init_report = None
    if not args.resume:
        init = cfg["initialization"]
        if init.get("mode") != "a1a_best_ged":
            raise ValueError("A.2 固定从 A.1a best GED checkpoint 初始化。")
        init_report = load_a1a_checkpoint(
            model,
            init["checkpoint"],
            device,
        )
        print("[INIT] A.1a best-GED -> A.2:")
        print(json.dumps(init_report, indent=2, ensure_ascii=False))

        expected = init.get("expected_source_epoch")
        if expected is not None and int(init_report["source_epoch"]) != int(expected):
            raise RuntimeError(
                f"期望 A.1a epoch={expected}，实际={init_report['source_epoch']}。"
            )

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
    )

    epochs = int(cfg["training"]["epochs"])
    train_max_batches = 2 if args.smoke_test else None
    val_max_batches = 2 if args.smoke_test else None

    total_steps = epochs * len(train_loader)
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
            rp,
            model=model,
            optimizer=optimizer,
            device=device,
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
            "temporary_scale_balance_reused_from_A1a_checkpoint": True,
            "scale_balance_during_A2": False,
            "direct_casewise_diversity_matching": True,
            "dmc_metric": "pairwise soft-Jaccard discrepancy; binary limit equals 1-IoU",
            "coverage_loss": False,
            "fidelity_set_loss": False,
            "mean_set_loss": False,
            "scale_specific_demand": False,
            "single_annotation_compatible": True,
        },
    }
    (save_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 116)
    print("EG-GM-PUNet v3 Stage A.2 — Direct Diversity-Matched MS-GMM")
    print("A.1a backbone + DMC + MS-DEH + bounded EG-ALC")
    print("=" * 116)
    print(f"Device: {device}")
    print(f"Train/Val: {len(train_loader.dataset)}/{len(val_loader.dataset)}")
    print(f"Output: {save_dir}")
    print(
        "[START Beijing]",
        datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
    )

    for epoch in range(start_epoch, epochs + 1):
        train_metrics, global_step = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            config=cfg,
            epoch=epoch,
            training=True,
            optimizer=optimizer,
            global_step=global_step,
            boundaries=boundaries,
            max_batches=train_max_batches,
        )

        # 固定 validation RNG，使各 epoch 的 4-sample GED/Div 更可比；
        # fork_rng 结束后恢复训练 RNG，不影响下一 epoch 训练随机性。
        cuda_devices = []
        if torch.cuda.is_available() and device.type == "cuda":
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]

        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(int(cfg["validation_seed"]))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(cfg["validation_seed"]))
            val_metrics, _ = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                config=cfg,
                epoch=epoch,
                training=False,
                optimizer=None,
                global_step=global_step,
                boundaries=boundaries,
                max_batches=val_max_batches,
            )

        history.append(
            flatten_record(
                epoch,
                global_step,
                train_metrics,
                val_metrics,
            )
        )
        write_history(save_dir, history)

        score = update_best_records(
            save_dir,
            best_records,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            cfg=cfg,
            val_metrics=val_metrics,
        )

        save_checkpoint(
            save_dir / "latest_eg_gm_punet_v3_a2.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            config=cfg,
            best_records=best_records,
            metrics=val_metrics,
        )

        if epoch % int(cfg["training"]["save_every_epochs"]) == 0:
            save_checkpoint(
                save_dir / f"epoch_{epoch:03d}_eg_gm_punet_v3_a2.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                config=cfg,
                best_records=best_records,
                metrics=val_metrics,
            )

        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"val_GED4={val_metrics.get('prior_ged2', float('nan')):.4f} "
            f"val_Cross={val_metrics.get('prior_cross_dice', float('nan')):.4f} "
            f"val_Best={val_metrics.get('prior_best_dice', float('nan')):.4f} "
            f"val_Div4={val_metrics.get('prior_sample_diversity', float('nan')):.4f} "
            f"AnnDiv={val_metrics.get('annotation_diversity_metric', float('nan')):.4f} "
            f"|gap|={val_metrics.get('prior_diversity_abs_gap', float('nan')):.4f} "
            f"score={score:.4f}"
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

        if train_metrics["dmc_updates"] > 0:
            print(
                "  [DMC] "
                f"ramp={train_metrics['dmc_ramp']:.2f} "
                f"predDiv={train_metrics['dmc_pred_div']:.4f} "
                f"targetDiv={train_metrics['dmc_target_div']:.4f} "
                f"gap={train_metrics['dmc_gap']:+.4f} "
                f"|gap|={train_metrics['dmc_abs_gap']:.4f} "
                f"under={train_metrics['dmc_under_fraction']:.3f} "
                f"weighted={train_metrics['dmc_weighted']:.2f}"
            )

    summary = {
        "experiment": cfg["experiment_name"],
        "completed_epochs": int(epochs),
        "best_records": best_records,
        "recommended_probability_checkpoints": {
            "primary": "best_val_ged4_eg_gm_punet_v3_a2.pth",
            "distribution_calibrated": "best_val_distribution_score_eg_gm_punet_v3_a2.pth",
            "minimum_diversity_gap": "best_val_distribution_gap_eg_gm_punet_v3_a2.pth",
        },
        "final_epoch": history[-1] if history else None,
        "architecture": manifest["notes"],
    }
    (save_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (save_dir / "best_records.json").write_text(
        json.dumps(best_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        "[END Beijing]",
        datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
    )
    print("[DONE] Best records:")
    print(json.dumps(best_records, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
