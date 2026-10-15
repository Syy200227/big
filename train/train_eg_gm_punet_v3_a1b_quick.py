"""
EG-GM-PUNet v3 Stage A.1a Quick Screen
======================
Hierarchical Scale-Decoupled MS-GMM + MS-DEH + Continuous Bounded EG-ALC

运行：
    python train/train_eg_gm_punet_v3_hierarchical_ms_gmm.py ^
      --config configs/eg_gm_punet_v3_hierarchical_ms_gmm_lidc.yaml ^
      --smoke-test

正式：
    python train/train_eg_gm_punet_v3_hierarchical_ms_gmm.py ^
      --config configs/eg_gm_punet_v3_hierarchical_ms_gmm_lidc.yaml

实验隔离原则：
- 保留三大模块 MS-GMM / MS-DEH / EG-ALC；
- 只修改 MS-GMM 内部 shared-k -> scale-decoupled hierarchical GMM density；
- EG-ALC 删除 categorical capacity 和 adaptive tau；
- demand 仍使用已经验证的 fine foreground expected-data entropy；
- 不启用 CCV / ambiguity alignment。
"""
from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import random
import sys
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
from models.EG_GM_PUNet_v3_A1b import EGGMPUNetV3A1b
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


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ValueError("config 必须为 dict。")
    return value


def build_model(config: dict[str, Any]) -> EGGMPUNetV3A1b:
    m = config["model"]
    bridge = m["scale_residual_bridge"]
    return EGGMPUNetV3A1b(
        in_channels=int(m["in_channels"]),
        num_classes=int(m["num_classes"]),
        num_components={
            str(k): int(v)
            for k, v in m["num_components"].items()
        },
        latent_dims={
            str(k): int(v)
            for k, v in m["latent_dims"].items()
        },
        num_channels=tuple(int(v) for v in m["num_channels"]),
        group_feature_indices={
            str(k): int(v)
            for k, v in m["group_feature_indices"].items()
        },
        group_decoder_indices={
            str(k): int(v)
            for k, v in m["group_decoder_indices"].items()
        },
        num_convs_per_block=int(m["num_convs_per_block"]),
        head_hidden_dim=int(m["head_hidden_dim"]),
        fusion_1x1_convs=int(m["fusion_1x1_convs"]),
        segmentation_head_1x1_convs=int(
            m["segmentation_head_1x1_convs"]
        ),
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
        evidence_projection_channels=int(
            m["evidence_projection_channels"]
        ),
        evidence_fusion_channels=int(
            m["evidence_fusion_channels"]
        ),
        feedback_config=m["feedback"],
        bridge_enabled_scales=tuple(str(v) for v in bridge["enabled_scales"]),
        bridge_hidden_ratio=float(bridge["hidden_ratio"]),
        bridge_residual_scale=float(bridge["residual_scale"]),
    )


def save_checkpoint(
    path: Path,
    *,
    model: EGGMPUNetV3A1b,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    best_records: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
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
                    if torch.cuda.is_available()
                    else None
                ),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        },
        path,
    )


def restore_checkpoint(
    path: Path,
    *,
    model: EGGMPUNetV3A1b,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    obj = torch.load(path, map_location=device)
    model.load_state_dict(obj["model_state_dict"], strict=True)
    optimizer.load_state_dict(obj["optimizer_state_dict"])

    rng = obj.get("rng_state")
    if isinstance(rng, dict):
        try:
            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
            if rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])
            if rng.get("python") is not None:
                random.setstate(rng["python"])
        except Exception:
            print("[WARN] RNG state restore failed; continue with checkpoint weights.")

    return obj


def set_probability_stage(
    model: EGGMPUNetV3A1b,
    *,
    epoch: int,
    warmup_epochs: int,
) -> str:
    """A.1b quick-screen：固定 A.1a，只训练新 middle/fine bridge。

    这是受控 decoder-responsibility intervention，不让 latent backbone 同时变化。
    因此能够用 10–12 epoch 直接判断：现有 z_middle/z_fine 是否包含可被 decoder
    利用的 stochastic information。
    """
    del epoch, warmup_epochs

    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.probabilistic_model.scale_residual_bridge.parameters():
        p.requires_grad_(True)

    # evidence 仍可前向计算诊断量，但不参与梯度。
    model.evidence_head.eval()
    return "a1b_middle_fine_bridge_only"


def set_learning_rates(
    optimizer: torch.optim.Optimizer,
    probability_lr: float,
    evidence_lr: float,
) -> None:
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
    """逐病例 mean pairwise (1-IoU)，不含对角线。"""

    masks = all_masks.detach().float()
    if masks.ndim == 5 and masks.shape[2] == 1:
        masks = masks[:, :, 0]
    b, r = masks.shape[:2]

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


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
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
    """逐病例 GMM 总 latent std RMS，用于 A.1a 快筛机制统计。"""
    pi = distribution.mixture_probs.unsqueeze(-1)
    mean = (pi * distribution.mu).sum(dim=1, keepdim=True)
    within = (pi * distribution.sigma.pow(2)).sum(dim=1)
    between = (pi * (distribution.mu - mean).pow(2)).sum(dim=1)
    total_var = within + between
    return torch.sqrt(total_var.mean(dim=1).clamp_min(1e-12))


def new_accumulator(
    model: EGGMPUNetV3A1b,
    *,
    ece_bins: int,
    ranking_bins: int,
):
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
    model: EGGMPUNetV3A1b,
):
    b = int(segmentation.shape[0])
    p = output.probabilistic
    c = output.adaptive_controls

    scores = binary_scores_from_logits(
        p.decode.logits.detach(),
        segmentation,
    )

    acc["samples"] += b
    acc["objective_loss"] += float(objective_loss.item()) * b
    acc["joint_loss"] += float(output.loss.item()) * b
    acc["probabilistic_loss"] += (
        float(output.probabilistic_loss.item()) * b
    )
    acc["weighted_evidence_loss"] += (
        float(output.weighted_evidential_loss.item()) * b
    )
    acc["regularization_loss"] += (
        float(regularization_loss.item()) * b
    )
    acc["reconstruction"] += (
        float(p.reconstruction_loss_sum.item()) * b
    )
    acc["weighted_kl"] += float(p.weighted_kl_mean.item()) * b
    acc["unweighted_kl"] += (
        float(p.kl_total_unweighted_mean.item()) * b
    )
    acc["prob_dice"] += float(scores["dice"].sum().item())
    acc["prob_iou"] += float(scores["iou"].sum().item())
    acc["evidence_loss"] += (
        float(output.evidential_loss.total_loss_mean.item()) * b
    )

    acc["raw_demand"] += float(c.raw_demand.sum().item())
    acc["relative_demand"] += float(c.ambiguity_demand.sum().item())

    dann = annotation_diversity_iou(all_segmentations)
    acc["case_raw_demand"].extend(
        c.raw_demand.detach().cpu().tolist()
    )
    acc["case_dann_iou"].extend(
        dann.detach().cpu().tolist()
    )

    for name in model.group_names:
        acc[f"{name}_kl"] += (
            float(p.kl_group_per_sample[name].sum().item())
        )
        # internal decomposition only; not separately controlled
        q = p.posterior_distributions[name]
        pr = p.prior_distributions[name]

        acc[f"{name}_kl_cat"] += (
            float(p.kl_categorical_mean[name].item()) * b
        )
        acc[f"{name}_kl_gauss"] += (
            float(p.kl_gaussian_mean[name].item()) * b
        )
        acc[f"{name}_capacity"] += (
            float(c.group_capacity[name].sum().item())
        )
        acc[f"{name}_error"] += (
            float(c.group_error[name].sum().item())
        )
        acc[f"{name}_deficit"] += (
            float(c.group_deficit[name].sum().item())
        )
        acc[f"{name}_surplus"] += (
            float(c.group_surplus[name].sum().item())
        )
        acc[f"{name}_beta"] += (
            float(c.beta_groups[name].sum().item())
        )
        acc[f"{name}_raw_capacity_kl"] += (
            float(c.raw_group_joint_kl[name].sum().item())
        )
        acc[f"{name}_prior_keff"] += float(
            effective_component_count(pr).sum().item()
        )
        acc[f"{name}_posterior_keff"] += float(
            effective_component_count(q).sum().item()
        )
        acc[f"{name}_alpha"] += float(
            p.scale_balance_alpha[name].detach().item()
        ) * b
        acc[f"{name}_prior_std_rms"] += float(
            mixture_total_std_rms(pr).sum().item()
        )
        acc[f"{name}_posterior_std_rms"] += float(
            mixture_total_std_rms(q).sum().item()
        )
        acc[f"{name}_kl_rate"] += float(
            p.dimension_normalized_kl_mean[name].detach().item()
        ) * b

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
            result[f"{name}_{key}"] = (
                acc[f"{name}_{key}"] / n
            )

    result["demand_annotation_pearson"] = corr(
        acc["case_raw_demand"],
        acc["case_dann_iou"],
        False,
    )
    result["demand_annotation_spearman"] = corr(
        acc["case_raw_demand"],
        acc["case_dann_iou"],
        True,
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

    bridge = model.probabilistic_model.scale_residual_bridge
    for name, value in bridge.parameter_l2_norms().items():
        result[f"bridge_{name}_parameter_l2"] = float(value)
    for name, value in bridge.output_projection_l2_norms().items():
        result[f"bridge_{name}_output_projection_l2"] = float(value)

    return result


def iter_loader(loader, max_batches):
    if max_batches is None:
        return loader, len(loader)
    count = min(int(max_batches), len(loader))
    return islice(loader, count), count


def run_epoch(
    *,
    model: EGGMPUNetV3A1b,
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

    if training:
        model.train()
        probability_stage = set_probability_stage(
            model,
            epoch=epoch,
            warmup_epochs=int(
                tcfg["transferred_backbone_warmup_epochs"]
            ),
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
    phase = "Train" if training else "Val"
    progress = tqdm(
        iterable,
        total=total,
        desc=f"v3 {phase} {epoch}",
        ncols=190,
    )

    last_plr = 0.0
    last_elr = 0.0

    for batch in progress:
        if training:
            plr = piecewise_learning_rate(
                global_step,
                boundaries,
                [
                    float(v)
                    for v in tcfg[
                        "probabilistic_learning_rate_values"
                    ]
                ],
            )
            elr = piecewise_learning_rate(
                global_step,
                boundaries,
                [
                    float(v)
                    for v in tcfg[
                        "evidence_learning_rate_values"
                    ]
                ],
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
                is_training=True,  # val 也需要 posterior capacity diagnostic
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
                evidential_annealing_epochs=int(
                    tcfg["evidential_annealing_epochs"]
                ),
                evidence_dice_weight=float(
                    tcfg["evidence_dice_weight"]
                ),
                evidence_kl_weight=float(
                    tcfg["evidence_kl_weight"]
                ),
                evidence_calibration_weight=float(
                    tcfg["evidence_calibration_weight"]
                ),
                evidence_auxiliary_scale_weight=float(
                    tcfg["evidence_auxiliary_scale_weight"]
                ),
                feedback_enabled=bool(tcfg["feedback_enabled"]),
                update_feedback_stats=training,
                scale_balance_enabled=(
                    int(epoch) <= int(tcfg["scale_balance_active_epochs"])
                    and bool(tcfg["scale_balance_enabled"])
                ),
                update_scale_balance_stats=training,
            )

            if training:
                regularization_loss = (
                    float(tcfg["regularization_weight"])
                    * calculate_l2_regularization(model)
                )
            else:
                regularization_loss = torch.zeros(
                    (), device=device
                )

            objective = output.loss + regularization_loss

        if training:
            if not torch.isfinite(objective):
                raise RuntimeError("objective contains NaN/Inf.")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
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
            "aC": f"{acc['coarse_alpha']/completed:.2f}",
            "aM": f"{acc['middle_alpha']/completed:.2f}",
            "aF": f"{acc['fine_alpha']/completed:.2f}",
        }
        if not training and acc["prior_metric_samples"] > 0:
            postfix["GED4"] = (
                f"{acc['prior_ged2']/acc['prior_metric_samples']:.4f}"
            )
        progress.set_postfix(postfix)

    result = finish_accumulator(acc, model)
    result["probability_stage"] = probability_stage
    result["probabilistic_learning_rate"] = last_plr
    result["evidence_learning_rate"] = last_elr
    return result, global_step


def flatten_record(epoch, global_step, train_metrics, val_metrics):
    row = {"epoch": epoch, "global_step": global_step}
    for prefix, metrics in (
        ("train", train_metrics),
        ("val", val_metrics),
    ):
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[f"{prefix}_{k}"] = v
    return row


def write_history(save_dir: Path, history: list[dict[str, Any]]) -> None:
    (save_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not history:
        return
    keys = []
    seen = set()
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
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def best_records_template():
    return {
        "val_joint_loss": {
            "value": float("inf"),
            "direction": "min",
            "checkpoint": "best_val_joint_loss_eg_gm_punet_v3.pth",
            "epoch": None,
        },
        "val_prior_ged4": {
            "value": float("inf"),
            "direction": "min",
            "checkpoint": "best_val_ged4_eg_gm_punet_v3.pth",
            "epoch": None,
        },
        "val_probabilistic_dice": {
            "value": float("-inf"),
            "direction": "max",
            "checkpoint": "best_val_probabilistic_dice_eg_gm_punet_v3.pth",
            "epoch": None,
        },
        "val_evidence_dice": {
            "value": float("-inf"),
            "direction": "max",
            "checkpoint": "best_val_evidence_dice_eg_gm_punet_v3.pth",
            "epoch": None,
        },
        "val_evidence_ece": {
            "value": float("inf"),
            "direction": "min",
            "checkpoint": "best_val_evidence_ece_eg_gm_punet_v3.pth",
            "epoch": None,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/eg_gm_punet_v3_a1b_lidc.yaml",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--quick-screen", action="store_true",
        help="运行 12-epoch A.1b bridge-only 快筛；feedback/evidence loss 关闭。",
    )
    parser.add_argument(
        "--stop-epoch", type=int, default=None,
        help="续训时指定停止 epoch，例如 40 或 150。",
    )
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(config_path)
    if args.smoke_test:
        cfg = copy.deepcopy(cfg)
        # A.1b smoke 重点检查 A.1a transfer + zero-init bridge + 前后向；不混入 EG-ALC。
        cfg["training"]["feedback_enabled"] = False
        cfg["model"]["feedback"]["enabled"] = False
        cfg["training"]["transferred_backbone_warmup_epochs"] = 0
        cfg["training"]["scale_balance_active_epochs"] = 0
        cfg["training"]["evidence_weight"] = 0.0
        cfg["training"]["regularization_weight"] = 0.0
    elif args.quick_screen:
        cfg = copy.deepcopy(cfg)
        # A.1b 只验证 decoder responsibility：feedback 与 evidence loss 全部关闭。
        cfg["training"]["feedback_enabled"] = False
        cfg["model"]["feedback"]["enabled"] = False
        cfg["training"]["evidence_weight"] = 0.0
        cfg["training"]["regularization_weight"] = 0.0

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
        init = cfg.get("initialization", {})
        if init.get("mode") != "a1a_best_ged":
            raise ValueError("A.1b 必须从 A.1a best-GED checkpoint 初始化。")
        init_report = model.load_a1a_checkpoint(
            init["checkpoint"],
            map_location=device,
        )
        print("[INIT] A.1a best-GED -> A.1b bridge transfer:")
        print(json.dumps(init_report, indent=2, ensure_ascii=False))
        if init_report.get("source_epoch") != int(init.get("expected_source_epoch", 9)):
            print(
                "[WARN] source epoch != expected epoch; please confirm checkpoint. ",
                init_report.get("source_epoch"),
            )
        initial_bridge_norms = model.probabilistic_model.scale_residual_bridge.output_projection_l2_norms()
        print("[INIT] zero-init bridge output projection norms:", initial_bridge_norms)
        if any(abs(float(v)) > 1e-12 for v in initial_bridge_norms.values()):
            raise RuntimeError("A.1b bridge 初始化不是 zero-init，请停止实验。")

    optimizer = Adam(
        [
            {
                "params": model.probabilistic_model.scale_residual_bridge.parameters(),
                "lr": float(
                    cfg["training"][
                        "probabilistic_learning_rate_values"
                    ][0]
                ),
                "group_name": "probability",
            },
            {
                "params": model.evidence_head.parameters(),
                "lr": float(
                    cfg["training"][
                        "evidence_learning_rate_values"
                    ][0]
                ),
                "group_name": "evidence",
            },
        ],
        weight_decay=0.0,
    )

    configured_epochs = int(cfg["training"]["epochs"])
    if args.smoke_test:
        epochs = 1
    elif args.quick_screen:
        epochs = int(cfg["training"]["quick_screen_epochs"])
    elif args.stop_epoch is not None:
        epochs = min(int(args.stop_epoch), configured_epochs)
    else:
        epochs = configured_epochs

    train_max_batches = 2 if args.smoke_test else None
    if args.smoke_test:
        val_max_batches = 2
    elif args.quick_screen:
        val_max_batches = int(cfg["validation"]["quick_screen_batches"])
    else:
        val_max_batches = None

    # A.1b 是独立 12-epoch bridge-only 快筛，LR schedule 只按本实验长度计算。
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
        resume_path = Path(resume)
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path
        obj = restore_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        start_epoch = int(obj["epoch"]) + 1
        global_step = int(obj.get("global_step", 0))
        best_records = obj.get(
            "best_records",
            best_records,
        )
        hp = save_dir / "history.json"
        if hp.exists():
            history = json.loads(hp.read_text(encoding="utf-8"))

    manifest = {
        "config": cfg,
        "initialization_report": init_report,
        "notes": {
            "shared_semantic_k": False,
            "scale_decoupled_gmm": True,
            "hierarchical_latents": True,
            "adaptive_categorical_beta": False,
            "adaptive_gumbel_temperature": False,
            "component_vitality": False,
            "scale_specific_demand": False,
            "a1b_centered_scale_residual_bridge": True,
            "a1b_bridge_only_training": True,
            "validated_shared_demand": (
                "fine_foreground_expected_data_entropy"
            ),
        },
    }
    (save_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 110)
    print("EG-GM-PUNet v3 Stage A.1b Quick Screen")
    print("A.1a frozen backbone + middle/fine centered stochastic residual bridge; feedback/evidence loss OFF")
    print("=" * 110)
    print(f"Device: {device}")
    print(f"Train/Val: {len(train_loader.dataset)}/{len(val_loader.dataset)}")
    print(f"Output: {save_dir}")

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

        row = flatten_record(
            epoch, global_step, train_metrics, val_metrics
        )
        history.append(row)
        write_history(save_dir, history)

        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"train_pDice={train_metrics['prob_dice']:.4f} "
            f"val_pDice={val_metrics['prob_dice']:.4f} "
            f"val_GED4={val_metrics.get('prior_ged2', float('nan')):.4f} "
            f"val_eDice={val_metrics.get('evidence_hard_dice_all_raters', float('nan')):.4f} "
            f"U={train_metrics['relative_demand']:.3f} "
            f"beta=({train_metrics['coarse_beta']:.3f},"
            f"{train_metrics['middle_beta']:.3f},"
            f"{train_metrics['fine_beta']:.3f})"
        )
        print(
            "  [A1b-base] "
            f"KL=({val_metrics['coarse_kl']:.4g},"
            f"{val_metrics['middle_kl']:.4g},"
            f"{val_metrics['fine_kl']:.4g}) "
            f"alpha=({train_metrics['coarse_alpha']:.2f},"
            f"{train_metrics['middle_alpha']:.2f},"
            f"{train_metrics['fine_alpha']:.2f}) "
            f"priorStd=({val_metrics['coarse_prior_std_rms']:.3f},"
            f"{val_metrics['middle_prior_std_rms']:.3f},"
            f"{val_metrics['fine_prior_std_rms']:.3f})"
        )
        print(
            "  [A1b-bridge] "
            f"div={val_metrics.get('prior_sample_diversity', float('nan')):.4f} "
            f"cross={val_metrics.get('prior_cross_dice', float('nan')):.4f} "
            f"best={val_metrics.get('prior_best_dice', float('nan')):.4f} "
            f"projNormM={val_metrics.get('bridge_middle_output_projection_l2', float('nan')):.4g} "
            f"projNormF={val_metrics.get('bridge_fine_output_projection_l2', float('nan')):.4g}"
        )

        monitored = {
            "val_joint_loss": val_metrics["joint_loss"],
            "val_prior_ged4": val_metrics.get(
                "prior_ged2", float("inf")
            ),
            "val_probabilistic_dice": val_metrics["prob_dice"],
            "val_evidence_dice": val_metrics.get(
                "evidence_hard_dice_all_raters",
                float("-inf"),
            ),
            "val_evidence_ece": val_metrics.get(
                "evidence_ece", float("inf")
            ),
        }

        for name, value in monitored.items():
            rec = best_records[name]
            improved = (
                value < rec["value"]
                if rec["direction"] == "min"
                else value > rec["value"]
            )
            if improved:
                rec["value"] = float(value)
                rec["epoch"] = int(epoch)
                save_checkpoint(
                    save_dir / rec["checkpoint"],
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    config=cfg,
                    best_records=best_records,
                    metrics=row,
                )
                print(
                    f"[BEST] {name}={value:.6f} @ epoch {epoch}"
                )

        save_checkpoint(
            save_dir / "latest_eg_gm_punet_v3.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            config=cfg,
            best_records=best_records,
            metrics=row,
        )

        if (
            epoch % int(cfg["training"]["save_every_epochs"]) == 0
            or epoch == epochs
        ):
            save_checkpoint(
                save_dir / "checkpoints" / f"epoch_{epoch:03d}.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                config=cfg,
                best_records=best_records,
                metrics=row,
            )

        (save_dir / "best_records.json").write_text(
            json.dumps(best_records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # -----------------------------------------------------------------
    # A.1b quick-screen 自动判定：只看 decoder-responsibility 是否开始恢复。
    # 这是继续/停止 gate，不作为论文最终统计。
    # -----------------------------------------------------------------
    baseline = cfg["screening_baseline"]

    numeric_history = [r for r in history if isinstance(r.get("val_prior_ged2"), (int, float))]
    final_val = numeric_history[-1] if numeric_history else {}
    tail = numeric_history[-3:] if numeric_history else []

    def _row_value(row, key, default=float("nan")):
        try:
            return float(row.get("val_" + key, default))
        except Exception:
            return default

    def _tail_mean(key):
        vals = [_row_value(r, key) for r in tail]
        vals = [v for v in vals if math.isfinite(v)]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    # 选 diversity 最大的一行，检查 stochastic coverage 提升是否伴随 fidelity 崩溃。
    best_div_row = max(
        numeric_history,
        key=lambda r: _row_value(r, "prior_sample_diversity", float("-inf")),
    ) if numeric_history else {}
    best_ged_row = min(
        numeric_history,
        key=lambda r: _row_value(r, "prior_ged2", float("inf")),
    ) if numeric_history else {}

    mechanism = {
        "baseline_ged4": float(baseline["ged4"]),
        "baseline_cross_dice": float(baseline["cross_dice"]),
        "baseline_best_dice": float(baseline["best_dice"]),
        "baseline_sample_diversity": float(baseline["sample_diversity"]),
        "best_diversity": _row_value(best_div_row, "prior_sample_diversity"),
        "best_diversity_epoch": int(best_div_row.get("epoch", -1)) if best_div_row else -1,
        "cross_dice_at_best_diversity": _row_value(best_div_row, "prior_cross_dice"),
        "ged4_at_best_diversity": _row_value(best_div_row, "prior_ged2"),
        "best_ged4": _row_value(best_ged_row, "prior_ged2"),
        "best_ged4_epoch": int(best_ged_row.get("epoch", -1)) if best_ged_row else -1,
        "tail3_ged4": _tail_mean("prior_ged2"),
        "tail3_cross_dice": _tail_mean("prior_cross_dice"),
        "tail3_best_dice": _tail_mean("prior_best_dice"),
        "tail3_sample_diversity": _tail_mean("prior_sample_diversity"),
        "final_annotation_diversity": _row_value(final_val, "annotation_diversity_metric"),
        "final_bridge_middle_output_projection_l2": _row_value(final_val, "bridge_middle_output_projection_l2"),
        "final_bridge_fine_output_projection_l2": _row_value(final_val, "bridge_fine_output_projection_l2"),
    }

    diversity_gain = mechanism["best_diversity"] - mechanism["baseline_sample_diversity"]
    finite = all(
        math.isfinite(v)
        for k, v in mechanism.items()
        if isinstance(v, float) and k not in {"final_annotation_diversity"}
    )
    criteria = {
        "finite": finite,
        "bridge_learned": (
            mechanism["final_bridge_middle_output_projection_l2"] > 1e-8
            or mechanism["final_bridge_fine_output_projection_l2"] > 1e-8
        ),
        "diversity_gain_at_least_0p02": diversity_gain >= 0.02,
        "cross_dice_not_collapsed": (
            mechanism["cross_dice_at_best_diversity"]
            >= mechanism["baseline_cross_dice"] - 0.025
        ),
        "ged4_not_severely_worse": (
            mechanism["best_ged4"]
            <= mechanism["baseline_ged4"] + 0.03
        ),
        "tail_cross_dice_not_collapsed": (
            mechanism["tail3_cross_dice"]
            >= mechanism["baseline_cross_dice"] - 0.025
        ),
    }

    if not finite or not criteria["cross_dice_not_collapsed"]:
        screen_status = "STOP"
    elif (
        criteria["bridge_learned"]
        and criteria["diversity_gain_at_least_0p02"]
        and criteria["ged4_not_severely_worse"]
        and criteria["tail_cross_dice_not_collapsed"]
    ):
        screen_status = "PROMISING_FOR_A1_FULL"
    elif diversity_gain < 0.01:
        screen_status = "STOP_NO_DECODER_RESPONSIBILITY_GAIN"
    else:
        screen_status = "REVIEW_ONCE"

    summary = {
        "experiment": cfg["experiment_name"],
        "completed_epochs": epochs,
        "screening": {
            "status": screen_status,
            "criteria": criteria,
            "mechanism": mechanism,
            "best_diversity_gain_vs_a1a_epoch9": diversity_gain,
        },
        "best_records": best_records,
        "probability_evaluation_checkpoint": best_records["val_prior_ged4"]["checkpoint"],
        "architecture": {
            "top_level": "MS-GMM + MS-DEH + EG-ALC",
            "shared_semantic_k": False,
            "hierarchical_scale_decoupled_gmm": True,
            "prior_anchored_residual_posterior": True,
            "temporary_scale_kl_balance": True,
            "a1b_centered_scale_residual_bridge": True,
            "a1b_bridge_scales": ["middle", "fine"],
            "a1b_base_backbone_frozen_during_quick_screen": True,
            "scale_specific_demand": False,
        },
    }
    (save_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 110)
    print("[DONE] EG-GM-PUNet v3 Stage A.1b quick-screen.")
    print(
        "Best GED4 checkpoint:",
        summary["probability_evaluation_checkpoint"],
    )
    print("[SCREEN]", screen_status)
    print(json.dumps(summary["screening"], indent=2, ensure_ascii=False))
    from datetime import datetime, timezone, timedelta
    bj = timezone(timedelta(hours=8))
    print("[END Beijing]", datetime.now(bj).strftime("%Y-%m-%d %H:%M:%S %z"))
    print("=" * 110)


if __name__ == "__main__":
    main()
