"""Formal V3 training: Multi-Scale Grouped GMM + Evidential Head.

Run smoke test:
    python train/train_evidential_multiscale_grouped_gmm.py ^
        --config configs/evidential_multiscale_grouped_gmm_lidc.yaml ^
        --smoke-test

Run full training:
    python train/train_evidential_multiscale_grouped_gmm.py ^
        --config configs/evidential_multiscale_grouped_gmm_lidc.yaml

Resume:
    python train/train_evidential_multiscale_grouped_gmm.py ^
        --config configs/evidential_multiscale_grouped_gmm_lidc.yaml ^
        --resume result/v3_evidential_multiscale_grouped_gmm_k5/latest_v3_evidential.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import islice
from pathlib import Path
from typing import Any

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
from models.Evidential_Multi_scale_Gaussian_Mixture_prob_unet.evidential_multiscale_gmm_model import (
    EvidentialMultiScaleGroupedGMMProbUNet,
)
from utils.evidential_metrics import EvidentialMetricAccumulator
from utils.gmm_experiment_utils import (
    binary_scores_from_logits,
    build_step_boundaries,
    calculate_l2_regularization,
    capture_rng_state,
    count_trainable_parameters,
    load_checkpoint,
    piecewise_learning_rate,
    prepare_device,
    prior_distribution_metrics,
    restore_rng_state,
    save_checkpoint,
    save_run_manifest,
    set_seed,
    write_history,
)

from datetime import datetime
from zoneinfo import ZoneInfo


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Config must contain a dictionary.")
    return config


def build_model(
    config: dict[str, Any],
) -> EvidentialMultiScaleGroupedGMMProbUNet:
    item = config["model"]

    return EvidentialMultiScaleGroupedGMMProbUNet(
        in_channels=int(item["in_channels"]),
        num_classes=int(item["num_classes"]),
        num_components=int(item["num_components"]),
        latent_dims={
            str(name): int(value)
            for name, value in item["latent_dims"].items()
        },
        num_channels=tuple(
            int(value)
            for value in item["num_channels"]
        ),
        group_feature_indices={
            str(name): int(value)
            for name, value
            in item["group_feature_indices"].items()
        },
        group_decoder_indices={
            str(name): int(value)
            for name, value
            in item["group_decoder_indices"].items()
        },
        num_convs_per_block=int(
            item["num_convs_per_block"]
        ),
        fusion_1x1_convs=int(
            item["fusion_1x1_convs"]
        ),
        segmentation_head_1x1_convs=int(
            item["segmentation_head_1x1_convs"]
        ),
        min_log_sigma=item.get("min_log_sigma"),
        max_log_sigma=item.get("max_log_sigma"),
        evidence_projection_channels=int(
            item["evidence_projection_channels"]
        ),
        evidence_fusion_channels=int(
            item["evidence_fusion_channels"]
        ),
    )


def set_probability_trainable(
    model: EvidentialMultiScaleGroupedGMMProbUNet,
    trainable: bool,
) -> None:
    for parameter in (
        model.probabilistic_model.parameters()
    ):
        parameter.requires_grad_(trainable)


def set_group_learning_rates(
    optimizer: torch.optim.Optimizer,
    probability_lr: float,
    evidence_lr: float,
) -> None:
    for group in optimizer.param_groups:
        name = group.get("group_name")
        if name == "probabilistic":
            group["lr"] = float(probability_lr)
        elif name == "evidence":
            group["lr"] = float(evidence_lr)
        else:
            raise RuntimeError(
                f"Unknown optimizer group: {name}"
            )


def new_accumulator(
    model: EvidentialMultiScaleGroupedGMMProbUNet,
    ece_bins: int,
    ranking_bins: int,
) -> dict[str, Any]:
    return {
        "samples": 0,
        "objective_loss": 0.0,
        "joint_loss": 0.0,
        "probabilistic_loss": 0.0,
        "weighted_evidential_loss": 0.0,
        "regularization_loss": 0.0,
        "reconstruction_loss_sum": 0.0,
        "reconstruction_loss_mean": 0.0,
        "weighted_kl": 0.0,
        "unweighted_kl": 0.0,
        "kl_categorical": 0.0,
        "kl_groups": {
            name: 0.0
            for name in model.group_names
        },
        "probabilistic_dice": 0.0,
        "probabilistic_iou": 0.0,
        "evidence_loss_mean": 0.0,
        "evidence_ace": 0.0,
        "evidence_dice_loss": 0.0,
        "evidence_dirichlet_kl": 0.0,
        "evidence_calibration": 0.0,
        "evidence_scale_loss": {
            name: 0.0
            for name in model.group_names
        },
        "prior_probability_sum": torch.zeros(
            model.num_components,
            dtype=torch.float64,
        ),
        "posterior_probability_sum": torch.zeros(
            model.num_components,
            dtype=torch.float64,
        ),
        "posterior_component_counts": torch.zeros(
            model.num_components,
            dtype=torch.long,
        ),
        "prior_ged2": 0.0,
        "prior_cross_dice": 0.0,
        "prior_best_dice": 0.0,
        "prior_sample_diversity": 0.0,
        "annotation_diversity": 0.0,
        "prior_metric_samples": 0,
        "evidence_metrics": EvidentialMetricAccumulator(
            num_bins=ece_bins,
            ranking_bins=ranking_bins,
        ),
    }


def update_accumulator(
    accumulator: dict[str, Any],
    output,
    state,
    segmentation: torch.Tensor,
    all_segmentations: torch.Tensor,
    objective_loss: torch.Tensor,
    regularization_loss: torch.Tensor,
    model: EvidentialMultiScaleGroupedGMMProbUNet,
) -> None:
    batch_size = segmentation.shape[0]
    probabilistic = output.probabilistic

    scores = binary_scores_from_logits(
        probabilistic.decode.logits.detach(),
        segmentation,
    )

    accumulator["samples"] += batch_size
    accumulator["objective_loss"] += (
        float(objective_loss.item()) * batch_size
    )
    accumulator["joint_loss"] += (
        float(output.loss.item()) * batch_size
    )
    accumulator["probabilistic_loss"] += (
        float(output.probabilistic_loss.item())
        * batch_size
    )
    accumulator["weighted_evidential_loss"] += (
        float(output.weighted_evidential_loss.item())
        * batch_size
    )
    accumulator["regularization_loss"] += (
        float(regularization_loss.item())
        * batch_size
    )

    accumulator["reconstruction_loss_sum"] += (
        float(
            probabilistic
            .reconstruction_loss_sum
            .item()
        )
        * batch_size
    )
    accumulator["reconstruction_loss_mean"] += (
        float(
            probabilistic
            .reconstruction_loss_mean
            .item()
        )
        * batch_size
    )
    accumulator["weighted_kl"] += (
        float(probabilistic.weighted_kl_mean.item())
        * batch_size
    )
    accumulator["unweighted_kl"] += (
        float(
            probabilistic
            .kl_total_unweighted_mean
            .item()
        )
        * batch_size
    )
    accumulator["kl_categorical"] += (
        float(
            probabilistic
            .kl_categorical_mean
            .item()
        )
        * batch_size
    )

    for name in model.group_names:
        accumulator["kl_groups"][name] += (
            float(
                probabilistic
                .kl_group_mean[name]
                .item()
            )
            * batch_size
        )

    accumulator["probabilistic_dice"] += float(
        scores["dice"].sum().item()
    )
    accumulator["probabilistic_iou"] += float(
        scores["iou"].sum().item()
    )

    evidence_loss = output.evidential_loss
    accumulator["evidence_loss_mean"] += (
        float(evidence_loss.total_loss_mean.item())
        * batch_size
    )
    accumulator["evidence_ace"] += (
        float(evidence_loss.fused.ace_mean.item())
        * batch_size
    )
    accumulator["evidence_dice_loss"] += (
        float(evidence_loss.fused.dice_mean.item())
        * batch_size
    )
    accumulator["evidence_dirichlet_kl"] += (
        float(evidence_loss.fused.kl_mean.item())
        * batch_size
    )
    accumulator["evidence_calibration"] += (
        float(
            evidence_loss
            .fused
            .calibration_mean
            .item()
        )
        * batch_size
    )

    for name in model.group_names:
        accumulator["evidence_scale_loss"][name] += (
            float(
                evidence_loss
                .scales[name]
                .total_mean
                .item()
            )
            * batch_size
        )

    accumulator["prior_probability_sum"] += (
        state.prior.mixture_probs.detach()
        .sum(dim=0).cpu().double()
    )
    accumulator["posterior_probability_sum"] += (
        state.posterior.mixture_probs.detach()
        .sum(dim=0).cpu().double()
    )
    accumulator["posterior_component_counts"] += (
        torch.bincount(
            probabilistic
            .decode
            .component_indices
            .detach()
            .cpu(),
            minlength=model.num_components,
        )
    )

    accumulator["evidence_metrics"].update(
        evidence_output=output.evidence.fused,
        evidence_summary=(
            output.evidence.summaries["fused"]
        ),
        segmentation=segmentation,
        all_segmentations=all_segmentations,
        soft_target=evidence_loss.soft_target,
    )


def finish_accumulator(
    accumulator: dict[str, Any],
    model: EvidentialMultiScaleGroupedGMMProbUNet,
) -> dict[str, Any]:
    count = int(accumulator["samples"])
    if count <= 0:
        raise RuntimeError("No samples were accumulated.")

    result: dict[str, Any] = {}

    scalar_names = (
        "objective_loss",
        "joint_loss",
        "probabilistic_loss",
        "weighted_evidential_loss",
        "regularization_loss",
        "reconstruction_loss_sum",
        "reconstruction_loss_mean",
        "weighted_kl",
        "unweighted_kl",
        "kl_categorical",
        "probabilistic_dice",
        "probabilistic_iou",
        "evidence_loss_mean",
        "evidence_ace",
        "evidence_dice_loss",
        "evidence_dirichlet_kl",
        "evidence_calibration",
    )

    for name in scalar_names:
        result[name] = accumulator[name] / count

    for name in model.group_names:
        result[f"kl_{name}"] = (
            accumulator["kl_groups"][name]
            / count
        )
        result[f"evidence_scale_loss_{name}"] = (
            accumulator[
                "evidence_scale_loss"
            ][name]
            / count
        )

    prior_probabilities = (
        accumulator["prior_probability_sum"]
        / count
    )
    posterior_probabilities = (
        accumulator["posterior_probability_sum"]
        / count
    )

    result[
        "prior_mean_component_probabilities"
    ] = prior_probabilities.tolist()

    result[
        "posterior_mean_component_probabilities"
    ] = posterior_probabilities.tolist()

    result["posterior_component_counts"] = (
        accumulator[
            "posterior_component_counts"
        ].tolist()
    )

    def effective_count(
        probabilities: torch.Tensor,
    ) -> float:
        safe = probabilities.clamp_min(1e-8)
        return float(
            torch.exp(
                -(safe * safe.log()).sum()
            ).item()
        )

    result["prior_effective_components"] = (
        effective_count(prior_probabilities)
    )
    result["posterior_effective_components"] = (
        effective_count(
            posterior_probabilities
        )
    )

    result.update(
        {
            f"evidence_{name}": value
            for name, value
            in accumulator[
                "evidence_metrics"
            ].compute().items()
        }
    )

    prior_count = int(
        accumulator["prior_metric_samples"]
    )
    if prior_count > 0:
        for name in (
            "prior_ged2",
            "prior_cross_dice",
            "prior_best_dice",
            "prior_sample_diversity",
            "annotation_diversity",
        ):
            result[name] = (
                accumulator[name] / prior_count
            )

    return result


def iter_loader(loader, max_batches: int | None):
    if max_batches is None:
        return loader, len(loader)
    count = min(len(loader), max_batches)
    return islice(loader, count), count


def run_epoch(
    model: EvidentialMultiScaleGroupedGMMProbUNet,
    loader,
    device: torch.device,
    config: dict[str, Any],
    epoch: int,
    training: bool,
    optimizer: torch.optim.Optimizer | None,
    global_step: int,
    boundaries: list[int],
    probability_lr_values: list[float],
    evidence_lr_values: list[float],
    max_batches: int | None,
) -> tuple[dict[str, Any], int]:
    training_config = config["training"]
    validation_config = config["validation"]

    if training:
        model.train()
    else:
        model.eval()

    frozen = (
        training
        and epoch
        <= int(
            training_config[
                "freeze_probabilistic_epochs"
            ]
        )
    )

    if training:
        set_probability_trainable(
            model,
            trainable=not frozen,
        )

    accumulator = new_accumulator(
        model=model,
        ece_bins=int(
            validation_config["ece_bins"]
        ),
        ranking_bins=int(
            validation_config["ranking_bins"]
        ),
    )

    iterable, total_batches = iter_loader(
        loader,
        max_batches,
    )

    phase = "Train" if training else "Val"
    progress = tqdm(
        iterable,
        total=total_batches,
        desc=f"V3 {phase} {epoch}",
        ncols=180,
    )

    last_probability_lr = 0.0
    last_evidence_lr = 0.0

    for batch in progress:
        if training:
            probability_lr = (
                0.0
                if frozen
                else piecewise_learning_rate(
                    global_step,
                    boundaries,
                    probability_lr_values,
                )
            )
            evidence_lr = piecewise_learning_rate(
                global_step,
                boundaries,
                evidence_lr_values,
            )

            set_group_learning_rates(
                optimizer,
                probability_lr,
                evidence_lr,
            )
            last_probability_lr = probability_lr
            last_evidence_lr = evidence_lr

        images = batch["image"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        segmentation = batch["mask"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        all_segmentations = batch["masks"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        context = (
            torch.enable_grad()
            if training
            else torch.no_grad()
        )

        with context:
            state = model(
                image=images,
                segmentation=segmentation,
                is_training=True,
            )

            output = model.elbo(
                state=state,
                segmentation=segmentation,
                all_segmentations=(
                    all_segmentations
                ),
                beta_categorical=float(
                    training_config[
                        "beta_categorical"
                    ]
                ),
                beta_groups={
                    str(name): float(value)
                    for name, value
                    in training_config[
                        "beta_groups"
                    ].items()
                },
                gumbel_temperature=float(
                    training_config[
                        "gumbel_temperature"
                    ]
                ),
                evidence_weight=float(
                    training_config[
                        "evidence_weight"
                    ]
                ),
                current_epoch=epoch,
                evidential_annealing_epochs=int(
                    training_config[
                        "evidential_annealing_epochs"
                    ]
                ),
                total_epochs=int(
                    training_config["epochs"]
                ),
                evidence_dice_weight=float(
                    training_config[
                        "evidence_dice_weight"
                    ]
                ),
                evidence_kl_weight=float(
                    training_config[
                        "evidence_kl_weight"
                    ]
                ),
                evidence_calibration_weight=float(
                    training_config[
                        "evidence_calibration_weight"
                    ]
                ),
                evidence_auxiliary_scale_weight=float(
                    training_config[
                        "evidence_auxiliary_scale_weight"
                    ]
                ),
            )

            if training:
                regularization_loss = (
                    float(
                        training_config[
                            "regularization_weight"
                        ]
                    )
                    * calculate_l2_regularization(
                        model
                    )
                )
                objective_loss = (
                    output.loss
                    + regularization_loss
                )
            else:
                regularization_loss = torch.zeros(
                    (),
                    device=device,
                )
                objective_loss = output.loss

        if training:
            if not torch.isfinite(
                objective_loss
            ):
                raise RuntimeError(
                    "Training objective contains "
                    "NaN or Inf."
                )

            optimizer.zero_grad(
                set_to_none=True
            )
            objective_loss.backward()
            optimizer.step()
            global_step += 1

        update_accumulator(
            accumulator=accumulator,
            output=output,
            state=state,
            segmentation=segmentation,
            all_segmentations=(
                all_segmentations
            ),
            objective_loss=objective_loss,
            regularization_loss=(
                regularization_loss
            ),
            model=model,
        )

        if not training:
            sample_logits, _ = model.sample(
                state=state,
                num_samples=int(
                    validation_config[
                        "prior_samples_each_epoch"
                    ]
                ),
                return_component_indices=True,
            )

            prior_metrics = (
                prior_distribution_metrics(
                    sample_logits,
                    all_segmentations,
                )
            )

            batch_size = images.shape[0]
            accumulator[
                "prior_metric_samples"
            ] += batch_size

            mapping = {
                "prior_ged2": "ged2",
                "prior_cross_dice": (
                    "cross_dice"
                ),
                "prior_best_dice": (
                    "best_dice"
                ),
                "prior_sample_diversity": (
                    "sample_diversity"
                ),
                "annotation_diversity": (
                    "annotation_diversity"
                ),
            }

            for destination, source in (
                mapping.items()
            ):
                accumulator[destination] += (
                    float(
                        prior_metrics[source]
                        .sum()
                        .item()
                    )
                )

        completed = max(
            accumulator["samples"],
            1,
        )

        postfix = {
            "pDice": (
                f"{accumulator['probabilistic_dice']/completed:.4f}"
            ),
            "eLoss": (
                f"{accumulator['evidence_loss_mean']/completed:.4f}"
            ),
            "vac": (
                f"{accumulator['evidence_metrics'].scalar_sums['ambiguity_weighted_vacuity']/completed:.3f}"
            ),
        }

        if not training:
            prior_completed = max(
                accumulator[
                    "prior_metric_samples"
                ],
                1,
            )
            postfix["GED4"] = (
                f"{accumulator['prior_ged2']/prior_completed:.4f}"
            )

        progress.set_postfix(postfix)

    result = finish_accumulator(
        accumulator,
        model,
    )

    result[
        "probabilistic_learning_rate"
    ] = last_probability_lr
    result[
        "evidence_learning_rate"
    ] = last_evidence_lr
    result[
        "probabilistic_frozen"
    ] = frozen

    return result, global_step


def flatten_record(
    epoch: int,
    global_step: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "epoch": int(epoch),
        "global_step": int(global_step),
    }

    for prefix, metrics in (
        ("train", train_metrics),
        ("val", val_metrics),
    ):
        for key, value in metrics.items():
            if isinstance(value, list):
                record[
                    f"{prefix}_{key}"
                ] = value
                for index, item in (
                    enumerate(value)
                ):
                    record[
                        f"{prefix}_{key}_{index}"
                    ] = item
            else:
                record[
                    f"{prefix}_{key}"
                ] = value

    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "evidential_multiscale_grouped_gmm_lidc.yaml"
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )
    parser.add_argument(
        "--resume",
        default=None,
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    set_seed(int(config["seed"]))
    device = prepare_device(
        config.get("device", "cuda")
    )

    training_config = config["training"]
    validation_config = config["validation"]

    output_directory = Path(
        config["save_dir"]
    )
    if args.smoke_test:
        output_directory = (
            output_directory / "smoke_test"
        )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_run_manifest(
        output_directory=output_directory,
        project_root=PROJECT_ROOT,
        config_path=config_path,
        tracked_files=[
            PROJECT_ROOT
            / (
                "models/"
                "Evidential_Multi_scale_Gaussian_Mixture_prob_unet/"
                "evidential_multiscale_gmm_model.py"
            ),
            PROJECT_ROOT
            / (
                "models/"
                "Evidential_Multi_scale_Gaussian_Mixture_prob_unet/"
                "evidential_multiscale_gmm_parts.py"
            ),
            PROJECT_ROOT
            / (
                "models/"
                "Multi_scale_Gaussian_Mixture_prob_unet/"
                "multiscale_grouped_gmm_model.py"
            ),
            PROJECT_ROOT
            / "train/train_evidential_multiscale_grouped_gmm.py",
            PROJECT_ROOT
            / "utils/evidential_metrics.py",
        ],
        config=config,
    )

    train_loader = build_lidc_dataloader(
        root=config["data_root"],
        split="train",
        batch_size=int(
            training_config["batch_size"]
        ),
        num_workers=int(
            training_config["num_workers"]
        ),
        shuffle=True,
        random_rater=True,
    )

    val_loader = build_lidc_dataloader(
        root=config["data_root"],
        split="val",
        batch_size=int(
            training_config["batch_size"]
        ),
        num_workers=int(
            training_config["num_workers"]
        ),
        shuffle=False,
        random_rater=False,
        rater_index=int(
            validation_config["rater_index"]
        ),
    )

    model = build_model(config).to(device)

    optimizer = Adam(
        [
            {
                "params": (
                    model
                    .probabilistic_model
                    .parameters()
                ),
                "lr": float(
                    training_config[
                        "probabilistic_learning_rate_values"
                    ][0]
                ),
                "group_name": "probabilistic",
            },
            {
                "params": (
                    model
                    .evidence_head
                    .parameters()
                ),
                "lr": float(
                    training_config[
                        "evidence_learning_rate_values"
                    ][0]
                ),
                "group_name": "evidence",
            },
        ],
        weight_decay=0.0,
    )

    configured_epochs = int(
        training_config["epochs"]
    )
    epochs = (
        1
        if args.smoke_test
        else configured_epochs
    )

    total_steps = (
        configured_epochs
        * len(train_loader)
    )
    boundaries = build_step_boundaries(
        total_steps,
        training_config[
            "learning_rate_boundary_fractions"
        ],
    )

    probability_lr_values = [
        float(value)
        for value
        in training_config[
            "probabilistic_learning_rate_values"
        ]
    ]
    evidence_lr_values = [
        float(value)
        for value
        in training_config[
            "evidence_learning_rate_values"
        ]
    ]

    start_epoch = 1
    global_step = 0
    history: list[dict[str, Any]] = []

    best_records = {
        "val_joint_loss": {
            "value": float("inf"),
            "epoch": None,
            "checkpoint": (
                "best_val_joint_loss_v3_evidential.pth"
            ),
        },
        "val_probabilistic_dice": {
            "value": float("-inf"),
            "epoch": None,
            "checkpoint": (
                "best_val_probabilistic_dice_v3_evidential.pth"
            ),
        },
        "val_prior_ged4": {
            "value": float("inf"),
            "epoch": None,
            "checkpoint": (
                "best_val_ged4_v3_evidential.pth"
            ),
        },
        "val_evidence_dice": {
            "value": float("-inf"),
            "epoch": None,
            "checkpoint": (
                "best_val_evidence_dice_v3_evidential.pth"
            ),
        },
        "val_evidence_ece": {
            "value": float("inf"),
            "epoch": None,
            "checkpoint": (
                "best_val_evidence_ece_v3_evidential.pth"
            ),
        },
    }

    resume_path = (
        args.resume
        or config.get("resume")
    )

    if resume_path:
        checkpoint = load_checkpoint(
            Path(resume_path),
            model,
            optimizer,
            device,
        )
        start_epoch = (
            int(checkpoint["epoch"]) + 1
        )
        global_step = int(
            checkpoint.get(
                "global_step",
                0,
            )
        )
        best_records = checkpoint.get(
            "best_records",
            best_records,
        )

        history_path = (
            output_directory
            / "history.json"
        )
        if history_path.exists():
            with open(
                history_path,
                "r",
                encoding="utf-8",
            ) as file:
                history = json.load(file)

        print(
            f"[INFO] Resumed V3 from "
            f"{resume_path}"
        )
    else:
        initialization = config[
            "initialization"
        ]
        checkpoint_path = Path(
            initialization[
                "probabilistic_checkpoint"
            ]
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "V2-B initialization checkpoint "
                f"not found: {checkpoint_path}"
            )

        checkpoint = (
            model.load_probabilistic_checkpoint(
                checkpoint_path,
                map_location=device,
                strict=bool(
                    initialization["strict"]
                ),
            )
        )

        print(
            "[INFO] Loaded V2-B probability "
            f"checkpoint: {checkpoint_path}"
        )
        print(
            "[INFO] V2-B checkpoint epoch: "
            f"{checkpoint.get('epoch')}"
        )

    print("=" * 92)
    print(
        "V3 Multi-Scale Grouped GMM + "
        "Evidential Head training"
    )
    print("=" * 92)
    print(f"[INFO] Device: {device}")
    print(
        f"[INFO] Parameters total: "
        f"{count_trainable_parameters(model):,}"
    )
    print(
        f"[INFO] Probability parameters: "
        f"{sum(p.numel() for p in model.probabilistic_model.parameters()):,}"
    )
    print(
        f"[INFO] Evidence parameters: "
        f"{sum(p.numel() for p in model.evidence_head.parameters()):,}"
    )
    print(
        f"[INFO] Train/Val samples: "
        f"{len(train_loader.dataset)}/"
        f"{len(val_loader.dataset)}"
    )
    print(
        "[INFO] V3 evidence feedback: "
        "disabled by design"
    )
    print(
        "[INFO] Probability freeze epochs: "
        f"{training_config['freeze_probabilistic_epochs']}"
    )
    print(
        f"[INFO] LR boundaries: "
        f"{boundaries}"
    )

    max_train_batches = (
        2 if args.smoke_test else None
    )
    max_val_batches = (
        2 if args.smoke_test else None
    )

    for epoch in range(
        start_epoch,
        epochs + 1,
    ):
        train_metrics, global_step = (
            run_epoch(
                model=model,
                loader=train_loader,
                device=device,
                config=config,
                epoch=epoch,
                training=True,
                optimizer=optimizer,
                global_step=global_step,
                boundaries=boundaries,
                probability_lr_values=(
                    probability_lr_values
                ),
                evidence_lr_values=(
                    evidence_lr_values
                ),
                max_batches=(
                    max_train_batches
                ),
            )
        )

        # Make validation sampling identical
        # across epochs, then restore the
        # training RNG state.
        rng_state = capture_rng_state()
        set_seed(
            int(config["validation_seed"])
        )

        val_metrics, _ = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            config=config,
            epoch=epoch,
            training=False,
            optimizer=None,
            global_step=global_step,
            boundaries=boundaries,
            probability_lr_values=(
                probability_lr_values
            ),
            evidence_lr_values=(
                evidence_lr_values
            ),
            max_batches=max_val_batches,
        )

        restore_rng_state(rng_state)

        record = flatten_record(
            epoch=epoch,
            global_step=global_step,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

        history.append(record)
        write_history(
            history,
            output_directory,
        )

        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"frozen={train_metrics['probabilistic_frozen']}, "
            f"train_pDice={train_metrics['probabilistic_dice']:.4f}, "
            f"train_eDice={train_metrics['evidence_hard_dice_all_raters']:.4f}, "
            f"val_pDice={val_metrics['probabilistic_dice']:.4f}, "
            f"val_eDice={val_metrics['evidence_hard_dice_all_raters']:.4f}, "
            f"val_GED4={val_metrics['prior_ged2']:.4f}, "
            f"val_ECE={val_metrics['evidence_ece']:.4f}, "
            f"val_errorAUROC={val_metrics['evidence_error_detection_auroc']:.4f}, "
            f"val_vacuity={val_metrics['evidence_ambiguity_weighted_vacuity']:.4f}"
        )

        monitored = {
            "val_joint_loss": (
                float(
                    val_metrics[
                        "joint_loss"
                    ]
                ),
                "min",
            ),
            "val_probabilistic_dice": (
                float(
                    val_metrics[
                        "probabilistic_dice"
                    ]
                ),
                "max",
            ),
            "val_prior_ged4": (
                float(
                    val_metrics[
                        "prior_ged2"
                    ]
                ),
                "min",
            ),
            "val_evidence_dice": (
                float(
                    val_metrics[
                        "evidence_hard_dice_all_raters"
                    ]
                ),
                "max",
            ),
            "val_evidence_ece": (
                float(
                    val_metrics[
                        "evidence_ece"
                    ]
                ),
                "min",
            ),
        }

        for name, (
            value,
            direction,
        ) in monitored.items():
            best_value = float(
                best_records[name]["value"]
            )

            improved = (
                value < best_value
                if direction == "min"
                else value > best_value
            )

            if improved:
                best_records[name][
                    "value"
                ] = value
                best_records[name][
                    "epoch"
                ] = epoch

                save_checkpoint(
                    path=(
                        output_directory
                        / best_records[name][
                            "checkpoint"
                        ]
                    ),
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    config=config,
                    best_records=(
                        best_records
                    ),
                    metrics=record,
                )

                print(
                    f"[INFO] New best "
                    f"{name}={value:.6f} "
                    f"at epoch {epoch}."
                )

        save_checkpoint(
            path=(
                output_directory
                / "latest_v3_evidential.pth"
            ),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            config=config,
            best_records=best_records,
            metrics=record,
        )

        save_every = int(
            training_config[
                "save_every_epochs"
            ]
        )

        if (
            epoch % save_every == 0
            or epoch == epochs
        ):
            save_checkpoint(
                path=(
                    output_directory
                    / "checkpoints"
                    / f"epoch_{epoch:03d}.pth"
                ),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                config=config,
                best_records=(
                    best_records
                ),
                metrics=record,
            )

        with open(
            output_directory
            / "best_records.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                best_records,
                file,
                indent=2,
                ensure_ascii=False,
            )

    summary = {
        "experiment": (
            config["experiment_name"]
        ),
        "completed_epochs": epochs,
        "best_records": best_records,
        "last_epoch": (
            history[-1]
            if history
            else None
        ),
        "probability_evaluation_checkpoint": (
            best_records[
                "val_prior_ged4"
            ]["checkpoint"]
        ),
        "evidence_segmentation_checkpoint": (
            best_records[
                "val_evidence_dice"
            ]["checkpoint"]
        ),
        "evidence_calibration_checkpoint": (
            best_records[
                "val_evidence_ece"
            ]["checkpoint"]
        ),
        "evidence_feedback_used": False,
    }

    with open(
        output_directory
        / "training_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 92)
    print("[DONE] V3 evidential training finished.")
    print(
        "[INFO] Probability checkpoint: "
        f"{summary['probability_evaluation_checkpoint']}"
    )
    print(
        "[INFO] Evidence checkpoint: "
        f"{summary['evidence_segmentation_checkpoint']}"
    )
    print(
        "[INFO] Calibration checkpoint: "
        f"{summary['evidence_calibration_checkpoint']}"
    )
    print(f"[INFO] Results: {output_directory}")
    print("=" * 92)


if __name__ == "__main__":
    main()


beijing_time = datetime.now(ZoneInfo("Asia/Shanghai"))

print("=" * 60)
print(f"模型训练完成！")
print(f"训练完成时间（北京时间）：{beijing_time.strftime('%Y年%m月%d日 %H:%M:%S')}")
print("=" * 60)