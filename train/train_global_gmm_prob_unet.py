"""Train the Global GMM Probabilistic U-Net on LIDC-IDRI.

Run a smoke test:
    python train/train_global_gmm_prob_unet.py \
        --config configs/global_gmm_prob_unet_lidc.yaml \
        --smoke-test

Run full training:
    python train/train_global_gmm_prob_unet.py \
        --config configs/global_gmm_prob_unet_lidc.yaml

The script saves:
    latest_global_gmm.pth
    best_val_loss_global_gmm.pth
    best_val_dice_global_gmm.pth
    best_val_ged4_global_gmm.pth
    checkpoints/epoch_XXX.pth
    history.json
    history.csv
    best_records.json
    training_summary.json
    run_manifest.json
"""

from __future__ import annotations

import argparse
import json
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


from dataset.lidc_dataset import build_lidc_dataloader
from models.gmm_prob_unet.gmm_prob_unet_model import GMMProbUNet
from utils.gmm_experiment_utils import (
    binary_scores_from_logits,
    build_step_boundaries,
    calculate_l2_regularization,
    count_trainable_parameters,
    effective_component_count,
    load_checkpoint,
    metric_is_better,
    piecewise_learning_rate,
    prepare_device,
    prior_distribution_metrics,
    save_checkpoint,
    save_run_manifest,
    set_optimizer_learning_rate,
    set_seed,
    write_history,
)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary.")
    return config


def build_model(config: dict[str, Any]) -> GMMProbUNet:
    model_config = config["model"]
    return GMMProbUNet(
        in_channels=int(model_config["in_channels"]),
        num_classes=int(model_config["num_classes"]),
        latent_dim=int(model_config["latent_dim"]),
        num_components=int(model_config["num_components"]),
        num_channels=tuple(
            int(value)
            for value in model_config["num_channels"]
        ),
        num_convs_per_block=int(
            model_config["num_convs_per_block"]
        ),
        num_1x1_convs=int(
            model_config["num_1x1_convs"]
        ),
        min_log_sigma=model_config.get("min_log_sigma"),
        max_log_sigma=model_config.get("max_log_sigma"),
    )


def _empty_accumulator(num_components: int) -> dict[str, Any]:
    return {
        "samples": 0,
        "objective": 0.0,
        "elbo_loss": 0.0,
        "regularization": 0.0,
        "ce_sum": 0.0,
        "ce_mean": 0.0,
        "kl_total": 0.0,
        "kl_categorical": 0.0,
        "kl_gaussian": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "prior_entropy": 0.0,
        "posterior_entropy": 0.0,
        "prior_normalized_entropy": 0.0,
        "posterior_normalized_entropy": 0.0,
        "prior_probability_sum": torch.zeros(
            num_components,
            dtype=torch.float64,
        ),
        "posterior_probability_sum": torch.zeros(
            num_components,
            dtype=torch.float64,
        ),
        "posterior_component_counts": torch.zeros(
            num_components,
            dtype=torch.long,
        ),
        "prior_ged2": 0.0,
        "prior_cross_dice": 0.0,
        "prior_cross_iou": 0.0,
        "prior_best_dice": 0.0,
        "prior_sample_diversity": 0.0,
        "annotation_diversity": 0.0,
        "prior_probability_variance": 0.0,
        "prior_empty_fraction": 0.0,
        "prior_metric_samples": 0,
    }


def _finish_accumulator(
    accumulator: dict[str, Any],
) -> dict[str, Any]:
    count = int(accumulator["samples"])
    if count == 0:
        raise RuntimeError("The loader produced no samples.")

    prior_mean_probabilities = (
        accumulator["prior_probability_sum"] / count
    ).tolist()

    posterior_mean_probabilities = (
        accumulator["posterior_probability_sum"] / count
    ).tolist()

    prior_probability_tensor = torch.tensor(
        prior_mean_probabilities,
        dtype=torch.float32,
    ).unsqueeze(0)

    posterior_probability_tensor = torch.tensor(
        posterior_mean_probabilities,
        dtype=torch.float32,
    ).unsqueeze(0)

    result = {
        "objective_loss": accumulator["objective"] / count,
        "elbo_loss": accumulator["elbo_loss"] / count,
        "regularization_loss": (
            accumulator["regularization"] / count
        ),
        "reconstruction_loss_sum": (
            accumulator["ce_sum"] / count
        ),
        "reconstruction_loss_mean": (
            accumulator["ce_mean"] / count
        ),
        "kl_total": accumulator["kl_total"] / count,
        "kl_categorical": (
            accumulator["kl_categorical"] / count
        ),
        "kl_gaussian": accumulator["kl_gaussian"] / count,
        "dice": accumulator["dice"] / count,
        "iou": accumulator["iou"] / count,
        "prior_entropy": (
            accumulator["prior_entropy"] / count
        ),
        "posterior_entropy": (
            accumulator["posterior_entropy"] / count
        ),
        "prior_normalized_entropy": (
            accumulator["prior_normalized_entropy"] / count
        ),
        "posterior_normalized_entropy": (
            accumulator["posterior_normalized_entropy"] / count
        ),
        "prior_effective_components": float(
            effective_component_count(
                prior_probability_tensor
            ).item()
        ),
        "posterior_effective_components": float(
            effective_component_count(
                posterior_probability_tensor
            ).item()
        ),
        "prior_mean_component_probabilities": (
            prior_mean_probabilities
        ),
        "posterior_mean_component_probabilities": (
            posterior_mean_probabilities
        ),
        "posterior_component_counts": (
            accumulator["posterior_component_counts"].tolist()
        ),
    }

    prior_count = int(accumulator["prior_metric_samples"])
    if prior_count > 0:
        for key in (
            "prior_ged2",
            "prior_cross_dice",
            "prior_cross_iou",
            "prior_best_dice",
            "prior_sample_diversity",
            "annotation_diversity",
            "prior_probability_variance",
            "prior_empty_fraction",
        ):
            result[key] = accumulator[key] / prior_count

    return result


def _update_common_metrics(
    accumulator: dict[str, Any],
    loss_output,
    state,
    masks: torch.Tensor,
    objective_loss: torch.Tensor,
    regularization_loss: torch.Tensor,
) -> None:
    scores = binary_scores_from_logits(
        loss_output.logits.detach(),
        masks,
    )

    batch_size = masks.shape[0]
    accumulator["samples"] += batch_size
    accumulator["objective"] += objective_loss.item() * batch_size
    accumulator["elbo_loss"] += loss_output.loss.item() * batch_size
    accumulator["regularization"] += (
        regularization_loss.item() * batch_size
    )
    accumulator["ce_sum"] += (
        loss_output.reconstruction_loss_sum.item() * batch_size
    )
    accumulator["ce_mean"] += (
        loss_output.reconstruction_loss_mean.item() * batch_size
    )
    accumulator["kl_total"] += (
        loss_output.kl_total_mean.item() * batch_size
    )
    accumulator["kl_categorical"] += (
        loss_output.kl_categorical_mean.item() * batch_size
    )
    accumulator["kl_gaussian"] += (
        loss_output.kl_gaussian_mean.item() * batch_size
    )
    accumulator["dice"] += scores["dice"].sum().item()
    accumulator["iou"] += scores["iou"].sum().item()
    accumulator["prior_entropy"] += (
        loss_output.prior_mixture_entropy_mean.item()
        * batch_size
    )
    accumulator["posterior_entropy"] += (
        loss_output.posterior_mixture_entropy_mean.item()
        * batch_size
    )
    accumulator["prior_normalized_entropy"] += (
        loss_output.prior_normalized_entropy_mean.item()
        * batch_size
    )
    accumulator["posterior_normalized_entropy"] += (
        loss_output.posterior_normalized_entropy_mean.item()
        * batch_size
    )

    accumulator["prior_probability_sum"] += (
        state.prior.mixture_probs.detach().sum(dim=0).cpu().double()
    )
    accumulator["posterior_probability_sum"] += (
        state.posterior.mixture_probs.detach().sum(dim=0).cpu().double()
    )

    accumulator["posterior_component_counts"] += torch.bincount(
        loss_output.posterior_component_indices.detach().cpu(),
        minlength=state.prior.mixture_probs.shape[1],
    )


def train_one_epoch(
    model: GMMProbUNet,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    beta: float,
    temperature: float,
    regularization_weight: float,
    global_step: int,
    boundaries: list[int],
    learning_rate_values: list[float],
    max_batches: int | None,
) -> tuple[dict[str, Any], int]:
    model.train()
    accumulator = _empty_accumulator(model.num_components)

    if max_batches is None:
        iterable = loader
        total_batches = len(loader)
    else:
        total_batches = min(len(loader), max_batches)
        iterable = islice(loader, total_batches)

    progress = tqdm(
        iterable,
        total=total_batches,
        desc=f"Global GMM Train {epoch}",
        ncols=155,
    )

    last_learning_rate = 0.0

    for batch in progress:
        learning_rate = piecewise_learning_rate(
            global_step,
            boundaries,
            learning_rate_values,
        )
        set_optimizer_learning_rate(
            optimizer,
            learning_rate,
        )
        last_learning_rate = learning_rate

        images = batch["image"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        masks = batch["mask"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        state = model(
            image=images,
            segmentation=masks,
            is_training=True,
        )
        loss_output = model.elbo(
            state=state,
            segmentation=masks,
            beta=beta,
            gumbel_temperature=temperature,
        )

        regularization_loss = (
            regularization_weight
            * calculate_l2_regularization(model)
        )
        objective_loss = (
            loss_output.loss + regularization_loss
        )

        if not torch.isfinite(objective_loss):
            raise RuntimeError("Training loss contains NaN or Inf.")

        optimizer.zero_grad(set_to_none=True)
        objective_loss.backward()
        optimizer.step()

        _update_common_metrics(
            accumulator,
            loss_output,
            state,
            masks,
            objective_loss,
            regularization_loss,
        )

        global_step += 1

        completed = max(accumulator["samples"], 1)
        progress.set_postfix(
            {
                "ce": f"{accumulator['ce_mean']/completed:.4f}",
                "kl": f"{accumulator['kl_total']/completed:.4f}",
                "dice": f"{accumulator['dice']/completed:.4f}",
                "lr": f"{learning_rate:.2e}",
            }
        )

    metrics = _finish_accumulator(accumulator)
    metrics["learning_rate"] = last_learning_rate
    return metrics, global_step


@torch.no_grad()
def validate_one_epoch(
    model: GMMProbUNet,
    loader,
    device: torch.device,
    beta: float,
    temperature: float,
    prior_samples: int,
    max_batches: int | None,
) -> dict[str, Any]:
    model.eval()
    accumulator = _empty_accumulator(model.num_components)

    if max_batches is None:
        iterable = loader
        total_batches = len(loader)
    else:
        total_batches = min(len(loader), max_batches)
        iterable = islice(loader, total_batches)

    progress = tqdm(
        iterable,
        total=total_batches,
        desc="Global GMM Val",
        ncols=155,
    )

    for batch in progress:
        images = batch["image"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        masks = batch["mask"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        all_masks = batch["masks"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        state = model(
            image=images,
            segmentation=masks,
            is_training=True,
        )
        loss_output = model.elbo(
            state=state,
            segmentation=masks,
            beta=beta,
            gumbel_temperature=temperature,
        )

        zero_regularization = torch.zeros(
            (),
            device=device,
        )
        _update_common_metrics(
            accumulator,
            loss_output,
            state,
            masks,
            loss_output.loss,
            zero_regularization,
        )

        sample_logits, _ = model.sample(
            state=state,
            num_samples=prior_samples,
            return_component_indices=True,
        )
        prior_metrics = prior_distribution_metrics(
            sample_logits,
            all_masks,
        )

        batch_size = images.shape[0]
        accumulator["prior_metric_samples"] += batch_size

        metric_mapping = {
            "prior_ged2": "ged2",
            "prior_cross_dice": "cross_dice",
            "prior_cross_iou": "cross_iou",
            "prior_best_dice": "best_dice",
            "prior_sample_diversity": "sample_diversity",
            "annotation_diversity": "annotation_diversity",
            "prior_probability_variance": "probability_variance",
            "prior_empty_fraction": "empty_fraction",
        }

        for destination, source in metric_mapping.items():
            accumulator[destination] += (
                prior_metrics[source].sum().item()
            )

        completed = max(
            accumulator["prior_metric_samples"],
            1,
        )
        progress.set_postfix(
            {
                f"GED{prior_samples}": (
                    f"{accumulator['prior_ged2']/completed:.4f}"
                ),
                "dice": (
                    f"{accumulator['dice']/accumulator['samples']:.4f}"
                ),
            }
        )

    return _finish_accumulator(accumulator)


def flatten_epoch_record(
    epoch: int,
    global_step: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
    }

    for prefix, metrics in (
        ("train", train_metrics),
        ("val", val_metrics),
    ):
        for key, value in metrics.items():
            if isinstance(value, list):
                record[f"{prefix}_{key}"] = value
                for index, item in enumerate(value):
                    record[
                        f"{prefix}_{key}_{index}"
                    ] = item
            else:
                record[f"{prefix}_{key}"] = value

    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/global_gmm_prob_unet_lidc.yaml",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    set_seed(int(config["seed"]))
    device = prepare_device(config.get("device", "cuda"))

    training_config = config["training"]
    validation_config = config["validation"]

    output_directory = Path(config["save_dir"])
    if args.smoke_test:
        output_directory = output_directory / "smoke_test"

    output_directory.mkdir(parents=True, exist_ok=True)

    save_run_manifest(
        output_directory=output_directory,
        project_root=PROJECT_ROOT,
        config_path=config_path,
        tracked_files=[
            PROJECT_ROOT
            / "models/gmm_prob_unet/gmm_prob_unet_model.py",
            PROJECT_ROOT
            / "models/gmm_prob_unet/gmm_prob_unet_parts.py",
            PROJECT_ROOT
            / "train/train_global_gmm_prob_unet.py",
            PROJECT_ROOT
            / "utils/gmm_experiment_utils.py",
        ],
        config=config,
    )

    train_loader = build_lidc_dataloader(
        root=config["data_root"],
        split="train",
        batch_size=int(training_config["batch_size"]),
        num_workers=int(training_config["num_workers"]),
        shuffle=True,
        random_rater=True,
    )
    val_loader = build_lidc_dataloader(
        root=config["data_root"],
        split="val",
        batch_size=int(training_config["batch_size"]),
        num_workers=int(training_config["num_workers"]),
        shuffle=False,
        random_rater=False,
        rater_index=int(validation_config["rater_index"]),
    )

    model = build_model(config).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=float(training_config["initial_learning_rate"]),
        weight_decay=0.0,
    )

    configured_epochs = int(training_config["epochs"])
    total_schedule_steps = configured_epochs * len(train_loader)

    boundaries = build_step_boundaries(
        total_schedule_steps,
        training_config["learning_rate_boundary_fractions"],
    )
    learning_rate_values = [
        float(value)
        for value in training_config["learning_rate_values"]
    ]

    epochs = 1 if args.smoke_test else configured_epochs
    max_train_batches = 2 if args.smoke_test else None
    max_val_batches = 2 if args.smoke_test else None

    start_epoch = 1
    global_step = 0
    history: list[dict[str, Any]] = []

    best_records = {
        "val_loss": {
            "value": float("inf"),
            "epoch": None,
            "checkpoint": "best_val_loss_global_gmm.pth",
        },
        "val_dice": {
            "value": float("-inf"),
            "epoch": None,
            "checkpoint": "best_val_dice_global_gmm.pth",
        },
        "val_ged2": {
            "value": float("inf"),
            "epoch": None,
            "checkpoint": "best_val_ged4_global_gmm.pth",
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
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_records = checkpoint.get(
            "best_records",
            best_records,
        )

        history_path = output_directory / "history.json"
        if history_path.exists():
            with open(
                history_path,
                "r",
                encoding="utf-8",
            ) as file:
                history = json.load(file)

        print(f"[INFO] Resumed from {resume_path}")

    print("=" * 80)
    print("Global GMM Probabilistic U-Net training")
    print("=" * 80)
    print(f"[INFO] Device: {device}")
    print(
        f"[INFO] Components: {model.num_components}; "
        f"latent dim: {model.latent_dim}"
    )
    print(
        f"[INFO] Parameters: "
        f"{count_trainable_parameters(model):,}"
    )
    print(
        f"[INFO] Train/Val samples: "
        f"{len(train_loader.dataset)}/{len(val_loader.dataset)}"
    )
    print(f"[INFO] LR boundaries: {boundaries}")
    print(
        f"[INFO] Validation prior samples per epoch: "
        f"{validation_config['prior_samples_each_epoch']}"
    )

    if args.smoke_test:
        print("[INFO] Smoke test: 2 train + 2 val batches.")

    beta = float(training_config["beta"])
    temperature = float(
        training_config["gumbel_temperature"]
    )
    regularization_weight = float(
        training_config["regularization_weight"]
    )

    for epoch in range(start_epoch, epochs + 1):
        train_metrics, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            beta=beta,
            temperature=temperature,
            regularization_weight=regularization_weight,
            global_step=global_step,
            boundaries=boundaries,
            learning_rate_values=learning_rate_values,
            max_batches=max_train_batches,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            beta=beta,
            temperature=temperature,
            prior_samples=int(
                validation_config[
                    "prior_samples_each_epoch"
                ]
            ),
            max_batches=max_val_batches,
        )

        record = flatten_epoch_record(
            epoch,
            global_step,
            train_metrics,
            val_metrics,
        )
        history.append(record)
        write_history(history, output_directory)

        monitored = {
            "val_loss": float(val_metrics["elbo_loss"]),
            "val_dice": float(val_metrics["dice"]),
            "val_ged2": float(val_metrics["prior_ged2"]),
        }

        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"train_ce={train_metrics['reconstruction_loss_mean']:.4f}, "
            f"train_kl={train_metrics['kl_total']:.4f}, "
            f"train_dice={train_metrics['dice']:.4f}, "
            f"val_ce={val_metrics['reconstruction_loss_mean']:.4f}, "
            f"val_kl={val_metrics['kl_total']:.4f}, "
            f"val_dice={val_metrics['dice']:.4f}, "
            f"val_GED4={val_metrics['prior_ged2']:.4f}, "
            f"val_Keff={val_metrics['prior_effective_components']:.3f}"
        )

        for metric_name, value in monitored.items():
            current_best = float(
                best_records[metric_name]["value"]
            )
            if metric_is_better(
                metric_name,
                value,
                current_best,
            ):
                best_records[metric_name]["value"] = value
                best_records[metric_name]["epoch"] = epoch

                save_checkpoint(
                    output_directory
                    / best_records[metric_name]["checkpoint"],
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    config,
                    best_records,
                    record,
                )
                print(
                    f"[INFO] New best {metric_name}: "
                    f"{value:.6f} at epoch {epoch}."
                )

        save_checkpoint(
            output_directory / "latest_global_gmm.pth",
            model,
            optimizer,
            epoch,
            global_step,
            config,
            best_records,
            record,
        )

        save_every = int(
            training_config["save_every_epochs"]
        )
        if epoch % save_every == 0 or epoch == epochs:
            save_checkpoint(
                output_directory
                / "checkpoints"
                / f"epoch_{epoch:03d}.pth",
                model,
                optimizer,
                epoch,
                global_step,
                config,
                best_records,
                record,
            )

        with open(
            output_directory / "best_records.json",
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
        "experiment": config["experiment_name"],
        "completed_epochs": epochs,
        "best_records": best_records,
        "last_epoch": history[-1] if history else None,
        "recommended_checkpoint_for_final_prior_evaluation": (
            best_records["val_ged2"]["checkpoint"]
        ),
    }

    with open(
        output_directory / "training_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("[DONE] Global GMM training finished.")
    print(
        "[INFO] Final prior evaluation checkpoint: "
        f"{best_records['val_ged2']['checkpoint']}"
    )
    print(f"[INFO] Results: {output_directory}")
    print("=" * 80)


if __name__ == "__main__":
    main()
