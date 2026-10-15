"""Formal training entry for EG-GM-PUNet.

This script reuses the validated V3 training/evaluation utilities, while:
1. Building the final EGGMPUNet model.
2. Enabling evidence-guided adaptive latent controls.
3. Supporting joint-checkpoint or probability-only initialization.
4. Logging uncertainty, adaptive KL weights, Gumbel temperature, and
   uncertainty-control correlations.
5. Saving checkpoints under final EG-GM-PUNet names.

Smoke test:
    python train/train_eg_gm_punet.py ^
        --config configs/eg_gm_punet_lidc.yaml ^
        --smoke-test

Full training:
    python train/train_eg_gm_punet.py ^
        --config configs/eg_gm_punet_lidc.yaml

Resume:
    python train/train_eg_gm_punet.py ^
        --config configs/eg_gm_punet_lidc.yaml ^
        --resume result/eg_gm_punet_k5_full/latest_eg_gm_punet.pth
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import torch
from torch.optim import Adam


from datetime import datetime
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


BASE_TRAINING_SCRIPT = (
    PROJECT_ROOT
    / "train"
    / "train_evidential_multiscale_grouped_gmm.py"
)
if not BASE_TRAINING_SCRIPT.exists():
    raise FileNotFoundError(
        "The validated base training script was not found: "
        f"{BASE_TRAINING_SCRIPT}"
    )

_spec = importlib.util.spec_from_file_location(
    "_eg_gm_punet_base_training",
    BASE_TRAINING_SCRIPT,
)
if _spec is None or _spec.loader is None:
    raise ImportError(
        f"Cannot import base training script: {BASE_TRAINING_SCRIPT}"
    )
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


from models.EG_GM_PUNet import EGGMPUNet


class ConfiguredEGGMPUNet(EGGMPUNet):
    """EG-GM-PUNet with training feedback switches stored on the model."""

    def __init__(
        self,
        *args,
        feedback_enabled: bool,
        categorical_feedback: bool,
        group_feedback: bool,
        temperature_feedback: bool,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.training_feedback_enabled = bool(feedback_enabled)
        self.training_categorical_feedback = bool(
            categorical_feedback
        )
        self.training_group_feedback = bool(group_feedback)
        self.training_temperature_feedback = bool(
            temperature_feedback
        )

    def elbo(self, *args, **kwargs):
        kwargs.setdefault(
            "feedback_enabled",
            self.training_feedback_enabled,
        )
        kwargs.setdefault(
            "categorical_feedback",
            self.training_categorical_feedback,
        )
        kwargs.setdefault(
            "group_feedback",
            self.training_group_feedback,
        )
        kwargs.setdefault(
            "temperature_feedback",
            self.training_temperature_feedback,
        )
        return super().elbo(*args, **kwargs)


def build_model(
    config: dict[str, Any],
) -> ConfiguredEGGMPUNet:
    item = config["model"]
    training = config["training"]

    return ConfiguredEGGMPUNet(
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
        feedback_config=item["feedback"],
        feedback_enabled=bool(
            training["feedback_enabled"]
        ),
        categorical_feedback=bool(
            training["categorical_feedback"]
        ),
        group_feedback=bool(
            training["group_feedback"]
        ),
        temperature_feedback=bool(
            training["temperature_feedback"]
        ),
    )


# ---------------------------------------------------------------------
# Extend the validated V3 accumulators with feedback-control diagnostics.
# base.run_epoch resolves these functions from the base module namespace,
# so replacing them keeps all validated loss and validation logic intact.
# ---------------------------------------------------------------------

_BASE_NEW_ACCUMULATOR = base.new_accumulator
_BASE_UPDATE_ACCUMULATOR = base.update_accumulator
_BASE_FINISH_ACCUMULATOR = base.finish_accumulator


def _new_pair_stats() -> dict[str, float]:
    return {
        "n": 0.0,
        "sum_x": 0.0,
        "sum_y": 0.0,
        "sum_x2": 0.0,
        "sum_y2": 0.0,
        "sum_xy": 0.0,
    }


def _update_pair_stats(
    stats: dict[str, float],
    x: torch.Tensor,
    y: torch.Tensor,
) -> None:
    x_cpu = x.detach().float().reshape(-1).cpu()
    y_cpu = y.detach().float().reshape(-1).cpu()
    if x_cpu.numel() != y_cpu.numel():
        raise ValueError(
            "Correlation tensors must contain the same number of values."
        )

    stats["n"] += float(x_cpu.numel())
    stats["sum_x"] += float(x_cpu.sum().item())
    stats["sum_y"] += float(y_cpu.sum().item())
    stats["sum_x2"] += float((x_cpu * x_cpu).sum().item())
    stats["sum_y2"] += float((y_cpu * y_cpu).sum().item())
    stats["sum_xy"] += float((x_cpu * y_cpu).sum().item())


def _pearson(stats: dict[str, float]) -> float | None:
    n = stats["n"]
    if n <= 1.0:
        return None

    numerator = (
        n * stats["sum_xy"]
        - stats["sum_x"] * stats["sum_y"]
    )
    var_x = (
        n * stats["sum_x2"]
        - stats["sum_x"] ** 2
    )
    var_y = (
        n * stats["sum_y2"]
        - stats["sum_y"] ** 2
    )
    denominator = math.sqrt(
        max(var_x, 0.0) * max(var_y, 0.0)
    )
    if denominator <= 1e-12:
        return None
    return float(numerator / denominator)


def new_accumulator(
    model: ConfiguredEGGMPUNet,
    ece_bins: int,
    ranking_bins: int,
) -> dict[str, Any]:
    accumulator = _BASE_NEW_ACCUMULATOR(
        model=model,
        ece_bins=ece_bins,
        ranking_bins=ranking_bins,
    )

    control_names = (
        "beta_categorical",
        "beta_coarse",
        "beta_middle",
        "beta_fine",
        "temperature",
    )
    uncertainty_names = (
        "categorical",
        "coarse",
        "middle",
        "fine",
    )

    accumulator["feedback_ramp_sum"] = 0.0
    accumulator["feedback_control_sum"] = {
        name: 0.0 for name in control_names
    }
    accumulator["feedback_control_min"] = {
        name: float("inf") for name in control_names
    }
    accumulator["feedback_control_max"] = {
        name: float("-inf") for name in control_names
    }
    accumulator["feedback_uncertainty_sum"] = {
        name: 0.0 for name in uncertainty_names
    }
    accumulator["feedback_uncertainty_min"] = {
        name: float("inf") for name in uncertainty_names
    }
    accumulator["feedback_uncertainty_max"] = {
        name: float("-inf") for name in uncertainty_names
    }
    accumulator["feedback_correlations"] = {
        "categorical_beta": _new_pair_stats(),
        "categorical_temperature": _new_pair_stats(),
        "coarse_beta": _new_pair_stats(),
        "middle_beta": _new_pair_stats(),
        "fine_beta": _new_pair_stats(),
    }
    return accumulator


def update_accumulator(
    accumulator: dict[str, Any],
    output,
    state,
    segmentation: torch.Tensor,
    all_segmentations: torch.Tensor,
    objective_loss: torch.Tensor,
    regularization_loss: torch.Tensor,
    model: ConfiguredEGGMPUNet,
) -> None:
    _BASE_UPDATE_ACCUMULATOR(
        accumulator=accumulator,
        output=output,
        state=state,
        segmentation=segmentation,
        all_segmentations=all_segmentations,
        objective_loss=objective_loss,
        regularization_loss=regularization_loss,
        model=model,
    )

    batch_size = int(segmentation.shape[0])
    controls = output.adaptive_controls

    ramp = float(
        controls.feedback_ramp.detach().cpu().item()
    )
    accumulator["feedback_ramp_sum"] += (
        ramp * batch_size
    )

    control_tensors = {
        "beta_categorical": controls.beta_categorical,
        "beta_coarse": controls.beta_groups["coarse"],
        "beta_middle": controls.beta_groups["middle"],
        "beta_fine": controls.beta_groups["fine"],
        "temperature": controls.gumbel_temperature,
    }
    uncertainty_tensors = {
        name: output.feedback_signals[name]
        for name in (
            "categorical",
            "coarse",
            "middle",
            "fine",
        )
    }

    for name, value in control_tensors.items():
        detached = value.detach().float()
        accumulator["feedback_control_sum"][name] += float(
            detached.sum().item()
        )
        accumulator["feedback_control_min"][name] = min(
            accumulator["feedback_control_min"][name],
            float(detached.min().item()),
        )
        accumulator["feedback_control_max"][name] = max(
            accumulator["feedback_control_max"][name],
            float(detached.max().item()),
        )

    for name, value in uncertainty_tensors.items():
        detached = value.detach().float()
        accumulator["feedback_uncertainty_sum"][name] += float(
            detached.sum().item()
        )
        accumulator["feedback_uncertainty_min"][name] = min(
            accumulator["feedback_uncertainty_min"][name],
            float(detached.min().item()),
        )
        accumulator["feedback_uncertainty_max"][name] = max(
            accumulator["feedback_uncertainty_max"][name],
            float(detached.max().item()),
        )

    pairs = accumulator["feedback_correlations"]
    _update_pair_stats(
        pairs["categorical_beta"],
        uncertainty_tensors["categorical"],
        control_tensors["beta_categorical"],
    )
    _update_pair_stats(
        pairs["categorical_temperature"],
        uncertainty_tensors["categorical"],
        control_tensors["temperature"],
    )
    for name in ("coarse", "middle", "fine"):
        _update_pair_stats(
            pairs[f"{name}_beta"],
            uncertainty_tensors[name],
            control_tensors[f"beta_{name}"],
        )


def finish_accumulator(
    accumulator: dict[str, Any],
    model: ConfiguredEGGMPUNet,
) -> dict[str, Any]:
    result = _BASE_FINISH_ACCUMULATOR(
        accumulator=accumulator,
        model=model,
    )
    count = int(accumulator["samples"])
    if count <= 0:
        raise RuntimeError("No feedback samples were accumulated.")

    result["feedback_ramp"] = (
        accumulator["feedback_ramp_sum"] / count
    )

    for name in accumulator["feedback_control_sum"]:
        result[f"adaptive_{name}"] = (
            accumulator["feedback_control_sum"][name]
            / count
        )
        result[f"adaptive_{name}_min"] = (
            accumulator["feedback_control_min"][name]
        )
        result[f"adaptive_{name}_max"] = (
            accumulator["feedback_control_max"][name]
        )

    for name in accumulator["feedback_uncertainty_sum"]:
        result[f"feedback_uncertainty_{name}"] = (
            accumulator["feedback_uncertainty_sum"][name]
            / count
        )
        result[f"feedback_uncertainty_{name}_min"] = (
            accumulator["feedback_uncertainty_min"][name]
        )
        result[f"feedback_uncertainty_{name}_max"] = (
            accumulator["feedback_uncertainty_max"][name]
        )

    pairs = accumulator["feedback_correlations"]
    result["corr_uncertainty_beta_categorical"] = _pearson(
        pairs["categorical_beta"]
    )
    result["corr_uncertainty_temperature"] = _pearson(
        pairs["categorical_temperature"]
    )
    for name in ("coarse", "middle", "fine"):
        result[f"corr_uncertainty_beta_{name}"] = _pearson(
            pairs[f"{name}_beta"]
        )

    return result


# Make the validated base epoch loop use the final-model extensions.
base.new_accumulator = new_accumulator
base.update_accumulator = update_accumulator
base.finish_accumulator = finish_accumulator


# Replace the old "V3 Train/Val" progress prefix without copying the
# validated run_epoch implementation.
_ORIGINAL_TQDM = base.tqdm


def _eg_tqdm(*args, **kwargs):
    description = kwargs.get("desc")
    if isinstance(description, str):
        kwargs["desc"] = description.replace(
            "V3 ",
            "EG-GM-PUNet ",
            1,
        )
    return _ORIGINAL_TQDM(*args, **kwargs)


base.tqdm = _eg_tqdm


def _path_or_none(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def resolve_initialization(
    config: dict[str, Any],
) -> tuple[str, Path]:
    initialization = config["initialization"]
    mode = str(
        initialization.get("mode", "auto")
    ).strip().lower()

    joint_path = _path_or_none(
        initialization.get("joint_checkpoint")
    )
    probability_path = _path_or_none(
        initialization.get("probabilistic_checkpoint")
    )

    if mode not in {
        "auto",
        "joint",
        "probabilistic",
    }:
        raise ValueError(
            "initialization.mode must be auto, joint, or probabilistic."
        )

    if mode == "joint":
        if joint_path is None or not joint_path.exists():
            raise FileNotFoundError(
                "Joint initialization checkpoint not found: "
                f"{joint_path}"
            )
        return "joint", joint_path

    if mode == "probabilistic":
        if (
            probability_path is None
            or not probability_path.exists()
        ):
            raise FileNotFoundError(
                "Probability initialization checkpoint not found: "
                f"{probability_path}"
            )
        return "probabilistic", probability_path

    if joint_path is not None and joint_path.exists():
        return "joint", joint_path
    if (
        probability_path is not None
        and probability_path.exists()
    ):
        return "probabilistic", probability_path

    raise FileNotFoundError(
        "No usable initialization checkpoint was found. "
        f"joint={joint_path}, probabilistic={probability_path}"
    )


def apply_initialization_schedule(
    config: dict[str, Any],
    initialization_kind: str,
) -> None:
    initialization = config["initialization"]
    feedback = config["model"]["feedback"]
    training = config["training"]

    if initialization_kind == "joint":
        feedback["start_epoch"] = int(
            initialization.get(
                "joint_feedback_start_epoch",
                feedback.get("start_epoch", 1),
            )
        )
        training["freeze_probabilistic_epochs"] = int(
            initialization.get(
                "joint_freeze_probabilistic_epochs",
                training.get(
                    "freeze_probabilistic_epochs",
                    2,
                ),
            )
        )
    elif initialization_kind == "probabilistic":
        feedback["start_epoch"] = int(
            initialization.get(
                "probabilistic_feedback_start_epoch",
                21,
            )
        )
        training["freeze_probabilistic_epochs"] = int(
            initialization.get(
                "probabilistic_freeze_probabilistic_epochs",
                5,
            )
        )
    else:
        raise ValueError(
            f"Unsupported initialization kind: {initialization_kind}"
        )


def build_best_records() -> dict[str, dict[str, Any]]:
    return {
        "val_joint_loss": {
            "value": float("inf"),
            "epoch": None,
            "checkpoint": (
                "best_val_joint_loss_eg_gm_punet.pth"
            ),
        },
        "val_probabilistic_dice": {
            "value": float("-inf"),
            "epoch": None,
            "checkpoint": (
                "best_val_probabilistic_dice_eg_gm_punet.pth"
            ),
        },
        "val_prior_ged4": {
            "value": float("inf"),
            "epoch": None,
            "checkpoint": (
                "best_val_ged4_eg_gm_punet.pth"
            ),
        },
        "val_evidence_dice": {
            "value": float("-inf"),
            "epoch": None,
            "checkpoint": (
                "best_val_evidence_dice_eg_gm_punet.pth"
            ),
        },
        "val_evidence_ece": {
            "value": float("inf"),
            "epoch": None,
            "checkpoint": (
                "best_val_evidence_ece_eg_gm_punet.pth"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/eg_gm_punet_lidc.yaml",
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
    config = base.load_config(config_path)

    base.set_seed(int(config["seed"]))
    device = base.prepare_device(
        config.get("device", "cuda")
    )

    resume_path = (
        args.resume
        or config.get("resume")
    )

    initialization_kind: str | None = None
    initialization_path: Path | None = None
    if not resume_path:
        (
            initialization_kind,
            initialization_path,
        ) = resolve_initialization(config)
        apply_initialization_schedule(
            config,
            initialization_kind,
        )

    training_config = config["training"]
    validation_config = config["validation"]

    output_directory = Path(config["save_dir"])
    if args.smoke_test:
        output_directory = (
            output_directory / "smoke_test"
        )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    base.save_run_manifest(
        output_directory=output_directory,
        project_root=PROJECT_ROOT,
        config_path=config_path,
        tracked_files=[
            PROJECT_ROOT
            / "models/EG_GM_PUNet/eg_gm_punet_model.py",
            PROJECT_ROOT
            / "models/EG_GM_PUNet/feedback_controller.py",
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
            / "train/train_eg_gm_punet.py",
            PROJECT_ROOT
            / "utils/evidential_metrics.py",
        ],
        config=config,
    )

    train_loader = base.build_lidc_dataloader(
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

    val_loader = base.build_lidc_dataloader(
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
                    model.probabilistic_model.parameters()
                ),
                "lr": float(
                    training_config[
                        "probabilistic_learning_rate_values"
                    ][0]
                ),
                "group_name": "probabilistic",
            },
            {
                "params": model.evidence_head.parameters(),
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
        1 if args.smoke_test else configured_epochs
    )

    total_steps = (
        configured_epochs * len(train_loader)
    )
    boundaries = base.build_step_boundaries(
        total_steps,
        training_config[
            "learning_rate_boundary_fractions"
        ],
    )

    probability_lr_values = [
        float(value)
        for value in training_config[
            "probabilistic_learning_rate_values"
        ]
    ]
    evidence_lr_values = [
        float(value)
        for value in training_config[
            "evidence_learning_rate_values"
        ]
    ]

    start_epoch = 1
    global_step = 0
    history: list[dict[str, Any]] = []
    best_records = build_best_records()

    if resume_path:
        checkpoint = base.load_checkpoint(
            Path(resume_path),
            model,
            optimizer,
            device,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(
            checkpoint.get("global_step", 0)
        )
        best_records = checkpoint.get(
            "best_records",
            best_records,
        )

        history_path = (
            output_directory / "history.json"
        )
        if history_path.exists():
            with open(
                history_path,
                "r",
                encoding="utf-8",
            ) as file:
                history = json.load(file)

        initialization_kind = str(
            checkpoint.get(
                "initialization_kind",
                "resume",
            )
        )
        initialization_path = Path(resume_path)
        print(
            "[INFO] Resumed EG-GM-PUNet from "
            f"{resume_path}"
        )
    else:
        assert initialization_kind is not None
        assert initialization_path is not None
        strict = bool(
            config["initialization"].get(
                "strict",
                True,
            )
        )

        if initialization_kind == "joint":
            checkpoint = model.load_joint_checkpoint(
                initialization_path,
                map_location=device,
                strict=strict,
            )
            print(
                "[INFO] Loaded joint GMM + Evidential "
                f"checkpoint: {initialization_path}"
            )
        else:
            checkpoint = (
                model.load_probabilistic_checkpoint(
                    initialization_path,
                    map_location=device,
                    strict=strict,
                )
            )
            print(
                "[INFO] Loaded probability checkpoint: "
                f"{initialization_path}"
            )
            print(
                "[INFO] Evidence head starts from random "
                "initialization; feedback remains neutral "
                "during the configured warm-up."
            )

        print(
            "[INFO] Initialization checkpoint epoch: "
            f"{checkpoint.get('epoch')}"
        )

    print("=" * 96)
    print("EG-GM-PUNet formal training")
    print("=" * 96)
    print(f"[INFO] Device: {device}")
    print(
        f"[INFO] Parameters total: "
        f"{base.count_trainable_parameters(model):,}"
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
        f"[INFO] Initialization: "
        f"{initialization_kind}"
    )
    print(
        "[INFO] Evidence feedback: "
        f"{training_config['feedback_enabled']}"
    )
    print(
        "[INFO] Feedback starts at epoch: "
        f"{config['model']['feedback']['start_epoch']}"
    )
    print(
        "[INFO] Feedback ramp epochs: "
        f"{config['model']['feedback']['ramp_epochs']}"
    )
    print(
        "[INFO] Probability freeze epochs: "
        f"{training_config['freeze_probabilistic_epochs']}"
    )
    print(f"[INFO] LR boundaries: {boundaries}")

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
        train_metrics, global_step = base.run_epoch(
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
            evidence_lr_values=evidence_lr_values,
            max_batches=max_train_batches,
        )

        rng_state = base.capture_rng_state()
        base.set_seed(int(config["validation_seed"]))

        val_metrics, _ = base.run_epoch(
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
            evidence_lr_values=evidence_lr_values,
            max_batches=max_val_batches,
        )

        base.restore_rng_state(rng_state)

        record = base.flatten_record(
            epoch=epoch,
            global_step=global_step,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )
        record["initialization_kind"] = (
            initialization_kind
        )
        history.append(record)
        base.write_history(
            history,
            output_directory,
        )

        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"frozen={train_metrics['probabilistic_frozen']}, "
            f"ramp={train_metrics['feedback_ramp']:.3f}, "
            f"train_pDice={train_metrics['probabilistic_dice']:.4f}, "
            f"train_eDice={train_metrics['evidence_hard_dice_all_raters']:.4f}, "
            f"val_pDice={val_metrics['probabilistic_dice']:.4f}, "
            f"val_eDice={val_metrics['evidence_hard_dice_all_raters']:.4f}, "
            f"val_GED4={val_metrics['prior_ged2']:.4f}, "
            f"val_ECE={val_metrics['evidence_ece']:.4f}"
        )
        print(
            "  [Feedback] "
            f"uC={train_metrics['feedback_uncertainty_categorical']:.3f}, "
            f"betaC={train_metrics['adaptive_beta_categorical']:.3f}, "
            f"betaCoarse={train_metrics['adaptive_beta_coarse']:.3f}, "
            f"betaMiddle={train_metrics['adaptive_beta_middle']:.3f}, "
            f"betaFine={train_metrics['adaptive_beta_fine']:.3f}, "
            f"tau={train_metrics['adaptive_temperature']:.3f}"
        )

        monitored = {
            "val_joint_loss": (
                float(val_metrics["joint_loss"]),
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
                float(val_metrics["prior_ged2"]),
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
                    val_metrics["evidence_ece"]
                ),
                "min",
            ),
        }

        for name, (value, direction) in monitored.items():
            best_value = float(
                best_records[name]["value"]
            )
            improved = (
                value < best_value
                if direction == "min"
                else value > best_value
            )

            if improved:
                best_records[name]["value"] = value
                best_records[name]["epoch"] = epoch

                base.save_checkpoint(
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
                    best_records=best_records,
                    metrics=record,
                )
                print(
                    f"[INFO] New best {name}="
                    f"{value:.6f} at epoch {epoch}."
                )

        base.save_checkpoint(
            path=(
                output_directory
                / "latest_eg_gm_punet.pth"
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
            training_config["save_every_epochs"]
        )
        if (
            epoch % save_every == 0
            or epoch == epochs
        ):
            base.save_checkpoint(
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
                best_records=best_records,
                metrics=record,
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
        "initialization_kind": (
            initialization_kind
        ),
        "initialization_checkpoint": (
            str(initialization_path)
            if initialization_path is not None
            else None
        ),
        "feedback_start_epoch": int(
            config["model"]["feedback"][
                "start_epoch"
            ]
        ),
        "feedback_ramp_epochs": int(
            config["model"]["feedback"][
                "ramp_epochs"
            ]
        ),
        "best_records": best_records,
        "last_epoch": (
            history[-1] if history else None
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
        "evidence_feedback_used": bool(
    training_config["feedback_enabled"]
    and config["model"]["feedback"]["enabled"]
),
    }

    with open(
        output_directory / "training_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 96)
    print("[DONE] EG-GM-PUNet training finished.")
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

    beijing_time = datetime.now(
        ZoneInfo("Asia/Shanghai")
    )
    print(
        "[INFO] Completion time: "
        f"{beijing_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("=" * 96)


if __name__ == "__main__":
    main()


