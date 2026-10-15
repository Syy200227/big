"""Final prior evaluation for the Global GMM Probabilistic U-Net."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from dataset.lidc_dataset import build_lidc_dataloader
from models.gmm_prob_unet.gmm_prob_unet_model import GMMProbUNet
from utils.gmm_experiment_utils import prepare_device, set_seed
from utils.gmm_prior_evaluation import (
    evaluate_prior,
    normalize_sample_counts,
)


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/global_gmm_prob_unet_lidc.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "result/v2_global_gmm_prob_unet_k5/"
            "best_val_ged4_global_gmm.pth"
        ),
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
    )
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument(
        "--sample-counts",
        type=int,
        nargs="*",
        default=None,
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-examples", type=int, default=5)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    set_seed(int(config["seed"]))
    device = prepare_device(config.get("device", "cuda"))

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    checkpoint_config = checkpoint.get("config", config)
    model = build_model(checkpoint_config).to(device)
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.eval()

    training_config = config["training"]
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else training_config["batch_size"]
    )
    num_workers = int(
        args.num_workers
        if args.num_workers is not None
        else training_config["num_workers"]
    )

    loader = build_lidc_dataloader(
        root=config["data_root"],
        split=args.split,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        random_rater=False,
        rater_index=int(
            config["validation"]["rater_index"]
        ),
    )

    sample_counts = normalize_sample_counts(
        args.sample_counts,
        args.num_samples,
    )

    if args.output_dir:
        output_directory = Path(args.output_dir)
    else:
        output_directory = (
            Path(config["save_dir"])
            / "evaluation"
            / args.split
            / f"samples_{args.num_samples}"
        )

    print("=" * 80)
    print("Global GMM Probabilistic U-Net prior evaluation")
    print("=" * 80)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"[INFO] Split: {args.split}")
    print(f"[INFO] Cases: {len(loader.dataset)}")
    print(
        f"[INFO] Samples/counts: "
        f"{args.num_samples}/{sample_counts}"
    )

    result = evaluate_prior(
        model=model,
        loader=loader,
        device=device,
        split=args.split,
        num_samples=args.num_samples,
        sample_counts=sample_counts,
        output_directory=output_directory,
        save_examples=max(0, int(args.save_examples)),
    )

    result["model"] = "Global GMM Probabilistic U-Net"
    result["checkpoint"] = str(checkpoint_path.resolve())

    print(f"[INFO] Results: {output_directory}")
    print("[DONE] Global GMM prior evaluation finished.")


if __name__ == "__main__":
    main()
