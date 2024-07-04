"""Manual-edit training entry point: paper-sized default and isolated loss ablations."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from stairnetv2.loss import StairNetV2Loss
# OPTIONAL: from stairnetv2.loss_alternatives import BinaryFocalLoss, BCEPlusDiceLoss
# OPTIONAL: from stairnetv2.initialization import initialize_he_normal, initialize_normal
from stairnetv2.metrics import (
    BinaryCellCounts,
    LocalizationErrors,
    binary_cell_counts,
    positive_location_errors,
)
from stairnetv2.model import StairNetV2
from stairnetv2.training_policy import EarlyStopping, validation_score
from stairnetv2.tiny_overfit import resolve_device
from stairnetv2.training_dataset import StairNetDataset


@dataclass(frozen=True)
class TrainingConfig:
    dataset_root: str
    output_dir: str = "artifacts/full_training"
    epochs: int = 200
    batch_size: int = 4
    learning_rate: float = 1e-3
    lr_step_size: int = 50
    lr_gamma: float = 0.5
    flip_probability: float = 0.5
    alpha: float = 10.0
    beta: float = 10.0
    sigma: float = 0.5
    seed: int = 202302175
    num_workers: int = 0
    device: str = "auto"
    resume: str | None = None
    max_train_batches: int | None = None
    max_validation_batches: int | None = None

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.lr_step_size < 1:
            raise ValueError("learning_rate and lr_step_size must be positive")
        if not 0.0 < self.lr_gamma <= 1.0:
            raise ValueError("lr_gamma must be in (0,1]")
        if not 0.0 <= self.flip_probability <= 1.0:
            raise ValueError("flip_probability must be in [0,1]")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.max_train_batches is not None and self.max_train_batches < 1:
            raise ValueError("max_train_batches must be positive when provided")
        if (
            self.max_validation_batches is not None
            and self.max_validation_batches < 1
        ):
            raise ValueError("max_validation_batches must be positive when provided")


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _move_batch(loaded: dict[str, object], device: torch.device) -> dict[str, Tensor]:
    return {
        key: loaded[key].to(device, non_blocking=device.type == "cuda")  # type: ignore[union-attr]
        for key in ("rgb", "depth", "classification", "location")
    }


def run_epoch(
    model: nn.Module,
    criterion: StairNetV2Loss,
    loader: Iterable[dict[str, object]],
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive when provided")
    training = optimizer is not None
    model.train(training)
    loss_sums = {"total": 0.0, "classification": 0.0, "x": 0.0, "y": 0.0}
    samples = 0
    counts = BinaryCellCounts()
    location_errors = LocalizationErrors(0.0, 0.0, 0)

    processed_batches = 0
    for loaded in loader:
        batch = _move_batch(loaded, device)
        batch_size = int(batch["rgb"].shape[0])
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            predictions = model(batch["rgb"], batch["depth"])
            losses = criterion(
                predictions["classification_logits"],
                predictions["location"],
                batch["classification"],
                batch["location"],
            )
            if training:
                losses.total.backward()
                optimizer.step()
        samples += batch_size
        loss_sums["total"] += float(losses.total.detach().item()) * batch_size
        loss_sums["classification"] += (
            float(losses.classification.detach().item()) * batch_size
        )
        loss_sums["x"] += float(losses.location_x.detach().item()) * batch_size
        loss_sums["y"] += float(losses.location_y.detach().item()) * batch_size
        counts += binary_cell_counts(
            predictions["classification_logits"].detach(), batch["classification"]
        )
        location_errors += positive_location_errors(
            predictions["location"].detach(),
            batch["location"],
            batch["classification"],
        )
        processed_batches += 1
        if max_batches is not None and processed_batches >= max_batches:
            break

    if samples == 0:
        raise ValueError("epoch loader produced no samples")
    return {
        "samples": samples,
        "batches": processed_batches,
        "total_loss": loss_sums["total"] / samples,
        "classification_loss": loss_sums["classification"] / samples,
        "location_x_grid_loss": loss_sums["x"] / samples,
        "location_y_grid_loss": loss_sums["y"] / samples,
        "true_positive": counts.true_positive,
        "false_positive": counts.false_positive,
        "false_negative": counts.false_negative,
        **counts.compute(),
        "location_x_error_sum": location_errors.x_error_sum,
        "location_y_error_sum": location_errors.y_error_sum,
        "positive_cells": location_errors.positive_cells,
        **location_errors.coordinate_means(),
    }


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def run_full_training(config: TrainingConfig) -> dict[str, Any]:
    config.validate()
    device = resolve_device(config.device)
    generator = seed_everything(config.seed)
    root = Path(config.dataset_root).expanduser().resolve()
    output = Path(config.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.jsonl"
    if config.resume is None and history_path.exists():
        raise FileExistsError(
            f"refusing to append a new run to existing history: {history_path}"
        )

    train_dataset = StairNetDataset(
        root,
        split="train",
        augment=True,
        flip_probability=config.flip_probability,
        overflow_policy="longest",
        rgb_normalization="baseline",  # 'unit' ONLY for FrozenResNet18Stem
        depth_normalization="baseline",  # verified 8-bit L sample; full dataset unchecked
    )
    validation_dataset = StairNetDataset(
        root, split="val", augment=False, overflow_policy="longest",
        rgb_normalization="baseline",  # match training setting
        depth_normalization="baseline",
    )
    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_options
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_options
    )

    model = StairNetV2().to(device)
    # OPTIONAL: initialize_he_normal(model) etc. Avoid reinitializing
    # frozen pretrained modules; initialization.py skips requires_grad=False.
    criterion = StairNetV2Loss(
        alpha=config.alpha,
        beta=config.beta,
        sigma=config.sigma,
        mask_mode="ground_truth",
        # OPTIONAL manually substitute either module here (ONE ablation/run):
        # classification_loss_fn=BinaryFocalLoss(gamma=2, alpha=0.25),
        # classification_loss_fn=BCEPlusDiceLoss(dice_weight=1),
        # location_loss_fn=nn.L1Loss(reduction="none"),
        # location_loss_fn=nn.SmoothL1Loss(beta=0.05, reduction="none"),
    ).to(device)
    # Adam betas and weight decay are absent from the paper. PyTorch defaults
    # (betas 0.9/0.999 and weight_decay 0) are an explicit implementation choice.
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.0)
    # OPTIONAL: manually change weight_decay above to 1e-4.
    scheduler = StepLR(
        optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma
    )
    # OPTIONAL: scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=8)
    stopper = None
    # OPTIONAL: stopper = EarlyStopping(patience=20, min_delta=1e-3)
    start_epoch = 1
    best_iou = -1.0
    best_score = -float("inf")

    if config.resume is not None:
        checkpoint = torch.load(
            Path(config.resume).expanduser().resolve(),
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        criterion.load_state_dict(checkpoint["criterion"])
        generator.set_state(checkpoint["data_generator_state"].cpu())
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"].cpu())
        start_epoch = int(checkpoint["epoch"]) + 1
        best_iou = float(checkpoint["best_iou"])
        best_score = float(checkpoint.get("best_score", -float("inf")))
        if stopper is not None and checkpoint.get("early_stopping") is not None:
            stopper.best = float(checkpoint["early_stopping"]["best"])
            stopper.bad_epochs = int(checkpoint["early_stopping"]["bad_epochs"])

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, config.epochs + 1):
        last_epoch = epoch
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = run_epoch(
            model,
            criterion,
            train_loader,
            device,
            optimizer=optimizer,
            max_batches=config.max_train_batches,
        )
        validation_metrics = run_epoch(
            model,
            criterion,
            validation_loader,
            device,
            optimizer=None,
            max_batches=config.max_validation_batches,
        )
        # Stable checkpoint/scheduler score: independent of dynamic alpha/beta.
        score = validation_score(validation_metrics)
        weight_update = criterion.update_weights(
            float(validation_metrics["location_x_error_sum"]),
            float(validation_metrics["location_y_error_sum"]),
        )
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(score)
        else:
            scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "validation": validation_metrics,
            "dynamic_weights": asdict(weight_update),
            "validation_combined_score": score,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

        validation_iou = float(validation_metrics["iou"])
        is_best = validation_iou > best_iou
        best_iou = max(best_iou, validation_iou)
        is_best_score = score > best_score
        best_score = max(best_score, score)
        should_stop = stopper.step(score) if stopper is not None else False
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "criterion": criterion.state_dict(),
            "data_generator_state": generator.get_state(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "best_iou": best_iou,
            "best_score": best_score,
            "early_stopping": (
                {"best": stopper.best, "bad_epochs": stopper.bad_epochs}
                if stopper is not None else None
            ),
            "config": asdict(config),
            "last_record": record,
        }
        _save_checkpoint(output / "latest.pt", state)
        # A pure tensor state dict can be inspected using weights_only=True;
        # full training checkpoints also include NumPy/Python RNG objects.
        _save_checkpoint(output / "model_weights.pt", model.state_dict())
        if is_best:
            _save_checkpoint(output / "best_iou.pt", state)
        if is_best_score:
            _save_checkpoint(output / "best_combined.pt", state)
        if should_stop:
            break

    result = {
        "completed_epochs": max(0, last_epoch - start_epoch + 1),
        "last_epoch": last_epoch,
        "best_iou": best_iou,
        "best_combined_score": best_score,
        "device": str(device),
        "history": str(history_path),
        "latest_checkpoint": str(output / "latest.pt"),
        "weights_only_checkpoint": str(output / "model_weights.pt"),
        "best_checkpoint": str(output / "best_iou.pt"),
        "best_combined_checkpoint": str(output / "best_combined.pt"),
        "truncated_batches": (
            config.max_train_batches is not None
            or config.max_validation_batches is not None
        ),
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", default="artifacts/full_training")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-step-size", type=int, default=50)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--flip-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=202302175)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_full_training(TrainingConfig(**vars(args)))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
