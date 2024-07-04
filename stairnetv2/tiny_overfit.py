"""Mandatory tiny-set memorization gate before full StairNetV2 training."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from stairnetv2.dataset import build_pairing_report
from stairnetv2.label_encoder import read_stair_lines
from stairnetv2.loss import StairNetV2Loss
from stairnetv2.metrics import binary_cell_counts, positive_location_errors
from stairnetv2.model import StairNetV2
from stairnetv2.training_dataset import StairNetDataset
from stairnetv2.visualize_predictions import render_prediction_comparison


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_batch(
    model: StairNetV2,
    criterion: StairNetV2Loss,
    batch: dict[str, Tensor],
) -> tuple[dict[str, float], dict[str, Tensor]]:
    model.eval()
    with torch.no_grad():
        predictions = model(batch["rgb"], batch["depth"])
        losses = criterion(
            predictions["classification_logits"],
            predictions["location"],
            batch["classification"],
            batch["location"],
        )
        counts = binary_cell_counts(
            predictions["classification_logits"], batch["classification"]
        )
        location_errors = positive_location_errors(
            predictions["location"],
            batch["location"],
            batch["classification"],
        )
    metrics = {
        "total_loss": float(losses.total.item()),
        "classification_loss": float(losses.classification.item()),
        "location_x_grid_loss": float(losses.location_x.item()),
        "location_y_grid_loss": float(losses.location_y.item()),
        **counts.compute(),
        **location_errors.coordinate_means(),
    }
    return metrics, predictions


def passes_gate(metrics: dict[str, float]) -> bool:
    return (
        metrics["classification_loss"] <= 0.1
        and metrics["precision"] >= 0.98
        and metrics["recall"] >= 0.98
        and metrics["iou"] >= 0.98
        and metrics["location_mae_x"] <= 0.03
        and metrics["location_mae_y"] <= 0.03
    )


def run_tiny_overfit(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    sample_count: int = 4,
    max_steps: int = 500,
    evaluate_every: int = 10,
    learning_rate: float = 1e-3,
    seed: int = 202302175,
    device_name: str = "auto",
) -> dict[str, Any]:
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2 for training BatchNorm")
    if max_steps < 1 or evaluate_every < 1:
        raise ValueError("max_steps and evaluate_every must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(device_name)
    root = Path(dataset_root).expanduser().resolve()
    report = build_pairing_report(
        root / "train" / "images",
        root / "train" / "depthes",
        root / "train" / "labels",
    )
    if len(report.samples) < sample_count:
        raise ValueError("training split is smaller than requested tiny set")
    selected = random.Random(seed).sample(list(report.samples), sample_count)
    sample_ids = [sample.sample_id for sample in selected]
    dataset = StairNetDataset(
        root,
        split="train",
        augment=False,
        overflow_policy="longest",
        sample_ids=sample_ids,
    )
    loaded = next(
        iter(DataLoader(dataset, batch_size=sample_count, shuffle=False, num_workers=0))
    )
    batch = {
        key: loaded[key].to(device)
        for key in ("rgb", "depth", "classification", "location")
    }
    model = StairNetV2().to(device)
    criterion = StairNetV2Loss(mask_mode="ground_truth").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    initial, _ = evaluate_batch(model, criterion, batch)
    history: list[dict[str, float]] = [{"step": 0, **initial}]
    final = initial
    final_predictions: dict[str, Tensor] | None = None
    completed_steps = 0

    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predictions = model(batch["rgb"], batch["depth"])
        losses = criterion(
            predictions["classification_logits"],
            predictions["location"],
            batch["classification"],
            batch["location"],
        )
        losses.total.backward()
        optimizer.step()
        completed_steps = step
        if step % evaluate_every == 0 or step == max_steps:
            final, final_predictions = evaluate_batch(model, criterion, batch)
            entry = {"step": step, **final}
            history.append(entry)
            print(json.dumps(entry), flush=True)
            if passes_gate(final):
                break

    if final_predictions is None:
        final, final_predictions = evaluate_batch(model, criterion, batch)
    success = passes_gate(final)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    probability = torch.sigmoid(
        final_predictions["classification_logits"]
    ).detach().cpu().numpy()
    location = final_predictions["location"].detach().cpu().numpy()
    for index, sample in enumerate(dataset.samples):
        render_prediction_comparison(
            sample.rgb_path,
            read_stair_lines(sample.label_path),
            probability[index],
            location[index],
            output / f"prediction_{index + 1:02d}_{sample.sample_id}.png",
            title=f"tiny-overfit train/{sample.sample_id}",
        )

    result: dict[str, Any] = {
        "success": success,
        "device": str(device),
        "seed": seed,
        "sample_ids": sample_ids,
        "sample_count": sample_count,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "completed_steps": completed_steps,
        "augmentation": False,
        "normalization": "uint8/255 then (x-0.5)/0.5",
        "location_mask_mode": criterion.mask_mode,
        "alpha": float(criterion.alpha.item()),
        "beta": float(criterion.beta.item()),
        "initial": initial,
        "final": final,
        "history": history,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "result": result,
        },
        output / "checkpoint.pt",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/tiny_overfit")
    )
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--evaluate-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=202302175)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_tiny_overfit(
        args.dataset_root,
        args.output_dir,
        sample_count=args.sample_count,
        max_steps=args.max_steps,
        evaluate_every=args.evaluate_every,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
