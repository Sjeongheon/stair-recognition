"""Estimate training-image channel stats and/or inspect checkpoint weight stats.

Examples:
  python scripts/inspect_distributions.py --image-dir DATA/train/images --limit 256
  python scripts/inspect_distributions.py --checkpoint artifacts/full_training/model_weights.pt --histogram-bins 20

Statistics are descriptive only; no custom initializer is chosen automatically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def image_distribution(folder: Path, limit: int) -> dict[str, object]:
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"))[:limit]
    if not images:
        raise ValueError("image directory contains no supported images")
    total = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    count = 0
    observed_modes: set[str] = set()
    for path in images:
        with Image.open(path) as image:
            observed_modes.add(image.mode)
            if image.mode != "RGB":
                raise ValueError(f"expected RGB 8-bit image for RGB mean/std: {path} ({image.mode}); inspect depth separately")
            array = np.asarray(image, dtype=np.float64) / 255.0
        values = array.reshape(-1, 3)
        total += values.sum(axis=0)
        squares += (values * values).sum(axis=0)
        count += len(values)
    mean = total / count
    std = np.sqrt(np.maximum(0.0, squares / count - mean * mean))
    return {"images": len(images), "modes": sorted(observed_modes), "mean": mean.tolist(), "std": std.tolist(), "pixels": count}


def checkpoint_distribution(path: Path, histogram_bins: int = 0) -> dict[str, object]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    report: dict[str, object] = {}
    for name, tensor in state.items():
        if not torch.is_tensor(tensor) or not tensor.is_floating_point() or not name.endswith("weight"):
            continue
        data = tensor.detach().float().flatten()
        if not data.numel():
            continue
        report[name] = {
            "shape": list(tensor.shape),
            "mean": float(data.mean()),
            "std": float(data.std(unbiased=False)),
            "min": float(data.min()),
            "max": float(data.max()),
            "q01_q50_q99": [float(v) for v in torch.quantile(data, torch.tensor([0.01, 0.5, 0.99]))],
        }
        if histogram_bins > 0:
            values, edges = np.histogram(data.numpy(), bins=histogram_bins)
            report[name]["histogram"] = {
                "counts": values.tolist(), "edges": edges.tolist()
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--histogram-bins", type=int, default=0)
    args = parser.parse_args()
    if args.image_dir is None and args.checkpoint is None:
        parser.error("supply --image-dir, --checkpoint, or both")
    if args.limit < 1 or args.histogram_bins < 0:
        parser.error("--limit must be positive and --histogram-bins non-negative")
    report = {}
    if args.image_dir is not None:
        report["rgb_training_distribution"] = image_distribution(args.image_dir, args.limit)
    if args.checkpoint is not None:
        report["model_weights"] = checkpoint_distribution(
            args.checkpoint, histogram_bins=args.histogram_bins
        )
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
