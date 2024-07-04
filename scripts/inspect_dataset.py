"""Check the archived reproduction's exact-stem RGB/Depth/label pairing.

Examples:
  python scripts/inspect_dataset.py DATASET --split train
  python scripts/inspect_dataset.py DATASET --split val --all

A complete pairing report is checked, then --limit samples are decoded and
checked for the actual baseline tensor format. This script neither changes
pixel values nor assumes that a zero-valued depth pixel is invalid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stairnetv2.dataset import (
    DatasetIntegrityError,
    build_pairing_report,
    inspect_image,
    validate_sample,
)
from stairnetv2.label_encoder import IMAGE_HEIGHT, IMAGE_WIDTH


def inspect_split(root: Path, split: str, limit: int | None = 8) -> dict[str, object]:
    if split not in ("train", "val"):
        raise ValueError("split must be train or val")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive or None")

    folder = root.expanduser().resolve() / split
    pairing = build_pairing_report(
        folder / "images", folder / "depthes", folder / "labels", strict=True
    )
    selected = pairing.samples if limit is None else pairing.samples[:limit]
    inspected = []
    for sample in selected:
        rgb = inspect_image(sample.rgb_path)
        depth, label = validate_sample(sample)
        if rgb.mode != "RGB" or rgb.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
            raise DatasetIntegrityError(
                f"{sample.sample_id}: RGB must be 512x512 RGB, "
                f"got {rgb.size}, {rgb.mode}"
            )
        if (
            depth.storage_mode != "L"
            or depth.dtype != "uint8"
            or depth.shape != (IMAGE_HEIGHT, IMAGE_WIDTH)
        ):
            raise DatasetIntegrityError(
                f"{sample.sample_id}: depth must be 512x512 uint8 L, "
                f"got {depth.shape}, {depth.dtype}, {depth.storage_mode}"
            )
        inspected.append(
            {
                "stem": sample.sample_id,
                "rgb_mode": rgb.mode,
                "depth_mode": depth.storage_mode,
                "depth_dtype": depth.dtype,
                "depth_min": depth.minimum,
                "depth_max": depth.maximum,
                "depth_zero_fraction": depth.zero_fraction,
                "label_lines": label.line_count,
            }
        )
    return {
        "split": split,
        "paired_samples": len(pairing.samples),
        "inspected_samples": len(inspected),
        "validation_scope": "complete exact-stem pairing; decoded subset" if limit is not None else "complete pairing and decoded files",
        "samples": inspected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--all", action="store_true", help="inspect every paired file")
    args = parser.parse_args()
    try:
        report = inspect_split(
            args.dataset_root, args.split, limit=None if args.all else args.limit
        )
    except (DatasetIntegrityError, ValueError) as exc:
        parser.exit(1, f"dataset inspection failed: {exc}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
