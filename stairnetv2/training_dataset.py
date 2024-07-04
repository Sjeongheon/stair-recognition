"""Training dataset retaining the archived reproduction's input semantics.

RGB: uint8 RGB -> /255 -> (x-0.5)/0.5; depth: uint8 L, ONE channel with
the same baseline normalization. Do not infer depth units or missing-value
semantics from a grayscale sample. Alternative normalizations are opt-in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from stairnetv2.augment import PairedRandomHorizontalFlip
from stairnetv2.dataset import PairedSample, build_pairing_report
from stairnetv2.label_encoder import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    OverflowPolicy,
    StairLine,
    encode_lines,
    read_stair_lines,
)


Split = Literal["train", "val"]
RGBNormalization = Literal["baseline", "unit", "imagenet", "dataset_mean_std"]
DepthNormalization = Literal["baseline", "unit", "per_image_minmax"]


def image_to_baseline_tensor(image: Image.Image, *, expected_mode: str) -> torch.Tensor:
    """Apply author-reference `ToTensor` and 0.5/0.5 normalization."""

    if image.mode != expected_mode:
        raise ValueError(f"expected image mode {expected_mode}, got {image.mode}")
    array = np.array(image, dtype=np.uint8, copy=True)
    if expected_mode == "L":
        array = array[..., None]
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(torch.float32) / 255.0
    return (tensor - 0.5) / 0.5


def image_to_unit_tensor(image: Image.Image, *, expected_mode: str) -> torch.Tensor:
    """8-bit /255 scaling, NOT independent per-image min-max normalization."""
    if image.mode != expected_mode:
        raise ValueError(f"expected image mode {expected_mode}, got {image.mode}")
    array = np.array(image, copy=True)
    if array.dtype != np.uint8:
        raise ValueError("8-bit image required; inspect original depth dtype first")
    if expected_mode == "L":
        array = array[..., None]
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0


def normalize_rgb(
    image: Image.Image,
    mode: RGBNormalization,
    *,
    dataset_mean: tuple[float, float, float] | None = None,
    dataset_std: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    tensor = image_to_unit_tensor(image, expected_mode="RGB")
    if mode == "unit":
        return tensor
    if mode == "baseline":
        return (tensor - 0.5) / 0.5
    if mode == "imagenet":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif mode == "dataset_mean_std":
        if dataset_mean is None or dataset_std is None:
            raise ValueError("dataset_mean_std requires training-set mean and std")
        mean, std = dataset_mean, dataset_std
    else:
        raise ValueError(f"unknown RGB normalization {mode}")
    if min(std) <= 0:
        raise ValueError("normalization standard deviations must be positive")
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype)[:, None, None]
    std_tensor = torch.tensor(std, dtype=tensor.dtype)[:, None, None]
    return (tensor - mean_tensor) / std_tensor


def normalize_depth(image: Image.Image, mode: DepthNormalization) -> torch.Tensor:
    tensor = image_to_unit_tensor(image, expected_mode="L")
    if mode == "unit":
        return tensor
    if mode == "baseline":
        return (tensor - 0.5) / 0.5
    if mode == "per_image_minmax":
        minimum, maximum = tensor.amin(), tensor.amax()
        return (tensor - minimum) / (maximum - minimum).clamp_min(1e-8)
    raise ValueError(f"unknown depth normalization {mode}")


class StairNetDataset(Dataset[dict[str, object]]):
    """Strict train/val dataset; the unlabeled release test directory is unsupported."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: Split,
        augment: bool = False,
        flip_probability: float = 0.5,
        overflow_policy: OverflowPolicy = "longest",
        sample_ids: Sequence[str] | None = None,
        rgb_normalization: RGBNormalization = "baseline",
        depth_normalization: DepthNormalization = "baseline",
        dataset_rgb_mean: tuple[float, float, float] | None = None,
        dataset_rgb_std: tuple[float, float, float] | None = None,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'; unlabeled test is excluded")
        if augment and split != "train":
            raise ValueError("augmentation is allowed only for the training split")
        root = Path(dataset_root).expanduser().resolve()
        split_root = root / split
        report = build_pairing_report(
            split_root / "images",
            split_root / "depthes",
            split_root / "labels",
        )
        indexed = {sample.sample_id: sample for sample in report.samples}
        if sample_ids is None:
            samples = list(report.samples)
        else:
            missing = sorted(set(sample_ids) - set(indexed))
            if missing:
                raise ValueError(f"unknown {split} sample IDs: {missing}")
            samples = [indexed[sample_id] for sample_id in sample_ids]
        if not samples:
            raise ValueError("dataset selection must contain at least one sample")

        self.dataset_root = root
        self.split = split
        self.samples: tuple[PairedSample, ...] = tuple(samples)
        self.lines: tuple[tuple[StairLine, ...], ...] = tuple(
            read_stair_lines(sample.label_path) for sample in self.samples
        )
        self.augmentation = (
            PairedRandomHorizontalFlip(flip_probability) if augment else None
        )
        self.overflow_policy = overflow_policy
        self.rgb_normalization = rgb_normalization
        self.depth_normalization = depth_normalization
        self.dataset_rgb_mean = dataset_rgb_mean
        self.dataset_rgb_std = dataset_rgb_std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        with Image.open(sample.rgb_path) as source:
            source.load()
            rgb = source.copy()
        with Image.open(sample.depth_path) as source:
            source.load()
            depth = source.copy()
        if rgb.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
            raise ValueError(f"unexpected RGB size for {sample.sample_id}: {rgb.size}")
        if depth.size != rgb.size:
            raise ValueError(
                f"RGB/depth size mismatch for {sample.sample_id}: {rgb.size} != {depth.size}"
            )
        if rgb.mode != "RGB" or depth.mode != "L":
            raise ValueError(
                f"unexpected modes for {sample.sample_id}: RGB={rgb.mode}, depth={depth.mode}"
            )

        lines = self.lines[index]
        flipped = False
        if self.augmentation is not None:
            augmented = self.augmentation(rgb, depth, lines)
            rgb, depth, lines = augmented.rgb, augmented.depth, augmented.lines
            flipped = augmented.flipped
        encoded = encode_lines(lines, overflow_policy=self.overflow_policy)
        return {
            "sample_id": sample.sample_id,
            "rgb": normalize_rgb(
                rgb, self.rgb_normalization,
                dataset_mean=self.dataset_rgb_mean,
                dataset_std=self.dataset_rgb_std,
            ),
            "depth": normalize_depth(depth, self.depth_normalization),
            "classification": torch.from_numpy(encoded.classification.copy()),
            "location": torch.from_numpy(encoded.location.copy()),
            "segment_count": torch.from_numpy(encoded.segment_count.copy()),
            "flipped": flipped,
        }
