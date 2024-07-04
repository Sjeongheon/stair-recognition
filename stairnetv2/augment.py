"""Paired geometric augmentation for RGB, depth, and original stair lines."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, Sequence

from PIL import Image

from stairnetv2.label_encoder import StairLine


class RandomSource(Protocol):
    def random(self) -> float: ...


@dataclass(frozen=True)
class AugmentedPair:
    rgb: Image.Image
    depth: Image.Image
    lines: tuple[StairLine, ...]
    flipped: bool


def horizontal_flip_lines(
    lines: Sequence[StairLine], *, image_width: int
) -> tuple[StairLine, ...]:
    """Flip endpoints and restore the paper's left-to-right endpoint order."""

    if image_width <= 0:
        raise ValueError("image_width must be positive")
    flipped: list[StairLine] = []
    for line in lines:
        if not (0.0 <= line.x1 <= image_width and 0.0 <= line.x2 <= image_width):
            raise ValueError("line x coordinate is outside the image")
        flipped.append(
            StairLine(
                cls=line.cls,
                x1=image_width - line.x2,
                y1=line.y2,
                x2=image_width - line.x1,
                y2=line.y1,
            )
        )
    return tuple(flipped)


def horizontal_flip_pair(
    rgb: Image.Image,
    depth: Image.Image,
    lines: Sequence[StairLine],
) -> AugmentedPair:
    """Apply one synchronized horizontal flip without color/depth conversion."""

    if rgb.size != depth.size:
        raise ValueError(f"RGB/depth size mismatch: {rgb.size} != {depth.size}")
    return AugmentedPair(
        rgb=rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        depth=depth.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        lines=horizontal_flip_lines(lines, image_width=rgb.width),
        flipped=True,
    )


class PairedRandomHorizontalFlip:
    """Paper augmentation with one random decision shared by all modalities."""

    def __init__(
        self,
        probability: float = 0.5,
        *,
        random_source: RandomSource | None = None,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        self.probability = probability
        self.random_source = random_source if random_source is not None else random

    def __call__(
        self,
        rgb: Image.Image,
        depth: Image.Image,
        lines: Sequence[StairLine],
    ) -> AugmentedPair:
        if rgb.size != depth.size:
            raise ValueError(f"RGB/depth size mismatch: {rgb.size} != {depth.size}")
        if self.random_source.random() < self.probability:
            return horizontal_flip_pair(rgb, depth, lines)
        return AugmentedPair(rgb=rgb, depth=depth, lines=tuple(lines), flipped=False)
