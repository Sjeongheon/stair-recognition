"""StairNetV2 grid-label encoding and decoding.

The paper fixes the output capacity at two line segments per cell, but does not
specify slot order, boundary ownership, or overflow handling. Those choices are
therefore explicit and deterministic here instead of being hidden in indexing
side effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np


IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
GRID_H = 32
GRID_W = 16
MAX_LINES_PER_CELL = 2

# PyTorch target layout is [C, H, W]. The eight location channels are
# interleaved point coordinates, matching the paper's two endpoint tuples.
LOCATION_CHANNELS = (
    "x1",
    "y1",
    "x2",
    "y2",
    "x3",
    "y3",
    "x4",
    "y4",
)

OverflowPolicy = Literal["error", "longest"]


class LabelEncodingError(ValueError):
    """Raised when an annotation cannot be represented without a stated policy."""


@dataclass(frozen=True)
class StairLine:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class CellSegment:
    row: int
    col: int
    source_line_index: int
    source_class: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def midpoint(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True)
class CellOverflow:
    row: int
    col: int
    source_line_indices: tuple[int, ...]
    retained_line_indices: tuple[int, ...]
    dropped_line_indices: tuple[int, ...]
    dropped_pixel_length: float


@dataclass(frozen=True)
class EncodedLabels:
    classification: np.ndarray
    location: np.ndarray
    segment_count: np.ndarray
    overflows: tuple[CellOverflow, ...]


def read_stair_lines(
    path: str | Path,
    *,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
) -> tuple[StairLine, ...]:
    """Read and strictly validate original ``cls x1 y1 x2 y2`` annotations."""

    label_path = Path(path).expanduser().resolve()
    lines: list[StairLine] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) != 5:
                raise LabelEncodingError(
                    f"{label_path}:{line_number}: expected 5 values, got {len(fields)}"
                )
            try:
                raw_class, x1, y1, x2, y2 = map(float, fields)
            except ValueError as error:
                raise LabelEncodingError(
                    f"{label_path}:{line_number}: non-numeric annotation"
                ) from error
            values = np.asarray((raw_class, x1, y1, x2, y2), dtype=np.float64)
            if not np.isfinite(values).all():
                raise LabelEncodingError(
                    f"{label_path}:{line_number}: non-finite annotation"
                )
            if raw_class not in (0.0, 1.0):
                raise LabelEncodingError(
                    f"{label_path}:{line_number}: class must be 0 or 1"
                )
            if not (0.0 <= x1 <= image_width and 0.0 <= x2 <= image_width):
                raise LabelEncodingError(
                    f"{label_path}:{line_number}: x coordinate outside image"
                )
            if not (0.0 <= y1 <= image_height and 0.0 <= y2 <= image_height):
                raise LabelEncodingError(
                    f"{label_path}:{line_number}: y coordinate outside image"
                )
            if (x1, y1) == (x2, y2):
                raise LabelEncodingError(
                    f"{label_path}:{line_number}: zero-length stair line"
                )
            if x1 > x2 or (x1 == x2 and y1 > y2):
                raise LabelEncodingError(
                    f"{label_path}:{line_number}: endpoints are not left-to-right"
                )
            lines.append(StairLine(int(raw_class), x1, y1, x2, y2))
    return tuple(lines)


def _unique_parameters(values: list[float], tolerance: float = 1e-12) -> list[float]:
    ordered = sorted(values)
    unique: list[float] = []
    for value in ordered:
        if not unique or abs(value - unique[-1]) > tolerance:
            unique.append(value)
    return unique


def clip_line_to_grid(
    line: StairLine,
    source_line_index: int,
    *,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
    grid_h: int = GRID_H,
    grid_w: int = GRID_W,
) -> tuple[CellSegment, ...]:
    """Split a line at grid boundaries and assign each nonzero piece once.

    A midpoint selects the owner of each piece. A line lying exactly on an
    internal boundary belongs to the cell below/right of that boundary. Outer
    image boundaries are clamped to the last row/column.
    """

    if image_width <= 0 or image_height <= 0 or grid_h <= 0 or grid_w <= 0:
        raise LabelEncodingError("image and grid dimensions must be positive")
    if not (
        0.0 <= line.x1 <= image_width
        and 0.0 <= line.x2 <= image_width
        and 0.0 <= line.y1 <= image_height
        and 0.0 <= line.y2 <= image_height
    ):
        raise LabelEncodingError("line coordinates are outside the image")
    if (line.x1, line.y1) == (line.x2, line.y2):
        raise LabelEncodingError("cannot encode a zero-length line")

    cell_width = image_width / grid_w
    cell_height = image_height / grid_h
    dx = line.x2 - line.x1
    dy = line.y2 - line.y1
    parameters = [0.0, 1.0]
    if dx != 0.0:
        for col in range(1, grid_w):
            parameter = (col * cell_width - line.x1) / dx
            if 0.0 < parameter < 1.0:
                parameters.append(parameter)
    if dy != 0.0:
        for row in range(1, grid_h):
            parameter = (row * cell_height - line.y1) / dy
            if 0.0 < parameter < 1.0:
                parameters.append(parameter)

    parameters = _unique_parameters(parameters)
    segments: list[CellSegment] = []
    for start_t, end_t in zip(parameters, parameters[1:]):
        if end_t - start_t <= 1e-12:
            continue
        midpoint_t = (start_t + end_t) / 2.0
        midpoint_x = line.x1 + dx * midpoint_t
        midpoint_y = line.y1 + dy * midpoint_t
        col = min(grid_w - 1, max(0, int(midpoint_x / cell_width)))
        row = min(grid_h - 1, max(0, int(midpoint_y / cell_height)))
        segments.append(
            CellSegment(
                row=row,
                col=col,
                source_line_index=source_line_index,
                source_class=line.cls,
                x1=line.x1 + dx * start_t,
                y1=line.y1 + dy * start_t,
                x2=line.x1 + dx * end_t,
                y2=line.y1 + dy * end_t,
            )
        )
    return tuple(segments)


def _slot_sort_key(segment: CellSegment) -> tuple[float, float, int]:
    midpoint_x, midpoint_y = segment.midpoint
    return midpoint_y, midpoint_x, segment.source_line_index


def _retain_segments(
    segments: list[CellSegment], policy: OverflowPolicy
) -> tuple[list[CellSegment], CellOverflow | None]:
    if len(segments) <= MAX_LINES_PER_CELL:
        return sorted(segments, key=_slot_sort_key), None
    if policy == "error":
        first = segments[0]
        raise LabelEncodingError(
            f"cell [y={first.row}, x={first.col}] contains {len(segments)} lines; "
            f"the paper output can represent only {MAX_LINES_PER_CELL}"
        )
    if policy != "longest":
        raise LabelEncodingError(f"unknown overflow policy: {policy}")

    ranked = sorted(
        segments,
        key=lambda segment: (-segment.length, segment.source_line_index),
    )
    retained = ranked[:MAX_LINES_PER_CELL]
    dropped = ranked[MAX_LINES_PER_CELL:]
    overflow = CellOverflow(
        row=segments[0].row,
        col=segments[0].col,
        source_line_indices=tuple(segment.source_line_index for segment in segments),
        retained_line_indices=tuple(
            segment.source_line_index for segment in retained
        ),
        dropped_line_indices=tuple(segment.source_line_index for segment in dropped),
        dropped_pixel_length=sum(segment.length for segment in dropped),
    )
    return sorted(retained, key=_slot_sort_key), overflow


def encode_lines(
    lines: Sequence[StairLine],
    *,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
    grid_h: int = GRID_H,
    grid_w: int = GRID_W,
    overflow_policy: OverflowPolicy = "error",
) -> EncodedLabels:
    """Encode original lines into `[1,H,W]` class and `[8,H,W]` location targets.

    A single segment is copied into both location slots, following the original
    StairNet paper. With two retained segments, slots are ordered by midpoint
    y then x. The StairNetV2 paper leaves this ordering unspecified.
    """

    by_cell: dict[tuple[int, int], list[CellSegment]] = {}
    for line_index, line in enumerate(lines):
        for segment in clip_line_to_grid(
            line,
            line_index,
            image_width=image_width,
            image_height=image_height,
            grid_h=grid_h,
            grid_w=grid_w,
        ):
            by_cell.setdefault((segment.row, segment.col), []).append(segment)

    classification = np.zeros((1, grid_h, grid_w), dtype=np.float32)
    location = np.zeros((8, grid_h, grid_w), dtype=np.float32)
    segment_count = np.zeros((grid_h, grid_w), dtype=np.uint8)
    overflows: list[CellOverflow] = []
    cell_width = image_width / grid_w
    cell_height = image_height / grid_h

    for (row, col), candidates in sorted(by_cell.items()):
        retained, overflow = _retain_segments(candidates, overflow_policy)
        if overflow is not None:
            overflows.append(overflow)
        classification[0, row, col] = 1.0
        segment_count[row, col] = len(retained)
        slots = retained if len(retained) == 2 else [retained[0], retained[0]]
        for slot, segment in enumerate(slots):
            local = np.asarray(
                (
                    (segment.x1 - col * cell_width) / cell_width,
                    (segment.y1 - row * cell_height) / cell_height,
                    (segment.x2 - col * cell_width) / cell_width,
                    (segment.y2 - row * cell_height) / cell_height,
                ),
                dtype=np.float64,
            )
            if np.any(local < -1e-9) or np.any(local > 1.0 + 1e-9):
                raise LabelEncodingError(
                    f"internal clipping error in cell [y={row}, x={col}]: {local}"
                )
            location[slot * 4 : slot * 4 + 4, row, col] = np.clip(
                local, 0.0, 1.0
            ).astype(np.float32)

    return EncodedLabels(
        classification=classification,
        location=location,
        segment_count=segment_count,
        overflows=tuple(overflows),
    )


def encode_label_file(
    path: str | Path,
    *,
    overflow_policy: OverflowPolicy = "error",
) -> EncodedLabels:
    return encode_lines(
        read_stair_lines(path), overflow_policy=overflow_policy
    )


def decode_segments(
    encoded: EncodedLabels,
    *,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
) -> tuple[CellSegment, ...]:
    """Decode retained ground-truth cell segments back to pixel coordinates."""

    if encoded.classification.ndim != 3 or encoded.classification.shape[0] != 1:
        raise LabelEncodingError("classification must have shape [1,H,W]")
    _, grid_h, grid_w = encoded.classification.shape
    if encoded.location.shape != (8, grid_h, grid_w):
        raise LabelEncodingError("location must have shape [8,H,W]")
    if encoded.segment_count.shape != (grid_h, grid_w):
        raise LabelEncodingError("segment_count must have shape [H,W]")

    cell_width = image_width / grid_w
    cell_height = image_height / grid_h
    decoded: list[CellSegment] = []
    for row in range(grid_h):
        for col in range(grid_w):
            count = int(encoded.segment_count[row, col])
            if count not in (0, 1, 2):
                raise LabelEncodingError(
                    f"invalid segment count {count} in cell [y={row}, x={col}]"
                )
            for slot in range(count):
                x1, y1, x2, y2 = encoded.location[
                    slot * 4 : slot * 4 + 4, row, col
                ]
                decoded.append(
                    CellSegment(
                        row=row,
                        col=col,
                        source_line_index=-1,
                        source_class=-1,
                        x1=(col + float(x1)) * cell_width,
                        y1=(row + float(y1)) * cell_height,
                        x2=(col + float(x2)) * cell_width,
                        y2=(row + float(y2)) * cell_height,
                    )
                )
    return tuple(decoded)


def decode_prediction_segments(
    classification_probabilities: np.ndarray,
    location: np.ndarray,
    *,
    probability_threshold: float = 0.5,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
) -> tuple[CellSegment, ...]:
    """Decode both predicted slots for cells strictly above the threshold."""

    if classification_probabilities.ndim != 3 or (
        classification_probabilities.shape[0] != 1
    ):
        raise LabelEncodingError(
            "classification probabilities must have shape [1,H,W]"
        )
    if not 0.0 <= probability_threshold <= 1.0:
        raise LabelEncodingError("probability threshold must be in [0,1]")
    _, grid_h, grid_w = classification_probabilities.shape
    if location.shape != (8, grid_h, grid_w):
        raise LabelEncodingError("prediction location must have shape [8,H,W]")
    if not np.isfinite(classification_probabilities).all() or not np.isfinite(
        location
    ).all():
        raise LabelEncodingError("predictions contain non-finite values")
    segment_count = np.where(
        classification_probabilities[0] > probability_threshold, 2, 0
    ).astype(np.uint8)
    encoded = EncodedLabels(
        classification=classification_probabilities.astype(np.float32, copy=False),
        location=location.astype(np.float32, copy=False),
        segment_count=segment_count,
        overflows=(),
    )
    return decode_segments(
        encoded, image_width=image_width, image_height=image_height
    )
