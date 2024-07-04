"""Ground-truth versus decoded prediction visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from stairnetv2.label_encoder import (
    GRID_H,
    GRID_W,
    StairLine,
    decode_prediction_segments,
)


CLASS_COLORS = {0: (0, 120, 255), 1: (255, 60, 40)}
PREDICTION_COLOR = (40, 220, 80)
GRID_COLOR = (130, 130, 130)


def render_prediction_comparison(
    image_path: str | Path,
    ground_truth: tuple[StairLine, ...],
    classification_probabilities: np.ndarray,
    location: np.ndarray,
    output_path: str | Path,
    *,
    probability_threshold: float = 0.5,
    title: str = "",
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    expected_probability_shape = (1, GRID_H, GRID_W)
    if classification_probabilities.shape != expected_probability_shape:
        raise ValueError(
            f"expected probabilities {expected_probability_shape}, "
            f"got {classification_probabilities.shape}"
        )

    truth_image = image.copy()
    truth_draw = ImageDraw.Draw(truth_image)
    for line in ground_truth:
        truth_draw.line(
            (line.x1, line.y1, line.x2, line.y2),
            fill=CLASS_COLORS[line.cls],
            width=3,
        )

    prediction_image = image.copy()
    prediction_draw = ImageDraw.Draw(prediction_image)
    cell_width = image.width / GRID_W
    cell_height = image.height / GRID_H
    for col in range(1, GRID_W):
        x = round(col * cell_width)
        prediction_draw.line((x, 0, x, image.height), fill=GRID_COLOR, width=1)
    for row in range(1, GRID_H):
        y = round(row * cell_height)
        prediction_draw.line((0, y, image.width, y), fill=GRID_COLOR, width=1)
    for segment in decode_prediction_segments(
        classification_probabilities,
        location,
        probability_threshold=probability_threshold,
        image_width=image.width,
        image_height=image.height,
    ):
        prediction_draw.line(
            (segment.x1, segment.y1, segment.x2, segment.y2),
            fill=PREDICTION_COLOR,
            width=3,
        )

    header_height = 28
    canvas = Image.new("RGB", (image.width * 2, image.height + header_height), "white")
    canvas.paste(truth_image, (0, header_height))
    canvas.paste(prediction_image, (image.width, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (4, 6),
        f"{title} | ground truth: blue/red | prediction > {probability_threshold}: green",
        fill="black",
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
