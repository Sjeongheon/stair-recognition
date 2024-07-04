"""Optional checkpoint/scheduler/early-stop scoring independent of alpha and beta."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


def validation_score(
    values: Mapping[str, float | int],
    *,
    iou_weight: float = 1.0,
    location_weight: float = 0.5,
) -> float:
    """Higher is better: IoU minus mean normalized X/Y MAE penalty.

    MAEs are in cell-normalized [0,1] coordinates, not pixel units. This is an
    experimental score, not a score defined by the paper; keep coefficients
    *fixed across epochs* and log its components separately.
    """
    if int(values["positive_cells"]) < 1:
        raise ValueError("combined score requires at least one positive validation cell")
    iou = float(values["iou"])
    x_mae = float(values["location_mae_x"])
    y_mae = float(values["location_mae_y"])
    result = iou_weight * iou - location_weight * (x_mae + y_mae) / 2
    if not math.isfinite(result):
        raise ValueError("non-finite validation score")
    return result


@dataclass
class EarlyStopping:
    """Track the fixed validation score; no model mutation or hidden checkpointing."""

    patience: int = 15
    min_delta: float = 0.0
    best: float = -float("inf")
    bad_epochs: int = 0

    def step(self, score: float) -> bool:
        """Return True when training should stop."""
        if self.patience < 1 or self.min_delta < 0:
            raise ValueError("patience must be >=1 and min_delta must be >=0")
        if not math.isfinite(score):
            raise ValueError("score must be finite")
        if score > self.best + self.min_delta:
            self.best = score
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience
