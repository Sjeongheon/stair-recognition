"""Paper-defined cell metrics and positive-cell localization errors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from stairnetv2.loss import X_CHANNELS, Y_CHANNELS


@dataclass(frozen=True)
class BinaryCellCounts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def __add__(self, other: "BinaryCellCounts") -> "BinaryCellCounts":
        return BinaryCellCounts(
            self.true_positive + other.true_positive,
            self.false_positive + other.false_positive,
            self.false_negative + other.false_negative,
        )

    def compute(self, *, zero_division: float = 0.0) -> dict[str, float]:
        precision_denominator = self.true_positive + self.false_positive
        recall_denominator = self.true_positive + self.false_negative
        iou_denominator = (
            self.true_positive + self.false_positive + self.false_negative
        )
        precision = (
            self.true_positive / precision_denominator
            if precision_denominator
            else zero_division
        )
        recall = (
            self.true_positive / recall_denominator
            if recall_denominator
            else zero_division
        )
        iou = (
            self.true_positive / iou_denominator
            if iou_denominator
            else zero_division
        )
        return {
            "precision": precision,
            "paper_accuracy": precision,
            "recall": recall,
            "iou": iou,
        }


@dataclass(frozen=True)
class LocalizationErrors:
    x_error_sum: float
    y_error_sum: float
    positive_cells: int

    def __add__(self, other: "LocalizationErrors") -> "LocalizationErrors":
        return LocalizationErrors(
            self.x_error_sum + other.x_error_sum,
            self.y_error_sum + other.y_error_sum,
            self.positive_cells + other.positive_cells,
        )

    def coordinate_means(self, *, zero_division: float = 0.0) -> dict[str, float]:
        denominator = self.positive_cells * 4
        return {
            "location_mae_x": (
                self.x_error_sum / denominator if denominator else zero_division
            ),
            "location_mae_y": (
                self.y_error_sum / denominator if denominator else zero_division
            ),
        }


def binary_cell_counts(
    classification_logits: Tensor,
    target_classification: Tensor,
    *,
    probability_threshold: float = 0.5,
) -> BinaryCellCounts:
    if classification_logits.shape != target_classification.shape:
        raise ValueError("classification prediction and target shapes must match")
    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("probability threshold must be in [0,1]")
    predicted = torch.sigmoid(classification_logits) > probability_threshold
    target = target_classification >= 0.5
    return BinaryCellCounts(
        true_positive=int(torch.count_nonzero(predicted & target).item()),
        false_positive=int(torch.count_nonzero(predicted & ~target).item()),
        false_negative=int(torch.count_nonzero(~predicted & target).item()),
    )


def positive_location_errors(
    predicted_location: Tensor,
    target_location: Tensor,
    target_classification: Tensor,
) -> LocalizationErrors:
    if predicted_location.shape != target_location.shape:
        raise ValueError("location prediction and target shapes must match")
    if predicted_location.ndim != 4 or predicted_location.shape[1] != 8:
        raise ValueError("location tensors must have shape [B,8,H,W]")
    expected_mask_shape = (
        predicted_location.shape[0],
        1,
        predicted_location.shape[2],
        predicted_location.shape[3],
    )
    if target_classification.shape != expected_mask_shape:
        raise ValueError("classification target must have shape [B,1,H,W]")
    absolute_error = torch.abs(predicted_location - target_location)
    positive_mask = (target_classification >= 0.5).to(absolute_error.dtype)
    x_error = (absolute_error[:, X_CHANNELS] * positive_mask).sum()
    y_error = (absolute_error[:, Y_CHANNELS] * positive_mask).sum()
    return LocalizationErrors(
        x_error_sum=float(x_error.item()),
        y_error_sum=float(y_error.item()),
        positive_cells=int(torch.count_nonzero(positive_mask).item()),
    )
