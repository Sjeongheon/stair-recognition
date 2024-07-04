"""StairNetV2 classification/location loss and dynamic coordinate weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
from torch import Tensor, nn


LocationMaskMode = Literal["ground_truth", "paper_literal_prediction"]
X_CHANNELS = (0, 2, 4, 6)
Y_CHANNELS = (1, 3, 5, 7)


class LossOutput(NamedTuple):
    total: Tensor
    classification: Tensor
    location_x: Tensor
    location_y: Tensor


@dataclass(frozen=True)
class WeightUpdate:
    alpha_before: float
    beta_before: float
    alpha_after: float
    beta_after: float
    x_error: float
    y_error: float


def _validate_shapes(
    classification_logits: Tensor,
    predicted_location: Tensor,
    target_classification: Tensor,
    target_location: Tensor,
) -> None:
    if classification_logits.ndim != 4 or classification_logits.shape[1] != 1:
        raise ValueError("classification logits must have shape [B,1,H,W]")
    if predicted_location.ndim != 4 or predicted_location.shape[1] != 8:
        raise ValueError("predicted location must have shape [B,8,H,W]")
    if target_classification.shape != classification_logits.shape:
        raise ValueError("classification target shape does not match logits")
    if target_location.shape != predicted_location.shape:
        raise ValueError("location target shape does not match prediction")
    if predicted_location.shape[0] != classification_logits.shape[0] or (
        predicted_location.shape[-2:] != classification_logits.shape[-2:]
    ):
        raise ValueError("classification and location batch/spatial shapes must match")


class StairNetV2Loss(nn.Module):
    """BCE classification plus grid-mean masked MSE coordinate losses.

    `ground_truth` masks positive cells, as chosen by the archived baseline.
    The printed paper Equation (2) multiplies by predicted p instead; select
    `paper_literal_prediction` only for an explicitly labeled ablation.
    """

    def __init__(
        self,
        *,
        alpha: float = 10.0,
        beta: float = 10.0,
        sigma: float = 0.5,
        mask_mode: LocationMaskMode = "ground_truth",
        classification_loss_fn: nn.Module | None = None,
        location_loss_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if alpha < sigma or beta < sigma:
            raise ValueError("initial alpha and beta must not be below sigma")
        if mask_mode not in ("ground_truth", "paper_literal_prediction"):
            raise ValueError(f"unknown location mask mode: {mask_mode}")
        self.register_buffer("alpha", torch.tensor(float(alpha)))
        self.register_buffer("beta", torch.tensor(float(beta)))
        self.sigma = float(sigma)
        self.mask_mode = mask_mode
        self.classification_loss_fn = (
            classification_loss_fn if classification_loss_fn is not None
            else nn.BCEWithLogitsLoss(reduction="mean")
        )
        self.location_loss_fn = (
            location_loss_fn if location_loss_fn is not None
            else nn.MSELoss(reduction="none")
        )

    def forward(
        self,
        classification_logits: Tensor,
        predicted_location: Tensor,
        target_classification: Tensor,
        target_location: Tensor,
    ) -> LossOutput:
        _validate_shapes(
            classification_logits,
            predicted_location,
            target_classification,
            target_location,
        )
        target_classification = target_classification.to(
            dtype=classification_logits.dtype
        )
        target_location = target_location.to(dtype=predicted_location.dtype)
        classification_loss = self.classification_loss_fn(
            classification_logits, target_classification
        )
        coordinate_error = self.location_loss_fn(
            predicted_location, target_location
        )
        if classification_loss.ndim != 0:
            raise ValueError("classification loss must return a scalar")
        if coordinate_error.shape != predicted_location.shape:
            raise ValueError("location loss must return per-coordinate errors [B,8,H,W]")
        x_error = coordinate_error[:, X_CHANNELS].sum(dim=1, keepdim=True)
        y_error = coordinate_error[:, Y_CHANNELS].sum(dim=1, keepdim=True)
        if self.mask_mode == "ground_truth":
            location_mask = target_classification
        else:
            location_mask = torch.sigmoid(classification_logits)
        grid_elements = (
            classification_logits.shape[0]
            * classification_logits.shape[-2]
            * classification_logits.shape[-1]
        )
        location_x = (location_mask * x_error).sum() / grid_elements
        location_y = (location_mask * y_error).sum() / grid_elements
        total = (
            classification_loss
            + self.alpha * location_x
            + self.beta * location_y
        )
        return LossOutput(total, classification_loss, location_x, location_y)

    @torch.no_grad()
    def update_weights(self, x_error: float, y_error: float) -> WeightUpdate:
        """Atomic threshold update: preserve alpha+beta and reject invalid pairs.

        Paper says values below sigma are not adjusted, but does not clarify
        whether the pair should be rejected together. Archived per-weight
        rejection caused beta drift; this is an explicit reproduction choice.
        """

        if x_error < 0.0 or y_error < 0.0:
            raise ValueError("validation errors must be non-negative")
        alpha_before = float(self.alpha.item())
        beta_before = float(self.beta.item())
        maximum = max(x_error, y_error)
        if maximum > 0.0:
            delta = (x_error - y_error) / maximum
            alpha_candidate = alpha_before + delta
            beta_candidate = beta_before - delta
            if alpha_candidate >= self.sigma and beta_candidate >= self.sigma:
                self.alpha.fill_(alpha_candidate)
                self.beta.fill_(beta_candidate)
        return WeightUpdate(
            alpha_before=alpha_before,
            beta_before=beta_before,
            alpha_after=float(self.alpha.item()),
            beta_after=float(self.beta.item()),
            x_error=x_error,
            y_error=y_error,
        )
