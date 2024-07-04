"""Loss options to import and manually substitute in stairnetv2/train.py.

The paper does not name exact classification/location training loss functions.
These are explicit *experimental implementations*, not claimed author settings.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BinaryFocalLoss(nn.Module):
    """Focal loss on logits; useful for examining class imbalance."""

    def __init__(self, gamma: float = 2.0, alpha: float | None = 0.25) -> None:
        super().__init__()
        if gamma < 0 or alpha is not None and not 0 <= alpha <= 1:
            raise ValueError("gamma >=0 and alpha in [0,1] are required")
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        pt = probabilities * targets + (1 - probabilities) * (1 - targets)
        result = (1 - pt).pow(self.gamma) * ce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            result = alpha_t * result
        return result.mean()


class SoftDiceLoss(nn.Module):
    """Global soft Dice loss over a batch of binary-cell predictions."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        if smooth <= 0:
            raise ValueError("smooth must be positive")
        self.smooth = smooth

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        prediction = torch.sigmoid(logits)
        intersection = (prediction * targets).sum(dim=(1, 2, 3))
        denominator = prediction.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        return (1 - (2 * intersection + self.smooth) / (denominator + self.smooth)).mean()


class BCEPlusDiceLoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.dice = SoftDiceLoss()

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets) + self.dice_weight * self.dice(logits, targets)


# Location alternatives can be passed directly: nn.L1Loss(reduction="none"),
# nn.SmoothL1Loss(beta=0.05, reduction="none"), nn.MSELoss(reduction="none").
# Do not use a reduction="mean" module here: the main loss must mask cells first.
