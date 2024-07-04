"""Drop-in RGB/depth feature fusion alternatives with output shape [B,C,H,W]."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _validate(rgb: Tensor, depth: Tensor, channels: int) -> None:
    if rgb.shape != depth.shape or rgb.ndim != 4 or rgb.shape[1] != channels:
        raise ValueError(
            f"expected paired [B,{channels},H,W] features, got "
            f"RGB={tuple(rgb.shape)} depth={tuple(depth.shape)}"
        )


class ConcatFusion(nn.Module):
    """No assumed channel correspondence before a learned 1x1 projection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.projection = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb: Tensor, depth: Tensor) -> Tensor:
        _validate(rgb, depth, self.channels)
        return self.projection(torch.cat((rgb, depth), dim=1))


class SeparateDescriptorFusion(nn.Module):
    """GAP each modality separately, then channel-wise modality softmax."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 2 * channels),
        )

    def weights(self, rgb: Tensor, depth: Tensor) -> Tensor:
        _validate(rgb, depth, self.channels)
        descriptor = torch.cat((self.pool(rgb).flatten(1), self.pool(depth).flatten(1)), dim=1)
        logits = self.gate(descriptor).reshape(rgb.shape[0], 2, self.channels, 1, 1)
        return torch.softmax(logits, dim=1)

    def forward(self, rgb: Tensor, depth: Tensor) -> Tensor:
        weights = self.weights(rgb, depth)
        return weights[:, 0] * rgb + weights[:, 1] * depth


class SpatialFusion(nn.Module):
    """One modality weight per pixel, shared by channels at that pixel."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, kernel_size=1),
        )

    def weights(self, rgb: Tensor, depth: Tensor) -> Tensor:
        _validate(rgb, depth, self.channels)
        return torch.softmax(self.gate(torch.cat((rgb, depth), dim=1)), dim=1)

    def forward(self, rgb: Tensor, depth: Tensor) -> Tensor:
        weights = self.weights(rgb, depth)
        return weights[:, 0:1] * rgb + weights[:, 1:2] * depth


class ChannelSpatialFusion(nn.Module):
    """Independent modality weights for each channel and spatial position."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2 * channels, kernel_size=1),
        )

    def weights(self, rgb: Tensor, depth: Tensor) -> Tensor:
        _validate(rgb, depth, self.channels)
        b, _, h, w = rgb.shape
        logits = self.gate(torch.cat((rgb, depth), dim=1)).reshape(
            b, 2, self.channels, h, w
        )
        return torch.softmax(logits, dim=1)

    def forward(self, rgb: Tensor, depth: Tensor) -> Tensor:
        weights = self.weights(rgb, depth)
        return weights[:, 0] * rgb + weights[:, 1] * depth
