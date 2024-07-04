"""Optional stride-2 shortcuts. Swap a single class in model.py by hand.

All modules accept [B,C,H,W] and return [B,out_channels,ceil(H/2),ceil(W/2)]
for even inputs used by StairNetV2. PixelUnshuffle/Haar require even H and W.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class AvgPoolShortcut(nn.Module):
    """ResNet-D-style 2x2 average pooling followed by channel projection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class BlurPool(nn.Module):
    """Fixed separable binomial LPF followed by sampling (not learned)."""

    def __init__(self, channels: int, *, filter_size: int = 3, stride: int = 2) -> None:
        super().__init__()
        if filter_size not in (3, 5):
            raise ValueError("supported odd filter sizes are 3 and 5")
        if stride < 1:
            raise ValueError("stride must be positive")
        coefficients = torch.tensor(
            [math.comb(filter_size - 1, i) for i in range(filter_size)],
            dtype=torch.float32,
        )
        kernel = coefficients[:, None] * coefficients[None, :]
        kernel /= kernel.sum()
        self.register_buffer("kernel", kernel[None, None].repeat(channels, 1, 1, 1))
        self.channels = channels
        self.stride = stride
        self.padding = filter_size // 2

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"expected [B,{self.channels},H,W], got {tuple(x.shape)}")
        return F.conv2d(
            x,
            self.kernel.to(dtype=x.dtype),
            stride=self.stride,
            padding=self.padding,
            groups=self.channels,
        )


class BlurPoolShortcut(nn.Module):
    """Blur spatially *before* projecting channels; preserves output shape."""

    def __init__(
        self, in_channels: int, out_channels: int, *, filter_size: int = 3
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            BlurPool(in_channels, filter_size=filter_size, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class PixelUnshuffleShortcut(nn.Module):
    """Invertible rearrangement if out_channels==4*in_channels (no projection)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.projection = (
            nn.Identity()
            if out_channels == 4 * in_channels
            else nn.Sequential(
                nn.Conv2d(4 * in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError("PixelUnshuffle needs even spatial dimensions")
        return self.projection(self.unshuffle(x))


class HaarDownsample(nn.Module):
    """Fixed orthonormal Haar DWT: [B,C,H,W] -> [B,4C,H/2,W/2].

    The band order is per original input channel: LL,LH,HL,HH. This module is
    invertible on even-sized inputs when *all four* bands are retained.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        basis = torch.tensor(
            [
                [[1., 1.], [1., 1.]],
                [[1., -1.], [1., -1.]],
                [[1., 1.], [-1., -1.]],
                [[1., -1.], [-1., 1.]],
            ],
            dtype=torch.float32,
        ) / 2.0
        self.register_buffer("weight", basis[:, None].repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {tuple(x.shape)}")
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError("Haar DWT requires even height and width")
        return F.conv2d(
            x,
            self.weight.to(dtype=x.dtype),
            stride=2,
            groups=self.channels,
        )


class HaarShortcut(nn.Module):
    """DWT followed by an optional learned projection; projection may lose data."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.dwt = HaarDownsample(in_channels)
        self.projection = (
            nn.Identity()
            if out_channels == 4 * in_channels
            else nn.Sequential(
                nn.Conv2d(4 * in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(self.dwt(x))
