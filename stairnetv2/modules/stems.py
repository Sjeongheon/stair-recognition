"""First 512x512 -> 256x256, 32-channel alternatives. Focus remains the baseline."""

from __future__ import annotations

from torch import Tensor, nn


class StridedConvStem(nn.Sequential):
    """Ordinary learnable spatial downsampling; unlike Focus it uses a 3x3 RF."""

    def __init__(self, in_channels: int, out_channels: int = 32) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class PoolStem(nn.Sequential):
    """2x2 average or max pooling, then learned 1x1 channel projection."""

    def __init__(self, in_channels: int, out_channels: int = 32, *, pooling: str = "avg") -> None:
        if pooling not in ("avg", "max"):
            raise ValueError("pooling must be 'avg' or 'max'")
        pool = nn.AvgPool2d(2, 2) if pooling == "avg" else nn.MaxPool2d(2, 2)
        super().__init__(
            pool,
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SharedSpatialStem(nn.Module):
    """Modality-specific 1x1 adapters + the *same* learned stride-2 spatial filter.

    Only the spatial conv weights are shared; normalization stays modality-specific
    to avoid mixing RGB/depth BatchNorm statistics.
    """

    def __init__(self, channels: int = 32, intermediate: int = 8) -> None:
        super().__init__()
        self.rgb_adapter = nn.Conv2d(3, intermediate, kernel_size=1, bias=False)
        self.depth_adapter = nn.Conv2d(1, intermediate, kernel_size=1, bias=False)
        self.shared_conv = nn.Conv2d(
            intermediate, channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.rgb_norm = nn.BatchNorm2d(channels)
        self.depth_norm = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, rgb: Tensor, depth: Tensor) -> tuple[Tensor, Tensor]:
        if rgb.shape[0] != depth.shape[0] or rgb.shape[-2:] != depth.shape[-2:]:
            raise ValueError("RGB/depth spatial/batch shapes must match")
        rgb_feature = self.activation(self.rgb_norm(self.shared_conv(self.rgb_adapter(rgb))))
        depth_feature = self.activation(self.depth_norm(self.shared_conv(self.depth_adapter(depth))))
        return rgb_feature, depth_feature
