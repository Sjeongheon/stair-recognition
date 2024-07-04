"""Optional whole-block anti-aliased replacement for stride-2 SE-ResNeXt blocks."""

from __future__ import annotations

from torch import Tensor, nn

from stairnetv2.modules.shortcuts import BlurPool, BlurPoolShortcut


class AntiAliasedSEResNeXtBottleneck(nn.Module):
    """Keep the SE-ResNeXt bottleneck but separate stride-2 filtering/sampling.

    At stride 1 it uses an ordinary grouped/dilated 3x3 convolution. At stride
    2 both the main path and projection shortcut get LPF before downsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 32,
        shrink_ratio: float = 0.5,
        se_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("only stride 1 or 2 is supported")
        middle = int(out_channels * shrink_ratio)
        if middle < 1 or middle % groups:
            raise ValueError("internal channels must be positive and divisible by groups")
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, middle, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                middle, middle, kernel_size=3, stride=1, dilation=dilation,
                padding=dilation, groups=groups, bias=False,
            ),
            nn.BatchNorm2d(middle),
            nn.ReLU(inplace=True),
            BlurPool(middle, stride=2) if stride == 2 else nn.Identity(),
            nn.Conv2d(middle, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # The original baseline SE class is imported lazily to avoid a model
        # import cycle when this alternative is imported in model.py.
        from stairnetv2.model import SqueezeExcitation

        self.se = SqueezeExcitation(out_channels, reduction_ratio=se_ratio)
        if stride == 2:
            self.shortcut = BlurPoolShortcut(in_channels, out_channels)
        elif in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.se(self.residual(x)) + self.shortcut(x))
