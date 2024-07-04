"""Optional parallel multi-scale module; not part of the StairNetV2 baseline."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ASPPBlock(nn.Module):
    """Parallel 1x1/dilated-3x3/GAP branches, concat then project."""

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        branch_channels: int = 64,
        dilations: tuple[int, ...] = (2, 4, 8, 16),
    ) -> None:
        super().__init__()
        if min(dilations, default=1) < 1:
            raise ValueError("dilations must be positive")
        self.local = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.dilated = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    in_channels, branch_channels, kernel_size=3,
                    dilation=d, padding=d, bias=False,
                ),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )
            for d in dilations
        )
        # No BatchNorm at the GAP [B,C,1,1] stage: batch size 1 should work.
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        n_branches = 2 + len(dilations)
        self.fuse = nn.Sequential(
            nn.Conv2d(n_branches * branch_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        features = [self.local(x), *(branch(x) for branch in self.dilated)]
        global_feature = self.global_branch(x)
        features.append(F.interpolate(global_feature, size=x.shape[-2:], mode="nearest"))
        return self.fuse(torch.cat(features, dim=1))
