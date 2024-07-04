"""Paper-first StairNetV2 1x architecture.

Table 1 fixes stage counts, channels, dilation, and output shapes. SE-ResNeXt
internals and the selective descriptor projections are not fully parameterized
in the paper; those choices follow the later author implementation and are
recorded in ``docs/model_architecture_report.md``.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import Tensor, nn


class Focus(nn.Module):
    """Lossless 2x spatial slicing followed by learned 1x1 channel fusion."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"Focus expects [B,C,H,W], got shape {inputs.shape}")
        if inputs.shape[-2] % 2 or inputs.shape[-1] % 2:
            raise ValueError("Focus requires even input height and width")
        sliced = torch.cat(
            (
                inputs[..., ::2, ::2],
                inputs[..., 1::2, ::2],
                inputs[..., ::2, 1::2],
                inputs[..., 1::2, 1::2],
            ),
            dim=1,
        )
        return self.projection(sliced)


class SqueezeExcitation(nn.Module):
    def __init__(
        self, channels: int, *, reduction_ratio: float = 0.25
    ) -> None:
        super().__init__()
        reduced_channels = int(channels * reduction_ratio)
        if reduced_channels < 1:
            raise ValueError("SE reduced channel count must be positive")
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * self.layers(inputs)


class SEResNeXtBottleneck(nn.Module):
    """SE-ResNeXt residual block with dilation on the spatial 3x3 conv."""

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
        internal_channels = int(out_channels * shrink_ratio)
        if internal_channels < 1 or internal_channels % groups:
            raise ValueError(
                "internal channels must be positive and divisible by groups"
            )
        self.spatial_conv = nn.Conv2d(
            internal_channels,
            internal_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            self.spatial_conv,
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.se = SqueezeExcitation(out_channels, reduction_ratio=se_ratio)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
            # OPTIONAL stride-2 ONLY: replace shortcut with one class from
            # stairnetv2.modules.shortcuts (AvgPoolShortcut, BlurPoolShortcut,
            # PixelUnshuffleShortcut, HaarShortcut). Keep stride-1 identity.
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        # OPTIONAL dropout after this Bottleneck is better controlled at the
        # model/stage call site; do not indiscriminately drop every SE output.
        return self.activation(self.se(self.residual(inputs)) + self.shortcut(inputs))


class SelectiveFusion(nn.Module):
    """Channel-wise competition between RGB and depth feature streams."""

    def __init__(
        self,
        channels: int,
        *,
        reduction: int = 1,
        minimum_descriptor_channels: int = 32,
    ) -> None:
        super().__init__()
        if reduction < 1:
            raise ValueError("reduction must be at least one")
        descriptor_channels = max(channels // reduction, minimum_descriptor_channels)
        self.channels = channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.descriptor = nn.Sequential(
            nn.Conv2d(channels, descriptor_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(descriptor_channels),
            nn.ReLU(inplace=True),
        )
        self.modality_logits = nn.Conv2d(
            descriptor_channels, channels * 2, kernel_size=1, bias=False
        )

    def weights(self, rgb: Tensor, depth: Tensor) -> Tensor:
        if rgb.shape != depth.shape:
            raise ValueError(
                f"selective inputs must have equal shapes: {rgb.shape} != {depth.shape}"
            )
        if rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError(
                f"expected [B,{self.channels},H,W], got shape {rgb.shape}"
            )
        descriptor = self.descriptor(self.pool(rgb + depth))
        batch_size = rgb.shape[0]
        logits = self.modality_logits(descriptor).reshape(
            batch_size, 2, self.channels, 1, 1
        )
        return torch.softmax(logits, dim=1)

    def forward(self, rgb: Tensor, depth: Tensor) -> Tensor:
        weights = self.weights(rgb, depth)
        return rgb * weights[:, 0] + depth * weights[:, 1]


def _shallow_branch(in_channels: int) -> nn.Sequential:
    return nn.Sequential(
        OrderedDict(
            (
                ("focus", Focus(in_channels, 32)),
                ("bottleneck_1_0", SEResNeXtBottleneck(32, 128, stride=2)),
                ("bottleneck_1_1", SEResNeXtBottleneck(128, 128)),
                ("bottleneck_1_2", SEResNeXtBottleneck(128, 128)),
            )
        )
    )


def _deep_stage(*, downsample_in_channels: int) -> nn.Sequential:
    return nn.Sequential(
        OrderedDict(
            (
                (
                    "bottleneck_0",
                    SEResNeXtBottleneck(
                        downsample_in_channels, 256, stride=2
                    ),
                ),
                ("bottleneck_1_d2", SEResNeXtBottleneck(256, 256, dilation=2)),
                ("bottleneck_2", SEResNeXtBottleneck(256, 256)),
                ("bottleneck_3_d4", SEResNeXtBottleneck(256, 256, dilation=4)),
                ("bottleneck_4", SEResNeXtBottleneck(256, 256)),
                ("bottleneck_5_d8", SEResNeXtBottleneck(256, 256, dilation=8)),
                ("bottleneck_6", SEResNeXtBottleneck(256, 256)),
                (
                    "bottleneck_7_d16",
                    SEResNeXtBottleneck(256, 256, dilation=16),
                ),
            )
        )
    )


def _prediction_head(out_channels: int, *, sigmoid: bool) -> nn.Sequential:
    layers: list[tuple[str, nn.Module]] = [
        ("conv_3x3", nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)),
        ("batch_norm", nn.BatchNorm2d(128)),
        ("activation", nn.ReLU(inplace=True)),
        ("conv_1x1", nn.Conv2d(128, out_channels, kernel_size=1)),
    ]
    if sigmoid:
        layers.append(("sigmoid", nn.Sigmoid()))
    return nn.Sequential(OrderedDict(layers))


class StairNetV2(nn.Module):
    """StairNetV2 1x baseline with separate RGB and one-channel depth branches."""

    def __init__(self) -> None:
        super().__init__()
        self.rgb_branch = _shallow_branch(3)
        self.depth_branch = _shallow_branch(1)
        # OPTIONAL:
        # from stairnetv2.modules.extractors import FrozenResNet18Stem
        # self.rgb_branch[0] = FrozenResNet18Stem()  # train dataset RGB must use 'unit'
        # from stairnetv2.modules.stems import PoolStem
        # self.rgb_branch[0] = PoolStem(3)
        # self.depth_branch[0] = PoolStem(1)
        self.selective_fusion = SelectiveFusion(128, reduction=1)
        # OPTIONAL (choose fusion replacement):
        # from stairnetv2.modules.fusions import SeparateDescriptorFusion
        # self.selective_fusion = SeparateDescriptorFusion(128)
        self.deep_stage_2 = _deep_stage(downsample_in_channels=128)
        self.deep_stage_3 = _deep_stage(downsample_in_channels=256)
        self.multiscale = nn.Identity()
        # OPTIONAL: from stairnetv2.modules.multiscale import ASPPBlock
        # self.multiscale = ASPPBlock(256, 256)  # additional, not matched compute
        self.width_downsample = nn.Sequential(
            OrderedDict(
                (
                    (
                        "conv_3x3",
                        nn.Conv2d(
                            256,
                            128,
                            kernel_size=3,
                            stride=(1, 2),
                            padding=1,
                            bias=False,
                        ),
                    ),
                    ("batch_norm", nn.BatchNorm2d(128)),
                    ("activation", nn.ReLU(inplace=True)),
                )
            )
        )
        self.classification_head = _prediction_head(1, sigmoid=False)
        # OPTIONAL nn.Dropout2d(p=0.05)
        self.location_head = _prediction_head(8, sigmoid=True)

    def forward(self, rgb: Tensor, depth: Tensor) -> dict[str, Tensor]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"RGB input must have shape [B,3,H,W], got {rgb.shape}")
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError(
                f"depth input must have shape [B,1,H,W], got {depth.shape}"
            )
        if rgb.shape[0] != depth.shape[0] or rgb.shape[-2:] != depth.shape[-2:]:
            raise ValueError("RGB and depth batch/spatial shapes must match")

        rgb_features = self.rgb_branch(rgb)
        depth_features = self.depth_branch(depth)
        features = self.selective_fusion(rgb_features, depth_features)
        # OPTIONAL dropout
        # features = nn.functional.dropout2d(features, p=0.05, training=self.training)
        features = self.deep_stage_2(features)
        features = self.deep_stage_3(features)
        features = self.multiscale(features)
        # OPTIONAL dropout nn.Dropout2d(p=0.05) ONLY
        features = self.width_downsample(features)
        return {
            "classification_logits": self.classification_head(features),
            "location": self.location_head(features),
        }
