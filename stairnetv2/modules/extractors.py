"""Optional pretrained feature adapter, manually substituted for the RGB Focus stem."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FrozenResNet18Stem(nn.Module):
    """RGB [B,3,512,512] -> [B,32,256,256] using a frozen ResNet18 encoder.

    The pretrained encoder runs at 256x256 and its layer1 features are 64x64.
    A *trainable* channel projection and spatial refinement upsample them, while
    a trainable 256x256 RGB skip retains high-resolution edge cues. Upsampling
    alone does not recover information lost by the pretrained encoder.

    Expects RGB tensors in [0,1]; ImageNet normalization happens *inside*.
    This option needs torchvision and a cached/downloadable pretrained weight.
    """

    def __init__(self, *, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as error:
            raise ImportError("FrozenResNet18Stem requires torchvision") from error
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        self.encoder = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool, model.layer1
        )
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.project = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        self.refine = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.detail_skip = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.activation = nn.ReLU(inplace=True)
        self.register_buffer("imagenet_mean", torch.tensor((0.485, 0.456, 0.406))[None, :, None, None])
        self.register_buffer("imagenet_std", torch.tensor((0.229, 0.224, 0.225))[None, :, None, None])

    def train(self, mode: bool = True) -> "FrozenResNet18Stem":
        super().train(mode)
        self.encoder.eval()  # keep frozen BN running statistics unchanged
        return self

    def forward(self, rgb: Tensor) -> Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"expected RGB [B,3,H,W], got {tuple(rgb.shape)}")
        image = F.interpolate(
            rgb, size=(256, 256), mode="bilinear", align_corners=False
        )
        normalized = (image - self.imagenet_mean) / self.imagenet_std
        with torch.no_grad():
            feature = self.encoder(normalized)
        feature = self.project(feature)
        feature = F.interpolate(feature, size=(256, 256), mode="bilinear", align_corners=False)
        return self.activation(self.refine(feature) + self.detail_skip(image))
