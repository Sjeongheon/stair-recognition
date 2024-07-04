"""Explicit, manually called alternatives. No initialization runs on import."""

from __future__ import annotations

import math

from torch import nn


def _initialize_trainable(module: nn.Module, method: str, *, mean: float = 0.0, std: float = 0.02) -> None:
    """Only initialize trainable weights; frozen pretrained extractors are untouched."""
    for layer in module.modules():
        if not isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            continue
        if layer.weight is not None and layer.weight.requires_grad:
            if method == "he":
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            elif method == "normal":
                nn.init.normal_(layer.weight, mean=mean, std=std)
            elif method == "custom":
                nn.init.normal_(layer.weight, mean=mean, std=std)
            else:
                raise ValueError(f"unsupported initialization: {method}")
        if layer.bias is not None and layer.bias.requires_grad:
            nn.init.zeros_(layer.bias)


def initialize_he_normal(module: nn.Module) -> None:
    """He normal with fan_out convention, useful for ReLU convolution paths."""
    _initialize_trainable(module, "he")


def initialize_normal(module: nn.Module, *, mean: float = 0.0, std: float = 0.02) -> None:
    if not math.isfinite(mean) or not 0 < std < float("inf"):
        raise ValueError("mean must be finite and std must be positive and finite")
    _initialize_trainable(module, "normal", mean=mean, std=std)


def initialize_custom_normal(module: nn.Module, *, mean: float, std: float) -> None:
    """Custom distribution is not guessed: caller must choose mean and std."""
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError("custom normal requires finite mean and positive std")
    _initialize_trainable(module, "custom", mean=mean, std=std)
