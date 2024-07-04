import pytest
import torch
from torch.nn import functional as F

from stairnetv2.model import StairNetV2
from stairnetv2.modules.blocks import AntiAliasedSEResNeXtBottleneck
from stairnetv2.modules.fusions import (
    ChannelSpatialFusion,
    ConcatFusion,
    SeparateDescriptorFusion,
    SpatialFusion,
)
from stairnetv2.modules.multiscale import ASPPBlock
from stairnetv2.modules.shortcuts import (
    AvgPoolShortcut,
    BlurPool,
    BlurPoolShortcut,
    HaarDownsample,
    HaarShortcut,
    PixelUnshuffleShortcut,
)
from stairnetv2.modules.stems import PoolStem, SharedSpatialStem, StridedConvStem


def test_downsampling_shortcuts_keep_residual_shape():
    x = torch.randn(2, 32, 16, 16)
    for layer in (
        AvgPoolShortcut(32, 128), BlurPoolShortcut(32, 128),
        PixelUnshuffleShortcut(32, 128), HaarShortcut(32, 128),
    ):
        assert layer(x).shape == (2, 128, 8, 8)


def test_pixel_unshuffle_is_lossless_and_haar_is_invertible():
    x = torch.randn(1, 2, 8, 8)
    rearranged = PixelUnshuffleShortcut(2, 8)(x)
    torch.testing.assert_close(F.pixel_shuffle(rearranged, 2), x)
    dwt = HaarDownsample(2)
    bands = dwt(x)
    assert bands.shape == (1, 8, 4, 4)
    torch.testing.assert_close(
        F.conv_transpose2d(bands, dwt.weight, stride=2, groups=2),
        x,
        rtol=1e-5,
        atol=1e-5,
    )


def test_blurpool_does_not_learn_and_rejects_wrong_channels():
    blur = BlurPool(3)
    assert len(list(blur.parameters())) == 0
    assert blur(torch.randn(1, 3, 16, 16)).shape == (1, 3, 8, 8)
    with pytest.raises(ValueError):
        blur(torch.randn(1, 2, 16, 16))


def test_fusion_shapes_and_softmax_axis():
    rgb = torch.randn(2, 8, 10, 12)
    depth = torch.randn(2, 8, 10, 12)
    for fusion in (ConcatFusion(8), SeparateDescriptorFusion(8), SpatialFusion(8), ChannelSpatialFusion(8)):
        assert fusion(rgb, depth).shape == (2, 8, 10, 12)
        if hasattr(fusion, "weights"):
            weights = fusion.weights(rgb, depth)
            torch.testing.assert_close(weights.sum(dim=1), torch.ones_like(weights.sum(dim=1)))


def test_stems_multiscale_and_anti_aliased_block():
    rgb = torch.randn(2, 3, 32, 32)
    depth = torch.randn(2, 1, 32, 32)
    assert StridedConvStem(3)(rgb).shape == (2, 32, 16, 16)
    assert PoolStem(3)(rgb).shape == (2, 32, 16, 16)
    assert PoolStem(1, pooling="max")(depth).shape == (2, 32, 16, 16)
    sr, sd = SharedSpatialStem()(rgb, depth)
    assert sr.shape == sd.shape == (2, 32, 16, 16)
    anti = AntiAliasedSEResNeXtBottleneck(32, 128, stride=2)
    assert anti(torch.randn(2, 32, 16, 16)).shape == (2, 128, 8, 8)
    aspp = ASPPBlock(32, 32, 8, dilations=(2, 4))
    assert aspp(torch.randn(2, 32, 8, 8)).shape == (2, 32, 8, 8)


def test_baseline_small_input_outputs_expected_grid():
    model = StairNetV2().eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64), torch.randn(1, 1, 64, 64))
    assert out["classification_logits"].shape == (1, 1, 4, 2)
    assert out["location"].shape == (1, 8, 4, 2)
    assert out["location"].min().item() >= 0
    assert out["location"].max().item() <= 1


def test_resnet_adapter_shape_without_downloading_weights():
    pytest.importorskip("torchvision")
    from stairnetv2.modules.extractors import FrozenResNet18Stem

    model = FrozenResNet18Stem(pretrained=False).train()
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert all(not isinstance(m, torch.nn.BatchNorm2d) or not m.training for m in model.encoder.modules())
    with torch.no_grad():
        output = model(torch.rand(1, 3, 32, 32))
    assert output.shape == (1, 32, 256, 256)
