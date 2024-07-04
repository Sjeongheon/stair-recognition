import pytest
import torch
import numpy as np
from PIL import Image
from torch import nn

from stairnetv2.initialization import (
    initialize_custom_normal,
    initialize_he_normal,
    initialize_normal,
)
from stairnetv2.loss import StairNetV2Loss
from stairnetv2.loss_alternatives import BCEPlusDiceLoss, BinaryFocalLoss, SoftDiceLoss
from stairnetv2.training_dataset import normalize_depth, normalize_rgb
from stairnetv2.training_dataset import StairNetDataset
from stairnetv2.dataset import DatasetIntegrityError
from stairnetv2.training_policy import EarlyStopping, validation_score


def test_location_loss_uses_gt_mask_and_accepts_alternative_losses():
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    cls = torch.zeros(1, 1, 2, 2)
    cls[0, 0, 0, 0] = 1
    loc = torch.zeros(1, 8, 2, 2, requires_grad=True)
    truth = torch.ones_like(loc)
    criterion = StairNetV2Loss(
        alpha=10, beta=10,
        classification_loss_fn=BinaryFocalLoss(),
        location_loss_fn=nn.SmoothL1Loss(beta=0.1, reduction="none"),
    )
    out = criterion(logits, loc, cls, truth)
    assert out.total.ndim == 0
    out.total.backward()
    assert loc.grad is not None
    torch.testing.assert_close(loc.grad[:, :, 1, 1], torch.zeros_like(loc.grad[:, :, 1, 1]))


def test_dynamic_weights_atomic_update_preserves_total():
    criterion = StairNetV2Loss(alpha=0.5, beta=19.5, sigma=0.5)
    update = criterion.update_weights(x_error=0.0, y_error=4.0)
    assert update.alpha_after == 0.5 and update.beta_after == 19.5
    criterion = StairNetV2Loss(alpha=10, beta=10, sigma=0.5)
    update = criterion.update_weights(x_error=3.0, y_error=1.0)
    assert update.alpha_after > 10 and update.beta_after < 10
    assert update.alpha_after + update.beta_after == pytest.approx(20.0)


def test_loss_rejects_reduced_coordinate_tensor():
    c = StairNetV2Loss(location_loss_fn=nn.L1Loss(reduction="mean"))
    with pytest.raises(ValueError, match="per-coordinate"):
        c(torch.zeros(1, 1, 2, 2), torch.zeros(1, 8, 2, 2),
          torch.zeros(1, 1, 2, 2), torch.zeros(1, 8, 2, 2))


def test_other_classification_losses_produce_scalar():
    x, y = torch.randn(2, 1, 4, 4), torch.randint(0, 2, (2, 1, 4, 4)).float()
    for loss in (BCEPlusDiceLoss(), SoftDiceLoss(), BinaryFocalLoss()):
        assert loss(x, y).shape == ()


def test_8bit_normalization_and_reject_unknown_modes():
    rgb = Image.fromarray(np.array([[[0, 128, 255]]], dtype=np.uint8), "RGB")
    depth = Image.fromarray(np.array([[0, 255]], dtype=np.uint8), "L")
    unit = normalize_rgb(rgb, "unit")
    torch.testing.assert_close(unit[:, 0, 0], torch.tensor([0., 128 / 255., 1.]))
    torch.testing.assert_close(normalize_depth(depth, "baseline")[0, 0], torch.tensor([-1., 1.]))
    with pytest.raises(ValueError, match="mean and std"):
        normalize_rgb(rgb, "dataset_mean_std")
    with pytest.raises(ValueError, match="expected image mode"):
        normalize_depth(rgb, "unit")


def test_supplied_depth_format_baseline_preserves_values_and_channel():
    """Synthetic 512x512 L PNG with the observed uploaded sample's value range."""
    array = np.zeros((512, 512), dtype=np.uint8)
    array[0, 0] = 239
    array[0, 1] = 51
    depth = Image.fromarray(array, mode="L")
    out = normalize_depth(depth, "baseline")
    assert out.shape == (1, 512, 512)
    assert out.dtype == torch.float32
    assert out[0, 1, 1].item() == pytest.approx(-1.0)  # 0 is NOT silently replaced
    assert out[0, 0, 0].item() == pytest.approx(2 * (239 / 255) - 1)
    assert out[0, 0, 1].item() == pytest.approx(2 * (51 / 255) - 1)
    assert normalize_depth(depth, "unit")[0, 0, 0].item() == pytest.approx(239 / 255)


def test_original_reproduction_exact_stem_pairing_and_dataset_shapes(tmp_path):
    stem = "color_0_night_l"  # the `_l` is part of the actual stem
    root = tmp_path / "dataset"
    for folder in ("images", "depthes", "labels"):
        (root / "train" / folder).mkdir(parents=True)
    rgb = np.zeros((512, 512, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    depth = np.zeros((512, 512), dtype=np.uint8)
    depth[20, 10] = 239
    Image.fromarray(rgb, mode="RGB").save(root / "train" / "images" / f"{stem}.png")
    Image.fromarray(depth, mode="L").save(root / "train" / "depthes" / f"{stem}.png")
    (root / "train" / "labels" / f"{stem}.txt").write_text(
        "0 20 40 420 40\n", encoding="utf-8"
    )

    dataset = StairNetDataset(root, split="train")
    sample = dataset[0]
    assert sample["sample_id"] == stem
    assert sample["rgb"].shape == (3, 512, 512)
    assert sample["depth"].shape == (1, 512, 512)
    assert sample["classification"].shape == (1, 32, 16)
    assert sample["location"].shape == (8, 32, 16)
    assert sample["depth"][0, 20, 10].item() == pytest.approx(2 * (239 / 255) - 1)

    from scripts.inspect_dataset import inspect_split

    report = inspect_split(root, "train", limit=None)
    assert report["paired_samples"] == report["inspected_samples"] == 1
    assert report["samples"][0]["depth_mode"] == "L"

    (root / "train" / "images" / f"{stem}.png").rename(
        root / "train" / "images" / "rgb_0_night_l.png"
    )
    with pytest.raises(DatasetIntegrityError, match="missing"):
        StairNetDataset(root, split="train")


def test_initializers_are_explicit_and_ignore_frozen_weights():
    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.Conv2d(8, 8, 1))
    model[0].requires_grad_(False)
    frozen = model[0].weight.detach().clone()
    initialize_he_normal(model)
    initialize_normal(model, mean=0.1, std=0.01)
    initialize_custom_normal(model, mean=0.0, std=0.02)
    torch.testing.assert_close(frozen, model[0].weight)
    with pytest.raises(ValueError):
        initialize_custom_normal(model, mean=0.0, std=0.0)


def test_validation_score_is_independent_from_dynamic_loss():
    metrics = {"iou": 0.8, "location_mae_x": 0.1,
               "location_mae_y": 0.3, "positive_cells": 10}
    assert validation_score(metrics) == pytest.approx(0.7)
    stopper = EarlyStopping(patience=2)
    assert not stopper.step(0.6)
    assert not stopper.step(0.5)
    assert stopper.step(0.4)
