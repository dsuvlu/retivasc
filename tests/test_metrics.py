import numpy as np

from retivasc.metrics import (
    dice_score,
    iou_score,
    precision_score_binary,
    recall_score_binary,
    sensitivity,
    specificity,
    specificity_score_binary,
)


def test_dice_identical_mask_is_one():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 2:5] = True

    assert dice_score(mask, mask) == 1.0


def test_dice_nonempty_vs_empty_is_zero():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 2:5] = True
    empty = np.zeros_like(mask)

    assert dice_score(mask, empty) == 0.0


def test_iou_identical_mask_is_one():
    mask = np.eye(8, dtype=bool)

    assert iou_score(mask, mask) == 1.0


def test_iou_empty_masks_are_one():
    empty = np.zeros((8, 8), dtype=bool)

    assert iou_score(empty, empty) == 1.0


def test_sensitivity_specificity_known_arrays():
    y_true = np.array([[1, 1, 0, 0]], dtype=bool)
    y_pred = np.array([[1, 0, 1, 0]], dtype=bool)

    assert sensitivity(y_true, y_pred) == 0.5
    assert specificity(y_true, y_pred) == 0.5
    assert precision_score_binary(y_true, y_pred) == 0.5
    assert recall_score_binary(y_true, y_pred) == 0.5
    assert specificity_score_binary(y_true, y_pred) == 0.5


def test_metric_mask_excludes_pixels_outside_field_of_view():
    y_true = np.array([[1, 0, 0, 0]], dtype=bool)
    y_pred = np.array([[1, 1, 0, 0]], dtype=bool)
    fov = np.array([[1, 1, 0, 0]], dtype=bool)

    assert specificity(y_true, y_pred) == 2 / 3
    assert specificity(y_true, y_pred, mask=fov) == 0.0
    assert specificity_score_binary(y_true, y_pred, mask=fov) == 0.0


def test_sensitivity_empty_positive_convention_is_one():
    y_true = np.zeros((8, 8), dtype=bool)
    y_pred = np.zeros((8, 8), dtype=bool)

    assert sensitivity(y_true, y_pred) == 1.0


def test_specificity_empty_negative_convention_is_one():
    y_true = np.ones((8, 8), dtype=bool)
    y_pred = np.ones((8, 8), dtype=bool)

    assert specificity(y_true, y_pred) == 1.0
