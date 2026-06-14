"""Binary segmentation metrics for vessel masks."""

from __future__ import annotations

import numpy as np


def _as_bool_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    if true.shape != pred.shape:
        msg = f"Shape mismatch: y_true {true.shape}, y_pred {pred.shape}."
        raise ValueError(msg)
    return true, pred


def dice_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Dice coefficient. Two empty masks are treated as a perfect match."""
    true, pred = _as_bool_pair(y_true, y_pred)
    denom = np.count_nonzero(true) + np.count_nonzero(pred)
    if denom == 0:
        return 1.0
    return float(2 * np.count_nonzero(true & pred) / denom)


def iou_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Intersection over union. Two empty masks are treated as a perfect match."""
    true, pred = _as_bool_pair(y_true, y_pred)
    union = np.count_nonzero(true | pred)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(true & pred) / union)


def sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True-positive rate. If there are no positives, return 1.0."""
    true, pred = _as_bool_pair(y_true, y_pred)
    tp = np.count_nonzero(true & pred)
    fn = np.count_nonzero(true & ~pred)
    denom = tp + fn
    if denom == 0:
        return 1.0
    return float(tp / denom)


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True-negative rate. If there are no negatives, return 1.0."""
    true, pred = _as_bool_pair(y_true, y_pred)
    tn = np.count_nonzero(~true & ~pred)
    fp = np.count_nonzero(~true & pred)
    denom = tn + fp
    if denom == 0:
        return 1.0
    return float(tn / denom)
