"""Binary segmentation metrics for vessel masks."""

from __future__ import annotations

import numpy as np


def _as_bool_pair(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    if true.shape != pred.shape:
        msg = f"Shape mismatch: y_true {true.shape}, y_pred {pred.shape}."
        raise ValueError(msg)
    if mask is not None:
        eval_mask = np.asarray(mask, dtype=bool)
        if eval_mask.shape != true.shape:
            msg = f"mask shape {eval_mask.shape} does not match y_true shape {true.shape}."
            raise ValueError(msg)
        true = true[eval_mask]
        pred = pred[eval_mask]
    return true, pred


def dice_score(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """Dice coefficient. Two empty masks are treated as a perfect match."""
    true, pred = _as_bool_pair(y_true, y_pred, mask)
    denom = np.count_nonzero(true) + np.count_nonzero(pred)
    if denom == 0:
        return 1.0
    return float(2 * np.count_nonzero(true & pred) / denom)


def iou_score(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """Intersection over union. Two empty masks are treated as a perfect match."""
    true, pred = _as_bool_pair(y_true, y_pred, mask)
    union = np.count_nonzero(true | pred)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(true & pred) / union)


def sensitivity(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """True-positive rate. If there are no positives, return 1.0."""
    true, pred = _as_bool_pair(y_true, y_pred, mask)
    tp = np.count_nonzero(true & pred)
    fn = np.count_nonzero(true & ~pred)
    denom = tp + fn
    if denom == 0:
        return 1.0
    return float(tp / denom)


def precision_score_binary(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """Positive predictive value. If there are no predicted positives, return 1.0."""
    true, pred = _as_bool_pair(y_true, y_pred, mask)
    tp = np.count_nonzero(true & pred)
    fp = np.count_nonzero(~true & pred)
    denom = tp + fp
    if denom == 0:
        return 1.0
    return float(tp / denom)


def recall_score_binary(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """Alias for sensitivity."""
    return sensitivity(y_true, y_pred, mask)


def specificity(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """True-negative rate. If there are no negatives, return 1.0."""
    true, pred = _as_bool_pair(y_true, y_pred, mask)
    tn = np.count_nonzero(~true & ~pred)
    fp = np.count_nonzero(~true & pred)
    denom = tn + fp
    if denom == 0:
        return 1.0
    return float(tn / denom)


def specificity_score_binary(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """Alias for specificity."""
    return specificity(y_true, y_pred, mask)
