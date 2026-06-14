"""Skeleton and skeleton-neighborhood utilities."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Return a 1-pixel-wide boolean skeleton."""
    return skeletonize(np.asarray(mask, dtype=bool))


def _neighbor_count(skel: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    return ndimage.convolve(np.asarray(skel, dtype=int), kernel, mode="constant", cval=0)


def branchpoint_mask(skel: np.ndarray) -> np.ndarray:
    """Return skeleton pixels with at least three 8-connected skeleton neighbors."""
    skel_bool = np.asarray(skel, dtype=bool)
    return skel_bool & (_neighbor_count(skel_bool) >= 3)


def endpoint_mask(skel: np.ndarray) -> np.ndarray:
    """Return skeleton pixels with exactly one 8-connected skeleton neighbor."""
    skel_bool = np.asarray(skel, dtype=bool)
    return skel_bool & (_neighbor_count(skel_bool) == 1)
