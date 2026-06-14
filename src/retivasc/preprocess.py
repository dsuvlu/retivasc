"""Small image preprocessing helpers."""

from __future__ import annotations

import numpy as np
from skimage.transform import resize


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA images to grayscale; leave 2D arrays unchanged."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        msg = f"Expected a 2D grayscale or 3D color image, got shape {arr.shape}."
        raise ValueError(msg)
    if arr.shape[-1] == 1:
        return arr[..., 0]
    if arr.shape[-1] < 3:
        msg = f"Expected at least 3 color channels, got shape {arr.shape}."
        raise ValueError(msg)

    rgb = arr[..., :3].astype(float, copy=False)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Return a finite float image scaled to [0, 1]."""
    arr = np.asarray(image, dtype=float)
    if arr.size == 0:
        return arr

    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)

    lo = float(np.nanmin(arr[finite]))
    hi = float(np.nanmax(arr[finite]))
    if hi <= lo:
        return np.zeros_like(arr, dtype=float)

    scaled = (arr - lo) / (hi - lo)
    return np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)


def resize_mask_to_max_dim(mask: np.ndarray, max_dim: int) -> np.ndarray:
    """Resize a 2D boolean mask so its largest dimension is at most max_dim."""
    if max_dim <= 0:
        msg = "max_dim must be positive."
        raise ValueError(msg)

    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2:
        msg = f"Expected a 2D mask, got shape {arr.shape}."
        raise ValueError(msg)
    current_max = max(arr.shape)
    if current_max <= max_dim:
        return arr

    scale = max_dim / current_max
    output_shape = tuple(max(1, int(round(dim * scale))) for dim in arr.shape)
    resized = resize(
        arr.astype(float),
        output_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return resized >= 0.5
