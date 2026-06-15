"""Classical, GPU-free vessel segmentation baseline."""

from __future__ import annotations

import numpy as np
from skimage import filters, morphology

from retivasc.preprocess import ensure_grayscale, normalize_image


def classical_vesselness_mask(
    image: np.ndarray, *, threshold: str = "otsu", black_ridges: bool = False
) -> np.ndarray:
    """Return a binary vessel mask using Frangi vesselness plus thresholding.

    The default assumes bright vessels, which matches OCTA projections. Set
    ``black_ridges=True`` for dark-vessel fundus-style inputs.
    """
    gray = normalize_image(ensure_grayscale(image))
    if gray.size == 0:
        return np.zeros_like(gray, dtype=bool)

    vesselness = filters.frangi(gray, black_ridges=black_ridges)
    vesselness = normalize_image(vesselness)
    if not vesselness.any():
        vesselness = gray

    if threshold == "otsu":
        cutoff = filters.threshold_otsu(vesselness)
    elif threshold == "yen":
        cutoff = filters.threshold_yen(vesselness)
    elif threshold.startswith("percentile:"):
        cutoff = float(np.percentile(vesselness, float(threshold.split(":", maxsplit=1)[1])))
    else:
        msg = "threshold must be 'otsu', 'yen', or 'percentile:<value>'."
        raise ValueError(msg)

    return cleanup_mask(vesselness > cutoff)


def cleanup_mask(mask: np.ndarray, *, min_size: int = 16) -> np.ndarray:
    """Remove small components and fill obvious holes."""
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.size == 0:
        return mask_bool
    max_size = max(0, min_size - 1)
    cleaned = morphology.remove_small_objects(mask_bool, max_size=max_size)
    cleaned = morphology.closing(cleaned, morphology.disk(1))
    cleaned = morphology.remove_small_holes(cleaned, max_size=max_size)
    return cleaned.astype(bool)
