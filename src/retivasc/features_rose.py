"""ROSE macula OCTA feature extraction."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import measure

from retivasc.features import (
    DEFAULT_TORTUOSITY_THRESHOLD,
    _as_bool_mask,
    _fov,
    branchpoint_density,
    connected_component_count,
    fractal_dimension_boxcount,
    segment_tortuosities,
    skeleton_length_density,
    vessel_density,
    vessel_widths_on_skeleton,
)
from retivasc.skeleton import endpoint_mask, skeletonize_mask

ROSE_MACULA_OCTA_CAVEAT = (
    "ROSE macula OCTA masks do not include the optic disc, artery/vein labels, "
    "or large feeder-vessel context needed for Reagan-style AVC or CRAE/CRVE/AVR."
)


def extract_rose_features(
    mask: np.ndarray,
    *,
    fov_mask: np.ndarray | None = None,
    layer: str | None = None,
    pixel_size_um: float | None = None,
) -> dict[str, float]:
    """Return ROSE-compatible macula OCTA vascular features.

    The honest early bridge to Reagan et al. 2025 is tortuosity burden. Density,
    dropout heterogeneity, fractal dimension, endpoints, and component summaries
    are late or interpretive context features. Caliber asymmetry and validated
    arteriovenous crossing features are intentionally excluded.
    """
    vessels, area = _analysis_mask_and_area(mask, fov_mask)
    skel = skeletonize_mask(vessels)
    endpoints = endpoint_mask(skel)
    labels = measure.label(vessels, connectivity=2)
    regions = measure.regionprops(labels)
    vessel_area = int(np.count_nonzero(vessels))
    largest_area = max((region.area for region in regions), default=0)
    small_area = sum(region.area for region in regions if region.area < 16)
    tortuosities, segment_lengths = segment_tortuosities(vessels)
    tortuous = tortuosities >= DEFAULT_TORTUOSITY_THRESHOLD
    total_arc = float(np.sum(segment_lengths))
    tortuous_arc = float(np.sum(segment_lengths[tortuous])) if total_arc > 0 else 0.0
    caliber = vessel_widths_on_skeleton(vessels)
    dropout_heterogeneity = _tile_density_cv(vessels)

    _ = (layer, pixel_size_um)
    return {
        "tortuous_segment_fraction": float(np.mean(tortuous)) if tortuosities.size else 0.0,
        "tortuous_length_fraction": float(tortuous_arc / total_arc) if total_arc > 0 else 0.0,
        "vessel_density": vessel_density(vessels, fov_mask),
        "dropout_heterogeneity": dropout_heterogeneity,
        "vessel_area_fraction": vessel_density(vessels, fov_mask),
        "skeleton_length_density": skeleton_length_density(vessels, fov_mask),
        "branchpoint_density": branchpoint_density(vessels, fov_mask),
        "endpoint_density": float(np.count_nonzero(endpoints) / area) if area else 0.0,
        "connected_component_count": float(connected_component_count(vessels)),
        "largest_component_fraction": float(largest_area / vessel_area) if vessel_area else 0.0,
        "fractal_dimension_boxcount": fractal_dimension_boxcount(vessels),
        "mean_segment_length_px": _nanmean(segment_lengths),
        "median_segment_length_px": _nanmedian(segment_lengths),
        "mean_tortuosity_arc_chord": _nanmean(tortuosities),
        "high_tortuosity_fraction": float(np.mean(tortuous)) if tortuosities.size else 0.0,
        "caliber_proxy_mean_px": _nanmean(caliber),
        "caliber_proxy_median_px": _nanmedian(caliber),
        "caliber_proxy_std_px": float(np.std(caliber)) if caliber.size else 0.0,
        "hole_fraction_or_dropout_proxy": _hole_fraction(vessels),
        "small_component_fraction": float(small_area / vessel_area) if vessel_area else 0.0,
        "orientation_entropy": _orientation_entropy(skel),
    }


def _analysis_mask_and_area(
    mask: np.ndarray, fov_mask: np.ndarray | None
) -> tuple[np.ndarray, int]:
    vessels = _as_bool_mask(mask)
    fov = _fov(vessels, fov_mask)
    return vessels & fov, int(np.count_nonzero(fov))


def _tile_density_cv(mask: np.ndarray, *, grid_size: int = 3) -> float:
    rows = np.array_split(np.arange(mask.shape[0]), grid_size)
    cols = np.array_split(np.arange(mask.shape[1]), grid_size)
    densities: list[float] = []
    for row_idx in rows:
        for col_idx in cols:
            if row_idx.size == 0 or col_idx.size == 0:
                continue
            tile = mask[np.ix_(row_idx, col_idx)]
            densities.append(float(np.count_nonzero(tile) / tile.size) if tile.size else 0.0)
    values = np.asarray(densities, dtype=float)
    mean_value = float(np.mean(values)) if values.size else 0.0
    return float(np.std(values) / mean_value) if mean_value > 0 else 0.0


def _hole_fraction(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    filled = ndimage.binary_fill_holes(mask)
    holes = filled & ~mask
    return float(np.count_nonzero(holes) / np.count_nonzero(filled)) if filled.any() else 0.0


def _orientation_entropy(skel: np.ndarray, bins: int = 8) -> float:
    coords = np.argwhere(skel)
    if coords.shape[0] < 2:
        return 0.0
    gradient_y, gradient_x = np.gradient(skel.astype(float))
    angles = np.arctan2(gradient_y[skel], gradient_x[skel])
    if angles.size == 0:
        return 0.0
    folded = np.mod(angles, np.pi)
    counts, _ = np.histogram(folded, bins=bins, range=(0.0, np.pi))
    probabilities = counts[counts > 0] / counts.sum() if counts.sum() else np.asarray([])
    if probabilities.size == 0:
        return 0.0
    return float(-(probabilities * np.log2(probabilities)).sum() / np.log2(bins))


def _nanmean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if arr.size else 0.0


def _nanmedian(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmedian(arr)) if arr.size else 0.0
