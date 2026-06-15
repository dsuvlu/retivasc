"""Interpretable vascular feature extraction from binary vessel masks."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from skimage import measure

from retivasc.skeleton import branchpoint_mask, skeletonize_mask


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        msg = f"Expected a 2D mask, got shape {arr.shape}."
        raise ValueError(msg)
    return arr.astype(bool)


def _fov(mask: np.ndarray, fov_mask: np.ndarray | None) -> np.ndarray:
    if fov_mask is None:
        return np.ones(mask.shape, dtype=bool)
    fov_arr = np.asarray(fov_mask, dtype=bool)
    if fov_arr.shape != mask.shape:
        msg = f"fov_mask shape {fov_arr.shape} does not match mask shape {mask.shape}."
        raise ValueError(msg)
    return fov_arr


def _area(fov_mask: np.ndarray) -> int:
    return int(np.count_nonzero(fov_mask))


def vessel_density(mask: np.ndarray, fov_mask: np.ndarray | None = None) -> float:
    """Fraction of field of view occupied by vessel pixels."""
    vessels = _as_bool_mask(mask)
    fov = _fov(vessels, fov_mask)
    area = _area(fov)
    if area == 0:
        return 0.0
    return float(np.count_nonzero(vessels & fov) / area)


def skeleton_length_density(mask: np.ndarray, fov_mask: np.ndarray | None = None) -> float:
    """Skeleton pixels per field-of-view pixel."""
    vessels = _as_bool_mask(mask)
    fov = _fov(vessels, fov_mask)
    area = _area(fov)
    if area == 0:
        return 0.0
    skel = skeletonize_mask(vessels & fov)
    return float(np.count_nonzero(skel) / area)


def branchpoint_density(mask: np.ndarray, fov_mask: np.ndarray | None = None) -> float:
    """Branchpoint junction count normalized by field-of-view area."""
    vessels = _as_bool_mask(mask)
    fov = _fov(vessels, fov_mask)
    area = _area(fov)
    if area == 0:
        return 0.0
    skel = skeletonize_mask(vessels & fov)
    branches = branchpoint_mask(skel)
    branch_labels = measure.label(branches, connectivity=2)
    return float(int(branch_labels.max()) / area)


def _component_arc_and_chord(coords: np.ndarray) -> tuple[float, float]:
    coord_set = {tuple(coord) for coord in coords}
    forward_neighbors = ((0, 1), (1, -1), (1, 0), (1, 1))
    all_neighbors = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    arc = 0.0
    endpoints = []
    for row, col in coord_set:
        neighbor_count = 0
        for d_row, d_col in all_neighbors:
            if (row + d_row, col + d_col) in coord_set:
                neighbor_count += 1
        if neighbor_count <= 1:
            endpoints.append((row, col))

        for d_row, d_col in forward_neighbors:
            if (row + d_row, col + d_col) in coord_set:
                arc += float(np.hypot(d_row, d_col))

    chord_coords = np.asarray(endpoints if len(endpoints) >= 2 else coords)
    distances = pdist(chord_coords)
    chord = float(distances.max()) if distances.size else 0.0
    return arc, chord


def segment_tortuosity(mask: np.ndarray, *, min_segment_length: int = 5) -> float:
    """Mean arc-to-chord tortuosity over vessel segments between junctions.

    Arc length uses 8-connected Euclidean skeleton step length; chord uses endpoint
    distance when endpoints are available. A straight segment is near 1.0; curved
    segments are greater than 1.0. This is a demo proxy, not a calibrated metric.
    """
    skel = skeletonize_mask(_as_bool_mask(mask))
    if not skel.any():
        return 0.0

    segments = skel & ~branchpoint_mask(skel)
    labels = measure.label(segments, connectivity=2)
    ratios: list[float] = []
    for region in measure.regionprops(labels):
        coords = region.coords
        if coords.shape[0] < min_segment_length:
            continue
        arc, chord = _component_arc_and_chord(coords)
        if chord > 0:
            ratios.append(arc / chord)

    if not ratios:
        return 0.0
    return float(np.mean(ratios))


def fractal_dimension_boxcount(mask: np.ndarray) -> float:
    """Estimate box-counting fractal dimension of a binary vessel mask."""
    vessels = _as_bool_mask(mask)
    if not vessels.any():
        return 0.0

    min_dim = min(vessels.shape)
    max_power = int(np.floor(np.log2(min_dim)))
    if max_power < 2:
        return 0.0

    sizes = 2 ** np.arange(1, max_power + 1)
    counts: list[int] = []
    valid_sizes: list[int] = []
    for size in sizes:
        rows = int(np.ceil(vessels.shape[0] / size) * size)
        cols = int(np.ceil(vessels.shape[1] / size) * size)
        padded = np.zeros((rows, cols), dtype=bool)
        padded[: vessels.shape[0], : vessels.shape[1]] = vessels
        boxes = padded.reshape(rows // size, size, cols // size, size).any(axis=(1, 3))
        count = int(np.count_nonzero(boxes))
        if count > 0:
            valid_sizes.append(int(size))
            counts.append(count)

    if len(counts) < 2:
        return 0.0

    slope, _ = np.polyfit(np.log(1 / np.asarray(valid_sizes)), np.log(counts), 1)
    return float(slope)


def connected_component_count(mask: np.ndarray) -> int:
    """Number of 8-connected vascular components."""
    vessels = _as_bool_mask(mask)
    if not vessels.any():
        return 0
    labels = measure.label(vessels, connectivity=2)
    return int(labels.max())


def extract_vascular_features(
    mask: np.ndarray, *, fov_mask: np.ndarray | None = None
) -> dict[str, float]:
    """Return all MVP vascular features in a flat dictionary."""
    return {
        "vessel_density": vessel_density(mask, fov_mask),
        "skeleton_length_density": skeleton_length_density(mask, fov_mask),
        "branchpoint_density": branchpoint_density(mask, fov_mask),
        "fractal_dimension_boxcount": fractal_dimension_boxcount(mask),
        "mean_segment_tortuosity": segment_tortuosity(mask),
        "connected_component_count": float(connected_component_count(mask)),
    }
