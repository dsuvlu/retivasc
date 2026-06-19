"""Interpretable vascular feature extraction from binary vessel masks."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve, distance_transform_edt
from scipy.spatial.distance import pdist
from skimage import measure

from retivasc.skeleton import branchpoint_mask, skeletonize_mask

DEFAULT_TORTUOSITY_THRESHOLD = 1.10
_NEIGHBOR_KERNEL = np.ones((3, 3), dtype=int)
_NEIGHBOR_KERNEL[1, 1] = 0
_OPPOSITE_NEIGHBOR_PAIRS = (
    ((-1, 0), (1, 0)),
    ((0, -1), (0, 1)),
    ((-1, -1), (1, 1)),
    ((-1, 1), (1, -1)),
)


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


def _skeleton_neighbor_count(skel: np.ndarray) -> np.ndarray:
    return convolve(np.asarray(skel, dtype=int), _NEIGHBOR_KERNEL, mode="constant", cval=0)


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


def segment_tortuosities(
    mask: np.ndarray, *, min_segment_length: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Per-segment arc-chord tortuosity and per-segment arc length in pixels.

    Returns tortuosities and arc lengths, one entry per non-branch skeleton segment
    with at least min_segment_length pixels. A straight segment is near 1.0, curved
    segments are greater. This is the unit of analysis Reagan et al. 2025 uses for
    tortuosity occurrence scoring.
    """
    skel = skeletonize_mask(_as_bool_mask(mask))
    if not skel.any():
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    segments = skel & ~branchpoint_mask(skel)
    labels = measure.label(segments, connectivity=2)
    ratios: list[float] = []
    arc_lengths: list[float] = []
    for region in measure.regionprops(labels):
        coords = region.coords
        if coords.shape[0] < min_segment_length:
            continue
        arc, chord = _component_arc_and_chord(coords)
        if chord <= 0:
            continue
        ratios.append(float(arc / chord))
        arc_lengths.append(float(arc))
    return np.asarray(ratios, dtype=float), np.asarray(arc_lengths, dtype=float)


def tortuosity_burden(
    mask: np.ndarray,
    *,
    threshold: float = DEFAULT_TORTUOSITY_THRESHOLD,
    min_segment_length: int = 5,
) -> dict[str, float]:
    """Tortuosity burden features, Reagan et al. 2025 occurrence percent.

    Faithful to the paper's per-vessel occurrence percent: a handful of twisted
    vessels are the phenotype, so report the fraction above a threshold, not only a
    whole-image mean. Timing: 6mo onset, statistical significance at 12mo CT/TT
    female. The threshold and absolute magnitudes are field-of-view and pixel-size
    dependent, comparable only within matched acquisition.
    """
    tortuosities, arc_lengths = segment_tortuosities(mask, min_segment_length=min_segment_length)
    if tortuosities.size == 0:
        return {
            "mean_segment_tortuosity": 0.0,
            "median_segment_tortuosity": 0.0,
            "p90_segment_tortuosity": 0.0,
            "p95_segment_tortuosity": 0.0,
            "tortuous_segment_fraction": 0.0,
            "tortuous_length_fraction": 0.0,
        }
    tortuous = tortuosities >= threshold
    total_arc = float(np.sum(arc_lengths))
    tortuous_arc = float(np.sum(arc_lengths[tortuous])) if total_arc > 0 else 0.0
    return {
        "mean_segment_tortuosity": float(np.mean(tortuosities)),
        "median_segment_tortuosity": float(np.median(tortuosities)),
        "p90_segment_tortuosity": float(np.percentile(tortuosities, 90)),
        "p95_segment_tortuosity": float(np.percentile(tortuosities, 95)),
        "tortuous_segment_fraction": float(np.mean(tortuous)),
        "tortuous_length_fraction": float(tortuous_arc / total_arc) if total_arc > 0 else 0.0,
    }


def vessel_widths_on_skeleton(mask: np.ndarray) -> np.ndarray:
    """Local vessel diameter in pixels, sampled on the centerline.

    This returns twice the Euclidean distance transform evaluated at skeleton
    pixels. Empty masks return an empty array.
    """
    vessels = _as_bool_mask(mask)
    if not vessels.any():
        return np.empty(0, dtype=float)
    dist = distance_transform_edt(vessels)
    skel = skeletonize_mask(vessels)
    return np.asarray(2.0 * dist[skel], dtype=float)


def caliber_features(mask: np.ndarray, *, large_vessel_quantile: float = 0.5) -> dict[str, float]:
    """Caliber dispersion, Reagan et al. 2025 arteriole and venule asymmetry.

    The paper found arteriole narrowing with venule widening at 6 months. Those
    are opposite-direction, per-type changes. A single mean averages them toward
    null and is dominated by capillaries. Dispersion of large-vessel widths captures
    that calibers diverged. It cannot attribute direction without artery and vein
    labels. The faithful per-type summary is CRAE, CRVE, and AVR.
    """
    widths = vessel_widths_on_skeleton(mask)
    if widths.size == 0:
        return {
            "mean_vessel_caliber_px": 0.0,
            "caliber_cv": 0.0,
            "caliber_p90_minus_p10_px": 0.0,
            "large_vessel_caliber_cv": 0.0,
        }
    mean_width = float(np.mean(widths))
    std_width = float(np.std(widths))
    quantile = float(np.clip(large_vessel_quantile, 0.0, 1.0))
    large_threshold = float(np.quantile(widths, quantile))
    large_widths = widths[widths >= large_threshold]
    large_mean = float(np.mean(large_widths)) if large_widths.size else 0.0
    large_std = float(np.std(large_widths)) if large_widths.size else 0.0
    return {
        "mean_vessel_caliber_px": mean_width,
        "caliber_cv": float(std_width / mean_width) if mean_width > 0 else 0.0,
        "caliber_p90_minus_p10_px": float(np.percentile(widths, 90) - np.percentile(widths, 10)),
        "large_vessel_caliber_cv": float(large_std / large_mean) if large_mean > 0 else 0.0,
    }


def large_vessel_skeleton(mask: np.ndarray, *, large_vessel_quantile: float = 0.75) -> np.ndarray:
    """Return skeleton pixels with local diameter in the upper vessel-width range.

    This supports the Reagan et al. 2025 late major-branch phenotype. It is a
    label-free proxy for large vessels and is only appropriate for images where
    large retinal vessels are actually present, such as FIVES fundus images.
    """
    vessels = _as_bool_mask(mask)
    skel = skeletonize_mask(vessels)
    if not skel.any():
        return np.zeros_like(skel, dtype=bool)
    dist = distance_transform_edt(vessels)
    diameter = 2.0 * dist
    widths = diameter[skel]
    quantile = float(np.clip(large_vessel_quantile, 0.0, 1.0))
    threshold = float(np.quantile(widths, quantile))
    return skel & (diameter >= threshold)


def major_branch_count(mask: np.ndarray, *, large_vessel_quantile: float = 0.75) -> int:
    """Count branchpoint components on the large-vessel skeleton.

    Reagan et al. 2025 measured a late reduction in major branches near the
    optic nerve head. This is a label-free FIVES proxy, not a capillary-scale
    branchpoint density and not suitable for macula-only ROSE OCTA masks.
    """
    large_skel = large_vessel_skeleton(mask, large_vessel_quantile=large_vessel_quantile)
    if not large_skel.any():
        return 0
    branch_labels = measure.label(branchpoint_mask(large_skel), connectivity=2)
    return int(branch_labels.max())


def candidate_crossing_count(mask: np.ndarray) -> int:
    """Count exploratory 4-way crossing candidates on the skeleton.

    This approximates the Reagan et al. 2025 arteriovenous crossing phenotype by
    counting skeleton nodes with two opposite neighbor pairs. It is not a
    validated artery-over-vein crossing detector and should only be used as a
    FIVES exploratory proxy.
    """
    skel = skeletonize_mask(_as_bool_mask(mask))
    crossing_candidates = _candidate_crossing_mask(skel)
    labels = measure.label(crossing_candidates, connectivity=2)
    return int(labels.max())


def candidate_crossing_density(mask: np.ndarray) -> float:
    """Crossing candidates normalized by skeleton length.

    The denominator makes images with different vessel amounts more comparable,
    but this is still a topology proxy, not validated arteriovenous crossing
    grading.
    """
    skel = skeletonize_mask(_as_bool_mask(mask))
    skeleton_length = int(np.count_nonzero(skel))
    if skeleton_length == 0:
        return 0.0
    crossing_candidates = _candidate_crossing_mask(skel)
    labels = measure.label(crossing_candidates, connectivity=2)
    return float(int(labels.max()) / skeleton_length)


def _candidate_crossing_mask(skel: np.ndarray) -> np.ndarray:
    skel_bool = np.asarray(skel, dtype=bool)
    if not skel_bool.any():
        return np.zeros_like(skel_bool, dtype=bool)
    candidates = skel_bool & (_skeleton_neighbor_count(skel_bool) >= 4)
    out = np.zeros_like(skel_bool, dtype=bool)
    padded = np.pad(skel_bool, 1, constant_values=False)
    for row, col in np.argwhere(candidates):
        prow = int(row) + 1
        pcol = int(col) + 1
        opposite_pairs = 0
        for first, second in _OPPOSITE_NEIGHBOR_PAIRS:
            if padded[prow + first[0], pcol + first[1]] and padded[
                prow + second[0], pcol + second[1]
            ]:
                opposite_pairs += 1
        out[row, col] = opposite_pairs >= 2
    return out


def segment_tortuosity(mask: np.ndarray, *, min_segment_length: int = 5) -> float:
    """Mean arc-to-chord tortuosity over vessel segments between junctions.

    Arc length uses 8-connected Euclidean skeleton step length; chord uses endpoint
    distance when endpoints are available. A straight segment is near 1.0; curved
    segments are greater than 1.0. This is a demo proxy, not a calibrated metric.
    """
    ratios, _arc_lengths = segment_tortuosities(mask, min_segment_length=min_segment_length)
    if ratios.size == 0:
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
    """Return the backward-compatible default vascular feature set.

    The default path is FIVES-like fundus feature extraction because the legacy
    report and modeling demos use this function for large-vessel retinal masks.
    """
    from retivasc.features_fives import extract_fives_features

    return extract_fives_features(mask, fov_mask=fov_mask)
