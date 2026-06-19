"""FIVES fundus feature extraction.

This module is the large-vessel, optic-disc-compatible feature path. It can host
paper-aligned Reagan et al. 2025 proxies for tortuosity burden, caliber
dispersion, crossing candidates, and late major-branch simplification. It still
returns the legacy density and topology keys used by existing demos.
"""

from __future__ import annotations

import numpy as np

from retivasc.features import (
    _as_bool_mask,
    _fov,
    branchpoint_density,
    caliber_features,
    candidate_crossing_count,
    candidate_crossing_density,
    connected_component_count,
    fractal_dimension_boxcount,
    major_branch_count,
    skeleton_length_density,
    tortuosity_burden,
    vessel_density,
)


def extract_fives_features(
    mask: np.ndarray,
    *,
    fov_mask: np.ndarray | None = None,
    pixel_size_um: float | None = None,
) -> dict[str, float]:
    """Return paper-aligned features for FIVES-style fundus vessel masks.

    Early phenotypes lead the dictionary: tortuosity burden, then label-free
    caliber dispersion. Crossing candidates and major branches are exploratory
    large-vessel proxies. Density and component features are kept as late/context
    summaries and for backward compatibility. Caliber values are in pixels unless
    a later calibrated path explicitly converts them with pixel_size_um.
    """
    vessels = _analysis_mask(mask, fov_mask)
    tortuosity = tortuosity_burden(vessels)
    caliber = caliber_features(vessels)

    _ = pixel_size_um
    return {
        "tortuous_segment_fraction": tortuosity["tortuous_segment_fraction"],
        "tortuous_length_fraction": tortuosity["tortuous_length_fraction"],
        "mean_segment_tortuosity": tortuosity["mean_segment_tortuosity"],
        "median_segment_tortuosity": tortuosity["median_segment_tortuosity"],
        "p90_segment_tortuosity": tortuosity["p90_segment_tortuosity"],
        "p95_segment_tortuosity": tortuosity["p95_segment_tortuosity"],
        "caliber_cv": caliber["caliber_cv"],
        "large_vessel_caliber_cv": caliber["large_vessel_caliber_cv"],
        "caliber_p90_minus_p10_px": caliber["caliber_p90_minus_p10_px"],
        "mean_vessel_caliber_px": caliber["mean_vessel_caliber_px"],
        "candidate_crossing_density": candidate_crossing_density(vessels),
        "candidate_crossing_count": float(candidate_crossing_count(vessels)),
        "major_branch_count": float(major_branch_count(vessels)),
        "fractal_dimension_boxcount": fractal_dimension_boxcount(vessels),
        "vessel_density": vessel_density(vessels, fov_mask),
        "skeleton_length_density": skeleton_length_density(vessels, fov_mask),
        "branchpoint_density": branchpoint_density(vessels, fov_mask),
        "connected_component_count": float(connected_component_count(vessels)),
    }


def _analysis_mask(mask: np.ndarray, fov_mask: np.ndarray | None) -> np.ndarray:
    vessels = _as_bool_mask(mask)
    fov = _fov(vessels, fov_mask)
    return vessels & fov
