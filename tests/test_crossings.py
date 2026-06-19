import numpy as np

from retivasc.features import (
    candidate_crossing_count,
    candidate_crossing_density,
    large_vessel_skeleton,
    major_branch_count,
)


def test_x_shape_counts_one_crossing_candidate():
    mask = np.zeros((25, 25), dtype=bool)
    center = 12
    for offset in range(-7, 8):
        mask[center + offset, center + offset] = True
        mask[center + offset, center - offset] = True

    assert candidate_crossing_count(mask) == 1
    assert candidate_crossing_density(mask) > 0.0


def test_y_shape_is_not_crossing_candidate():
    mask = np.zeros((25, 25), dtype=bool)
    center = 12
    mask[center, 5:20] = True
    for offset in range(0, 8):
        mask[center - offset, center - offset] = True
        mask[center - offset, center + offset] = True

    assert candidate_crossing_count(mask) == 0
    assert candidate_crossing_density(mask) == 0.0


def test_empty_mask_crossing_density_is_zero():
    mask = np.zeros((16, 16), dtype=bool)

    assert candidate_crossing_count(mask) == 0
    assert candidate_crossing_density(mask) == 0.0


def test_large_vessel_skeleton_excludes_thin_spur():
    mask = np.zeros((64, 64), dtype=bool)
    mask[8:56, 20:28] = True
    mask[8:56, 45:47] = True

    large = large_vessel_skeleton(mask)
    thin_region = np.zeros_like(mask)
    thin_region[8:56, 45:47] = True

    assert large.any()
    assert np.count_nonzero(large & thin_region) == 0


def test_major_branch_count_uses_large_vessel_network():
    mask = np.zeros((64, 64), dtype=bool)
    mask[28:36, 8:56] = True
    mask[8:56, 28:36] = True

    assert major_branch_count(mask) >= 1
