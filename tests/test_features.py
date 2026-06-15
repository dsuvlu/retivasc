import math

import numpy as np

from retivasc.features import (
    branchpoint_density,
    connected_component_count,
    extract_vascular_features,
    fractal_dimension_boxcount,
    segment_tortuosity,
    skeleton_length_density,
    vessel_density,
)
from retivasc.preprocess import resize_mask_to_max_dim
from retivasc.skeleton import branchpoint_mask, skeletonize_mask


def test_empty_mask_vessel_density_is_zero():
    mask = np.zeros((10, 10), dtype=bool)

    assert vessel_density(mask) == 0.0


def test_full_mask_vessel_density_is_one():
    mask = np.ones((10, 10), dtype=bool)

    assert vessel_density(mask) == 1.0


def test_single_line_skeleton_length_density_positive():
    mask = np.zeros((12, 12), dtype=bool)
    mask[6, 2:10] = True

    assert skeleton_length_density(mask) > 0.0


def test_y_shape_has_branchpoint():
    mask = np.zeros((15, 15), dtype=bool)
    mask[7, 3:12] = True
    for offset in range(5):
        mask[7 - offset, 7 - offset] = True
        mask[7 - offset, 7 + offset] = True

    skel = skeletonize_mask(mask)

    assert branchpoint_mask(skel).sum() >= 1
    assert branchpoint_density(mask) > 0.0


def test_y_shape_counts_one_branchpoint_junction():
    mask = np.zeros((15, 15), dtype=bool)
    mask[7, 3:12] = True
    for offset in range(5):
        mask[7 - offset, 7 - offset] = True
        mask[7 - offset, 7 + offset] = True

    assert branchpoint_density(mask) == 1 / mask.size


def test_connected_component_count_handles_empty_and_disconnected_masks():
    empty = np.zeros((12, 12), dtype=bool)
    blobs = empty.copy()
    blobs[1:3, 1:3] = True
    blobs[8:10, 8:10] = True

    assert connected_component_count(empty) == 0
    assert connected_component_count(blobs) == 2


def test_fractal_dimension_grid_is_finite():
    mask = np.zeros((32, 32), dtype=bool)
    mask[::4, :] = True
    mask[:, ::4] = True

    value = fractal_dimension_boxcount(mask)

    assert math.isfinite(value)
    assert value > 0


def test_fractal_dimension_line_is_less_than_grid():
    line = np.zeros((64, 64), dtype=bool)
    line[32, 8:56] = True
    grid = np.zeros((64, 64), dtype=bool)
    grid[::4, :] = True
    grid[:, ::4] = True

    line_value = fractal_dimension_boxcount(line)
    grid_value = fractal_dimension_boxcount(grid)

    assert 0.8 <= line_value <= 1.2
    assert line_value < grid_value < 2.0


def test_segment_tortuosity_is_higher_for_bent_segment_than_straight_line():
    straight = np.zeros((40, 40), dtype=bool)
    straight[20, 5:35] = True
    bent = np.zeros((40, 40), dtype=bool)
    bent[20, 5:30] = True
    bent[10:21, 30] = True

    straight_value = segment_tortuosity(straight)
    bent_value = segment_tortuosity(bent)

    assert 1.0 <= straight_value <= 1.2
    assert bent_value > straight_value + 0.2


def test_extract_vascular_features_returns_required_keys():
    mask = np.zeros((12, 12), dtype=bool)
    mask[3:9, 6] = True

    features = extract_vascular_features(mask)

    assert {
        "vessel_density",
        "skeleton_length_density",
        "branchpoint_density",
        "fractal_dimension_boxcount",
        "mean_segment_tortuosity",
        "connected_component_count",
    } <= set(features)


def test_resize_mask_to_max_dim_preserves_binary_mask():
    mask = np.zeros((40, 20), dtype=bool)
    mask[10:30, 8:12] = True

    resized = resize_mask_to_max_dim(mask, 10)

    assert resized.dtype == bool
    assert resized.shape == (10, 5)
    assert resized.any()
