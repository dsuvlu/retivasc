import numpy as np

from retivasc.features import extract_vascular_features
from retivasc.features_fives import extract_fives_features


def test_fives_features_lead_with_paper_aligned_early_features():
    mask = np.zeros((64, 64), dtype=bool)
    mask[32, 8:56] = True
    mask[16:48, 40] = True

    features = extract_fives_features(mask)
    keys = list(features)

    assert keys[0] == "tortuous_segment_fraction"
    assert keys.index("caliber_cv") < keys.index("vessel_density")
    assert keys.index("vessel_density") > keys.index("tortuous_segment_fraction")


def test_fives_features_include_legacy_and_large_vessel_keys():
    mask = np.zeros((64, 64), dtype=bool)
    mask[28:36, 8:56] = True
    mask[8:56, 28:36] = True

    features = extract_fives_features(mask)

    assert {
        "vessel_density",
        "skeleton_length_density",
        "branchpoint_density",
        "fractal_dimension_boxcount",
        "mean_segment_tortuosity",
        "connected_component_count",
        "candidate_crossing_density",
        "candidate_crossing_count",
        "major_branch_count",
        "caliber_cv",
    } <= set(features)


def test_extract_vascular_features_remains_backward_compatible():
    mask = np.zeros((24, 24), dtype=bool)
    mask[12, 4:20] = True

    features = extract_vascular_features(mask)

    assert {
        "vessel_density",
        "skeleton_length_density",
        "branchpoint_density",
        "fractal_dimension_boxcount",
        "mean_segment_tortuosity",
        "connected_component_count",
    } <= set(features)
