import numpy as np

from retivasc.embeddings import compute_mask_embedding_features
from retivasc.features import caliber_features, vessel_widths_on_skeleton


def test_vessel_widths_on_skeleton_matches_known_bar_width():
    mask = np.zeros((64, 64), dtype=bool)
    mask[8:56, 20:28] = True

    widths = vessel_widths_on_skeleton(mask)
    features = caliber_features(mask)

    assert widths.size > 0
    assert abs(features["mean_vessel_caliber_px"] - 8.0) <= 1.0


def test_caliber_cv_increases_with_width_divergence():
    equal = np.zeros((64, 64), dtype=bool)
    equal[8:56, 12:18] = True
    equal[8:56, 40:46] = True

    divergent = np.zeros((64, 64), dtype=bool)
    divergent[8:56, 12:16] = True
    divergent[8:56, 36:48] = True

    equal_features = caliber_features(equal)
    divergent_features = caliber_features(divergent)

    assert divergent_features["caliber_cv"] > equal_features["caliber_cv"]
    assert (
        divergent_features["caliber_p90_minus_p10_px"] > equal_features["caliber_p90_minus_p10_px"]
    )


def test_empty_mask_caliber_features_return_zeros():
    features = caliber_features(np.zeros((24, 24), dtype=bool))

    assert features == {
        "mean_vessel_caliber_px": 0.0,
        "caliber_cv": 0.0,
        "caliber_p90_minus_p10_px": 0.0,
        "large_vessel_caliber_cv": 0.0,
    }


def test_rose_embedding_caliber_proxy_uses_doubled_skeleton_widths():
    mask = np.zeros((64, 64), dtype=bool)
    mask[8:56, 20:28] = True

    features = compute_mask_embedding_features(mask)

    assert abs(features["caliber_proxy_mean_px"] - 8.0) <= 1.0
