import numpy as np

from retivasc.features import (
    DEFAULT_TORTUOSITY_THRESHOLD,
    segment_tortuosities,
    tortuosity_burden,
)


def test_straight_bars_have_zero_tortuosity_burden():
    mask = np.zeros((48, 48), dtype=bool)
    mask[12, 5:42] = True
    mask[32, 5:42] = True

    features = tortuosity_burden(mask)

    assert features["tortuous_segment_fraction"] == 0.0
    assert features["tortuous_length_fraction"] == 0.0
    assert 1.0 <= features["mean_segment_tortuosity"] <= 1.05


def test_mixed_straight_and_bent_segments_report_threshold_fraction():
    mask = np.zeros((64, 64), dtype=bool)
    mask[12, 5:45] = True
    mask[45, 5:35] = True
    mask[25:46, 35] = True

    tortuosities, arc_lengths = segment_tortuosities(mask)
    features = tortuosity_burden(mask, threshold=DEFAULT_TORTUOSITY_THRESHOLD)

    expected_segment_fraction = float(np.mean(tortuosities >= DEFAULT_TORTUOSITY_THRESHOLD))
    expected_length_fraction = float(
        arc_lengths[tortuosities >= DEFAULT_TORTUOSITY_THRESHOLD].sum() / arc_lengths.sum()
    )

    assert tortuosities.size == 2
    assert 0 < features["tortuous_segment_fraction"] <= 1
    assert features["tortuous_segment_fraction"] == expected_segment_fraction
    assert features["tortuous_length_fraction"] == expected_length_fraction


def test_empty_mask_tortuosity_burden_returns_zeros():
    features = tortuosity_burden(np.zeros((24, 24), dtype=bool))

    assert set(features) == {
        "mean_segment_tortuosity",
        "median_segment_tortuosity",
        "p90_segment_tortuosity",
        "p95_segment_tortuosity",
        "tortuous_segment_fraction",
        "tortuous_length_fraction",
    }
    assert all(value == 0.0 for value in features.values())
