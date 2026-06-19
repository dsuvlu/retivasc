import numpy as np
import pandas as pd
import pytest

from retivasc.stats import (
    bootstrap_difference,
    compare_groups_featurewise,
    fdr_bh,
    permutation_test_difference,
    validate_rose_feature_table,
)


def _feature_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": ["s1", "s2", "s3", "s4"],
            "image_id": ["i1", "i2", "i3", "i4"],
            "layer": ["SVC", "SVC", "DVC", "DVC"],
            "diagnosis": ["AD", "control", "AD", "control"],
            "label_source": ["manifest"] * 4,
            "mask_path": ["a.png", "b.png", "c.png", "d.png"],
            "vessel_area_fraction": [0.4, 0.2, 0.5, 0.25],
            "branchpoint_density": [0.1, 0.08, 0.11, 0.07],
        }
    )


def test_validate_rose_feature_table_rejects_missing_required_columns():
    table = _feature_table().drop(columns=["mask_path"])

    with pytest.raises(ValueError, match="Missing required"):
        validate_rose_feature_table(table)


def test_validate_rose_feature_table_rejects_conflicting_subject_labels():
    table = _feature_table()
    table.loc[2, "subject_id"] = "s1"
    table.loc[2, "diagnosis"] = "control"

    with pytest.raises(ValueError, match="conflicting diagnosis"):
        validate_rose_feature_table(table)


def test_bootstrap_difference_is_deterministic():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 1.5, 2.0])

    first = bootstrap_difference(x, y, n_boot=100, random_state=7)
    second = bootstrap_difference(x, y, n_boot=100, random_state=7)

    assert first == second


def test_bootstrap_difference_detects_separated_groups():
    low, high = bootstrap_difference(
        np.array([10.0, 11.0, 12.0]),
        np.array([1.0, 2.0, 3.0]),
        n_boot=200,
        random_state=0,
    )

    assert low > 0
    assert high > low


def test_permutation_test_difference_identical_groups_high_p_value():
    p_value = permutation_test_difference(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
        n_perm=200,
        random_state=0,
    )

    assert p_value > 0.5


def test_permutation_test_difference_separated_groups_low_p_value():
    p_value = permutation_test_difference(
        np.arange(10.0, 20.0),
        np.arange(0.0, 10.0),
        statistic="mean",
        n_perm=500,
        random_state=0,
    )

    assert p_value < 0.05


def test_fdr_bh_values_between_zero_and_one():
    adjusted = fdr_bh([0.01, 0.2, np.nan, 0.03])

    assert np.isnan(adjusted[2])
    assert np.nanmin(adjusted) >= 0
    assert np.nanmax(adjusted) <= 1


def test_compare_groups_featurewise_returns_effect_columns():
    table = pd.DataFrame(
        {
            "diagnosis": ["AD"] * 4 + ["control"] * 4,
            "feature_a": [5, 6, 7, 8, 1, 2, 3, 4],
        }
    )

    effects = compare_groups_featurewise(
        table,
        ["feature_a"],
        n_boot=100,
        n_perm=100,
        random_state=0,
    )

    assert effects.loc[0, "diff_median_AD_minus_control"] > 0
    assert 0 <= effects.loc[0, "fdr_bh_permutation_p"] <= 1
