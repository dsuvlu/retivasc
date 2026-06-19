import pandas as pd

from retivasc.outliers import compute_outlier_scores, compute_outlier_sensitivity


def test_outlier_scoring_flags_extreme_synthetic_row():
    table = pd.DataFrame(
        {
            "subject_id": ["s1", "s2", "s3", "s4", "s5"],
            "image_id": ["i1", "i2", "i3", "i4", "i5"],
            "layer": ["SVC"] * 5,
            "diagnosis": ["AD", "AD", "control", "control", "AD"],
            "mask_path": ["m.png"] * 5,
            "feature_a": [1.0, 1.1, 0.9, 1.2, 10.0],
            "feature_b": [2.0, 2.1, 1.9, 2.2, 20.0],
        }
    )

    scores = compute_outlier_scores(table, ["feature_a", "feature_b"])

    extreme = scores.loc[scores["subject_id"] == "s5"].iloc[0]
    assert bool(extreme["is_feature_outlier"])
    assert extreme["max_abs_zscore"] > 3


def test_outlier_sensitivity_labels_sign_flip():
    full = pd.DataFrame(
        {
            "analysis": ["layer_specific"],
            "layer": ["SVC"],
            "feature": ["feature_a"],
            "diff_median_AD_minus_control": [1.0],
            "permutation_p": [0.1],
        }
    )
    filtered = pd.DataFrame(
        {
            "analysis": ["layer_specific"],
            "layer": ["SVC"],
            "feature": ["feature_a"],
            "diff_median_AD_minus_control": [-0.2],
            "permutation_p": [0.8],
        }
    )

    sensitivity = compute_outlier_sensitivity(full, filtered)

    assert sensitivity.loc[0, "interpretation_flag"] == "outlier_sensitive"
