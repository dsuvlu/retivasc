import pandas as pd

from retivasc.plots_effects import (
    plot_layer_contrast_effect_sizes,
    plot_layer_effect_sizes,
    plot_subject_level_pca,
)


def test_subject_level_pca_runs_and_saves_figure(tmp_path):
    subject_df = pd.DataFrame(
        {
            "subject_id": ["s1", "s2", "s3", "s4"],
            "diagnosis": ["AD", "AD", "control", "control"],
            "feature_a": [1.0, 1.2, 2.0, 2.2],
            "feature_b": [0.5, 0.7, 1.4, 1.5],
            "is_feature_outlier": [False, False, False, True],
        }
    )

    fig, metadata = plot_subject_level_pca(
        subject_df,
        ["feature_a", "feature_b"],
        tmp_path / "subject_pca.png",
    )

    assert len(fig.axes) == 1
    assert metadata["explained_variance_ratio_pc1"] is not None
    assert (tmp_path / "subject_pca.png").exists()


def test_layer_effect_size_plot_writes_file(tmp_path):
    effects = pd.DataFrame(
        {
            "layer": ["SVC", "DVC", "SVC+DVC"],
            "feature": ["vessel_area_fraction"] * 3,
            "diff_median_AD_minus_control": [0.1, -0.2, 0.05],
            "bootstrap_CI_low": [0.0, -0.4, -0.05],
            "bootstrap_CI_high": [0.2, 0.0, 0.15],
            "hedges_g": [0.5, -0.7, 0.2],
            "fdr_bh_permutation_p": [0.2, 0.08, 0.5],
        }
    )

    fig = plot_layer_effect_sizes(effects, tmp_path / "layer_effects.png")

    assert len(fig.axes) == 3
    assert (tmp_path / "layer_effects.png").exists()


def test_layer_contrast_effect_size_plot_writes_file(tmp_path):
    effects = pd.DataFrame(
        {
            "feature": ["DVC_minus_SVC_vessel_area_fraction", "DVC_div_SVC_branchpoint_density"],
            "diff_median_AD_minus_control": [0.1, -0.2],
            "bootstrap_CI_low": [0.0, -0.4],
            "bootstrap_CI_high": [0.2, 0.0],
            "hedges_g": [0.5, -0.7],
            "fdr_bh_permutation_p": [0.2, 0.08],
        }
    )

    fig = plot_layer_contrast_effect_sizes(effects, tmp_path / "contrast_effects.png")

    assert len(fig.axes) == 1
    assert (tmp_path / "contrast_effects.png").exists()
