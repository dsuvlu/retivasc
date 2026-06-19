import pandas as pd

from retivasc.plots_embeddings import plot_layer_faceted_embedding, plot_pca_feature_loadings


def test_plot_layer_faceted_embedding_has_three_axes(tmp_path):
    coords = pd.DataFrame(
        {
            "component_1": [0.0, 1.0, 0.2],
            "component_2": [0.5, 1.5, 0.3],
            "layer": ["SVC", "DVC", "SVC+DVC"],
            "diagnosis": ["AD", "control", "unknown"],
        }
    )

    fig = plot_layer_faceted_embedding(
        coords,
        "component_1",
        "component_2",
        out_path=tmp_path / "embedding.png",
    )

    assert len(fig.axes) == 3
    assert (tmp_path / "embedding.png").exists()


def test_plot_does_not_fail_with_unknown_labels():
    coords = pd.DataFrame(
        {
            "component_1": [0.0],
            "component_2": [0.5],
            "layer": ["SVC"],
            "diagnosis": ["unknown"],
        }
    )

    fig = plot_layer_faceted_embedding(coords, "component_1", "component_2")

    assert len(fig.axes) == 3


def test_plot_pca_feature_loadings_writes_file(tmp_path):
    metadata = {
        "loadings": [
            {"feature": "vessel_area_fraction", "PC1": 0.5, "PC2": -0.2},
            {"feature": "branchpoint_density", "PC1": -0.3, "PC2": 0.7},
            {"feature": "orientation_entropy", "PC1": 0.1, "PC2": 0.2},
        ]
    }

    fig = plot_pca_feature_loadings(metadata, tmp_path / "loadings.png")

    assert len(fig.axes) == 2
    assert (tmp_path / "loadings.png").exists()
