from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from skimage import io as skio

from retivasc.embeddings import (
    MASK_EMBEDDING_FEATURES,
    build_mask_feature_table,
    compute_group_separation,
    compute_mask_embedding_features,
    fit_pca_embedding,
    fit_tsne_embedding,
    fit_umap_embedding,
    load_binary_mask,
    normalize_diagnosis,
    prepare_embedding_matrix,
)


def _write_mask(path: Path, *, offset: int = 0) -> Path:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[16, 4 + offset : 28] = 255
    mask[8:24, 12 + offset] = 255
    skio.imsave(path, mask, check_contrast=False)
    return path


def _manifest(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for subject, diagnosis, offset in (("s1", "AD", 0), ("s2", "control", 2)):
        for layer in ("SVC", "DVC", "SVC+DVC"):
            mask_path = _write_mask(
                tmp_path / f"{subject}_{layer.replace('+', '')}.png",
                offset=offset,
            )
            rows.append(
                {
                    "subject_id": subject,
                    "image_id": f"{subject}_{layer}",
                    "layer": layer,
                    "mask_path": str(mask_path),
                    "diagnosis": diagnosis,
                    "label_source": "official_metadata",
                    "split_group": subject,
                }
            )
    return pd.DataFrame(rows)


def test_load_binary_mask_reads_nonzero_pixels(tmp_path):
    mask_path = _write_mask(tmp_path / "mask.png")

    mask = load_binary_mask(mask_path)

    assert mask.dtype == bool
    assert mask.shape == (32, 32)
    assert mask.any()


def test_build_mask_feature_table_one_row_per_subject_layer(tmp_path):
    manifest = _manifest(tmp_path)

    table = build_mask_feature_table(manifest)

    assert len(table) == 6
    assert table[["subject_id", "layer"]].drop_duplicates().shape[0] == 6
    assert set(MASK_EMBEDDING_FEATURES) <= set(table.columns)
    assert set(table["diagnosis"]) == {"AD", "control"}


def test_compute_mask_embedding_features_preserves_public_column_set():
    mask = np.zeros((32, 32), dtype=bool)
    mask[16, 4:28] = True
    mask[8:24, 12] = True

    features = compute_mask_embedding_features(mask)

    assert list(features) == MASK_EMBEDDING_FEATURES


def test_build_mask_feature_table_rejects_missing_explicit_label_source(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.loc[0, "label_source"] = "unknown"

    with pytest.raises(ValueError, match="explicit label_source"):
        build_mask_feature_table(manifest)


def test_unknown_labels_allowed(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["diagnosis"] = "unknown"
    manifest["label_source"] = "unknown"

    table = build_mask_feature_table(manifest)

    assert set(table["diagnosis"]) == {"unknown"}


def test_prepare_embedding_matrix_imputes_nan_and_scales_finite(tmp_path):
    table = build_mask_feature_table(_manifest(tmp_path))
    table.loc[0, MASK_EMBEDDING_FEATURES[0]] = np.nan

    matrix, metadata = prepare_embedding_matrix(table, MASK_EMBEDDING_FEATURES)

    assert matrix.shape == (6, len(MASK_EMBEDDING_FEATURES))
    assert np.isfinite(matrix).all()
    assert metadata["n_missing_values_imputed"] == 1


def test_fit_pca_embedding_returns_expected_columns(tmp_path):
    table = build_mask_feature_table(_manifest(tmp_path))
    matrix, _ = prepare_embedding_matrix(table, MASK_EMBEDDING_FEATURES)

    coords, metadata = fit_pca_embedding(matrix, MASK_EMBEDDING_FEATURES)

    assert {"component_1", "component_2"} <= set(coords.columns)
    assert metadata["explained_variance_ratio_pc1"] is not None
    assert metadata["top_pc1_loadings"]


def test_tsne_perplexity_less_than_n_samples(tmp_path):
    table = build_mask_feature_table(_manifest(tmp_path))
    matrix, _ = prepare_embedding_matrix(table, MASK_EMBEDDING_FEATURES)

    coords, metadata = fit_tsne_embedding(matrix, perplexity=100)

    assert len(coords) == len(table)
    assert metadata["perplexity"] < len(table)


def test_umap_optional_import_error_is_informative(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "umap":
            raise ModuleNotFoundError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match="umap-learn"):
        fit_umap_embedding(np.zeros((5, 2)))


def test_compute_group_separation_reports_centroid_distance():
    coords = pd.DataFrame(
        {
            "component_1": [0.0, 0.2, 2.0, 2.2],
            "component_2": [0.0, 0.1, 2.0, 2.1],
            "diagnosis": ["AD", "AD", "control", "control"],
        }
    )

    summary = compute_group_separation(coords, n_permutations=20, random_state=0)

    assert summary["centroid_distance"] > 0
    assert 0 <= summary["p_value"] <= 1


def test_normalize_diagnosis_does_not_infer_from_filename_like_values():
    assert normalize_diagnosis("subject_01_AD_like_filename") == "unknown"
