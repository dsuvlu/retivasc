import numpy as np

from retivasc.embeddings import MASK_EMBEDDING_FEATURES, compute_mask_embedding_features
from retivasc.features_rose import ROSE_MACULA_OCTA_CAVEAT, extract_rose_features


def test_rose_features_exclude_fives_only_caliber_and_crossing_proxies():
    mask = np.zeros((48, 48), dtype=bool)
    mask[24, 8:40] = True
    mask[12:36, 20] = True

    features = extract_rose_features(mask)

    assert "caliber_cv" not in features
    assert "candidate_crossing_density" not in features
    assert "major_branch_count" not in features
    assert "tortuous_segment_fraction" in features
    assert "dropout_heterogeneity" in features
    assert "optic disc" in ROSE_MACULA_OCTA_CAVEAT


def test_rose_features_preserve_embedding_columns():
    mask = np.zeros((48, 48), dtype=bool)
    mask[24, 8:40] = True
    mask[12:36, 20] = True

    rose_features = extract_rose_features(mask)
    embedding_features = compute_mask_embedding_features(mask)

    assert set(MASK_EMBEDDING_FEATURES) <= set(rose_features)
    assert list(embedding_features) == MASK_EMBEDDING_FEATURES
    assert set(embedding_features) == set(MASK_EMBEDDING_FEATURES)
    assert rose_features["high_tortuosity_fraction"] == rose_features[
        "tortuous_segment_fraction"
    ]
