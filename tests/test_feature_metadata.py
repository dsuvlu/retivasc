import numpy as np

from retivasc.feature_metadata import FEATURE_METADATA, metadata_for_feature
from retivasc.features import caliber_features, extract_vascular_features, tortuosity_burden
from retivasc.features_fives import extract_fives_features
from retivasc.features_rose import extract_rose_features


def test_current_feature_outputs_have_metadata():
    mask = np.zeros((32, 32), dtype=bool)
    mask[16, 4:28] = True
    mask[8:24, 12] = True

    keys = set(extract_vascular_features(mask))
    keys.update(extract_fives_features(mask))
    keys.update(extract_rose_features(mask))
    keys.update(tortuosity_burden(mask))
    keys.update(caliber_features(mask))

    for key in keys:
        metadata = metadata_for_feature(key)
        assert metadata["timing"]
        assert metadata["source"]
        assert metadata["datasets"]


def test_paper_aligned_metadata_tags():
    assert FEATURE_METADATA["vessel_density"]["timing"] == "late"
    assert FEATURE_METADATA["tortuous_segment_fraction"]["timing"] == "early"
    assert FEATURE_METADATA["caliber_cv"]["timing"] == "early"
    assert FEATURE_METADATA["fractal_dimension_boxcount"]["timing"] == "context"
    assert "Reagan" not in str(FEATURE_METADATA["fractal_dimension_boxcount"]["source"])
    assert FEATURE_METADATA["dropout_heterogeneity"]["timing"] == "interpretive"


def test_metadata_for_derived_layer_feature_uses_base_feature():
    metadata = metadata_for_feature("DVC_minus_SVC_tortuous_segment_fraction")

    assert metadata["timing"] == "early"
