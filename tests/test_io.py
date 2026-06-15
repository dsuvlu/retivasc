import numpy as np
import pytest
from skimage import io as skio

from retivasc.io import load_fives_manifest, load_rose_manifest


def _write_png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    skio.imsave(path, array, check_contrast=False)


def test_load_fives_manifest_detects_official_layout(tmp_path):
    root = tmp_path / "fives"
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 3] = 255

    for split, filenames in {
        "train": ["1_A.png", "2_N.png"],
        "test": ["1_D.png", "2_G.png"],
    }.items():
        for filename in filenames:
            _write_png(root / split / "Original" / filename, rgb)
            _write_png(root / split / "Ground Truth" / filename, mask)

    manifest = load_fives_manifest(root)

    assert len(manifest) == 4
    assert set(manifest["official_split"]) == {"train", "test"}
    assert set(manifest["label"]) == {"AMD", "DR", "glaucoma", "normal"}
    assert manifest["image_id"].is_unique
    assert manifest["image_path"].str.contains("Original").all()
    assert manifest["mask_path"].str.contains("Ground Truth").all()


def test_load_rose_manifest_detects_official_rose1_layout(tmp_path):
    root = tmp_path / "rose"
    image = np.zeros((8, 8), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 3] = 255

    for layer in ("SVC", "DVC", "SVC_DVC"):
        _write_png(root / "ROSE" / "ROSE-1" / layer / "train" / "img" / "01.tif", image)
        _write_png(root / "ROSE" / "ROSE-1" / layer / "train" / "gt" / "01.tif", mask)

    manifest = load_rose_manifest(root)

    assert len(manifest) == 3
    assert set(manifest["dataset"]) == {"ROSE-1"}
    assert set(manifest["layer"]) == {"SVC", "DVC", "SVC+DVC"}
    assert manifest["label"].isna().all()
    assert manifest["subject_id"].nunique() == 1
    assert manifest["split_group"].nunique() == 1
    assert set(manifest["split_group"]) == set(manifest["subject_id"])


def test_load_rose_manifest_detects_official_rose2_layout(tmp_path):
    root = tmp_path / "rose"
    image = np.zeros((8, 8), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 3] = 255

    _write_png(root / "ROSE" / "ROSE-2" / "test" / "original" / "11_OD_SVP.png", image)
    _write_png(root / "ROSE" / "ROSE-2" / "test" / "gt" / "11_OD_SVP.png", mask)

    manifest = load_rose_manifest(root)

    assert len(manifest) == 1
    assert manifest.loc[0, "dataset"] == "ROSE-2"
    assert manifest.loc[0, "subject_id"] == "ROSE-2_11"
    assert manifest.loc[0, "split_group"] == "ROSE-2_11"
    assert manifest.loc[0, "layer"] == "SVP"
    assert manifest.loc[0, "label"] is None


def test_load_rose_manifest_rejects_official_cross_split_subject_collision(tmp_path):
    root = tmp_path / "rose"
    image = np.zeros((8, 8), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 3] = 255

    for split in ("train", "test"):
        _write_png(root / "ROSE" / "ROSE-1" / "SVC" / split / "img" / "01.tif", image)
        _write_png(root / "ROSE" / "ROSE-1" / "SVC" / split / "gt" / "01.tif", mask)

    with pytest.raises(ValueError, match="multiple official splits"):
        load_rose_manifest(root)
