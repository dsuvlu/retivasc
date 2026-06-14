import numpy as np
from skimage import io as skio

from retivasc.io import load_fives_manifest


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
