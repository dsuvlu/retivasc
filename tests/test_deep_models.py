from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from skimage import io as skio

from retivasc.deep_models import (
    DeepSegmenterConfig,
    select_rose_deep_demo_rows,
    torch_available,
    train_predict_deep_segmenters,
)


def _write_case(root: Path, stem: str) -> tuple[Path, Path]:
    image = np.zeros((24, 24), dtype=np.uint8)
    image[12, 4:20] = 255
    image[6:18, 10] = 180
    mask = image > 0
    image_path = root / f"{stem}.png"
    mask_path = root / f"{stem}_mask.png"
    skio.imsave(image_path, image, check_contrast=False)
    skio.imsave(mask_path, mask.astype(np.uint8) * 255, check_contrast=False)
    return image_path, mask_path


def test_select_rose_deep_demo_rows_prefers_rose1_svc_balanced_split(tmp_path):
    image_a, mask_a = _write_case(tmp_path, "train_disease")
    image_b, mask_b = _write_case(tmp_path, "train_control")
    image_c, mask_c = _write_case(tmp_path, "test_disease")
    image_d, mask_d = _write_case(tmp_path, "test_control")
    manifest = pd.DataFrame(
        {
            "dataset": ["ROSE-1"] * 4,
            "image_id": ["a", "b", "c", "d"],
            "subject_id": ["sa", "sb", "sc", "sd"],
            "layer": ["SVC"] * 4,
            "label": ["disease", "control", "disease", "control"],
            "official_split": ["train", "train", "test", "test"],
            "image_path": [str(image_a), str(image_b), str(image_c), str(image_d)],
            "mask_path": [str(mask_a), str(mask_b), str(mask_c), str(mask_d)],
        }
    )

    train_rows, eval_rows = select_rose_deep_demo_rows(manifest, train_max_rows=2, eval_max_rows=2)

    assert set(train_rows["official_split"]) == {"train"}
    assert set(eval_rows["official_split"]) == {"test"}
    assert set(train_rows["label"]) == {"disease", "control"}
    assert set(eval_rows["label"]) == {"disease", "control"}


def test_train_predict_deep_segmenters_requires_or_uses_torch(tmp_path):
    image_a, mask_a = _write_case(tmp_path, "case_a")
    image_b, mask_b = _write_case(tmp_path, "case_b")
    train_manifest = pd.DataFrame({"image_path": [str(image_a)], "mask_path": [str(mask_a)]})
    eval_manifest = pd.DataFrame(
        {
            "image_id": ["case_b"],
            "image_path": [str(image_b)],
            "mask_path": [str(mask_b)],
        }
    )

    if not torch_available():
        with pytest.raises(RuntimeError, match="PyTorch"):
            train_predict_deep_segmenters(train_manifest, eval_manifest, tmp_path / "out")
        return

    predictions, history = train_predict_deep_segmenters(
        train_manifest,
        eval_manifest,
        tmp_path / "out",
        methods=("unet_lite",),
        config=DeepSegmenterConfig(image_size=32, epochs=1, batch_size=1, base_channels=2),
    )

    assert "unet_lite_prediction_path" in predictions.columns
    assert Path(predictions.loc[0, "unet_lite_prediction_path"]).exists()
    assert history.loc[0, "method"] == "unet_lite"
