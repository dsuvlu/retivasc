import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from skimage import io as skio

from retivasc.evaluation import (
    benchmark_native_segmenters,
    benchmark_prediction_columns,
    compare_mask_features,
    compare_segmentation_masks,
    evaluate_score_postprocessing,
    postprocess_score_mask,
    tune_native_segmenters,
    tune_score_postprocessing,
)


def _mask():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 12] = True
    mask[16, 12:24] = True
    return mask


def test_compare_mask_features_identical_masks_zero_error():
    mask = _mask()

    comparison = compare_mask_features(mask, mask)

    error_keys = [key for key in comparison if key.startswith("abs_error_")]
    assert error_keys
    assert all(comparison[key] == 0.0 for key in error_keys)


def test_compare_mask_features_empty_prediction_has_density_error():
    manual = _mask()
    predicted = np.zeros_like(manual)

    comparison = compare_mask_features(manual, predicted)

    assert comparison["manual_vessel_density"] > 0
    assert comparison["predicted_vessel_density"] == 0.0
    assert comparison["abs_error_vessel_density"] > 0
    assert math.isfinite(comparison["rel_error_vessel_density"])


def test_compare_segmentation_masks_returns_expected_keys():
    manual = _mask()
    predicted = manual.copy()
    predicted[16, 20:24] = False

    metrics = compare_segmentation_masks(manual, predicted)

    assert set(metrics) == {"dice", "iou", "precision", "recall", "specificity"}
    assert metrics["dice"] < 1.0
    assert metrics["precision"] == 1.0


def test_compare_segmentation_masks_uses_fov_for_specificity():
    manual = np.array([[1, 0, 0, 0]], dtype=bool)
    predicted = np.array([[1, 1, 0, 0]], dtype=bool)
    fov = np.array([[1, 1, 0, 0]], dtype=bool)

    metrics = compare_segmentation_masks(manual, predicted, fov_mask=fov)

    assert metrics["specificity"] == 0.0


def test_benchmark_native_segmenters_writes_outputs(tmp_path):
    image = np.zeros((32, 32), dtype=np.uint8)
    image[16, 6:26] = 255
    image[8:24, 12] = 220
    mask = image > 0
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    skio.imsave(image_path, image, check_contrast=False)
    skio.imsave(mask_path, mask.astype(np.uint8) * 255, check_contrast=False)
    manifest = pd.DataFrame(
        {
            "dataset": ["fixture"],
            "image_id": ["case_001"],
            "subject_id": ["subject_001"],
            "layer": ["SVC"],
            "official_split": ["test"],
            "image_path": [str(image_path)],
            "mask_path": [str(mask_path)],
        }
    )

    table = benchmark_native_segmenters(
        manifest,
        methods=["frangi", "diffusion"],
        output_root=tmp_path / "benchmark",
        method_params={
            "frangi": {"threshold": "otsu", "min_size": 2},
            "diffusion": {"n_iter": 1, "threshold": "otsu", "min_size": 2, "clahe": False},
        },
    )

    assert len(table) == 2
    assert set(table["method"]) == {"frangi", "diffusion_threshold"}
    assert table["error"].isna().all()
    assert {"dice", "iou", "abs_error_vessel_density"} <= set(table.columns)
    assert all(pd.notna(path) and Path(path).exists() for path in table["pred_mask_path"])
    assert all(pd.notna(path) and Path(path).exists() for path in table["score_path"])


def test_benchmark_native_segmenters_rejects_unknown_method():
    manifest = pd.DataFrame({"image_path": ["image.png"], "mask_path": ["mask.png"]})

    with pytest.raises(ValueError, match="Unknown native segmentation"):
        benchmark_native_segmenters(manifest, methods=["missing"])


def test_benchmark_prediction_columns_scores_external_masks(tmp_path):
    image = np.zeros((24, 24), dtype=np.uint8)
    image[12, 4:20] = 255
    manual = image > 0
    pred = manual.copy()
    pred[12, 18:20] = False
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    pred_path = tmp_path / "unet_prediction.png"
    skio.imsave(image_path, image, check_contrast=False)
    skio.imsave(mask_path, manual.astype(np.uint8) * 255, check_contrast=False)
    skio.imsave(pred_path, pred.astype(np.uint8) * 255, check_contrast=False)
    manifest = pd.DataFrame(
        {
            "image_id": ["case_001"],
            "image_path": [str(image_path)],
            "mask_path": [str(mask_path)],
            "unet_prediction_path": [str(pred_path)],
        }
    )

    table = benchmark_prediction_columns(
        manifest,
        {"u_net": "unet_prediction_path"},
    )

    assert len(table) == 1
    assert table.loc[0, "method"] == "u_net"
    assert table.loc[0, "error"] is None
    assert table.loc[0, "dice"] < 1.0
    assert table.loc[0, "pred_mask_path"] == str(pred_path)


def test_benchmark_prediction_columns_reports_missing_prediction_path(tmp_path):
    image = np.zeros((16, 16), dtype=np.uint8)
    image[8, 4:12] = 255
    mask = image > 0
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    skio.imsave(image_path, image, check_contrast=False)
    skio.imsave(mask_path, mask.astype(np.uint8) * 255, check_contrast=False)
    manifest = pd.DataFrame(
        {
            "image_path": [str(image_path)],
            "mask_path": [str(mask_path)],
            "octa_net_prediction_path": [str(tmp_path / "missing.png")],
        }
    )

    table = benchmark_prediction_columns(
        manifest,
        {"octa_net": "octa_net_prediction_path"},
    )

    assert len(table) == 1
    assert "does not exist" in table.loc[0, "error"]


def test_postprocess_score_mask_threshold_and_cleanup():
    score = np.zeros((16, 16), dtype=float)
    score[8, 3:13] = 0.8
    score[1, 1] = 0.9

    mask = postprocess_score_mask(score, threshold=0.5, min_size=3)

    assert mask[8, 6]
    assert not mask[1, 1]


def test_tune_and_evaluate_score_postprocessing_selects_better_threshold(tmp_path):
    image = np.zeros((24, 24), dtype=np.uint8)
    manual = np.zeros((24, 24), dtype=bool)
    manual[12, 5:19] = True
    image[manual] = 255
    score = np.zeros((24, 24), dtype=np.float32)
    score[manual] = 0.7
    score[4:8, 4:8] = 0.45
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    score_path = tmp_path / "score.npy"
    skio.imsave(image_path, image, check_contrast=False)
    skio.imsave(mask_path, manual.astype(np.uint8) * 255, check_contrast=False)
    np.save(score_path, score)
    manifest = pd.DataFrame(
        {
            "image_id": ["case_001"],
            "image_path": [str(image_path)],
            "mask_path": [str(mask_path)],
            "unet_score_path": [str(score_path)],
        }
    )

    tuning_table, tuning_summary, best_params = tune_score_postprocessing(
        manifest,
        {"unet_lite": "unet_score_path"},
        [
            {"threshold": 0.4, "min_size": 1},
            {"threshold": 0.6, "min_size": 1},
        ],
    )
    evaluated = evaluate_score_postprocessing(
        manifest,
        {"unet_lite": "unet_score_path"},
        best_params,
        output_root=tmp_path / "tuned_masks",
    )

    assert len(tuning_table) == 2
    assert tuning_summary["selected"].sum() == 1
    assert best_params["unet_lite"]["threshold"] == 0.6
    assert evaluated.loc[0, "dice"] == 1.0
    assert Path(evaluated.loc[0, "pred_mask_path"]).exists()


def test_tune_native_segmenters_selects_one_candidate_per_method(tmp_path):
    image = np.zeros((32, 32), dtype=np.uint8)
    image[16, 6:26] = 255
    image[8:24, 12] = 220
    mask = image > 0
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    skio.imsave(image_path, image, check_contrast=False)
    skio.imsave(mask_path, mask.astype(np.uint8) * 255, check_contrast=False)
    manifest = pd.DataFrame(
        {
            "image_id": ["case_001"],
            "image_path": [str(image_path)],
            "mask_path": [str(mask_path)],
        }
    )

    tuning_table, tuning_summary, best_params = tune_native_segmenters(
        manifest,
        {
            "diffusion": {
                "n_iter": [1],
                "threshold": ["otsu", "yen"],
                "min_size": [2],
                "clahe": [False],
            }
        },
        methods=["diffusion"],
    )

    assert len(tuning_table) == 2
    assert len(tuning_summary) == 2
    assert tuning_summary["selected"].sum() == 1
    assert set(best_params) == {"diffusion"}
    assert best_params["diffusion"]["threshold"] in {"otsu", "yen"}


def test_tune_native_segmenters_rejects_missing_grid():
    manifest = pd.DataFrame({"image_path": ["image.png"], "mask_path": ["mask.png"]})

    with pytest.raises(ValueError, match="Missing parameter grid"):
        tune_native_segmenters(manifest, {"frangi": []}, methods=["diffusion"])
