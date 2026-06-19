import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from skimage import io as skio

from retivasc.external.nnunet import (
    build_nnunet_predict_commands,
    build_nnunet_train_commands,
    import_nnunet_predictions,
    prepare_nnunet_dataset,
)
from retivasc.external.octa_net import (
    build_octa_net_commands,
    export_for_octa_net,
    import_octa_net_predictions,
)
from retivasc.external.registry import MODEL_REGISTRY


def _write_image_pair(root: Path, stem: str) -> tuple[Path, Path]:
    image = np.zeros((16, 16), dtype=np.uint8)
    image[8, 3:13] = 255
    mask = image > 0
    image_path = root / f"{stem}.png"
    mask_path = root / f"{stem}_mask.png"
    skio.imsave(image_path, image, check_contrast=False)
    skio.imsave(mask_path, mask.astype(np.uint8) * 255, check_contrast=False)
    return image_path, mask_path


def _manifest_csv(tmp_path: Path) -> Path:
    image_a, mask_a = _write_image_pair(tmp_path, "case_001")
    image_b, mask_b = _write_image_pair(tmp_path, "case_002")
    manifest = pd.DataFrame(
        {
            "dataset": ["ROSE", "ROSE"],
            "image_id": ["case_001", "case_002"],
            "subject_id": ["subject_001", "subject_002"],
            "layer": ["SVC", "DVC"],
            "official_split": ["train", "test"],
            "image_path": [image_a.name, image_b.name],
            "mask_path": [mask_a.name, mask_b.name],
        }
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest_path


def test_octa_net_export_commands_and_prediction_import(tmp_path):
    manifest_path = _manifest_csv(tmp_path)
    export_root = tmp_path / "octa_export"

    summary = export_for_octa_net(manifest_path, export_root)

    export_manifest = Path(str(summary["manifest_path"]))
    exported = pd.read_csv(export_manifest)
    assert summary["image_count"] == 2
    assert export_manifest.exists()
    assert Path(exported.loc[0, "octa_net_image_path"]).exists()
    assert Path(exported.loc[0, "octa_net_mask_path"]).exists()

    commands = build_octa_net_commands(
        export_root,
        tmp_path / "OCTA-Net",
        tmp_path / "weights.pth",
        tmp_path / "octa_predictions",
    )
    assert "OCTA-Net" in commands[0]
    assert "weights.pth" in commands[0]
    assert "octa_predictions" in commands[0]

    prediction_root = tmp_path / "octa_predictions"
    prediction_root.mkdir()
    skio.imsave(
        prediction_root / "case_001_pred.png",
        np.zeros((16, 16), dtype=np.uint8),
        check_contrast=False,
    )
    output_manifest = import_octa_net_predictions(
        prediction_root, export_manifest, tmp_path / "octa_predictions.csv"
    )
    imported = pd.read_csv(output_manifest)
    assert imported.loc[0, "octa_net_prediction_path"].endswith("case_001_pred.png")


def test_nnunet_prepare_commands_and_prediction_import(tmp_path):
    manifest_path = _manifest_csv(tmp_path)

    dataset_root = Path(
        prepare_nnunet_dataset(
            manifest_path,
            tmp_path / "nnunet_raw",
            dataset_id=501,
            dataset_name="RetiVascRose",
        )
    )

    dataset_json = json.loads((dataset_root / "dataset.json").read_text())
    assert dataset_json["file_ending"] == ".png"
    assert dataset_json["overwrite_image_reader_writer"] == "NaturalImage2DIO"
    assert dataset_json["numTraining"] == 1
    assert (dataset_root / "imagesTr" / "case_001_0000.png").exists()
    assert (dataset_root / "labelsTr" / "case_001.png").exists()
    assert (dataset_root / "imagesTs" / "case_002_0000.png").exists()

    train_commands = build_nnunet_train_commands(501)
    predict_commands = build_nnunet_predict_commands(
        501, dataset_root / "imagesTs", tmp_path / "nnunet_predictions"
    )
    assert train_commands[0].startswith("nnUNetv2_plan_and_preprocess")
    assert train_commands[1].startswith("nnUNetv2_train")
    assert predict_commands[0].startswith("nnUNetv2_predict")

    prediction_root = tmp_path / "nnunet_predictions"
    prediction_root.mkdir()
    skio.imsave(
        prediction_root / "case_002.png",
        np.zeros((16, 16), dtype=np.uint8),
        check_contrast=False,
    )
    output_manifest = import_nnunet_predictions(
        prediction_root, manifest_path, tmp_path / "nnunet_predictions.csv"
    )
    imported = pd.read_csv(output_manifest)
    assert imported.loc[1, "nnunet_prediction_path"].endswith("case_002.png")


def test_nnunet_nifti_export_is_explicitly_optional(tmp_path):
    manifest_path = _manifest_csv(tmp_path)

    with pytest.raises(NotImplementedError, match="NIfTI export"):
        prepare_nnunet_dataset(
            manifest_path,
            tmp_path / "nnunet_raw",
            dataset_id=501,
            dataset_name="RetiVascRose",
            use_nifti_single_slice=True,
        )


def test_model_registry_marks_external_tools_as_optional():
    assert MODEL_REGISTRY["octa_net"]["requires_external_tool"] is True
    assert MODEL_REGISTRY["u_net"]["requires_external_tool"] is True
    assert MODEL_REGISTRY["nnunet"]["requires_external_tool"] is True
    assert MODEL_REGISTRY["unet_lite"]["requires_external_tool"] is False
    assert MODEL_REGISTRY["octa_net_lite"]["requires_external_tool"] is False
    assert MODEL_REGISTRY["nnunet_lite"]["requires_external_tool"] is False
    assert MODEL_REGISTRY["frangi"]["requires_external_tool"] is False
