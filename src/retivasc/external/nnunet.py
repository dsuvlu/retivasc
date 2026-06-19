"""Adapter helpers for optional nnU-Net v2 benchmarking."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io as skio

from retivasc.preprocess import ensure_grayscale

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def prepare_nnunet_dataset(
    manifest_path: str | Path | pd.DataFrame,
    output_root: str | Path,
    dataset_id: int,
    dataset_name: str,
    image_col: str = "image_path",
    mask_col: str = "mask_path",
    channel_names: dict[int, str] | None = None,
    labels: dict[str, int] | None = None,
    use_nifti_single_slice: bool = False,
    split_col: str = "official_split",
) -> str:
    """Write an nnU-Net v2 2D dataset scaffold from a RetiVasc manifest.

    By default this writes PNG images and labels for nnU-Net v2's natural-image 2D
    reader. NIfTI export is intentionally not implemented here to avoid adding heavy
    optional dependencies to the default demo environment.
    """
    if use_nifti_single_slice:
        msg = (
            "NIfTI export requires an optional medical-image writer. "
            "Call with use_nifti_single_slice=False for the lightweight PNG adapter."
        )
        raise NotImplementedError(msg)
    manifest, base_dir = _read_manifest(manifest_path)
    _require_columns(manifest, [image_col, mask_col])

    dataset_root = Path(output_root) / f"Dataset{dataset_id:03d}_{_safe_stem(dataset_name)}"
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"
    images_ts = dataset_root / "imagesTs"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    images_ts.mkdir(parents=True, exist_ok=True)

    training_count = 0
    test_count = 0
    cases = []
    for row_index, row in manifest.iterrows():
        case_id = _case_id(row, row_index)
        image = skio.imread(_resolve_path(row[image_col], base_dir))
        split = str(row.get(split_col, "train")).lower() if split_col in row else "train"
        is_test = split == "test"
        image_dir = images_ts if is_test else images_tr
        image_path = image_dir / f"{case_id}_0000.png"
        skio.imsave(image_path, image, check_contrast=False)

        label_path = None
        if not is_test:
            mask = ensure_grayscale(skio.imread(_resolve_path(row[mask_col], base_dir))) > 0
            label_path = labels_tr / f"{case_id}.png"
            skio.imsave(label_path, mask.astype(np.uint8), check_contrast=False)
            training_count += 1
        else:
            test_count += 1
        cases.append(
            {
                "case_id": case_id,
                "split": "test" if is_test else "train",
                "image_path": str(image_path),
                "label_path": str(label_path) if label_path is not None else None,
            }
        )

    dataset_json = {
        "channel_names": channel_names or {"0": "retinal_image"},
        "labels": labels or {"background": 0, "vessel": 1},
        "numTraining": training_count,
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    with (dataset_root / "dataset.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_json, handle, indent=2, sort_keys=True)
        handle.write("\n")
    pd.DataFrame(cases).to_csv(dataset_root / "retivasc_cases.csv", index=False)
    return str(dataset_root)


def build_nnunet_train_commands(
    dataset_id: int,
    configuration: str = "2d",
    fold: str | int = "all",
    trainer: str = "nnUNetTrainer",
) -> list[str]:
    """Return nnU-Net v2 planning/preprocessing and training commands."""
    dataset = str(int(dataset_id))
    return [
        " ".join(
            shlex.quote(part)
            for part in [
                "nnUNetv2_plan_and_preprocess",
                "-d",
                dataset,
                "--verify_dataset_integrity",
            ]
        ),
        " ".join(
            shlex.quote(part)
            for part in [
                "nnUNetv2_train",
                dataset,
                configuration,
                str(fold),
                "-tr",
                trainer,
            ]
        ),
    ]


def build_nnunet_predict_commands(
    dataset_id: int,
    input_dir: str | Path,
    output_dir: str | Path,
    configuration: str = "2d",
    fold: str | int = "all",
    trainer: str = "nnUNetTrainer",
) -> list[str]:
    """Return an nnU-Net v2 prediction command."""
    return [
        " ".join(
            shlex.quote(part)
            for part in [
                "nnUNetv2_predict",
                "-d",
                str(int(dataset_id)),
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
                "-c",
                configuration,
                "-f",
                str(fold),
                "-tr",
                trainer,
            ]
        )
    ]


def import_nnunet_predictions(
    prediction_root: str | Path,
    manifest_path: str | Path | pd.DataFrame,
    output_manifest_path: str | Path,
) -> str:
    """Attach nnU-Net prediction paths to a RetiVasc manifest."""
    manifest, _ = _read_manifest(manifest_path)
    prediction_lookup = _prediction_lookup(Path(prediction_root))
    rows = []
    for row_index, row in manifest.iterrows():
        case_id = _case_id(row, row_index)
        prediction = prediction_lookup.get(case_id)
        out_row = row.to_dict()
        out_row["nnunet_prediction_path"] = str(prediction) if prediction is not None else None
        rows.append(out_row)

    output_manifest_path = Path(output_manifest_path)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_manifest_path, index=False)
    return str(output_manifest_path)


def _read_manifest(manifest_path: str | Path | pd.DataFrame) -> tuple[pd.DataFrame, Path]:
    if isinstance(manifest_path, pd.DataFrame):
        return manifest_path.copy(), Path.cwd()
    path = Path(manifest_path)
    return pd.read_csv(path), path.parent


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        msg = f"Missing required manifest columns: {', '.join(missing)}"
        raise ValueError(msg)


def _resolve_path(value: object, base_dir: Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        msg = f"Required path does not exist: {path}"
        raise FileNotFoundError(msg)
    return path


def _case_id(row: pd.Series, row_index: object) -> str:
    value = row.get("image_id", None) or row.get("subject_id", None) or row_index
    return _safe_stem(str(value))


def _safe_stem(value: str) -> str:
    cleaned = [char if char.isalnum() or char in {"-", "_"} else "_" for char in value]
    return "".join(cleaned).strip("_") or "case"


def _prediction_lookup(prediction_root: Path) -> dict[str, Path]:
    if not prediction_root.exists():
        return {}
    return {
        path.stem: path
        for path in sorted(prediction_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }

