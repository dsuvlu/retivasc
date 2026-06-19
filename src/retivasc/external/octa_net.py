"""Adapter helpers for optional OCTA-Net benchmarking.

The functions in this module only prepare RetiVasc data for an external OCTA-Net
checkout and import predictions after that tool has been run separately.
"""

from __future__ import annotations

import shlex
import shutil
from pathlib import Path

import pandas as pd

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def export_for_octa_net(
    manifest_path: str | Path | pd.DataFrame,
    output_root: str | Path,
    image_col: str = "image_path",
    mask_col: str = "mask_path",
    layer_col: str = "layer",
) -> dict[str, object]:
    """Create a simple OCTA-Net-compatible export folder from a RetiVasc manifest."""
    manifest, base_dir = _read_manifest(manifest_path)
    _require_columns(manifest, [image_col, mask_col])

    output_root = Path(output_root)
    images_root = output_root / "images"
    masks_root = output_root / "masks"
    rows = []
    for row_index, row in manifest.iterrows():
        image_path = _resolve_path(row[image_col], base_dir)
        mask_path = _resolve_path(row[mask_col], base_dir)
        layer = _safe_stem(row.get(layer_col, "unknown") or "unknown")
        case_id = _case_id(row, row_index)
        image_dst = images_root / layer / f"{case_id}{image_path.suffix.lower()}"
        mask_dst = masks_root / layer / f"{case_id}{mask_path.suffix.lower()}"
        image_dst.parent.mkdir(parents=True, exist_ok=True)
        mask_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, image_dst)
        shutil.copy2(mask_path, mask_dst)

        out_row = row.to_dict()
        out_row.update(
            {
                "octa_net_case_id": case_id,
                "octa_net_image_path": str(image_dst),
                "octa_net_mask_path": str(mask_dst),
            }
        )
        rows.append(out_row)

    output_root.mkdir(parents=True, exist_ok=True)
    export_manifest = output_root / "octa_net_manifest.csv"
    pd.DataFrame(rows).to_csv(export_manifest, index=False)
    return {
        "export_root": str(output_root),
        "manifest_path": str(export_manifest),
        "image_count": len(rows),
    }


def build_octa_net_commands(
    export_root: str | Path,
    octa_net_repo: str | Path,
    weights_path: str | Path | None,
    output_root: str | Path,
    dataset_name: str = "ROSE",
) -> list[str]:
    """Return shell commands for running OCTA-Net in its own environment."""
    args = [
        "python",
        "test.py",
        "--dataset",
        dataset_name,
        "--data-root",
        str(export_root),
        "--output",
        str(output_root),
    ]
    if weights_path is not None:
        args.extend(["--weights", str(weights_path)])
    command = f"cd {shlex.quote(str(octa_net_repo))} && " + " ".join(
        shlex.quote(part) for part in args
    )
    return [command]


def import_octa_net_predictions(
    prediction_root: str | Path,
    manifest_path: str | Path | pd.DataFrame,
    output_manifest_path: str | Path,
) -> str:
    """Attach OCTA-Net prediction paths to a RetiVasc manifest."""
    manifest, _ = _read_manifest(manifest_path)
    prediction_lookup = _prediction_lookup(Path(prediction_root))
    rows = []
    for row_index, row in manifest.iterrows():
        case_id = row.get("octa_net_case_id") or _case_id(row, row_index)
        prediction = _find_prediction(prediction_lookup, str(case_id))
        out_row = row.to_dict()
        out_row["octa_net_prediction_path"] = str(prediction) if prediction is not None else None
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


def _find_prediction(prediction_lookup: dict[str, Path], case_id: str) -> Path | None:
    for candidate in (case_id, f"{case_id}_pred", f"{case_id}_prediction"):
        if candidate in prediction_lookup:
            return prediction_lookup[candidate]
    prefix_matches = [
        path for stem, path in prediction_lookup.items() if stem.startswith(f"{case_id}_")
    ]
    return prefix_matches[0] if prefix_matches else None

