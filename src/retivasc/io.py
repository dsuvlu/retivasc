"""Local dataset manifest loading.

The loaders intentionally do not download medical images. They either read a local
manifest or perform conservative discovery. If metadata are ambiguous, they ask for
a small manifest instead of inventing labels.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MASK_TOKENS = (
    "mask",
    "masks",
    "manual",
    "label",
    "labels",
    "gt",
    "groundtruth",
    "ground_truth",
    "ground truth",
    "annotation",
    "annotations",
)
FIVES_LABEL_CODES = {
    "A": "AMD",
    "D": "DR",
    "G": "glaucoma",
    "N": "normal",
}


class DataNotFoundError(FileNotFoundError):
    """Raised when an expected local dataset root is absent or empty."""


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear ValueError if required metadata columns are absent."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        msg = f"Missing required metadata columns: {', '.join(missing)}"
        raise ValueError(msg)


def _dataset_missing_message(name: str, root: Path, extra: str = "") -> str:
    tail = f"\n{extra}" if extra else ""
    return (
        f"{name} data were not found at {root}.\n"
        f"Place the dataset under {root} or add a manifest.csv there. "
        "Raw medical images are intentionally not downloaded or committed."
        f"{tail}"
    )


def _read_manifest(root: Path) -> pd.DataFrame | None:
    for filename in ("manifest.csv", "metadata.csv", "rose_manifest.csv", "fives_manifest.csv"):
        path = root / filename
        if path.exists():
            return pd.read_csv(path)
    return None


def _resolve_path_column(df: pd.DataFrame, root: Path, column: str) -> None:
    if column not in df.columns:
        return

    def resolve(value: object) -> str:
        path = Path(str(value))
        if not path.is_absolute():
            path = root / path
        return str(path)

    df[column] = df[column].map(resolve)


def _standardize_manifest(
    df: pd.DataFrame,
    root: Path,
    *,
    dataset: str,
    modality: str,
    split_group_default: str = "subject_id",
) -> pd.DataFrame:
    out = df.copy()
    out["dataset"] = out.get("dataset", dataset)
    out["modality"] = out.get("modality", modality)
    if "image_id" not in out.columns and "image_path" in out.columns:
        out["image_id"] = out["image_path"].map(lambda value: Path(str(value)).stem)
    if "split_group" not in out.columns and split_group_default in out.columns:
        out["split_group"] = out[split_group_default]
    _resolve_path_column(out, root, "image_path")
    _resolve_path_column(out, root, "mask_path")
    return out


def _is_mask_path(path: Path) -> bool:
    lowered = path.as_posix().lower()
    return any(token in lowered for token in MASK_TOKENS)


def _image_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _normalized_stem(path: Path) -> str:
    stem = path.stem.lower()
    for token in MASK_TOKENS:
        stem = re.sub(rf"(^|[_\-\s]){re.escape(token)}($|[_\-\s])", "_", stem)
    stem = re.sub(r"[_\-\s]+", "_", stem).strip("_")
    return stem


def _pair_images_and_masks(root: Path) -> list[tuple[Path, Path]]:
    files = _image_files(root)
    masks = [path for path in files if _is_mask_path(path)]
    images = [path for path in files if not _is_mask_path(path)]
    mask_by_stem = {_normalized_stem(mask): mask for mask in masks}
    pairs = []
    for image in images:
        mask = mask_by_stem.get(_normalized_stem(image))
        if mask is not None:
            pairs.append((image, mask))
    return pairs


def _infer_layer(path: Path) -> str | None:
    text = path.as_posix().lower()
    if "svc" in text and "dvc" in text:
        return "SVC+DVC"
    if "svc" in text:
        return "SVC"
    if "dvc" in text:
        return "DVC"
    return None


def _infer_label(path: Path) -> str | None:
    parts = [part.lower() for part in path.parts]
    if any(part in {"ad", "alzheimers", "alzheimer", "dementia"} for part in parts):
        return "AD"
    if any(part in {"control", "controls", "normal", "healthy", "cn"} for part in parts):
        return "control"
    disease_names = {"amd", "dr", "glaucoma", "normal"}
    for part in parts:
        if part in disease_names:
            return part
    return None


def _infer_fives_label_from_name(path: Path) -> tuple[str | None, str | None]:
    match = re.search(r"(?:^|[_\-\s])([ADGN])$", path.stem, flags=re.I)
    if not match:
        return _infer_label(path), None
    code = match.group(1).upper()
    return FIVES_LABEL_CODES[code], code


def _normalized_dir_name(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "", path.name.lower())


def _find_child_dir(parent: Path, names: set[str]) -> Path | None:
    normalized_names = {re.sub(r"[^a-z0-9]+", "", name.lower()) for name in names}
    for child in sorted(parent.iterdir()):
        if child.is_dir() and _normalized_dir_name(child) in normalized_names:
            return child
    return None


def _candidate_roots(root: Path) -> list[Path]:
    candidates = [root]
    candidates.extend(sorted(child for child in root.iterdir() if child.is_dir()))
    return candidates


def _load_fives_official_manifest(root: Path) -> pd.DataFrame | None:
    """Load the official FIVES train/test Original/Ground Truth layout if present."""
    rows = []
    for candidate in _candidate_roots(root):
        for split in ("train", "test"):
            split_dir = _find_child_dir(candidate, {split})
            if split_dir is None:
                continue
            original_dir = _find_child_dir(split_dir, {"Original", "Images", "Image"})
            mask_dir = _find_child_dir(
                split_dir,
                {
                    "Ground Truth",
                    "GroundTruth",
                    "Ground_Truth",
                    "Masks",
                    "Mask",
                    "Annotations",
                },
            )
            if original_dir is None or mask_dir is None:
                continue

            masks_by_stem = {path.stem: path for path in _image_files(mask_dir)}
            for image_path in _image_files(original_dir):
                mask_path = masks_by_stem.get(image_path.stem)
                if mask_path is None:
                    continue
                label, diagnosis_code = _infer_fives_label_from_name(image_path)
                image_id = f"{split}_{image_path.stem}"
                rows.append(
                    {
                        "dataset": "FIVES",
                        "subject_id": image_id,
                        "image_id": image_id,
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "modality": "fundus",
                        "label": label,
                        "diagnosis_code": diagnosis_code,
                        "official_split": split,
                        "split_group": image_id,
                    }
                )

        if rows:
            return pd.DataFrame(rows)
    return None


def _infer_subject_id(path: Path) -> str | None:
    stem = path.stem
    match = re.search(r"(?:subject|subj|sub|patient|pt)[_\-\s]*(\d+)", stem, flags=re.I)
    if match:
        return match.group(1)
    first_number = re.search(r"\d+", stem)
    if first_number:
        return first_number.group(0)
    return None


def load_rose_manifest(root: str | Path) -> pd.DataFrame:
    """Return one row per ROSE image/layer with paths and subject-level split groups."""
    root = Path(root)
    if not root.exists():
        raise DataNotFoundError(_dataset_missing_message("ROSE", root))

    manifest = _read_manifest(root)
    if manifest is not None:
        out = _standardize_manifest(manifest, root, dataset="ROSE", modality="OCTA")
        require_columns(out, ["image_path", "mask_path", "subject_id", "layer", "label"])
        if "split_group" not in out.columns:
            out["split_group"] = out["subject_id"]
        return out

    pairs = _pair_images_and_masks(root)
    if not pairs:
        raise DataNotFoundError(
            _dataset_missing_message(
                "ROSE",
                root,
                "Expected a manifest.csv or image/mask files with matching stems.",
            )
        )

    rows = []
    uncertain = []
    for image_path, mask_path in pairs:
        layer = _infer_layer(image_path)
        label = _infer_label(image_path)
        subject_id = _infer_subject_id(image_path)
        if layer is None or label is None or subject_id is None:
            uncertain.append(image_path)
            continue
        rows.append(
            {
                "dataset": "ROSE",
                "subject_id": subject_id,
                "image_id": image_path.stem,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "modality": "OCTA",
                "layer": layer,
                "label": label,
                "split_group": subject_id,
            }
        )

    if uncertain or not rows:
        preview = "\n".join(str(path) for path in uncertain[:5])
        msg = (
            "Could not confidently infer ROSE subject_id, layer, and label from filenames. "
            "Add data/raw/rose/manifest.csv with columns image_path, mask_path, subject_id, "
            "layer, label, and split_group.\n"
            f"Ambiguous examples:\n{preview}"
        )
        raise ValueError(msg)

    return pd.DataFrame(rows)


def load_fives_manifest(root: str | Path) -> pd.DataFrame:
    """Return one row per FIVES image with image_path, mask_path, and labels if available."""
    root = Path(root)
    if not root.exists():
        raise DataNotFoundError(_dataset_missing_message("FIVES", root))

    manifest = _read_manifest(root)
    if manifest is not None:
        out = _standardize_manifest(manifest, root, dataset="FIVES", modality="fundus")
        require_columns(out, ["image_path", "mask_path"])
        if "subject_id" not in out.columns:
            out["subject_id"] = out["image_id"]
        if "split_group" not in out.columns:
            out["split_group"] = out["subject_id"]
        return out

    official = _load_fives_official_manifest(root)
    if official is not None:
        return official

    pairs = _pair_images_and_masks(root)
    if not pairs:
        raise DataNotFoundError(
            _dataset_missing_message(
                "FIVES",
                root,
                "Expected a manifest.csv or image/mask files with matching stems.",
            )
        )

    rows = []
    for image_path, mask_path in pairs:
        label, diagnosis_code = _infer_fives_label_from_name(image_path)
        image_id = image_path.stem
        rows.append(
            {
                "dataset": "FIVES",
                "subject_id": image_id,
                "image_id": image_id,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "modality": "fundus",
                "label": label,
                "diagnosis_code": diagnosis_code,
                "split_group": image_id,
            }
        )
    return pd.DataFrame(rows)
