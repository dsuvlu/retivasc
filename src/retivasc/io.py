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
ROSE1_ALZHEIMERS_LABEL_RANGES = {
    "train": {
        "disease": range(1, 21),
        "control": range(21, 31),
    },
    "test": {
        "disease": range(1, 7),
        "control": range(7, 10),
    },
}
ROSE1_LABEL_SOURCE = (
    "ROSE-1 official AD/control cohort ordering: train 1-20 disease, "
    "train 21-30 control, test 1-6 disease, test 7-9 control"
)


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


def _infer_rose2_subject_id(path: Path) -> str | None:
    stem = path.stem
    match = re.match(r"([A-Za-z]*\d+(?:-\d+)?)", stem)
    if match:
        return match.group(1)
    return _infer_subject_id(path)


def _infer_rose1_label(split: str, image_stem: str) -> str | None:
    try:
        subject_number = int(image_stem)
    except ValueError:
        return None
    for label, subject_range in ROSE1_ALZHEIMERS_LABEL_RANGES.get(split, {}).items():
        if subject_number in subject_range:
            return label
    return None


def _raise_on_cross_split_subjects(df: pd.DataFrame, *, dataset_name: str) -> None:
    if "official_split" not in df.columns or df.empty:
        return
    split_counts = df.groupby("subject_id")["official_split"].nunique(dropna=True)
    collisions = sorted(str(subject_id) for subject_id, count in split_counts.items() if count > 1)
    if not collisions:
        return
    preview = ", ".join(collisions[:5])
    msg = (
        f"{dataset_name} subject_id appears under multiple official splits: {preview}. "
        "Supply a manifest.csv with explicit subject_id and split_group columns before "
        "using these data for split-sensitive analysis."
    )
    raise ValueError(msg)


def _load_rose_official_manifest(
    root: Path, *, require_split_safe: bool = True
) -> pd.DataFrame | None:
    """Load the official ROSE/ROSE-O segmentation layouts if present."""
    rows = []

    for rose1_root in sorted(root.rglob("ROSE-1")):
        for layer_dir in sorted(child for child in rose1_root.iterdir() if child.is_dir()):
            layer = _infer_layer(layer_dir)
            if layer is None:
                continue
            for split in ("train", "test"):
                split_dir = _find_child_dir(layer_dir, {split})
                if split_dir is None:
                    continue
                image_dir = _find_child_dir(split_dir, {"img", "image", "images", "original"})
                mask_dir = _find_child_dir(split_dir, {"gt", "mask", "masks", "vessel"})
                if image_dir is None or mask_dir is None:
                    continue

                masks_by_stem = {path.stem: path for path in _image_files(mask_dir)}
                for image_path in _image_files(image_dir):
                    mask_path = masks_by_stem.get(image_path.stem)
                    if mask_path is None:
                        continue
                    subject_id = f"ROSE-1_{split}_{image_path.stem}"
                    label = _infer_rose1_label(split, image_path.stem)
                    diagnosis = "Alzheimer's disease" if label == "disease" else label
                    rows.append(
                        {
                            "dataset": "ROSE-1",
                            "subject_id": subject_id,
                            "image_id": f"{subject_id}_{layer}",
                            "image_path": str(image_path),
                            "mask_path": str(mask_path),
                            "modality": "OCTA",
                            "layer": layer,
                            "label": label,
                            "diagnosis": diagnosis,
                            "label_source": ROSE1_LABEL_SOURCE if label is not None else None,
                            "official_split": split,
                            "split_group": subject_id,
                        }
                    )

    for rose2_root in sorted(root.rglob("ROSE-2")):
        for split in ("train", "test"):
            split_dir = _find_child_dir(rose2_root, {split})
            if split_dir is None:
                continue
            image_dir = _find_child_dir(split_dir, {"original", "img", "image", "images"})
            mask_dir = _find_child_dir(split_dir, {"gt", "mask", "masks", "vessel"})
            if image_dir is None or mask_dir is None:
                continue

            masks_by_stem = {path.stem: path for path in _image_files(mask_dir)}
            for image_path in _image_files(image_dir):
                mask_path = masks_by_stem.get(image_path.stem)
                if mask_path is None:
                    continue
                raw_subject_id = _infer_rose2_subject_id(image_path) or image_path.stem
                subject_id = f"ROSE-2_{raw_subject_id}"
                layer = _infer_layer(image_path) or "SVP"
                rows.append(
                    {
                        "dataset": "ROSE-2",
                        "subject_id": subject_id,
                        "image_id": f"ROSE-2_{split}_{image_path.stem}",
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "modality": "OCTA",
                        "layer": layer,
                        "label": None,
                        "official_split": split,
                        "split_group": subject_id,
                    }
                )

    if rows:
        out = pd.DataFrame(rows)
        if require_split_safe:
            _raise_on_cross_split_subjects(out, dataset_name="ROSE")
        return out
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


def _infer_split_name(path: Path) -> str | None:
    split_terms = {"train", "test", "val", "valid", "validation"}
    for part in path.parts:
        lowered = part.lower()
        if lowered in split_terms:
            return lowered
    return None


def load_rose_manifest(root: str | Path, *, require_split_safe: bool = True) -> pd.DataFrame:
    """Return one row per ROSE image/layer with paths and subject-level split groups.

    By default, official ROSE layouts are rejected when a discovered subject id appears
    under multiple official split folders. Demo notebooks can set
    ``require_split_safe=False`` to load image/mask rows for non-split-sensitive
    visualization and feature extraction.
    """
    root = Path(root)
    if not root.exists():
        raise DataNotFoundError(_dataset_missing_message("ROSE", root))

    manifest = _read_manifest(root)
    if manifest is not None:
        out = _standardize_manifest(manifest, root, dataset="ROSE", modality="OCTA")
        require_columns(out, ["image_path", "mask_path", "subject_id", "layer"])
        if "label" not in out.columns:
            out["label"] = None
        if "split_group" not in out.columns:
            out["split_group"] = out["subject_id"]
        return out

    official = _load_rose_official_manifest(root, require_split_safe=require_split_safe)
    if official is not None:
        return official

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
        split_name = _infer_split_name(image_path)
        if layer is None or subject_id is None or split_name is not None:
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
            "Could not confidently infer ROSE subject_id and layer from filenames. "
            "Add data/raw/rose/manifest.csv with columns image_path, mask_path, subject_id, "
            "layer, optional label, and split_group. If files are arranged under split "
            "directories, use a manifest so split_group cannot be hidden by path names.\n"
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
