"""Evaluation helpers for segmentation model comparison."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io as skio
from skimage import morphology

from retivasc.features import extract_vascular_features
from retivasc.metrics import (
    dice_score,
    iou_score,
    precision_score_binary,
    recall_score_binary,
    specificity_score_binary,
)
from retivasc.preprocess import ensure_grayscale
from retivasc.segment import (
    SegmentResult,
    diffusion_threshold_segment,
    frangi_segment,
    geodesic_voting_segment,
    random_walker_segment,
)

NATIVE_SEGMENTERS = {
    "frangi": frangi_segment,
    "diffusion": diffusion_threshold_segment,
    "diffusion_threshold": diffusion_threshold_segment,
    "random_walker": random_walker_segment,
    "geodesic": geodesic_voting_segment,
    "geodesic_voting": geodesic_voting_segment,
}


def compare_mask_features(
    manual_mask: np.ndarray,
    pred_mask: np.ndarray,
    *,
    fov_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compare vascular features from a predicted mask against a manual mask."""
    manual = extract_vascular_features(manual_mask, fov_mask=fov_mask)
    predicted = extract_vascular_features(pred_mask, fov_mask=fov_mask)
    out: dict[str, float] = {}
    for feature, manual_value in manual.items():
        predicted_value = predicted[feature]
        abs_error = abs(predicted_value - manual_value)
        if manual_value == 0:
            rel_error = 0.0 if predicted_value == 0 else float("inf")
        else:
            rel_error = abs_error / abs(manual_value)
        out[f"manual_{feature}"] = float(manual_value)
        out[f"predicted_{feature}"] = float(predicted_value)
        out[f"abs_error_{feature}"] = float(abs_error)
        out[f"rel_error_{feature}"] = float(rel_error)
    return out


def compare_segmentation_masks(
    manual_mask: np.ndarray,
    pred_mask: np.ndarray,
    *,
    fov_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Return mask quality metrics for one manual/predicted mask pair."""
    true = np.asarray(manual_mask, dtype=bool)
    pred = np.asarray(pred_mask, dtype=bool)
    if true.shape != pred.shape:
        msg = f"Shape mismatch: manual_mask {true.shape}, pred_mask {pred.shape}."
        raise ValueError(msg)
    fov = None
    if fov_mask is not None:
        fov = np.asarray(fov_mask, dtype=bool)
        if fov.shape != true.shape:
            msg = f"fov_mask shape {fov.shape} does not match mask shape {true.shape}."
            raise ValueError(msg)
    return {
        "dice": dice_score(true, pred, mask=fov),
        "iou": iou_score(true, pred, mask=fov),
        "precision": precision_score_binary(true, pred, mask=fov),
        "recall": recall_score_binary(true, pred, mask=fov),
        "specificity": specificity_score_binary(true, pred, mask=fov),
    }


def available_native_methods() -> tuple[str, ...]:
    """Return accepted method names for native segmentation benchmarks."""
    return tuple(NATIVE_SEGMENTERS)


def benchmark_native_segmenters(
    manifest: pd.DataFrame,
    methods: Sequence[str] | None = None,
    *,
    image_col: str = "image_path",
    mask_col: str = "mask_path",
    fov_col: str | None = None,
    output_root: str | Path | None = None,
    method_params: Mapping[str, Mapping[str, object]] | None = None,
    max_rows: int | None = None,
    continue_on_error: bool = True,
) -> pd.DataFrame:
    """Run native segmentation methods over a manifest and return one benchmark table.

    The returned table keeps segmentation quality and vascular-feature stability in the
    same row so downstream notebooks can rank methods without recomputing masks.
    """
    method_names = list(methods or ("frangi", "diffusion", "random_walker", "geodesic"))
    unknown = sorted({name for name in method_names if name not in NATIVE_SEGMENTERS})
    if unknown:
        msg = f"Unknown native segmentation method(s): {', '.join(unknown)}"
        raise ValueError(msg)

    required_columns = [image_col, mask_col]
    if fov_col is not None:
        required_columns.append(fov_col)
    _require_columns(manifest, required_columns)

    output_path = Path(output_root) if output_root is not None else None
    if output_path is not None:
        (output_path / "masks").mkdir(parents=True, exist_ok=True)
        (output_path / "scores").mkdir(parents=True, exist_ok=True)

    rows = manifest.head(max_rows) if max_rows is not None else manifest
    records: list[dict[str, object]] = []
    for row_index, row in rows.iterrows():
        base_record = _benchmark_base_record(row, image_col=image_col, mask_col=mask_col)
        try:
            image = skio.imread(row[image_col])
            manual_mask = ensure_grayscale(skio.imread(row[mask_col])) > 0
            fov_mask = _read_optional_mask(row, fov_col)
        except Exception as exc:
            if not continue_on_error:
                raise
            for method_name in method_names:
                records.append(
                    {
                        **base_record,
                        "method": method_name,
                        "error": f"input load failed: {exc}",
                    }
                )
            continue

        for method_name in method_names:
            params = dict((method_params or {}).get(method_name, {}))
            params.update((method_params or {}).get(NATIVE_SEGMENTERS[method_name].__name__, {}))
            started = time.perf_counter()
            try:
                result = NATIVE_SEGMENTERS[method_name](image, fov_mask=fov_mask, **params)
                elapsed = time.perf_counter() - started
                pred_path, score_path = _write_benchmark_outputs(
                    result,
                    output_path=output_path,
                    row_index=row_index,
                    image_id=base_record.get("image_id"),
                )
                mask_metrics = compare_segmentation_masks(
                    manual_mask, result.mask, fov_mask=fov_mask
                )
                feature_metrics = compare_mask_features(
                    manual_mask, result.mask, fov_mask=fov_mask
                )
                records.append(
                    {
                        **base_record,
                        "method": result.method,
                        "pred_mask_path": pred_path,
                        "score_path": score_path,
                        "runtime_seconds": elapsed,
                        "params_json": _to_json(result.params),
                        "diagnostics_json": _to_json(result.diagnostics),
                        "error": None,
                        **mask_metrics,
                        **feature_metrics,
                    }
                )
            except Exception as exc:
                if not continue_on_error:
                    raise
                records.append(
                    {
                        **base_record,
                        "method": method_name,
                        "runtime_seconds": time.perf_counter() - started,
                        "params_json": _to_json(params),
                        "error": str(exc),
                    }
                )
    return pd.DataFrame(records)


def benchmark_prediction_columns(
    manifest: pd.DataFrame,
    prediction_cols: Mapping[str, str],
    *,
    image_col: str = "image_path",
    mask_col: str = "mask_path",
    fov_col: str | None = None,
    max_rows: int | None = None,
    continue_on_error: bool = True,
) -> pd.DataFrame:
    """Evaluate externally generated prediction masks listed in manifest columns."""
    required_columns = [image_col, mask_col]
    if fov_col is not None:
        required_columns.append(fov_col)
    _require_columns(manifest, required_columns)

    rows = manifest.head(max_rows) if max_rows is not None else manifest
    records: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        base_record = _benchmark_base_record(row, image_col=image_col, mask_col=mask_col)
        try:
            manual_mask = ensure_grayscale(skio.imread(row[mask_col])) > 0
            fov_mask = _read_optional_mask(row, fov_col)
        except Exception as exc:
            if not continue_on_error:
                raise
            for method_name in prediction_cols:
                records.append(
                    {
                        **base_record,
                        "method": method_name,
                        "error": f"input load failed: {exc}",
                    }
                )
            continue

        for method_name, prediction_col in prediction_cols.items():
            started = time.perf_counter()
            try:
                if prediction_col not in row.index:
                    raise ValueError(f"prediction column {prediction_col!r} is missing.")
                pred_path = _coerce_prediction_path(row[prediction_col])
                pred_mask = ensure_grayscale(skio.imread(pred_path)) > 0
                mask_metrics = compare_segmentation_masks(
                    manual_mask, pred_mask, fov_mask=fov_mask
                )
                feature_metrics = compare_mask_features(
                    manual_mask, pred_mask, fov_mask=fov_mask
                )
                records.append(
                    {
                        **base_record,
                        "method": method_name,
                        "pred_mask_path": str(pred_path),
                        "score_path": None,
                        "runtime_seconds": time.perf_counter() - started,
                        "params_json": "{}",
                        "diagnostics_json": _to_json({"source_column": prediction_col}),
                        "error": None,
                        **mask_metrics,
                        **feature_metrics,
                    }
                )
            except Exception as exc:
                if not continue_on_error:
                    raise
                records.append(
                    {
                        **base_record,
                        "method": method_name,
                        "pred_mask_path": row.get(prediction_col, None),
                        "runtime_seconds": time.perf_counter() - started,
                        "params_json": "{}",
                        "diagnostics_json": _to_json({"source_column": prediction_col}),
                        "error": str(exc),
                    }
                )
    return pd.DataFrame(records)


def postprocess_score_mask(
    score: np.ndarray,
    *,
    threshold: float = 0.5,
    min_size: int = 16,
    closing_radius: int = 0,
    dilation_radius: int = 0,
    fov_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a continuous score/probability map to a cleaned binary mask."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")
    if min_size < 0:
        raise ValueError("min_size must be non-negative.")
    if closing_radius < 0:
        raise ValueError("closing_radius must be non-negative.")
    if dilation_radius < 0:
        raise ValueError("dilation_radius must be non-negative.")

    arr = _normalize_score(score)
    mask = arr >= threshold
    if closing_radius > 0 and mask.any():
        mask = morphology.closing(mask, morphology.disk(closing_radius))
    if dilation_radius > 0 and mask.any():
        mask = morphology.dilation(mask, morphology.disk(dilation_radius))
    if min_size > 0 and mask.any():
        max_size = max(0, min_size - 1)
        mask = morphology.remove_small_objects(mask, max_size=max_size)
        mask = morphology.remove_small_holes(mask, max_size=max_size)
    if fov_mask is not None:
        fov = np.asarray(fov_mask, dtype=bool)
        if fov.shape != mask.shape:
            msg = f"fov_mask shape {fov.shape} does not match score shape {mask.shape}."
            raise ValueError(msg)
        mask = mask & fov
    return mask.astype(bool, copy=False)


def tune_score_postprocessing(
    manifest: pd.DataFrame,
    score_cols: Mapping[str, str],
    param_grid: Sequence[Mapping[str, object]] | Mapping[str, Sequence[object]],
    *,
    image_col: str = "image_path",
    mask_col: str = "mask_path",
    fov_col: str | None = None,
    score_col: str = "dice",
    higher_is_better: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    """Tune score-map threshold/morphology parameters for probability-style outputs."""
    _require_columns(manifest, [image_col, mask_col, *score_cols.values()])
    candidates = _expand_param_grid(param_grid)
    if not candidates:
        raise ValueError("param_grid must contain at least one candidate.")

    records: list[dict[str, object]] = []
    for method_name, score_path_col in score_cols.items():
        for candidate_index, params in enumerate(candidates):
            table = evaluate_score_postprocessing(
                manifest,
                {method_name: score_path_col},
                {method_name: params},
                image_col=image_col,
                mask_col=mask_col,
                fov_col=fov_col,
            )
            table["requested_method"] = method_name
            table["candidate_index"] = candidate_index
            table["candidate_params_json"] = _to_json(params)
            records.append(table)

    tuning_table = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    tuning_summary = summarize_tuning_results(
        tuning_table,
        score_col=score_col,
        higher_is_better=higher_is_better,
    )
    best_params = {
        row.requested_method: json.loads(row.candidate_params_json)
        for row in tuning_summary.itertuples(index=False)
        if bool(row.selected)
    }
    return tuning_table, tuning_summary, best_params


def evaluate_score_postprocessing(
    manifest: pd.DataFrame,
    score_cols: Mapping[str, str],
    method_params: Mapping[str, Mapping[str, object]] | None = None,
    *,
    image_col: str = "image_path",
    mask_col: str = "mask_path",
    fov_col: str | None = None,
    output_root: str | Path | None = None,
    prediction_col_suffix: str = "_tuned_prediction_path",
    max_rows: int | None = None,
    continue_on_error: bool = True,
) -> pd.DataFrame:
    """Evaluate score maps after applying tuned threshold/morphology settings."""
    required_columns = [image_col, mask_col, *score_cols.values()]
    if fov_col is not None:
        required_columns.append(fov_col)
    _require_columns(manifest, required_columns)

    output_path = Path(output_root) if output_root is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    rows = manifest.head(max_rows) if max_rows is not None else manifest
    records: list[dict[str, object]] = []
    for row_index, row in rows.iterrows():
        base_record = _benchmark_base_record(row, image_col=image_col, mask_col=mask_col)
        try:
            manual_mask = ensure_grayscale(skio.imread(row[mask_col])) > 0
            fov_mask = _read_optional_mask(row, fov_col)
        except Exception as exc:
            if not continue_on_error:
                raise
            for method_name in score_cols:
                records.append(
                    {
                        **base_record,
                        "method": method_name,
                        "error": f"input load failed: {exc}",
                    }
                )
            continue

        for method_name, score_path_col in score_cols.items():
            started = time.perf_counter()
            params = dict((method_params or {}).get(method_name, {}))
            try:
                score_path = _coerce_prediction_path(row[score_path_col])
                score = _read_score_map(score_path)
                if score.shape != manual_mask.shape:
                    msg = (
                        f"score shape {score.shape} does not match "
                        f"mask shape {manual_mask.shape}."
                    )
                    raise ValueError(msg)
                pred_mask = postprocess_score_mask(score, fov_mask=fov_mask, **params)
                pred_path = _write_score_postprocess_mask(
                    pred_mask,
                    output_path=output_path,
                    row_index=row_index,
                    image_id=base_record.get("image_id"),
                    method=method_name,
                    suffix=prediction_col_suffix,
                )
                mask_metrics = compare_segmentation_masks(
                    manual_mask, pred_mask, fov_mask=fov_mask
                )
                feature_metrics = compare_mask_features(
                    manual_mask, pred_mask, fov_mask=fov_mask
                )
                records.append(
                    {
                        **base_record,
                        "method": method_name,
                        "pred_mask_path": pred_path,
                        "score_path": str(score_path),
                        "runtime_seconds": time.perf_counter() - started,
                        "params_json": _to_json(params),
                        "diagnostics_json": _to_json({"source_column": score_path_col}),
                        "error": None,
                        **mask_metrics,
                        **feature_metrics,
                    }
                )
            except Exception as exc:
                if not continue_on_error:
                    raise
                records.append(
                    {
                        **base_record,
                        "method": method_name,
                        "score_path": row.get(score_path_col, None),
                        "runtime_seconds": time.perf_counter() - started,
                        "params_json": _to_json(params),
                        "diagnostics_json": _to_json({"source_column": score_path_col}),
                        "error": str(exc),
                    }
                )
    return pd.DataFrame(records)


def tune_native_segmenters(
    tuning_manifest: pd.DataFrame,
    param_grids: Mapping[str, Sequence[Mapping[str, object]] | Mapping[str, Sequence[object]]],
    *,
    methods: Sequence[str] | None = None,
    score_col: str = "dice",
    higher_is_better: bool = True,
    image_col: str = "image_path",
    mask_col: str = "mask_path",
    fov_col: str | None = None,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    """Tune native segmenter parameters on a manifest of tuning rows.

    The raw run table and compact candidate summary are both returned so reports can
    show what was tuned before reporting held-out metrics.
    """
    method_names = list(methods or param_grids.keys())
    unknown = sorted({name for name in method_names if name not in NATIVE_SEGMENTERS})
    if unknown:
        msg = f"Unknown native segmentation method(s): {', '.join(unknown)}"
        raise ValueError(msg)
    missing_grids = sorted({name for name in method_names if name not in param_grids})
    if missing_grids:
        msg = f"Missing parameter grid(s): {', '.join(missing_grids)}"
        raise ValueError(msg)

    records = []
    for method_name in method_names:
        candidates = _expand_param_grid(param_grids[method_name])
        if not candidates:
            msg = f"Parameter grid for {method_name!r} is empty."
            raise ValueError(msg)
        for candidate_index, params in enumerate(candidates):
            table = benchmark_native_segmenters(
                tuning_manifest,
                methods=[method_name],
                image_col=image_col,
                mask_col=mask_col,
                fov_col=fov_col,
                method_params={method_name: params},
                max_rows=max_rows,
                continue_on_error=True,
            )
            table["requested_method"] = method_name
            table["candidate_index"] = candidate_index
            table["candidate_params_json"] = _to_json(params)
            records.append(table)

    tuning_table = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    tuning_summary = summarize_tuning_results(
        tuning_table,
        score_col=score_col,
        higher_is_better=higher_is_better,
    )
    best_params = {
        row.requested_method: json.loads(row.candidate_params_json)
        for row in tuning_summary.itertuples(index=False)
        if bool(row.selected)
    }
    return tuning_table, tuning_summary, best_params


def summarize_tuning_results(
    tuning_table: pd.DataFrame,
    *,
    score_col: str = "dice",
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Summarize candidate-level tuning runs and mark one selected row per method."""
    if tuning_table.empty:
        return pd.DataFrame()
    required = {"requested_method", "method", "candidate_index", "candidate_params_json", "error"}
    missing = sorted(required - set(tuning_table.columns))
    if missing:
        msg = f"Missing tuning columns: {', '.join(missing)}"
        raise ValueError(msg)
    if score_col not in tuning_table.columns:
        msg = f"Missing score column {score_col!r}."
        raise ValueError(msg)

    metric_cols = [
        score_col,
        "iou",
        "precision",
        "recall",
        "specificity",
        "abs_error_vessel_density",
        "abs_error_skeleton_length_density",
        "runtime_seconds",
    ]
    available_metrics = [col for col in metric_cols if col in tuning_table.columns]
    grouped_rows = []
    group_cols = ["requested_method", "method", "candidate_index", "candidate_params_json"]
    for group_key, group in tuning_table.groupby(group_cols, dropna=False, sort=False):
        successful = group.loc[group["error"].isna()]
        row = dict(zip(group_cols, group_key, strict=True))
        row["n_rows"] = len(group)
        row["n_success"] = len(successful)
        for metric in available_metrics:
            row[f"mean_{metric}"] = (
                float(successful[metric].mean()) if len(successful) else np.nan
            )
        grouped_rows.append(row)

    summary = pd.DataFrame(grouped_rows)
    summary["selected"] = False
    score_summary_col = f"mean_{score_col}"
    if score_summary_col not in summary.columns:
        return summary
    for _, method_rows in summary.groupby("requested_method", sort=False):
        candidates = method_rows.loc[method_rows["n_success"] > 0].copy()
        if candidates.empty:
            continue
        sort_cols = [score_summary_col, "candidate_index"]
        ascending = [not higher_is_better, True]
        if "mean_abs_error_vessel_density" in candidates.columns:
            sort_cols.insert(1, "mean_abs_error_vessel_density")
            ascending.insert(1, True)
        if "mean_runtime_seconds" in candidates.columns:
            sort_cols.insert(-1, "mean_runtime_seconds")
            ascending.insert(-1, True)
        candidates = candidates.sort_values(
            by=sort_cols,
            ascending=ascending,
            na_position="last",
        )
        summary.loc[candidates.index[0], "selected"] = True
    return summary


def _expand_param_grid(
    grid: Sequence[Mapping[str, object]] | Mapping[str, Sequence[object]],
) -> list[dict[str, object]]:
    if isinstance(grid, Mapping):
        keys = list(grid)
        value_lists = [_as_grid_values(grid[key]) for key in keys]
        return [dict(zip(keys, values, strict=True)) for values in product(*value_lists)]
    return [dict(candidate) for candidate in grid]


def _as_grid_values(value: object) -> list[object]:
    if isinstance(value, str | bytes):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        msg = f"Missing required manifest columns: {', '.join(missing)}"
        raise ValueError(msg)


def _benchmark_base_record(
    row: pd.Series, *, image_col: str, mask_col: str
) -> dict[str, object]:
    split = row.get("official_split", row.get("split", None))
    return {
        "dataset": row.get("dataset", None),
        "image_id": row.get("image_id", None),
        "subject_id": row.get("subject_id", None),
        "layer": row.get("layer", None),
        "split": split,
        "image_path": row[image_col],
        "manual_mask_path": row[mask_col],
    }


def _read_optional_mask(row: pd.Series, fov_col: str | None) -> np.ndarray | None:
    if fov_col is None:
        return None
    value = row[fov_col]
    if pd.isna(value):
        return None
    return ensure_grayscale(skio.imread(value)) > 0


def _coerce_prediction_path(value: object) -> Path:
    if pd.isna(value) or str(value).strip() == "":
        raise ValueError("prediction path is missing.")
    path = Path(str(value))
    if not path.exists():
        raise FileNotFoundError(f"prediction path does not exist: {path}")
    return path


def _read_score_map(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        score = np.load(path)
    else:
        score = ensure_grayscale(skio.imread(path))
    return _normalize_score(score)


def _normalize_score(score: np.ndarray) -> np.ndarray:
    arr = np.asarray(score, dtype=float)
    if arr.ndim != 2:
        msg = f"Expected a 2D score map, got shape {arr.shape}."
        raise ValueError(msg)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    clean = np.where(finite, arr, 0.0)
    min_value = float(clean[finite].min())
    max_value = float(clean[finite].max())
    if 0.0 <= min_value and max_value <= 1.0:
        return clean.astype(float, copy=False)
    if max_value <= min_value:
        return np.zeros_like(clean, dtype=float)
    return (clean - min_value) / (max_value - min_value)


def _write_benchmark_outputs(
    result: SegmentResult,
    *,
    output_path: Path | None,
    row_index: object,
    image_id: object,
) -> tuple[str | None, str | None]:
    if output_path is None:
        return None, None
    stem = _safe_stem(f"{row_index}_{image_id or 'image'}_{result.method}")
    pred_path = output_path / "masks" / f"{stem}.png"
    score_path = output_path / "scores" / f"{stem}.npy" if result.score is not None else None
    skio.imsave(pred_path, result.mask.astype(np.uint8) * 255, check_contrast=False)
    if score_path is not None:
        np.save(score_path, np.asarray(result.score, dtype=np.float32))
    return str(pred_path), str(score_path) if score_path is not None else None


def _write_score_postprocess_mask(
    mask: np.ndarray,
    *,
    output_path: Path | None,
    row_index: object,
    image_id: object,
    method: str,
    suffix: str,
) -> str | None:
    if output_path is None:
        return None
    stem = _safe_stem(f"{row_index}_{image_id or 'image'}_{method}{suffix}")
    pred_path = output_path / f"{stem}.png"
    skio.imsave(
        pred_path,
        np.asarray(mask, dtype=bool).astype(np.uint8) * 255,
        check_contrast=False,
    )
    return str(pred_path)


def _safe_stem(value: str) -> str:
    cleaned = [char if char.isalnum() or char in {"-", "_"} else "_" for char in value]
    return "".join(cleaned).strip("_") or "image"


def _to_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=_json_default)


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run native RetiVasc segmentation benchmarks.")
    parser.add_argument(
        "--manifest",
        required=True,
        help="CSV manifest with image_path and mask_path.",
    )
    parser.add_argument("--out", required=True, help="Output CSV benchmark table.")
    parser.add_argument(
        "--methods",
        default="frangi,diffusion,random_walker,geodesic",
        help="Comma-separated native methods.",
    )
    parser.add_argument("--output-root", help="Optional folder for masks and score maps.")
    parser.add_argument("--max-rows", type=int, help="Optional row limit for quick demos.")
    args = parser.parse_args(argv)

    manifest = pd.read_csv(args.manifest)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    benchmark = benchmark_native_segmenters(
        manifest,
        methods=methods,
        output_root=args.output_root,
        max_rows=args.max_rows,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    benchmark.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
