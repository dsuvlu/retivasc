"""Outlier scoring and sensitivity helpers for ROSE mask-derived features."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def compute_outlier_scores(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Score feature outliers within comparable groups, usually OCTA layer."""
    if group_cols is None:
        group_cols = ["layer"]
    rows = []
    group_iter = df.groupby(group_cols, dropna=False, sort=False) if group_cols else [((), df)]
    for _group_key, group in group_iter:
        group_values = group[list(feature_cols)].apply(pd.to_numeric, errors="coerce")
        zscores = _robust_zscores(group_values)
        pca_leverage = _pca_leverage(group_values)
        threshold = _leverage_threshold(pca_leverage)
        distances = np.sqrt(np.nansum(np.square(zscores.to_numpy(dtype=float)), axis=1))
        distances = distances / np.sqrt(max(1, len(feature_cols)))
        for local_idx, (_, original_row) in enumerate(group.iterrows()):
            zrow = zscores.iloc[local_idx]
            abs_z = np.abs(zrow.to_numpy(dtype=float))
            max_abs = float(np.nanmax(abs_z)) if np.isfinite(abs_z).any() else 0.0
            mean_abs = float(np.nanmean(abs_z)) if np.isfinite(abs_z).any() else 0.0
            extreme_count = int(np.nansum(abs_z > 3.0))
            score_row = {
                "subject_id": original_row.get("subject_id", None),
                "image_id": original_row.get("image_id", None),
                "layer": original_row.get("layer", None),
                "diagnosis": original_row.get("diagnosis", None),
                "mask_path": original_row.get("mask_path", None),
                "max_abs_zscore": max_abs,
                "mean_abs_zscore": mean_abs,
                "robust_mahalanobis_distance": float(distances[local_idx]),
                "pca_leverage_score": float(pca_leverage[local_idx]),
                "pca_leverage_threshold": float(threshold),
                "n_extreme_features_abs_z_gt_3": extreme_count,
                "top_extreme_features": _top_feature_zscores(zrow),
            }
            for col in group_cols:
                score_row[f"group_{col}"] = original_row.get(col, None)
            for feature in feature_cols:
                score_row[f"z__{feature}"] = (
                    float(zrow[feature]) if np.isfinite(zrow[feature]) else np.nan
                )
            score_row["is_feature_outlier"] = bool(
                max_abs > 3.0 or pca_leverage[local_idx] > threshold
            )
            rows.append(score_row)
    return pd.DataFrame(rows)


def compute_outlier_sensitivity(
    full_effects: pd.DataFrame,
    filtered_effects: pd.DataFrame,
    *,
    key_cols: Sequence[str] = ("analysis", "layer", "feature"),
    effect_col: str = "diff_median_AD_minus_control",
    p_col: str = "permutation_p",
) -> pd.DataFrame:
    """Compare effect tables before and after removing flagged outliers."""
    available_keys = [
        col for col in key_cols if col in full_effects.columns and col in filtered_effects.columns
    ]
    if "feature" not in available_keys:
        available_keys.append("feature")
    merged = full_effects.merge(
        filtered_effects,
        on=available_keys,
        how="left",
        suffixes=("_full", "_without_outliers"),
    )
    full_name = f"{effect_col}_full"
    filtered_name = f"{effect_col}_without_outliers"
    p_full = f"{p_col}_full"
    p_filtered = f"{p_col}_without_outliers"
    if full_name not in merged.columns and effect_col in merged.columns:
        merged = merged.rename(columns={effect_col: full_name})
    merged["effect_full"] = pd.to_numeric(merged.get(full_name), errors="coerce")
    merged["effect_without_outliers"] = pd.to_numeric(merged.get(filtered_name), errors="coerce")
    merged["delta_effect"] = merged["effect_without_outliers"] - merged["effect_full"]
    merged["p_full"] = pd.to_numeric(merged.get(p_full), errors="coerce")
    merged["p_without_outliers"] = pd.to_numeric(merged.get(p_filtered), errors="coerce")
    merged["interpretation_flag"] = [
        _sensitivity_flag(full, filtered)
        for full, filtered in zip(
            merged["effect_full"].to_numpy(dtype=float),
            merged["effect_without_outliers"].to_numpy(dtype=float),
            strict=False,
        )
    ]
    keep = list(available_keys) + [
        "effect_full",
        "effect_without_outliers",
        "delta_effect",
        "p_full",
        "p_without_outliers",
        "interpretation_flag",
    ]
    return merged[keep]


def _robust_zscores(values: pd.DataFrame) -> pd.DataFrame:
    numeric = values.apply(pd.to_numeric, errors="coerce")
    medians = numeric.median(axis=0, skipna=True)
    filled = numeric.fillna(medians).fillna(0.0)
    mad = (filled - medians).abs().median(axis=0, skipna=True) * 1.4826
    std = filled.std(axis=0, ddof=1).replace(0, np.nan)
    scale = mad.where(mad > 0, std).replace(0, np.nan).fillna(1.0)
    return (filled - medians) / scale


def _pca_leverage(values: pd.DataFrame) -> np.ndarray:
    numeric = values.apply(pd.to_numeric, errors="coerce")
    if len(numeric) < 3 or numeric.shape[1] == 0:
        return np.zeros(len(numeric), dtype=float)
    zscores = _robust_zscores(numeric).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n_components = min(2, zscores.shape[1], zscores.shape[0] - 1)
    if n_components < 1:
        return np.zeros(len(numeric), dtype=float)
    try:
        coords = PCA(n_components=n_components, random_state=0).fit_transform(zscores)
    except ValueError:
        return np.zeros(len(numeric), dtype=float)
    center = np.median(coords, axis=0)
    return np.sum(np.square(coords - center), axis=1)


def _leverage_threshold(leverage: np.ndarray) -> float:
    finite = leverage[np.isfinite(leverage)]
    if finite.size == 0:
        return 0.0
    if finite.size < 5:
        return float(np.nanmax(finite) + 1.0)
    return float(max(3.0, np.nanpercentile(finite, 95)))


def _top_feature_zscores(zrow: pd.Series, *, top_n: int = 5) -> str:
    ranked = zrow.abs().sort_values(ascending=False).head(top_n)
    return ";".join(f"{feature}:{float(zrow[feature]):.2f}" for feature in ranked.index)


def _sensitivity_flag(full: float, filtered: float) -> str:
    if not np.isfinite(full) or not np.isfinite(filtered):
        return "too_few_samples"
    if np.sign(full) != np.sign(filtered) and abs(full) > 0:
        return "outlier_sensitive"
    if abs(filtered - full) > 0.5 * max(abs(full), 1e-12):
        return "outlier_sensitive"
    return "stable"
