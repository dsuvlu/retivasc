"""Subject-level aggregation and paired layer contrasts for ROSE features."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from retivasc.embeddings import LAYER_ORDER, normalize_diagnosis
from retivasc.stats import infer_feature_columns


def aggregate_subject_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate subject-layer rows to one row per subject without imputing layers."""
    if feature_cols is None:
        feature_cols = infer_feature_columns(df)
    rows = []
    for subject_id, group in df.groupby("subject_id", sort=True):
        diagnoses = sorted(set(group["diagnosis"].map(normalize_diagnosis).dropna()) - {"unknown"})
        diagnosis = (
            diagnoses[0] if len(diagnoses) == 1 else ("unknown" if not diagnoses else "conflict")
        )
        available_layers = sorted(set(group["layer"].dropna().astype(str)))
        missing_layers = [layer for layer in LAYER_ORDER if layer not in available_layers]
        row: dict[str, object] = {
            "subject_id": subject_id,
            "diagnosis": diagnosis,
            "label_source": _join_unique(group.get("label_source", pd.Series(dtype=object))),
            "n_layers_available": int(len(available_layers)),
            "missing_layers": ",".join(missing_layers),
        }
        if "split_group" in group.columns:
            row["split_group"] = _join_unique(group["split_group"])
        if "official_split" in group.columns:
            row["official_split"] = _join_unique(group["official_split"])
        numeric = group[list(feature_cols)].apply(pd.to_numeric, errors="coerce")
        for feature in feature_cols:
            values = numeric[feature]
            row[f"mean_{feature}"] = (
                float(values.mean(skipna=True)) if values.notna().any() else np.nan
            )
            row[f"min_{feature}"] = (
                float(values.min(skipna=True)) if values.notna().any() else np.nan
            )
            row[f"max_{feature}"] = (
                float(values.max(skipna=True)) if values.notna().any() else np.nan
            )
            row[f"range_{feature}"] = (
                float(values.max(skipna=True) - values.min(skipna=True))
                if values.notna().any()
                else np.nan
            )
            row[f"std_{feature}"] = (
                float(values.std(skipna=True, ddof=1)) if values.notna().sum() > 1 else 0.0
            )
        for layer in LAYER_ORDER:
            layer_rows = group.loc[group["layer"].astype("string") == layer]
            if layer_rows.empty:
                for feature in feature_cols:
                    row[f"{layer}_{feature}"] = np.nan
                continue
            layer_row = layer_rows.iloc[0]
            for feature in feature_cols:
                row[f"{layer}_{feature}"] = pd.to_numeric(
                    pd.Series([layer_row[feature]]), errors="coerce"
                ).iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_layer_contrasts(
    df: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    *,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """Compute paired SVC/DVC/SVC+DVC differences, ratios, and log-ratios."""
    if feature_cols is None:
        feature_cols = infer_feature_columns(df)
    rows = []
    for subject_id, group in df.groupby("subject_id", sort=True):
        diagnoses = sorted(set(group["diagnosis"].map(normalize_diagnosis).dropna()) - {"unknown"})
        diagnosis = (
            diagnoses[0] if len(diagnoses) == 1 else ("unknown" if not diagnoses else "conflict")
        )
        row: dict[str, object] = {
            "subject_id": subject_id,
            "diagnosis": diagnosis,
            "label_source": _join_unique(group.get("label_source", pd.Series(dtype=object))),
        }
        if "split_group" in group.columns:
            row["split_group"] = _join_unique(group["split_group"])
        layer_values = {
            layer: _layer_feature_values(group, layer, feature_cols) for layer in LAYER_ORDER
        }
        pairs = (
            ("DVC_minus_SVC", "DVC", "SVC"),
            ("SVCplusDVC_minus_SVC", "SVC+DVC", "SVC"),
            ("SVCplusDVC_minus_DVC", "SVC+DVC", "DVC"),
        )
        ratio_pairs = (
            ("DVC_div_SVC", "DVC_logratio_SVC", "DVC", "SVC"),
            ("SVCplusDVC_div_SVC", "SVCplusDVC_logratio_SVC", "SVC+DVC", "SVC"),
            ("SVCplusDVC_div_DVC", "SVCplusDVC_logratio_DVC", "SVC+DVC", "DVC"),
        )
        for prefix, numerator_layer, denominator_layer in pairs:
            for feature in feature_cols:
                numerator = layer_values[numerator_layer].get(feature, np.nan)
                denominator = layer_values[denominator_layer].get(feature, np.nan)
                row[f"{prefix}_{feature}"] = numerator - denominator
        for ratio_prefix, log_prefix, numerator_layer, denominator_layer in ratio_pairs:
            for feature in feature_cols:
                numerator = layer_values[numerator_layer].get(feature, np.nan)
                denominator = layer_values[denominator_layer].get(feature, np.nan)
                row[f"{ratio_prefix}_{feature}"] = _safe_ratio(numerator, denominator, eps)
                row[f"{log_prefix}_{feature}"] = _safe_logratio(numerator, denominator, eps)
        rows.append(row)
    return pd.DataFrame(rows)


def _layer_feature_values(
    group: pd.DataFrame,
    layer: str,
    feature_cols: Sequence[str],
) -> dict[str, float]:
    layer_rows = group.loc[group["layer"].astype("string") == layer]
    if layer_rows.empty:
        return {feature: np.nan for feature in feature_cols}
    row = layer_rows.iloc[0]
    return {
        feature: float(pd.to_numeric(pd.Series([row[feature]]), errors="coerce").iloc[0])
        for feature in feature_cols
    }


def _safe_ratio(numerator: float, denominator: float, eps: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    return float(numerator / (denominator + eps))


def _safe_logratio(numerator: float, denominator: float, eps: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if numerator + eps <= 0 or denominator + eps <= 0:
        return float("nan")
    return float(np.log((numerator + eps) / (denominator + eps)))


def _join_unique(values: pd.Series) -> str:
    unique = [str(value) for value in values.dropna().unique() if str(value)]
    return ";".join(sorted(unique))
