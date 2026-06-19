"""Mask and metadata artifact audits for ROSE exploratory analyses."""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from retivasc.embeddings import normalize_diagnosis
from retivasc.stats import compare_groups_featurewise

DEFAULT_ARTIFACT_NUMERIC_COLUMNS = (
    "mask_height",
    "mask_width",
    "mask_area_pixels",
    "foreground_fraction",
    "connected_component_count",
    "largest_component_fraction",
    "small_component_fraction",
    "skeleton_length_density",
    "edge_touch_fraction",
    "empty_border_fraction",
)


def audit_mask_artifacts(
    df: pd.DataFrame,
    numeric_cols: Sequence[str] | None = None,
    *,
    n_boot: int = 1000,
    n_perm: int = 1000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Audit whether diagnosis is associated with QC or acquisition proxies."""
    work = df.copy()
    work["diagnosis"] = work["diagnosis"].map(normalize_diagnosis)
    if "mask_area_pixels" not in work.columns and {"mask_height", "mask_width"} <= set(
        work.columns
    ):
        work["mask_area_pixels"] = pd.to_numeric(
            work["mask_height"], errors="coerce"
        ) * pd.to_numeric(work["mask_width"], errors="coerce")
    if numeric_cols is None:
        numeric_cols = [col for col in DEFAULT_ARTIFACT_NUMERIC_COLUMNS if col in work.columns]

    rows = []
    if numeric_cols:
        numeric = compare_groups_featurewise(
            work,
            list(numeric_cols),
            n_boot=n_boot,
            n_perm=n_perm,
            random_state=random_state,
        )
        numeric = numeric.rename(columns={"feature": "variable"})
        numeric.insert(1, "variable_type", "numeric")
        numeric["test"] = "AD-control numeric comparison"
        rows.extend(numeric.to_dict(orient="records"))

    for col in ("official_split", "split", "layer"):
        if col in work.columns:
            rows.append(_categorical_association_row(work, col))

    layer_availability = _subject_layer_availability(work)
    if not layer_availability.empty:
        availability = compare_groups_featurewise(
            layer_availability,
            ["n_layers_available"],
            n_boot=n_boot,
            n_perm=n_perm,
            random_state=random_state + 13,
        )
        availability = availability.rename(columns={"feature": "variable"})
        availability.insert(1, "variable_type", "numeric")
        availability["test"] = "subject layer availability"
        rows.extend(availability.to_dict(orient="records"))
        rows.append(_categorical_association_row(layer_availability, "available_layer_set"))

    return pd.DataFrame(rows)


def _categorical_association_row(df: pd.DataFrame, column: str) -> dict[str, object]:
    subset = df.loc[df["diagnosis"].isin(["AD", "control"]), ["diagnosis", column]].dropna()
    base = {
        "variable": column,
        "variable_type": "categorical",
        "test": "chi-square association",
        "n_AD": int((subset["diagnosis"] == "AD").sum()),
        "n_control": int((subset["diagnosis"] == "control").sum()),
        "mean_AD": np.nan,
        "mean_control": np.nan,
        "median_AD": np.nan,
        "median_control": np.nan,
        "diff_mean_AD_minus_control": np.nan,
        "diff_median_AD_minus_control": np.nan,
        "cohens_d": np.nan,
        "hedges_g": np.nan,
        "rank_biserial_correlation": np.nan,
        "bootstrap_CI_low": np.nan,
        "bootstrap_CI_high": np.nan,
        "permutation_p": np.nan,
        "mannwhitney_p": np.nan,
        "ks_p": np.nan,
        "fdr_bh_permutation_p": np.nan,
        "fdr_bh_mannwhitney_p": np.nan,
        "levels": "",
    }
    if subset.empty or subset["diagnosis"].nunique() < 2 or subset[column].nunique() < 2:
        base["levels"] = json.dumps({})
        return base
    contingency = pd.crosstab(subset["diagnosis"], subset[column])
    try:
        chi2, p_value, dof, expected = scipy_stats.chi2_contingency(contingency)
        base.update(
            {
                "chi2": float(chi2),
                "categorical_p": float(p_value),
                "categorical_dof": int(dof),
                "min_expected_count": float(np.min(expected)),
            }
        )
    except ValueError as exc:
        base["categorical_warning"] = str(exc)
    base["levels"] = json.dumps(contingency.to_dict(), sort_keys=True)
    return base


def _subject_layer_availability(df: pd.DataFrame) -> pd.DataFrame:
    if not {"subject_id", "diagnosis", "layer"} <= set(df.columns):
        return pd.DataFrame()
    rows = []
    for subject_id, group in df.groupby("subject_id", sort=True):
        diagnoses = sorted(set(group["diagnosis"].dropna()) - {"unknown"})
        diagnosis = (
            diagnoses[0] if len(diagnoses) == 1 else ("unknown" if not diagnoses else "conflict")
        )
        layers = sorted(str(layer) for layer in group["layer"].dropna().unique())
        rows.append(
            {
                "subject_id": subject_id,
                "diagnosis": diagnosis,
                "n_layers_available": len(layers),
                "available_layer_set": "+".join(layers),
            }
        )
    return pd.DataFrame(rows)
