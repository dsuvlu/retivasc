"""Statistical helpers for exploratory ROSE layer-aware feature analysis."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from retivasc.embeddings import LAYER_ORDER, normalize_diagnosis

ALLOWED_DIAGNOSES = ("AD", "control", "unknown")

METADATA_COLUMNS = {
    "dataset",
    "subject_id",
    "image_id",
    "eye_id",
    "layer",
    "diagnosis",
    "label",
    "label_source",
    "mask_path",
    "image_path",
    "split",
    "official_split",
    "split_group",
    "source_row_count",
    "scanner",
    "field_of_view",
    "pixel_size_um",
    "quality_flag",
    "missing_layers",
    "n_layers_available",
    "is_feature_outlier",
}


def validate_rose_feature_table(df: pd.DataFrame) -> None:
    """Validate the long ROSE subject-layer feature table.

    The function intentionally does not infer labels. A diagnosis column must already
    be present, and non-empty labels are normalized only to AD/control/unknown.
    """
    required = {"subject_id", "image_id", "layer", "diagnosis", "label_source", "mask_path"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required ROSE feature table columns: {', '.join(missing)}")
    if df["subject_id"].isna().any():
        raise ValueError("subject_id contains null values.")

    layers = set(df["layer"].dropna().astype(str))
    invalid_layers = sorted(layers - set(LAYER_ORDER))
    if invalid_layers:
        raise ValueError(f"Unexpected ROSE layer values: {', '.join(invalid_layers)}")

    diagnoses = df["diagnosis"].map(normalize_diagnosis)
    invalid_diagnoses = sorted(set(diagnoses.dropna()) - set(ALLOWED_DIAGNOSES))
    if invalid_diagnoses:
        raise ValueError(f"Unexpected diagnosis values: {', '.join(invalid_diagnoses)}")

    duplicated = df.duplicated(["subject_id", "layer"], keep=False)
    if duplicated.any():
        examples = (
            df.loc[duplicated, ["subject_id", "layer"]]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(f"Each subject can have at most one row per layer. Examples: {examples}")

    labels_by_subject = (
        df.assign(_diagnosis=diagnoses)
        .loc[lambda frame: frame["_diagnosis"] != "unknown"]
        .groupby("subject_id")["_diagnosis"]
        .nunique()
    )
    conflicts = labels_by_subject[labels_by_subject > 1]
    if not conflicts.empty:
        raise ValueError(
            "Subjects have conflicting diagnosis labels: "
            + ", ".join(str(value) for value in conflicts.index[:10])
        )

    feature_cols = infer_feature_columns(df)
    if not feature_cols:
        raise ValueError("No numeric feature columns were found.")
    non_numeric = [
        col
        for col in feature_cols
        if not pd.api.types.is_numeric_dtype(pd.to_numeric(df[col], errors="coerce"))
    ]
    if non_numeric:
        raise ValueError(f"Feature columns must be numeric: {', '.join(non_numeric)}")
    all_nan = [col for col in feature_cols if pd.to_numeric(df[col], errors="coerce").isna().all()]
    if all_nan:
        raise ValueError(f"Feature columns are entirely NaN: {', '.join(all_nan)}")


def infer_feature_columns(
    df: pd.DataFrame,
    *,
    include_qc: bool = False,
    extra_exclude: Iterable[str] = (),
) -> list[str]:
    """Infer numeric vascular feature columns from a ROSE feature table."""
    exclude = set(METADATA_COLUMNS) | set(extra_exclude)
    if not include_qc:
        exclude |= {
            "mask_height",
            "mask_width",
            "mask_area_pixels",
            "foreground_fraction",
            "edge_touch_fraction",
            "empty_border_fraction",
        }
    cols: list[str] = []
    for col in df.columns:
        if col in exclude or col.startswith("_"):
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            cols.append(col)
    return cols


def summarize_rose_feature_table(
    df: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
) -> dict[str, object]:
    """Return compact table-level QC metadata for the ROSE feature table."""
    if feature_cols is None:
        feature_cols = infer_feature_columns(df)
    diagnosis_by_subject = (
        df[["subject_id", "diagnosis"]]
        .drop_duplicates()
        .assign(diagnosis=lambda frame: frame["diagnosis"].map(normalize_diagnosis))
    )
    numeric = (
        df[list(feature_cols)].apply(pd.to_numeric, errors="coerce")
        if feature_cols
        else pd.DataFrame()
    )
    return {
        "n_subjects": int(df["subject_id"].nunique()) if "subject_id" in df else 0,
        "n_rows": int(len(df)),
        "subject_counts_by_diagnosis": {
            str(key): int(value)
            for key, value in diagnosis_by_subject["diagnosis"].value_counts(dropna=False).items()
        }
        if not diagnosis_by_subject.empty
        else {},
        "rows_per_layer": {
            str(key): int(value)
            for key, value in df.get("layer", pd.Series(dtype=str)).value_counts().items()
        },
        "missingness_per_feature": {
            col: float(numeric[col].isna().mean()) for col in numeric.columns
        },
        "variance_per_feature": {
            col: float(numeric[col].var(ddof=1)) if numeric[col].notna().sum() > 1 else 0.0
            for col in numeric.columns
        },
    }


def bootstrap_difference(
    x: np.ndarray,
    y: np.ndarray,
    statistic: str = "median",
    n_boot: int = 5000,
    random_state: int = 0,
) -> tuple[float, float]:
    """Return a percentile 95% bootstrap CI for statistic(x) - statistic(y)."""
    x_clean = _drop_nan(x)
    y_clean = _drop_nan(y)
    if x_clean.size == 0 or y_clean.size == 0:
        return (float("nan"), float("nan"))
    stat_func = _statistic_func(statistic)
    rng = np.random.default_rng(random_state)
    diffs = np.empty(int(n_boot), dtype=float)
    for idx in range(int(n_boot)):
        x_boot = rng.choice(x_clean, size=x_clean.size, replace=True)
        y_boot = rng.choice(y_clean, size=y_clean.size, replace=True)
        diffs[idx] = stat_func(x_boot) - stat_func(y_boot)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low), float(high)


def permutation_test_difference(
    x: np.ndarray,
    y: np.ndarray,
    statistic: str = "median",
    n_perm: int = 10000,
    random_state: int = 0,
) -> float:
    """Two-sided permutation p-value for statistic(x) - statistic(y)."""
    x_clean = _drop_nan(x)
    y_clean = _drop_nan(y)
    if x_clean.size == 0 or y_clean.size == 0:
        return float("nan")
    stat_func = _statistic_func(statistic)
    observed = stat_func(x_clean) - stat_func(y_clean)
    combined = np.concatenate([x_clean, y_clean])
    n_x = x_clean.size
    rng = np.random.default_rng(random_state)
    extreme = 0
    for _ in range(int(n_perm)):
        shuffled = rng.permutation(combined)
        null_diff = stat_func(shuffled[:n_x]) - stat_func(shuffled[n_x:])
        if abs(null_diff) >= abs(observed):
            extreme += 1
    return float((extreme + 1) / (int(n_perm) + 1))


def compare_groups_featurewise(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str = "diagnosis",
    group_a: str = "AD",
    group_b: str = "control",
    n_boot: int = 5000,
    n_perm: int = 10000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Compare AD/control feature distributions with exploratory effect summaries."""
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")
    labels = df[group_col].map(normalize_diagnosis)
    rows = []
    for idx, feature in enumerate(feature_cols):
        values = pd.to_numeric(df[feature], errors="coerce")
        x = values.loc[labels == group_a].to_numpy(dtype=float)
        y = values.loc[labels == group_b].to_numpy(dtype=float)
        x = _drop_nan(x)
        y = _drop_nan(y)
        seed = int(random_state) + idx * 7919
        row = _compare_one_feature(
            feature,
            x,
            y,
            group_a=group_a,
            group_b=group_b,
            n_boot=n_boot,
            n_perm=n_perm,
            random_state=seed,
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_bh_permutation_p"] = fdr_bh(out["permutation_p"].to_numpy())
        out["fdr_bh_mannwhitney_p"] = fdr_bh(out["mannwhitney_p"].to_numpy())
    return out


def fdr_bh(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction with NaN preservation."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return adjusted
    valid_p = p[valid]
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    n = ranked.size
    scaled = ranked * n / np.arange(1, n + 1)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    clipped = np.clip(monotone, 0.0, 1.0)
    valid_adjusted = np.empty_like(clipped)
    valid_adjusted[order] = clipped
    adjusted[valid] = valid_adjusted
    return adjusted


def fit_layer_mixed_effects(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Fit exploratory repeated-measures models when statsmodels is available."""
    rows = []
    try:
        import statsmodels.formula.api as smf
    except ModuleNotFoundError as exc:
        return pd.DataFrame(
            [
                {
                    "feature": feature,
                    "n_subjects": int(df["subject_id"].nunique()) if "subject_id" in df else 0,
                    "n_rows": int(len(df)),
                    "converged": False,
                    "coef_diagnosis_AD": np.nan,
                    "p_diagnosis_AD": np.nan,
                    "coef_layer_DVC": np.nan,
                    "p_layer_DVC": np.nan,
                    "coef_layer_SVCplusDVC": np.nan,
                    "p_layer_SVCplusDVC": np.nan,
                    "coef_diagnosis_AD_x_layer_DVC": np.nan,
                    "p_interaction_DVC": np.nan,
                    "coef_diagnosis_AD_x_layer_SVCplusDVC": np.nan,
                    "p_interaction_SVCplusDVC": np.nan,
                    "model_warning": f"statsmodels unavailable: {exc}",
                }
                for feature in feature_cols
            ]
        )

    model_df = df.copy()
    model_df["diagnosis"] = model_df["diagnosis"].map(normalize_diagnosis)
    model_df = model_df.loc[model_df["diagnosis"].isin(["AD", "control"])].copy()
    model_df["layer_model"] = (
        model_df["layer"].astype("string").str.replace("+", "plus", regex=False)
    )
    for feature in feature_cols:
        feature_df = model_df[["subject_id", "diagnosis", "layer_model", feature]].copy()
        feature_df[feature] = pd.to_numeric(feature_df[feature], errors="coerce")
        feature_df = feature_df.dropna()
        base_row = {
            "feature": feature,
            "n_subjects": int(feature_df["subject_id"].nunique()) if not feature_df.empty else 0,
            "n_rows": int(len(feature_df)),
            "converged": False,
            "coef_diagnosis_AD": np.nan,
            "p_diagnosis_AD": np.nan,
            "coef_layer_DVC": np.nan,
            "p_layer_DVC": np.nan,
            "coef_layer_SVCplusDVC": np.nan,
            "p_layer_SVCplusDVC": np.nan,
            "coef_diagnosis_AD_x_layer_DVC": np.nan,
            "p_interaction_DVC": np.nan,
            "coef_diagnosis_AD_x_layer_SVCplusDVC": np.nan,
            "p_interaction_SVCplusDVC": np.nan,
            "model_warning": "",
        }
        if feature_df["subject_id"].nunique() < 3 or feature_df["diagnosis"].nunique() < 2:
            base_row["model_warning"] = "too few subjects or diagnosis groups"
            rows.append(base_row)
            continue
        work_col = "_feature_value"
        feature_df = feature_df.rename(columns={feature: work_col})
        try:
            model = smf.mixedlm(
                f"{work_col} ~ C(diagnosis, Treatment(reference='control'))"
                " * C(layer_model, Treatment(reference='SVC'))",
                feature_df,
                groups=feature_df["subject_id"],
            )
            result = model.fit(reml=False, method="lbfgs", disp=False)
            params = result.params
            pvalues = result.pvalues
            base_row.update(
                {
                    "converged": bool(getattr(result, "converged", False)),
                    "coef_diagnosis_AD": _lookup_param(
                        params, "C(diagnosis, Treatment(reference='control'))[T.AD]"
                    ),
                    "p_diagnosis_AD": _lookup_param(
                        pvalues, "C(diagnosis, Treatment(reference='control'))[T.AD]"
                    ),
                    "coef_layer_DVC": _lookup_param(
                        params, "C(layer_model, Treatment(reference='SVC'))[T.DVC]"
                    ),
                    "p_layer_DVC": _lookup_param(
                        pvalues, "C(layer_model, Treatment(reference='SVC'))[T.DVC]"
                    ),
                    "coef_layer_SVCplusDVC": _lookup_param(
                        params, "C(layer_model, Treatment(reference='SVC'))[T.SVCplusDVC]"
                    ),
                    "p_layer_SVCplusDVC": _lookup_param(
                        pvalues, "C(layer_model, Treatment(reference='SVC'))[T.SVCplusDVC]"
                    ),
                    "coef_diagnosis_AD_x_layer_DVC": _lookup_param_contains(
                        params, ["diagnosis", "T.AD", "layer_model", "T.DVC"]
                    ),
                    "p_interaction_DVC": _lookup_param_contains(
                        pvalues, ["diagnosis", "T.AD", "layer_model", "T.DVC"]
                    ),
                    "coef_diagnosis_AD_x_layer_SVCplusDVC": _lookup_param_contains(
                        params, ["diagnosis", "T.AD", "layer_model", "T.SVCplusDVC"]
                    ),
                    "p_interaction_SVCplusDVC": _lookup_param_contains(
                        pvalues, ["diagnosis", "T.AD", "layer_model", "T.SVCplusDVC"]
                    ),
                }
            )
        except Exception as exc:
            base_row["model_warning"] = str(exc)
        rows.append(base_row)
    return pd.DataFrame(rows)


def _compare_one_feature(
    feature: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    group_a: str,
    group_b: str,
    n_boot: int,
    n_perm: int,
    random_state: int,
) -> dict[str, float | int | str]:
    n_x = int(x.size)
    n_y = int(y.size)
    mean_x = _safe_stat(x, np.mean)
    mean_y = _safe_stat(y, np.mean)
    median_x = _safe_stat(x, np.median)
    median_y = _safe_stat(y, np.median)
    ci_low, ci_high = bootstrap_difference(
        x, y, statistic="median", n_boot=n_boot, random_state=random_state
    )
    perm_p = permutation_test_difference(
        x, y, statistic="median", n_perm=n_perm, random_state=random_state + 17
    )
    mann_p = float("nan")
    rank_biserial = float("nan")
    ks_p = float("nan")
    if n_x > 0 and n_y > 0:
        try:
            mann = scipy_stats.mannwhitneyu(x, y, alternative="two-sided")
            mann_p = float(mann.pvalue)
            rank_biserial = float(2 * mann.statistic / (n_x * n_y) - 1)
        except ValueError:
            pass
        try:
            ks_p = float(scipy_stats.ks_2samp(x, y).pvalue)
        except ValueError:
            pass
    return {
        "feature": feature,
        f"n_{group_a}": n_x,
        f"n_{group_b}": n_y,
        f"mean_{group_a}": mean_x,
        f"mean_{group_b}": mean_y,
        f"median_{group_a}": median_x,
        f"median_{group_b}": median_y,
        "diff_mean_AD_minus_control": mean_x - mean_y,
        "diff_median_AD_minus_control": median_x - median_y,
        "cohens_d": _cohens_d(x, y),
        "hedges_g": _hedges_g(x, y),
        "rank_biserial_correlation": rank_biserial,
        "bootstrap_CI_low": ci_low,
        "bootstrap_CI_high": ci_high,
        "permutation_p": perm_p,
        "mannwhitney_p": mann_p,
        "ks_p": ks_p,
    }


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    pooled_num = (x.size - 1) * np.var(x, ddof=1) + (y.size - 1) * np.var(y, ddof=1)
    pooled_den = x.size + y.size - 2
    if pooled_den <= 0:
        return float("nan")
    pooled = np.sqrt(pooled_num / pooled_den)
    if pooled == 0:
        return 0.0 if np.mean(x) == np.mean(y) else float("nan")
    return float((np.mean(x) - np.mean(y)) / pooled)


def _hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    d = _cohens_d(x, y)
    if not np.isfinite(d):
        return d
    df = x.size + y.size - 2
    if df <= 1:
        return d
    correction = 1 - 3 / (4 * df - 1)
    return float(d * correction)


def _drop_nan(values: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _statistic_func(statistic: str):
    if statistic == "mean":
        return lambda arr: float(np.mean(arr))
    if statistic == "median":
        return lambda arr: float(np.median(arr))
    raise ValueError("statistic must be 'mean' or 'median'.")


def _safe_stat(values: np.ndarray, func) -> float:
    if values.size == 0:
        return float("nan")
    return float(func(values))


def _lookup_param(series: pd.Series, key: str) -> float:
    return float(series[key]) if key in series.index else float("nan")


def _lookup_param_contains(series: pd.Series, tokens: Sequence[str]) -> float:
    for key, value in series.items():
        if all(token in str(key) for token in tokens):
            return float(value)
    return float("nan")
