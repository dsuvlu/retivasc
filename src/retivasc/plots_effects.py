"""Effect-size, PCA, QC, and outlier figures for ROSE layer-aware statistics."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import measure, morphology
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from retivasc.embeddings import (
    LAYER_ORDER,
    compute_group_separation,
    load_binary_mask,
    normalize_diagnosis,
)
from retivasc.plots_embeddings import DIAGNOSIS_COLORS

CORE_FEATURE_LABELS = (
    "vessel_area_fraction",
    "skeleton_length_density",
    "branchpoint_density",
    "fractal_dimension_boxcount",
    "orientation_entropy",
    "small_component_fraction",
    "mean_tortuosity_arc_chord",
)


def plot_subject_level_pca(
    subject_df: pd.DataFrame,
    feature_cols: list[str],
    out: str | Path,
) -> tuple[plt.Figure, dict[str, object]]:
    """Plot one-dot-per-subject PCA and return variance/loading metadata."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if subject_df.empty or len(feature_cols) < 1 or len(subject_df) < 2:
        fig = _placeholder("Subject-level PCA unavailable", "Too few subject rows.", out_path)
        return fig, {"skipped": "too few rows"}

    values = subject_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    values = values.fillna(values.median(axis=0, skipna=True)).fillna(0.0)
    scaled = RobustScaler().fit_transform(values.to_numpy(dtype=float))
    n_components = min(2, scaled.shape[0] - 1, scaled.shape[1])
    if n_components < 1:
        fig = _placeholder("Subject-level PCA unavailable", "No usable numeric features.", out_path)
        return fig, {"skipped": "no usable numeric features"}
    pca = PCA(n_components=n_components, random_state=0)
    coords = pca.fit_transform(scaled)
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])
    plot_df = subject_df[["subject_id", "diagnosis"]].copy()
    plot_df["diagnosis"] = plot_df["diagnosis"].map(normalize_diagnosis)
    plot_df["component_1"] = coords[:, 0]
    plot_df["component_2"] = coords[:, 1]
    if "is_feature_outlier" in subject_df.columns:
        plot_df["is_feature_outlier"] = subject_df["is_feature_outlier"].fillna(False).astype(bool)
    else:
        plot_df["is_feature_outlier"] = False

    fig, axis = plt.subplots(figsize=(7.4, 5.3))
    for diagnosis, color in DIAGNOSIS_COLORS.items():
        for is_outlier, marker in ((False, "o"), (True, "X")):
            subset = plot_df.loc[
                (plot_df["diagnosis"] == diagnosis) & (plot_df["is_feature_outlier"] == is_outlier)
            ]
            if subset.empty:
                continue
            axis.scatter(
                subset["component_1"],
                subset["component_2"],
                s=80 if is_outlier else 54,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                alpha=0.9,
                label=f"{diagnosis}{' outlier' if is_outlier else ''}",
            )
    for _, row in plot_df.loc[plot_df["is_feature_outlier"]].iterrows():
        axis.text(row["component_1"], row["component_2"], str(row["subject_id"]), fontsize=8)
    evr = pca.explained_variance_ratio_
    axis.set_xlabel(f"PC1 ({_safe_percent(evr, 0)})")
    axis.set_ylabel(f"PC2 ({_safe_percent(evr, 1)})")
    axis.grid(True, color="#d9e1e5", linewidth=0.7, alpha=0.8)
    axis.set_title("ROSE subject-level PCA from aggregated mask features")
    axis.legend(frameon=False, loc="best")
    fig.text(
        0.5,
        0.01,
        fill(
            "One dot is one subject. Features combine layer summaries and paired layer "
            "contrasts. Exploratory visualization only; not disease classification.",
            width=110,
        ),
        ha="center",
        fontsize=9,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"PC{idx + 1}" for idx in range(pca.components_.shape[0])],
    )
    separation = compute_group_separation(plot_df, n_permutations=500)
    metadata = {
        "explained_variance_ratio": [float(value) for value in evr],
        "explained_variance_ratio_pc1": float(evr[0]) if len(evr) > 0 else None,
        "explained_variance_ratio_pc2": float(evr[1]) if len(evr) > 1 else None,
        "top_pc1_loadings": _top_loadings(loadings, "PC1"),
        "top_pc2_loadings": _top_loadings(loadings, "PC2") if "PC2" in loadings else [],
        "separation": separation,
    }
    return fig, metadata


def plot_layer_effect_sizes(
    effect_df: pd.DataFrame,
    out: str | Path,
    *,
    top_n: int = 15,
) -> plt.Figure:
    """Plot layer-faceted AD/control median differences with bootstrap CIs."""
    out_path = Path(out)
    selected = _select_effect_rows(effect_df, top_n=top_n, require_layer=True)
    if selected.empty:
        return _placeholder("Layer effect sizes unavailable", "No effect rows.", out_path)
    features = _ordered_features(selected)
    fig, axes = plt.subplots(
        1, len(LAYER_ORDER), figsize=(13.5, max(4.8, 0.34 * len(features))), sharey=True
    )
    for axis, layer in zip(axes, LAYER_ORDER, strict=True):
        layer_rows = selected.loc[selected["layer"].astype("string") == layer].set_index("feature")
        _draw_effect_axis(axis, layer_rows, features, title=layer)
    axes[0].set_yticks(np.arange(len(features)))
    axes[0].set_yticklabels([_short_feature_label(feature) for feature in features])
    for axis in axes[1:]:
        axis.tick_params(labelleft=False)
    fig.suptitle("ROSE AD-control effect sizes by OCTA layer", y=1.01)
    fig.text(
        0.5,
        0.01,
        "Points show median AD-control differences; bars show bootstrap 95% CIs. Exploratory only.",
        ha="center",
        fontsize=9,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


def plot_layer_contrast_effect_sizes(
    effect_df: pd.DataFrame,
    out: str | Path,
    *,
    top_n: int = 18,
) -> plt.Figure:
    """Plot top paired layer-contrast AD/control effects."""
    out_path = Path(out)
    selected = _select_effect_rows(effect_df, top_n=top_n, require_layer=False)
    if selected.empty:
        return _placeholder("Layer contrast effects unavailable", "No effect rows.", out_path)
    selected = selected.sort_values("diff_median_AD_minus_control")
    y = np.arange(len(selected))
    fig, axis = plt.subplots(figsize=(8.5, max(4.8, 0.35 * len(selected))))
    colors = _effect_colors(selected)
    x = selected["diff_median_AD_minus_control"].to_numpy(dtype=float)
    low = selected["bootstrap_CI_low"].to_numpy(dtype=float)
    high = selected["bootstrap_CI_high"].to_numpy(dtype=float)
    xerr = np.vstack([np.abs(x - low), np.abs(high - x)])
    axis.errorbar(x, y, xerr=xerr, fmt="none", ecolor="0.35", elinewidth=0.9, alpha=0.8)
    axis.scatter(x, y, color=colors, s=52, edgecolor="white", linewidth=0.6, zorder=3)
    axis.axvline(0, color="0.2", linewidth=0.8)
    axis.set_yticks(y)
    axis.set_yticklabels([_short_feature_label(feature) for feature in selected["feature"]])
    axis.set_xlabel("Median AD-control difference")
    axis.set_title("Paired layer-contrast effects")
    axis.grid(True, axis="x", color="#d9e1e5", linewidth=0.7, alpha=0.8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


def plot_outlier_audit_panel(
    outlier_scores: pd.DataFrame,
    out: str | Path,
    *,
    top_k: int = 4,
) -> plt.Figure:
    """Create a compact mask/skeleton/component/z-score panel for top outliers."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if outlier_scores.empty:
        return _placeholder("Outlier audit unavailable", "No outlier rows.", out_path)
    ranked = outlier_scores.sort_values(
        ["is_feature_outlier", "max_abs_zscore", "pca_leverage_score"],
        ascending=[False, False, False],
    ).head(top_k)
    fig, axes = plt.subplots(len(ranked), 4, figsize=(13.5, 3.0 * len(ranked)), squeeze=False)
    for row_idx, (_, row) in enumerate(ranked.iterrows()):
        title = (
            f"{row.get('subject_id', '')} {row.get('layer', '')} | "
            f"max z {row.get('max_abs_zscore', np.nan):.2f}"
        )
        mask = _load_mask_or_none(row.get("mask_path", None))
        if mask is None:
            for col_idx in range(3):
                axes[row_idx, col_idx].text(0.5, 0.5, "mask unavailable", ha="center", va="center")
                axes[row_idx, col_idx].set_axis_off()
        else:
            skeleton = morphology.skeletonize(mask)
            components = measure.label(mask, connectivity=2)
            axes[row_idx, 0].imshow(mask, cmap="gray")
            axes[row_idx, 0].set_title("mask")
            axes[row_idx, 1].imshow(skeleton, cmap="gray")
            axes[row_idx, 1].set_title("skeleton")
            axes[row_idx, 2].imshow(components, cmap="tab20")
            axes[row_idx, 2].set_title("components")
            for col_idx in range(3):
                axes[row_idx, col_idx].set_xticks([])
                axes[row_idx, col_idx].set_yticks([])
        axes[row_idx, 0].set_ylabel(title, fontsize=9)
        _draw_zscore_bar(axes[row_idx, 3], row)
    fig.suptitle("ROSE mask outlier audit", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


def plot_feature_qc_heatmap(
    summary: dict[str, object],
    out: str | Path,
    *,
    top_n: int = 20,
) -> plt.Figure:
    """Plot feature missingness and variance as a small QC heatmap."""
    out_path = Path(out)
    missing = pd.Series(summary.get("missingness_per_feature", {}), dtype=float)
    variance = pd.Series(summary.get("variance_per_feature", {}), dtype=float)
    if missing.empty:
        return _placeholder("Feature QC unavailable", "No feature summary.", out_path)
    score = variance.rank(pct=True).fillna(0) + missing.rank(pct=True).fillna(0)
    features = score.sort_values(ascending=False).head(top_n).index.tolist()
    matrix = pd.DataFrame(
        {
            "missingness": missing.reindex(features).fillna(0.0),
            "variance_rank": variance.rank(pct=True).reindex(features).fillna(0.0),
        }
    )
    fig, axis = plt.subplots(figsize=(5.6, max(4.0, 0.28 * len(features))))
    image = axis.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axis.set_xticks(np.arange(matrix.shape[1]))
    axis.set_xticklabels(["Missing", "Variance rank"])
    axis.set_yticks(np.arange(len(features)))
    axis.set_yticklabels([_short_feature_label(feature) for feature in features])
    axis.set_title("Feature QC heatmap")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


def plot_mask_artifact_audit(
    artifact_df: pd.DataFrame,
    out: str | Path,
    *,
    top_n: int = 12,
) -> plt.Figure:
    """Plot the strongest QC/artifact associations with diagnosis."""
    out_path = Path(out)
    if artifact_df.empty:
        return _placeholder("Mask artifact audit unavailable", "No artifact rows.", out_path)
    work = artifact_df.copy()
    work["display_score"] = pd.to_numeric(work.get("hedges_g"), errors="coerce").abs()
    categorical_p = pd.to_numeric(work.get("categorical_p"), errors="coerce")
    work.loc[categorical_p.notna(), "display_score"] = -np.log10(categorical_p.clip(lower=1e-12))
    work = (
        work.sort_values("display_score", ascending=False).head(top_n).sort_values("display_score")
    )
    fig, axis = plt.subplots(figsize=(7.8, max(4.2, 0.34 * len(work))))
    colors = np.where(work["variable_type"].astype(str) == "categorical", "#7c3aed", "#1b7f79")
    axis.barh(np.arange(len(work)), work["display_score"], color=colors)
    axis.set_yticks(np.arange(len(work)))
    axis.set_yticklabels([_short_feature_label(value) for value in work["variable"]])
    axis.set_xlabel("|Hedges g| or -log10(categorical p)")
    axis.set_title("Mask and metadata artifact audit")
    axis.grid(True, axis="x", color="#d9e1e5", linewidth=0.7, alpha=0.8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


def _draw_effect_axis(axis, layer_rows: pd.DataFrame, features: list[str], *, title: str) -> None:
    y = np.arange(len(features))
    aligned = layer_rows.reindex(features)
    x = pd.to_numeric(aligned["diff_median_AD_minus_control"], errors="coerce").to_numpy(
        dtype=float
    )
    low = pd.to_numeric(aligned["bootstrap_CI_low"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(aligned["bootstrap_CI_high"], errors="coerce").to_numpy(dtype=float)
    colors = _effect_colors(aligned.reset_index())
    valid = np.isfinite(x)
    xerr = np.vstack([np.abs(x - low), np.abs(high - x)])
    axis.errorbar(
        x[valid],
        y[valid],
        xerr=xerr[:, valid],
        fmt="none",
        ecolor="0.35",
        elinewidth=0.9,
        alpha=0.8,
    )
    axis.scatter(
        x[valid],
        y[valid],
        color=np.asarray(colors)[valid],
        s=46,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    axis.axvline(0, color="0.2", linewidth=0.8)
    axis.set_title(title)
    axis.set_xlabel("Median AD-control difference")
    axis.grid(True, axis="x", color="#d9e1e5", linewidth=0.7, alpha=0.8)
    axis.set_ylim(-0.7, len(features) - 0.3)


def _select_effect_rows(
    effect_df: pd.DataFrame, *, top_n: int, require_layer: bool
) -> pd.DataFrame:
    if effect_df.empty or "feature" not in effect_df.columns:
        return pd.DataFrame()
    work = effect_df.copy()
    work["abs_effect"] = pd.to_numeric(work.get("hedges_g"), errors="coerce").abs()
    work["abs_effect"] = work["abs_effect"].fillna(
        pd.to_numeric(work.get("diff_median_AD_minus_control"), errors="coerce").abs()
    )
    top_features = (
        work.groupby("feature")["abs_effect"].max().sort_values(ascending=False).head(top_n).index
    )
    core = [feature for feature in CORE_FEATURE_LABELS if feature in set(work["feature"])]
    selected_features = list(dict.fromkeys([*core, *top_features]))
    selected = work.loc[work["feature"].isin(selected_features)].copy()
    if require_layer and "layer" not in selected.columns:
        return pd.DataFrame()
    return selected


def _ordered_features(effect_df: pd.DataFrame) -> list[str]:
    order = (
        effect_df.groupby("feature")["abs_effect"].max().sort_values(ascending=True).index.tolist()
        if "abs_effect" in effect_df
        else sorted(effect_df["feature"].unique())
    )
    return order


def _effect_colors(effect_df: pd.DataFrame) -> list[str]:
    fdr = pd.to_numeric(effect_df.get("fdr_bh_permutation_p"), errors="coerce")
    flags = effect_df.get("interpretation_flag", pd.Series(["stable"] * len(effect_df)))
    colors = []
    for p_value, flag in zip(fdr, flags, strict=False):
        if str(flag) == "outlier_sensitive":
            colors.append("#d97706")
        elif np.isfinite(p_value) and p_value < 0.1:
            colors.append("#c73e1d")
        else:
            colors.append("#1b7f79")
    return colors


def _draw_zscore_bar(axis, row: pd.Series, *, top_n: int = 7) -> None:
    z = {
        col.replace("z__", ""): float(value)
        for col, value in row.items()
        if str(col).startswith("z__") and np.isfinite(float(value))
    }
    if not z:
        axis.text(0.5, 0.5, "no z-scores", ha="center", va="center")
        axis.set_axis_off()
        return
    series = pd.Series(z).sort_values(key=lambda values: values.abs(), ascending=False).head(top_n)
    series = series.sort_values()
    colors = np.where(series >= 0, "#c73e1d", "#1b7f79")
    axis.barh(np.arange(len(series)), series.to_numpy(dtype=float), color=colors)
    axis.axvline(0, color="0.2", linewidth=0.8)
    axis.set_yticks(np.arange(len(series)))
    axis.set_yticklabels([_short_feature_label(feature) for feature in series.index], fontsize=8)
    axis.set_xlabel("Robust z")
    axis.set_title("top feature z-scores")
    axis.grid(True, axis="x", color="#d9e1e5", linewidth=0.7, alpha=0.8)


def _load_mask_or_none(path) -> np.ndarray | None:
    if path is None or pd.isna(path):
        return None
    try:
        return load_binary_mask(path)
    except Exception:
        return None


def _placeholder(title: str, message: str, out: Path) -> plt.Figure:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.0, 3.2))
    axis.text(0.5, 0.58, title, ha="center", va="center", fontsize=14)
    axis.text(0.5, 0.4, message, ha="center", va="center", fontsize=10, color="0.35")
    axis.set_axis_off()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def _top_loadings(
    loadings: pd.DataFrame, component: str, *, n: int = 8
) -> list[dict[str, float | str]]:
    if component not in loadings:
        return []
    selected = loadings[component].abs().sort_values(ascending=False).head(n).index
    return [
        {"feature": str(feature), "loading": float(loadings.loc[feature, component])}
        for feature in selected
    ]


def _short_feature_label(feature: object) -> str:
    text = str(feature)
    replacements = {
        "vessel_area_fraction": "vessel area",
        "skeleton_length_density": "skeleton density",
        "branchpoint_density": "branchpoint density",
        "fractal_dimension_boxcount": "fractal dimension",
        "orientation_entropy": "orientation entropy",
        "small_component_fraction": "small components",
        "mean_tortuosity_arc_chord": "mean tortuosity",
        "connected_component_count": "components",
        "largest_component_fraction": "largest component",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    text = text.replace("SVCplusDVC", "SVC+DVC")
    text = text.replace("_", " ")
    return fill(text, width=34)


def _safe_percent(values: np.ndarray, idx: int) -> str:
    if len(values) <= idx:
        return "0.0%"
    return f"{float(values[idx]):.1%}"
