"""Plot helpers for mask-derived vascular embeddings."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retivasc.embeddings import LAYER_ORDER, normalize_diagnosis

DIAGNOSIS_COLORS = {
    "AD": "#c73e1d",
    "control": "#1b7f79",
    "unknown": "#8d99ae",
}


def plot_layer_faceted_embedding(
    coords: pd.DataFrame,
    x_col: str,
    y_col: str,
    layer_col: str = "layer",
    label_col: str = "diagnosis",
    layers: tuple[str, ...] = LAYER_ORDER,
    title: str = "",
    out_path: str | Path | None = None,
):
    """Create a 1x3 embedding figure with one dot per subject-layer row."""
    required = {x_col, y_col, layer_col, label_col}
    missing = sorted(required - set(coords.columns))
    if missing:
        raise ValueError(f"Missing embedding columns: {', '.join(missing)}")

    fig, axes = plt.subplots(
        1,
        len(layers),
        figsize=(4.2 * len(layers), 4.0),
        sharex=True,
        sharey=True,
    )
    if len(layers) == 1:
        axes = [axes]
    for axis, layer in zip(axes, layers, strict=True):
        layer_rows = coords.loc[coords[layer_col].astype("string") == layer].copy()
        if layer_rows.empty:
            axis.text(0.5, 0.5, "No rows", ha="center", va="center", transform=axis.transAxes)
        for diagnosis, color in DIAGNOSIS_COLORS.items():
            labels = layer_rows[label_col].map(normalize_diagnosis)
            subset = layer_rows.loc[labels == diagnosis]
            if subset.empty:
                continue
            axis.scatter(
                subset[x_col],
                subset[y_col],
                s=46,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                alpha=0.9,
                label=diagnosis,
            )
        axis.set_title(layer)
        axis.set_xlabel(x_col.replace("_", " ").title())
        axis.grid(True, color="#d9e1e5", linewidth=0.7, alpha=0.8)
    axes[0].set_ylabel(y_col.replace("_", " ").title())

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=max(1, len(labels)),
            frameon=False,
        )
    caption = (
        "Mask-derived feature embedding. One dot is one subject in one OCTA layer. "
        "Coordinates are fit jointly across layer-specific rows and then faceted by layer. "
        "Exploratory visualization only; not predictive validation."
    )
    fig.suptitle(title or "ROSE-1 mask-derived embedding", y=1.02)
    fig.text(
        0.5,
        0.02,
        fill(caption, width=130),
        ha="center",
        va="bottom",
        fontsize=9,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.9))
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def plot_pca_feature_loadings(
    pca_metadata: dict,
    out_path: str | Path | None = None,
    *,
    top_n: int = 8,
):
    """Plot top positive/negative feature loadings for PC1 and PC2."""
    loadings = pd.DataFrame(pca_metadata.get("loadings", []))
    if loadings.empty or "feature" not in loadings.columns:
        raise ValueError("pca_metadata does not contain feature loadings.")
    components = [component for component in ("PC1", "PC2") if component in loadings.columns]
    if not components:
        raise ValueError("pca_metadata does not contain PC1 or PC2 loadings.")

    fig, axes = plt.subplots(
        1,
        len(components),
        figsize=(5.4 * len(components), 4.4),
        squeeze=False,
    )
    for axis, component in zip(axes.ravel(), components, strict=True):
        selected = _top_signed_loadings(loadings[["feature", component]], component, top_n=top_n)
        colors = np.where(selected[component] >= 0, "#1b7f79", "#c73e1d")
        axis.barh(selected["feature"], selected[component], color=colors)
        axis.axvline(0, color="0.25", linewidth=0.8)
        axis.set_title(component)
        axis.set_xlabel("Loading")
        axis.invert_yaxis()
        axis.grid(True, axis="x", color="#d9e1e5", linewidth=0.7, alpha=0.8)
    fig.suptitle("PCA feature loadings", y=1.02)
    fig.tight_layout()
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def _top_signed_loadings(loadings: pd.DataFrame, component: str, *, top_n: int) -> pd.DataFrame:
    positive = loadings.sort_values(component, ascending=False).head(top_n // 2)
    negative = loadings.sort_values(component, ascending=True).head(top_n - len(positive))
    selected = pd.concat([positive, negative], ignore_index=True)
    return selected.sort_values(component, ascending=True)
