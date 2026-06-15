"""Matplotlib figure helpers for the demo notebooks."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from retivasc.preprocess import ensure_grayscale, normalize_image


def _prepare_out_path(out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _mask_overlay(
    image: np.ndarray, mask: np.ndarray, color: tuple[float, float, float]
) -> np.ndarray:
    gray = normalize_image(ensure_grayscale(image))
    rgb = np.repeat(gray[..., None], 3, axis=2)
    mask_bool = np.asarray(mask, dtype=bool)
    overlay = rgb.copy()
    for channel, value in enumerate(color):
        overlay[..., channel] = np.where(
            mask_bool, 0.65 * value + 0.35 * rgb[..., channel], rgb[..., channel]
        )
    return overlay


def plot_rose_pipeline_panel(
    image: np.ndarray,
    manual_mask: np.ndarray,
    predicted_mask: np.ndarray,
    skeleton: np.ndarray,
    out_path: str | Path,
):
    """Save a raw/manual/predicted/skeleton ROSE pipeline panel and return the figure."""
    out = _prepare_out_path(out_path)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), constrained_layout=True)

    axes[0].imshow(normalize_image(ensure_grayscale(image)), cmap="gray")
    axes[0].set_title("Raw OCTA")
    axes[1].imshow(np.asarray(manual_mask, dtype=bool), cmap="gray")
    axes[1].set_title("Manual annotation")
    axes[2].imshow(_mask_overlay(image, predicted_mask, (1.0, 0.2, 0.1)))
    axes[2].set_title("Classical baseline overlay")
    axes[3].imshow(np.asarray(skeleton, dtype=bool), cmap="magma")
    axes[3].set_title("Manual-mask skeleton")

    for axis in axes:
        axis.set_axis_off()

    fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def plot_feature_distributions(
    features_df: pd.DataFrame,
    feature_names: list[str],
    label_col: str,
    out_path: str | Path,
):
    """Save grouped feature distributions and return the figure."""
    if features_df.empty:
        raise ValueError("features_df is empty; cannot plot feature distributions.")
    missing = [feature for feature in feature_names if feature not in features_df.columns]
    if missing:
        msg = f"Missing feature columns: {', '.join(missing)}"
        raise ValueError(msg)
    if label_col not in features_df.columns:
        msg = f"Missing label column {label_col!r}."
        raise ValueError(msg)

    out = _prepare_out_path(out_path)
    labels = [label for label in sorted(features_df[label_col].dropna().unique())]
    fig, axes = plt.subplots(1, len(feature_names), figsize=(4 * len(feature_names), 3.8))
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)

    for axis, feature in zip(axes, feature_names, strict=True):
        values = [
            features_df.loc[features_df[label_col] == label, feature].dropna().to_numpy()
            for label in labels
        ]
        axis.boxplot(values, labels=labels, showfliers=False)
        for idx, vals in enumerate(values, start=1):
            if vals.size:
                jitter = rng.normal(0, 0.035, size=vals.size)
                axis.scatter(np.full(vals.size, idx) + jitter, vals, s=16, alpha=0.75)
        axis.set_title(feature.replace("_", " "))
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle("Exploratory ROSE manual-mask vascular features", y=1.04)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_path: str | Path,
    *,
    title: str = "Calibration",
    metrics: dict[str, float | int | str | tuple[float, float]] | None = None,
):
    """Save a calibration curve with Brier score and return the figure."""
    true = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    if true.shape != prob.shape:
        msg = f"Shape mismatch: y_true {true.shape}, y_prob {prob.shape}."
        raise ValueError(msg)
    if len(np.unique(true)) != 2:
        raise ValueError("Calibration requires exactly two classes in y_true.")

    out = _prepare_out_path(out_path)
    frac_pos, mean_pred = calibration_curve(true, prob, n_bins=8, strategy="quantile")
    brier = brier_score_loss(true, prob)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.1), constrained_layout=True)
    axes[0].plot([0, 1], [0, 1], color="0.55", linestyle="--", linewidth=1, label="Ideal")
    axes[0].plot(mean_pred, frac_pos, marker="o", linewidth=1.8, label="Observed")
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Observed fraction")
    axes[0].set_title(title)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper left", frameon=False)

    axes[1].hist(
        prob[true == 0],
        bins=np.linspace(0, 1, 11),
        color="#4c78a8",
        edgecolor="white",
        alpha=0.75,
        label="Normal",
    )
    axes[1].hist(
        prob[true == 1],
        bins=np.linspace(0, 1, 11),
        color="#f58518",
        edgecolor="white",
        alpha=0.7,
        label="Disease",
    )
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Prediction distribution")
    axes[1].legend(loc="upper center", frameon=False, ncol=2)

    metric_values = {"Brier": brier}
    if metrics:
        metric_values.update(metrics)
    metric_lines = []
    for key, value in metric_values.items():
        if isinstance(value, float):
            metric_lines.append(f"{key}: {value:.3f}")
        elif isinstance(value, tuple) and len(value) == 2:
            metric_lines.append(f"{key}: {value[0]:.3f}-{value[1]:.3f}")
        else:
            metric_lines.append(f"{key}: {value}")
    axes[1].text(
        0.98,
        0.96,
        "\n".join(metric_lines),
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75"},
    )

    fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def plot_cross_species_roadmap(out_path: str | Path):
    """Save a species-agnostic retinal vascular feature roadmap schematic."""
    out = _prepare_out_path(out_path)
    fig, axis = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    axis.set_axis_off()

    boxes = [
        ("human", 0.05, 0.68, 0.25, 0.16, "Human retina\nOCTA / fundus"),
        ("mouse", 0.05, 0.34, 0.25, 0.16, "Mouse retina\nMthfr677C>T"),
        ("pipeline", 0.38, 0.51, 0.26, 0.18, "Vessel mask + skeleton\nsame pipeline"),
        (
            "features",
            0.33,
            0.17,
            0.40,
            0.18,
            "Shared feature vector\ndensity, branching,\nfractal dimension, tortuosity",
        ),
        (
            "context",
            0.78,
            0.16,
            0.18,
            0.31,
            "Align with\np-tau217, Abeta ratio,\nGFAP, NfL, genomic,\nand clinical context",
        ),
    ]

    for _key, x, y, width, height, text in boxes:
        axis.add_patch(
            plt.Rectangle(
                (x, y),
                width,
                height,
                facecolor="#f7f7f2",
                edgecolor="#30343f",
                linewidth=1.4,
            )
        )
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=10)

    arrows = [
        ((0.30, 0.76), (0.38, 0.61)),
        ((0.30, 0.42), (0.38, 0.56)),
        ((0.51, 0.51), (0.51, 0.35)),
        ((0.73, 0.26), (0.78, 0.30)),
    ]
    for start, end in arrows:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "linewidth": 1.7, "color": "#30343f"},
        )

    caption = (
        "The extraction functions are species-agnostic: the same definitions of density, "
        "branching, fractal dimension, and tortuosity can be applied to human OCTA/fundus "
        "images and mouse retinal images, enabling cross-species comparison once Roux/JAX "
        "data are available."
    )
    axis.text(0.05, 0.06, caption, ha="left", va="top", fontsize=9, wrap=True)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def plot_data_audit_flow(out_path: str | Path):
    """Save a code-aware end-to-end data audit schematic."""
    out = _prepare_out_path(out_path)
    fig, axis = plt.subplots(figsize=(15.5, 9.3), constrained_layout=True)
    axis.set_axis_off()

    rows = [
        (
            "1",
            "Local Raw Data",
            "data/raw/fives with official train/test Original and Ground truth folders",
            "Raw medical images and masks stay local and gitignored.",
            "src/retivasc/io.py\nload_fives_manifest",
        ),
        (
            "2",
            "Manifest Audit",
            "800 rows with image_path, mask_path, label, official_split, split_group",
            "Pair fundus images to manual vessel masks and preserve official split metadata.",
            "notebooks/02_fives_modeling_calibration_demo.py\nsrc/retivasc/io.py",
        ),
        (
            "3",
            "Mask Preprocessing",
            "Manual vessel masks loaded from PNG files",
            "Convert masks to grayscale and downsample each mask to max dimension 512.",
            "src/retivasc/preprocess.py\nensure_grayscale\nresize_mask_to_max_dim",
        ),
        (
            "4",
            "Vascular Features",
            "One feature vector per fundus image",
            "Compute vessel density, skeleton length, branchpoint density, "
            "fractal dimension, and components.",
            "src/retivasc/features.py\nextract_vascular_features\nsrc/retivasc/skeleton.py",
        ),
        (
            "5",
            "Cached Feature Table",
            "data/interim/fives_features_max512.parquet",
            "Persist derived features so reruns do not reprocess every manual mask.",
            "notebooks/02_fives_modeling_calibration_demo.py\npandas.to_parquet",
        ),
        (
            "6",
            "Modeling Split",
            "Official split: 600 train rows, 200 test rows",
            "Build disease-vs-normal target and verify split separation before fitting.",
            "src/retivasc/splits.py\nassert_group_split_safe\nsklearn Pipeline",
        ),
        (
            "7",
            "Calibration Outputs",
            "figures/fives_calibration_demo.png and reports/fives_metrics.json",
            "Fit logistic model; compute AUROC, AUPRC, Brier score, and bootstrap CIs.",
            "src/retivasc/plotting.py\nplot_calibration\nsklearn.metrics",
        ),
        (
            "8",
            "PI Report",
            "reports/retivasc_pi_demo.html",
            "Embed only derived figures, metrics, caveats, audit trail, and roadmap.",
            "notebooks/03_pi_demo_report.py\nsrc/retivasc/plotting.py\nplot_data_audit_flow",
        ),
    ]

    axis.text(
        0.03,
        0.955,
        "End-to-End Data Audit: FIVES Processing Trail",
        ha="left",
        va="center",
        fontsize=15,
        weight="bold",
    )
    axis.text(
        0.03,
        0.918,
        "Top-to-bottom trace of each data state, transformation, and owning code path.",
        ha="left",
        va="center",
        fontsize=9.5,
        color="#37474f",
    )

    table_left = 0.03
    table_top = 0.875
    table_bottom = 0.105
    header_h = 0.06
    col_widths = [0.16, 0.26, 0.30, 0.23]
    col_lefts = [table_left]
    for width in col_widths[:-1]:
        col_lefts.append(col_lefts[-1] + width)

    headers = ["Stage", "Data / Artifact", "Operation", "Code Location"]
    header_color = "#263238"
    row_colors = ["#f8fbff", "#f6f8f4"]
    line_color = "#c7d1d8"

    for x, width, header in zip(col_lefts, col_widths, headers, strict=True):
        axis.add_patch(
            plt.Rectangle(
                (x, table_top - header_h),
                width,
                header_h,
                facecolor=header_color,
                edgecolor="white",
                linewidth=1,
            )
        )
        axis.text(
            x + 0.01,
            table_top - header_h / 2,
            header,
            ha="left",
            va="center",
            fontsize=9,
            weight="bold",
            color="white",
        )

    row_h = (table_top - header_h - table_bottom) / len(rows)
    wrap_widths = [18, 34, 42, 30]
    text_color = "#263238"

    for idx, (number, stage, artifact, operation, code_ref) in enumerate(rows):
        y_top = table_top - header_h - idx * row_h
        y = y_top - row_h
        facecolor = row_colors[idx % 2]
        for x, width in zip(col_lefts, col_widths, strict=True):
            axis.add_patch(
                plt.Rectangle(
                    (x, y),
                    width,
                    row_h,
                    facecolor=facecolor,
                    edgecolor=line_color,
                    linewidth=0.8,
                )
            )

        stage_text = f"{number}. {stage}"
        cell_values = [
            fill(stage_text, width=wrap_widths[0]),
            fill(artifact, width=wrap_widths[1]),
            fill(operation, width=wrap_widths[2]),
            fill(code_ref, width=wrap_widths[3]),
        ]
        font_sizes = [8.2, 7.5, 7.5, 6.9]
        families = ["sans-serif", "sans-serif", "sans-serif", "monospace"]

        for col_idx, (value, font_size, family) in enumerate(
            zip(cell_values, font_sizes, families, strict=True)
        ):
            axis.text(
                col_lefts[col_idx] + 0.01,
                y_top - 0.014,
                value,
                ha="left",
                va="top",
                fontsize=font_size,
                family=family,
                color=text_color,
                linespacing=1.16,
            )

    axis.text(
        0.03,
        0.055,
        (
            "Audit note: raw medical images remain under gitignored data/raw/. "
            "The report consumes derived figures, a cached feature table, and metrics JSON; "
            "no synthetic ADRD or plasma-biomarker results are generated."
        ),
        ha="left",
        va="center",
        fontsize=9,
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig
