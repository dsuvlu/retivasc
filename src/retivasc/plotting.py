"""Matplotlib figure helpers for the demo notebooks."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io as skio
from skimage import measure, transform
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from retivasc.features import extract_vascular_features
from retivasc.preprocess import ensure_grayscale, normalize_image, resize_mask_to_max_dim
from retivasc.segment import classical_vesselness_mask
from retivasc.skeleton import branchpoint_mask, skeletonize_mask


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


def _format_feature_value(feature: str, value: float) -> str:
    if feature == "vessel_density":
        return f"{value:.1%}"
    if feature == "connected_component_count":
        return f"{value:.0f}"
    if feature.endswith("_density"):
        return f"{value:.4f}"
    return f"{value:.3f}"


def _segment_arc_chord_endpoints(coords: np.ndarray):
    coord_set = {tuple(int(value) for value in coord) for coord in coords}
    forward_neighbors = ((0, 1), (1, -1), (1, 0), (1, 1))
    all_neighbors = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    arc = 0.0
    endpoints = []
    for row, col in coord_set:
        neighbor_count = sum(
            (row + d_row, col + d_col) in coord_set for d_row, d_col in all_neighbors
        )
        if neighbor_count <= 1:
            endpoints.append((row, col))
        for d_row, d_col in forward_neighbors:
            if (row + d_row, col + d_col) in coord_set:
                arc += float(np.hypot(d_row, d_col))

    candidates = np.asarray(endpoints if len(endpoints) >= 2 else coords, dtype=float)
    if candidates.shape[0] < 2:
        return arc, 0.0, None
    deltas = candidates[:, None, :] - candidates[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    start_idx, end_idx = np.unravel_index(int(np.argmax(distances)), distances.shape)
    chord = float(distances[start_idx, end_idx])
    return arc, chord, (candidates[start_idx], candidates[end_idx])


def _most_tortuous_segment(
    skeleton: np.ndarray, *, min_segment_length: int = 5, min_chord: float = 0.0
):
    segment_labels = measure.label(skeleton & ~branchpoint_mask(skeleton), connectivity=2)
    best_ratio = 0.0
    best_mask = np.zeros_like(skeleton, dtype=bool)
    best_endpoints = None

    for region in measure.regionprops(segment_labels):
        coords = region.coords
        if coords.shape[0] < min_segment_length:
            continue
        arc, chord, endpoints = _segment_arc_chord_endpoints(coords)
        if chord <= min_chord:
            continue
        ratio = arc / chord
        if ratio > best_ratio:
            best_ratio = float(ratio)
            best_mask = segment_labels == region.label
            best_endpoints = endpoints

    return best_mask, best_endpoints, best_ratio


def _boxcount_grid(mask: np.ndarray) -> tuple[int, np.ndarray]:
    min_dim = min(mask.shape)
    if min_dim <= 0:
        return 1, np.zeros((1, 1), dtype=bool)
    max_power = int(np.floor(np.log2(min_dim)))
    size = int(2 ** max(3, max_power - 3))
    size = max(4, min(size, min_dim))
    rows = int(np.ceil(mask.shape[0] / size) * size)
    cols = int(np.ceil(mask.shape[1] / size) * size)
    padded = np.zeros((rows, cols), dtype=bool)
    padded[: mask.shape[0], : mask.shape[1]] = mask
    occupied = padded.reshape(rows // size, size, cols // size, size).any(axis=(1, 3))
    return size, occupied


def _draw_boxcount_visual(axis, mask: np.ndarray, *, title: str) -> None:
    axis.imshow(mask, cmap="gray", interpolation="nearest")
    size, occupied = _boxcount_grid(mask)
    for box_row, box_col in np.argwhere(occupied):
        axis.add_patch(
            plt.Rectangle(
                (box_col * size - 0.5, box_row * size - 0.5),
                size,
                size,
                edgecolor="#00a6d6",
                facecolor="none",
                linewidth=0.7,
                alpha=0.75,
            )
        )
    axis.set_xlim(-0.5, mask.shape[1] - 0.5)
    axis.set_ylim(mask.shape[0] - 0.5, -0.5)
    axis.set_title(title)


def plot_rose_feature_visuals(
    image: np.ndarray,
    manual_mask: np.ndarray,
    out_path: str | Path,
    *,
    max_dim: int = 512,
):
    """Save feature-specific ROSE visuals from one OCTA image and manual mask."""
    out = _prepare_out_path(out_path)
    image = _resize_image_to_max_dim(image, max_dim)
    mask = resize_mask_to_max_dim(ensure_grayscale(manual_mask) > 0, max_dim)
    gray = normalize_image(ensure_grayscale(image))
    skeleton = skeletonize_mask(mask)
    branches = branchpoint_mask(skeleton)
    branch_labels = measure.label(branches, connectivity=2)
    branch_centroids = np.asarray(
        [region.centroid for region in measure.regionprops(branch_labels)]
    )
    component_labels = measure.label(mask, connectivity=2)
    component_display = np.ma.masked_where(component_labels == 0, component_labels)
    min_tortuosity_chord = max(8.0, 0.03 * min(skeleton.shape))
    min_tortuosity_length = max(10, int(0.04 * min(skeleton.shape)))
    tortuous_segment, tortuous_endpoints, tortuous_value = _most_tortuous_segment(
        skeleton,
        min_segment_length=min_tortuosity_length,
        min_chord=min_tortuosity_chord,
    )
    features = extract_vascular_features(mask)
    value_text = {name: _format_feature_value(name, value) for name, value in features.items()}

    fig = plt.figure(figsize=(13, 10.5), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=[0.9, 1.0, 1.0])
    reference_axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
    ]
    visual_axes = [
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
        fig.add_subplot(grid[2, 0]),
        fig.add_subplot(grid[2, 1]),
        fig.add_subplot(grid[2, 2]),
    ]

    reference_axes[0].imshow(gray, cmap="gray")
    reference_axes[0].set_title("Original ROSE OCTA")
    reference_axes[1].imshow(mask, cmap="gray", interpolation="nearest")
    reference_axes[1].set_title("Manual vessel mask")

    feature_lines = [
        f"Vessel density: {value_text['vessel_density']}",
        f"Skeleton length density: {value_text['skeleton_length_density']}",
        f"Branchpoint density: {value_text['branchpoint_density']}",
        f"Fractal dimension: {value_text['fractal_dimension_boxcount']}",
        f"Mean segment tortuosity: {value_text['mean_segment_tortuosity']}",
        f"Connected components: {value_text['connected_component_count']}",
    ]
    reference_axes[2].text(
        0.0,
        0.98,
        "Feature values\n\n" + "\n".join(feature_lines),
        ha="left",
        va="top",
        fontsize=10,
        transform=reference_axes[2].transAxes,
    )
    reference_axes[2].text(
        0.0,
        0.08,
        fill("Values come from the manual mask for this one image.", width=40),
        ha="left",
        va="bottom",
        color="0.35",
        fontsize=9,
        transform=reference_axes[2].transAxes,
    )

    visual_axes[0].imshow(_mask_overlay(image, mask, (0.0, 0.75, 0.85)))
    visual_axes[0].set_title(
        "Vessel density\n"
        f"vessel pixels: {value_text['vessel_density']}"
    )

    visual_axes[1].imshow(_mask_overlay(image, skeleton, (1.0, 0.78, 0.05)))
    visual_axes[1].set_title(
        "Skeleton length density\n"
        f"centerline density: {value_text['skeleton_length_density']}"
    )

    visual_axes[2].imshow(_mask_overlay(image, skeleton, (0.85, 0.8, 0.2)))
    if branch_centroids.size:
        visual_axes[2].scatter(
            branch_centroids[:, 1],
            branch_centroids[:, 0],
            s=11,
            color="#e7292f",
            edgecolors="white",
            linewidths=0.25,
        )
    visual_axes[2].set_title(
        "Branchpoint density\n"
        f"junction density: {value_text['branchpoint_density']}"
    )

    _draw_boxcount_visual(
        visual_axes[3],
        mask,
        title=(
            "Fractal dimension\n"
            f"box-count slope: {value_text['fractal_dimension_boxcount']}"
        ),
    )

    visual_axes[4].imshow(_mask_overlay(image, skeleton, (0.8, 0.78, 0.2)))
    if tortuous_segment.any():
        visual_axes[4].imshow(
            np.ma.masked_where(~tortuous_segment, tortuous_segment),
            cmap="cool",
            interpolation="nearest",
            alpha=0.95,
        )
    if tortuous_endpoints is not None:
        start, end = tortuous_endpoints
        visual_axes[4].plot(
            [start[1], end[1]],
            [start[0], end[0]],
            color="#00a6d6",
            linestyle="--",
            linewidth=1.8,
        )
        visual_axes[4].scatter(
            [start[1], end[1]],
            [start[0], end[0]],
            s=24,
            color="#00a6d6",
            edgecolors="white",
            linewidths=0.4,
        )
    title_value = value_text["mean_segment_tortuosity"]
    if tortuous_value > 0:
        title_value = f"mean {title_value}, shown {tortuous_value:.3f}"
    visual_axes[4].set_title(f"Segment tortuosity\n{title_value}")

    visual_axes[5].imshow(gray, cmap="gray", alpha=0.35)
    visual_axes[5].imshow(component_display, cmap="tab20", interpolation="nearest", alpha=0.85)
    visual_axes[5].set_title(
        "Connected components\n"
        f"components: {value_text['connected_component_count']}"
    )

    for axis in [*reference_axes, *visual_axes]:
        axis.set_axis_off()

    fig.suptitle("ROSE feature visual glossary", y=1.015)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def plot_segmentation_comparison_grid(
    benchmark_df: pd.DataFrame,
    out_path: str | Path,
    *,
    method_order: list[str] | None = None,
    max_cases: int = 2,
    max_dim: int = 360,
):
    """Save a raw/manual/predicted-mask grid from a benchmark table."""
    if benchmark_df.empty:
        raise ValueError("benchmark_df is empty; cannot plot a comparison grid.")
    required = {"image_path", "manual_mask_path", "method", "pred_mask_path"}
    missing = sorted(required - set(benchmark_df.columns))
    if missing:
        msg = f"Missing benchmark columns: {', '.join(missing)}"
        raise ValueError(msg)

    out = _prepare_out_path(out_path)
    method_order = method_order or [
        "frangi",
        "diffusion_threshold",
        "random_walker",
        "geodesic_voting",
        "octa_net",
        "u_net",
        "nnunet",
        "unet_lite",
        "octa_net_lite",
        "nnunet_lite",
    ]
    available_methods = [method for method in method_order if method in set(benchmark_df["method"])]
    extra_methods = [
        method
        for method in benchmark_df["method"].dropna().unique()
        if method not in available_methods
    ]
    methods = [*available_methods, *extra_methods]
    if not methods:
        raise ValueError("benchmark_df has no plottable methods.")

    case_cols = ["image_path", "manual_mask_path"]
    for optional_col in ("image_id", "subject_id", "layer", "label"):
        if optional_col in benchmark_df.columns:
            case_cols.append(optional_col)
    cases = benchmark_df[case_cols].drop_duplicates().head(max_cases)
    n_rows = len(cases)
    n_cols = 2 + len(methods)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.25 * n_cols, 2.45 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, (_, case) in enumerate(cases.iterrows()):
        image = _resize_image_to_max_dim(skio.imread(case["image_path"]), max_dim)
        manual_mask = resize_mask_to_max_dim(
            ensure_grayscale(skio.imread(case["manual_mask_path"])) > 0,
            max_dim,
        )
        case_title = _case_title(case)

        axes[row_idx, 0].imshow(_display_image(image), cmap="gray")
        axes[row_idx, 0].set_title(case_title if case_title else "Raw OCTA")
        axes[row_idx, 1].imshow(_mask_overlay(image, manual_mask, (0.0, 0.75, 0.85)))
        axes[row_idx, 1].set_title("Manual mask")

        for method_idx, method in enumerate(methods, start=2):
            axis = axes[row_idx, method_idx]
            match = benchmark_df[
                (benchmark_df["image_path"] == case["image_path"])
                & (benchmark_df["manual_mask_path"] == case["manual_mask_path"])
                & (benchmark_df["method"] == method)
            ]
            if match.empty:
                _draw_missing_method(axis, method, "not run")
                continue
            result = match.iloc[0]
            pred_path = result.get("pred_mask_path")
            if pd.isna(pred_path) or not Path(str(pred_path)).exists():
                reason = result.get("error", "missing prediction")
                _draw_missing_method(axis, method, str(reason))
                continue
            pred_mask = resize_mask_to_max_dim(
                ensure_grayscale(skio.imread(pred_path)) > 0,
                max_dim,
            )
            axis.imshow(_mask_overlay(image, pred_mask, (1.0, 0.35, 0.08)))
            axis.set_title(_method_title(method, result))

    for axis in axes.ravel():
        axis.set_axis_off()

    fig.suptitle("ROSE segmentation comparator sample", y=1.02)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def _case_title(case: pd.Series) -> str:
    parts = []
    for column in ("image_id", "layer", "label"):
        value = case.get(column, None)
        if pd.notna(value):
            parts.append(str(value))
    return "\n".join(parts[:2])


def _method_title(method: str, result: pd.Series) -> str:
    labels = {
        "frangi": "Frangi",
        "diffusion_threshold": "Diffusion",
        "random_walker": "Random walker",
        "geodesic_voting": "Geodesic",
        "octa_net": "OCTA-Net",
        "u_net": "U-Net",
        "nnunet": "nnU-Net",
        "unet_lite": "U-Net Lite",
        "octa_net_lite": "OCTA-Net Lite",
        "nnunet_lite": "nnU-Net Lite",
    }
    title = labels.get(method, method.replace("_", " ").title())
    dice = result.get("dice", np.nan)
    if pd.notna(dice):
        title += f"\nDice {float(dice):.2f}"
    return title


def _draw_missing_method(axis, method: str, reason: str) -> None:
    axis.text(
        0.5,
        0.5,
        fill(reason, width=18),
        ha="center",
        va="center",
        fontsize=8,
        color="0.35",
        transform=axis.transAxes,
    )
    axis.set_title(method.replace("_", " ").title())


def _resize_image_to_max_dim(image: np.ndarray, max_dim: int) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim not in {2, 3}:
        msg = f"Expected a 2D or 3D image, got shape {arr.shape}."
        raise ValueError(msg)
    rows, cols = arr.shape[:2]
    current_max = max(rows, cols)
    if current_max <= max_dim:
        return arr
    scale = max_dim / current_max
    output_shape = (max(1, round(rows * scale)), max(1, round(cols * scale)), *arr.shape[2:])
    return transform.resize(arr, output_shape, preserve_range=True, anti_aliasing=True)


def _display_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return normalize_image(arr)
    return np.clip(arr.astype(float) / 255.0, 0.0, 1.0)


def plot_processing_example_panel(
    image: np.ndarray,
    manual_mask: np.ndarray,
    out_path: str | Path,
    *,
    title: str = "FIVES real-image processing example",
    black_ridges: bool = True,
    max_dim: int = 512,
):
    """Save a real image/mask/baseline/skeleton processing example."""
    out = _prepare_out_path(out_path)
    image = _resize_image_to_max_dim(image, max_dim)
    manual_mask = resize_mask_to_max_dim(ensure_grayscale(manual_mask) > 0, max_dim)
    baseline_mask = classical_vesselness_mask(
        image,
        threshold="percentile:90",
        black_ridges=black_ridges,
    )
    skeleton = skeletonize_mask(manual_mask)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), constrained_layout=True)
    panels = [
        (_display_image(image), "FIVES fundus image", None),
        (manual_mask, "Manual vessel mask", "gray"),
        (baseline_mask, "Classical baseline mask", "gray"),
        (skeleton, "Manual-mask skeleton", "magma"),
    ]
    for axis, (panel, panel_title, cmap) in zip(axes, panels, strict=True):
        axis.imshow(panel, cmap=cmap) if cmap else axis.imshow(panel)
        axis.set_title(panel_title)
        axis.set_axis_off()
    fig.suptitle(title, y=1.05)
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
