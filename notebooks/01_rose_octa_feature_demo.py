# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.9",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # ROSE OCTA Feature Demo

    ROSE is used here as an OCTA vessel-segmentation dataset. ROSE-1 is documented
    as an AD/control OCTA subset, but this notebook uses ROSE only for computer-vision
    sanity checks; it does not build or claim an ADRD diagnostic model.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import pandas as pd
    from skimage import io as skio

    from retivasc.deep_models import (
        PREDICTION_COLUMNS,
        DeepSegmenterConfig,
        torch_available,
        train_predict_deep_segmenters,
    )
    from retivasc.evaluation import (
        benchmark_native_segmenters,
        benchmark_prediction_columns,
        tune_native_segmenters,
    )
    from retivasc.external.registry import MODEL_REGISTRY
    from retivasc.features_rose import extract_rose_features
    from retivasc.io import DataNotFoundError, load_rose_manifest
    from retivasc.plotting import (
        plot_feature_distributions,
        plot_rose_feature_visuals,
        plot_rose_pipeline_panel,
        plot_segmentation_comparison_grid,
    )
    from retivasc.preprocess import ensure_grayscale
    from retivasc.report_text import ROSE_MANUAL_MASK_CAVEAT, ROSE_NO_PREDICTION_CAVEAT
    from retivasc.segment import classical_vesselness_mask
    from retivasc.skeleton import skeletonize_mask
    from retivasc.splits import assert_group_split_safe, grouped_train_test_split

    return (
        DataNotFoundError,
        DeepSegmenterConfig,
        MODEL_REGISTRY,
        PREDICTION_COLUMNS,
        Path,
        ROSE_MANUAL_MASK_CAVEAT,
        ROSE_NO_PREDICTION_CAVEAT,
        assert_group_split_safe,
        benchmark_native_segmenters,
        benchmark_prediction_columns,
        classical_vesselness_mask,
        ensure_grayscale,
        extract_rose_features,
        grouped_train_test_split,
        load_rose_manifest,
        pd,
        plot_feature_distributions,
        plot_rose_feature_visuals,
        plot_rose_pipeline_panel,
        plot_segmentation_comparison_grid,
        skeletonize_mask,
        skio,
        torch_available,
        tune_native_segmenters,
        train_predict_deep_segmenters,
    )


@app.cell
def _(DataNotFoundError, Path, load_rose_manifest, mo):
    rose_root = Path("data/raw/rose")
    rose_error = None
    rose_warning = None
    try:
        rose_manifest = load_rose_manifest(rose_root)
    except DataNotFoundError as exc:
        rose_manifest = None
        rose_error = str(exc)
    except ValueError as exc:
        rose_warning = f"ROSE metadata warning:\n{exc}"
        try:
            rose_manifest = load_rose_manifest(rose_root, require_split_safe=False)
        except DataNotFoundError as fallback_exc:
            rose_manifest = None
            rose_error = str(fallback_exc)
        except ValueError as fallback_exc:
            rose_manifest = None
            rose_error = f"ROSE metadata error:\n{fallback_exc}"

    if rose_error is None:
        warning_block = (
            f"""
            Loaded for non-split-sensitive demonstration only.

            ```text
            {rose_warning}
            ```

            Add `data/raw/rose/manifest.csv` with explicit `subject_id` and
            `split_group` columns before using these rows for split-sensitive analysis.
            """
            if rose_warning is not None
            else ""
        )
        status_message = f"""
            ## Local Data Check

            Loaded `{len(rose_manifest)}` ROSE image rows from `{rose_root}`.

            {warning_block}
            """
    else:
        status_message = f"""## Local ROSE Data Required

    ```text
    {rose_error}
    ```
    """
    mo.md(status_message)
    return rose_manifest, rose_warning


@app.cell
def _(mo, rose_manifest):
    if rose_manifest is None:
        audit_message = "## Manifest Audit\n\nWaiting for local ROSE data."
    else:
        subject_count = rose_manifest["subject_id"].nunique()
        dataset_counts = rose_manifest["dataset"].value_counts(dropna=False).to_dict()
        layer_counts = rose_manifest["layer"].value_counts(dropna=False).to_dict()
        label_counts = rose_manifest["label"].value_counts(dropna=False).to_dict()
        split_counts = (
            rose_manifest["official_split"].value_counts(dropna=False).to_dict()
            if "official_split" in rose_manifest.columns
            else {}
        )
        audit_message = f"""
            ## Manifest Audit

            - Rows: `{len(rose_manifest)}`
            - Subjects: `{subject_count}`
            - Datasets: `{dataset_counts}`
            - Layers: `{layer_counts}`
            - Labels: `{label_counts}`
            - Official splits: `{split_counts}`
            """
    mo.md(audit_message)
    return


@app.cell
def _(
    assert_group_split_safe,
    grouped_train_test_split,
    mo,
    rose_manifest,
    rose_warning,
):
    if rose_manifest is None:
        split_message = "Waiting for local ROSE data."
    elif rose_warning is not None:
        split_message = (
            "Split-sensitive ROSE validation is skipped because the official archive layout "
            "has ambiguous subject ids across split folders. The notebook still uses the "
            "available image/mask pairs for local computer-vision and feature demonstrations."
        )
    elif rose_manifest["subject_id"].nunique() >= 2:
        rose_train, rose_test = grouped_train_test_split(
            rose_manifest,
            group_col="subject_id",
            label_col="label" if rose_manifest["label"].notna().any() else None,
            test_size=0.25,
            random_state=0,
        )
        assert_group_split_safe(rose_train, rose_test, "subject_id")
        split_message = (
            f"Subject-level split check passed: {rose_train['subject_id'].nunique()} train "
            f"subjects and {rose_test['subject_id'].nunique()} test subjects."
        )
    else:
        split_message = "Subject-level split check skipped because fewer than two subjects loaded."

    mo.md(
        f"""
        ## Leakage Note

        Multiple ROSE angiograms/layers can come from the same subject, so any
        evaluation split must be by `subject_id`. The official ROSE-1 layout is
        annotated with disease/control labels from the published AD/control cohort
        ordering; ROSE-2 rows remain unlabeled. This check demonstrates split hygiene
        for any exploratory disease/control grouping.

        `{split_message}`
        """
    )
    return


@app.cell
def _(
    Path,
    classical_vesselness_mask,
    ensure_grayscale,
    plot_rose_pipeline_panel,
    rose_manifest,
    skeletonize_mask,
    skio,
):
    pipeline_path = Path("figures/rose_pipeline_panel.png")
    if rose_manifest is not None:
        _svc_rows = rose_manifest.loc[rose_manifest["layer"].astype("string") == "SVC"]
        _first = _svc_rows.iloc[0] if not _svc_rows.empty else rose_manifest.iloc[0]
        _image = skio.imread(_first["image_path"])
        _manual_mask = ensure_grayscale(skio.imread(_first["mask_path"])) > 0
        _predicted_mask = classical_vesselness_mask(_image)
        _skeleton = skeletonize_mask(_manual_mask)

        _ = plot_rose_pipeline_panel(
            _image,
            _manual_mask,
            _predicted_mask,
            _skeleton,
            pipeline_path,
        )
    return (pipeline_path,)


@app.cell
def _(mo, pipeline_path, rose_manifest):
    if rose_manifest is None or not pipeline_path.exists():
        _message = f"`{pipeline_path}` not generated because the ROSE manifest could not be loaded."
    else:
        _message = (
            "Saved raw OCTA, manual annotation, classical baseline overlay, and skeleton "
            f"panel to `{pipeline_path}`."
        )
    (
        mo.vstack(
            [
                mo.md(f"## Pipeline Figure\n\n{_message}"),
                mo.image(pipeline_path, alt="ROSE OCTA pipeline panel"),
            ]
        )
        if rose_manifest is not None and pipeline_path.exists()
        else mo.md(f"## Pipeline Figure\n\n{_message}")
    )
    return


@app.cell
def _(
    Path,
    ensure_grayscale,
    plot_rose_feature_visuals,
    rose_manifest,
    skio,
):
    feature_visuals_path = Path("figures/rose_feature_visuals.png")
    if rose_manifest is not None:
        _svc_rows = rose_manifest.loc[rose_manifest["layer"].astype("string") == "SVC"]
        _first = _svc_rows.iloc[0] if not _svc_rows.empty else rose_manifest.iloc[0]
        _image = skio.imread(_first["image_path"])
        _manual_mask = ensure_grayscale(skio.imread(_first["mask_path"])) > 0
        _ = plot_rose_feature_visuals(_image, _manual_mask, feature_visuals_path)
    return (feature_visuals_path,)


@app.cell
def _(mo):
    feature_visual_zoom = mo.ui.slider(
        start=80,
        stop=240,
        step=10,
        value=120,
        show_value=True,
        label="Feature visual zoom (%)",
        full_width=True,
    )
    return (feature_visual_zoom,)


@app.cell
def _(feature_visual_zoom, feature_visuals_path, mo, rose_manifest):
    def _zoomable_image(path):
        _image = mo.image(
            path,
            alt="ROSE feature visual glossary",
            width=f"{feature_visual_zoom.value}%",
            style={"display": "block", "max-width": "none"},
        )
        _zoom_html = f"""
        <div style="overflow: auto; max-height: 760px; background: white;
                    border: 1px solid #d8e0e5; border-radius: 8px; padding: 8px;">
          {_image.text}
        </div>
        """
        return mo.Html(_zoom_html)

    _missing_message = (
        f"`{feature_visuals_path}` not generated because the ROSE manifest could not be loaded."
    )
    _loaded_message = (
        "Saved a feature-by-feature visual glossary for one ROSE OCTA image to "
        f"`{feature_visuals_path}`."
    )
    (
        mo.md(f"## Feature Visuals\n\n{_missing_message}")
        if rose_manifest is None or not feature_visuals_path.exists()
        else mo.vstack(
            [
                mo.md(f"## Feature Visuals\n\n{_loaded_message}"),
                feature_visual_zoom,
                _zoomable_image(feature_visuals_path),
            ]
        )
    )
    return


@app.cell
def _(ROSE_MANUAL_MASK_CAVEAT, ROSE_NO_PREDICTION_CAVEAT, mo):
    mo.md(f"""
    ## Scientific Caveats

    {ROSE_MANUAL_MASK_CAVEAT}

    {ROSE_NO_PREDICTION_CAVEAT}
    """)
    return


@app.cell
def _(pd, rose_manifest):
    def _rose_comparator_candidates(manifest):
        if manifest is None or manifest.empty:
            return pd.DataFrame()

        candidates = manifest.copy()
        if "dataset" in candidates.columns:
            rose1 = candidates.loc[candidates["dataset"].astype("string") == "ROSE-1"]
            if not rose1.empty:
                candidates = rose1
        if "layer" in candidates.columns:
            svc = candidates.loc[candidates["layer"].astype("string") == "SVC"]
            if not svc.empty:
                candidates = svc

        return candidates.reset_index(drop=True)

    def _balanced_sample(
        candidates, *, max_rows: int, split: str | None = None, exclude_subjects=()
    ):
        if candidates.empty:
            return pd.DataFrame()
        rows = candidates.copy()
        if split is not None and "official_split" in rows.columns:
            split_rows = rows.loc[rows["official_split"].astype("string").str.lower() == split]
            if not split_rows.empty:
                rows = split_rows
        if exclude_subjects and "subject_id" in rows.columns:
            rows = rows.loc[~rows["subject_id"].astype("string").isin(set(exclude_subjects))]
        if rows.empty:
            return pd.DataFrame()

        selected_indexes = []
        if "label" in rows.columns:
            labels = rows["label"].astype("string").str.lower()
            for label in ("disease", "control"):
                label_rows = rows.loc[labels == label]
                if not label_rows.empty:
                    selected_indexes.extend(label_rows.head(max(1, max_rows // 2)).index)

        for row_index in rows.index:
            if row_index not in selected_indexes:
                selected_indexes.append(row_index)
            if len(selected_indexes) >= max_rows:
                break

        selected = rows.loc[selected_indexes].copy()
        if "image_id" in selected.columns:
            selected = selected.drop_duplicates("image_id")
        return selected.head(max_rows).reset_index(drop=True)

    rose_candidate_rows = _rose_comparator_candidates(rose_manifest)
    rose_tuning_rows = _balanced_sample(rose_candidate_rows, max_rows=4, split="train")
    rose_deep_tuning_rows = _balanced_sample(rose_candidate_rows, max_rows=8, split="train")
    excluded_subjects = (
        set(rose_tuning_rows["subject_id"].astype("string"))
        if "subject_id" in rose_tuning_rows.columns
        else set()
    )
    rose_comparison_rows = _balanced_sample(
        rose_candidate_rows,
        max_rows=2,
        split="test",
        exclude_subjects=excluded_subjects,
    )
    if rose_comparison_rows.empty:
        rose_comparison_rows = _balanced_sample(
            rose_candidate_rows,
            max_rows=2,
            exclude_subjects=excluded_subjects,
        )
    return rose_candidate_rows, rose_comparison_rows, rose_deep_tuning_rows, rose_tuning_rows


@app.cell
def _(
    DeepSegmenterConfig,
    PREDICTION_COLUMNS,
    Path,
    pd,
    rose_comparison_rows,
    rose_deep_tuning_rows,
    rose_tuning_rows,
    torch_available,
    train_predict_deep_segmenters,
):
    def _merge_prediction_manifest(base_rows, manifest_path, prediction_cols):
        if base_rows.empty or not manifest_path.exists():
            return base_rows
        external = pd.read_csv(manifest_path)
        if "image_id" not in external.columns:
            return base_rows
        available_cols = [col for col in prediction_cols if col in external.columns]
        if not available_cols:
            return base_rows
        for col in available_cols:
            external[col] = external[col].map(
                lambda value: _resolve_prediction_path(value, manifest_path.parent)
            )
        merge_cols = ["image_id", *available_cols]
        merged = base_rows.merge(
            external[merge_cols].drop_duplicates("image_id"),
            on="image_id",
            how="left",
            suffixes=("", "_external"),
        )
        for col in available_cols:
            external_col = f"{col}_external"
            if external_col in merged.columns:
                if col in base_rows.columns:
                    merged[col] = merged[col].combine_first(merged[external_col])
                else:
                    merged[col] = merged[external_col]
                merged = merged.drop(columns=[external_col])
        return merged

    def _resolve_prediction_path(value, base_dir):
        if pd.isna(value) or str(value).strip() == "":
            return value
        path = Path(str(value))
        if path.is_absolute() or path.exists():
            return str(path)
        return str(base_dir / path)

    rose_deep_prediction_manifest_path = Path("data/interim/rose_inrepo_deep_predictions.csv")
    rose_deep_training_history_path = Path("data/interim/rose_inrepo_deep_training_history.csv")
    rose_deep_output_dir = Path("data/interim/rose_inrepo_deep_comparison")
    rose_deep_status = "not run"
    rose_deep_methods = ("unet_lite", "octa_net_lite", "nnunet_lite")
    rose_deep_postprocess_cols = [
        f"{method}_postprocess_params_json" for method in rose_deep_methods
    ]

    if rose_deep_prediction_manifest_path.exists() and all(
        col in pd.read_csv(rose_deep_prediction_manifest_path, nrows=0).columns
        for col in rose_deep_postprocess_cols
    ):
        rose_deep_status = (
            f"using cached tuned predictions from `{rose_deep_prediction_manifest_path}`"
        )
    elif rose_deep_tuning_rows.empty or rose_comparison_rows.empty:
        rose_deep_status = "waiting for ROSE tuning and comparison rows"
    elif not torch_available():
        rose_deep_status = (
            "PyTorch is not installed, so in-repo deep comparators were skipped. "
            "Install a PyTorch-enabled environment or provide external prediction CSVs."
        )
    else:
        try:
            rose_deep_predictions, rose_deep_history = train_predict_deep_segmenters(
                rose_deep_tuning_rows,
                rose_comparison_rows,
                rose_deep_output_dir,
                methods=rose_deep_methods,
                config=DeepSegmenterConfig(
                    image_size=128,
                    epochs=6,
                    batch_size=2,
                    base_channels=8,
                    device="auto",
                    seed=0,
                ),
            )
            rose_deep_prediction_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            rose_deep_predictions.to_csv(rose_deep_prediction_manifest_path, index=False)
            rose_deep_history.to_csv(rose_deep_training_history_path, index=False)
            rose_deep_status = (
                f"trained local deep comparators, tuned threshold/morphology on "
                f"ROSE tuning rows, and cached predictions at "
                f"`{rose_deep_prediction_manifest_path}`"
            )
        except Exception as exc:
            rose_deep_status = f"in-repo deep comparator training failed: {exc}"

    rose_external_manifest_specs = [
        (
            "OCTA-Net",
            Path("data/interim/rose_octa_net_predictions.csv"),
            ["octa_net_prediction_path"],
        ),
        (
            "U-Net",
            Path("data/interim/rose_unet_predictions.csv"),
            ["unet_prediction_path"],
        ),
        (
            "nnU-Net",
            Path("data/interim/rose_nnunet_predictions.csv"),
            ["nnunet_prediction_path"],
        ),
        (
            "U-Net Lite",
            rose_deep_prediction_manifest_path,
            [PREDICTION_COLUMNS["unet_lite"]],
        ),
        (
            "OCTA-Net Lite",
            rose_deep_prediction_manifest_path,
            [PREDICTION_COLUMNS["octa_net_lite"]],
        ),
        (
            "nnU-Net Lite",
            rose_deep_prediction_manifest_path,
            [PREDICTION_COLUMNS["nnunet_lite"]],
        ),
    ]
    rose_comparison_rows_with_external = rose_comparison_rows.copy()
    for _, manifest_path, prediction_cols in rose_external_manifest_specs:
        rose_comparison_rows_with_external = _merge_prediction_manifest(
            rose_comparison_rows_with_external,
            manifest_path,
            prediction_cols,
        )

    rose_external_prediction_cols = {}
    prediction_column_methods = {
        "octa_net_prediction_path": "octa_net",
        "unet_prediction_path": "u_net",
        "nnunet_prediction_path": "nnunet",
        PREDICTION_COLUMNS["unet_lite"]: "unet_lite",
        PREDICTION_COLUMNS["octa_net_lite"]: "octa_net_lite",
        PREDICTION_COLUMNS["nnunet_lite"]: "nnunet_lite",
    }
    for prediction_col, method in prediction_column_methods.items():
        if prediction_col in rose_comparison_rows_with_external.columns:
            rose_external_prediction_cols[method] = prediction_col

    return (
        rose_comparison_rows_with_external,
        rose_deep_prediction_manifest_path,
        rose_deep_status,
        rose_deep_training_history_path,
        rose_external_manifest_specs,
        rose_external_prediction_cols,
    )


@app.cell
def _(
    Path,
    benchmark_native_segmenters,
    benchmark_prediction_columns,
    pd,
    rose_comparison_rows,
    rose_comparison_rows_with_external,
    rose_external_prediction_cols,
    rose_tuning_rows,
    tune_native_segmenters,
):
    rose_comparison_output_dir = Path("data/interim/rose_model_comparison")
    rose_comparison_metrics_path = Path("data/interim/rose_model_comparison_metrics.csv")
    rose_external_metrics_path = Path("data/interim/rose_external_model_comparison_metrics.csv")
    rose_tuning_metrics_path = Path("data/interim/rose_model_tuning_metrics.csv")
    rose_tuning_summary_path = Path("data/interim/rose_model_tuning_summary.csv")
    rose_comparison_grid_path = Path("figures/rose_model_comparison_grid.png")
    rose_methods = ["frangi", "diffusion", "random_walker", "geodesic"]
    rose_param_grids = {
        "frangi": [
            {"threshold": "otsu", "min_size": 8},
            {"threshold": "yen", "min_size": 8},
            {"threshold": "percentile:85", "min_size": 8},
            {"threshold": "percentile:90", "min_size": 8},
        ],
        "diffusion": [
            {
                "n_iter": 3,
                "threshold": "otsu",
                "window_size": 31,
                "min_size": 8,
                "clahe": True,
            },
            {
                "n_iter": 6,
                "threshold": "otsu",
                "window_size": 31,
                "min_size": 8,
                "clahe": True,
            },
            {
                "n_iter": 6,
                "threshold": "yen",
                "window_size": 31,
                "min_size": 8,
                "clahe": True,
            },
            {
                "n_iter": 6,
                "threshold": "percentile:85",
                "window_size": 31,
                "min_size": 8,
                "clahe": True,
            },
        ],
        "random_walker": [
            {
                "vessel_seed_quantile": 0.94,
                "background_seed_quantile": 0.20,
                "beta": 50.0,
                "mode": "bf",
                "min_size": 8,
            },
            {
                "vessel_seed_quantile": 0.96,
                "background_seed_quantile": 0.20,
                "beta": 80.0,
                "mode": "bf",
                "min_size": 8,
            },
            {
                "vessel_seed_quantile": 0.98,
                "background_seed_quantile": 0.20,
                "beta": 80.0,
                "mode": "bf",
                "min_size": 8,
            },
            {
                "vessel_seed_quantile": 0.96,
                "background_seed_quantile": 0.15,
                "beta": 80.0,
                "mode": "bf",
                "min_size": 8,
            },
        ],
        "geodesic": [
            {
                "max_seeds": 16,
                "max_pairs": 48,
                "vote_threshold": 0.03,
                "downsample_max_dim": 256,
                "min_size": 8,
                "dilation_radius": 1,
                "random_state": 0,
            },
            {
                "max_seeds": 16,
                "max_pairs": 48,
                "vote_threshold": 0.05,
                "downsample_max_dim": 256,
                "min_size": 8,
                "dilation_radius": 2,
                "random_state": 0,
            },
            {
                "max_seeds": 24,
                "max_pairs": 80,
                "vote_threshold": 0.05,
                "downsample_max_dim": 256,
                "min_size": 8,
                "dilation_radius": 2,
                "random_state": 0,
            },
            {
                "max_seeds": 24,
                "max_pairs": 80,
                "vote_threshold": 0.08,
                "downsample_max_dim": 256,
                "min_size": 8,
                "dilation_radius": 3,
                "random_state": 0,
            },
        ],
    }

    if rose_tuning_rows.empty:
        rose_tuning_table = pd.DataFrame()
        rose_tuning_summary = pd.DataFrame()
        rose_best_params = {}
    else:
        rose_tuning_table, rose_tuning_summary, rose_best_params = tune_native_segmenters(
            rose_tuning_rows,
            rose_param_grids,
            methods=rose_methods,
            score_col="dice",
        )
        rose_tuning_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        rose_tuning_table.to_csv(rose_tuning_metrics_path, index=False)
        rose_tuning_summary.to_csv(rose_tuning_summary_path, index=False)

    if rose_comparison_rows.empty or not rose_best_params:
        rose_native_comparison_table = pd.DataFrame()
    else:
        rose_native_comparison_table = benchmark_native_segmenters(
            rose_comparison_rows,
            methods=rose_methods,
            output_root=rose_comparison_output_dir,
            method_params=rose_best_params,
        )

    if rose_comparison_rows_with_external.empty or not rose_external_prediction_cols:
        rose_external_comparison_table = pd.DataFrame()
    else:
        rose_external_comparison_table = benchmark_prediction_columns(
            rose_comparison_rows_with_external,
            rose_external_prediction_cols,
        )
        rose_external_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        rose_external_comparison_table.to_csv(rose_external_metrics_path, index=False)

    rose_comparison_table = pd.concat(
        [rose_native_comparison_table, rose_external_comparison_table],
        ignore_index=True,
    )
    if not rose_comparison_table.empty:
        rose_comparison_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        rose_comparison_table.to_csv(rose_comparison_metrics_path, index=False)

    return (
        rose_best_params,
        rose_comparison_grid_path,
        rose_comparison_metrics_path,
        rose_comparison_table,
        rose_external_comparison_table,
        rose_external_metrics_path,
        rose_methods,
        rose_native_comparison_table,
        rose_tuning_metrics_path,
        rose_tuning_rows,
        rose_tuning_summary,
        rose_tuning_summary_path,
        rose_tuning_table,
    )


@app.cell
def _(plot_segmentation_comparison_grid, rose_comparison_grid_path, rose_comparison_table):
    if (
        not rose_comparison_table.empty
        and "pred_mask_path" in rose_comparison_table.columns
        and rose_comparison_table["pred_mask_path"].notna().any()
    ):
        _ = plot_segmentation_comparison_grid(
            rose_comparison_table,
            rose_comparison_grid_path,
            max_cases=2,
            max_dim=360,
        )
    return


@app.cell
def _(
    MODEL_REGISTRY,
    mo,
    pd,
    rose_best_params,
    rose_comparison_grid_path,
    rose_comparison_metrics_path,
    rose_comparison_rows,
    rose_comparison_table,
    rose_comparison_rows_with_external,
    rose_deep_prediction_manifest_path,
    rose_deep_status,
    rose_deep_training_history_path,
    rose_external_manifest_specs,
    rose_external_prediction_cols,
    rose_tuning_rows,
    rose_tuning_summary,
    rose_tuning_summary_path,
):
    def _format_summary(table):
        display_cols = [
            "image_id",
            "layer",
            "label",
            "method",
            "dice",
            "iou",
            "precision",
            "recall",
            "specificity",
            "abs_error_vessel_density",
            "abs_error_skeleton_length_density",
            "runtime_seconds",
            "error",
        ]
        summary = table[[col for col in display_cols if col in table.columns]].copy()
        method_order = {
            "frangi": 0,
            "diffusion_threshold": 1,
            "random_walker": 2,
            "geodesic_voting": 3,
            "octa_net": 4,
            "u_net": 5,
            "nnunet": 6,
            "unet_lite": 7,
            "octa_net_lite": 8,
            "nnunet_lite": 9,
        }
        summary["_method_order"] = summary["method"].map(method_order).fillna(99)
        summary = summary.sort_values(["image_id", "_method_order"]).drop(columns="_method_order")
        numeric_cols = [
            "dice",
            "iou",
            "precision",
            "recall",
            "specificity",
            "abs_error_vessel_density",
            "abs_error_skeleton_length_density",
            "runtime_seconds",
        ]
        for col in numeric_cols:
            if col in summary.columns:
                summary[col] = summary[col].map(
                    lambda value: "" if pd.isna(value) else f"{float(value):.3f}"
                )
        return summary

    def _format_tuning_summary(table):
        if table.empty:
            return table
        display_cols = [
            "requested_method",
            "method",
            "candidate_index",
            "selected",
            "n_success",
            "mean_dice",
            "mean_iou",
            "mean_abs_error_vessel_density",
            "mean_runtime_seconds",
            "candidate_params_json",
        ]
        summary = table[[col for col in display_cols if col in table.columns]].copy()
        numeric_cols = [
            "mean_dice",
            "mean_iou",
            "mean_abs_error_vessel_density",
            "mean_runtime_seconds",
        ]
        for col in numeric_cols:
            if col in summary.columns:
                summary[col] = summary[col].map(
                    lambda value: "" if pd.isna(value) else f"{float(value):.3f}"
                )
        return summary.sort_values(["requested_method", "candidate_index"])

    def _row_ids(rows):
        if rows.empty or "image_id" not in rows.columns:
            return ""
        return ", ".join(str(value) for value in rows["image_id"].head(6))

    def _subject_overlap(left, right):
        if (
            left.empty
            or right.empty
            or "subject_id" not in left.columns
            or "subject_id" not in right.columns
        ):
            return set()
        return set(left["subject_id"].astype("string")) & set(right["subject_id"].astype("string"))

    def _external_status_table():
        rows = []
        method_info = {
            "OCTA-Net": ("octa_net_prediction_path", MODEL_REGISTRY["octa_net"]["description"]),
            "U-Net": ("unet_prediction_path", MODEL_REGISTRY["u_net"]["description"]),
            "nnU-Net": ("nnunet_prediction_path", MODEL_REGISTRY["nnunet"]["description"]),
            "U-Net Lite": (
                "unet_lite_prediction_path",
                MODEL_REGISTRY["unet_lite"]["description"],
            ),
            "OCTA-Net Lite": (
                "octa_net_lite_prediction_path",
                MODEL_REGISTRY["octa_net_lite"]["description"],
            ),
            "nnU-Net Lite": (
                "nnunet_lite_prediction_path",
                MODEL_REGISTRY["nnunet_lite"]["description"],
            ),
        }
        for label, manifest_path, _prediction_cols in rose_external_manifest_specs:
            prediction_col, role = method_info[label]
            if prediction_col in rose_comparison_rows_with_external.columns:
                available = int(rose_comparison_rows_with_external[prediction_col].notna().sum())
                status = f"{available} prediction path(s) available"
            else:
                status = f"no `{prediction_col}` column; expected `{manifest_path}`"
            rows.append({"method": label, "status": status, "role": role})
        return pd.DataFrame(rows)

    def _table_html(table):
        return mo.Html(
            "<div style='overflow-x:auto'>"
            + table.to_html(index=False, border=0, na_rep="", escape=False)
            + "</div>"
        )

    def _comparison_view():
        if rose_comparison_rows.empty:
            return mo.md(
                "## Segmentation Comparisons\n\nWaiting for local ROSE data."
            )
        if rose_comparison_table.empty:
            return mo.md(
                "## Segmentation Comparisons\n\nNo held-out comparison was generated. "
                "This usually means tuning rows were unavailable or no candidate completed."
            )
        completed = int(rose_comparison_table["error"].isna().sum())
        total = len(rose_comparison_table)
        external_count = len(rose_external_prediction_cols)
        summary = _format_summary(rose_comparison_table)
        tuning_summary = _format_tuning_summary(rose_tuning_summary)
        external_status = _external_status_table()
        overlap = _subject_overlap(rose_tuning_rows, rose_comparison_rows)
        overlap_message = (
            "No tuning/evaluation subject overlap."
            if not overlap
            else f"Subject overlap detected: {sorted(overlap)}"
        )
        intro = f"""
        ## Segmentation Comparisons

        This section tunes a small parameter grid on ROSE tuning rows, then reports
        metrics on separate held-out rows. Frangi is included in the same tuning
        discipline so the comparison is not biased by hand-picking parameters only for
        the newer methods. Selection uses mean tuning Dice, with vessel-density error
        and runtime as tie-breakers.

        - Tuning rows: `{len(rose_tuning_rows)}` (`{_row_ids(rose_tuning_rows)}`)
        - Held-out comparison rows: `{len(rose_comparison_rows)}`
          (`{_row_ids(rose_comparison_rows)}`)
        - Split check: `{overlap_message}`
        - Selected methods: `{len(rose_best_params)}`
        - External prediction sources found: `{external_count}`
        - In-repo deep comparator status: `{rose_deep_status}`
        - Completed held-out runs: `{completed}` of `{total}`

        Saved tuning summary to `{rose_tuning_summary_path}` and held-out metrics to
        `{rose_comparison_metrics_path}`. These are segmentation and feature-stability
        checks only; they are not disease-prediction metrics.

        In-repo U-Net Lite, OCTA-Net Lite, and nnU-Net Lite predictions are cached at
        `{rose_deep_prediction_manifest_path}` when PyTorch is available; training
        history is saved to `{rose_deep_training_history_path}`. These local models
        are demo comparators, not the official OCTA-Net or nnU-Net implementations.

        To add official OCTA-Net, U-Net, or nnU-Net rows, provide a CSV under
        `data/interim/` with
        `image_id` plus one of `octa_net_prediction_path`, `unet_prediction_path`, or
        `nnunet_prediction_path`. The prediction masks must already exist locally.
        """
        items = [mo.md(intro)]
        if not tuning_summary.empty:
            items.extend(
                [
                    mo.md("### Tuning Summary"),
                    _table_html(tuning_summary),
                ]
            )
        if rose_comparison_grid_path.exists():
            items.append(
                mo.image(
                    rose_comparison_grid_path,
                    alt="ROSE segmentation comparator grid",
                )
            )
        items.extend(
            [
                mo.md("### Held-Out Comparison"),
                _table_html(summary),
                mo.md("### External Comparator Status"),
                _table_html(external_status),
            ]
        )
        return mo.vstack(items)

    _comparison_view()
    return


@app.cell
def _(
    Path,
    ensure_grayscale,
    extract_rose_features,
    pd,
    rose_manifest,
    skio,
):
    feature_rows = []
    if rose_manifest is not None:
        for row in rose_manifest.itertuples(index=False):
            mask = ensure_grayscale(skio.imread(row.mask_path)) > 0
            features = extract_rose_features(mask)
            features.update(
                {
                    "dataset": row.dataset,
                    "subject_id": row.subject_id,
                    "image_id": row.image_id,
                    "layer": row.layer,
                    "label": row.label,
                    "feature_group": row.label if pd.notna(row.label) else row.layer,
                }
            )
            feature_rows.append(features)

    rose_features = pd.DataFrame(feature_rows)
    feature_path = Path("figures/rose_feature_distributions.png")
    return feature_path, rose_features


@app.cell
def _(feature_path, plot_feature_distributions, rose_features):
    if not rose_features.empty:
        group_col = "label" if rose_features["label"].notna().any() else "feature_group"
        _ = plot_feature_distributions(
            rose_features,
            [
                "vessel_density",
                "branchpoint_density",
                "fractal_dimension_boxcount",
                "mean_segment_tortuosity",
            ],
            group_col,
            feature_path,
        )
    return


@app.cell
def _(feature_path, mo, rose_features):
    if rose_features.empty or not feature_path.exists():
        feature_message = f"""
            ## Exploratory Manual-Mask Features

            `{feature_path}` not generated because no ROSE feature rows were loaded.
            """
    else:
        feature_message = f"""
            ## Exploratory Manual-Mask Features

            Extracted `{len(rose_features)}` manual-mask feature rows and saved exploratory
            vascular feature distributions to `{feature_path}`.

            ROSE-1 rows are grouped by the official disease/control cohort labels;
            unlabeled ROSE-2 rows are excluded from label-grouped plots.
            Absolute magnitudes differ across layers and ROSE subsets largely because of
            annotation density and acquisition, so these distributions are a computer-vision
            sanity check, not a predictive Alzheimer's model.
            """
    (
        mo.vstack(
            [
                mo.md(feature_message),
                mo.image(feature_path, alt="ROSE manual-mask feature distributions"),
            ]
        )
        if not rose_features.empty and feature_path.exists()
        else mo.md(feature_message)
    )
    return


if __name__ == "__main__":
    app.run()
