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

    from retivasc.features import extract_vascular_features
    from retivasc.io import DataNotFoundError, load_rose_manifest
    from retivasc.plotting import plot_feature_distributions, plot_rose_pipeline_panel
    from retivasc.preprocess import ensure_grayscale
    from retivasc.report_text import ROSE_MANUAL_MASK_CAVEAT, ROSE_NO_PREDICTION_CAVEAT
    from retivasc.segment import classical_vesselness_mask
    from retivasc.skeleton import skeletonize_mask
    from retivasc.splits import assert_group_split_safe, grouped_train_test_split

    return (
        DataNotFoundError,
        Path,
        ROSE_MANUAL_MASK_CAVEAT,
        ROSE_NO_PREDICTION_CAVEAT,
        assert_group_split_safe,
        classical_vesselness_mask,
        ensure_grayscale,
        extract_vascular_features,
        grouped_train_test_split,
        load_rose_manifest,
        pd,
        plot_feature_distributions,
        plot_rose_pipeline_panel,
        skeletonize_mask,
        skio,
    )


@app.cell
def _(DataNotFoundError, Path, load_rose_manifest, mo):
    rose_root = Path("data/raw/rose")
    rose_error = None
    try:
        rose_manifest = load_rose_manifest(rose_root)
    except DataNotFoundError as exc:
        rose_manifest = None
        rose_error = str(exc)
    except ValueError as exc:
        rose_manifest = None
        rose_error = f"ROSE metadata error:\n{exc}"

    if rose_error is None:
        status_message = f"""
            ## Local Data Check

            Loaded `{len(rose_manifest)}` ROSE image rows from `{rose_root}`.
            """
    else:
        status_message = f"""## Local ROSE Data Required

    ```text
    {rose_error}
    ```
    """
    mo.md(status_message)
    return (rose_manifest,)


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
def _(assert_group_split_safe, grouped_train_test_split, mo, rose_manifest):
    if rose_manifest is None:
        split_message = "Waiting for local ROSE data."
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
        evaluation split must be by `subject_id`. Labels are used only if explicitly
        supplied in a local manifest; none are inferred from filenames. This check
        demonstrates split hygiene without disease stratification when labels are absent.

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
        svc_rows = rose_manifest.loc[rose_manifest["layer"].astype("string") == "SVC"]
        first = svc_rows.iloc[0] if not svc_rows.empty else rose_manifest.iloc[0]
        image = skio.imread(first["image_path"])
        manual_mask = ensure_grayscale(skio.imread(first["mask_path"])) > 0
        predicted_mask = classical_vesselness_mask(image)
        skeleton = skeletonize_mask(manual_mask)

        _ = plot_rose_pipeline_panel(
            image,
            manual_mask,
            predicted_mask,
            skeleton,
            pipeline_path,
        )
    return (pipeline_path,)


@app.cell
def _(mo, pipeline_path, rose_manifest):
    if rose_manifest is None or not pipeline_path.exists():
        message = f"`{pipeline_path}` not generated because ROSE data are unavailable."
    else:
        message = (
            "Saved raw OCTA, manual annotation, classical baseline overlay, and skeleton "
            f"panel to `{pipeline_path}`."
        )
    (
        mo.vstack(
            [
                mo.md(f"## Pipeline Figure\n\n{message}"),
                mo.image(pipeline_path, alt="ROSE OCTA pipeline panel"),
            ]
        )
        if rose_manifest is not None and pipeline_path.exists()
        else mo.md(f"## Pipeline Figure\n\n{message}")
    )
    return


@app.cell
def _(ROSE_MANUAL_MASK_CAVEAT, ROSE_NO_PREDICTION_CAVEAT, mo):
    mo.md(
        f"""
        ## Scientific Caveats

        {ROSE_MANUAL_MASK_CAVEAT}

        {ROSE_NO_PREDICTION_CAVEAT}
        """
    )
    return


@app.cell
def _(
    Path,
    ensure_grayscale,
    extract_vascular_features,
    pd,
    rose_manifest,
    skio,
):
    feature_rows = []
    if rose_manifest is not None:
        for row in rose_manifest.itertuples(index=False):
            mask = ensure_grayscale(skio.imread(row.mask_path)) > 0
            features = extract_vascular_features(mask)
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

            `{feature_path}` not generated because ROSE data are unavailable.
            """
    else:
        feature_message = f"""
            ## Exploratory Manual-Mask Features

            Extracted `{len(rose_features)}` manual-mask feature rows and saved exploratory
            vascular feature distributions to `{feature_path}`.

            Grouped by available layer and subset metadata, not by disease status.
            Absolute magnitudes differ across layers and ROSE subsets largely because of
            annotation density and acquisition, so these distributions are a computer-vision
            sanity check, not a biological comparison.
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
