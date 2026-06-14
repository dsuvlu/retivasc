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
    # FIVES Modeling And Calibration Demo

    FIVES is not an ADRD dataset. It is used here to demonstrate modeling
    discipline on a larger annotated fundus dataset when clean labels are present.
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from skimage import io as skio
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from retivasc.features import extract_vascular_features
    from retivasc.io import DataNotFoundError, load_fives_manifest
    from retivasc.plotting import plot_calibration
    from retivasc.preprocess import ensure_grayscale, resize_mask_to_max_dim
    from retivasc.splits import assert_group_split_safe, grouped_train_test_split

    return (
        DataNotFoundError,
        LogisticRegression,
        Path,
        StandardScaler,
        assert_group_split_safe,
        average_precision_score,
        brier_score_loss,
        ensure_grayscale,
        extract_vascular_features,
        grouped_train_test_split,
        json,
        load_fives_manifest,
        make_pipeline,
        np,
        pd,
        plot_calibration,
        resize_mask_to_max_dim,
        roc_auc_score,
        skio,
    )


@app.cell
def _(DataNotFoundError, Path, load_fives_manifest, mo):
    fives_root = Path("data/raw/fives")
    fives_error = None
    try:
        fives_manifest = load_fives_manifest(fives_root)
    except DataNotFoundError as exc:
        fives_manifest = None
        fives_error = str(exc)
    except ValueError as exc:
        fives_manifest = None
        fives_error = f"FIVES metadata error:\n{exc}"

    if fives_error is None:
        status_message = f"""
            ## Local Data Check

            Loaded `{len(fives_manifest)}` FIVES image rows from `{fives_root}`.
            """
    else:
        status_message = f"""## Local FIVES Data Required

    ```text
    {fives_error}
    ```
    """
    mo.md(status_message)
    return (fives_manifest,)


@app.cell
def _(
    Path,
    ensure_grayscale,
    extract_vascular_features,
    fives_manifest,
    pd,
    resize_mask_to_max_dim,
    skio,
):
    feature_max_dim = 512
    feature_cache_path = Path("data/interim/fives_features_max512.parquet")

    cache_ok = False
    if fives_manifest is not None and feature_cache_path.exists():
        cached_features = pd.read_parquet(feature_cache_path)
        cache_ok = (
            len(cached_features) == len(fives_manifest)
            and "feature_max_dim" in cached_features.columns
            and set(cached_features["feature_max_dim"]) == {feature_max_dim}
        )
    else:
        cached_features = pd.DataFrame()

    if cache_ok:
        fives_features = cached_features
    else:
        rows = []
        if fives_manifest is not None:
            feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
            for row in fives_manifest.itertuples(index=False):
                mask = ensure_grayscale(skio.imread(row.mask_path)) > 0
                mask = resize_mask_to_max_dim(mask, feature_max_dim)
                features = extract_vascular_features(mask)
                features.update(
                    {
                        "dataset": row.dataset,
                        "subject_id": getattr(row, "subject_id", row.image_id),
                        "image_id": row.image_id,
                        "label": getattr(row, "label", None),
                        "official_split": getattr(row, "official_split", None),
                        "split_group": getattr(
                            row, "split_group", getattr(row, "subject_id", row.image_id)
                        ),
                        "feature_max_dim": feature_max_dim,
                    }
                )
                rows.append(features)

        fives_features = pd.DataFrame(rows)
        if fives_manifest is not None:
            fives_features.to_parquet(feature_cache_path, index=False)
    return feature_cache_path, feature_max_dim, fives_features


@app.cell
def _(feature_cache_path, feature_max_dim, fives_features, fives_manifest, mo):
    if fives_manifest is None:
        feature_message = "## Feature Extraction\n\nWaiting for local FIVES data."
    else:
        label_counts = (
            fives_features["label"].fillna("<missing>").value_counts(dropna=False).to_dict()
            if "label" in fives_features.columns
            else {}
        )
        feature_message = f"""
            ## Feature Extraction

            Extracted vascular features from `{len(fives_features)}` FIVES masks.
            Masks were downsampled to maximum dimension `{feature_max_dim}` for this
            fast demo run and cached at `{feature_cache_path}`.

            Label counts: `{label_counts}`
            """
    mo.md(feature_message)
    return


@app.cell
def _(fives_features, mo, np, pd):
    def make_binary_target(features: pd.DataFrame) -> pd.DataFrame:
        if "label" not in features.columns:
            raise ValueError("FIVES manifest has no label column; calibration is not available.")

        out = features.copy()
        labels = out["label"].astype("string").str.strip()
        valid = labels.notna() & (labels != "")
        out = out.loc[valid].copy()
        labels = labels.loc[valid].str.lower()
        unique = sorted(labels.unique())
        normal_terms = {"normal", "control", "healthy", "no disease", "no_disease"}

        if any(label in normal_terms for label in unique) and any(
            label not in normal_terms for label in unique
        ):
            out["target"] = (~labels.isin(normal_terms)).astype(int).to_numpy()
            out["target_name"] = "disease_vs_normal"
            return out

        if len(unique) == 2:
            mapping = {unique[0]: 0, unique[1]: 1}
            out["target"] = labels.map(mapping).astype(int).to_numpy()
            out["target_name"] = f"{unique[1]}_vs_{unique[0]}"
            return out

        raise ValueError(
            "FIVES labels are unavailable or not cleanly binary. Add a manifest.csv with "
            "a binary label column or normal-vs-disease labels before running calibration."
        )

    modeling_df = pd.DataFrame()
    numeric_cols = []
    calibration_message = None

    if fives_features.empty:
        calibration_message = "Waiting for local FIVES data."
    else:
        try:
            modeling_df = make_binary_target(fives_features)
        except ValueError as exc:
            calibration_message = str(exc)

    calibration_status = None
    if calibration_message is not None:
        calibration_status = f"""## Calibration Not Generated

    ```text
    {calibration_message}
    ```

    No labels were invented. Add a clean binary label column to the local FIVES manifest
    before using the calibration section.
    """
    elif modeling_df["target"].nunique() != 2 or modeling_df["target"].value_counts().min() < 4:
        calibration_status = """
            ## Calibration Not Generated

            FIVES calibration requires at least two classes with four or more examples each.
            """
        modeling_df = pd.DataFrame()
    else:
        numeric_cols = [
            "vessel_density",
            "skeleton_length_density",
            "branchpoint_density",
            "fractal_dimension_boxcount",
            "connected_component_count",
        ]
        modeling_df = modeling_df.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_cols)

    mo.md(calibration_status) if calibration_status is not None else None
    return modeling_df, numeric_cols


@app.cell
def _(
    LogisticRegression,
    StandardScaler,
    assert_group_split_safe,
    grouped_train_test_split,
    make_pipeline,
    modeling_df,
    np,
    numeric_cols,
):
    train_df = None
    test_df = None
    y_true = np.array([])
    y_prob = np.array([])
    train_message = None

    if modeling_df.empty:
        train_message = "Calibration not generated."
    else:
        try:
            if "official_split" in modeling_df.columns and {
                "train",
                "test",
            } <= set(modeling_df["official_split"].dropna()):
                train_df = modeling_df[modeling_df["official_split"] == "train"].copy()
                test_df = modeling_df[modeling_df["official_split"] == "test"].copy()
            else:
                train_df, test_df = grouped_train_test_split(
                    modeling_df,
                    group_col="split_group",
                    label_col="target",
                    test_size=0.25,
                    random_state=0,
                )
            assert_group_split_safe(train_df, test_df, "split_group")
            if train_df["target"].nunique() != 2 or test_df["target"].nunique() != 2:
                raise ValueError("Train and test sets must each contain both binary classes.")

            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
            )
            model.fit(train_df[numeric_cols], train_df["target"])
            y_true = test_df["target"].to_numpy()
            y_prob = model.predict_proba(test_df[numeric_cols])[:, 1]
        except ValueError as exc:
            train_message = str(exc)
    return test_df, train_df, train_message, y_prob, y_true


@app.cell
def _(
    Path,
    average_precision_score,
    brier_score_loss,
    json,
    np,
    plot_calibration,
    roc_auc_score,
    test_df,
    train_df,
    train_message,
    y_prob,
    y_true,
):
    calibration_path = Path("figures/fives_calibration_demo.png")
    metrics_path = Path("reports/fives_metrics.json")
    metrics_summary = None

    def bootstrap_ci(metric_func, true, prob, *, n_boot=500, random_state=0):
        rng = np.random.default_rng(random_state)
        values = []
        indices = np.arange(true.size)
        for _ in range(n_boot):
            sample = rng.choice(indices, size=indices.size, replace=True)
            if len(np.unique(true[sample])) < 2:
                continue
            values.append(metric_func(true[sample], prob[sample]))
        if not values:
            return (float("nan"), float("nan"))
        return tuple(np.percentile(values, [2.5, 97.5]))

    if train_message is None:
        roc_auc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        roc_ci = bootstrap_ci(roc_auc_score, y_true, y_prob)
        auprc_ci = bootstrap_ci(average_precision_score, y_true, y_prob)

        _ = plot_calibration(
            y_true,
            y_prob,
            calibration_path,
            title="FIVES disease-vs-normal calibration",
            metrics={
                "AUROC": roc_auc,
                "AUPRC": auprc,
                "Brier": brier,
                "n": int(y_true.size),
            },
        )
        metrics_summary = {
            "AUROC": float(roc_auc),
            "AUROC 95% CI": [float(roc_ci[0]), float(roc_ci[1])],
            "AUPRC": float(auprc),
            "AUPRC 95% CI": [float(auprc_ci[0]), float(auprc_ci[1])],
            "Brier score": float(brier),
            "train rows": int(len(train_df)),
            "test rows": int(len(test_df)),
            "target": "disease_vs_normal",
            "positive class": "AMD/DR/glaucoma",
            "negative class": "normal",
            "feature max dimension": 512,
        }
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics_summary, file, indent=2)
    return calibration_path, metrics_path, metrics_summary


@app.cell
def _(
    calibration_path,
    metrics_path,
    metrics_summary,
    mo,
    test_df,
    train_df,
    train_message,
):
    if metrics_summary is None:
        reason = train_message or "Local FIVES data or clean binary labels are unavailable."
        evaluation_message = f"""
            ## Evaluation

            `{calibration_path}` not generated.

            ```text
            {reason}
            ```
            """
    else:
        evaluation_message = f"""
            ## Evaluation

            Train rows: `{len(train_df)}`. Test rows: `{len(test_df)}`.

            Metrics: `{metrics_summary}`

            Saved calibration figure to `{calibration_path}`.
            Saved metrics summary to `{metrics_path}`.

            Limitation: FIVES disease labels are not ADRD labels, and fundus imaging is not OCTA.
            This section demonstrates validation and calibration mechanics, not ADRD biology.
            """
    (
        mo.vstack(
            [
                mo.md(evaluation_message),
                mo.image(
                    calibration_path,
                    alt="FIVES calibration curve and probability distribution",
                ),
            ]
        )
        if metrics_summary is not None and calibration_path.exists()
        else mo.md(evaluation_message)
    )
    return


if __name__ == "__main__":
    app.run()
