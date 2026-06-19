import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    from retivasc.artifacts import audit_mask_artifacts
    from retivasc.embeddings import (
        LAYER_ORDER,
        MASK_EMBEDDING_FEATURES,
        build_mask_feature_table,
        normalize_diagnosis,
        safe_write_table,
        write_json,
    )
    from retivasc.io import DataNotFoundError, load_rose_manifest
    from retivasc.layer_contrasts import aggregate_subject_features, compute_layer_contrasts
    from retivasc.outliers import compute_outlier_scores, compute_outlier_sensitivity
    from retivasc.plots_effects import (
        plot_feature_qc_heatmap,
        plot_layer_contrast_effect_sizes,
        plot_layer_effect_sizes,
        plot_mask_artifact_audit,
        plot_outlier_audit_panel,
        plot_subject_level_pca,
    )
    from retivasc.stats import (
        compare_groups_featurewise,
        fit_layer_mixed_effects,
        infer_feature_columns,
        summarize_rose_feature_table,
        validate_rose_feature_table,
    )

    return (
        DataNotFoundError,
        LAYER_ORDER,
        MASK_EMBEDDING_FEATURES,
        Path,
        aggregate_subject_features,
        audit_mask_artifacts,
        build_mask_feature_table,
        compare_groups_featurewise,
        compute_layer_contrasts,
        compute_outlier_scores,
        compute_outlier_sensitivity,
        fit_layer_mixed_effects,
        infer_feature_columns,
        json,
        load_rose_manifest,
        mo,
        normalize_diagnosis,
        pd,
        plot_feature_qc_heatmap,
        plot_layer_contrast_effect_sizes,
        plot_layer_effect_sizes,
        plot_mask_artifact_audit,
        plot_outlier_audit_panel,
        plot_subject_level_pca,
        safe_write_table,
        summarize_rose_feature_table,
        validate_rose_feature_table,
        write_json,
    )


@app.cell
def _(mo):
    mo.md("""
    # ROSE-1 Layer-Aware Feature Statistics

    This notebook follows up the ROSE mask-embedding result with subject-level,
    layer-aware exploratory statistics. It compares AD and control masks only after
    preserving the subject structure of the data: one subject can contribute SVC,
    DVC, and SVC+DVC rows.

    **Claim boundary:** this is not a ROSE classifier, AUROC analysis, calibration
    analysis, or diagnostic claim. It is a cautious feature/QC audit for a demo
    retinal vascular processing package.
    """)
    return


@app.cell
def _(
    DataNotFoundError,
    LAYER_ORDER,
    MASK_EMBEDDING_FEATURES,
    Path,
    aggregate_subject_features,
    audit_mask_artifacts,
    build_mask_feature_table,
    compare_groups_featurewise,
    compute_layer_contrasts,
    compute_outlier_scores,
    compute_outlier_sensitivity,
    fit_layer_mixed_effects,
    infer_feature_columns,
    json,
    load_rose_manifest,
    normalize_diagnosis,
    pd,
    plot_feature_qc_heatmap,
    plot_layer_contrast_effect_sizes,
    plot_layer_effect_sizes,
    plot_mask_artifact_audit,
    plot_outlier_audit_panel,
    plot_subject_level_pca,
    safe_write_table,
    summarize_rose_feature_table,
    validate_rose_feature_table,
    write_json,
):
    OUTPUT_DIR = Path("outputs/rose_layer_stats")
    REPORT_DIR = Path("reports")
    FIGURE_DIR = Path("figures")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    N_BOOT = 500
    N_PERM = 1000

    def _read_table(path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def _load_or_build_feature_table() -> tuple[pd.DataFrame, str, str | None]:
        candidates = [
            Path("outputs/rose_embeddings/rose_mask_features_long.parquet"),
            Path("outputs/rose_embeddings/rose_mask_features_long.csv"),
            Path("reports/rose_mask_embedding_features.parquet"),
            Path("reports/rose_mask_embedding_features.csv"),
        ]
        for path in candidates:
            if path.exists():
                return _read_table(path), str(path), None

        try:
            manifest = load_rose_manifest(Path("data/raw/rose"))
        except DataNotFoundError as exc:
            return pd.DataFrame(), "unavailable", str(exc)
        except ValueError:
            manifest = load_rose_manifest(Path("data/raw/rose"), require_split_safe=False)

        rose1_manifest = manifest.loc[
            (manifest["dataset"].astype("string") == "ROSE-1")
            & manifest["layer"].astype("string").isin(LAYER_ORDER)
        ].copy()
        rose1_manifest["diagnosis"] = rose1_manifest["diagnosis"].map(normalize_diagnosis)
        rose1_manifest["label_source"] = rose1_manifest["label_source"].fillna("unknown")
        if rose1_manifest.empty:
            return pd.DataFrame(), "unavailable", "No ROSE-1 rows were found."

        feature_table = build_mask_feature_table(rose1_manifest, min_component_size=0)
        feature_out = Path("outputs/rose_embeddings/rose_mask_features_long.parquet")
        written = safe_write_table(feature_table, feature_out)
        feature_table.to_csv(
            Path("outputs/rose_embeddings/rose_mask_features_long.csv"),
            index=False,
        )
        return feature_table, str(written), None

    def _write_csv(df: pd.DataFrame, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return str(path)

    def _layer_effects(table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        rows = []
        for layer in LAYER_ORDER:
            layer_df = table.loc[table["layer"].astype("string") == layer].copy()
            effects = compare_groups_featurewise(
                layer_df,
                features,
                n_boot=N_BOOT,
                n_perm=N_PERM,
                random_state=101 + len(rows),
            )
            if effects.empty:
                continue
            effects.insert(0, "analysis", "layer_specific")
            effects.insert(1, "layer", layer)
            rows.append(effects)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def _effect_table(table: pd.DataFrame, features: list[str], analysis: str) -> pd.DataFrame:
        effects = compare_groups_featurewise(
            table,
            features,
            n_boot=N_BOOT,
            n_perm=N_PERM,
            random_state=303,
        )
        if not effects.empty:
            effects.insert(0, "analysis", analysis)
        return effects

    def _top_effects_html(df: pd.DataFrame, *, rows: int = 8) -> str:
        if df.empty:
            return "<p>No effect rows were generated.</p>"
        display_cols = [
            col
            for col in [
                "analysis",
                "layer",
                "feature",
                "diff_median_AD_minus_control",
                "bootstrap_CI_low",
                "bootstrap_CI_high",
                "hedges_g",
                "fdr_bh_permutation_p",
            ]
            if col in df.columns
        ]
        work = df.copy()
        work["_abs"] = pd.to_numeric(work.get("hedges_g"), errors="coerce").abs()
        work = work.sort_values("_abs", ascending=False).head(rows)
        return work[display_cols].to_html(index=False, float_format=lambda value: f"{value:.4g}")

    def _write_report(
        *,
        summary: dict,
        paths: dict[str, str],
        pca_metadata: dict,
        layer_effects: pd.DataFrame,
        contrast_effects: pd.DataFrame,
        subject_effects: pd.DataFrame,
        mixed_effects: pd.DataFrame,
    ) -> str:
        report_path = REPORT_DIR / "rose_layer_aware_statistics.html"
        pca_ev1 = pca_metadata.get("explained_variance_ratio_pc1")
        pca_ev2 = pca_metadata.get("explained_variance_ratio_pc2")
        pca_text = (
            f"Subject-level PC1 explains {pca_ev1:.1%} and PC2 explains {pca_ev2:.1%} "
            "of the scaled aggregated/contrast feature variance."
            if pca_ev1 is not None and pca_ev2 is not None
            else "Subject-level PCA was skipped or did not have variance metadata."
        )
        mixed_warning = ""
        if not mixed_effects.empty and mixed_effects.get("model_warning") is not None:
            warnings = sorted(
                set(str(value) for value in mixed_effects["model_warning"] if str(value))
            )
            if warnings:
                mixed_warning = (
                    "<p><strong>Mixed-effects note:</strong> " + "; ".join(warnings[:3]) + "</p>"
                )
        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>ROSE layer-aware feature statistics</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1f2933; }}
    main {{ max-width: 1160px; margin: auto; }}
    img {{ max-width: 100%; border: 1px solid #d8e0e5; margin: 1rem 0 2rem; }}
    table {{ border-collapse: collapse; font-size: 0.88rem; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #d8e0e5; padding: 0.35rem 0.5rem; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    code {{ background: #eef3f6; padding: 0.1rem 0.25rem; border-radius: 3px; }}
  </style>
</head>
<body>
<main>
  <h1>ROSE-1 layer-aware feature statistics</h1>
  <p>
    This report is a demo for processing retinal vascular OCTA data. It uses
    manual/reference ROSE-1 vessel masks to compute interpretable vascular features,
    then asks whether AD/control differences are layer-specific, subject-level, or
    sensitive to outliers and mask artifacts.
  </p>
  <p>
    <strong>Claim boundary:</strong> no ROSE classification, AUROC, calibration, or
    AD diagnostic claim is made. Effect sizes and bootstrap intervals are emphasized
    over nominal p-values.
  </p>
  <h2>Dataset summary</h2>
  <ul>
    <li>Subjects: <code>{summary.get("n_subjects", 0)}</code></li>
    <li>Subject-layer rows: <code>{summary.get("n_rows", 0)}</code></li>
    <li>Rows per layer: <code>{summary.get("rows_per_layer", {})}</code></li>
    <li>Subjects by diagnosis: <code>{summary.get("subject_counts_by_diagnosis", {})}</code></li>
    <li>Feature source: <code>{summary.get("feature_source", "unknown")}</code></li>
  </ul>
  <h2>Layer-specific effects</h2>
  <p>
    Each panel compares AD and control within one OCTA slab. This avoids treating
    the 117 subject-layer rows as 117 independent subjects.
  </p>
  <img src="../figures/rose_layer_effect_sizes.png" alt="Layer-specific effect sizes" />
  {_top_effects_html(layer_effects)}
  <h2>Paired layer contrasts</h2>
  <p>
    These features compare layers within the same subject, for example DVC minus SVC
    or log-ratios. This can reveal layer relationships that global embeddings hide.
  </p>
  <img src="../figures/rose_layer_contrast_effect_sizes.png" alt="Layer contrast effects" />
  {_top_effects_html(contrast_effects)}
  <h2>Subject-level PCA</h2>
  <p>{pca_text}</p>
  <img src="../figures/rose_subject_level_pca.png" alt="Subject-level PCA" />
  <h2>Outlier and QC audit</h2>
  <p>
    Outlier checks use robust feature z-scores and PCA leverage within layers. If an
    effect changes sign or magnitude after removing flagged rows, it is marked
    outlier-sensitive in the machine-readable outputs.
  </p>
  <img src="../figures/rose_outlier_audit_panel.png" alt="Outlier audit panel" />
  <img src="../figures/rose_feature_qc_heatmap.png" alt="Feature QC heatmap" />
  <img src="../figures/rose_mask_artifact_audit.png" alt="Mask artifact audit" />
  <h2>Subject-level effects</h2>
  {_top_effects_html(subject_effects)}
  <h2>Mixed-effects caveat</h2>
  <p>
    Mixed-effects models are exploratory and small-sample estimates may be unstable.
    They are included only as a repeated-measures sanity check.
  </p>
  {mixed_warning}
  <h2>Outputs</h2>
  <ul>
    {"".join(f"<li><code>{key}</code>: <code>{value}</code></li>" for key, value in paths.items())}
  </ul>
  <h2>Interpretation</h2>
  <p>
    The cautious interpretation remains that ROSE-1 global mask-derived features do
    not provide a robust AD/control separation by themselves. The useful next step is
    to inspect layer-aware effects, paired contrasts, outlier sensitivity, and mask
    artifacts before carrying any feature into a future multimodal ADRD model.
  </p>
</main>
</body>
</html>
"""
        report_path.write_text(html, encoding="utf-8")
        return str(report_path)

    feature_table, feature_source, load_error = _load_or_build_feature_table()
    if feature_table.empty:
        analysis_summary = {
            "status": "skipped",
            "reason": load_error or "No feature rows were available.",
            "feature_source": feature_source,
            "claim_boundary": "No ROSE classification or AD diagnostic claim is made.",
        }
        summary_path = write_json(OUTPUT_DIR / "analysis_summary.json", analysis_summary)
        report_path = REPORT_DIR / "rose_layer_aware_statistics.html"
        report_path.write_text(
            "<!doctype html><html><body><h1>ROSE layer-aware feature statistics</h1>"
            f"<p>Analysis skipped: {analysis_summary['reason']}</p>"
            "<p>No ROSE classification or AD diagnostic claim is made.</p></body></html>",
            encoding="utf-8",
        )
        result_message = f"Analysis skipped: {analysis_summary['reason']}"
        paths = {"analysis_summary": str(summary_path), "report": str(report_path)}
    else:
        feature_table = feature_table.copy()
        feature_table["diagnosis"] = feature_table["diagnosis"].map(normalize_diagnosis)
        validate_rose_feature_table(feature_table)
        feature_cols = [
            feature for feature in MASK_EMBEDDING_FEATURES if feature in feature_table.columns
        ]
        if not feature_cols:
            feature_cols = infer_feature_columns(feature_table)

        table_summary = summarize_rose_feature_table(feature_table, feature_cols)
        table_summary["feature_source"] = feature_source

        subject_features = aggregate_subject_features(feature_table, feature_cols)
        layer_contrast_features = compute_layer_contrasts(feature_table, feature_cols)

        layer_specific_effects = _layer_effects(feature_table, feature_cols)
        subject_feature_cols = infer_feature_columns(subject_features)
        subject_level_effects = _effect_table(
            subject_features,
            subject_feature_cols,
            "subject_aggregated",
        )
        contrast_feature_cols = infer_feature_columns(layer_contrast_features)
        layer_contrast_effects = _effect_table(
            layer_contrast_features,
            contrast_feature_cols,
            "layer_contrast",
        )

        outlier_scores = compute_outlier_scores(feature_table, feature_cols)
        outlier_flags = outlier_scores[["subject_id", "layer", "is_feature_outlier"]].copy()
        table_with_flags = feature_table.merge(
            outlier_flags, on=["subject_id", "layer"], how="left"
        )
        filtered_table = table_with_flags.loc[
            ~table_with_flags["is_feature_outlier"].fillna(False)
        ].drop(columns=["is_feature_outlier"])
        filtered_layer_effects = _layer_effects(filtered_table, feature_cols)

        subject_outliers = (
            outlier_scores.groupby("subject_id")["is_feature_outlier"]
            .any()
            .rename("is_feature_outlier")
        )
        subject_features = subject_features.merge(subject_outliers, on="subject_id", how="left")
        subject_features["is_feature_outlier"] = subject_features["is_feature_outlier"].fillna(
            False
        )
        filtered_subject_features = subject_features.loc[
            ~subject_features["is_feature_outlier"]
        ].copy()
        filtered_subject_effects = _effect_table(
            filtered_subject_features,
            subject_feature_cols,
            "subject_aggregated",
        )
        contrast_with_flags = layer_contrast_features.merge(
            subject_outliers, on="subject_id", how="left"
        )
        contrast_with_flags["is_feature_outlier"] = contrast_with_flags[
            "is_feature_outlier"
        ].fillna(False)
        filtered_contrast_effects = _effect_table(
            contrast_with_flags.loc[~contrast_with_flags["is_feature_outlier"]],
            contrast_feature_cols,
            "layer_contrast",
        )
        sensitivity = pd.concat(
            [
                compute_outlier_sensitivity(
                    layer_specific_effects,
                    filtered_layer_effects,
                    key_cols=("analysis", "layer", "feature"),
                ),
                compute_outlier_sensitivity(
                    subject_level_effects,
                    filtered_subject_effects,
                    key_cols=("analysis", "feature"),
                ),
                compute_outlier_sensitivity(
                    layer_contrast_effects,
                    filtered_contrast_effects,
                    key_cols=("analysis", "feature"),
                ),
            ],
            ignore_index=True,
        )

        artifact_audit = audit_mask_artifacts(
            feature_table,
            n_boot=N_BOOT,
            n_perm=N_PERM,
            random_state=505,
        )
        mixed_effects = fit_layer_mixed_effects(feature_table, feature_cols)

        subject_pca_base = subject_features.drop(columns=["is_feature_outlier"]).merge(
            layer_contrast_features[["subject_id", *contrast_feature_cols]],
            on="subject_id",
            how="left",
        )
        subject_pca_base = subject_pca_base.merge(subject_outliers, on="subject_id", how="left")
        subject_pca_base["is_feature_outlier"] = subject_pca_base["is_feature_outlier"].fillna(
            False
        )
        subject_pca_cols = [
            col
            for col in infer_feature_columns(subject_pca_base, include_qc=True)
            if "_div_" not in col
        ]
        _, pca_metadata = plot_subject_level_pca(
            subject_pca_base,
            subject_pca_cols,
            FIGURE_DIR / "rose_subject_level_pca.png",
        )
        plot_layer_effect_sizes(layer_specific_effects, FIGURE_DIR / "rose_layer_effect_sizes.png")
        plot_layer_contrast_effect_sizes(
            layer_contrast_effects,
            FIGURE_DIR / "rose_layer_contrast_effect_sizes.png",
        )
        plot_outlier_audit_panel(outlier_scores, FIGURE_DIR / "rose_outlier_audit_panel.png")
        plot_feature_qc_heatmap(table_summary, FIGURE_DIR / "rose_feature_qc_heatmap.png")
        plot_mask_artifact_audit(artifact_audit, FIGURE_DIR / "rose_mask_artifact_audit.png")

        paths = {
            "layer_specific_effects": _write_csv(
                layer_specific_effects,
                OUTPUT_DIR / "layer_specific_effects.csv",
            ),
            "layer_contrast_features": _write_csv(
                layer_contrast_features,
                OUTPUT_DIR / "layer_contrast_features.csv",
            ),
            "layer_contrast_effects": _write_csv(
                layer_contrast_effects,
                OUTPUT_DIR / "layer_contrast_effects.csv",
            ),
            "subject_aggregated_features": _write_csv(
                subject_features,
                OUTPUT_DIR / "subject_aggregated_features.csv",
            ),
            "subject_level_effects": _write_csv(
                subject_level_effects,
                OUTPUT_DIR / "subject_level_effects.csv",
            ),
            "mixed_effects_summary": _write_csv(
                mixed_effects,
                OUTPUT_DIR / "mixed_effects_summary.csv",
            ),
            "outlier_scores": _write_csv(outlier_scores, OUTPUT_DIR / "outlier_scores.csv"),
            "outlier_sensitivity": _write_csv(
                sensitivity,
                OUTPUT_DIR / "outlier_sensitivity.csv",
            ),
            "artifact_audit": _write_csv(artifact_audit, OUTPUT_DIR / "artifact_audit.csv"),
            "feature_table_summary": _write_csv(
                pd.DataFrame(
                    {
                        "feature": list(table_summary["missingness_per_feature"].keys()),
                        "missingness": list(table_summary["missingness_per_feature"].values()),
                        "variance": [
                            table_summary["variance_per_feature"].get(feature, 0.0)
                            for feature in table_summary["missingness_per_feature"]
                        ],
                    }
                ),
                OUTPUT_DIR / "feature_table_summary.csv",
            ),
        }
        analysis_summary = {
            **table_summary,
            "status": "complete",
            "feature_source": feature_source,
            "feature_cols": feature_cols,
            "subject_level_feature_count": len(subject_feature_cols),
            "layer_contrast_feature_count": len(contrast_feature_cols),
            "subject_pca_feature_count": len(subject_pca_cols),
            "resampling": {"n_boot": N_BOOT, "n_perm": N_PERM},
            "n_flagged_subject_layer_outliers": int(outlier_scores["is_feature_outlier"].sum()),
            "n_flagged_subjects": int(subject_outliers.sum()),
            "subject_pca": pca_metadata,
            "claim_boundary": "No ROSE classification or AD diagnostic claim is made.",
        }
        summary_path = write_json(OUTPUT_DIR / "analysis_summary.json", analysis_summary)
        paths["analysis_summary"] = str(summary_path)
        paths["report"] = _write_report(
            summary=analysis_summary,
            paths=paths,
            pca_metadata=pca_metadata,
            layer_effects=layer_specific_effects,
            contrast_effects=layer_contrast_effects,
            subject_effects=subject_level_effects,
            mixed_effects=mixed_effects,
        )
        result_message = (
            f"Completed layer-aware statistics for {analysis_summary['n_subjects']} subjects "
            f"and {analysis_summary['n_rows']} subject-layer rows."
        )

    return analysis_summary, paths, result_message


@app.cell
def _(analysis_summary, mo, paths, result_message):
    mo.md(f"""
    ## Analysis Status

    `{result_message}`

    - Summary: `{paths.get("analysis_summary")}`
    - Report: `{paths.get("report")}`
    - Status: `{analysis_summary.get("status")}`

    The report explicitly avoids ROSE classification, AUROC, calibration, or AD
    diagnostic claims.
    """)
    return


@app.cell
def _(Path, mo, pd):
    def layer_stats_image(path: str | Path, alt: str):
        image_path = Path(path)
        if image_path.exists():
            return mo.image(image_path, alt=alt)
        return mo.md(f"_Missing `{image_path}`._")

    def layer_stats_table(
        path: str | Path | None,
        *,
        columns: list[str] | None = None,
        rows: int = 10,
        sort_col: str | None = "hedges_g",
        abs_sort: bool = True,
    ):
        if path is None:
            return mo.md("_No table path was produced._")
        table_path = Path(path)
        if not table_path.exists():
            return mo.md(f"_Missing `{table_path}`._")
        table = pd.read_csv(table_path)
        if table.empty:
            return mo.md(f"_`{table_path}` is empty._")
        display = table.copy()
        if sort_col is not None and sort_col in display.columns:
            sort_values = pd.to_numeric(display[sort_col], errors="coerce")
            if abs_sort:
                sort_values = sort_values.abs()
            display = (
                display.assign(_sort=sort_values)
                .sort_values("_sort", ascending=False, na_position="last")
                .drop(columns=["_sort"])
            )
        if columns is not None:
            display = display[[col for col in columns if col in display.columns]]
        display = display.head(rows)
        return mo.Html(
            "<div style='overflow-x:auto'>"
            + display.to_html(
                index=False, border=0, na_rep="", float_format=lambda value: f"{value:.4g}"
            )
            + "</div>"
        )

    def percent(value):
        return "not available" if value is None else f"{value:.1%}"

    def number(value, digits: int = 3):
        try:
            return f"{float(value):.{digits}g}"
        except (TypeError, ValueError):
            return "not available"

    return layer_stats_image, layer_stats_table, number, percent


@app.cell
def _(analysis_summary, mo, number, percent):
    pca = analysis_summary.get("subject_pca", {})
    separation = pca.get("separation", {})
    resampling = analysis_summary.get("resampling", {})
    flagged_rows = analysis_summary.get("n_flagged_subject_layer_outliers", 0)
    flagged_subjects = analysis_summary.get("n_flagged_subjects", 0)
    mo.md(f"""
    ## Results Overview

    - Subjects: `{analysis_summary.get("n_subjects", 0)}`.
    - Subject-layer rows: `{analysis_summary.get("n_rows", 0)}`.
    - Rows per layer: `{analysis_summary.get("rows_per_layer", {})}`.
    - Subjects by diagnosis: `{analysis_summary.get("subject_counts_by_diagnosis", {})}`.
    - Resampling: `{resampling.get("n_boot", "NA")}` bootstrap draws and
      `{resampling.get("n_perm", "NA")}` permutations per feature comparison.
    - Flagged subject-layer outliers: `{flagged_rows}`.
    - Flagged subjects: `{flagged_subjects}`.

    Subject-level PCA uses aggregated features plus paired layer differences/log-ratios.
    PC1 explains `{percent(pca.get("explained_variance_ratio_pc1"))}` and PC2 explains
    `{percent(pca.get("explained_variance_ratio_pc2"))}`. The AD/control centroid
    permutation p-value is `{number(separation.get("p_value"))}`, with silhouette
    `{number(separation.get("silhouette_score"))}`.

    **Interpretation:** this remains an exploratory feature/QC audit. No ROSE
    classifier, AUROC, calibration, or AD diagnostic claim is made.
    """)
    return


@app.cell
def _(Path, layer_stats_image, layer_stats_table, mo, paths):
    layer_columns = [
        "layer",
        "feature",
        "n_AD",
        "n_control",
        "diff_median_AD_minus_control",
        "bootstrap_CI_low",
        "bootstrap_CI_high",
        "hedges_g",
        "fdr_bh_permutation_p",
    ]
    mo.vstack(
        [
            mo.md("""
            ## Layer-Specific AD/Control Effects

            Each panel compares AD and control within one OCTA layer. This avoids
            treating SVC, DVC, and SVC+DVC rows from the same subject as independent
            people. The table shows the largest absolute standardized effects.
            """),
            layer_stats_image(
                Path("figures/rose_layer_effect_sizes.png"),
                "ROSE layer-specific effect sizes",
            ),
            layer_stats_table(
                paths.get("layer_specific_effects"),
                columns=layer_columns,
                rows=12,
            ),
        ]
    )
    return


@app.cell
def _(Path, layer_stats_image, layer_stats_table, mo, paths):
    contrast_columns = [
        "feature",
        "n_AD",
        "n_control",
        "diff_median_AD_minus_control",
        "bootstrap_CI_low",
        "bootstrap_CI_high",
        "hedges_g",
        "fdr_bh_permutation_p",
    ]
    mo.vstack(
        [
            mo.md("""
            ## Paired Layer Contrasts

            These rows compare layers within the same subject, such as DVC minus SVC
            or SVC+DVC log-ratio DVC. They ask whether layer relationships carry
            signal that global mask embeddings hide.
            """),
            layer_stats_image(
                Path("figures/rose_layer_contrast_effect_sizes.png"),
                "ROSE paired layer contrast effect sizes",
            ),
            layer_stats_table(
                paths.get("layer_contrast_effects"),
                columns=contrast_columns,
                rows=14,
            ),
        ]
    )
    return


@app.cell
def _(Path, layer_stats_image, layer_stats_table, mo, paths):
    subject_columns = [
        "feature",
        "n_AD",
        "n_control",
        "diff_median_AD_minus_control",
        "bootstrap_CI_low",
        "bootstrap_CI_high",
        "hedges_g",
        "fdr_bh_permutation_p",
    ]
    mo.vstack(
        [
            mo.md("""
            ## Subject-Level Summary

            This view collapses each participant to one row before testing group
            differences. The PCA uses subject-level summaries and paired layer
            contrasts; one dot is one subject.
            """),
            layer_stats_image(
                Path("figures/rose_subject_level_pca.png"),
                "ROSE subject-level PCA",
            ),
            mo.md("### Largest Subject-Level Effects"),
            layer_stats_table(
                paths.get("subject_level_effects"),
                columns=subject_columns,
                rows=12,
            ),
        ]
    )
    return


@app.cell
def _(Path, layer_stats_image, layer_stats_table, mo, paths):
    outlier_columns = [
        "subject_id",
        "image_id",
        "layer",
        "diagnosis",
        "max_abs_zscore",
        "mean_abs_zscore",
        "pca_leverage_score",
        "n_extreme_features_abs_z_gt_3",
        "top_extreme_features",
        "is_feature_outlier",
    ]
    sensitivity_columns = [
        "analysis",
        "layer",
        "feature",
        "effect_full",
        "effect_without_outliers",
        "delta_effect",
        "p_full",
        "p_without_outliers",
        "interpretation_flag",
    ]
    mo.vstack(
        [
            mo.md("""
            ## Outlier Sensitivity

            Outlier scores are computed within layer using robust feature z-scores and
            PCA leverage. The panel shows masks/skeletons/components for the most
            extreme rows when masks are available locally.
            """),
            layer_stats_image(
                Path("figures/rose_outlier_audit_panel.png"),
                "ROSE outlier audit panel",
            ),
            mo.md("### Highest Outlier Scores"),
            layer_stats_table(
                paths.get("outlier_scores"),
                columns=outlier_columns,
                rows=10,
                sort_col="max_abs_zscore",
            ),
            mo.md("### Most Outlier-Sensitive Effects"),
            layer_stats_table(
                paths.get("outlier_sensitivity"),
                columns=sensitivity_columns,
                rows=10,
                sort_col="delta_effect",
            ),
        ]
    )
    return


@app.cell
def _(Path, layer_stats_image, layer_stats_table, mo, paths):
    artifact_columns = [
        "variable",
        "variable_type",
        "test",
        "n_AD",
        "n_control",
        "diff_median_AD_minus_control",
        "hedges_g",
        "fdr_bh_permutation_p",
        "categorical_p",
    ]
    mo.vstack(
        [
            mo.md("""
            ## Feature QC and Mask Artifact Audit

            These checks ask whether diagnosis labels are associated with mask size,
            foreground fraction, fragmentation, official split, layer, or layer
            availability. If QC variables differ by diagnosis, feature differences
            should not be read biologically without adjustment.
            """),
            layer_stats_image(
                Path("figures/rose_feature_qc_heatmap.png"),
                "ROSE feature QC heatmap",
            ),
            layer_stats_image(
                Path("figures/rose_mask_artifact_audit.png"),
                "ROSE mask artifact audit",
            ),
            layer_stats_table(
                paths.get("artifact_audit"),
                columns=artifact_columns,
                rows=12,
                sort_col="hedges_g",
            ),
        ]
    )
    return


@app.cell
def _(layer_stats_table, mo, paths):
    mixed_columns = [
        "feature",
        "n_subjects",
        "n_rows",
        "converged",
        "coef_diagnosis_AD",
        "p_diagnosis_AD",
        "coef_diagnosis_AD_x_layer_DVC",
        "p_interaction_DVC",
        "coef_diagnosis_AD_x_layer_SVCplusDVC",
        "p_interaction_SVCplusDVC",
        "model_warning",
    ]
    mo.vstack(
        [
            mo.md("""
            ## Mixed-Effects Sanity Check

            When `statsmodels` is installed, this section fits exploratory repeated
            measures models with subject random intercepts. In environments without
            `statsmodels`, the CSV records a clear skipped-model warning.
            """),
            layer_stats_table(
                paths.get("mixed_effects_summary"),
                columns=mixed_columns,
                rows=12,
                sort_col="p_diagnosis_AD",
                abs_sort=False,
            ),
        ]
    )
    return


@app.cell
def _(mo, paths):
    mo.md(f"""
    ## Generated Files

    - Layer-specific effects: `{paths.get("layer_specific_effects")}`
    - Layer-contrast effects: `{paths.get("layer_contrast_effects")}`
    - Subject-level effects: `{paths.get("subject_level_effects")}`
    - Outlier scores: `{paths.get("outlier_scores")}`
    - Artifact audit: `{paths.get("artifact_audit")}`
    - Summary JSON: `{paths.get("analysis_summary")}`
    - HTML report: `{paths.get("report")}`
    """)
    return


if __name__ == "__main__":
    app.run()
