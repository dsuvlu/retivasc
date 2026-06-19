import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    from retivasc.embeddings import (
        LAYER_ORDER,
        MASK_EMBEDDING_FEATURES,
        ROSE_MASK_EMBEDDING_CLAIM_BOUNDARY,
        attach_metadata,
        build_mask_feature_table,
        compute_group_separation,
        fit_pca_embedding,
        fit_tsne_embedding,
        fit_umap_embedding,
        normalize_diagnosis,
        prepare_embedding_matrix,
        safe_write_table,
        write_json,
    )
    from retivasc.io import DataNotFoundError, load_rose_manifest
    from retivasc.plots_embeddings import (
        plot_layer_faceted_embedding,
        plot_pca_feature_loadings,
    )

    return (
        DataNotFoundError,
        LAYER_ORDER,
        MASK_EMBEDDING_FEATURES,
        Path,
        ROSE_MASK_EMBEDDING_CLAIM_BOUNDARY,
        attach_metadata,
        build_mask_feature_table,
        compute_group_separation,
        fit_pca_embedding,
        fit_tsne_embedding,
        fit_umap_embedding,
        json,
        load_rose_manifest,
        mo,
        normalize_diagnosis,
        pd,
        plot_layer_faceted_embedding,
        plot_pca_feature_loadings,
        plt,
        prepare_embedding_matrix,
        safe_write_table,
        write_json,
    )


@app.cell
def _(ROSE_MASK_EMBEDDING_CLAIM_BOUNDARY, mo):
    mo.md(f"""
    # ROSE-1 Mask-Derived Vascular Embeddings

    This notebook uses manual/reference vessel masks, not raw OCTA intensity images,
    to visualize ROSE-1 vascular phenotype space. Each dot is one subject in one OCTA
    layer. Color shows explicit AD/control labels recorded in the local manifest.

    `{ROSE_MASK_EMBEDDING_CLAIM_BOUNDARY}`
    """)
    return


@app.cell
def _(DataNotFoundError, LAYER_ORDER, Path, load_rose_manifest, normalize_diagnosis, pd):
    rose_root = Path("data/raw/rose")
    rose_error = None
    rose_warning = None
    try:
        rose_manifest = load_rose_manifest(rose_root)
    except DataNotFoundError as exc:
        rose_manifest = None
        rose_error = str(exc)
    except ValueError as exc:
        rose_warning = str(exc)
        try:
            rose_manifest = load_rose_manifest(rose_root, require_split_safe=False)
        except Exception as fallback_exc:
            rose_manifest = None
            rose_error = str(fallback_exc)

    if rose_manifest is None:
        rose1_manifest = pd.DataFrame()
    else:
        rose1_manifest = rose_manifest.loc[
            (rose_manifest["dataset"].astype("string") == "ROSE-1")
            & rose_manifest["layer"].astype("string").isin(LAYER_ORDER)
        ].copy()
        rose1_manifest["diagnosis"] = rose1_manifest["diagnosis"].map(normalize_diagnosis)
        rose1_manifest["label_source"] = rose1_manifest["label_source"].fillna("unknown")

    return rose1_manifest, rose_error, rose_warning


@app.cell
def _(mo, rose1_manifest, rose_error, rose_warning):
    if rose1_manifest.empty:
        message = f"""
        ## Manifest Audit

        Waiting for local ROSE-1 data.

        ```text
        {rose_error or "No ROSE-1 rows were loaded."}
        ```
        """
    else:
        _audit_label_counts = (
            rose1_manifest.groupby(["layer", "diagnosis"], dropna=False)
            .size()
            .unstack(fill_value=0)
            .to_dict()
        )
        warning_block = (
            f"""

            Split-safety warning from automatic discovery:

            ```text
            {rose_warning}
            ```
            """
            if rose_warning
            else ""
        )
        message = f"""
        ## Manifest Audit

        - Rows: `{len(rose1_manifest)}`
        - Subjects: `{rose1_manifest["subject_id"].nunique()}`
        - Layers: `{sorted(rose1_manifest["layer"].dropna().unique())}`
        - Diagnosis counts by layer: `{_audit_label_counts}`

        The embedding workflow does not infer diagnosis from filenames. AD/control
        rows must carry a non-empty `label_source`.
        {warning_block}
        """
    mo.md(message)
    return


@app.cell
def _(
    MASK_EMBEDDING_FEATURES,
    build_mask_feature_table,
    prepare_embedding_matrix,
    rose1_manifest,
    safe_write_table,
):
    feature_path = "reports/rose_mask_embedding_features.parquet"
    if rose1_manifest.empty:
        feature_table = rose1_manifest.copy()
        embedding_matrix = None
        preprocessing_metadata = {}
    else:
        feature_table = build_mask_feature_table(rose1_manifest, min_component_size=0)
        embedding_matrix, preprocessing_metadata = prepare_embedding_matrix(
            feature_table,
            MASK_EMBEDDING_FEATURES,
        )
        feature_path = str(safe_write_table(feature_table, feature_path))
    return embedding_matrix, feature_path, feature_table, preprocessing_metadata


@app.cell
def _(
    MASK_EMBEDDING_FEATURES,
    attach_metadata,
    feature_table,
    fit_pca_embedding,
    fit_tsne_embedding,
    fit_umap_embedding,
    plot_layer_faceted_embedding,
    plot_pca_feature_loadings,
    plt,
    embedding_matrix,
):
    def _placeholder_figure(title, message, out_path):
        fig, axis = plt.subplots(figsize=(8, 3.5))
        axis.text(0.5, 0.55, title, ha="center", va="center", fontsize=14)
        axis.text(0.5, 0.38, message, ha="center", va="center", fontsize=10, color="0.35")
        axis.set_axis_off()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        return fig

    if embedding_matrix is None or feature_table.empty:
        pca_coords = feature_table.copy()
        umap_coords = feature_table.copy()
        tsne_coords = feature_table.copy()
        pca_metadata = {}
        umap_metadata = {"skipped": "no ROSE-1 rows"}
        tsne_metadata = {}
    else:
        pca_raw_coords, pca_metadata = fit_pca_embedding(
            embedding_matrix,
            MASK_EMBEDDING_FEATURES,
        )
        pca_coords = attach_metadata(feature_table, pca_raw_coords, "pca")
        _ = plot_layer_faceted_embedding(
            pca_coords,
            "component_1",
            "component_2",
            title="ROSE-1 feature PCA from vessel masks",
            out_path="figures/rose_mask_embeddings_pca.png",
        )
        _ = plot_pca_feature_loadings(
            pca_metadata,
            out_path="figures/rose_mask_embedding_feature_loadings.png",
        )

        try:
            umap_raw_coords, umap_metadata = fit_umap_embedding(embedding_matrix)
            umap_coords = attach_metadata(feature_table, umap_raw_coords, "umap")
            _ = plot_layer_faceted_embedding(
                umap_coords,
                "component_1",
                "component_2",
                title="ROSE-1 feature UMAP from vessel masks",
                out_path="figures/rose_mask_embeddings_umap.png",
            )
        except ImportError as exc:
            umap_coords = feature_table.head(0).copy()
            umap_metadata = {"method": "umap", "skipped": str(exc)}
            _ = _placeholder_figure(
                "UMAP not generated",
                str(exc),
                "figures/rose_mask_embeddings_umap.png",
            )

        tsne_raw_coords, tsne_metadata = fit_tsne_embedding(embedding_matrix)
        tsne_coords = attach_metadata(feature_table, tsne_raw_coords, "tsne")
        _ = plot_layer_faceted_embedding(
            tsne_coords,
            "component_1",
            "component_2",
            title="ROSE-1 feature t-SNE from vessel masks",
            out_path="figures/rose_mask_embeddings_tsne.png",
        )
    return pca_coords, pca_metadata, tsne_coords, tsne_metadata, umap_coords, umap_metadata


@app.cell
def _(
    compute_group_separation,
    feature_table,
    json,
    pca_coords,
    pca_metadata,
    pd,
    preprocessing_metadata,
    rose1_manifest,
    safe_write_table,
    tsne_coords,
    tsne_metadata,
    umap_coords,
    umap_metadata,
    write_json,
):
    coordinates = []
    separation = {}
    for method, coords in (
        ("pca", pca_coords),
        ("umap", umap_coords),
        ("tsne", tsne_coords),
    ):
        if not coords.empty and "component_1" in coords.columns:
            coordinates.append(coords)
            separation[method] = compute_group_separation(coords, n_permutations=250)
    coordinates_table = (
        pd.concat(coordinates, ignore_index=True) if coordinates else feature_table.head(0).copy()
    )
    coordinates_path = "reports/rose_mask_embedding_coordinates.parquet"
    if not coordinates_table.empty:
        coordinates_path = str(safe_write_table(coordinates_table, coordinates_path))

    _summary_label_counts = (
        feature_table.groupby(["layer", "diagnosis"], dropna=False).size().to_dict()
        if not feature_table.empty
        else {}
    )
    summary = {
        "n_subjects": int(feature_table["subject_id"].nunique()) if not feature_table.empty else 0,
        "n_rows": int(len(feature_table)),
        "layers": (
            sorted(feature_table["layer"].dropna().unique()) if not feature_table.empty else []
        ),
        "label_counts_by_layer": {
            str(key): int(value) for key, value in _summary_label_counts.items()
        },
        "features_used": list(preprocessing_metadata.get("feature_cols", [])),
        "n_missing_values_raw": preprocessing_metadata.get("n_missing_values_raw", 0),
        "n_missing_values_imputed": preprocessing_metadata.get("n_missing_values_imputed", 0),
        "pca": pca_metadata,
        "umap": umap_metadata,
        "tsne": tsne_metadata,
        "separation": separation,
        "manifest_rows": int(len(rose1_manifest)),
        "claim_boundary": (
            "Exploratory mask-derived visualization only; not predictive validation."
        ),
    }
    summary_path = write_json("reports/rose_mask_embedding_summary.json", summary)
    _summary_text = json.dumps(summary, indent=2, sort_keys=True, default=str)
    return coordinates_path, coordinates_table, summary, summary_path


@app.cell
def _(Path, coordinates_path, feature_path, summary, summary_path):
    def _write_embedding_report():
        report_path = Path("reports/rose_mask_embedding_report.html")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        pca_ev1 = summary.get("pca", {}).get("explained_variance_ratio_pc1")
        pca_ev2 = summary.get("pca", {}).get("explained_variance_ratio_pc2")
        variance_text = (
            f"PC1 explains {pca_ev1:.1%} and PC2 explains {pca_ev2:.1%} of scaled "
            "feature variance."
            if pca_ev1 is not None and pca_ev2 is not None
            else "PCA variance metadata are unavailable."
        )
        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>ROSE mask embedding report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1f2933; }}
    main {{ max-width: 1100px; margin: auto; }}
    img {{ max-width: 100%; border: 1px solid #d8e0e5; margin: 1rem 0 2rem; }}
    code {{ background: #eef3f6; padding: 0.1rem 0.25rem; border-radius: 3px; }}
  </style>
</head>
<body>
<main>
  <h1>ROSE-1 mask-derived vascular embeddings</h1>
  <p>
    This local report visualizes manual/reference vessel masks as interpretable vascular
    feature vectors. Each dot is one subject in one OCTA layer. Coordinates are fit
    jointly across subject-layer rows and faceted by SVC, DVC, and SVC+DVC.
  </p>
  <p><strong>Claim boundary:</strong> {summary["claim_boundary"]}</p>
  <p>{variance_text}</p>
  <h2>PCA</h2>
  <img src="../figures/rose_mask_embeddings_pca.png" alt="ROSE mask PCA embedding" />
  <h2>PCA feature loadings</h2>
  <img src="../figures/rose_mask_embedding_feature_loadings.png" alt="PCA loadings" />
  <h2>UMAP</h2>
  <img src="../figures/rose_mask_embeddings_umap.png" alt="ROSE mask UMAP embedding" />
  <h2>t-SNE</h2>
  <img src="../figures/rose_mask_embeddings_tsne.png" alt="ROSE mask t-SNE embedding" />
  <h2>Artifacts</h2>
  <p>
    Features: <code>{feature_path}</code><br />
    Coordinates: <code>{coordinates_path}</code><br />
    Summary: <code>{summary_path}</code>
  </p>
  <h2>Limitations</h2>
  <p>
    ROSE-1 is small. UMAP and t-SNE are parameter-sensitive and should not be read as
    evidence of validated disease separation. These plots motivate feature inspection;
    they do not estimate AUROC, calibration, or clinical risk.
  </p>
</main>
</body>
</html>
"""
        report_path.write_text(html, encoding="utf-8")
        return report_path

    report_path = _write_embedding_report()
    return (report_path,)


@app.cell
def _(coordinates_path, feature_path, mo, report_path, summary_path):
    mo.md(f"""
    ## Outputs

    - Feature table: `{feature_path}`
    - Coordinate table: `{coordinates_path}`
    - Summary JSON: `{summary_path}`
    - HTML report: `{report_path}`
    - Figures:
      - `figures/rose_mask_embeddings_pca.png`
      - `figures/rose_mask_embeddings_umap.png`
      - `figures/rose_mask_embeddings_tsne.png`
      - `figures/rose_mask_embedding_feature_loadings.png`
    """)
    return


if __name__ == "__main__":
    app.run()
