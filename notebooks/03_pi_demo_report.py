import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import json
    from pathlib import Path

    from retivasc.plotting import plot_cross_species_roadmap, plot_data_audit_flow
    from retivasc.report_text import ROSE_MANUAL_MASK_CAVEAT, ROSE_NO_PREDICTION_CAVEAT

    return (
        Path,
        ROSE_MANUAL_MASK_CAVEAT,
        ROSE_NO_PREDICTION_CAVEAT,
        json,
        plot_cross_species_roadmap,
        plot_data_audit_flow,
    )


@app.cell
def _(mo):
    mo.md("""
    # retivasc PI Demo Report

    Christine flagged computer vision as a need; this compact prototype demonstrates
    retinal image ingestion, vessel-mask processing, skeletonization, feature extraction,
    leakage-aware validation, and report generation for early ADRD biomarker research.

    This is not an Alzheimer's diagnostic model.
    """)
    return


@app.cell
def _(ROSE_MANUAL_MASK_CAVEAT, ROSE_NO_PREDICTION_CAVEAT, mo):
    mo.md(
        f"""
        ## Project Fit

        The package connects retinal vascular image analysis to a future Roux/JAX workflow
        involving human retinal imaging, plasma biomarkers, genomic context, clinical
        covariates, and Howell-lab mouse retinal images.

        {ROSE_MANUAL_MASK_CAVEAT}

        {ROSE_NO_PREDICTION_CAVEAT}
        """
    )
    return


@app.cell
def _(Path, json, mo):
    def image_or_missing(path: Path, label: str):
        if path.exists():
            return mo.image(path, alt=label)
        return mo.md(
            f"_Missing `{path}`. Run the corresponding dataset notebook after "
            "local data are available._"
        )

    def format_ci(values: list[float]) -> str:
        if len(values) != 2:
            return "not available"
        return f"{values[0]:.3f}-{values[1]:.3f}"

    rose_pipeline = image_or_missing(Path("figures/rose_pipeline_panel.png"), "ROSE pipeline")
    rose_features = image_or_missing(
        Path("figures/rose_feature_distributions.png"), "ROSE feature distributions"
    )
    fives_calibration = image_or_missing(
        Path("figures/fives_calibration_demo.png"), "FIVES calibration"
    )
    fives_metrics_path = Path("reports/fives_metrics.json")
    if fives_metrics_path.exists():
        fives_metrics = json.loads(fives_metrics_path.read_text(encoding="utf-8"))
        fives_metrics_text = f"""
        FIVES calibration summary:

        - Target: `{fives_metrics["target"]}`; positive class is
          `{fives_metrics["positive class"]}`, negative class is
          `{fives_metrics["negative class"]}`.
        - Official split: `{fives_metrics["train rows"]}` train rows and
          `{fives_metrics["test rows"]}` test rows.
        - AUROC: `{fives_metrics["AUROC"]:.3f}`
          (95% bootstrap CI `{format_ci(fives_metrics["AUROC 95% CI"])}`).
        - AUPRC: `{fives_metrics["AUPRC"]:.3f}`
          (95% bootstrap CI `{format_ci(fives_metrics["AUPRC 95% CI"])}`).
        - Brier score: `{fives_metrics["Brier score"]:.3f}`.

        This is a modeling-discipline demo on fundus disease labels, not ADRD prediction.
        """
    else:
        fives_metrics_text = (
            "_Missing `reports/fives_metrics.json`. Run the FIVES notebook after local "
            "data are available._"
        )

    mo.vstack(
        [
            mo.md("## ROSE CV Panel"),
            rose_pipeline,
            mo.md("## ROSE Exploratory Features"),
            rose_features,
            mo.md("## FIVES Modeling Discipline"),
            fives_calibration,
            mo.md(fives_metrics_text),
        ],
        gap=1.0,
    )
    return


@app.cell
def _(Path, plot_data_audit_flow):
    data_audit_path = Path("figures/data_audit_flow.png")
    _ = plot_data_audit_flow(data_audit_path)
    return (data_audit_path,)


@app.cell
def _(data_audit_path, mo):
    audit_text = """
        The audit trail is deliberately code-aware: each transformation names the module
        or notebook cell responsible for it. Raw medical images stay under `data/raw/`,
        derived features are cached under `data/interim/`, and the report consumes only
        derived figures and metrics.
        """
    (
        mo.vstack(
            [
                mo.md("## End-to-End Data Audit"),
                mo.image(data_audit_path, alt="End-to-end data audit"),
                mo.md(audit_text),
            ],
            gap=1.0,
        )
        if data_audit_path.exists()
        else mo.md(f"## End-to-End Data Audit\n\nMissing `{data_audit_path}`.")
    )
    return


@app.cell
def _(Path, plot_cross_species_roadmap):
    roadmap_path = Path("figures/cross_species_roadmap.png")
    _ = plot_cross_species_roadmap(roadmap_path)
    return (roadmap_path,)


@app.cell
def _(mo, roadmap_path):
    roadmap_text = """
        The same feature definitions can be applied to human OCTA/fundus images and
        Howell/Reagan mouse retinal images. That creates a shared phenotype table for
        comparing retinal vascular signatures across species.
        """
    (
        mo.vstack(
            [
                mo.md("## Cross-Species Roadmap"),
                mo.image(roadmap_path, alt="Cross-species roadmap"),
                mo.md(roadmap_text),
            ],
            gap=1.0,
        )
        if roadmap_path.exists()
        else mo.md(f"## Cross-Species Roadmap\n\nMissing `{roadmap_path}`.")
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Multimodal Roadmap

    Once real Roux/JAX data are available, the public dataset loaders should be
    replaced with project-specific loaders that join retinal vascular features with
    plasma p-tau217, amyloid ratios, GFAP, NfL, genomic context, clinical covariates,
    and mouse-model metadata.

    The analysis should use leakage-safe splits, missingness checks, site/batch
    analysis, calibration diagnostics, and biologically informed ablations. No
    synthetic biomarker results are shown here.
    """)
    return


if __name__ == "__main__":
    app.run()
