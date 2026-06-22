# retivasc

`retivasc` is a proof-of-capability retinal vascular analysis prototype designed for
early ADRD biomarker research. It demonstrates OCTA/fundus image ingestion,
vessel-mask processing, skeletonization, vascular feature extraction, leakage-aware
validation, and report generation. The initial demo uses ROSE for OCTA vessel-mask
computer vision and FIVES for a larger-scale modeling/calibration example. It does
not claim clinical ADRD prediction from public data; instead, it demonstrates the
reproducible computer-vision scaffold needed to connect retinal vascular phenotypes
with plasma biomarkers, genomic context, and mouse-model biology.

## What It Is

A compact interview-demo package for retinal vascular computer vision:

- local dataset manifest loading
- classical vessel segmentation overlays
- manual-mask vascular feature extraction
- skeleton and branchpoint analysis
- leakage-aware train/test splitting
- calibration/report figures

## What It Is Not

This is not an Alzheimer's diagnostic model, a production clinical tool, a deep
learning competition entry, or a claim that public ROSE data validate ADRD risk
prediction.

## Scientific Motivation

The goal is to show a reproducible scaffold for extracting interpretable retinal
vascular phenotypes that could later be connected to plasma biomarkers, genomic
context, clinical covariates, and mouse-model retinal vascular data.

## Datasets And Access

Raw medical images are not committed. Place local datasets under:

```text
data/raw/rose/
data/raw/fives/
```

If automatic filename parsing is uncertain, provide a `manifest.csv` with at least
`image_path`, `mask_path`, and dataset-specific metadata columns.

The official FIVES layout is detected automatically after unzipping under
`data/raw/fives`; its `train/test` folders and filename suffixes are used for the
modeling demo.

## Install With Pixi

```bash
pixi install
```

## Run Demo

```bash
pixi run test
pixi run tutorial
pixi run rose-demo
pixi run fives-demo
pixi run rose-embeddings
pixi run report
```

`pixi run report` writes both local review files under `reports/` and a GitHub
Pages-ready static site under `docs/`.

ROSE-derived visual panels are generated locally and are not committed or published
by default. Public reports use a schematic or placeholder unless redistribution of
image examples is permitted by the dataset terms.

## Optional Deep Segmentation Comparisons

The ROSE notebook can compare the classical methods against prediction masks from
official external tools or against small in-repo U-Net-family demo comparators.

Generate local demo predictions with the Pixi environment:

```bash
pixi run python -m retivasc.deep_models --rose-root data/raw/rose
```

This writes cached masks and manifests under `data/interim/`, including
`data/interim/rose_inrepo_deep_predictions.csv`. The notebook picks those up and
adds U-Net Lite, OCTA-Net Lite, and nnU-Net Lite rows to the same metric table and
comparison figure. The command also tunes threshold and light morphology on the ROSE
tuning rows before writing held-out prediction masks. These local models are demo
comparators; they are not the official OCTA-Net or nnU-Net implementations.

For official OCTA-Net or nnU-Net comparisons, run those frameworks in their own
environment and provide prediction CSVs under `data/interim/` with `image_id` plus
`octa_net_prediction_path` or `nnunet_prediction_path`.

## ROSE Mask Embeddings

`pixi run rose-embeddings` opens a Marimo notebook that computes mask-derived
vascular feature embeddings for ROSE-1. The workflow uses vessel masks, not raw OCTA
intensity images, and writes local PCA, UMAP, and t-SNE figures plus a summary report
under `figures/` and `reports/`. These ROSE-derived artifacts are ignored by Git.

## ROSE Layer-Aware Statistics

`pixi run rose-layer-stats` runs a follow-up Marimo workflow that summarizes ROSE-1
mask-derived vascular features at the subject and OCTA-layer levels. It writes local
CSV tables under `outputs/rose_layer_stats/`, figures under `figures/`, and an HTML
report at `reports/rose_layer_aware_statistics.html`. This analysis reports
exploratory effect sizes, bootstrap confidence intervals, paired layer contrasts,
outlier sensitivity, and mask-artifact checks; it does not train a ROSE classifier or
make an AD diagnostic claim.

## Expected Outputs

```text
figures/rose_pipeline_panel.png
figures/rose_feature_distributions.png
figures/rose_model_comparison_grid.png
figures/rose_mask_embeddings_pca.png
figures/rose_mask_embeddings_umap.png
figures/rose_mask_embeddings_tsne.png
figures/rose_mask_embedding_feature_loadings.png
figures/rose_subject_level_pca.png
figures/rose_layer_effect_sizes.png
figures/rose_layer_contrast_effect_sizes.png
figures/rose_outlier_audit_panel.png
figures/rose_feature_qc_heatmap.png
figures/rose_mask_artifact_audit.png
figures/processing_example_panel.png
figures/fives_calibration_demo.png
reports/rose_mask_embedding_report.html
reports/rose_layer_aware_statistics.html
reports/retivasc_pi_demo.html
reports/data_audit_flow.html
docs/index.html
docs/data_audit_flow.html
docs/assets/processing_example_panel.png
docs/assets/fives_calibration_demo.png
```

## Publish With GitHub Pages

The generated `docs/` directory is a self-contained static website. To publish it
from GitHub:

1. Run `pixi run report` and commit the refreshed `docs/` files.
2. In the GitHub repository, open Settings -> Pages.
3. Set the source to deploy from the main branch's `/docs` folder.
4. Save the setting; GitHub Pages will serve `docs/index.html` as the project site.

## Limitations

ROSE-1 is documented as an AD/control OCTA subset in the published dataset, but
this demo deliberately does not use ROSE for predictive modeling, AUROC, or
calibration. ROSE-1 is small and lacks the plasma, amyloid/tau, genomic, and
longitudinal context needed for any ADRD biomarker claim, so all ROSE analyses
here are exploratory computer-vision sanity checks. The official ROSE-1 layout is
annotated with disease/control labels from the published AD/control cohort
ordering, but those labels are used only for exploratory grouping. FIVES is not
ADRD; it is used only to demonstrate modeling and calibration discipline on a
larger fundus dataset when valid labels are available. Cross-cohort,
cross-device, and cross-species use will require pixel-size or field-of-view
normalization before interpreting absolute feature magnitudes. No synthetic plasma
biomarker results are generated.

## References

- Reagan AM, MacLean M, Cossette TL, Howell GR. 2025. Retinal vascular dysfunction
  in the Mthfr677C>T mouse model of cerebrovascular disease.
  https://pubmed.ncbi.nlm.nih.gov/40741711/
- Bader et al. 2024. Rationale and design of the BeyeOMARKER study.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11340081/
- ROSE retinal OCTA vessel segmentation dataset:
  https://zenodo.org/records/12775880
- FIVES fundus image vessel segmentation dataset:
  https://www.nature.com/articles/s41597-022-01564-3
- RASTA retinal OCTA and cardiovascular status dataset:
  https://rasta.u-bourgogne.fr/
