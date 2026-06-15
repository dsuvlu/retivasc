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
pixi run report
```

`pixi run report` writes both local review files under `reports/` and a GitHub
Pages-ready static site under `docs/`.

ROSE-derived visual panels are generated locally and are not committed or published
by default. Public reports use a schematic or placeholder unless redistribution of
image examples is permitted by the dataset terms.

## Expected Outputs

```text
figures/rose_pipeline_panel.png
figures/rose_feature_distributions.png
figures/processing_example_panel.png
figures/fives_calibration_demo.png
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
here are exploratory computer-vision sanity checks. Labels are used only if
explicitly supplied in a local manifest; none are inferred from filenames. FIVES
is not ADRD; it is used only to demonstrate modeling and calibration discipline on
a larger fundus dataset when valid labels are available. Cross-cohort,
cross-device, and cross-species use will require pixel-size or field-of-view
normalization before interpreting absolute feature magnitudes. No synthetic plasma
biomarker results are generated.

## References

- Northeastern/Roux/JAX postdoctoral research fellow ad:
  https://northeastern.wd1.myworkdayjobs.com/en-US/careers/job/Portland-ME/Postdoctoral-Research-Fellow_R140142?q=postdoctoral+fellow
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
