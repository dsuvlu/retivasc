# `retivasc` development handoff plan for Codex/Claude

## 0. Executive brief

Build a **demo-sized**, reproducible retinal vascular computer-vision prototype for an upcoming 30-minute PI interview. This is not a production library and not an ADRD diagnostic model.

The package should support an **eight-minute demo** showing:

1. **ROSE OCTA CV pipeline:** raw OCTA image → manual vessel mask → simple classical segmentation overlay → skeletonized vascular graph.
2. **ROSE exploratory vascular-feature analysis:** vessel density, branch density, and fractal dimension by AD/control group, clearly labeled as exploratory because ROSE-1 is small.
3. **FIVES modeling/calibration demo:** use a larger fundus dataset to demonstrate honest model evaluation, calibration, Brier score, and confidence intervals. FIVES is not ADRD; it is a modeling-discipline demo.
4. **Roadmap figure:** how the same feature vectors could later connect human retinal images, plasma biomarkers, and Howell-lab mouse retinal images.

Use **pixi** for environment management and tasks. Use **marimo** notebooks for the demo/report. Keep code small, tested, and transparent.

## 1. Scientific and interview framing

### One-sentence pitch

`retivasc` is a proof-of-capability retinal vascular analysis prototype that demonstrates OCTA/fundus image ingestion, vessel-mask processing, skeletonization, vascular feature extraction, leakage-aware validation, and report generation for early ADRD biomarker research.

### What this package is

- A reproducible scaffold for retinal vascular computer vision.
- A way to show that we can extract interpretable vascular phenotypes from retinal images.
- A framework that can later receive Roux/JAX data: retinal imaging, plasma biomarkers, genomic context, clinical covariates, and mouse-model metadata.

### What this package is not

- Not an Alzheimer’s diagnostic model.
- Not a claim that public ROSE data validate an ADRD risk model.
- Not a deep-learning competition entry.
- Not a production-grade clinical or regulatory tool.

### Non-negotiable scientific caveats

1. **No predictive AD modeling on ROSE-1.** ROSE-1 has too few subjects for meaningful AUROC/calibration claims.
2. **Use ROSE manual masks for biological feature extraction.** This isolates vascular-feature analysis from segmentation error.
3. **Benchmark any segmentation baseline separately.** The segmentation overlay proves CV capability; the manual-mask features support the biological teaser.
4. **Enforce subject-level splits for ROSE.** Multiple angiograms/layers from the same subject must never appear in different folds.
5. **No fake biomarker fusion numbers.** Do not create synthetic p-tau217/GFAP/NfL results. Use a diagram only, unless using real RASTA clinical covariates as a stretch goal.
6. **Do not commit raw medical images.** Data loaders should expect local data under `data/raw/...`, which is gitignored.

## 2. Source anchors and references

Use these references in the README/report. Do not over-explain them in code.

### Job/project anchor

- Northeastern/Roux/JAX postdoctoral research fellow ad: complementary human and mouse studies evaluating plasma biomarkers with retinal vascular health as ADRD biomarkers.
  - URL: https://northeastern.wd1.myworkdayjobs.com/en-US/careers/job/Portland-ME/Postdoctoral-Research-Fellow_R140142?q=postdoctoral+fellow

### Mouse-side anchor

- Reagan AM, MacLean M, Cossette TL, Howell GR. 2025. “Retinal vascular dysfunction in the Mthfr677C>T mouse model of cerebrovascular disease.” *Alzheimer’s & Dementia*.
  - PubMed: https://pubmed.ncbi.nlm.nih.gov/40741711/
  - Wiley: https://alz-journals.onlinelibrary.wiley.com/doi/10.1002/alz.70501
  - Key point for demo: mouse retinal vascular phenotypes align with cerebrovascular phenotypes; retina and brain share AD-relevant molecular signatures.

### Human-side anchor

- Bader et al. 2024. “Rationale and design of the BeyeOMARKER study: prospective evaluation of blood and eye-based biomarkers for early detection of Alzheimer’s disease.”
  - PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11340081/
  - Key point for demo: human eye-clinic study combining plasma p-tau217 with retinal imaging, including hyperspectral imaging, widefield imaging, OCT, OCTA, Aβ-PET, and tau-PET in an enriched subcohort.

### Dataset anchors

- ROSE: Retinal OCT-Angiography Vessel Segmentation Dataset.
  - Zenodo: https://zenodo.org/records/12775880
  - PubMed: https://pubmed.ncbi.nlm.nih.gov/33284751/
  - arXiv: https://arxiv.org/abs/2007.05201
  - Use: OCTA vessel segmentation and AD-relevant exploratory vascular features.

- FIVES: Fundus Image Vessel Segmentation dataset.
  - Nature Scientific Data: https://www.nature.com/articles/s41597-022-01564-3
  - PubMed: https://pubmed.ncbi.nlm.nih.gov/35927290/
  - Figshare: https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169
  - Use: larger fundus vessel-feature modeling/calibration demo.

- RASTA: Retinal OCT-Angiography and Cardiovascular Status dataset.
  - Dataset site: https://rasta.u-bourgogne.fr/
  - Paper: https://www.mdpi.com/2306-5729/8/10/147
  - Use: stretch goal for OCTA + real clinical covariate fusion.

### Tooling anchors

- pixi: reproducible package/environment manager with lock files and task runner.
  - Docs: https://pixi.prefix.dev/
  - GitHub: https://github.com/prefix-dev/pixi

- marimo: reactive, Git-friendly Python notebooks stored as Python files and runnable as scripts/apps.
  - Site: https://marimo.io/
  - GitHub: https://github.com/marimo-team/marimo

## 3. MVP deliverables

The MVP is complete when these artifacts exist:

```text
reports/retivasc_pi_demo.html
figures/rose_pipeline_panel.png
figures/rose_feature_distributions.png
figures/fives_calibration_demo.png
figures/cross_species_roadmap.png
```

And these commands work:

```bash
pixi run test
pixi run rose-demo
pixi run fives-demo
pixi run report
```

If local ROSE/FIVES data are unavailable, notebooks should fail with a clear message explaining where to place data, not with an opaque traceback. Unit tests must use synthetic toy masks and must not require real medical images.

## 4. Repository structure

Keep this compact. Do not add a docs site, CI matrix, model cards, or package-publishing machinery before the interview.

```text
retivasc/
  README.md
  pyproject.toml
  pixi.toml
  pixi.lock
  .gitignore

  src/
    retivasc/
      __init__.py
      io.py
      preprocess.py
      segment.py
      skeleton.py
      features.py
      metrics.py
      splits.py
      plotting.py
      report_text.py

  notebooks/
    01_rose_octa_feature_demo.py
    02_fives_modeling_calibration_demo.py
    03_pi_demo_report.py

  tests/
    test_features.py
    test_metrics.py
    test_splits.py

  data/
    README.md

  figures/
    .gitkeep

  reports/
    .gitkeep
```

## 5. Environment and task spec

Use pixi as the only environment manager.

### `pixi.toml` requirements

Dependencies:

```text
python >=3.11
numpy
scipy
pandas
pyarrow
scikit-image
opencv
matplotlib
scikit-learn
networkx
pydantic
marimo
pytest
ruff
```

Optional only if easy:

```text
tqdm
rich
```

### Required pixi tasks

```toml
[tasks]
test = "pytest -q"
lint = "ruff check src tests notebooks"
format = "ruff format src tests notebooks"
rose-demo = "marimo run notebooks/01_rose_octa_feature_demo.py"
fives-demo = "marimo run notebooks/02_fives_modeling_calibration_demo.py"
report = "marimo export html notebooks/03_pi_demo_report.py -o reports/retivasc_pi_demo.html"
check = { depends-on = ["lint", "test"] }
```

Do not add CI unless all MVP figures are finished.

## 6. Core implementation details

### 6.1 `io.py`

Implement lightweight local-data loaders. Do not download restricted datasets automatically.

Functions:

```python
def load_rose_manifest(root: str | Path) -> pd.DataFrame:
    """Return one row per ROSE image/layer with image_path, mask_path, subject_id, layer, label."""


def load_fives_manifest(root: str | Path) -> pd.DataFrame:
    """Return one row per FIVES image with image_path, mask_path, label fields if available."""


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear ValueError if required metadata columns are absent."""
```

Expected common manifest columns:

```text
dataset
subject_id
image_id
image_path
mask_path
modality
layer
label
split_group
```

For ROSE, set:

```text
split_group = subject_id
modality = OCTA
layer ∈ {SVC, DVC, SVC+DVC or equivalent labels inferred from filenames}
```

If filename parsing is uncertain, fail loudly and ask the user to supply a small metadata CSV. Do not silently invent labels.

### 6.2 `preprocess.py`

Functions:

```python
def normalize_image(image: np.ndarray) -> np.ndarray:
    """Return float image scaled to [0, 1]."""


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA to grayscale; leave 2D arrays unchanged."""
```

Keep preprocessing minimal. Avoid heavy enhancement unless necessary for the classical overlay.

### 6.3 `segment.py`

Use a classical, GPU-free baseline only.

Functions:

```python
def classical_vesselness_mask(image: np.ndarray, *, threshold: str = "otsu") -> np.ndarray:
    """Return a binary vessel mask using Frangi/Sato-style vesselness + thresholding."""


def cleanup_mask(mask: np.ndarray, *, min_size: int = 16) -> np.ndarray:
    """Remove small components and fill obvious holes."""
```

Implementation suggestion:

```text
1. normalize image
2. invert if needed after visual inspection/config flag
3. apply skimage.filters.frangi or sato
4. threshold with Otsu or percentile
5. morphological cleanup
```

Do not train a U-Net in the MVP.

### 6.4 `skeleton.py`

Functions:

```python
def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Return 1-pixel-wide skeleton."""


def branchpoint_mask(skel: np.ndarray) -> np.ndarray:
    """Return boolean mask of skeleton pixels with >=3 skeleton neighbors."""


def endpoint_mask(skel: np.ndarray) -> np.ndarray:
    """Return boolean mask of skeleton pixels with exactly 1 skeleton neighbor."""
```

Use simple 8-neighborhood convolution for branchpoints/endpoints. NetworkX graph conversion is optional; do not overengineer.

### 6.5 `features.py`

Required feature functions:

```python
def vessel_density(mask: np.ndarray, fov_mask: np.ndarray | None = None) -> float:
    """Fraction of field of view occupied by vessel pixels."""


def skeleton_length_density(mask: np.ndarray, fov_mask: np.ndarray | None = None) -> float:
    """Skeleton pixels per field-of-view pixel."""


def branchpoint_density(mask: np.ndarray, fov_mask: np.ndarray | None = None) -> float:
    """Branchpoint count normalized by field-of-view area."""


def fractal_dimension_boxcount(mask: np.ndarray) -> float:
    """Estimate box-counting fractal dimension of a binary vessel mask."""


def connected_component_count(mask: np.ndarray) -> int:
    """Number of connected vascular components."""


def extract_vascular_features(mask: np.ndarray, *, fov_mask: np.ndarray | None = None) -> dict[str, float]:
    """Return all MVP features in a flat dictionary."""
```

Optional if time:

```python
def tortuosity_proxy(mask: np.ndarray) -> float:
    """Approximate tortuosity using skeleton segment arc/chord ratios."""
```

If tortuosity is fragile, leave it as optional and mark it as experimental in the report.

### 6.6 `metrics.py`

Functions:

```python
def dice_score(y_true: np.ndarray, y_pred: np.ndarray) -> float: ...
def iou_score(y_true: np.ndarray, y_pred: np.ndarray) -> float: ...
def sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float: ...
def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float: ...
```

Handle empty masks explicitly and document behavior.

### 6.7 `splits.py`

Functions:

```python
def assert_group_split_safe(train: pd.DataFrame, test: pd.DataFrame, group_col: str) -> None:
    """Raise if any group appears in both train and test."""


def grouped_train_test_split(df: pd.DataFrame, group_col: str, label_col: str | None = None, test_size: float = 0.25, random_state: int = 0):
    """Return leakage-safe train/test dataframes."""
```

ROSE notebook must visibly call `assert_group_split_safe`.

### 6.8 `plotting.py`

Functions should save figures to disk and return matplotlib figures.

```python
def plot_rose_pipeline_panel(image, manual_mask, predicted_mask, skeleton, out_path): ...
def plot_feature_distributions(features_df, feature_names, label_col, out_path): ...
def plot_calibration(y_true, y_prob, out_path): ...
def plot_cross_species_roadmap(out_path): ...
```

Use matplotlib only. No seaborn required.

## 7. Notebook specs

### Notebook 1: `01_rose_octa_feature_demo.py`

Purpose: demonstrate OCTA computer vision and AD-relevant vascular-feature extraction.

Required sections:

1. **Motivation:** ROSE is an OCTA vessel-segmentation dataset with an AD/control subset; this is a small exploratory CV scaffold, not an ADRD model.
2. **Local data check:** explain expected `data/raw/rose/...` layout and fail clearly if absent.
3. **Manifest audit:** show number of images, subjects, layers, labels.
4. **Leakage note:** show why splitting must be by `subject_id`.
5. **Pipeline figure:** raw OCTA → manual mask → classical segmentation → skeleton.
6. **Feature table:** extract density, skeleton-length density, branchpoint density, fractal dimension from manual masks.
7. **Exploratory AD/control distributions:** plot 2–3 features by group.
8. **Limitations box:** n too small, diagnosis labels not enough for predictive claims, no plasma biomarkers, no amyloid/tau confirmation unless dataset metadata proves otherwise.

Required saved outputs:

```text
figures/rose_pipeline_panel.png
figures/rose_feature_distributions.png
```

Language to include verbatim somewhere in the notebook/report:

> For the AD/control feature teaser, I use the manual masks rather than predicted masks so that the biological comparison is not confounded by segmentation error. The segmentation baseline is benchmarked separately as a computer-vision component.

### Notebook 2: `02_fives_modeling_calibration_demo.py`

Purpose: demonstrate modeling/calibration discipline on a dataset large enough for it to be less silly.

Required sections:

1. **Motivation:** FIVES is not ADRD, but it has 800 annotated fundus images and supports an honest feature/modeling demo.
2. **Local data check:** explain expected `data/raw/fives/...` layout and fail clearly if absent.
3. **Feature extraction:** extract vascular features from provided masks.
4. **Simple classifier:** classify available FIVES disease labels or normal-vs-disease if labels are cleanly available.
5. **Evaluation:** AUROC/AUPRC if appropriate, Brier score, calibration curve, bootstrap confidence intervals.
6. **Limitations:** disease labels are not ADRD, fundus is not OCTA, patient-level splitting only if patient IDs are available.

Models:

```text
Primary: regularized logistic regression
Optional: random forest or gradient boosting
```

Required saved output:

```text
figures/fives_calibration_demo.png
```

If FIVES labels are unavailable or messy, do not invent them. Instead, run segmentation/feature benchmarking and produce calibration only if a valid label column exists.

### Notebook 3: `03_pi_demo_report.py`

Purpose: create the PI-facing HTML artifact.

Required sections:

1. **Why this exists:** Christine flagged computer vision as a need; this is a focused prototype.
2. **Project fit:** retinal vascular imaging + plasma biomarkers + human/mouse translation.
3. **ROSE CV panel:** include `figures/rose_pipeline_panel.png`.
4. **ROSE exploratory features:** include `figures/rose_feature_distributions.png`.
5. **FIVES modeling discipline:** include `figures/fives_calibration_demo.png` if available.
6. **Cross-species roadmap:** include figure and text connecting human vascular features to Howell/Reagan mouse retinal phenotypes.
7. **Multimodal roadmap:** retinal features + plasma p-tau217/Aβ42 ratio + GFAP/NfL + genomics + clinical covariates, but no fake numbers.
8. **What I would do with Roux/JAX data:** replace public loaders, add plasma/mouse data, validate with leakage-safe splits, missingness checks, site/batch analysis, and biologically informed ablations.

Required saved output:

```text
reports/retivasc_pi_demo.html
```

## 8. Cross-species roadmap figure

Create a simple schematic. No data required.

Text/structure:

```text
Human retinal images
    → vessel mask / skeleton
    → density, branching, fractal dimension, tortuosity

Mouse retinal images from Mthfr677C>T studies
    → same vessel-mask / skeleton pipeline
    → same feature vector definitions

Shared phenotype table
    → compare human/mouse retinal vascular signatures
    → align with plasma/proteomic/genomic pathways
    → generate testable vascular, inflammatory, and metabolic hypotheses
```

Caption:

> The extraction functions are species-agnostic: the same definitions of density, branching, fractal dimension, and tortuosity can be applied to human OCTA/fundus images and mouse retinal images, enabling cross-species comparison once Roux/JAX data are available.

## 9. Tests

Required tests should be quick and data-free.

### `test_metrics.py`

```text
dice(mask, mask) == 1
dice(mask, empty) == 0 for non-empty mask
IoU identical masks == 1
sensitivity/specificity handle simple known arrays
```

### `test_features.py`

```text
empty mask vessel_density == 0
full mask vessel_density == 1
single-line skeleton_length_density > 0
synthetic Y-shape branchpoint count >= 1
fractal_dimension_boxcount returns finite value for grid-like mask
extract_vascular_features returns required keys
```

### `test_splits.py`

```text
assert_group_split_safe passes with disjoint groups
assert_group_split_safe raises on group overlap
grouped_train_test_split creates no overlap
```

## 10. README requirements

Keep README short but useful.

Required sections:

```text
# retivasc
What it is
What it is not
Scientific motivation
Datasets and access
Install with pixi
Run demo
Expected outputs
Limitations
References
```

README paragraph:

> `retivasc` is a proof-of-capability retinal vascular analysis prototype designed for early ADRD biomarker research. It demonstrates OCTA/fundus image ingestion, vessel-mask processing, skeletonization, vascular feature extraction, leakage-aware validation, and report generation. The initial demo uses ROSE for AD-relevant OCTA vascular features and FIVES for a larger-scale modeling/calibration example. It does not claim clinical ADRD prediction from public data; instead, it demonstrates the reproducible computer-vision scaffold needed to connect retinal vascular phenotypes with plasma biomarkers, genomic context, and mouse-model biology.

## 11. Suggested three-day implementation plan

### Day 1: ROSE pipeline and core functions

Codex tasks:

```text
1. Create repo structure.
2. Add pixi.toml, pyproject.toml, .gitignore.
3. Implement metrics.py, skeleton.py, features.py, splits.py.
4. Add data-free tests.
5. Implement basic ROSE manifest loader with clear local-data assumptions.
6. Implement Notebook 1 up to pipeline figure.
```

Acceptance criteria:

```bash
pixi run test
pixi run rose-demo
```

Required outputs:

```text
figures/rose_pipeline_panel.png
```

### Day 2: feature distributions and FIVES modeling demo

Codex tasks:

```text
1. Finish ROSE feature extraction from manual masks.
2. Plot exploratory ROSE AD/control feature distributions.
3. Implement FIVES manifest loader.
4. Extract FIVES features from masks.
5. Implement regularized logistic regression demo if labels are available.
6. Add calibration curve, Brier score, bootstrap confidence intervals.
```

Acceptance criteria:

```bash
pixi run rose-demo
pixi run fives-demo
```

Required outputs:

```text
figures/rose_feature_distributions.png
figures/fives_calibration_demo.png
```

### Day 3: report and polish

Claude/Codex tasks:

```text
1. Build PI report notebook.
2. Add cross-species roadmap figure.
3. Add multimodal fusion roadmap diagram without fake numbers.
4. Tighten README.
5. Run tests and notebooks.
6. Export final HTML report.
```

Acceptance criteria:

```bash
pixi run test
pixi run report
```

Required output:

```text
reports/retivasc_pi_demo.html
```

## 12. Division of labor for Codex and Claude

### Codex: implementation owner

Primary responsibilities:

```text
repo skeleton
pixi setup
feature/metric/split functions
data loaders
notebook execution
figure generation
tests
```

Codex should avoid:

```text
training deep networks
adding CI before figures exist
building a large CLI
inventing labels or metadata
committing raw data
```

### Claude: scientific reviewer and report owner

Primary responsibilities:

```text
review methodology for overclaiming
ensure ROSE language is exploratory only
polish PI report narrative
check dataset/source citations
check leakage and split language
make roadmap text crisp
```

Claude should flag:

```text
any AUROC/calibration claim on ROSE
any subject-level leakage
any synthetic biomarker “results”
any unsupported AD diagnostic language
any unnecessarily broad package scope
```

## 13. Copy-paste prompt for Codex

You are implementing a compact interview-demo package called `retivasc`. Do not build a production library. The goal is an eight-minute PI demo for a Roux/JAX postdoc interview about retinal vascular imaging, plasma biomarkers, and early ADRD risk. Use pixi for the environment and marimo notebooks for the report.

Implement the repository exactly as specified in `retivasc_development_handoff.md`. The MVP deliverables are:

```text
figures/rose_pipeline_panel.png
figures/rose_feature_distributions.png
figures/fives_calibration_demo.png
figures/cross_species_roadmap.png
reports/retivasc_pi_demo.html
```

Non-negotiables:

```text
No U-Net training in MVP.
No AD predictive model on ROSE.
Use ROSE manual masks for AD/control exploratory feature distributions.
Benchmark classical segmentation separately.
Enforce subject-level splits for ROSE.
No synthetic biomarker results.
Do not commit raw data.
All tests must run without real medical images.
```

Start by creating the repo skeleton, pixi environment, feature/metric/split utilities, and tests. Then implement the ROSE notebook, then the FIVES notebook, then the PI report.

## 14. Copy-paste prompt for Claude

You are reviewing and polishing a compact interview-demo package called `retivasc` for a Roux/JAX postdoc interview. The candidate wants to show retinal vascular computer vision skill without overclaiming clinical ADRD prediction.

Your job is to review the code, notebooks, README, and report for scientific rigor and interview impact. Focus especially on:

```text
ROSE is exploratory only; no predictive AD claims.
ROSE splits must be subject-level because multiple angiograms/layers come from each subject.
Manual masks should be used for biological feature extraction; predicted masks are a separate segmentation component.
FIVES or RASTA should carry the modeling/calibration demonstration, not ROSE.
No fake p-tau217/GFAP/NfL data should be shown as results.
The cross-species hook to Howell/Reagan mouse retinal data must be explicit.
The final report should be suitable for an eight-minute PI demo.
```

Produce concise edits, not scope expansion. Preserve the demo-sized plan.

## 15. Eight-minute demo talk track

### Minute 0–1: why this exists

> Christine flagged computer vision as a need, so I built a small prototype around open retinal datasets. I am not claiming this predicts ADRD. The point is to demonstrate the reproducible scaffold: retinal image ingestion, vessel segmentation, vascular feature extraction, leakage-aware validation, and a roadmap for connecting this to plasma biomarkers and mouse data.

### Minute 1–3: ROSE OCTA computer vision

Show raw OCTA → manual mask → classical segmentation overlay → skeleton.

> ROSE is useful because it is OCTA and includes an AD/control subset. For the biological feature analysis, I use manual masks so segmentation error does not contaminate the comparison. The segmentation baseline is benchmarked separately.

### Minute 3–4: ROSE exploratory features

Show feature distributions.

> This is exploratory only. ROSE-1 is small, so I would not present a predictive model or calibration curve from this dataset.

### Minute 4–6: FIVES modeling discipline

Show feature table and calibration/model diagnostics.

> This is where I demonstrate modeling discipline on a dataset with a more appropriate sample size. FIVES is not ADRD, but it supports an honest vessel-feature modeling and calibration example.

### Minute 6–7: cross-species extension

Show human ↔ mouse shared feature-space diagram.

> The same vascular features are species-agnostic. That means the Howell lab’s Mthfr mouse retinal images could be mapped into the same feature space and compared with human retinal phenotypes.

### Minute 7–8: what happens with real Roux/JAX data

> With the real Roux/JAX data, I would replace the public dataset loaders with project-specific loaders, add plasma biomarkers and genomic context, and then evaluate calibrated multimodal models with leakage-safe splits, missingness analysis, site/batch checks, and biologically informed ablations.

## 16. Stop conditions

Stop building once the four figures and report exist. Do not add new tools or datasets unless one of the core outputs is missing.

Priority order if time is short:

```text
1. ROSE pipeline panel
2. ROSE feature distributions with caveats
3. Cross-species roadmap
4. FIVES calibration demo
5. README polish
6. Extra tests
```

If only one dataset is ready, make ROSE work and use the modeling/calibration section as a roadmap rather than forcing weak modeling.
