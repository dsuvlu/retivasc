"""Reusable report language for scientific caveats."""

ROSE_MANUAL_MASK_CAVEAT = (
    "For ROSE feature extraction, I use the manual vessel masks rather than predicted "
    "masks so that feature sanity checks are not confounded by segmentation error. "
    "The classical segmentation baseline is shown separately as a computer-vision component."
)

ROSE_NO_PREDICTION_CAVEAT = (
    "ROSE-1 is documented as an AD/control OCTA subset in the published dataset, "
    "but this demo deliberately does not use ROSE for predictive modeling, AUROC, or "
    "calibration. ROSE-1 is small and lacks the plasma, amyloid/tau, genomic, and "
    "longitudinal context needed for any ADRD biomarker claim, so all ROSE analyses here "
    "are exploratory computer-vision sanity checks. The official ROSE-1 layout is "
    "annotated with disease/control labels from the published AD/control cohort ordering, "
    "but those labels are used only for exploratory grouping."
)

FIVES_SPLIT_CAVEAT = (
    "FIVES is split here using its official image-level train/test partition. "
    "The leakage check confirms image-disjoint, not patient-disjoint, splits: "
    "the public FIVES release does not publish patient identifiers, so subject-level "
    "splitting is not possible and the official split may place two eyes of one person "
    "across train and test. With 800 images from 573 subjects this is a known limitation "
    "that can modestly inflate performance. The disease-vs-normal base rate in the FIVES "
    "test set is about 75 percent, so AUPRC is prevalence-driven and AUROC is the cleaner "
    "headline."
)

FEATURE_SCALING_CAVEAT = (
    "Cross-cohort, cross-device, and cross-species use will require pixel-size or "
    "field-of-view normalization before interpreting absolute feature magnitudes."
)

EARLY_VS_LATE_BIOMARKER_NOTE = (
    "Reagan et al. 2025 ranks the vascular findings by timing: tortuosity "
    "occurrences and arteriovenous crossing events appear by 6 months, and "
    "arteriole/venule caliber asymmetry appears by 6 months before density changes. "
    "Vessel density and major branching change later at 12 months and the authors "
    "describe density as a weak early biomarker. Fractal dimension is retained here "
    "as external retinal-imaging context, not as a measured phenotype from that paper."
)
