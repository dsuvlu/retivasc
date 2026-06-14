"""Reusable report language for scientific caveats."""

ROSE_MANUAL_MASK_CAVEAT = (
    "For the AD/control feature teaser, I use the manual masks rather than predicted "
    "masks so that the biological comparison is not confounded by segmentation error. "
    "The segmentation baseline is benchmarked separately as a computer-vision component."
)

ROSE_NO_PREDICTION_CAVEAT = (
    "ROSE is used here as a small exploratory OCTA vascular-feature dataset, not as a "
    "basis for AD prediction, AUROC claims, or calibration claims."
)
