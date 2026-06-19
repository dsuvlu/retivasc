# ROSE Mask Embedding Interpretation

## Current Results

The ROSE-1 mask-derived embedding workflow ran on:

- 39 subjects
- 117 subject-layer rows
- 3 OCTA layers: DVC, SVC, and SVC+DVC
- 26 AD and 13 control subjects per layer

The PCA embedding used mask-derived vascular features, not raw OCTA image intensity.
PC1 explained 45.6% of scaled feature variance and PC2 explained 26.6%, for about
72.1% across the first two principal components.

The current embeddings do not show clear AD/control separation. PCA and t-SNE both
show substantial overlap between groups:

- PCA centroid permutation p-value: ~0.52
- PCA silhouette score: ~0.05
- t-SNE centroid permutation p-value: ~0.53
- t-SNE silhouette score: ~0.005

UMAP was not run because `umap-learn` is not installed; the notebook generated an
informative placeholder figure.

## Feature Drivers

The strongest PCA loading directions were:

- PC1: mostly mean tortuosity, small-component fraction, and segment length features.
- PC2: mostly small-component fraction and tortuosity in the opposite direction.

The DVC layer shows more spread than SVC and SVC+DVC, including at least one clear
outlier. This suggests that layer-specific mask morphology, annotation/acquisition
effects, or outlier masks may dominate the current embedding more than diagnosis.

Group-average feature values were directionally similar. Controls were slightly higher
on vessel area fraction, skeleton length density, branchpoint density, fractal
dimension, and orientation entropy, but the differences are small and should not be
interpreted as biological evidence from this analysis alone.

## Current Interpretation

The honest conclusion is:

> The embedding workflow works and produces interpretable vascular feature spaces, but
> the current ROSE-1 mask features do not provide convincing exploratory evidence of
> AD/control separation. The visible variation is more consistent with layer-specific
> mask morphology, annotation/acquisition differences, and outliers than a robust
> disease-group pattern.

The effective sample size is 39 subjects, not 117 independent observations, because
each subject contributes multiple layers.

## Suggested Next Analyses

1. Layer-specific feature tests

   Analyze SVC, DVC, and SVC+DVC separately. For each feature, compare AD vs control
   using effect sizes, bootstrap confidence intervals, and permutation p-values.

2. Paired cross-layer contrasts

   Compute within-subject differences and ratios such as:

   - DVC - SVC
   - SVC+DVC - SVC
   - DVC / SVC

   Disease effects may appear in how layers differ within a subject rather than in
   absolute feature values.

3. Robust outlier audit

   Inspect high-leverage DVC outliers and rerun PCA/statistics with and without them.
   Determine whether they reflect acquisition, annotation, or true vascular structure.

4. Feature-level effect plots

   Create a volcano/effect-size plot showing AD/control median difference, bootstrap
   confidence interval, permutation p-value, and false-discovery-adjusted p-value for
   every vascular feature.

5. Subject-level aggregation

   Collapse layers into one row per subject using means, maxima, minima, and layer
   contrasts. Re-run PCA and feature tests at the true subject level.

6. Mixed-effects modeling

   For each vascular feature, fit an exploratory repeated-measures model:

   ```text
   feature ~ diagnosis + layer + diagnosis:layer + (1 | subject)
   ```

   This respects repeated layers per subject and asks whether diagnosis effects vary
   by layer.

7. Mask artifact audit

   Test whether AD/control groups differ in mask dimensions, foreground fraction,
   connected components, official split, or other acquisition/annotation proxies.

8. Distributional tests

   Compare full feature distributions with KS tests, energy distance, or permutation
   tests. Some differences may appear in variance or tails rather than group means.

9. Skeleton graph features

   Add graph-derived summaries from skeletons: branch length distribution, node degree
   distribution, terminal branch count, connected component size distribution, and
   dropout-like proxies.

10. Spatially localized features

    Divide masks into image regions or fovea-centered zones and compute regional
    density/branching. Global averages may hide local capillary dropout patterns.

## Recommended Next Step

The most defensible next implementation is a subject-level, layer-aware statistical
summary:

- effect sizes
- bootstrap confidence intervals
- permutation p-values
- paired cross-layer contrasts
- outlier sensitivity checks

This is more interpretable and scientifically cautious than adding another embedding
method or attempting disease classification on ROSE.
