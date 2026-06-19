# Local Data

Raw retinal datasets are intentionally excluded from version control.

Expected local roots:

```text
data/raw/rose/
data/raw/fives/
```

For the most reliable loading, place a `manifest.csv` in the dataset root.

ROSE manifest columns:

```text
dataset,subject_id,image_id,image_path,mask_path,modality,layer,split_group
```

`label` may be supplied when clinical labels are explicitly available in a local
manifest. For the official ROSE-1 layout, the loader records the published
Alzheimer's disease/control cohort labels from the official subject ordering:
train 1-20 disease, train 21-30 control, test 1-6 disease, and test 7-9 control.
It does not infer ROSE labels from arbitrary filenames.

FIVES manifest columns:

```text
dataset,subject_id,image_id,image_path,mask_path,modality,label,split_group
```

Relative `image_path` and `mask_path` values are resolved against the dataset root.

The official ROSE archives can also be used without a manifest if unzipped under:

```text
data/raw/rose/
  ROSE/
    ROSE-1/
    ROSE-2/
  ROSE-O/
```

The loader uses ROSE-1/ROSE-2 image-mask pairs and records layer/split metadata.
ROSE-O is kept available locally but is not needed for the default notebook.
If official filenames collide across train/test splits, the loader raises and asks
for a local manifest with explicit `subject_id` and `split_group` columns rather
than hiding split membership inside inferred subject identifiers.

The official FIVES archive can also be used without a manifest if it is unzipped as:

```text
data/raw/fives/
  train/
    Original/
      1_A.png
      ...
    Ground Truth/
      1_A.png
      ...
  test/
    Original/
      1_D.png
      ...
    Ground Truth/
      1_D.png
      ...
  Quality Assessment.xlsx
```

FIVES filename suffixes are parsed as `A=AMD`, `D=DR`, `G=glaucoma`, and
`N=normal`. The public archive does not provide patient identifiers in the image
filenames, so the loader uses `image_id` as `split_group` unless a manifest supplies
patient-level metadata.
