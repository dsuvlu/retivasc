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
dataset,subject_id,image_id,image_path,mask_path,modality,layer,label,split_group
```

FIVES manifest columns:

```text
dataset,subject_id,image_id,image_path,mask_path,modality,label,split_group
```

Relative `image_path` and `mask_path` values are resolved against the dataset root.

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
