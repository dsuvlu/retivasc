import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Data Flow Tutorial

    This bare-bones notebook shows the same processing pattern for each public dataset:

    1. Load a local manifest.
    2. Inspect image and mask metadata.
    3. Read one image-mask pair.
    4. Run the classical vessel segmentation baseline.
    5. Skeletonize the manual mask.
    6. Extract interpretable vascular features.

    The notebook does not download data and does not invent labels. If a local dataset is
    missing or ambiguous, it shows the metadata error and stops that dataset's example.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from skimage import io as skio
    from skimage import transform

    from retivasc.features import extract_vascular_features
    from retivasc.io import DataNotFoundError, load_fives_manifest, load_rose_manifest
    from retivasc.preprocess import ensure_grayscale, resize_mask_to_max_dim
    from retivasc.segment import classical_vesselness_mask
    from retivasc.skeleton import skeletonize_mask

    return (
        DataNotFoundError,
        Path,
        classical_vesselness_mask,
        ensure_grayscale,
        extract_vascular_features,
        load_fives_manifest,
        load_rose_manifest,
        np,
        pd,
        plt,
        resize_mask_to_max_dim,
        skeletonize_mask,
        skio,
        transform,
    )


@app.cell
def _(
    DataNotFoundError,
    Path,
    classical_vesselness_mask,
    ensure_grayscale,
    extract_vascular_features,
    np,
    pd,
    plt,
    resize_mask_to_max_dim,
    skeletonize_mask,
    skio,
    transform,
):
    def load_or_error(loader, root: Path, **kwargs):
        try:
            return loader(root, **kwargs), None
        except DataNotFoundError as exc:
            return None, str(exc)
        except ValueError as exc:
            return None, str(exc)

    def load_rose_for_demo(loader, root: Path):
        manifest, error = load_or_error(loader, root)
        if manifest is not None or error is None:
            return manifest, error
        manifest, fallback_error = load_or_error(loader, root, require_split_safe=False)
        if manifest is None:
            return None, fallback_error or error
        warning = (
            "Loaded for non-split-sensitive demonstration only. "
            "Add a local manifest.csv with explicit subject_id and split_group columns "
            "before ROSE split-sensitive analysis.\n\n"
            f"{error}"
        )
        return manifest, warning

    def manifest_summary(name: str, root: Path, manifest: pd.DataFrame | None, error: str | None):
        if manifest is None:
            return f"""
            ## {name} Manifest

            Could not load `{root}`.

            ```text
            {error}
            ```
            """

        lines = [
            f"## {name} Manifest",
            "",
            f"- Root: `{root}`",
            f"- Rows: `{len(manifest)}`",
        ]
        for column in ("dataset", "modality", "layer", "label", "official_split"):
            if column in manifest.columns:
                counts = manifest[column].value_counts(dropna=False).to_dict()
                lines.append(f"- `{column}` counts: `{counts}`")
        if "subject_id" in manifest.columns:
            lines.append(f"- Unique `subject_id`: `{manifest['subject_id'].nunique()}`")
        if "split_group" in manifest.columns:
            lines.append(f"- Unique `split_group`: `{manifest['split_group'].nunique()}`")
        if error is not None:
            lines.extend(["", "### Warning", "", "```text", error, "```"])
        return "\n".join(lines)

    def choose_sample(manifest: pd.DataFrame | None, *, prefer_layer: str | None = None):
        if manifest is None or manifest.empty:
            return None
        if prefer_layer is not None and "layer" in manifest.columns:
            preferred = manifest.loc[manifest["layer"].astype("string") == prefer_layer]
            if not preferred.empty:
                return preferred.iloc[0]
        return manifest.iloc[0]

    def resize_image_to_max_dim(image: np.ndarray, max_dim: int = 512) -> np.ndarray:
        gray = ensure_grayscale(image)
        rows, cols = gray.shape
        current_max = max(rows, cols)
        if current_max <= max_dim:
            return gray
        scale = max_dim / current_max
        shape = (max(1, round(rows * scale)), max(1, round(cols * scale)))
        return transform.resize(gray, shape, preserve_range=True, anti_aliasing=True)

    def feature_markdown(features: dict[str, float]) -> str:
        lines = ["### Feature Values", ""]
        for key, value in features.items():
            lines.append(f"- `{key}`: `{float(value):.4f}`")
        return "\n".join(lines)

    def plot_data_flow(row: pd.Series, out_path: Path, title: str):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image = resize_image_to_max_dim(skio.imread(row["image_path"]))
        mask = ensure_grayscale(skio.imread(row["mask_path"])) > 0
        mask = resize_mask_to_max_dim(mask, 512)
        baseline_mask = classical_vesselness_mask(image, threshold="percentile:90")
        skeleton = skeletonize_mask(mask)
        features = extract_vascular_features(mask)

        fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), constrained_layout=True)
        panels = [
            (image, "Image", "gray"),
            (mask, "Manual mask", "gray"),
            (baseline_mask, "Classical baseline", "gray"),
            (skeleton, "Manual-mask skeleton", "magma"),
        ]
        for axis, (panel, panel_title, cmap) in zip(axes, panels, strict=True):
            axis.imshow(panel, cmap=cmap)
            axis.set_title(panel_title)
            axis.set_axis_off()
        fig.suptitle(title, y=1.05)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return out_path, features

    return (
        choose_sample,
        feature_markdown,
        load_or_error,
        load_rose_for_demo,
        manifest_summary,
        plot_data_flow,
    )


@app.cell
def _(Path, load_fives_manifest, load_or_error, load_rose_for_demo, load_rose_manifest):
    rose_root = Path("data/raw/rose")
    fives_root = Path("data/raw/fives")
    rose_manifest, rose_error = load_rose_for_demo(load_rose_manifest, rose_root)
    fives_manifest, fives_error = load_or_error(load_fives_manifest, fives_root)
    return fives_error, fives_manifest, fives_root, rose_error, rose_manifest, rose_root


@app.cell
def _(
    fives_error,
    fives_manifest,
    fives_root,
    manifest_summary,
    mo,
    rose_error,
    rose_manifest,
    rose_root,
):
    mo.vstack(
        [
            mo.md(manifest_summary("ROSE", rose_root, rose_manifest, rose_error)),
            mo.md(manifest_summary("FIVES", fives_root, fives_manifest, fives_error)),
        ]
    )
    return


@app.cell
def _(Path, choose_sample, feature_markdown, mo, plot_data_flow, rose_manifest):
    def render_rose_flow():
        rose_sample = choose_sample(rose_manifest, prefer_layer="SVC")
        if rose_sample is None:
            return mo.md(
                """
                ## ROSE Data Flow

                No ROSE example is shown because the local ROSE manifest could not be loaded.
                If official filenames collide across train/test splits, add a local
                `data/raw/rose/manifest.csv` with explicit `subject_id` and `split_group`
                columns.
                """
            )

        rose_path, rose_features = plot_data_flow(
            rose_sample,
            Path("figures/tutorial_rose_data_flow.png"),
            "ROSE local OCTA data flow",
        )
        return mo.vstack(
            [
                mo.md(
                    """
                    ## ROSE Data Flow

                    ROSE demonstrates OCTA ingestion, manual-mask processing,
                    skeletonization, and vascular feature extraction.
                    """
                ),
                mo.image(rose_path, alt="ROSE local image, mask, baseline, and skeleton"),
                mo.md(feature_markdown(rose_features)),
            ]
        )

    render_rose_flow()
    return


@app.cell
def _(Path, choose_sample, feature_markdown, fives_manifest, mo, plot_data_flow):
    def render_fives_flow():
        fives_sample = choose_sample(fives_manifest)
        if fives_sample is None:
            return mo.md(
                """
                ## FIVES Data Flow

                No FIVES example is shown because the local FIVES manifest could not be loaded.
                Place the official FIVES archive under `data/raw/fives/` or supply a manifest.
                """
            )

        fives_path, fives_features = plot_data_flow(
            fives_sample,
            Path("figures/tutorial_fives_data_flow.png"),
            "FIVES local fundus data flow",
        )
        return mo.vstack(
            [
                mo.md(
                    """
                    ## FIVES Data Flow

                    FIVES demonstrates the same feature-extraction path on a larger fundus
                    dataset. The modeling notebook then feeds these features into a small
                    logistic-regression baseline and calibration analysis.
                    """
                ),
                mo.image(fives_path, alt="FIVES local image, mask, baseline, and skeleton"),
                mo.md(feature_markdown(fives_features)),
            ]
        )

    render_fives_flow()
    return


@app.cell
def _(mo):
    mo.md("""
    ## What To Read Next

    - `notebooks/01_rose_octa_feature_demo.py` expands the OCTA feature extraction path.
    - `notebooks/02_fives_modeling_calibration_demo.py` expands the modeling and
      calibration path.
    - `src/retivasc/report_html.py` builds the static report and data-flow audit.
    """)
    return


if __name__ == "__main__":
    app.run()
