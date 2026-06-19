"""Mask-derived embedding workflow for exploratory ROSE vascular phenotypes."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io as skio
from skimage import morphology
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

from retivasc.features_rose import extract_rose_features
from retivasc.preprocess import ensure_grayscale

MASK_EMBEDDING_FEATURES = [
    "vessel_area_fraction",
    "skeleton_length_density",
    "branchpoint_density",
    "endpoint_density",
    "connected_component_count",
    "largest_component_fraction",
    "fractal_dimension_boxcount",
    "mean_segment_length_px",
    "median_segment_length_px",
    "mean_tortuosity_arc_chord",
    "high_tortuosity_fraction",
    "caliber_proxy_mean_px",
    "caliber_proxy_median_px",
    "caliber_proxy_std_px",
    "hole_fraction_or_dropout_proxy",
    "small_component_fraction",
    "orientation_entropy",
]

ROSE_MASK_EMBEDDING_CLAIM_BOUNDARY = (
    "Exploratory mask-derived visualization only; not predictive validation."
)
LAYER_ORDER = ("SVC", "DVC", "SVC+DVC")


def load_binary_mask(path: str | Path, *, min_component_size: int = 0) -> np.ndarray:
    """Read a grayscale image or NumPy array and return a boolean vessel mask."""
    mask_path = Path(path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask path does not exist: {mask_path}")
    if mask_path.suffix.lower() == ".npy":
        arr = np.load(mask_path)
    else:
        arr = skio.imread(mask_path)
    gray = ensure_grayscale(arr)
    mask = np.asarray(gray) != 0
    if min_component_size > 0 and mask.any():
        mask = morphology.remove_small_objects(mask, max_size=max(0, min_component_size - 1))
    return mask.astype(bool, copy=False)


def build_mask_feature_table(
    manifest: pd.DataFrame,
    *,
    min_component_size: int = 0,
    mask_col: str = "mask_path",
) -> pd.DataFrame:
    """Compute one vascular feature row per subject_id x layer from binary masks."""
    _require_columns(
        manifest,
        ["subject_id", "image_id", "layer", mask_col, "diagnosis", "label_source"],
    )
    _validate_label_sources(manifest)
    rows = []
    grouped = manifest.sort_values(["subject_id", "layer", "image_id"]).groupby(
        ["subject_id", "layer"], sort=False, dropna=False
    )
    for (subject_id, layer), group in grouped:
        row = group.iloc[0]
        mask = load_binary_mask(row[mask_col], min_component_size=min_component_size)
        features = compute_mask_embedding_features(mask)
        out = {
            "subject_id": subject_id,
            "image_id": row.get("image_id", None),
            "layer": layer,
            "diagnosis": normalize_diagnosis(row.get("diagnosis", "unknown")),
            "label_source": row.get("label_source", "unknown"),
            "mask_path": str(row[mask_col]),
            "source_row_count": int(len(group)),
            "mask_height": int(mask.shape[0]),
            "mask_width": int(mask.shape[1]),
            "foreground_fraction": float(np.count_nonzero(mask) / mask.size) if mask.size else 0.0,
            **features,
        }
        for optional_col in (
            "eye_id",
            "scanner",
            "field_of_view",
            "pixel_size_um",
            "quality_flag",
            "official_split",
            "split_group",
        ):
            if optional_col in row.index:
                out[optional_col] = row.get(optional_col, None)
        rows.append(out)
    return pd.DataFrame(rows)


def compute_mask_embedding_features(mask: np.ndarray) -> dict[str, float]:
    """Return embedding-oriented vascular features from one binary mask."""
    rose_features = extract_rose_features(mask)
    return {feature: rose_features[feature] for feature in MASK_EMBEDDING_FEATURES}


def prepare_embedding_matrix(
    features: pd.DataFrame,
    feature_cols: list[str],
    scaler: str = "robust",
    impute: str = "median",
) -> tuple[np.ndarray, dict]:
    """Return scaled feature matrix and preprocessing metadata."""
    _require_columns(features, feature_cols)
    values = features[feature_cols].apply(pd.to_numeric, errors="coerce")
    raw_missing = values.isna()
    if impute != "median":
        raise ValueError("Only median imputation is currently supported.")
    medians = values.median(axis=0, skipna=True).fillna(0.0)
    imputed = values.fillna(medians)
    if scaler == "robust":
        scaler_obj = RobustScaler()
    elif scaler == "standard":
        scaler_obj = StandardScaler()
    elif scaler in {"none", None}:
        scaler_obj = None
    else:
        raise ValueError("scaler must be 'robust', 'standard', or 'none'.")
    matrix = imputed.to_numpy(dtype=float)
    scaled = scaler_obj.fit_transform(matrix) if scaler_obj is not None else matrix
    metadata = {
        "feature_cols": list(feature_cols),
        "scaler": scaler or "none",
        "impute": impute,
        "n_missing_values_raw": int(raw_missing.to_numpy().sum()),
        "n_missing_values_imputed": int(raw_missing.to_numpy().sum()),
        "imputation_counts_by_feature": {col: int(raw_missing[col].sum()) for col in feature_cols},
        "imputation_medians": {col: float(medians[col]) for col in feature_cols},
    }
    if scaler_obj is not None:
        metadata["scaler_center"] = _array_to_float_list(getattr(scaler_obj, "center_", []))
        metadata["scaler_scale"] = _array_to_float_list(getattr(scaler_obj, "scale_", []))
    return np.asarray(scaled, dtype=float), metadata


def fit_pca_embedding(
    X: np.ndarray,
    feature_cols: list[str],
    n_components: int = 10,
    random_state: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Return PCA coordinates and variance/loading metadata."""
    matrix = _as_2d_matrix(X)
    if matrix.shape[0] < 2:
        coords = pd.DataFrame({"component_1": np.zeros(matrix.shape[0]), "component_2": 0.0})
        return coords, {"method": "pca", "skipped": "fewer than two rows"}
    n_fit_components = min(n_components, matrix.shape[1], matrix.shape[0] - 1)
    pca = PCA(n_components=n_fit_components, random_state=random_state)
    transformed = pca.fit_transform(matrix)
    data = {f"component_{idx + 1}": transformed[:, idx] for idx in range(transformed.shape[1])}
    if "component_2" not in data:
        data["component_2"] = np.zeros(matrix.shape[0])
    coords = pd.DataFrame(data)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"PC{idx + 1}" for idx in range(pca.components_.shape[0])],
    )
    metadata = {
        "method": "pca",
        "n_components": int(n_fit_components),
        "random_state": int(random_state),
        "explained_variance_ratio": _array_to_float_list(pca.explained_variance_ratio_),
        "explained_variance_ratio_pc1": _safe_component(pca.explained_variance_ratio_, 0),
        "explained_variance_ratio_pc2": _safe_component(pca.explained_variance_ratio_, 1),
        "loadings": loadings.reset_index(names="feature").to_dict(orient="records"),
        "top_pc1_loadings": _top_loadings(loadings, "PC1"),
        "top_pc2_loadings": _top_loadings(loadings, "PC2") if "PC2" in loadings else [],
    }
    return coords, metadata


def fit_umap_embedding(
    X: np.ndarray,
    n_neighbors: int = 8,
    min_dist: float = 0.25,
    metric: str = "euclidean",
    random_state: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Return UMAP coordinates and metadata; requires optional umap-learn."""
    matrix = _as_2d_matrix(X)
    if matrix.shape[0] < 3:
        coords = pd.DataFrame({"component_1": np.zeros(matrix.shape[0]), "component_2": 0.0})
        return coords, {"method": "umap", "skipped": "fewer than three rows"}
    try:
        import umap
    except ModuleNotFoundError as exc:
        raise ImportError(
            "UMAP embeddings require the optional 'umap-learn' package. "
            "Install it to generate UMAP figures, or use PCA/t-SNE outputs."
        ) from exc
    adjusted_neighbors = min(max(2, int(n_neighbors)), matrix.shape[0] - 1)
    model = umap.UMAP(
        n_components=2,
        n_neighbors=adjusted_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    coords_array = model.fit_transform(matrix)
    coords = pd.DataFrame({"component_1": coords_array[:, 0], "component_2": coords_array[:, 1]})
    metadata = {
        "method": "umap",
        "n_neighbors": int(adjusted_neighbors),
        "requested_n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "metric": metric,
        "random_state": int(random_state),
    }
    return coords, metadata


def fit_tsne_embedding(
    X: np.ndarray,
    perplexity: float | None = None,
    random_state: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Return t-SNE coordinates with small-sample-safe perplexity."""
    matrix = _as_2d_matrix(X)
    if matrix.shape[0] < 3:
        coords = pd.DataFrame({"component_1": np.zeros(matrix.shape[0]), "component_2": 0.0})
        return coords, {"method": "tsne", "skipped": "fewer than three rows"}
    selected_perplexity = (
        min(10.0, max(5.0, float((matrix.shape[0] - 1) // 4)))
        if perplexity is None
        else float(perplexity)
    )
    selected_perplexity = min(selected_perplexity, float(matrix.shape[0] - 1))
    model = TSNE(
        n_components=2,
        perplexity=selected_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
        metric="euclidean",
    )
    coords_array = model.fit_transform(matrix)
    coords = pd.DataFrame({"component_1": coords_array[:, 0], "component_2": coords_array[:, 1]})
    metadata = {
        "method": "tsne",
        "perplexity": float(selected_perplexity),
        "init": "pca",
        "learning_rate": "auto",
        "metric": "euclidean",
        "random_state": int(random_state),
    }
    return coords, metadata


def compute_group_separation(
    coords: pd.DataFrame,
    label_col: str = "diagnosis",
    group_a: str = "AD",
    group_b: str = "control",
    n_permutations: int = 1000,
    random_state: int = 0,
) -> dict:
    """Compute exploratory centroid-distance separation with a permutation p-value."""
    if coords.empty or label_col not in coords.columns:
        return {"n_group_a": 0, "n_group_b": 0, "centroid_distance": None, "p_value": None}
    coordinate_cols = _infer_coordinate_cols(coords)
    labels = coords[label_col].map(normalize_diagnosis)
    values = coords[coordinate_cols].to_numpy(dtype=float)
    group_a_mask = labels == group_a
    group_b_mask = labels == group_b
    n_a = int(group_a_mask.sum())
    n_b = int(group_b_mask.sum())
    if n_a == 0 or n_b == 0:
        return {"n_group_a": n_a, "n_group_b": n_b, "centroid_distance": None, "p_value": None}
    observed = _centroid_distance(values, group_a_mask.to_numpy(), group_b_mask.to_numpy())
    rng = np.random.default_rng(random_state)
    combined_mask = (group_a_mask | group_b_mask).to_numpy()
    subset_values = values[combined_mask]
    subset_labels = labels[combined_mask].to_numpy()
    permuted_distances = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(subset_labels)
        perm_a = shuffled == group_a
        perm_b = shuffled == group_b
        permuted_distances.append(_centroid_distance(subset_values, perm_a, perm_b))
    p_value = float(
        (np.count_nonzero(np.asarray(permuted_distances) >= observed) + 1) / (n_permutations + 1)
    )
    silhouette = None
    if n_a + n_b >= 3 and len({group_a, group_b}) == 2:
        try:
            silhouette = float(silhouette_score(subset_values, subset_labels))
        except ValueError:
            silhouette = None
    return {
        "n_group_a": n_a,
        "n_group_b": n_b,
        "centroid_distance": float(observed),
        "n_permutations": int(n_permutations),
        "p_value": p_value,
        "silhouette_score": silhouette,
        "claim_boundary": (
            "Exploratory separation summary only; not validation of disease prediction."
        ),
    }


def attach_metadata(feature_table: pd.DataFrame, coords: pd.DataFrame, method: str) -> pd.DataFrame:
    """Attach subject/layer/diagnosis columns to coordinate output."""
    base_cols = [
        col
        for col in ("subject_id", "image_id", "layer", "diagnosis", "label_source")
        if col in feature_table.columns
    ]
    out = pd.concat(
        [feature_table[base_cols].reset_index(drop=True), coords.reset_index(drop=True)],
        axis=1,
    )
    out["embedding_method"] = method
    return out


def normalize_diagnosis(value: object) -> str:
    """Normalize explicit diagnosis labels to AD/control/unknown."""
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    if text in {"ad", "alzheimers", "alzheimer", "alzheimer's disease", "disease"}:
        return "AD"
    if text in {"control", "controls", "healthy", "normal", "cn"}:
        return "control"
    if text in {"unknown", "nan", "none", ""}:
        return "unknown"
    return "unknown"


def write_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    """Write a JSON file with stable formatting."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    return out


def safe_write_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write parquet when possible and fall back to CSV if the engine is unavailable."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out, index=False)
        return out
    except Exception:
        fallback = out.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        return fallback


def _validate_label_sources(manifest: pd.DataFrame) -> None:
    diagnosis = manifest["diagnosis"].map(normalize_diagnosis)
    label_source = manifest["label_source"].astype("string")
    missing = diagnosis.isin(["AD", "control"]) & (
        label_source.isna() | (label_source.str.strip() == "") | (label_source == "unknown")
    )
    if missing.any():
        msg = (
            "AD/control rows require an explicit label_source. "
            "Mask embeddings do not infer diagnosis from filenames."
        )
        raise ValueError(msg)


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _as_2d_matrix(X: np.ndarray) -> np.ndarray:
    matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding matrix contains NaN or infinite values.")
    return matrix


def _infer_coordinate_cols(coords: pd.DataFrame) -> list[str]:
    for candidates in (["component_1", "component_2"], ["x", "y"]):
        if all(col in coords.columns for col in candidates):
            return candidates
    numeric_cols = [
        col
        for col in coords.select_dtypes(include=[np.number]).columns
        if not col.startswith("subject")
    ]
    if len(numeric_cols) < 2:
        raise ValueError("Could not infer at least two coordinate columns.")
    return numeric_cols[:2]


def _centroid_distance(
    values: np.ndarray, group_a_mask: np.ndarray, group_b_mask: np.ndarray
) -> float:
    centroid_a = values[group_a_mask].mean(axis=0)
    centroid_b = values[group_b_mask].mean(axis=0)
    return float(np.linalg.norm(centroid_a - centroid_b))


def _top_loadings(loadings: pd.DataFrame, component: str, n: int = 5) -> list[dict[str, object]]:
    if component not in loadings:
        return []
    component_loadings = loadings[component].sort_values(
        key=lambda values: values.abs(),
        ascending=False,
    )
    return [
        {"feature": feature, "loading": float(value)}
        for feature, value in component_loadings.head(n).items()
    ]


def _safe_component(values: np.ndarray, index: int) -> float | None:
    return float(values[index]) if len(values) > index else None


def _array_to_float_list(values: Sequence[object]) -> list[float]:
    return [float(value) for value in values]


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
