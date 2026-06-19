"""Classical, GPU-free vessel segmentation baseline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from skimage import exposure, filters, graph, measure, morphology, transform

from retivasc.preprocess import ensure_grayscale, normalize_image
from retivasc.skeleton import branchpoint_mask, endpoint_mask, skeletonize_mask


@dataclass(frozen=True)
class SegmentResult:
    """Common result object for native and external segmentation comparators."""

    method: str
    mask: np.ndarray
    score: np.ndarray | None
    params: dict
    diagnostics: dict


def classical_vesselness_mask(
    image: np.ndarray, *, threshold: str = "otsu", black_ridges: bool = False
) -> np.ndarray:
    """Return a binary vessel mask using Frangi vesselness plus thresholding.

    The default assumes bright vessels, which matches OCTA projections. Set
    ``black_ridges=True`` for dark-vessel fundus-style inputs.
    """
    gray = normalize_image(ensure_grayscale(image))
    if gray.size == 0:
        return np.zeros_like(gray, dtype=bool)

    vesselness = filters.frangi(gray, black_ridges=black_ridges)
    vesselness = normalize_image(vesselness)
    if not vesselness.any():
        vesselness = gray

    if threshold == "otsu":
        cutoff = filters.threshold_otsu(vesselness)
    elif threshold == "yen":
        cutoff = filters.threshold_yen(vesselness)
    elif threshold.startswith("percentile:"):
        cutoff = float(np.percentile(vesselness, float(threshold.split(":", maxsplit=1)[1])))
    else:
        msg = "threshold must be 'otsu', 'yen', or 'percentile:<value>'."
        raise ValueError(msg)

    return cleanup_mask(vesselness > cutoff)


def frangi_segment(
    image: np.ndarray,
    fov_mask: np.ndarray | None = None,
    *,
    threshold: str = "otsu",
    black_ridges: bool = False,
    min_size: int = 16,
) -> SegmentResult:
    """Return Frangi vesselness as a comparator-compatible result."""
    gray = _prepare_gray(image)
    score = _frangi_score(gray, black_ridges=black_ridges)
    mask = _threshold_score(score, threshold)
    mask = cleanup_mask(mask, min_size=min_size)
    mask = _apply_fov(mask, fov_mask)
    score = _apply_fov_score(score, fov_mask)
    return SegmentResult(
        method="frangi",
        mask=mask,
        score=score,
        params={
            "threshold": threshold,
            "black_ridges": black_ridges,
            "min_size": min_size,
        },
        diagnostics={"vessel_pixels": int(np.count_nonzero(mask))},
    )


def diffusion_threshold_segment(
    image: np.ndarray,
    fov_mask: np.ndarray | None = None,
    *,
    n_iter: int = 15,
    kappa: float = 0.08,
    gamma: float = 0.15,
    threshold: str | float = "sauvola",
    window_size: int = 31,
    min_size: int = 24,
    clahe: bool = True,
    black_ridges: bool = False,
) -> SegmentResult:
    """Segment vessels with anisotropic diffusion, vesselness, and thresholding."""
    if n_iter < 0:
        raise ValueError("n_iter must be non-negative.")
    if kappa <= 0:
        raise ValueError("kappa must be positive.")
    if not 0 < gamma <= 0.25:
        raise ValueError("gamma must be in (0, 0.25] for stable 4-neighbor diffusion.")

    gray = _prepare_gray(image)
    working = 1.0 - gray if black_ridges else gray
    if clahe and working.size:
        working = exposure.equalize_adapthist(working, clip_limit=0.01)
    diffused = anisotropic_diffusion(working, n_iter=n_iter, kappa=kappa, gamma=gamma)
    score = _frangi_score(diffused, black_ridges=False)
    if not score.any():
        score = normalize_image(diffused)
    mask = _threshold_score(score, threshold, window_size=window_size)
    mask = cleanup_mask(mask, min_size=min_size)
    mask = _apply_fov(mask, fov_mask)
    score = _apply_fov_score(score, fov_mask)
    return SegmentResult(
        method="diffusion_threshold",
        mask=mask,
        score=score,
        params={
            "n_iter": n_iter,
            "kappa": kappa,
            "gamma": gamma,
            "threshold": threshold,
            "window_size": window_size,
            "min_size": min_size,
            "clahe": clahe,
            "black_ridges": black_ridges,
        },
        diagnostics={"vessel_pixels": int(np.count_nonzero(mask))},
    )


def random_walker_segment(
    image: np.ndarray,
    fov_mask: np.ndarray | None = None,
    *,
    vessel_score: np.ndarray | None = None,
    vessel_seed_quantile: float = 0.97,
    background_seed_quantile: float = 0.20,
    beta: float = 90.0,
    mode: str = "cg_j",
    min_size: int = 24,
    black_ridges: bool = False,
) -> SegmentResult:
    """Segment vessels by propagating vessel/background seeds with random walker."""
    gray = _prepare_gray(image)
    fov = _coerce_fov(fov_mask, gray.shape)
    score = _coerce_score(vessel_score, gray.shape) if vessel_score is not None else None
    if score is None:
        score = _frangi_score(gray, black_ridges=black_ridges)
    score = normalize_image(score)

    markers = _random_walker_markers(
        score,
        fov,
        vessel_seed_quantile=vessel_seed_quantile,
        background_seed_quantile=background_seed_quantile,
    )
    vessel_seed_count = int(np.count_nonzero(markers == 2))
    background_seed_count = int(np.count_nonzero(markers == 1))
    if vessel_seed_count == 0:
        raise ValueError("random_walker_segment could not find vessel seeds.")
    if background_seed_count == 0:
        raise ValueError("random_walker_segment could not find background seeds.")

    if not np.any(markers == 0):
        vessel_probability = (markers == 2).astype(float)
    else:
        try:
            from skimage.segmentation import random_walker

            probabilities = random_walker(
                gray,
                markers,
                beta=beta,
                mode=mode,
                return_full_prob=True,
            )
            probabilities = np.asarray(probabilities)
            if probabilities.ndim == gray.ndim:
                vessel_probability = (probabilities == 2).astype(float)
            else:
                vessel_probability = np.asarray(probabilities[1], dtype=float)
        except Exception as exc:
            raise ValueError(f"random_walker_segment failed: {exc}") from exc

    vessel_probability = normalize_image(vessel_probability)
    mask = cleanup_mask(vessel_probability >= 0.5, min_size=min_size)
    mask = _apply_fov(mask, fov)
    score = _apply_fov_score(vessel_probability, fov)
    return SegmentResult(
        method="random_walker",
        mask=mask,
        score=score,
        params={
            "vessel_seed_quantile": vessel_seed_quantile,
            "background_seed_quantile": background_seed_quantile,
            "beta": beta,
            "mode": mode,
            "min_size": min_size,
            "black_ridges": black_ridges,
        },
        diagnostics={
            "vessel_seed_count": vessel_seed_count,
            "background_seed_count": background_seed_count,
            "vessel_pixels": int(np.count_nonzero(mask)),
        },
    )


def geodesic_voting_segment(
    image: np.ndarray,
    fov_mask: np.ndarray | None = None,
    *,
    vessel_score: np.ndarray | None = None,
    max_seeds: int = 64,
    max_pairs: int = 512,
    cost_power: float = 2.0,
    vote_threshold: str | float = "otsu",
    downsample_max_dim: int = 512,
    min_size: int = 24,
    dilation_radius: int = 1,
    random_state: int = 0,
    black_ridges: bool = False,
) -> SegmentResult:
    """Segment vessels by accumulating minimal paths through high-vesselness pixels."""
    start_time = time.perf_counter()
    if max_seeds < 2:
        raise ValueError("max_seeds must be at least 2.")
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive.")
    if cost_power <= 0:
        raise ValueError("cost_power must be positive.")
    if dilation_radius < 0:
        raise ValueError("dilation_radius must be non-negative.")

    gray_original = _prepare_gray(image)
    fov_original = _coerce_fov(fov_mask, gray_original.shape)
    gray, fov, scale = _resize_for_geodesic(gray_original, fov_original, downsample_max_dim)
    score = _coerce_score(vessel_score, gray_original.shape) if vessel_score is not None else None
    if score is None:
        score = _frangi_score(gray_original, black_ridges=black_ridges)
    if scale != 1.0:
        score = transform.resize(
            score,
            gray.shape,
            preserve_range=True,
            anti_aliasing=True,
        )
    score = normalize_image(score)
    score = _apply_fov_score(score, fov)

    seeds = _geodesic_seed_points(score, fov, max_seeds=max_seeds)
    if len(seeds) < 2:
        empty = np.zeros_like(gray_original, dtype=bool)
        return SegmentResult(
            method="geodesic_voting",
            mask=empty,
            score=np.zeros_like(gray_original, dtype=float),
            params={
                "max_seeds": max_seeds,
                "max_pairs": max_pairs,
                "cost_power": cost_power,
                "vote_threshold": vote_threshold,
                "downsample_max_dim": downsample_max_dim,
                "min_size": min_size,
                "dilation_radius": dilation_radius,
                "random_state": random_state,
                "black_ridges": black_ridges,
            },
            diagnostics={
                "num_seeds": len(seeds),
                "num_pairs_attempted": 0,
                "num_paths_found": 0,
                "runtime_seconds": time.perf_counter() - start_time,
                "downsample_factor": scale,
            },
        )

    pairs = _seed_pairs(seeds, max_pairs=max_pairs, random_state=random_state)
    cost = 1.0 / (1.0e-3 + np.clip(score, 0.0, 1.0) ** cost_power)
    cost = np.where(fov, cost, float(cost.max() * 100.0 + 1.0))
    votes = np.zeros_like(score, dtype=float)
    path_lengths = []
    for start, end in pairs:
        try:
            path, _ = graph.route_through_array(
                cost,
                tuple(start),
                tuple(end),
                fully_connected=True,
                geometric=True,
            )
        except Exception:
            continue
        if not path:
            continue
        coords = np.asarray(path, dtype=int)
        votes[coords[:, 0], coords[:, 1]] += 1.0
        path_lengths.append(int(coords.shape[0]))

    vote_score = normalize_image(votes)
    if scale != 1.0:
        vote_score = transform.resize(
            vote_score,
            gray_original.shape,
            preserve_range=True,
            anti_aliasing=True,
        )
    mask = _threshold_score(vote_score, vote_threshold)
    if dilation_radius > 0 and mask.any():
        mask = morphology.dilation(mask, morphology.disk(dilation_radius))
    mask = cleanup_mask(mask, min_size=min_size)
    mask = _apply_fov(mask, fov_original)
    vote_score = _apply_fov_score(vote_score, fov_original)
    return SegmentResult(
        method="geodesic_voting",
        mask=mask,
        score=vote_score,
        params={
            "max_seeds": max_seeds,
            "max_pairs": max_pairs,
            "cost_power": cost_power,
            "vote_threshold": vote_threshold,
            "downsample_max_dim": downsample_max_dim,
            "min_size": min_size,
            "dilation_radius": dilation_radius,
            "random_state": random_state,
            "black_ridges": black_ridges,
        },
        diagnostics={
            "num_seeds": len(seeds),
            "num_pairs_attempted": len(pairs),
            "num_paths_found": len(path_lengths),
            "mean_path_length": float(np.mean(path_lengths)) if path_lengths else 0.0,
            "runtime_seconds": time.perf_counter() - start_time,
            "downsample_factor": scale,
            "vessel_pixels": int(np.count_nonzero(mask)),
        },
    )


def anisotropic_diffusion(
    image: np.ndarray,
    *,
    n_iter: int = 15,
    kappa: float = 0.08,
    gamma: float = 0.15,
) -> np.ndarray:
    """Perona-Malik-style 4-neighbor anisotropic diffusion on a normalized image."""
    out = _prepare_gray(image).astype(float, copy=True)
    if out.size == 0 or n_iter == 0:
        return out
    for _ in range(n_iter):
        north = np.zeros_like(out)
        south = np.zeros_like(out)
        east = np.zeros_like(out)
        west = np.zeros_like(out)
        north[1:, :] = out[:-1, :] - out[1:, :]
        south[:-1, :] = out[1:, :] - out[:-1, :]
        east[:, :-1] = out[:, 1:] - out[:, :-1]
        west[:, 1:] = out[:, :-1] - out[:, 1:]
        update = sum(
            np.exp(-((delta / kappa) ** 2)) * delta
            for delta in (north, south, east, west)
        )
        out = np.clip(out + gamma * update, 0.0, 1.0)
    return out


def cleanup_mask(mask: np.ndarray, *, min_size: int = 16) -> np.ndarray:
    """Remove small components and fill obvious holes."""
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.size == 0:
        return mask_bool
    max_size = max(0, min_size - 1)
    cleaned = morphology.remove_small_objects(mask_bool, max_size=max_size)
    cleaned = morphology.closing(cleaned, morphology.disk(1))
    cleaned = morphology.remove_small_holes(cleaned, max_size=max_size)
    return cleaned.astype(bool)


def _prepare_gray(image: np.ndarray) -> np.ndarray:
    return normalize_image(ensure_grayscale(image)).astype(np.float32, copy=False)


def _frangi_score(image: np.ndarray, *, black_ridges: bool = False) -> np.ndarray:
    gray = _prepare_gray(image)
    if gray.size == 0:
        return np.zeros_like(gray, dtype=float)
    score = filters.frangi(gray, black_ridges=black_ridges)
    score = normalize_image(score)
    return score if score.any() else np.zeros_like(gray, dtype=float)


def _threshold_score(
    score: np.ndarray,
    threshold: str | float,
    *,
    window_size: int = 31,
) -> np.ndarray:
    arr = normalize_image(score)
    if arr.size == 0 or not arr.any():
        return np.zeros_like(arr, dtype=bool)
    if isinstance(threshold, int | float):
        cutoff = float(threshold)
        return arr > cutoff
    if threshold == "otsu":
        cutoff = filters.threshold_otsu(arr)
        return arr > cutoff
    if threshold == "yen":
        cutoff = filters.threshold_yen(arr)
        return arr > cutoff
    if threshold == "li":
        cutoff = filters.threshold_li(arr)
        return arr > cutoff
    if threshold == "sauvola":
        odd_window = max(3, int(window_size) | 1)
        cutoff = filters.threshold_sauvola(arr, window_size=odd_window)
        return arr > cutoff
    if threshold.startswith("percentile:"):
        cutoff = float(np.percentile(arr, float(threshold.split(":", maxsplit=1)[1])))
        return arr > cutoff
    msg = "threshold must be a float or one of 'otsu', 'yen', 'li', 'sauvola', percentile:<value>."
    raise ValueError(msg)


def _coerce_fov(fov_mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if fov_mask is None:
        return np.ones(shape, dtype=bool)
    fov = np.asarray(fov_mask, dtype=bool)
    if fov.shape != shape:
        msg = f"fov_mask shape {fov.shape} does not match image shape {shape}."
        raise ValueError(msg)
    return fov


def _apply_fov(mask: np.ndarray, fov_mask: np.ndarray | None) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if fov_mask is None:
        return mask_bool
    return mask_bool & _coerce_fov(fov_mask, mask_bool.shape)


def _apply_fov_score(score: np.ndarray, fov_mask: np.ndarray | None) -> np.ndarray:
    arr = np.asarray(score, dtype=float)
    if fov_mask is None:
        return arr
    return np.where(_coerce_fov(fov_mask, arr.shape), arr, 0.0)


def _coerce_score(score: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(score, dtype=float)
    if arr.shape != shape:
        msg = f"vessel_score shape {arr.shape} does not match image shape {shape}."
        raise ValueError(msg)
    return arr


def _random_walker_markers(
    score: np.ndarray,
    fov: np.ndarray,
    *,
    vessel_seed_quantile: float,
    background_seed_quantile: float,
) -> np.ndarray:
    if not 0 <= background_seed_quantile < vessel_seed_quantile <= 1:
        msg = "Require 0 <= background_seed_quantile < vessel_seed_quantile <= 1."
        raise ValueError(msg)
    markers = np.zeros(score.shape, dtype=np.int32)
    in_fov_scores = score[fov]
    if in_fov_scores.size == 0:
        return markers
    if not np.any(in_fov_scores):
        markers[fov] = 1
        markers[~fov] = 1
        return markers
    vessel_cutoff = float(np.quantile(in_fov_scores, vessel_seed_quantile))
    background_cutoff = float(np.quantile(in_fov_scores, background_seed_quantile))
    if vessel_cutoff <= background_cutoff:
        vessel_cutoff = float(np.max(in_fov_scores))
    markers[fov & (score <= background_cutoff)] = 1
    markers[fov & (score >= vessel_cutoff)] = 2
    markers[~fov] = 1
    return markers


def _resize_for_geodesic(
    image: np.ndarray,
    fov: np.ndarray,
    max_dim: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if max_dim <= 0:
        raise ValueError("downsample_max_dim must be positive.")
    current_max = max(image.shape)
    if current_max <= max_dim:
        return image, fov, 1.0
    scale = max_dim / current_max
    shape = tuple(max(1, int(round(dim * scale))) for dim in image.shape)
    resized = transform.resize(image, shape, preserve_range=True, anti_aliasing=True)
    resized_fov = transform.resize(
        fov.astype(float),
        shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return resized.astype(np.float32, copy=False), resized_fov >= 0.5, scale


def _geodesic_seed_points(score: np.ndarray, fov: np.ndarray, *, max_seeds: int) -> np.ndarray:
    if not score.any():
        return np.empty((0, 2), dtype=int)
    high = score >= np.quantile(score[fov], 0.9)
    high = cleanup_mask(high & fov, min_size=4)
    skel = skeletonize_mask(high)
    seed_mask = (endpoint_mask(skel) | branchpoint_mask(skel)) & fov
    coords = np.argwhere(seed_mask)
    if coords.shape[0] < 2:
        labels = measure.label(high, connectivity=2)
        centroids = [
            tuple(int(round(value)) for value in region.centroid)
            for region in measure.regionprops(labels)
        ]
        coords = np.asarray(centroids, dtype=int) if centroids else np.argwhere(skel & fov)
    if coords.shape[0] < 2:
        coords = np.argwhere(high & fov)
    if coords.shape[0] <= max_seeds:
        return coords.astype(int, copy=False)
    values = score[coords[:, 0], coords[:, 1]]
    order = np.argsort(values)[::-1]
    return coords[order[:max_seeds]].astype(int, copy=False)


def _seed_pairs(
    seeds: np.ndarray, *, max_pairs: int, random_state: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    all_pairs = list(combinations(seeds, 2))
    if len(all_pairs) <= max_pairs:
        return all_pairs
    rng = np.random.default_rng(random_state)
    selected = rng.choice(len(all_pairs), size=max_pairs, replace=False)
    return [all_pairs[int(idx)] for idx in np.sort(selected)]
