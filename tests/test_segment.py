import numpy as np
import pytest

from retivasc.segment import (
    SegmentResult,
    classical_vesselness_mask,
    diffusion_threshold_segment,
    geodesic_voting_segment,
    random_walker_segment,
)


def _synthetic_vessel_image(shape=(64, 64)):
    image = np.zeros(shape, dtype=float)
    image[shape[0] // 2, 8 : shape[1] - 8] = 1.0
    image[12 : shape[0] - 12, shape[1] // 3] = 0.9
    image[24:48, 42] = 0.85
    image = image + 0.05 * np.linspace(0.0, 1.0, shape[1])[None, :]
    return image


def _fov(shape=(64, 64)):
    fov = np.ones(shape, dtype=bool)
    fov[:8, :] = False
    fov[:, :4] = False
    return fov


def test_classical_vesselness_accepts_dark_ridge_mode():
    image = np.ones((32, 32), dtype=float)
    image[16, 4:28] = 0.0

    mask = classical_vesselness_mask(image, black_ridges=True)

    assert mask.shape == image.shape
    assert mask.dtype == bool


def test_diffusion_threshold_segment_output_shape_bool_and_fov():
    image = _synthetic_vessel_image()
    fov = _fov(image.shape)

    result = diffusion_threshold_segment(
        image,
        fov_mask=fov,
        n_iter=3,
        threshold="otsu",
        min_size=2,
        clahe=False,
    )

    assert isinstance(result, SegmentResult)
    assert result.mask.shape == image.shape
    assert result.mask.dtype == bool
    assert result.score is not None
    assert not result.mask[~fov].any()


def test_random_walker_segment_output_shape_bool_and_fov():
    image = _synthetic_vessel_image()
    fov = _fov(image.shape)

    result = random_walker_segment(
        image,
        fov_mask=fov,
        vessel_score=image,
        vessel_seed_quantile=0.95,
        background_seed_quantile=0.25,
        mode="bf",
        min_size=2,
    )

    assert result.mask.shape == image.shape
    assert result.mask.dtype == bool
    assert result.score is not None
    assert result.diagnostics["vessel_seed_count"] > 0
    assert result.diagnostics["background_seed_count"] > 0
    assert not result.mask[~fov].any()


def test_random_walker_segment_fails_cleanly_without_vessel_seeds():
    image = np.zeros((16, 16), dtype=float)

    with pytest.raises(ValueError, match="vessel seeds"):
        random_walker_segment(
            image,
            vessel_score=np.zeros_like(image),
            vessel_seed_quantile=1.0,
            background_seed_quantile=0.5,
            mode="bf",
        )


def test_geodesic_voting_segment_deterministic_and_respects_max_pairs():
    image = _synthetic_vessel_image()
    fov = _fov(image.shape)

    first = geodesic_voting_segment(
        image,
        fov_mask=fov,
        vessel_score=image,
        max_seeds=12,
        max_pairs=20,
        vote_threshold=0.05,
        min_size=2,
        random_state=3,
    )
    second = geodesic_voting_segment(
        image,
        fov_mask=fov,
        vessel_score=image,
        max_seeds=12,
        max_pairs=20,
        vote_threshold=0.05,
        min_size=2,
        random_state=3,
    )

    assert first.mask.shape == image.shape
    assert first.mask.dtype == bool
    assert np.array_equal(first.mask, second.mask)
    assert first.diagnostics["num_pairs_attempted"] <= 20
    assert not first.mask[~fov].any()


def test_geodesic_voting_dilation_radius_increases_mask_area():
    image = _synthetic_vessel_image()

    thin = geodesic_voting_segment(
        image,
        vessel_score=image,
        max_seeds=12,
        max_pairs=20,
        vote_threshold=0.05,
        min_size=2,
        dilation_radius=0,
        random_state=3,
    )
    widened = geodesic_voting_segment(
        image,
        vessel_score=image,
        max_seeds=12,
        max_pairs=20,
        vote_threshold=0.05,
        min_size=2,
        dilation_radius=2,
        random_state=3,
    )

    assert np.count_nonzero(widened.mask) >= np.count_nonzero(thin.mask)
