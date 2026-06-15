import numpy as np

from retivasc.segment import classical_vesselness_mask


def test_classical_vesselness_accepts_dark_ridge_mode():
    image = np.ones((32, 32), dtype=float)
    image[16, 4:28] = 0.0

    mask = classical_vesselness_mask(image, black_ridges=True)

    assert mask.shape == image.shape
    assert mask.dtype == bool
