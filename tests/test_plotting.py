import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io as skio

from retivasc.plotting import plot_rose_feature_visuals, plot_segmentation_comparison_grid


def test_plot_rose_feature_visuals_writes_file(tmp_path):
    image = np.zeros((48, 48), dtype=float)
    mask = np.zeros((48, 48), dtype=bool)
    mask[6:42, 12] = True
    mask[18, 12:36] = True
    mask[18:40, 32] = True
    mask[34, 24:42] = True
    image[mask] = 1.0
    image += np.linspace(0.0, 0.2, image.shape[1])[None, :]

    out_path = tmp_path / "rose_feature_visuals.png"
    fig = plot_rose_feature_visuals(image, mask, out_path, max_dim=64)
    plt.close(fig)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_segmentation_comparison_grid_writes_file(tmp_path):
    image = np.zeros((40, 40), dtype=np.uint8)
    image[20, 6:34] = 255
    manual = image > 0
    pred = np.zeros_like(manual)
    pred[20, 8:30] = True
    image_path = tmp_path / "image.png"
    manual_path = tmp_path / "manual.png"
    pred_path = tmp_path / "pred.png"
    skio.imsave(image_path, image, check_contrast=False)
    skio.imsave(manual_path, manual.astype(np.uint8) * 255, check_contrast=False)
    skio.imsave(pred_path, pred.astype(np.uint8) * 255, check_contrast=False)
    benchmark = pd.DataFrame(
        {
            "image_id": ["case_001"],
            "layer": ["SVC"],
            "label": ["control"],
            "image_path": [str(image_path)],
            "manual_mask_path": [str(manual_path)],
            "method": ["frangi"],
            "pred_mask_path": [str(pred_path)],
            "dice": [0.88],
        }
    )

    out_path = tmp_path / "comparison.png"
    fig = plot_segmentation_comparison_grid(benchmark, out_path, max_dim=64)
    plt.close(fig)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
