import json
from pathlib import Path

import numpy as np
from skimage import io as skio

from retivasc.report_html import build_report, render_data_audit_component
from retivasc.report_text import EARLY_VS_LATE_BIOMARKER_NOTE


def _write_demo_metrics(root: Path) -> None:
    reports_dir = root / "reports"
    reports_dir.mkdir()
    metrics = {
        "AUROC": 0.8264,
        "AUROC 95% CI": [0.77, 0.89],
        "AUPRC": 0.935,
        "AUPRC 95% CI": [0.90, 0.97],
        "Brier score": 0.150,
        "train rows": 600,
        "test rows": 200,
        "target": "disease_vs_normal",
        "positive class": "AMD/DR/glaucoma",
        "negative class": "normal",
        "feature max dimension": 512,
        "split level": (
            "image-level (FIVES official); patient linkage unavailable in public release"
        ),
        "test disease prevalence": 0.75,
    }
    (reports_dir / "fives_metrics.json").write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )


def _write_demo_feature_summary(root: Path) -> None:
    summary = {
        "rows": 800,
        "label_counts": {"AMD": 200, "DR": 200, "glaucoma": 200, "normal": 200},
        "split_counts": {"train": 600, "test": 200},
        "feature_max_dim": 512,
    }
    (root / "reports" / "fives_feature_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )


def _write_demo_calibration(root: Path) -> None:
    figures_dir = root / "figures"
    figures_dir.mkdir()
    (figures_dir / "fives_calibration_demo.png").write_bytes(b"demo image")


def _write_demo_fives_data(root: Path) -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[..., 1] = 70
    image[14:18, 4:28, :] = 180
    image[4:28, 15:18, :] = 180
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[14:18, 4:28] = 255
    mask[4:28, 15:18] = 255

    image_path = root / "data" / "raw" / "fives" / "test" / "Original" / "001_N.png"
    mask_path = root / "data" / "raw" / "fives" / "test" / "Ground Truth" / "001_N.png"
    image_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    skio.imsave(image_path, image, check_contrast=False)
    skio.imsave(mask_path, mask, check_contrast=False)


def test_data_audit_component_is_native_html_not_an_image():
    html = render_data_audit_component()

    assert 'id="data-audit-flow"' in html
    assert "src/retivasc/io.py" in html
    assert "data/raw/fives/" in html
    assert "<img" not in html
    assert "figures/data_audit_flow.png" not in html


def test_build_report_writes_local_report_and_pages_site(tmp_path):
    _write_demo_metrics(tmp_path)
    _write_demo_feature_summary(tmp_path)
    _write_demo_calibration(tmp_path)
    _write_demo_fives_data(tmp_path)
    stale_assets = tmp_path / "docs" / "assets"
    stale_assets.mkdir(parents=True)
    (stale_assets / "rose_pipeline_panel.png").write_bytes(b"stale")
    (stale_assets / "rose_feature_distributions.png").write_bytes(b"stale")

    outputs = build_report(tmp_path)

    assert outputs.report_path.exists()
    assert outputs.audit_path.exists()
    assert outputs.site_index_path == tmp_path / "docs" / "index.html"
    assert outputs.site_index_path.exists()
    assert outputs.site_audit_path.exists()
    assert (tmp_path / "docs" / ".nojekyll").exists()
    assert (tmp_path / "docs" / "assets" / "processing_example_panel.png").exists()
    assert (tmp_path / "docs" / "assets" / "fives_calibration_demo.png").exists()
    assert not (tmp_path / "docs" / "assets" / "cross_species_roadmap.png").exists()
    assert not (tmp_path / "docs" / "assets" / "rose_pipeline_panel.png").exists()
    assert not (tmp_path / "docs" / "assets" / "rose_feature_distributions.png").exists()

    site_html = outputs.site_index_path.read_text(encoding="utf-8")
    assert "Retinal vascular data processing demo" in site_html
    assert 'src="assets/processing_example_panel.png"' in site_html
    assert 'src="assets/fives_calibration_demo.png"' in site_html
    assert "Cross-species roadmap" not in site_html
    assert "Feature cache contents" not in site_html
    assert "Dataset caveats" not in site_html
    assert "Next implementation targets" not in site_html
    assert "https://zenodo.org/records/12775880" in site_html
    assert "https://www.nature.com/articles/s41597-022-01564-3" in site_html
    assert "rose_pipeline_panel" not in site_html
    assert "rose_feature_distributions" not in site_html
    assert "Real FIVES fundus example" in site_html
    assert "image-disjoint, not patient-disjoint" in site_html
    assert "test prevalence 75.0%" in site_html
    assert EARLY_VS_LATE_BIOMARKER_NOTE in site_html
    assert "tortuous_segment_fraction" in site_html
    assert "<strong>Timing:</strong> late" in site_html
    assert "density, skeleton length, branch density" not in site_html
    assert "Pending" not in site_html
    assert 'href="../src/retivasc/io.py"' not in site_html
    assert "src/retivasc/io.py" in site_html
