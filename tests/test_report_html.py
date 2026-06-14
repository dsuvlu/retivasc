import json
from pathlib import Path

from retivasc.report_html import build_report, render_data_audit_component


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
    }
    (reports_dir / "fives_metrics.json").write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )


def _write_demo_calibration(root: Path) -> None:
    figures_dir = root / "figures"
    figures_dir.mkdir()
    (figures_dir / "fives_calibration_demo.png").write_bytes(b"demo image")


def test_data_audit_component_is_native_html_not_an_image():
    html = render_data_audit_component()

    assert 'id="data-audit-flow"' in html
    assert "src/retivasc/io.py" in html
    assert "data/raw/fives/" in html
    assert "<img" not in html
    assert "figures/data_audit_flow.png" not in html


def test_build_report_writes_local_report_and_pages_site(tmp_path):
    _write_demo_metrics(tmp_path)
    _write_demo_calibration(tmp_path)

    outputs = build_report(tmp_path)

    assert outputs.report_path.exists()
    assert outputs.audit_path.exists()
    assert outputs.site_index_path == tmp_path / "docs" / "index.html"
    assert outputs.site_index_path.exists()
    assert outputs.site_audit_path.exists()
    assert (tmp_path / "docs" / ".nojekyll").exists()
    assert (tmp_path / "docs" / "assets" / "fives_calibration_demo.png").exists()

    site_html = outputs.site_index_path.read_text(encoding="utf-8")
    assert 'src="assets/fives_calibration_demo.png"' in site_html
    assert 'href="../src/retivasc/io.py"' not in site_html
    assert "src/retivasc/io.py" in site_html
