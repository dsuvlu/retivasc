"""Static HTML report builder for the retivasc PI demo."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import date
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from retivasc.report_text import ROSE_MANUAL_MASK_CAVEAT, ROSE_NO_PREDICTION_CAVEAT

DATA_AUDIT_STEPS: list[dict[str, Any]] = [
    {
        "number": "01",
        "stage": "Raw data landing zone",
        "artifact": "data/raw/fives/",
        "operation": (
            "The official FIVES archive is unpacked locally. Fundus images and manual "
            "vessel masks remain outside version control."
        ),
        "code": [{"path": "src/retivasc/io.py", "symbol": "load_fives_manifest"}],
        "checks": ["raw images stay gitignored", "official train/test folders retained"],
    },
    {
        "number": "02",
        "stage": "Manifest audit",
        "artifact": "800 image-mask rows with label and official_split metadata",
        "operation": (
            "Each fundus image is paired with its manual vessel mask. Dataset, image id, "
            "label, official split, and split group are normalized into a table."
        ),
        "code": [
            {"path": "src/retivasc/io.py", "symbol": "load_fives_manifest"},
            {
                "path": "notebooks/02_fives_modeling_calibration_demo.py",
                "symbol": "local data check",
            },
        ],
        "checks": ["image-mask pairing validated", "split metadata preserved"],
    },
    {
        "number": "03",
        "stage": "Mask preprocessing",
        "artifact": "manual vessel mask arrays",
        "operation": (
            "Mask PNGs are loaded, converted to grayscale, binarized, and downsampled "
            "to max dimension 512 for fast reproducible feature extraction."
        ),
        "code": [
            {"path": "src/retivasc/preprocess.py", "symbol": "ensure_grayscale"},
            {"path": "src/retivasc/preprocess.py", "symbol": "resize_mask_to_max_dim"},
        ],
        "checks": ["no synthetic masks created", "feature_max_dim recorded"],
    },
    {
        "number": "04",
        "stage": "Vascular feature extraction",
        "artifact": "one interpretable feature vector per image",
        "operation": (
            "The binary mask is skeletonized and summarized as density, skeleton length "
            "density, branchpoint density, fractal dimension, and connected components."
        ),
        "code": [
            {"path": "src/retivasc/features.py", "symbol": "extract_vascular_features"},
            {"path": "src/retivasc/skeleton.py", "symbol": "skeletonize_mask"},
        ],
        "checks": ["species-agnostic feature definitions", "derived table only"],
    },
    {
        "number": "05",
        "stage": "Feature cache",
        "artifact": "data/interim/fives_features_max512.parquet",
        "operation": (
            "The derived feature table is cached so report regeneration does not re-read "
            "and reprocess every raw mask."
        ),
        "code": [
            {
                "path": "notebooks/02_fives_modeling_calibration_demo.py",
                "symbol": "cache check",
            },
            {"label": "pandas.to_parquet"},
        ],
        "checks": ["800 cached rows expected", "cache includes feature_max_dim"],
    },
    {
        "number": "06",
        "stage": "Leakage-aware split",
        "artifact": "600 train rows and 200 test rows",
        "operation": (
            "The official FIVES split is used when present. Split groups are checked so "
            "the model cannot train and test on the same grouped unit."
        ),
        "code": [
            {"path": "src/retivasc/splits.py", "symbol": "assert_group_split_safe"},
            {"label": "sklearn Pipeline"},
        ],
        "checks": ["official split preferred", "group overlap rejected"],
    },
    {
        "number": "07",
        "stage": "Modeling and calibration",
        "artifact": "figures/fives_calibration_demo.png, reports/fives_metrics.json",
        "operation": (
            "A simple logistic model demonstrates evaluation discipline on FIVES "
            "disease-vs-normal labels using AUROC, AUPRC, Brier score, and bootstrap CIs."
        ),
        "code": [
            {"path": "src/retivasc/plotting.py", "symbol": "plot_calibration"},
            {"label": "sklearn.metrics"},
        ],
        "checks": ["calibration reported", "not an ADRD prediction claim"],
    },
    {
        "number": "08",
        "stage": "Static PI report",
        "artifact": "reports/retivasc_pi_demo.html",
        "operation": (
            "The deliverable consumes derived figures, cached features, metric JSON, "
            "explicit caveats, and this HTML audit trail."
        ),
        "code": [{"path": "src/retivasc/report_html.py", "symbol": "build_report"}],
        "checks": ["raw files not embedded", "audit rendered as HTML"],
    },
]


@dataclass(frozen=True)
class ReportOutputs:
    """Paths written by the HTML report builder."""

    report_path: Path
    audit_path: Path
    site_index_path: Path
    site_audit_path: Path


def _read_metrics(root: Path) -> dict[str, Any]:
    metrics_path = root / "reports" / "fives_metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _feature_cache_summary(root: Path) -> dict[str, Any]:
    cache_path = root / "data" / "interim" / "fives_features_max512.parquet"
    if not cache_path.exists():
        return {"path": cache_path, "available": False}

    features = pd.read_parquet(cache_path)
    summary: dict[str, Any] = {
        "path": cache_path,
        "available": True,
        "rows": int(len(features)),
        "columns": list(features.columns),
    }
    if "label" in features.columns:
        summary["label_counts"] = {
            str(label): int(count)
            for label, count in features["label"].value_counts(dropna=False).items()
        }
    if "official_split" in features.columns:
        summary["split_counts"] = {
            str(split): int(count)
            for split, count in features["official_split"].value_counts(dropna=False).items()
        }
    if "feature_max_dim" in features.columns and not features.empty:
        dims = sorted(int(value) for value in features["feature_max_dim"].dropna().unique())
        summary["feature_max_dim"] = dims
    return summary


def _fmt_float(value: Any) -> str:
    if value is None:
        return "Pending"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "Pending"


def _fmt_ci(values: Any) -> str:
    if not isinstance(values, list | tuple) or len(values) != 2:
        return "CI pending"
    return f"95% CI {_fmt_float(values[0])}-{_fmt_float(values[1])}"


def _path_exists(root: Path, relative_path: str) -> bool:
    return (root / relative_path).exists()


def _status_label(available: bool) -> str:
    return "available" if available else "pending"


def _code_ref_html(ref: dict[str, str], code_href_prefix: str | None) -> str:
    if "path" in ref:
        path = ref["path"]
        label = ref.get("symbol", path)
        if code_href_prefix is None:
            return (
                '<span class="code-ref">'
                f"<code>{escape(path)}</code>"
                f"<span>{escape(label)}</span>"
                "</span>"
            )
        return (
            f'<a class="code-ref" href="{escape(code_href_prefix)}{escape(path)}">'
            f"<code>{escape(path)}</code>"
            f"<span>{escape(label)}</span>"
            "</a>"
        )
    return f'<span class="code-ref"><code>{escape(ref["label"])}</code></span>'


def _definition_items(items: dict[str, Any]) -> str:
    rows = []
    for key, value in items.items():
        rows.append(f"<div><dt>{escape(str(key))}</dt><dd>{escape(str(value))}</dd></div>")
    return "\n".join(rows)


def _metric_card(label: str, value: str, detail: str) -> str:
    return (
        '<article class="metric-card">'
        f"<span>{escape(label)}</span>"
        f"<strong>{escape(value)}</strong>"
        f"<em>{escape(detail)}</em>"
        "</article>"
    )


def _table_rows(values: dict[str, int]) -> str:
    if not values:
        return '<tr><td colspan="2">Pending</td></tr>'
    rows = []
    for label, count in sorted(values.items()):
        rows.append(f"<tr><td>{escape(label)}</td><td>{count}</td></tr>")
    return "\n".join(rows)


def _render_asset_status(root: Path) -> str:
    assets = {
        "FIVES calibration figure": "figures/fives_calibration_demo.png",
        "FIVES metrics JSON": "reports/fives_metrics.json",
        "FIVES feature cache": "data/interim/fives_features_max512.parquet",
        "ROSE pipeline figure": "figures/rose_pipeline_panel.png",
        "ROSE feature figure": "figures/rose_feature_distributions.png",
    }
    items = []
    for label, relative_path in assets.items():
        available = _path_exists(root, relative_path)
        items.append(
            '<li class="asset-row">'
            f"<span>{escape(label)}</span>"
            f"<code>{escape(relative_path)}</code>"
            f'<b class="{_status_label(available)}">{_status_label(available)}</b>'
            "</li>"
        )
    return "\n".join(items)


def render_data_audit_component(code_href_prefix: str | None = "../") -> str:
    """Render the end-to-end data audit as native HTML."""
    code_context = (
        "directly linked to the code paths that perform each operation"
        if code_href_prefix is not None
        else "paired with the code paths that perform each operation"
    )
    steps = []
    for step in DATA_AUDIT_STEPS:
        code_refs = "\n".join(_code_ref_html(ref, code_href_prefix) for ref in step["code"])
        checks = "\n".join(f"<li>{escape(check)}</li>" for check in step["checks"])
        steps.append(
            '<article class="audit-step">'
            '<div class="audit-index">'
            f"<span>{escape(step['number'])}</span>"
            "</div>"
            '<div class="audit-body">'
            f'<p class="audit-artifact">{escape(step["artifact"])}</p>'
            f"<h3>{escape(step['stage'])}</h3>"
            f"<p>{escape(step['operation'])}</p>"
            '<div class="code-ref-list">'
            f"{code_refs}"
            "</div>"
            '<ul class="check-list">'
            f"{checks}"
            "</ul>"
            "</div>"
            "</article>"
        )

    return (
        '<section class="panel audit-panel" id="data-audit-flow">'
        '<div class="section-heading">'
        '<p class="eyebrow">End-to-end data audit</p>'
        "<h2>What happens to the data, step by step</h2>"
        "<p>"
        "This is rendered as HTML so the report remains readable, searchable, and "
        f"{code_context}."
        "</p>"
        "</div>"
        '<div class="audit-timeline">'
        f"{''.join(steps)}"
        "</div>"
        "</section>"
    )


def _styles() -> str:
    return """
    :root {
      color-scheme: light;
      --bg: #f5f7f8;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #52616b;
      --line: #d8e0e5;
      --code-bg: #eef3f5;
      --teal: #007f7a;
      --blue: #2f5d8c;
      --amber: #a45f00;
      --rose: #b84545;
      --green-bg: #e8f4ef;
      --amber-bg: #fff4df;
      --red-bg: #faeaea;
      --shadow: 0 14px 35px rgba(31, 41, 51, 0.08);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family:
        Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
        sans-serif;
      line-height: 1.5;
    }

    a {
      color: inherit;
    }

    .report-shell {
      width: min(1180px, calc(100% - 40px));
      margin: 0 auto;
    }

    .report-header {
      padding: 42px 0 26px;
    }

    .eyebrow {
      margin: 0 0 8px;
      color: var(--teal);
      font-size: 0.75rem;
      font-weight: 800;
      letter-spacing: 0;
      text-transform: uppercase;
    }

    h1,
    h2,
    h3 {
      letter-spacing: 0;
      line-height: 1.15;
    }

    h1 {
      max-width: 900px;
      margin: 0;
      font-size: clamp(2.1rem, 5vw, 4.3rem);
    }

    h2 {
      margin: 0;
      font-size: clamp(1.45rem, 3vw, 2.3rem);
    }

    h3 {
      margin: 0;
      font-size: 1rem;
    }

    .lede {
      max-width: 820px;
      margin: 18px 0 0;
      color: var(--muted);
      font-size: 1.08rem;
    }

    .status-strip,
    .metrics-grid,
    .method-grid,
    .roadmap-grid {
      display: grid;
      gap: 14px;
    }

    .status-strip {
      grid-template-columns: repeat(4, minmax(0, 1fr));
      margin-top: 28px;
    }

    .status-pill,
    .metric-card,
    .method-block,
    .roadmap-item,
    .note-block {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }

    .status-pill {
      padding: 14px 16px;
    }

    .status-pill span,
    .metric-card span {
      display: block;
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
    }

    .status-pill strong {
      display: block;
      margin-top: 4px;
      font-size: 1.02rem;
    }

    main {
      display: grid;
      gap: 18px;
      padding: 0 0 56px;
    }

    .panel {
      padding: 26px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }

    .section-heading {
      max-width: 820px;
      margin-bottom: 20px;
    }

    .section-heading p:last-child {
      margin-bottom: 0;
      color: var(--muted);
    }

    .summary-grid {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 18px;
      align-items: start;
    }

    .callout {
      padding: 18px;
      border-left: 4px solid var(--teal);
      border-radius: 8px;
      background: #eef8f7;
    }

    .callout.warning {
      border-left-color: var(--amber);
      background: var(--amber-bg);
    }

    .callout p {
      margin: 0;
    }

    .compact-list {
      margin: 0;
      padding-left: 20px;
      color: var(--muted);
    }

    .compact-list li + li {
      margin-top: 8px;
    }

    .metrics-grid {
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }

    .metric-card {
      padding: 16px;
    }

    .metric-card strong {
      display: block;
      margin-top: 8px;
      font-size: 1.8rem;
      line-height: 1;
    }

    .metric-card em {
      display: block;
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.86rem;
      font-style: normal;
    }

    .method-grid,
    .roadmap-grid {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }

    .method-block,
    .roadmap-item,
    .note-block {
      padding: 18px;
      box-shadow: none;
    }

    .method-block p,
    .roadmap-item p,
    .note-block p {
      margin-bottom: 0;
      color: var(--muted);
    }

    .calibration-layout {
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(280px, 0.75fr);
      gap: 22px;
      align-items: start;
    }

    figure {
      margin: 0;
    }

    figure img {
      display: block;
      width: 100%;
      height: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
    }

    figcaption {
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.88rem;
    }

    dl.data-list {
      display: grid;
      gap: 10px;
      margin: 0;
    }

    dl.data-list div {
      display: grid;
      grid-template-columns: minmax(90px, 0.45fr) 1fr;
      gap: 12px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--line);
    }

    dt {
      color: var(--muted);
      font-weight: 700;
    }

    dd {
      margin: 0;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }

    th,
    td {
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
    }

    th {
      color: var(--muted);
      font-size: 0.76rem;
      text-transform: uppercase;
    }

    code {
      padding: 0.12rem 0.3rem;
      border-radius: 5px;
      background: var(--code-bg);
      font-family:
        "SFMono-Regular", Consolas, "Liberation Mono", Menlo, ui-monospace,
        monospace;
      font-size: 0.88em;
    }

    .audit-panel {
      overflow: hidden;
    }

    .audit-timeline {
      position: relative;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }

    .audit-step {
      display: grid;
      grid-template-columns: 52px 1fr;
      min-height: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
    }

    .audit-index {
      display: flex;
      justify-content: center;
      padding-top: 18px;
      border-right: 1px solid var(--line);
      background: #eef4f4;
      border-radius: 8px 0 0 8px;
    }

    .audit-index span {
      width: 30px;
      height: 30px;
      border-radius: 999px;
      background: var(--teal);
      color: white;
      font-size: 0.78rem;
      font-weight: 800;
      line-height: 30px;
      text-align: center;
    }

    .audit-body {
      padding: 16px;
    }

    .audit-artifact {
      margin: 0 0 8px;
      color: var(--blue);
      font-family:
        "SFMono-Regular", Consolas, "Liberation Mono", Menlo, ui-monospace,
        monospace;
      font-size: 0.82rem;
      font-weight: 700;
    }

    .audit-body > p:not(.audit-artifact) {
      margin: 10px 0 12px;
      color: var(--muted);
    }

    .code-ref-list {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 12px 0;
    }

    .code-ref {
      display: inline-flex;
      flex-direction: column;
      gap: 2px;
      min-width: 150px;
      padding: 8px 9px;
      border: 1px solid #c9d8dc;
      border-radius: 7px;
      background: white;
      color: var(--ink);
      text-decoration: none;
    }

    .code-ref span {
      color: var(--muted);
      font-size: 0.74rem;
    }

    .check-list {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 0;
      padding: 0;
      list-style: none;
    }

    .check-list li {
      padding: 5px 8px;
      border-radius: 999px;
      background: var(--green-bg);
      color: #255443;
      font-size: 0.76rem;
      font-weight: 700;
    }

    .asset-list {
      display: grid;
      gap: 8px;
      margin: 0;
      padding: 0;
      list-style: none;
    }

    .asset-row {
      display: grid;
      grid-template-columns: 1fr minmax(220px, 0.8fr) 86px;
      gap: 10px;
      align-items: center;
      padding: 10px 0;
      border-bottom: 1px solid var(--line);
    }

    .asset-row b {
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 0.75rem;
      text-align: center;
      text-transform: uppercase;
    }

    .available {
      background: var(--green-bg);
      color: #255443;
    }

    .pending {
      background: var(--red-bg);
      color: var(--rose);
    }

    .footer {
      padding: 24px 0 42px;
      color: var(--muted);
      font-size: 0.88rem;
    }

    @media (max-width: 920px) {
      .status-strip,
      .metrics-grid,
      .method-grid,
      .roadmap-grid,
      .audit-timeline,
      .summary-grid,
      .calibration-layout {
        grid-template-columns: 1fr;
      }

      .asset-row {
        grid-template-columns: 1fr;
      }

      .panel {
        padding: 20px;
      }
    }
    """


def _document(title: str, body: str) -> str:
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape(title)}</title>\n"
        f"<style>{_styles()}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


def render_report(
    root: Path | str = ".",
    *,
    asset_prefix: str = "../figures",
    code_href_prefix: str | None = "../",
) -> str:
    """Render the complete PI report as static HTML."""
    project_root = Path(root)
    metrics = _read_metrics(project_root)
    cache_summary = _feature_cache_summary(project_root)
    calibration_available = _path_exists(project_root, "figures/fives_calibration_demo.png")

    metric_cards = "\n".join(
        [
            _metric_card(
                "AUROC",
                _fmt_float(metrics.get("AUROC")),
                _fmt_ci(metrics.get("AUROC 95% CI")),
            ),
            _metric_card(
                "AUPRC",
                _fmt_float(metrics.get("AUPRC")),
                _fmt_ci(metrics.get("AUPRC 95% CI")),
            ),
            _metric_card(
                "Brier score",
                _fmt_float(metrics.get("Brier score")),
                "lower is better",
            ),
            _metric_card(
                "Official split",
                (f"{metrics.get('train rows', 'Pending')} / {metrics.get('test rows', 'Pending')}"),
                "train / test rows",
            ),
        ]
    )

    fives_details = {
        "Target": metrics.get("target", "Pending"),
        "Positive class": metrics.get("positive class", "Pending"),
        "Negative class": metrics.get("negative class", "Pending"),
        "Feature max dim": metrics.get("feature max dimension", "Pending"),
        "Feature cache rows": cache_summary.get("rows", "Pending"),
    }

    if calibration_available:
        calibration_figure = (
            "<figure>"
            f'<img src="{escape(asset_prefix)}/fives_calibration_demo.png" '
            'alt="FIVES calibration and prediction distribution">'
            "<figcaption>"
            "FIVES disease-vs-normal calibration. This is a validation mechanics demo, "
            "not an ADRD model."
            "</figcaption>"
            "</figure>"
        )
    else:
        calibration_figure = (
            '<div class="callout warning">'
            "<p>"
            "The calibration figure has not been generated. Run "
            "<code>pixi run fives-demo</code> after local FIVES data are available."
            "</p>"
            "</div>"
        )

    generated = date.today().isoformat()
    body = f"""
    <div class="report-shell">
      <header class="report-header">
        <p class="eyebrow">retivasc prototype report</p>
        <h1>Retinal vascular computer-vision scaffold for ADRD biomarker work</h1>
        <p class="lede">
          A compact, auditable pipeline for moving retinal vessel masks from local
          medical-image files into interpretable vascular features, leakage-aware
          validation, calibration diagnostics, and a future multimodal Roux/JAX workflow.
        </p>
        <div class="status-strip">
          <div class="status-pill">
            <span>FIVES status</span>
            <strong>{_status_label(bool(metrics))}</strong>
          </div>
          <div class="status-pill">
            <span>ROSE status</span>
            <strong>pending local access</strong>
          </div>
          <div class="status-pill">
            <span>ADRD claims</span>
            <strong>none from public data</strong>
          </div>
          <div class="status-pill">
            <span>Generated</span>
            <strong>{generated}</strong>
          </div>
        </div>
      </header>

      <main>
        <section class="panel">
          <div class="summary-grid">
            <div>
              <p class="eyebrow">Executive summary</p>
              <h2>What this demo establishes now</h2>
              <p>
                The current working dataset is FIVES. It is not an ADRD dataset, but it
                is large enough to demonstrate the engineering discipline needed before
                private Roux/JAX retinal, plasma, clinical, genomic, and mouse-model data
                are joined.
              </p>
              <ul class="compact-list">
                <li>Raw medical images stay local under <code>data/raw/</code>.</li>
                <li>Derived vascular features are cached under <code>data/interim/</code>.</li>
                <li>
                  Validation uses held-out data and reports calibration, not only rank
                  metrics.
                </li>
                <li>
                  ROSE and true ADRD biomarker analyses remain blocked until those data
                  are available.
                </li>
              </ul>
            </div>
            <div class="callout warning">
              <p>
                This report demonstrates a reproducible computer-vision scaffold. It does
                not claim Alzheimer's diagnosis, ADRD risk prediction, or plasma biomarker
                association from FIVES.
              </p>
            </div>
          </div>
        </section>

        <section class="panel">
          <div class="section-heading">
            <p class="eyebrow">FIVES modeling discipline</p>
            <h2>Current quantitative output</h2>
            <p>
              These metrics summarize a simple disease-vs-normal fundus-label model
              trained and evaluated on the official FIVES split.
            </p>
          </div>
          <div class="metrics-grid">
            {metric_cards}
          </div>
        </section>

        <section class="panel">
          <div class="calibration-layout">
            {calibration_figure}
            <div>
              <p class="eyebrow">Model context</p>
              <h2>How to read this section</h2>
              <p>
                FIVES supplies fundus vessel masks and disease labels. The model is a
                deliberately small logistic baseline over interpretable vascular features.
                Its purpose is to show split hygiene, metric reporting, and calibration
                mechanics that will be reused when ADRD-relevant data arrive.
              </p>
              <dl class="data-list">
                {_definition_items(fives_details)}
              </dl>
            </div>
          </div>
        </section>

        {render_data_audit_component(code_href_prefix=code_href_prefix)}

        <section class="panel">
          <div class="section-heading">
            <p class="eyebrow">Feature cache contents</p>
            <h2>What has been materialized locally</h2>
            <p>
              The report reads derived artifacts only. This table is generated from the
              cached feature table when it is present.
            </p>
          </div>
          <div class="method-grid">
            <div class="method-block">
              <h3>Label counts</h3>
              <table>
                <thead><tr><th>Label</th><th>Rows</th></tr></thead>
                <tbody>
                  {_table_rows(cache_summary.get("label_counts", {}))}
                </tbody>
              </table>
            </div>
            <div class="method-block">
              <h3>Split counts</h3>
              <table>
                <thead><tr><th>Split</th><th>Rows</th></tr></thead>
                <tbody>
                  {_table_rows(cache_summary.get("split_counts", {}))}
                </tbody>
              </table>
            </div>
            <div class="method-block">
              <h3>Current assets</h3>
              <ul class="asset-list">
                {_render_asset_status(project_root)}
              </ul>
            </div>
          </div>
        </section>

        <section class="panel">
          <div class="section-heading">
            <p class="eyebrow">Dataset caveats</p>
            <h2>What remains outside the current evidence base</h2>
          </div>
          <div class="method-grid">
            <div class="note-block">
              <h3>ROSE OCTA</h3>
              <p>{escape(ROSE_MANUAL_MASK_CAVEAT)}</p>
              <p>{escape(ROSE_NO_PREDICTION_CAVEAT)}</p>
            </div>
            <div class="note-block">
              <h3>Roux/JAX human data</h3>
              <p>
                The production analysis should replace public-data loaders with
                project-specific joins to retinal imaging, plasma p-tau217, amyloid
                ratios, GFAP, NfL, genomics, and clinical covariates.
              </p>
            </div>
            <div class="note-block">
              <h3>Howell/Reagan mouse data</h3>
              <p>
                The same vascular feature definitions can be applied to mouse retinal
                images, creating a shared phenotype table for cross-species comparison.
              </p>
            </div>
          </div>
        </section>

        <section class="panel">
          <div class="section-heading">
            <p class="eyebrow">Next implementation targets</p>
            <h2>What should come next</h2>
          </div>
          <div class="roadmap-grid">
            <div class="roadmap-item">
              <h3>Bring in ROSE or project OCTA</h3>
              <p>
                Generate the OCTA-specific CV panel and manual-mask feature distributions
                once the local dataset is accessible.
              </p>
            </div>
            <div class="roadmap-item">
              <h3>Add cohort joins</h3>
              <p>
                Create typed loaders for biomarker, clinical, genomic, and imaging
                metadata with explicit missingness and site or batch checks.
              </p>
            </div>
            <div class="roadmap-item">
              <h3>Broaden validation</h3>
              <p>
                Add calibration diagnostics, biologically informed ablations, and
                sensitivity analyses before making any ADRD-facing claim.
              </p>
            </div>
          </div>
        </section>
      </main>

      <footer class="footer">
        Generated by <code>retivasc.report_html</code>. Notebooks remain as provenance
        for exploratory analysis; this static report is the presentation deliverable.
      </footer>
    </div>
    """
    return _document("retivasc PI Demo Report", body)


def render_data_audit_page(code_href_prefix: str | None = "../") -> str:
    body = (
        '<div class="report-shell">'
        '<header class="report-header">'
        '<p class="eyebrow">retivasc audit artifact</p>'
        "<h1>End-to-end data audit</h1>"
        '<p class="lede">'
        "A standalone HTML view of the data operations and owning code paths used by "
        "the current FIVES pipeline."
        "</p>"
        "</header>"
        "<main>"
        f"{render_data_audit_component(code_href_prefix=code_href_prefix)}"
        "</main>"
        "</div>"
    )
    return _document("retivasc Data Audit", body)


def _copy_site_assets(project_root: Path, site_dir: Path) -> None:
    assets_dir = site_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    asset_map = {
        project_root / "figures" / "fives_calibration_demo.png": (
            assets_dir / "fives_calibration_demo.png"
        ),
    }
    for source, destination in asset_map.items():
        if source.exists():
            shutil.copy2(source, destination)


def build_report(root: Path | str = ".") -> ReportOutputs:
    """Write local report files and the GitHub Pages-ready docs site."""
    project_root = Path(root)
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    site_dir = project_root / "docs"
    site_dir.mkdir(parents=True, exist_ok=True)
    _copy_site_assets(project_root, site_dir)

    report_path = reports_dir / "retivasc_pi_demo.html"
    audit_path = reports_dir / "data_audit_flow.html"
    site_index_path = site_dir / "index.html"
    site_audit_path = site_dir / "data_audit_flow.html"
    report_path.write_text(render_report(project_root), encoding="utf-8")
    audit_path.write_text(render_data_audit_page(), encoding="utf-8")
    site_index_path.write_text(
        render_report(project_root, asset_prefix="assets", code_href_prefix=None),
        encoding="utf-8",
    )
    site_audit_path.write_text(
        render_data_audit_page(code_href_prefix=None),
        encoding="utf-8",
    )
    (site_dir / ".nojekyll").write_text("", encoding="utf-8")
    return ReportOutputs(
        report_path=report_path,
        audit_path=audit_path,
        site_index_path=site_index_path,
        site_audit_path=site_audit_path,
    )


def main() -> None:
    outputs = build_report(Path.cwd())
    print(f"Wrote {outputs.report_path}")
    print(f"Wrote {outputs.audit_path}")
    print(f"Wrote {outputs.site_index_path}")
    print(f"Wrote {outputs.site_audit_path}")


if __name__ == "__main__":
    main()
