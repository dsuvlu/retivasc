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
from skimage import io as skio

from retivasc.feature_metadata import FEATURE_METADATA
from retivasc.io import DataNotFoundError, load_fives_manifest
from retivasc.plotting import plot_processing_example_panel
from retivasc.report_text import EARLY_VS_LATE_BIOMARKER_NOTE, FIVES_SPLIT_CAVEAT

DATA_AUDIT_STEPS: list[dict[str, Any]] = [
    {
        "number": "01",
        "kind": "Ingest",
        "stage": "Find and pair image data",
        "artifact": "manifest table with image_path and mask_path",
        "plain": (
            "The package first builds a tidy table where each retinal image is paired with "
            "its matching hand-drawn vessel mask."
        ),
        "detail": (
            "ROSE and FIVES loaders detect official folder layouts or read a local manifest. "
            "They normalize image ids, mask paths, ROSE-1 official disease/control labels, "
            "layer metadata, and split groups without downloading or committing medical images. "
            "Local data are expected under data/raw/rose/ and data/raw/fives/."
        ),
        "code": [
            {"path": "src/retivasc/io.py", "symbol": "load_rose_manifest"},
            {"path": "src/retivasc/io.py", "symbol": "load_fives_manifest"},
        ],
        "checks": [
            "raw images stay local",
            "image-mask pairing checked",
            "ROSE-1 labels from official ordering only",
        ],
    },
    {
        "number": "02",
        "kind": "Prepare",
        "stage": "Tidy images and masks",
        "artifact": "normalized arrays for fast processing",
        "plain": (
            "Color images are converted to grayscale, brightness is normalized, and large "
            "masks are resized for a quick demo run."
        ),
        "detail": (
            "The preprocessing helpers standardize inputs before segmentation and feature "
            "extraction, while preserving binary mask semantics."
        ),
        "code": [
            {"path": "src/retivasc/preprocess.py", "symbol": "ensure_grayscale"},
            {"path": "src/retivasc/preprocess.py", "symbol": "resize_mask_to_max_dim"},
        ],
        "checks": ["grayscale conversion", "brightness normalized", "mask remains binary"],
    },
    {
        "number": "03",
        "kind": "Baseline",
        "stage": "Run the classical vessel detector",
        "artifact": "binary baseline vessel mask",
        "plain": (
            "A classic ridge detector lights up thin tube-like structures, thresholds that "
            "response, and removes small specks."
        ),
        "detail": (
            "The Frangi filter is used as an untrained, GPU-free floor for vessel detection. "
            "For this demo it is a baseline, not the final computer-vision claim."
        ),
        "code": [
            {"path": "src/retivasc/segment.py", "symbol": "classical_vesselness_mask"},
            {"path": "src/retivasc/segment.py", "symbol": "cleanup_mask"},
        ],
        "checks": ["Frangi vesselness", "Otsu/Yen/percentile threshold", "morphology cleanup"],
    },
    {
        "number": "04",
        "kind": "Graph",
        "stage": "Trace skeletons and junctions",
        "artifact": "one-pixel centerline plus branch points",
        "plain": (
            "The vessel map is thinned to a centerline, then branch points and endpoints "
            "are marked so the vascular network can be measured."
        ),
        "detail": (
            "Skeletonization turns vessel blobs into graph-like structure, making length, "
            "branching, and tortuosity easier to quantify."
        ),
        "code": [
            {"path": "src/retivasc/skeleton.py", "symbol": "skeletonize_mask"},
            {"path": "src/retivasc/skeleton.py", "symbol": "branchpoint_mask"},
        ],
        "checks": ["centerline extracted", "junctions counted", "endpoints available"],
    },
    {
        "number": "05",
        "kind": "Measure",
        "stage": "Extract vascular features",
        "artifact": "vascular fingerprint per image",
        "plain": (
            "Each mask is reduced to interpretable numbers. Tortuosity burden and caliber "
            "dispersion lead the paper-aligned set; density, branch, and component features "
            "are retained as later-stage or context descriptors."
        ),
        "detail": (
            "Feature metadata records whether each descriptor is early, late, context, or "
            "interpretive, so dataset reports do not overstate density or context features."
        ),
        "code": [
            {"path": "src/retivasc/features.py", "symbol": "extract_vascular_features"},
            {"path": "src/retivasc/features.py", "symbol": "tortuosity_burden"},
        ],
        "checks": ["timing metadata", "no synthetic biomarkers", "derived table only"],
    },
    {
        "number": "06",
        "kind": "Score",
        "stage": "Grade segmentation overlap",
        "artifact": "Dice, IoU, sensitivity, specificity",
        "plain": (
            "Predicted vessel masks can be compared against manual masks to score overlap "
            "and separate missed vessels from false vessel pixels."
        ),
        "detail": (
            "Dice and IoU focus on foreground vessel overlap, while sensitivity and "
            "specificity separate true-vessel recovery from background rejection."
        ),
        "code": [
            {"path": "src/retivasc/metrics.py", "symbol": "dice_score"},
            {"path": "src/retivasc/metrics.py", "symbol": "iou_score"},
        ],
        "checks": [
            "foreground metrics",
            "empty-mask conventions tested",
            "same-shape masks required",
        ],
    },
    {
        "number": "07",
        "kind": "Split",
        "stage": "Split data without leakage",
        "artifact": "train/test groups and leakage checks",
        "plain": (
            "The model is trained on one group of images and tested on a held-out group, "
            "with explicit checks against group overlap."
        ),
        "detail": (
            "FIVES only supports the official image-level split in the public release. "
            "The report states that this is image-disjoint, not patient-disjoint."
        ),
        "code": [{"path": "src/retivasc/splits.py", "symbol": "assert_group_split_safe"}],
        "checks": ["group overlap rejected", "split level stated", "patient IDs not invented"],
    },
    {
        "number": "08",
        "kind": "Model",
        "stage": "Fit the classification baseline",
        "artifact": "disease-vs-normal logistic baseline",
        "plain": (
            "A simple logistic regression uses the vascular feature table to separate "
            "diseased from normal FIVES fundus images."
        ),
        "detail": (
            "The scikit-learn pipeline standardizes features, applies L2-regularized "
            "logistic regression, and uses class balancing for the 75/25 test-set base rate."
        ),
        "code": [
            {
                "path": "notebooks/02_fives_modeling_calibration_demo.py",
                "symbol": "make_pipeline",
            },
            {"label": "StandardScaler + LogisticRegression"},
        ],
        "checks": ["interpretable model", "training-only scaling", "not an ADRD model"],
    },
    {
        "number": "09",
        "kind": "Validate",
        "stage": "Report ranking and calibration",
        "artifact": "AUROC, AUPRC, Brier score, calibration curve",
        "plain": (
            "The demo reports whether the baseline ranks examples well and whether its "
            "probabilities are trustworthy."
        ),
        "detail": (
            "AUROC and AUPRC are ranking metrics. Brier score and the reliability diagram "
            "test probability calibration, which matters for any future risk model."
        ),
        "code": [
            {"path": "src/retivasc/plotting.py", "symbol": "plot_calibration"},
            {"label": "sklearn.metrics"},
        ],
        "checks": ["bootstrap CIs", "prevalence caveat", "calibration shown"],
    },
    {
        "number": "10",
        "kind": "Publish",
        "stage": "Build the static report",
        "artifact": "reports/retivasc_pi_demo.html and docs/index.html",
        "plain": (
            "The report collects the figures, metrics, citations, and caveats into one "
            "static artifact that can be inspected locally or published from docs."
        ),
        "detail": (
            "The public report can include a real FIVES processing example generated from "
            "local data. ROSE-derived image panels remain local-only by default."
        ),
        "code": [{"path": "src/retivasc/report_html.py", "symbol": "build_report"}],
        "checks": ["ROSE images not embedded", "dataset links included", "audit rendered as HTML"],
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
        summary_path = root / "reports" / "fives_feature_summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {"path": cache_path, "available": False}
            summary["path"] = cache_path
            summary["summary_path"] = summary_path
            summary["available"] = False
            summary["source"] = "summary"
            return summary
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


def _fmt_percent(value: Any) -> str:
    if value is None:
        return "Pending"
    try:
        return f"{100 * float(value):.1f}%"
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


def _method_block(title: str, body: str, code: str) -> str:
    return (
        '<article class="method-block">'
        f"<h3>{escape(title)}</h3>"
        f"<p>{escape(body)}</p>"
        f"<code>{escape(code)}</code>"
        "</article>"
    )


def _dataset_reference_cards() -> str:
    references = [
        {
            "title": "ROSE OCTA vessel segmentation dataset",
            "body": (
                "Used here as the OCTA computer-vision scaffold for local ingestion, "
                "manual-mask feature extraction, and skeletonization. ROSE-derived image "
                "examples are not published by default."
            ),
            "url": "https://zenodo.org/records/12775880",
            "label": "Zenodo record",
        },
        {
            "title": "FIVES fundus vessel segmentation dataset",
            "body": (
                "Used here for the larger-scale modeling and calibration demonstration on "
                "manual vessel masks and disease-vs-normal labels."
            ),
            "url": "https://www.nature.com/articles/s41597-022-01564-3",
            "label": "Scientific Data article",
        },
    ]
    cards = []
    for ref in references:
        cards.append(
            '<article class="method-block reference-card">'
            f"<h3>{escape(ref['title'])}</h3>"
            f"<p>{escape(ref['body'])}</p>"
            f'<a href="{escape(ref["url"])}" target="_blank" rel="noopener noreferrer">'
            f"{escape(ref['label'])}"
            "</a>"
            "</article>"
        )
    return "\n".join(cards)


def _processing_method_blocks() -> str:
    blocks = [
        (
            "Baseline vessel segmentation",
            (
                "The segmentation baseline is an untrained Frangi ridge detector followed "
                "by thresholding and morphology cleanup. It is a transparent floor, not a "
                "state-of-the-art model."
            ),
            "segment.py",
        ),
        (
            "Manual-mask feature extraction",
            (
                "Manual vessel masks are converted into a vascular fingerprint with "
                "paper-aligned timing labels. Tortuosity burden leads the portable "
                "feature set; density is retained as a later-stage descriptor."
            ),
            "features.py",
        ),
        (
            "Skeleton and graph structure",
            (
                "The mask is thinned to a one-pixel centerline so branch points, endpoints, "
                "network length, and curviness can be measured consistently."
            ),
            "skeleton.py",
        ),
        (
            "Segmentation scoring",
            (
                "Dice, IoU, sensitivity, and specificity compare a predicted vessel mask "
                "against a manual vessel tracing without being dominated by background pixels."
            ),
            "metrics.py",
        ),
        (
            "Classification baseline",
            (
                "The FIVES model standardizes the vascular features and fits a small "
                "L2-regularized logistic regression with class balancing."
            ),
            "notebook 02",
        ),
        (
            "Calibration reporting",
            (
                "AUROC and AUPRC describe ranking. The Brier score and calibration curve "
                "ask whether predicted probabilities are numerically trustworthy."
            ),
            "plotting.py",
        ),
    ]
    return "\n".join(_method_block(title, body, code) for title, body, code in blocks)


def _feature_timing_cards() -> str:
    selected = [
        "tortuous_segment_fraction",
        "caliber_cv",
        "candidate_crossing_density",
        "vessel_density",
        "major_branch_count",
        "fractal_dimension_boxcount",
        "dropout_heterogeneity",
    ]
    cards = []
    for key in selected:
        metadata = FEATURE_METADATA[key]
        cards.append(
            '<article class="method-block">'
            f"<h3>{escape(key)}</h3>"
            f"<p><strong>Timing:</strong> {escape(str(metadata['timing']))}</p>"
            f"<p>{escape(str(metadata['note']))}</p>"
            "</article>"
        )
    return "\n".join(cards)


def _table_rows(values: dict[str, int]) -> str:
    if not values:
        return '<tr><td colspan="2">Pending</td></tr>'
    rows = []
    for label, count in sorted(values.items()):
        rows.append(f"<tr><td>{escape(label)}</td><td>{count}</td></tr>")
    return "\n".join(rows)


def _render_asset_status(root: Path) -> str:
    assets = {
        "Processing example panel": "figures/processing_example_panel.png",
        "FIVES calibration figure": "figures/fives_calibration_demo.png",
        "FIVES metrics JSON": "reports/fives_metrics.json",
        "FIVES feature cache": "data/interim/fives_features_max512.parquet",
        "Cross-species roadmap": "figures/cross_species_roadmap.png",
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
            f'<p class="audit-kind">{escape(step["kind"])}</p>'
            f"<h3>{escape(step['stage'])}</h3>"
            f'<p class="audit-plain">{escape(step["plain"])}</p>'
            f'<p class="audit-detail">{escape(step["detail"])}</p>'
            '<p class="audit-artifact-label">Output</p>'
            f'<p class="audit-artifact">{escape(step["artifact"])}</p>'
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
        "The audit follows the package from local image files to masks, skeletons, "
        "features, baseline models, calibration, and the final static report. It is "
        f"rendered as HTML so it remains readable, searchable, and {code_context}."
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

    .public-placeholder {
      display: grid;
      gap: 14px;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
    }

    .pipeline-schematic {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      align-items: stretch;
    }

    .pipeline-schematic span {
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 74px;
      padding: 12px;
      border: 1px solid #c9d8dc;
      border-radius: 8px;
      background: white;
      color: var(--ink);
      font-size: 0.88rem;
      font-weight: 750;
      text-align: center;
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

    .method-block code {
      display: inline-block;
      margin-top: 12px;
      color: var(--blue);
      font-weight: 700;
    }

    .reference-card a {
      display: inline-block;
      margin-top: 12px;
      color: var(--teal);
      font-weight: 800;
      text-decoration: none;
    }

    .reference-card a:hover {
      text-decoration: underline;
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
      gap: 16px;
    }

    .audit-step {
      display: grid;
      grid-template-columns: 64px 1fr;
      min-height: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      box-shadow: 0 10px 24px rgba(31, 41, 51, 0.06);
    }

    .audit-index {
      display: flex;
      justify-content: center;
      padding-top: 20px;
      border-right: 1px solid var(--line);
      background: linear-gradient(180deg, #e8f4ef, #eef4f4);
      border-radius: 8px 0 0 8px;
    }

    .audit-index span {
      width: 36px;
      height: 36px;
      border-radius: 999px;
      background: var(--teal);
      color: white;
      font-size: 0.78rem;
      font-weight: 800;
      line-height: 36px;
      text-align: center;
    }

    .audit-body {
      padding: 18px;
    }

    .audit-kind {
      display: inline-block;
      margin: 0 0 9px;
      padding: 4px 8px;
      border-radius: 999px;
      background: #e7eef7;
      color: var(--blue);
      font-size: 0.72rem;
      font-weight: 800;
      text-transform: uppercase;
    }

    .audit-plain {
      margin: 10px 0 8px;
      color: var(--ink);
    }

    .audit-detail {
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 0.92rem;
    }

    .audit-artifact-label {
      margin: 0 0 4px;
      color: var(--muted);
      font-size: 0.7rem;
      font-weight: 800;
      text-transform: uppercase;
    }

    .audit-artifact {
      margin: 0 0 10px;
      color: var(--blue);
      font-family:
        "SFMono-Regular", Consolas, "Liberation Mono", Menlo, ui-monospace,
        monospace;
      font-size: 0.82rem;
      font-weight: 700;
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
      .calibration-layout,
      .pipeline-schematic {
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
    processing_example_available = _path_exists(
        project_root, "figures/processing_example_panel.png"
    )
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
                (
                    f"{_fmt_ci(metrics.get('AUPRC 95% CI'))}; "
                    f"test prevalence {_fmt_percent(metrics.get('test disease prevalence'))}"
                ),
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
        "Split level": metrics.get("split level", "Pending"),
        "Test disease prevalence": _fmt_percent(metrics.get("test disease prevalence")),
        "Feature max dim": metrics.get("feature max dimension", "Pending"),
        "Feature cache rows": cache_summary.get("rows", "Pending"),
    }

    if processing_example_available:
        processing_example_figure = (
            "<figure>"
            f'<img src="{escape(asset_prefix)}/processing_example_panel.png" '
            'alt="FIVES fundus image, manual vessel mask, baseline mask, and skeleton">'
            "<figcaption>"
            "Real FIVES fundus example showing the package's processing stages: image, "
            "manual vessel mask, classical baseline mask, and manual-mask skeleton. The "
            "baseline mask is deliberately simple and may be noisy; the manual mask drives "
            "the feature extraction shown in this demo."
            "</figcaption>"
            "</figure>"
        )
    else:
        processing_example_figure = (
            '<div class="callout warning">'
            "<p>"
            "The real-image processing example has not been generated because local FIVES "
            "data are unavailable. Place FIVES under <code>data/raw/fives/</code> and run "
            "<code>pixi run report</code>."
            "</p>"
            "</div>"
        )

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
        <h1>Retinal vascular data processing demo</h1>
        <p class="lede">
          This report shows how the package processes retinal vascular data: it pairs
          images with vessel masks, runs a simple segmentation baseline, skeletonizes
          the vessels, extracts interpretable vascular features, and demonstrates
          careful validation on a larger public fundus dataset.
        </p>
        <div class="status-strip">
          <div class="status-pill">
            <span>FIVES status</span>
            <strong>{_status_label(bool(metrics))}</strong>
          </div>
          <div class="status-pill">
            <span>ROSE status</span>
            <strong>local-only visuals</strong>
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
              <h2>What this demo is for</h2>
              <p>
                This is a working demonstration of retinal vessel processing, not a clinical
                diagnostic. In plain terms, it traces vessels at the back of the eye, reduces
                their shape to a small set of interpretable numbers, and shows how those
                numbers can be evaluated with honest train/test splits and calibration checks.
              </p>
              <ul class="compact-list">
                <li>Raw medical images stay local under <code>data/raw/</code>.</li>
                <li>ROSE exercises the OCTA image-processing scaffold.</li>
                <li>FIVES exercises feature modeling and calibration discipline.</li>
                <li>
                  The package uses baseline models deliberately: simple, inspectable methods
                  that establish a floor before any specialized model is justified.
                </li>
                <li>
                  True ADRD biomarker analyses remain blocked until project-specific
                  clinical and biomarker data are available.
                </li>
              </ul>
            </div>
            <div class="callout warning">
              <p>
                The public datasets here are stand-ins for demonstrating the processing
                workflow. The report does not claim Alzheimer's diagnosis, ADRD risk
                prediction, or plasma biomarker association from FIVES or ROSE.
              </p>
            </div>
          </div>
        </section>

        <section class="panel">
          <div class="section-heading">
            <p class="eyebrow">Dataset roles</p>
            <h2>What each public dataset is used for</h2>
            <p>
              The two public datasets have different jobs in this demo. ROSE is used to
              exercise OCTA vessel-mask ingestion and feature extraction. FIVES is used
              for the real-image walkthrough and the quantitative modeling example because
              it has a larger public train/test split with disease-vs-normal labels.
            </p>
          </div>
          <div class="method-grid">
            {_dataset_reference_cards()}
          </div>
        </section>

        <section class="panel">
          <div class="section-heading">
            <p class="eyebrow">Processing example</p>
            <h2>From real fundus image to vessel mask to skeleton</h2>
            <p>
              The figure below uses a real local FIVES image and its manual vessel mask to
              show the processing stages. ROSE-derived visual panels remain local-only by
              default because the ROSE archive has more restrictive redistribution terms.
            </p>
          </div>
          {processing_example_figure}
        </section>

        <section class="panel">
          <div class="section-heading">
            <p class="eyebrow">Technical overview</p>
            <h2>What the package does under the hood</h2>
            <p>
              The package is intentionally compact: it uses transparent image-processing
              and statistical baselines so the evidence trail is easy to inspect.
            </p>
          </div>
          <div class="method-grid">
            {_processing_method_blocks()}
          </div>
          <div class="callout info" style="margin-top: 16px;">
            <p>{escape(EARLY_VS_LATE_BIOMARKER_NOTE)}</p>
          </div>
          <div class="method-grid" style="margin-top: 16px;">
            {_feature_timing_cards()}
          </div>
        </section>

        <section class="panel">
          <div class="section-heading">
            <p class="eyebrow">FIVES modeling discipline</p>
            <h2>Current quantitative output</h2>
            <p>
              These metrics summarize a simple disease-vs-normal fundus-label model
              trained and evaluated on the official FIVES image-level split.
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
              <div class="callout warning" style="margin-top: 16px;">
                <p>{escape(FIVES_SPLIT_CAVEAT)}</p>
              </div>
            </div>
          </div>
        </section>

        {render_data_audit_component(code_href_prefix=code_href_prefix)}
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
    for stale_name in (
        "rose_pipeline_panel.png",
        "rose_feature_distributions.png",
        "cross_species_roadmap.png",
        "processing_example_panel.png",
    ):
        stale_path = assets_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    asset_map = {
        project_root / "figures" / "processing_example_panel.png": (
            assets_dir / "processing_example_panel.png"
        ),
        project_root / "figures" / "fives_calibration_demo.png": (
            assets_dir / "fives_calibration_demo.png"
        ),
    }
    for source, destination in asset_map.items():
        if source.exists():
            shutil.copy2(source, destination)


def _generate_fives_processing_example(project_root: Path) -> None:
    out_path = project_root / "figures" / "processing_example_panel.png"
    try:
        manifest = load_fives_manifest(project_root / "data" / "raw" / "fives")
    except (DataNotFoundError, ValueError):
        if out_path.exists():
            out_path.unlink()
        return
    if manifest.empty:
        if out_path.exists():
            out_path.unlink()
        return

    candidates = manifest.copy()
    if "official_split" in candidates.columns:
        test_rows = candidates.loc[candidates["official_split"] == "test"]
        if not test_rows.empty:
            candidates = test_rows
    if "label" in candidates.columns:
        normal_rows = candidates.loc[candidates["label"].astype("string").str.lower() == "normal"]
        if not normal_rows.empty:
            candidates = normal_rows

    row = candidates.iloc[0]
    image = skio.imread(row["image_path"])
    mask = skio.imread(row["mask_path"])
    plot_processing_example_panel(
        image,
        mask,
        out_path,
        title="FIVES real fundus image processing example",
        black_ridges=True,
    )


def build_report(root: Path | str = ".") -> ReportOutputs:
    """Write local report files and the GitHub Pages-ready docs site."""
    project_root = Path(root)
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _generate_fives_processing_example(project_root)

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
