"""Paper-aligned feature metadata and caveats."""

from __future__ import annotations

FEATURE_METADATA: dict[str, dict[str, object]] = {
    "tortuous_segment_fraction": {
        "timing": "early",
        "significance": (
            "6mo onset (TT-only occurrence); statistical significance at 12mo CT/TT female"
        ),
        "source": "Reagan et al. 2025, tortuosity occurrence percent (Fig 3D-F)",
        "datasets": ["FIVES", "ROSE"],
        "note": (
            "Faithful to the paper's per-vessel occurrence percent. Magnitude is "
            "field-of-view and pixel-size dependent."
        ),
    },
    "tortuous_length_fraction": {
        "timing": "early",
        "significance": "6mo onset; 12mo CT/TT female significance",
        "source": "Reagan et al. 2025, tortuosity occurrence percent (Fig 3D-F)",
        "datasets": ["FIVES", "ROSE"],
        "note": "Length-weighted burden. Demo proxy, field-of-view and pixel-size dependent.",
    },
    "mean_segment_tortuosity": {
        "timing": "early",
        "significance": "6mo onset; 12mo CT/TT female significance",
        "source": "Reagan et al. 2025, tortuosity occurrence percent (Fig 3D-F)",
        "datasets": ["FIVES", "ROSE"],
        "note": "Secondary descriptor. The burden fraction is the primary paper-aligned metric.",
    },
    "median_segment_tortuosity": {
        "timing": "early",
        "significance": "6mo onset; 12mo CT/TT female significance",
        "source": "Reagan et al. 2025, tortuosity occurrence percent (Fig 3D-F)",
        "datasets": ["FIVES", "ROSE"],
        "note": "Secondary descriptor. The burden fraction is the primary paper-aligned metric.",
    },
    "p90_segment_tortuosity": {
        "timing": "early",
        "significance": "6mo onset; 12mo CT/TT female significance",
        "source": "Reagan et al. 2025, tortuosity occurrence percent (Fig 3D-F)",
        "datasets": ["FIVES", "ROSE"],
        "note": "Upper-tail tortuosity descriptor, not the primary burden metric.",
    },
    "p95_segment_tortuosity": {
        "timing": "early",
        "significance": "6mo onset; 12mo CT/TT female significance",
        "source": "Reagan et al. 2025, tortuosity occurrence percent (Fig 3D-F)",
        "datasets": ["FIVES", "ROSE"],
        "note": "Upper-tail tortuosity descriptor, not the primary burden metric.",
    },
    "caliber_cv": {
        "timing": "early",
        "significance": (
            "6mo, genotype-dose (CC/CT/TT); arteriole narrowing (****) with "
            "venule widening (*); not stated female-specific"
        ),
        "source": "Reagan et al. 2025, arteriole/venule diameter near optic nerve head (Fig 3G-I)",
        "datasets": ["FIVES"],
        "note": (
            "Dispersion proxy only: detects that calibers diverged, cannot attribute "
            "arteriole-narrowing vs venule-widening without artery/vein labels. "
            "Faithful per-type version is CRAE/CRVE/AVR (deferred)."
        ),
    },
    "mean_vessel_caliber_px": {
        "timing": "early",
        "significance": "6mo caliber finding, but this scalar is signed-asymmetry blind",
        "source": "Reagan et al. 2025 (Fig 3G-I)",
        "datasets": ["FIVES"],
        "note": (
            "Secondary descriptor only. A single mean averages arteriole narrowing "
            "against venule widening toward null and is capillary-dominated. Do not "
            "use as the caliber biomarker, use caliber_cv."
        ),
    },
    "caliber_p90_minus_p10_px": {
        "timing": "early",
        "significance": "6mo caliber asymmetry proxy",
        "source": "Reagan et al. 2025 (Fig 3G-I)",
        "datasets": ["FIVES"],
        "note": "Width tail-spread proxy. Direction requires artery/vein labels.",
    },
    "large_vessel_caliber_cv": {
        "timing": "early",
        "significance": "6mo caliber asymmetry proxy",
        "source": "Reagan et al. 2025 (Fig 3G-I)",
        "datasets": ["FIVES"],
        "note": "Large-vessel dispersion proxy. Direction requires artery/vein labels.",
    },
    "candidate_crossing_density": {
        "timing": "early",
        "significance": "6mo for females (CT/TT); peaks at 12mo",
        "source": "Reagan et al. 2025, AVC percent (Fig 3A-C)",
        "datasets": ["FIVES"],
        "note": (
            "Proxy: counts degree-4 skeleton nodes, not validated arteriole-over-venule "
            "crossings. A 2D projection of two non-touching vessels also makes a 4-way node."
        ),
    },
    "candidate_crossing_count": {
        "timing": "early",
        "significance": "6mo for females (CT/TT); peaks at 12mo",
        "source": "Reagan et al. 2025, AVC percent (Fig 3A-C)",
        "datasets": ["FIVES"],
        "note": "Raw count supporting candidate_crossing_density. Exploratory only.",
    },
    "vessel_density": {
        "timing": "late",
        "significance": (
            "12mo female TT only; authors call density 'may not be ideal' as an early biomarker"
        ),
        "source": "Reagan et al. 2025 (Fig 2A-B)",
        "datasets": ["FIVES", "ROSE"],
        "note": "Late and weak. Never present as an early or sex-neutral biomarker.",
    },
    "vessel_area_fraction": {
        "timing": "late",
        "significance": (
            "12mo female TT only; authors call density 'may not be ideal' as an early biomarker"
        ),
        "source": "Reagan et al. 2025 (Fig 2A-B)",
        "datasets": ["ROSE"],
        "note": "ROSE name for vessel_density. Late, weak, and exploratory.",
    },
    "skeleton_length_density": {
        "timing": "late",
        "significance": "covaries with density, 12mo",
        "source": "Reagan et al. 2025 (Fiji vascular length density)",
        "datasets": ["FIVES", "ROSE"],
        "note": "Late. Covaries with vessel_density.",
    },
    "major_branch_count": {
        "timing": "late",
        "significance": (
            "12mo female TT; large-vessel network simplification near the optic nerve head"
        ),
        "source": "Reagan et al. 2025 (Fig 2C)",
        "datasets": ["FIVES"],
        "note": "Paper finding is large-vessel major branches near the disc.",
    },
    "branchpoint_density": {
        "timing": "late",
        "significance": "capillary-scale fragmentation proxy, not the paper's major-branch count",
        "source": "network simplification analogue, 12mo",
        "datasets": ["FIVES", "ROSE"],
        "note": "On ROSE this is capillary-scale fragmentation, not disc major-branch count.",
    },
    "connected_component_count": {
        "timing": "late",
        "significance": "fragmentation proxy",
        "source": "network simplification analogue, 12mo",
        "datasets": ["FIVES", "ROSE"],
        "note": "Fragmentation proxy.",
    },
    "fractal_dimension_boxcount": {
        "timing": "context",
        "significance": "not measured in the mouse paper",
        "source": "human oculomics plus ROSE OCTA AD/control context",
        "datasets": ["FIVES", "ROSE"],
        "note": "External complexity context only. Covaries with density.",
    },
    "dropout_heterogeneity": {
        "timing": "interpretive",
        "significance": "not measured in the mouse paper",
        "source": "hypoperfusion hypothesis in the Discussion, not a measured phenotype",
        "datasets": ["ROSE"],
        "note": "Density-complement or spatial heterogeneity. Do not call it hypoperfusion.",
    },
    "hole_fraction_or_dropout_proxy": {
        "timing": "interpretive",
        "significance": "not measured in the mouse paper",
        "source": "hypoperfusion hypothesis in the Discussion, not a measured phenotype",
        "datasets": ["ROSE"],
        "note": "Legacy ROSE dropout-style proxy. Interpretive and density-related.",
    },
    "endpoint_density": {
        "timing": "late",
        "significance": "fragmentation proxy",
        "source": "network simplification analogue, 12mo",
        "datasets": ["ROSE"],
        "note": "Capillary-scale fragmentation descriptor.",
    },
    "largest_component_fraction": {
        "timing": "late",
        "significance": "fragmentation proxy",
        "source": "network simplification analogue, 12mo",
        "datasets": ["ROSE"],
        "note": "Connectedness descriptor for capillary masks.",
    },
    "small_component_fraction": {
        "timing": "late",
        "significance": "fragmentation proxy",
        "source": "network simplification analogue, 12mo",
        "datasets": ["ROSE"],
        "note": "Small-component fragmentation descriptor.",
    },
    "mean_segment_length_px": {
        "timing": "late",
        "significance": "fragmentation proxy",
        "source": "network simplification analogue, 12mo",
        "datasets": ["ROSE"],
        "note": "Segment-length descriptor for capillary masks.",
    },
    "median_segment_length_px": {
        "timing": "late",
        "significance": "fragmentation proxy",
        "source": "network simplification analogue, 12mo",
        "datasets": ["ROSE"],
        "note": "Segment-length descriptor for capillary masks.",
    },
    "mean_tortuosity_arc_chord": {
        "timing": "early",
        "significance": "6mo onset; 12mo CT/TT female significance",
        "source": "Reagan et al. 2025, tortuosity occurrence percent (Fig 3D-F)",
        "datasets": ["ROSE"],
        "note": "Legacy ROSE mean. Burden fraction is the primary paper-aligned metric.",
    },
    "high_tortuosity_fraction": {
        "timing": "early",
        "significance": "6mo onset; 12mo CT/TT female significance",
        "source": "Reagan et al. 2025, tortuosity occurrence percent (Fig 3D-F)",
        "datasets": ["ROSE"],
        "note": "Legacy ROSE name for tortuous_segment_fraction.",
    },
    "caliber_proxy_mean_px": {
        "timing": "context",
        "significance": "ROSE cannot test optic-disc caliber asymmetry",
        "source": "ROSE capillary OCTA width context",
        "datasets": ["ROSE"],
        "note": "Macula capillary width context only, not Reagan caliber asymmetry.",
    },
    "caliber_proxy_median_px": {
        "timing": "context",
        "significance": "ROSE cannot test optic-disc caliber asymmetry",
        "source": "ROSE capillary OCTA width context",
        "datasets": ["ROSE"],
        "note": "Macula capillary width context only, not Reagan caliber asymmetry.",
    },
    "caliber_proxy_std_px": {
        "timing": "context",
        "significance": "ROSE cannot test optic-disc caliber asymmetry",
        "source": "ROSE capillary OCTA width context",
        "datasets": ["ROSE"],
        "note": "Macula capillary width context only, not Reagan caliber asymmetry.",
    },
    "orientation_entropy": {
        "timing": "context",
        "significance": "not measured in the mouse paper",
        "source": "ROSE capillary OCTA orientation context",
        "datasets": ["ROSE"],
        "note": "Orientation heterogeneity context only.",
    },
}


def metadata_for_feature(feature: str) -> dict[str, object]:
    """Return metadata for a feature key, or context metadata for derived aliases."""
    if feature in FEATURE_METADATA:
        return FEATURE_METADATA[feature]
    for prefix in (
        "DVC_minus_SVC_",
        "SVCplusDVC_minus_SVC_",
        "SVCplusDVC_minus_DVC_",
        "DVC_div_SVC_",
        "SVCplusDVC_div_SVC_",
        "SVCplusDVC_div_DVC_",
        "DVC_logratio_SVC_",
        "SVCplusDVC_logratio_SVC_",
        "SVCplusDVC_logratio_DVC_",
    ):
        if feature.startswith(prefix):
            return metadata_for_feature(feature.removeprefix(prefix))
    for prefix in ("mean_", "min_", "max_", "range_", "std_", "SVC_", "DVC_", "SVC+DVC_"):
        if feature.startswith(prefix):
            return metadata_for_feature(feature.removeprefix(prefix))
    return {
        "timing": "context",
        "significance": "derived or exploratory feature",
        "source": "retivasc derived feature",
        "datasets": ["FIVES", "ROSE"],
        "note": "Derived feature without a direct paper claim.",
    }
