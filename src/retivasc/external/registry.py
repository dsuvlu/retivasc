"""Comparator metadata for reports and notebooks."""

from __future__ import annotations

MODEL_REGISTRY = {
    "frangi": {
        "family": "native",
        "description": "Transparent Frangi vesselness floor with morphology cleanup.",
        "requires_external_tool": False,
    },
    "diffusion_threshold": {
        "family": "native",
        "description": "Anisotropic-diffusion smoothing followed by vesselness thresholding.",
        "requires_external_tool": False,
    },
    "random_walker": {
        "family": "native",
        "description": "Seeded label diffusion over an image-guided random-walker graph.",
        "requires_external_tool": False,
    },
    "geodesic_voting": {
        "family": "native",
        "description": "Minimal-path voting through high-vesselness pixels.",
        "requires_external_tool": False,
    },
    "octa_net": {
        "family": "external_adapter",
        "description": "ROSE-native OCTA deep segmentation comparator; not bundled.",
        "requires_external_tool": True,
    },
    "u_net": {
        "family": "external_prediction",
        "description": "Externally trained U-Net-family segmentation predictions; not bundled.",
        "requires_external_tool": True,
    },
    "unet_lite": {
        "family": "in_repo_deep",
        "description": "Small locally trained U-Net comparator for demo-scale segmentation checks.",
        "requires_external_tool": False,
    },
    "octa_net_lite": {
        "family": "in_repo_deep",
        "description": (
            "Small OCTA-oriented U-Net comparator using image plus vesselness channels; "
            "not the official OCTA-Net implementation."
        ),
        "requires_external_tool": False,
    },
    "nnunet_lite": {
        "family": "in_repo_deep",
        "description": (
            "Small nnU-Net-style local comparator with instance-normalized U-Net blocks; "
            "official nnU-Net remains an optional external framework."
        ),
        "requires_external_tool": False,
    },
    "nnunet": {
        "family": "external_adapter",
        "description": "Generic self-configuring biomedical segmentation comparator; not bundled.",
        "requires_external_tool": True,
    },
}
