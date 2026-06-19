"""Artery and vein caliber summaries for future fundus workflows."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def compute_avr(
    arteriole_widths: Sequence[float] | np.ndarray,
    venule_widths: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Compute CRAE, CRVE, and AVR from preclassified vessel widths.

    This implements the Knudtson iterative equivalent formula given measured
    arteriole and venule widths, typically the six widest of each in an optic
    disc-centered annulus. It does not classify arteries and veins or select the
    disc zone. Those dependencies are deferred.
    """
    crae = _equivalent_width(arteriole_widths, coefficient=0.88)
    crve = _equivalent_width(venule_widths, coefficient=0.95)
    avr = float(crae / crve) if np.isfinite(crve) and crve != 0 else float("nan")
    return {"CRAE": crae, "CRVE": crve, "AVR": avr}


def classify_artery_vein(*args: object, **kwargs: object) -> None:
    """Deferred artery/vein classification scaffold."""
    raise NotImplementedError(
        "Artery/vein classification is deferred. It requires vessel centerlines, "
        "optic-disc context, and color or flow cues from a labeled fundus or OCTA "
        "training substrate."
    )


def measure_crae_crve(*args: object, **kwargs: object) -> None:
    """Deferred end-to-end CRAE/CRVE measurement scaffold."""
    raise NotImplementedError(
        "End-to-end CRAE/CRVE measurement is deferred. First classify arteries "
        "and veins and select calibrated vessel widths in the optic-disc annulus, "
        "then pass those widths to compute_avr."
    )


def resolve_av_crossings(*args: object, **kwargs: object) -> None:
    """Deferred arteriovenous crossing resolution scaffold."""
    raise NotImplementedError(
        "A/V-resolved crossings are deferred. The current crossing count is only "
        "a 4-way skeleton proxy and needs artery/vein labels before it can be "
        "called an arteriovenous crossing detector."
    )


def grade_nicking(*args: object, **kwargs: object) -> None:
    """Deferred venous nicking grading scaffold."""
    raise NotImplementedError(
        "Nicking grading is deferred. It requires resolved artery-over-vein "
        "crossings and a calibrated venular caliber profile through each crossing."
    )


def _equivalent_width(
    widths: Sequence[float] | np.ndarray, *, coefficient: float
) -> float:
    values = [float(value) for value in np.asarray(widths, dtype=float).ravel()]
    if not values:
        return float("nan")
    if not np.isfinite(values).all():
        raise ValueError("Widths must be finite.")
    if any(value <= 0 for value in values):
        raise ValueError("Widths must be positive.")

    values.sort(reverse=True)
    while len(values) > 1:
        largest = values.pop(0)
        smallest = values.pop(-1)
        values.append(coefficient * float(np.hypot(largest, smallest)))
        values.sort(reverse=True)
    return float(values[0])
