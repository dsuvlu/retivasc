import math

import pytest

from retivasc.vascular_avr import (
    classify_artery_vein,
    compute_avr,
    grade_nicking,
    measure_crae_crve,
    resolve_av_crossings,
)


def test_compute_avr_uses_knudtson_equivalent_widths():
    summary = compute_avr([3.0, 4.0], [5.0, 12.0])

    assert summary["CRAE"] == pytest.approx(4.4)
    assert summary["CRVE"] == pytest.approx(12.35)
    assert summary["AVR"] == pytest.approx(4.4 / 12.35)


def test_compute_avr_pairs_widest_with_narrowest_iteratively():
    summary = compute_avr([50.0, 40.0, 30.0], [70.0, 60.0, 50.0])
    first_artery_pair = 0.88 * math.hypot(50.0, 30.0)
    expected_crae = 0.88 * math.hypot(first_artery_pair, 40.0)
    first_vein_pair = 0.95 * math.hypot(70.0, 50.0)
    expected_crve = 0.95 * math.hypot(first_vein_pair, 60.0)

    assert summary["CRAE"] == pytest.approx(expected_crae)
    assert summary["CRVE"] == pytest.approx(expected_crve)
    assert summary["AVR"] == pytest.approx(expected_crae / expected_crve)


def test_compute_avr_rejects_nonpositive_widths():
    with pytest.raises(ValueError, match="positive"):
        compute_avr([3.0, 0.0], [5.0, 12.0])


@pytest.mark.parametrize(
    "func",
    [classify_artery_vein, measure_crae_crve, resolve_av_crossings, grade_nicking],
)
def test_deferred_av_workflows_raise_informative_errors(func):
    with pytest.raises(NotImplementedError, match="deferred"):
        func()
