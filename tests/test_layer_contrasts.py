import numpy as np
import pandas as pd

from retivasc.layer_contrasts import aggregate_subject_features, compute_layer_contrasts


def _layer_table() -> pd.DataFrame:
    rows = []
    for subject, diagnosis, offset in (("s1", "AD", 0.0), ("s2", "control", 1.0)):
        for layer, value in (("SVC", 1.0), ("DVC", 3.0), ("SVC+DVC", 5.0)):
            rows.append(
                {
                    "subject_id": subject,
                    "image_id": f"{subject}_{layer}",
                    "layer": layer,
                    "diagnosis": diagnosis,
                    "label_source": "manifest",
                    "mask_path": "mask.png",
                    "vessel_area_fraction": value + offset,
                    "branchpoint_density": value * 2 + offset,
                }
            )
    return pd.DataFrame(rows)


def test_aggregate_subject_features_produces_one_row_per_subject():
    aggregated = aggregate_subject_features(_layer_table(), ["vessel_area_fraction"])

    assert len(aggregated) == 2
    assert set(aggregated["subject_id"]) == {"s1", "s2"}
    assert (
        aggregated.loc[aggregated["subject_id"] == "s1", "mean_vessel_area_fraction"].iloc[0] == 3.0
    )
    assert aggregated.loc[aggregated["subject_id"] == "s1", "missing_layers"].iloc[0] == ""


def test_compute_layer_contrasts_expected_difference():
    contrasts = compute_layer_contrasts(_layer_table(), ["vessel_area_fraction"])

    s1 = contrasts.loc[contrasts["subject_id"] == "s1"].iloc[0]

    assert s1["DVC_minus_SVC_vessel_area_fraction"] == 2.0
    assert s1["SVCplusDVC_minus_DVC_vessel_area_fraction"] == 2.0


def test_compute_layer_contrasts_safe_ratios_do_not_divide_by_zero():
    table = _layer_table()
    table.loc[
        (table["subject_id"] == "s1") & (table["layer"] == "SVC"),
        "vessel_area_fraction",
    ] = 0.0

    contrasts = compute_layer_contrasts(table, ["vessel_area_fraction"])
    value = contrasts.loc[
        contrasts["subject_id"] == "s1",
        "DVC_div_SVC_vessel_area_fraction",
    ].iloc[0]

    assert np.isfinite(value)
