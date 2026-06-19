import pandas as pd

from retivasc.artifacts import audit_mask_artifacts


def test_artifact_audit_runs_on_toy_metadata():
    table = pd.DataFrame(
        {
            "subject_id": ["s1", "s2", "s3", "s4"],
            "diagnosis": ["AD", "AD", "control", "control"],
            "layer": ["SVC", "DVC", "SVC", "DVC"],
            "official_split": ["train", "train", "test", "test"],
            "mask_height": [64, 64, 64, 64],
            "mask_width": [64, 64, 64, 64],
            "foreground_fraction": [0.2, 0.25, 0.18, 0.23],
            "connected_component_count": [3, 4, 3, 4],
            "largest_component_fraction": [0.8, 0.75, 0.82, 0.76],
            "small_component_fraction": [0.05, 0.06, 0.04, 0.05],
            "skeleton_length_density": [0.1, 0.12, 0.09, 0.11],
        }
    )

    audit = audit_mask_artifacts(table, n_boot=50, n_perm=50, random_state=0)

    assert not audit.empty
    assert {"foreground_fraction", "official_split"} <= set(audit["variable"])
