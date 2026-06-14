import pandas as pd
import pytest

from retivasc.splits import assert_group_split_safe, grouped_train_test_split


def test_assert_group_split_safe_passes_with_disjoint_groups():
    train = pd.DataFrame({"subject_id": ["a", "b"]})
    test = pd.DataFrame({"subject_id": ["c"]})

    assert_group_split_safe(train, test, "subject_id")


def test_assert_group_split_safe_raises_on_overlap():
    train = pd.DataFrame({"subject_id": ["a", "b"]})
    test = pd.DataFrame({"subject_id": ["b", "c"]})

    with pytest.raises(ValueError, match="Group leakage"):
        assert_group_split_safe(train, test, "subject_id")


def test_grouped_train_test_split_creates_no_overlap():
    df = pd.DataFrame(
        {
            "subject_id": [f"sub-{idx}" for idx in range(12) for _ in range(2)],
            "label": ["AD" if idx % 2 else "control" for idx in range(12) for _ in range(2)],
            "value": range(24),
        }
    )

    train, test = grouped_train_test_split(
        df, group_col="subject_id", label_col="label", random_state=1
    )

    assert len(train) > 0
    assert len(test) > 0
    assert_group_split_safe(train, test, "subject_id")
