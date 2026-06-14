"""Leakage-aware dataset splitting helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


def assert_group_split_safe(train: pd.DataFrame, test: pd.DataFrame, group_col: str) -> None:
    """Raise if any group appears in both train and test."""
    if group_col not in train.columns or group_col not in test.columns:
        msg = f"Column {group_col!r} must be present in both train and test dataframes."
        raise ValueError(msg)
    overlap = set(train[group_col].dropna()) & set(test[group_col].dropna())
    if overlap:
        preview = ", ".join(map(str, sorted(overlap)[:5]))
        msg = f"Group leakage detected in column {group_col!r}: {preview}"
        raise ValueError(msg)


def grouped_train_test_split(
    df: pd.DataFrame,
    group_col: str,
    label_col: str | None = None,
    test_size: float = 0.25,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return leakage-safe train/test dataframes."""
    if group_col not in df.columns:
        msg = f"Missing required group column {group_col!r}."
        raise ValueError(msg)
    if not 0 < test_size < 1:
        msg = "test_size must be between 0 and 1."
        raise ValueError(msg)

    work = df.reset_index(drop=True)
    groups = work[group_col]
    unique_groups = groups.dropna().unique()
    if len(unique_groups) < 2:
        msg = "At least two non-null groups are required for a grouped train/test split."
        raise ValueError(msg)

    if label_col is not None and label_col in work.columns:
        group_labels = work[[group_col, label_col]].drop_duplicates()
        conflicts = group_labels.groupby(group_col)[label_col].nunique(dropna=False)
        if (conflicts > 1).any():
            bad = ", ".join(map(str, conflicts[conflicts > 1].index[:5]))
            msg = f"Groups have conflicting labels in {label_col!r}: {bad}"
            raise ValueError(msg)
        group_table = group_labels.dropna(subset=[group_col, label_col]).reset_index(drop=True)
        class_counts = group_table[label_col].value_counts()
        if len(class_counts) >= 2 and int(class_counts.min()) >= 2:
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
            train_group_idx, test_group_idx = next(
                splitter.split(group_table[[group_col]], group_table[label_col])
            )
            train_groups = set(group_table.loc[train_group_idx, group_col])
            test_groups = set(group_table.loc[test_group_idx, group_col])
            train = work[work[group_col].isin(train_groups)].copy()
            test = work[work[group_col].isin(test_groups)].copy()
            assert_group_split_safe(train, test, group_col)
            return train, test

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(work, groups=groups))
    train = work.iloc[train_idx].copy()
    test = work.iloc[test_idx].copy()
    assert_group_split_safe(train, test, group_col)
    return train, test
