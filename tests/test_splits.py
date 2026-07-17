"""Tests for deterministic patient-level splitting."""

from pathlib import Path

import pandas as pd

from src.splits import load_case_splits, split_case_ids


def test_case_splits_are_disjoint_and_deterministic() -> None:
    case_ids = list(range(1, 21))

    first = split_case_ids(case_ids, seed=42)
    second = split_case_ids(reversed(case_ids), seed=42)

    assert first == second
    assert not (set(first.train) & set(first.val))
    assert not (set(first.train) & set(first.test))
    assert not (set(first.val) & set(first.test))
    assert set(first.train) | set(first.val) | set(first.test) == set(case_ids)


def test_smallest_supported_split_keeps_all_three_sets_nonempty() -> None:
    splits = split_case_ids([10, 20, 30], seed=42)

    assert len(splits.train) == len(splits.val) == len(splits.test) == 1


def test_reference_split_is_reused_exactly_without_reassignment(tmp_path: Path) -> None:
    expected = {"train": [7, 1], "val": [3], "test": [9]}
    for name, case_ids in expected.items():
        pd.DataFrame({"caseid": case_ids}).to_csv(
            tmp_path / f"{name}_cases.csv", index=False
        )

    splits = load_case_splits(tmp_path, [1, 3, 7, 9])

    assert splits.train == (7, 1)
    assert splits.val == (3,)
    assert splits.test == (9,)
