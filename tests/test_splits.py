"""Tests for deterministic patient-level splitting."""

from src.splits import split_case_ids


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

