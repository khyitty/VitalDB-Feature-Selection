"""Deterministic patient/case-level dataset splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CaseSplits:
    """Disjoint case identifiers assigned to each modeling split."""

    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]

    def as_dict(self) -> dict[str, tuple[int, ...]]:
        """Return split names mapped to their case identifiers."""

        return {"train": self.train, "val": self.val, "test": self.test}

    def assert_disjoint(self) -> None:
        """Raise if any case appears in more than one split."""

        train, val, test = map(set, (self.train, self.val, self.test))
        if train & val or train & test or val & test:
            raise AssertionError("Case-level splits overlap.")


def _allocate_counts(n_cases: int, fractions: tuple[float, float, float]) -> np.ndarray:
    raw = np.asarray(fractions, dtype=float) * n_cases
    counts = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - counts), kind="stable")[: n_cases - counts.sum()]:
        counts[index] += 1

    if n_cases >= 3:
        for empty_index in np.flatnonzero(counts == 0):
            donor_index = int(np.argmax(counts))
            if counts[donor_index] <= 1:
                raise ValueError("Unable to create three non-empty case-level splits.")
            counts[donor_index] -= 1
            counts[empty_index] += 1
    return counts


def split_case_ids(
    case_ids: Iterable[int],
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> CaseSplits:
    """Split unique case IDs deterministically before window construction."""

    unique_ids = np.asarray(sorted({int(case_id) for case_id in case_ids}), dtype=int)
    if len(unique_ids) < 3:
        raise ValueError("At least three unique cases are required for train/val/test splits.")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must sum to 1.")

    shuffled = np.random.default_rng(seed).permutation(unique_ids)
    n_train, n_val, _ = _allocate_counts(len(unique_ids), fractions)
    splits = CaseSplits(
        train=tuple(int(value) for value in shuffled[:n_train]),
        val=tuple(int(value) for value in shuffled[n_train : n_train + n_val]),
        test=tuple(int(value) for value in shuffled[n_train + n_val :]),
    )
    splits.assert_disjoint()
    if set().union(*map(set, splits.as_dict().values())) != set(unique_ids):
        raise AssertionError("Case-level split did not preserve every selected case.")
    return splits


def save_case_splits(splits: CaseSplits, output_dir: Path) -> None:
    """Save one case-ID CSV per split."""

    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name, case_ids in splits.as_dict().items():
        pd.DataFrame({"caseid": case_ids}).to_csv(
            split_dir / f"{split_name}_cases.csv", index=False
        )

