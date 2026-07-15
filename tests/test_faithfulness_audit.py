"""Focused tests for validation-only perturbation and faithfulness helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.faithfulness_audit import (
    ablate_feature_arrays,
    ablate_named_group,
    align_attention_contribution,
    contribution_stability_summary,
    patient_bootstrap_interval,
    patient_equal_mae,
    within_patient_circular_permutation,
)
from src.redundancy_audit import FEATURE_GROUPS, REDUCED_FEATURES


def _arrays() -> tuple[np.ndarray, np.ndarray]:
    values = np.arange(4 * 2 * 4, dtype=np.float32).reshape(4, 2, 4)
    masks = (values % 3 != 0)
    return values, masks


def test_individual_ablation_changes_only_selected_value_and_mask_columns() -> None:
    values, masks = _arrays()
    changed_values, changed_masks = ablate_feature_arrays(values, masks, [2])

    assert np.count_nonzero(changed_values[:, :, 2]) == 0
    assert not changed_masks[:, :, 2].any()
    np.testing.assert_array_equal(changed_values[:, :, [0, 1, 3]], values[:, :, [0, 1, 3]])
    np.testing.assert_array_equal(changed_masks[:, :, [0, 1, 3]], masks[:, :, [0, 1, 3]])
    assert not np.shares_memory(changed_values, values)


def test_group_ablation_uses_exact_declared_membership() -> None:
    values = np.ones((2, 6, len(REDUCED_FEATURES)), dtype=np.float32)
    masks = np.ones_like(values, dtype=bool)
    members = FEATURE_GROUPS["respiratory"]
    changed_values, changed_masks = ablate_named_group(
        values, masks, REDUCED_FEATURES, members
    )
    selected = [REDUCED_FEATURES.index(name) for name in members]
    retained = [index for index in range(len(REDUCED_FEATURES)) if index not in selected]

    assert not changed_values[:, :, selected].any()
    assert not changed_masks[:, :, selected].any()
    assert changed_values[:, :, retained].all()
    assert changed_masks[:, :, retained].all()


def test_within_patient_permutation_preserves_trajectories_masks_and_boundaries() -> None:
    case_ids = np.array([10, 10, 10, 20, 20, 20])
    values = np.arange(6 * 2 * 3, dtype=np.float32).reshape(6, 2, 3)
    masks = values % 2 == 0
    changed_values, changed_masks, shifts = within_patient_circular_permutation(
        values, masks, case_ids, [1], repetition=0, permutation_seed=7
    )

    assert set(shifts) == {10, 20}
    assert all(shift > 0 for shift in shifts.values())
    for case_id in (10, 20):
        rows = np.flatnonzero(case_ids == case_id)
        original_pairs = sorted(
            (tuple(values[row, :, 1]), tuple(masks[row, :, 1])) for row in rows
        )
        changed_pairs = sorted(
            (tuple(changed_values[row, :, 1]), tuple(changed_masks[row, :, 1]))
            for row in rows
        )
        assert changed_pairs == original_pairs
        assert not np.array_equal(changed_values[rows, :, 1], values[rows, :, 1])
    np.testing.assert_array_equal(changed_values[:, :, [0, 2]], values[:, :, [0, 2]])


def test_patient_equal_mae_is_case_balanced() -> None:
    y_true = np.zeros(4)
    y_pred = np.array([1.0, 1.0, 1.0, 9.0])
    case_ids = np.array([1, 1, 1, 2])

    assert patient_equal_mae(y_true, y_pred, case_ids) == pytest.approx(5.0)


def test_patient_bootstrap_is_deterministic_and_uses_patient_vector() -> None:
    differences = np.array([-1.0, 0.0, 2.0])
    first = patient_bootstrap_interval(differences, replicates=1000, seed=13)
    second = patient_bootstrap_interval(differences, replicates=1000, seed=13)

    assert first == second
    assert first["patient_count"] == 3
    assert first["mean_delta_patient_mae"] == pytest.approx(1.0 / 3.0)


def test_attention_contribution_alignment_rejects_missing_rows() -> None:
    attention = pd.DataFrame(
        {"seed": [7, 7], "feature": ["bis", "hr"], "attention": [0.9, 0.1]}
    )
    contribution = pd.DataFrame(
        {"seed": [7, 7], "feature": ["hr", "bis"], "delta_patient_mae": [0.2, 1.0]}
    )
    aligned = align_attention_contribution(attention, contribution, "feature")
    assert aligned["feature"].tolist() == ["bis", "hr"]

    with pytest.raises(ValueError, match="not exactly aligned"):
        align_attention_contribution(attention, contribution.iloc[:1], "feature")


def test_descriptive_stability_requires_positive_mean_not_only_four_positive_seeds() -> None:
    frame = pd.DataFrame(
        {
            "model": ["attention"] * 10,
            "seed": [7, 21, 42, 84, 123] * 2,
            "feature": ["a"] * 5 + ["b"] * 5,
            "delta_patient_mae": [0.1, 0.1, 0.1, 0.1, -1.0] + [0.2] * 5,
        }
    )
    summary = contribution_stability_summary(
        frame, "individual_feature_ablation", "feature"
    ).set_index("item")

    assert bool(summary.loc["a", "descriptively_stable"]) is False
    assert bool(summary.loc["b", "descriptively_stable"]) is True
