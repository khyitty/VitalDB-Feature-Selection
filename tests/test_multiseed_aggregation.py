"""Focused tests for fixed-seed GRU aggregation helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.aggregate_multiseed_gru import (
    REQUIRED_ARTIFACTS,
    align_prediction_rows,
    discover_complete_seeds,
    patient_bootstrap_mean_ci,
    summarize_numeric,
    validate_patient_seed_alignment,
)


def _write_artifact_names(directory: Path, names: tuple[str, ...]) -> None:
    directory.mkdir(parents=True)
    for name in names:
        (directory / name).touch()


def test_seed_discovery_returns_only_complete_numeric_seed_runs(tmp_path: Path) -> None:
    _write_artifact_names(tmp_path / "seed_7", REQUIRED_ARTIFACTS)
    _write_artifact_names(tmp_path / "seed_21", REQUIRED_ARTIFACTS)
    _write_artifact_names(tmp_path / "smoke_seed_42", REQUIRED_ARTIFACTS)

    complete, incomplete = discover_complete_seeds(tmp_path)

    assert complete == [7, 21]
    assert incomplete == {}


def test_incomplete_run_is_reported_and_not_treated_as_complete(tmp_path: Path) -> None:
    missing_name = "test_predictions.csv"
    present = tuple(name for name in REQUIRED_ARTIFACTS if name != missing_name)
    _write_artifact_names(tmp_path / "seed_84", present)

    complete, incomplete = discover_complete_seeds(tmp_path)

    assert complete == []
    assert incomplete == {84: [missing_name]}


def test_metric_aggregation_uses_sample_standard_deviation() -> None:
    summary = summarize_numeric([1.0, 2.0, 3.0])

    assert summary == {
        "count": 3,
        "mean": 2.0,
        "standard_deviation": 1.0,
        "minimum": 1.0,
        "maximum": 3.0,
    }


def test_patient_by_seed_alignment_requires_every_seed_once() -> None:
    aligned = pd.DataFrame(
        {
            "split": ["test"] * 4,
            "case_id": [97, 97, 154, 154],
            "seed": [7, 21, 7, 21],
            "persistence_mae": [3.0, 3.0, 4.0, 4.0],
            "number_of_windows": [10, 10, 12, 12],
        }
    )
    validate_patient_seed_alignment(aligned, [7, 21])

    with pytest.raises(ValueError, match="case 154 has seeds"):
        validate_patient_seed_alignment(aligned.iloc[:-1], [7, 21])


def test_bootstrap_uses_one_seed_averaged_value_per_patient() -> None:
    patient_differences = np.array([-1.0, 1.0, 3.0])

    result = patient_bootstrap_mean_ci(
        patient_differences, bootstrap_seed=11, replicates=10_000
    )

    assert result["patient_count"] == 3
    assert result["resampling_unit"] == "patient"
    assert result["point_estimate"] == pytest.approx(1.0)


def test_missing_or_inconsistent_prediction_rows_raise_clear_error() -> None:
    reference = pd.DataFrame(
        {
            "sample_index": [0, 1],
            "case_id": [97, 97],
            "target_timestamp": [100, 110],
            "observed_future_bis": [50.0, 51.0],
            "predicted_future_bis": [49.0, 50.0],
            "high_bis_label": [0, 0],
            "low_bis_label": [0, 0],
        }
    )
    missing_row = reference.iloc[:1].copy()
    reordered = reference.iloc[::-1].reset_index(drop=True)

    with pytest.raises(ValueError, match="row count"):
        align_prediction_rows(
            reference, missing_row, split="test", candidate_name="seed 7"
        )
    with pytest.raises(ValueError, match="reordered, or inconsistent"):
        align_prediction_rows(
            reference, reordered, split="test", candidate_name="seed 7"
        )
