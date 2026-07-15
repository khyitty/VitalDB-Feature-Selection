"""Focused tests for held-out key and float32 target alignment."""

from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import pytest

from src.frozen_predictive_test_evaluation import verify_prediction_alignment


def _expected() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_index": np.arange(3, dtype=np.int64),
            "case_id": np.asarray([17, 17, 21], dtype=np.int64),
            "target_timestamp": np.asarray([190, 260, 310], dtype=np.int64),
            "observed_future_bis": np.asarray(
                [97.3, 65.005966, 42.55], dtype=np.float32
            ).astype(float),
        }
    )


def _csv_round_trip(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.read_csv(StringIO(frame.to_csv(index=False)))


def test_float32_csv_round_trip_passes_despite_more_than_1e6_absolute_noise() -> None:
    expected = _expected()
    observed = _csv_round_trip(
        expected.assign(
            observed_future_bis=expected["observed_future_bis"].to_numpy(
                dtype=np.float32
            )
        )
    )
    result = verify_prediction_alignment(expected, observed, context="round trip")
    assert result["max_absolute_target_difference"] > 1e-6
    assert result["target_mismatch_count"] == 0


def test_target_difference_above_float32_round_trip_bound_fails() -> None:
    expected = _expected()
    observed = expected.copy()
    observed.loc[0, "observed_future_bis"] += 1e-3
    with pytest.raises(ValueError, match='"mismatch_field": "observed_future_bis"'):
        verify_prediction_alignment(expected, observed, context="changed target")


@pytest.mark.parametrize("field", ["sample_index", "case_id", "target_timestamp"])
def test_integer_alignment_field_change_fails(field: str) -> None:
    expected = _expected()
    observed = expected.copy()
    observed.loc[1, field] += 1
    with pytest.raises(ValueError, match=rf'"mismatch_field": "{field}"'):
        verify_prediction_alignment(expected, observed, context=f"changed {field}")


def test_row_order_change_fails_without_sorting() -> None:
    expected = _expected()
    observed = expected.iloc[[1, 0, 2]].reset_index(drop=True)
    with pytest.raises(ValueError, match='"mismatch_field": "sample_index"'):
        verify_prediction_alignment(expected, observed, context="changed order")


def test_duplicate_key_fails() -> None:
    expected = _expected()
    observed = expected.copy()
    observed.iloc[2] = observed.iloc[0]
    with pytest.raises(ValueError, match='"mismatch_field": "duplicate_key"'):
        verify_prediction_alignment(expected, observed, context="duplicate")


@pytest.mark.parametrize("mode", ["missing", "extra"])
def test_missing_or_extra_row_fails(mode: str) -> None:
    expected = _expected()
    observed = (
        expected.iloc[:-1].copy()
        if mode == "missing"
        else pd.concat([expected, expected.iloc[[0]]], ignore_index=True)
    )
    with pytest.raises(ValueError, match='"mismatch_field": "row_count"'):
        verify_prediction_alignment(expected, observed, context=mode)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_target_fails(value: float) -> None:
    expected = _expected()
    observed = expected.copy()
    observed.loc[0, "observed_future_bis"] = value
    with pytest.raises(
        ValueError, match='"mismatch_field": "observed_future_bis_non_finite"'
    ):
        verify_prediction_alignment(expected, observed, context="non-finite")
