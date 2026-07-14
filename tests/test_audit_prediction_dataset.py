"""Focused tests for full-dataset audit calculations."""

import pandas as pd

from scripts.audit_prediction_dataset import _numeric_summary, _window_candidate_counts


def test_numeric_summary_reports_median_and_iqr() -> None:
    summary = _numeric_summary([1.0, 2.0, 3.0, 4.0])

    assert summary["median"] == 2.5
    assert summary["q1"] == 1.75
    assert summary["q3"] == 3.25
    assert summary["interquartile_range"] == 1.5


def test_window_candidate_audit_matches_exact_window_convention() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": list(range(0, 101, 10)),
            "bis": [50.0] * 11,
        }
    )

    counts = _window_candidate_counts(
        frame, history_steps=6, interval_seconds=10, horizon_seconds=30
    )

    assert counts == {
        "candidate_windows": 6,
        "included_windows": 3,
        "excluded_history_gap": 0,
        "excluded_unavailable_future_bis": 3,
    }

