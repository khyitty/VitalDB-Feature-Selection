"""Tests for pooled, patient-level, classification, and threshold metrics."""

import numpy as np
import pytest

from src.metrics import (
    classification_metrics,
    patient_level_evaluation,
    regression_metrics,
    select_validation_thresholds,
    threshold_metrics,
)


def test_known_regression_metrics() -> None:
    metrics = regression_metrics([0.0, 2.0], [1.0, 4.0])

    assert metrics["mae"] == pytest.approx(1.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2.5))


def test_patient_level_metrics_equal_weight_cases_and_mark_undefined_auc() -> None:
    y_true = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 60.0, 70.0])
    y_pred = np.array([51.0] * 9 + [70.0, 70.0])
    case_ids = np.array([1] * 9 + [2] * 2)

    result = patient_level_evaluation(y_true, y_pred, case_ids)

    assert result.summary["mae"]["mean"] == pytest.approx(3.0)
    assert result.summary["number_of_evaluated_cases"] == 2
    assert result.summary["number_of_cases_auroc_defined"] == 1
    undefined = result.case_metrics[result.case_metrics.case_id == 1].iloc[0]
    assert not undefined.high_bis_auroc_defined
    assert np.isnan(undefined.high_bis_auroc)


def test_perfect_classification_and_validation_only_threshold_application() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = classification_metrics(labels, scores)

    assert metrics == {"auprc": 1.0, "auroc": 1.0}
    y_val = np.array([35.0, 45.0, 65.0, 70.0])
    pred_val = np.array([40.0, 50.0, 62.0, 68.0])
    thresholds = select_validation_thresholds(y_val, pred_val)
    test_result = threshold_metrics(
        np.array([0, 1]), np.array([55.0, 66.0]), thresholds["high_bis_score"]
    )
    assert test_result["threshold"] == thresholds["high_bis_score"]


def test_one_class_auc_and_auprc_are_explicitly_undefined() -> None:
    assert classification_metrics(np.zeros(3), np.arange(3.0)) == {
        "auprc": None,
        "auroc": None,
    }

