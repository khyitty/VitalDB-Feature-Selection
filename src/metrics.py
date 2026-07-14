"""Pooled and patient-balanced metrics for future-BIS predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float | None]:
    """Calculate pooled MAE, RMSE, and R-squared."""

    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    error = y_pred_array - y_true_array
    r_squared = None
    if len(y_true_array) >= 2 and np.ptp(y_true_array) > 0.0:
        r_squared = float(r2_score(y_true_array, y_pred_array))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "r_squared": r_squared,
    }


def bis_region_mae(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    """Calculate MAE in regions defined by observed future BIS."""

    regions = {
        "bis_below_40": y_true < 40.0,
        "bis_40_to_60": (y_true >= 40.0) & (y_true <= 60.0),
        "bis_above_60": y_true > 60.0,
    }
    return {
        name: float(np.mean(np.abs(y_pred[mask] - y_true[mask]))) if mask.any() else None
        for name, mask in regions.items()
    }


def classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    """Calculate AUROC/AUPRC, preserving undefined one-class results as None."""

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if np.unique(labels).size < 2:
        return {"auprc": None, "auroc": None}
    return {
        "auprc": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
    }


def select_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Select a score threshold maximizing F1 on validation predictions only."""

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if np.unique(labels).size < 2:
        raise ValueError("F1 threshold selection requires both positive and negative labels.")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    return float(thresholds[int(np.argmax(f1))])


def threshold_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    """Apply a fixed score threshold and report F1, sensitivity, and specificity."""

    labels = np.asarray(labels, dtype=bool)
    predicted = np.asarray(scores, dtype=float) >= threshold
    true_positive = int(np.sum(predicted & labels))
    true_negative = int(np.sum(~predicted & ~labels))
    false_positive = int(np.sum(predicted & ~labels))
    false_negative = int(np.sum(~predicted & labels))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    f1 = 2.0 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "threshold": float(threshold),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def pooled_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Calculate pooled regression, region, and derived classification metrics."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    high_labels = (y_true > 60.0).astype(int)
    low_labels = (y_true < 40.0).astype(int)
    result: dict[str, Any] = {
        "regression": regression_metrics(y_true, y_pred),
        "bis_region_mae": bis_region_mae(y_true, y_pred),
        "high_bis_classification": classification_metrics(high_labels, y_pred),
        "low_bis_classification": classification_metrics(low_labels, -y_pred),
    }
    if thresholds is not None:
        result["high_bis_threshold_metrics"] = threshold_metrics(
            high_labels, y_pred, thresholds["high_bis_score"]
        )
        result["low_bis_threshold_metrics"] = threshold_metrics(
            low_labels, -y_pred, thresholds["low_bis_score"]
        )
    return result


def select_validation_thresholds(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Select independent high/low score thresholds from validation data."""

    return {
        "high_bis_score": select_f1_threshold((y_true > 60.0).astype(int), y_pred),
        "low_bis_score": select_f1_threshold((y_true < 40.0).astype(int), -y_pred),
    }


def _distribution_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count_defined": 0,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "interquartile_range": None,
        }
    array = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(array, [25, 75])
    return {
        "count_defined": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "standard_deviation": float(array.std(ddof=0)),
        "interquartile_range": float(q3 - q1),
    }


@dataclass(frozen=True)
class PatientEvaluation:
    """Patient-balanced summary and one row of metrics per case."""

    summary: dict[str, Any]
    case_metrics: pd.DataFrame


def patient_level_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    case_ids: np.ndarray,
) -> PatientEvaluation:
    """Calculate metrics independently per case, then summarize with equal weight."""

    frame = pd.DataFrame(
        {"case_id": case_ids.astype(int), "y_true": y_true, "y_pred": y_pred}
    )
    rows: list[dict[str, Any]] = []
    for case_id, case_frame in frame.groupby("case_id", sort=True):
        observed = case_frame["y_true"].to_numpy(dtype=float)
        predicted = case_frame["y_pred"].to_numpy(dtype=float)
        regression = regression_metrics(observed, predicted)
        high_labels = (observed > 60.0).astype(int)
        high_metrics = classification_metrics(high_labels, predicted)
        rows.append(
            {
                "case_id": int(case_id),
                "number_of_windows": len(case_frame),
                "mae": regression["mae"],
                "rmse": regression["rmse"],
                "high_bis_auprc": high_metrics["auprc"],
                "high_bis_auroc": high_metrics["auroc"],
                "high_bis_auprc_defined": high_metrics["auprc"] is not None,
                "high_bis_auroc_defined": high_metrics["auroc"] is not None,
            }
        )
    case_metrics = pd.DataFrame(rows)
    summary = {
        "number_of_evaluated_cases": len(case_metrics),
        "number_of_cases_auprc_defined": int(case_metrics["high_bis_auprc_defined"].sum()),
        "number_of_cases_auroc_defined": int(case_metrics["high_bis_auroc_defined"].sum()),
        "mae": _distribution_summary(case_metrics["mae"].dropna().tolist()),
        "rmse": _distribution_summary(case_metrics["rmse"].dropna().tolist()),
        "high_bis_auprc": _distribution_summary(
            case_metrics["high_bis_auprc"].dropna().tolist()
        ),
        "high_bis_auroc": _distribution_summary(
            case_metrics["high_bis_auroc"].dropna().tolist()
        ),
    }
    return PatientEvaluation(summary=summary, case_metrics=case_metrics)

