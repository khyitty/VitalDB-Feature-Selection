"""Focused tests for case-balanced and masked attention audit helpers."""

import numpy as np
import pandas as pd
import pytest

from src.attention_audit import (
    SplitAttentionData,
    align_prediction_rows,
    attention_normalization_audit,
    case_balanced_feature_summary,
    normalized_feature_entropy,
)


def _split_data(
    feature_attention: np.ndarray,
    mask: np.ndarray,
    case_ids: np.ndarray,
) -> SplitAttentionData:
    temporal = np.full(feature_attention.shape[:2], 1 / feature_attention.shape[1])
    combined = temporal[:, :, None] * feature_attention
    predictions = pd.DataFrame(
        {
            "sample_index": np.arange(len(case_ids)),
            "case_id": case_ids,
            "target_timestamp": np.arange(len(case_ids)) * 10,
            "observed_future_bis": np.full(len(case_ids), 50.0),
            "predicted_future_bis": np.full(len(case_ids), 50.0),
            "high_bis_label": np.zeros(len(case_ids), dtype=int),
            "low_bis_label": np.zeros(len(case_ids), dtype=int),
        }
    )
    return SplitAttentionData(
        split="val",
        predictions=predictions,
        sample_indices=np.arange(len(case_ids)),
        case_ids=case_ids,
        observation_mask=mask,
        feature_attention=feature_attention,
        temporal_attention=temporal,
        combined_attention=combined,
    )


def test_case_balanced_summary_does_not_let_long_case_dominate() -> None:
    case_ids = np.array([1] * 100 + [2] * 2)
    feature = np.zeros((102, 2, 2), dtype=float)
    feature[:100, :, 0] = 1.0
    feature[100:, :, 1] = 1.0
    data = _split_data(feature, np.ones_like(feature, dtype=bool), case_ids)

    summary = case_balanced_feature_summary(data, ["long_case_feature", "short_case_feature"])

    assert summary.set_index("feature").loc[
        "long_case_feature", "mean_feature_attention"
    ] == pytest.approx(0.5)
    assert summary.set_index("feature").loc[
        "short_case_feature", "mean_feature_attention"
    ] == pytest.approx(0.5)


def test_normalized_entropy_accounts_for_varying_observed_counts() -> None:
    feature = np.array(
        [
            [
                [0.5, 0.5, 0.0, 0.0],
                [0.25, 0.25, 0.25, 0.25],
                [1.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    mask = feature > 0

    entropy = normalized_feature_entropy(feature, mask)

    assert entropy[0, 0] == pytest.approx(1.0)
    assert entropy[0, 1] == pytest.approx(1.0)
    assert entropy[0, 2] == pytest.approx(0.0)


def test_masked_attention_audit_reports_exact_zero_and_valid_sums() -> None:
    feature = np.array([[[0.7, 0.3, 0.0], [0.2, 0.0, 0.8]]])
    mask = np.array([[[True, True, False], [True, False, True]]])
    data = _split_data(feature, mask, np.array([9]))

    audit = attention_normalization_audit(data)

    assert audit["feature_attention"]["maximum_unobserved_feature_weight"] == 0.0
    assert audit["feature_attention"]["unobserved_feature_nonzero_count"] == 0
    assert audit["feature_attention"]["time_steps_violating_tolerance_1e_5"] == 0
    assert audit["temporal_attention"]["rows_violating_tolerance_1e_5"] == 0
    assert audit["combined_attention"]["maximum_definition_error"] == 0.0


def test_prediction_comparison_rejects_misaligned_rows() -> None:
    frame = pd.DataFrame(
        {
            "sample_index": [0, 1],
            "case_id": [3, 3],
            "target_timestamp": [80, 90],
            "observed_future_bis": [50.0, 51.0],
            "predicted_future_bis": [49.0, 50.0],
            "high_bis_label": [0, 0],
            "low_bis_label": [0, 0],
        }
    )
    misaligned = frame.iloc[::-1].reset_index(drop=True)

    with pytest.raises(ValueError, match="misaligned"):
        align_prediction_rows(
            frame, misaligned, split="test", candidate_name="candidate"
        )
