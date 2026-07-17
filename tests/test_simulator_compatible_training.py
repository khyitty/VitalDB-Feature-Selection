"""Scientific guards for the new validation-only prediction wrappers."""

from __future__ import annotations

import json

import pytest

from src.prediction_feature_profiles import (
    SIMULATOR_COMPATIBLE_PROFILE,
    get_prediction_feature_profile,
)
from src.simulator_compatible_training import validate_main_prediction_run


def _write_metadata(tmp_path, **overrides) -> None:
    profile = get_prediction_feature_profile(SIMULATOR_COMPATIBLE_PROFILE)
    payload = {
        **profile.as_metadata(),
        "preprocessing_fit_split": "train_only",
        "feature_selection_split_accessed": False,
        "test_results_inspected": False,
        "test_target_summary_sealed": True,
        **overrides,
    }
    (tmp_path / "dataset_metadata.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    (tmp_path / "pkpd_reconstruction_audit.json").write_text(
        json.dumps(
            {
                "causal": True,
                "target_concentration_used": False,
                "recorded_cp_ce_used_as_model_features": False,
            }
        ),
        encoding="utf-8",
    )


def test_main_prediction_run_accepts_only_locked_validation_dataset(tmp_path) -> None:
    _write_metadata(tmp_path)
    metadata = validate_main_prediction_run(tmp_path, validation_only=True)
    assert metadata["feature_profile"] == SIMULATOR_COMPATIBLE_PROFILE


def test_main_prediction_run_rejects_test_evaluation(tmp_path) -> None:
    _write_metadata(tmp_path)
    with pytest.raises(ValueError, match="validation-only"):
        validate_main_prediction_run(tmp_path, validation_only=False)


def test_main_prediction_run_rejects_non_train_preprocessing(tmp_path) -> None:
    _write_metadata(tmp_path, preprocessing_fit_split="all")
    with pytest.raises(ValueError, match="scientific guards"):
        validate_main_prediction_run(tmp_path, validation_only=True)
