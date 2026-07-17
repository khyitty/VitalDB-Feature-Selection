"""Contracts for the final simulator-compatible prediction feature universe."""

from __future__ import annotations

import json

import pytest

from src.prediction_feature_profiles import (
    LEGACY_DYNAMIC_FEATURES,
    SIMULATOR_COMPATIBLE_PROFILE,
    get_prediction_feature_profile,
    load_and_validate_legacy_exploratory_dataset,
    prediction_rl_definition_rows,
    validate_dataset_feature_profile,
    validate_simulator_compatible_features,
)
from src.rl_env.state_manifests import (
    END_TO_END_DYNAMIC_FEATURES,
    END_TO_END_STATIC_FEATURES,
    FEATURE_REGISTRY,
)


def test_prediction_and_rl_use_identical_names_units_windows_and_parents() -> None:
    profile = get_prediction_feature_profile(SIMULATOR_COMPATIBLE_PROFILE)
    assert profile.dynamic_feature_names == END_TO_END_DYNAMIC_FEATURES
    assert profile.static_feature_names == END_TO_END_STATIC_FEATURES
    assert len(profile.dynamic_feature_names) == 13
    rows = prediction_rl_definition_rows()
    assert [row["name"] for row in rows] == list(profile.feature_names)
    for row in rows:
        rl = FEATURE_REGISTRY[row["name"]]
        assert row["units"] == rl.units
        assert row["temporal_window_seconds"] == rl.temporal_window_seconds
        assert tuple(row["deterministic_parents"]) == rl.deterministic_parents
        assert row["end_to_end_eligible"] is True


@pytest.mark.parametrize(
    "feature",
    ["hr", "pleth_hr", "mbp", "spo2", "etco2", "hrv", "bis_sqi"],
)
def test_unsupported_physiology_is_rejected_from_main_prediction_profile(
    feature: str,
) -> None:
    with pytest.raises(ValueError, match="Unsupported physiological"):
        validate_simulator_compatible_features(
            [*END_TO_END_DYNAMIC_FEATURES, feature], END_TO_END_STATIC_FEATURES
        )


def test_ambiguous_legacy_bis_slope_is_absent_from_final_profile() -> None:
    profile = get_prediction_feature_profile(SIMULATOR_COMPATIBLE_PROFILE)
    assert "bis_delta_10s" in profile.dynamic_feature_names
    assert "bis_slope" not in profile.dynamic_feature_names
    assert "bis_slope" in LEGACY_DYNAMIC_FEATURES


def test_unversioned_physiological_dataset_is_rejected_as_main() -> None:
    metadata = {
        "dynamic_feature_names": list(LEGACY_DYNAMIC_FEATURES),
        "static_feature_names": ["age", "sex_male", "height", "weight", "bmi", "asa"],
    }
    with pytest.raises(ValueError, match="legacy exploratory"):
        validate_dataset_feature_profile(metadata)


def test_profile_metadata_is_strict_json_and_does_not_claim_selection() -> None:
    profile = get_prediction_feature_profile(SIMULATOR_COMPATIBLE_PROFILE)
    payload = profile.as_metadata()
    assert payload["final_selected_feature_set_decided"] is False
    assert json.loads(json.dumps(payload, allow_nan=False))["feature_profile_version"] == 2


def test_main_dataset_cannot_be_relabelled_as_legacy_to_bypass_guard(tmp_path) -> None:
    profile = get_prediction_feature_profile(SIMULATOR_COMPATIBLE_PROFILE)
    (tmp_path / "dataset_metadata.json").write_text(
        json.dumps(profile.as_metadata()), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="cannot be relabeled"):
        load_and_validate_legacy_exploratory_dataset(tmp_path)
