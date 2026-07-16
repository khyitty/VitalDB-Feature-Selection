"""Focused contracts for canonical RL profiles and selected-state manifests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.rl_env.state_adapters import get_state_profile
from src.rl_env.state_manifests import (
    PendingStateSelectionError,
    StateManifestError,
    load_selected_state_manifest,
    validate_selected_state_manifest,
)
from src.rl_training.policy_registry import (
    primary_state_policy_contract,
    validate_state_only_comparison,
)


ROOT = Path(__file__).resolve().parents[1]


def _resolved_manifest(**overrides):
    payload = {
        "schema_version": 1,
        "profile_name": "selected",
        "feature_names": ["bis", "propofol_rate_mg_per_min", "age_years"],
        "selection_method": "multiseed_attention_stability",
        "selection_source_artifact": "outputs/example/validation_attention.csv",
        "selection_split": "validation",
        "seeds": [7, 21, 42, 84, 123],
        "patient_aggregation_rule": "equal patient weight",
        "feature_aggregation_rule": "median normalized attention across seeds",
        "threshold_or_top_k_rule": "selection frequency >= predeclared threshold",
        "timestamp": "2026-07-16T00:00:00+00:00",
        "git_commit": "a" * 40,
        "notes": "Non-scientific test fixture.",
    }
    payload.update(overrides)
    return payload


def test_pending_template_is_parseable_but_cannot_execute() -> None:
    path = ROOT / "configs/rl_state_profiles/selected_template.json"
    manifest = load_selected_state_manifest(path, require_resolved=False)
    assert manifest.pending and manifest.feature_names == ()
    with pytest.raises(PendingStateSelectionError, match="pending supervisor"):
        load_selected_state_manifest(path)


def test_resolved_selected_manifest_preserves_exact_feature_order(tmp_path: Path) -> None:
    path = tmp_path / "selected.json"
    path.write_text(json.dumps(_resolved_manifest()), encoding="utf-8")
    profile = get_state_profile("selected", selected_manifest_path=path)
    assert profile.dynamic_feature_names == ("bis", "propofol_rate_mg_per_min")
    assert profile.static_feature_names == ("age_years",)
    assert profile.ordered_feature_names == tuple(_resolved_manifest()["feature_names"])
    assert profile.observation_dimension() == 20


@pytest.mark.parametrize("feature", ["hr", "mbp", "spo2", "etco2", "bis_sqi", "hrv"])
def test_prediction_only_physiology_is_rejected_from_rl_manifest(feature: str) -> None:
    with pytest.raises(StateManifestError, match="prediction-only or simulator-unsupported"):
        validate_selected_state_manifest(
            _resolved_manifest(feature_names=["bis", feature])
        )


def test_test_split_selection_and_unknown_features_are_rejected() -> None:
    with pytest.raises(StateManifestError, match="test is forbidden"):
        validate_selected_state_manifest(_resolved_manifest(selection_split="test"))
    with pytest.raises(StateManifestError, match="Unknown RL state feature"):
        validate_selected_state_manifest(_resolved_manifest(feature_names=["made_up_track"]))


def test_selected_manifest_rejects_order_that_cannot_be_preserved() -> None:
    with pytest.raises(StateManifestError, match="dynamic features before static"):
        validate_selected_state_manifest(
            _resolved_manifest(feature_names=["age_years", "bis"])
        )


@pytest.mark.parametrize("alias", ["original_yun", "yun_reconstructed"])
def test_original_aliases_warn_and_resolve_canonically(alias: str) -> None:
    with pytest.warns(DeprecationWarning, match="deprecated"):
        profile = get_state_profile(alias)
    assert profile.name == "original_reconstructed"


def test_legacy_selected_name_is_not_the_proposed_selected_state() -> None:
    with pytest.warns(DeprecationWarning):
        profile = get_state_profile("selected_control_aware")
    assert profile.name == "legacy_control_aware"
    assert "not the proposed selected state" in profile.purpose


def test_primary_state_policy_contract_is_architecture_identical() -> None:
    original = primary_state_policy_contract("original_reconstructed")
    all_supported = primary_state_policy_contract("all_supported")
    validate_state_only_comparison([original, all_supported])
    assert original.architecture_signature == all_supported.architecture_signature
    assert original.observation_dimension != all_supported.observation_dimension


def test_state_only_guard_rejects_changed_encoder() -> None:
    original = primary_state_policy_contract("original_reconstructed")
    changed = replace(
        primary_state_policy_contract("all_supported"),
        feature_extractor="custom_attention",
    )
    with pytest.raises(ValueError, match="changes policy or feature-extractor"):
        validate_state_only_comparison([original, changed])
