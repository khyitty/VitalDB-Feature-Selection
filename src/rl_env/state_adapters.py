"""Ordered observation profiles that never alter simulator transitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from src.pkpd.demographics import PatientDemographics

from .config import StateProfileName
from .history import HistoryBuffer
from .state_manifests import (
    FEATURE_REGISTRY,
    SelectedStateManifest,
    feature_registry_metadata,
    load_selected_state_manifest,
)


STATIC_FEATURE_NAMES = ("age_years", "sex_male", "height_cm", "weight_kg")
STATIC_UNITS = ("year", "binary", "cm", "kg")

ORIGINAL_YUN_FEATURES = (
    "bis",
    "bis_delta_10s",
    "bis_target_error",
    "propofol_rate_mg_per_min",
    "propofol_recent_dose_mg",
    "remifentanil_rate_micrograms_per_min",
    "remifentanil_recent_dose_micrograms",
)

ALL_SUPPORTED_FEATURES = (
    "bis",
    "bis_delta_10s",
    "bis_target_error",
    "propofol_rate_mg_per_min",
    "propofol_recent_dose_mg",
    "propofol_cumulative_dose_mg",
    "remifentanil_rate_micrograms_per_min",
    "remifentanil_recent_dose_micrograms",
    "remifentanil_cumulative_dose_micrograms",
)

SELECTED_CONTROL_AWARE_FEATURES = (
    "bis",
    "bis_delta_10s",
    "bis_target_error",
    "propofol_rate_mg_per_min",
    "propofol_recent_dose_mg",
    "propofol_cp_mg_per_l",
    "remifentanil_rate_micrograms_per_min",
    "remifentanil_ce_micrograms_per_l",
)

@dataclass(frozen=True)
class StateProfile:
    name: str
    dynamic_feature_names: tuple[str, ...]
    static_feature_names: tuple[str, ...] = STATIC_FEATURE_NAMES
    purpose: str = ""
    selected_manifest: SelectedStateManifest | None = None

    @property
    def ordered_feature_names(self) -> tuple[str, ...]:
        """Return the exact dynamic-then-static policy feature order."""

        return (*self.dynamic_feature_names, *self.static_feature_names)

    def observation_dimension(self, history_steps: int = 6) -> int:
        """Return the flattened dimension including mask and target context."""

        return (
            history_steps * len(self.dynamic_feature_names)
            + history_steps
            + len(self.static_feature_names)
            + 1
        )

    def observation(
        self,
        history: HistoryBuffer,
        demographics: PatientDemographics,
        target_bis: float,
    ) -> dict[str, np.ndarray]:
        static_values = {
            "age_years": demographics.age_years,
            "sex_male": float(demographics.sex == "male"),
            "height_cm": demographics.height_cm,
            "weight_kg": demographics.weight_kg,
        }
        static = np.asarray(
            [static_values[name] for name in self.static_feature_names], dtype=np.float32
        )
        observation = {
            "history": history.matrix(self.dynamic_feature_names),
            "history_mask": history.mask,
            "static": static,
            "target_bis": np.asarray([target_bis], dtype=np.float32),
        }
        if not all(np.isfinite(value).all() for value in observation.values()):
            raise FloatingPointError("Observation contains NaN or infinity.")
        return observation

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "ordered_feature_names": list(self.ordered_feature_names),
            "dynamic_feature_names": list(self.dynamic_feature_names),
            "dynamic_feature_units": [
                FEATURE_REGISTRY[name].units for name in self.dynamic_feature_names
            ],
            "static_feature_names": list(self.static_feature_names),
            "static_feature_units": [
                FEATURE_REGISTRY[name].units for name in self.static_feature_names
            ],
            "observation_dimension_60s_10s": self.observation_dimension(),
            "history_window_seconds": 60,
            "decision_interval_seconds": 10,
            "normalization": (
                "raw physical values in the environment; fixed physical scaling in policy"
            ),
            "bis_signal": "raw causal observed BIS; offline LOWESS is not reproduced",
            "features": [
                FEATURE_REGISTRY[name].as_dict() for name in self.ordered_feature_names
            ],
            "selected_manifest": (
                self.selected_manifest.as_dict() if self.selected_manifest is not None else None
            ),
        }


STATE_PROFILES: dict[str, StateProfile] = {
    "original_reconstructed": StateProfile(
        "original_reconstructed",
        ORIGINAL_YUN_FEATURES,
        purpose=(
            "Reconstructed Yun-informed baseline using raw causal BIS because the paper "
            "does not specify an online causal LOWESS procedure or unpublished code."
        ),
    ),
    "all_supported": StateProfile(
        "all_supported",
        ALL_SUPPORTED_FEATURES,
        purpose="All end-to-end prediction/simulator-compatible control observations.",
    ),
    "attention_ready": StateProfile(
        "attention_ready",
        ALL_SUPPORTED_FEATURES,
        purpose=(
            "Same raw information as all_supported in structured history/static form; "
            "attention is a future policy-encoder responsibility."
        ),
    ),
    "legacy_control_aware": StateProfile(
        "legacy_control_aware",
        SELECTED_CONTROL_AWARE_FEATURES,
        purpose=(
            "Legacy debugging subset based on a predictive intersection plus protected "
            "control variables; this is not the proposed selected state."
        ),
    ),
}

STATE_PROFILE_ALIASES: dict[str, str] = {
    "original_yun": "original_reconstructed",
    "yun_reconstructed": "original_reconstructed",
    "selected_control_aware": "legacy_control_aware",
}

PREDICTIVE_STRICT_FEATURES = (
    "bis",
    "bis_sqi",
    "ppf_rate",
    "ppf_volume",
    "ppf_cp",
    "rftn_volume",
    "bis_slope",
)
PREDICTIVE_INTERSECTION = ("bis", "bis_slope", "ppf_rate", "ppf_cp")
UNSUPPORTED_PREDICTIVE_FEATURES = ("bis_sqi", "ppf_volume", "rftn_volume")
UNSUPPORTED_VITAL_SIGNS = ("HR", "MBP", "SBP", "DBP", "SpO2", "ETCO2", "HRV")
EXCLUDED_LATENT_STATES = (
    "propofol_x1",
    "propofol_x2",
    "propofol_x3",
    "remifentanil_x1",
    "remifentanil_x2",
    "remifentanil_x3",
)


def get_state_profile(
    name: StateProfileName | str,
    *,
    selected_manifest_path: Path | None = None,
) -> StateProfile:
    """Resolve canonical profiles while warning for temporary legacy names."""

    canonical = STATE_PROFILE_ALIASES.get(name, name)
    if canonical != name:
        warnings.warn(
            f"State profile {name!r} is deprecated; use {canonical!r}.",
            DeprecationWarning,
            stacklevel=2,
        )
    if canonical == "selected":
        if selected_manifest_path is None:
            raise ValueError("The selected profile requires selected_state_manifest.")
        manifest = load_selected_state_manifest(selected_manifest_path, require_resolved=True)
        dynamic = tuple(
            feature
            for feature in manifest.feature_names
            if FEATURE_REGISTRY[feature].static_or_dynamic == "dynamic"
        )
        static = tuple(
            feature
            for feature in manifest.feature_names
            if FEATURE_REGISTRY[feature].static_or_dynamic == "static"
        )
        return StateProfile(
            name="selected",
            dynamic_feature_names=dynamic,
            static_feature_names=static,
            purpose="Versioned manifest-selected simulator-supported state.",
            selected_manifest=manifest,
        )
    try:
        return STATE_PROFILES[canonical]
    except KeyError as exc:
        raise ValueError(f"Unknown state profile {name!r}.") from exc


def state_profile_registry() -> dict[str, Any]:
    return {
        "profiles": {name: profile.metadata() for name, profile in STATE_PROFILES.items()},
        "selected_profile": {
            "name": "selected",
            "status": "requires_resolved_versioned_manifest",
            "hard_coded_features": False,
        },
        "aliases": dict(STATE_PROFILE_ALIASES),
        "canonical_primary_profiles": [
            "original_reconstructed",
            "all_supported",
            "selected",
        ],
        "official_experiment_baseline_name": "original_reconstructed",
        "original_reconstructed_is_exact_reproduction": False,
        "all_attention_raw_information_equal": (
            STATE_PROFILES["all_supported"].dynamic_feature_names
            == STATE_PROFILES["attention_ready"].dynamic_feature_names
        ),
        "predictive_strict_features": list(PREDICTIVE_STRICT_FEATURES),
        "predictive_intersection": list(PREDICTIVE_INTERSECTION),
        "unsupported_predictive_features_removed": list(UNSUPPORTED_PREDICTIVE_FEATURES),
        "selected_control_protected_additions": [
            "bis_target_error",
            "propofol_recent_dose_mg",
            "remifentanil_rate_micrograms_per_min",
            "remifentanil_ce_micrograms_per_l",
            *STATIC_FEATURE_NAMES,
        ],
        "unsupported_vital_signs": list(UNSUPPORTED_VITAL_SIGNS),
        "excluded_internal_latent_states": list(EXCLUDED_LATENT_STATES),
        "feature_registry": feature_registry_metadata(),
    }
