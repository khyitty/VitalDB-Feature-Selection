"""Ordered observation profiles that never alter simulator transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.pkpd.demographics import PatientDemographics

from .config import StateProfileName
from .history import HistoryBuffer


STATIC_FEATURE_NAMES = ("age_years", "sex_male", "height_cm", "weight_kg")
STATIC_UNITS = ("year", "binary", "cm", "kg")

ORIGINAL_YUN_FEATURES = (
    "bis",
    "bis_slope",
    "bis_target_error",
    "propofol_rate_mg_per_min",
    "propofol_recent_dose_mg",
    "remifentanil_rate_micrograms_per_min",
    "remifentanil_recent_dose_micrograms",
)

ALL_SUPPORTED_FEATURES = (
    "bis",
    "bis_slope",
    "bis_target_error",
    "propofol_rate_mg_per_min",
    "propofol_recent_dose_mg",
    "propofol_cumulative_dose_mg",
    "propofol_cp_mg_per_l",
    "propofol_ce_mg_per_l",
    "remifentanil_rate_micrograms_per_min",
    "remifentanil_recent_dose_micrograms",
    "remifentanil_cumulative_dose_micrograms",
    "remifentanil_cp_micrograms_per_l",
    "remifentanil_ce_micrograms_per_l",
)

SELECTED_CONTROL_AWARE_FEATURES = (
    "bis",
    "bis_slope",
    "bis_target_error",
    "propofol_rate_mg_per_min",
    "propofol_recent_dose_mg",
    "propofol_cp_mg_per_l",
    "remifentanil_rate_micrograms_per_min",
    "remifentanil_ce_micrograms_per_l",
)

FEATURE_UNITS = {
    "bis": "BIS unit",
    "bis_slope": "BIS unit per 10-second decision",
    "bis_target_error": "BIS unit",
    "propofol_rate_mg_per_min": "mg/min",
    "propofol_recent_dose_mg": "mg per causal 60-second window",
    "propofol_cumulative_dose_mg": "mg",
    "propofol_cp_mg_per_l": "mg/L",
    "propofol_ce_mg_per_l": "mg/L",
    "remifentanil_rate_micrograms_per_min": "microgram/min",
    "remifentanil_recent_dose_micrograms": "microgram per causal 60-second window",
    "remifentanil_cumulative_dose_micrograms": "microgram",
    "remifentanil_cp_micrograms_per_l": "microgram/L",
    "remifentanil_ce_micrograms_per_l": "microgram/L",
}


@dataclass(frozen=True)
class StateProfile:
    name: StateProfileName
    dynamic_feature_names: tuple[str, ...]
    static_feature_names: tuple[str, ...] = STATIC_FEATURE_NAMES
    purpose: str = ""

    def observation(
        self,
        history: HistoryBuffer,
        demographics: PatientDemographics,
        target_bis: float,
    ) -> dict[str, np.ndarray]:
        static = np.asarray(
            [
                demographics.age_years,
                float(demographics.sex == "male"),
                demographics.height_cm,
                demographics.weight_kg,
            ],
            dtype=np.float32,
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
            "dynamic_feature_names": list(self.dynamic_feature_names),
            "dynamic_feature_units": [FEATURE_UNITS[name] for name in self.dynamic_feature_names],
            "static_feature_names": list(self.static_feature_names),
            "static_feature_units": list(STATIC_UNITS),
            "normalization": "raw_physical_values; no fitted statistics in environment core",
            "bis_signal": "raw causal observed BIS; offline LOWESS is not reproduced",
        }


STATE_PROFILES: dict[StateProfileName, StateProfile] = {
    "original_yun": StateProfile(
        "original_yun",
        ORIGINAL_YUN_FEATURES,
        purpose=(
            "Yun 2023 seven-concept baseline reconstructed with raw causal BIS because "
            "the source does not specify an online causal LOWESS procedure or W."
        ),
    ),
    "all_supported": StateProfile(
        "all_supported",
        ALL_SUPPORTED_FEATURES,
        purpose="All non-latent simulator-supported control observations.",
    ),
    "attention_ready": StateProfile(
        "attention_ready",
        ALL_SUPPORTED_FEATURES,
        purpose=(
            "Same raw information as all_supported in structured history/static form; "
            "attention is a future policy-encoder responsibility."
        ),
    ),
    "selected_control_aware": StateProfile(
        "selected_control_aware",
        SELECTED_CONTROL_AWARE_FEATURES,
        purpose=(
            "Simulator-supported predictive intersection plus protected target, action, "
            "remifentanil disturbance, and demographic variables."
        ),
    ),
}

STATE_PROFILE_ALIASES: dict[str, str] = {
    "yun_reconstructed": "original_yun",
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


def get_state_profile(name: StateProfileName) -> StateProfile:
    if name == "yun_reconstructed":
        original = STATE_PROFILES["original_yun"]
        return StateProfile(
            name="yun_reconstructed",
            dynamic_feature_names=original.dynamic_feature_names,
            static_feature_names=original.static_feature_names,
            purpose=(
                "Official experiment name for the raw-causal reconstructed Yun baseline; "
                "this is not a complete reproduction of Yun's unspecified LOWESS pipeline."
            ),
        )
    try:
        return STATE_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown state profile {name!r}.") from exc


def state_profile_registry() -> dict[str, Any]:
    return {
        "profiles": {name: profile.metadata() for name, profile in STATE_PROFILES.items()},
        "aliases": dict(STATE_PROFILE_ALIASES),
        "official_experiment_baseline_name": "yun_reconstructed",
        "yun_reconstructed_is_exact_reproduction": False,
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
    }
