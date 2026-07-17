"""Canonical prediction feature profiles and prediction-to-RL consistency checks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from src.rl_env.state_manifests import (
    END_TO_END_DYNAMIC_FEATURES,
    END_TO_END_STATIC_FEATURES,
    FEATURE_REGISTRY,
)


PredictionFeatureProfileName = Literal[
    "simulator_compatible",
    "legacy_physiological_exploratory",
]

SIMULATOR_COMPATIBLE_PROFILE = "simulator_compatible"
LEGACY_PHYSIOLOGICAL_PROFILE = "legacy_physiological_exploratory"
SIMULATOR_COMPATIBLE_PROFILE_VERSION = 1

LEGACY_DYNAMIC_FEATURES = (
    "bis",
    "bis_sqi",
    "hr",
    "mbp",
    "sbp",
    "dbp",
    "spo2",
    "etco2",
    "ppf_rate",
    "ppf_volume",
    "ppf_cp",
    "ppf_ce",
    "rftn_rate",
    "rftn_volume",
    "rftn_cp",
    "rftn_ce",
    "bis_slope",
    "bis_error",
)
LEGACY_STATIC_FEATURES = ("age", "sex_male", "height", "weight", "bmi", "asa")

UNSUPPORTED_PHYSIOLOGICAL_FEATURES = frozenset(
    {
        "hr",
        "pleth_hr",
        "mbp",
        "sbp",
        "dbp",
        "nibp_mbp",
        "nibp_sbp",
        "nibp_dbp",
        "spo2",
        "etco2",
        "respiratory_rate",
        "respiratory_variables",
        "hrv",
        "pleth_waveform",
        "pleth_waveform_features",
        "bis_sqi",
    }
)


@dataclass(frozen=True)
class PredictionFeatureProfile:
    """Ordered feature contract for one prediction-dataset family."""

    name: PredictionFeatureProfileName
    version: int
    dynamic_feature_names: tuple[str, ...]
    static_feature_names: tuple[str, ...]
    scientific_role: str
    final_selection_decided: bool

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (*self.dynamic_feature_names, *self.static_feature_names)

    def as_metadata(self) -> dict[str, Any]:
        feature_definitions = []
        for name in self.feature_names:
            metadata = FEATURE_REGISTRY.get(name)
            feature_definitions.append(
                metadata.as_dict()
                if metadata is not None
                else {
                    "name": name,
                    "legacy_definition_only": True,
                    "end_to_end_eligible": False,
                }
            )
        return {
            "feature_profile": self.name,
            "feature_profile_version": self.version,
            "scientific_role": self.scientific_role,
            "dynamic_feature_names": list(self.dynamic_feature_names),
            "static_feature_names": list(self.static_feature_names),
            "final_selected_feature_set_decided": self.final_selection_decided,
            "feature_definitions": feature_definitions,
        }


FEATURE_PROFILES: dict[str, PredictionFeatureProfile] = {
    SIMULATOR_COMPATIBLE_PROFILE: PredictionFeatureProfile(
        name=SIMULATOR_COMPATIBLE_PROFILE,
        version=SIMULATOR_COMPATIBLE_PROFILE_VERSION,
        dynamic_feature_names=END_TO_END_DYNAMIC_FEATURES,
        static_feature_names=END_TO_END_STATIC_FEATURES,
        scientific_role="main_end_to_end_candidate_universe",
        final_selection_decided=False,
    ),
    LEGACY_PHYSIOLOGICAL_PROFILE: PredictionFeatureProfile(
        name=LEGACY_PHYSIOLOGICAL_PROFILE,
        version=1,
        dynamic_feature_names=LEGACY_DYNAMIC_FEATURES,
        static_feature_names=LEGACY_STATIC_FEATURES,
        scientific_role="legacy_exploratory_not_valid_for_final_selection",
        final_selection_decided=False,
    ),
}


def get_prediction_feature_profile(name: str) -> PredictionFeatureProfile:
    """Resolve one named profile without inferring a fallback."""

    try:
        return FEATURE_PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown prediction feature profile {name!r}; choices={sorted(FEATURE_PROFILES)}."
        ) from exc


def validate_simulator_compatible_features(
    dynamic_features: Sequence[str],
    static_features: Sequence[str],
) -> None:
    """Require the exact ordered end-to-end universe and reject physiology."""

    dynamic = tuple(dynamic_features)
    static = tuple(static_features)
    requested = set(dynamic) | set(static)
    unsupported = sorted(requested & UNSUPPORTED_PHYSIOLOGICAL_FEATURES)
    if unsupported:
        raise ValueError(
            "Unsupported physiological variables cannot enter the simulator-compatible "
            f"prediction profile: {unsupported}."
        )
    if dynamic != END_TO_END_DYNAMIC_FEATURES or static != END_TO_END_STATIC_FEATURES:
        raise ValueError(
            "The simulator-compatible prediction profile requires exact ordered features; "
            f"dynamic={list(END_TO_END_DYNAMIC_FEATURES)}, "
            f"static={list(END_TO_END_STATIC_FEATURES)}."
        )
    for name in (*dynamic, *static):
        metadata = FEATURE_REGISTRY[name]
        if not metadata.simulator_supported or not metadata.end_to_end_eligible:
            raise ValueError(
                f"Feature {name!r} is not end-to-end eligible: {metadata.eligibility_note}"
            )


def validate_dataset_feature_profile(
    metadata: Mapping[str, Any],
    *,
    required_profile: str = SIMULATOR_COMPATIBLE_PROFILE,
) -> None:
    """Validate dataset identity before a main prediction run starts."""

    observed = metadata.get("feature_profile")
    if observed != required_profile:
        legacy = observed or "unversioned_legacy"
        raise ValueError(
            f"Dataset feature profile is {legacy!r}, not {required_profile!r}. "
            "Prior physiological-inclusive datasets are legacy exploratory inputs."
        )
    profile = get_prediction_feature_profile(required_profile)
    if required_profile == SIMULATOR_COMPATIBLE_PROFILE:
        validate_simulator_compatible_features(
            metadata.get("dynamic_feature_names", ()),
            metadata.get("static_feature_names", ()),
        )
    if int(metadata.get("feature_profile_version", -1)) != profile.version:
        raise ValueError("Dataset feature-profile version is incompatible.")


def load_dataset_feature_profile_metadata(dataset_dir: Path) -> dict[str, Any]:
    """Read dataset metadata without inferring a profile for legacy artifacts."""

    path = dataset_dir / "dataset_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset metadata is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset metadata root must be an object: {path}")
    return payload


def load_and_validate_dataset_feature_profile(
    dataset_dir: Path,
    *,
    required_profile: str = SIMULATOR_COMPATIBLE_PROFILE,
) -> dict[str, Any]:
    """Read and validate dataset metadata before a main CLI dispatches training."""

    payload = load_dataset_feature_profile_metadata(dataset_dir)
    validate_dataset_feature_profile(payload, required_profile=required_profile)
    return payload


def load_and_validate_legacy_exploratory_dataset(dataset_dir: Path) -> dict[str, Any]:
    """Require an old or explicitly legacy dataset for a legacy-only CLI run."""

    payload = load_dataset_feature_profile_metadata(dataset_dir)
    observed = payload.get("feature_profile")
    if observed == SIMULATOR_COMPATIBLE_PROFILE:
        raise ValueError(
            "The simulator-compatible dataset cannot be relabeled as legacy exploratory "
            "to bypass the complete-candidate-universe guard."
        )
    if observed not in {None, LEGACY_PHYSIOLOGICAL_PROFILE}:
        raise ValueError(f"Dataset has an unsupported legacy profile: {observed!r}.")
    return payload


def prediction_rl_definition_rows() -> list[dict[str, Any]]:
    """Return the shared names, units, windows, and deterministic definitions."""

    validate_simulator_compatible_features(
        END_TO_END_DYNAMIC_FEATURES, END_TO_END_STATIC_FEATURES
    )
    feature_names = (*END_TO_END_DYNAMIC_FEATURES, *END_TO_END_STATIC_FEATURES)
    return [FEATURE_REGISTRY[name].as_dict() for name in feature_names]
