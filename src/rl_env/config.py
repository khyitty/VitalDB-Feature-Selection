"""Validated configuration for the research propofol-control environment."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Literal


StateProfileName = Literal[
    "original_reconstructed",
    "prediction_minimal",
    "selected_control_core",
    "selected",
    "legacy_control_aware",
    "original_yun",
    "yun_reconstructed",
    "all_supported",
    "attention_ready",
    "selected_control_aware",
]
ActionMode = Literal["strict", "clip"]

RESEARCH_ONLY_WARNING = (
    "This environment is a research-only reconstruction around a published "
    "PK-PD simulator. It is not a medical device and must not be used for "
    "clinical dosing or patient care."
)


@dataclass(frozen=True)
class ActionBounds:
    """Physical propofol-rate limits and their provenance."""

    low_mg_per_min: float
    high_mg_per_min: float
    profile_name: str
    provenance: str

    def __post_init__(self) -> None:
        low = float(self.low_mg_per_min)
        high = float(self.high_mg_per_min)
        if not math.isfinite(low) or not math.isfinite(high) or low < 0.0 or high <= low:
            raise ValueError("Action bounds require finite 0 <= low < high.")
        object.__setattr__(self, "low_mg_per_min", low)
        object.__setattr__(self, "high_mg_per_min", high)


YUN_2023_CONVERTED_ACTION_BOUNDS = ActionBounds(
    low_mg_per_min=0.0,
    high_mg_per_min=166.2,
    profile_name="yun2023_converted",
    provenance=(
        "Yun 2023 PDF p.5 states 0--27.7 mg over each 10-second decision; "
        "converted to the simulator API by multiplying by 60/10."
    ),
)

YUN_REPORTED_ACTION_BOUNDS = ActionBounds(
    low_mg_per_min=0.0,
    high_mg_per_min=166.2,
    profile_name="yun_reported_action_range",
    provenance=(
        "Official experiment alias for Yun 2023 PDF p.5 action range: "
        "27.7 mg per 10 seconds converted exactly to 166.2 mg/min."
    ),
)

SYNTHETIC_NONCLINICAL_ACTION_BOUNDS = ActionBounds(
    low_mg_per_min=0.0,
    high_mg_per_min=12.0,
    profile_name="synthetic_nonclinical_v1",
    provenance=(
        "Repository-designed narrow bound for synthetic validation only; this is "
        "not a Yun 2023 action bound and is not a clinical dosing recommendation."
    ),
)


def action_bounds_from_profile(profile_name: str) -> ActionBounds:
    """Resolve a named, traceable action-bound profile."""

    profiles = {
        YUN_2023_CONVERTED_ACTION_BOUNDS.profile_name: YUN_2023_CONVERTED_ACTION_BOUNDS,
        YUN_REPORTED_ACTION_BOUNDS.profile_name: YUN_REPORTED_ACTION_BOUNDS,
        SYNTHETIC_NONCLINICAL_ACTION_BOUNDS.profile_name: (
            SYNTHETIC_NONCLINICAL_ACTION_BOUNDS
        ),
    }
    try:
        return profiles[profile_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown action-bound profile {profile_name!r}; choices={sorted(profiles)}."
        ) from exc


@dataclass(frozen=True)
class EnvironmentConfig:
    """Environment contract shared by every observation profile."""

    action_interval_seconds: float = 10.0
    internal_dt_seconds: float = 1.0
    history_window_seconds: float = 60.0
    episode_duration_seconds: float = 1800.0
    target_bis: float = 50.0
    safe_bis_low: float = 40.0
    safe_bis_high: float = 60.0
    deterministic: bool = True
    integrator: Literal["exact", "solve_ivp"] = "exact"
    action_bounds: ActionBounds = field(default_factory=lambda: YUN_2023_CONVERTED_ACTION_BOUNDS)
    action_mode: ActionMode = "strict"
    state_profile: StateProfileName = "original_reconstructed"
    selected_state_manifest: Path | None = None
    reward_profile: str = "transparent_tracking_v1"
    paper_reward_alpha: float | None = None
    action_magnitude_coefficient: float = 0.0
    action_change_coefficient: float = 0.0
    concentration_safety_coefficient: float = 0.0
    propofol_ce_threshold_mg_per_l: float | None = None
    excessive_action_change_threshold_mg_per_min: float = 4.0

    def __post_init__(self) -> None:
        finite_positive = {
            "action_interval_seconds": self.action_interval_seconds,
            "internal_dt_seconds": self.internal_dt_seconds,
            "history_window_seconds": self.history_window_seconds,
            "episode_duration_seconds": self.episode_duration_seconds,
        }
        for name, value in finite_positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.action_interval_seconds != 10.0:
            raise ValueError("Module 5 fixes action_interval_seconds at 10 seconds.")
        if self.internal_dt_seconds != 1.0:
            raise ValueError("Module 5 fixes internal_dt_seconds at 1 second.")
        if self.history_window_seconds != 60.0:
            raise ValueError("Module 5 fixes history_window_seconds at 60 seconds.")
        if self.episode_duration_seconds % self.action_interval_seconds != 0.0:
            raise ValueError("episode_duration_seconds must be an exact multiple of 10.")
        if not 0.0 <= self.safe_bis_low < self.target_bis < self.safe_bis_high <= 100.0:
            raise ValueError("Require 0 <= safe low < target < safe high <= 100.")
        if self.action_mode not in ("strict", "clip"):
            raise ValueError("action_mode must be 'strict' or 'clip'.")
        if self.integrator not in ("exact", "solve_ivp"):
            raise ValueError("integrator must be 'exact' or 'solve_ivp'.")
        if self.reward_profile not in (
            "transparent_tracking_v1",
            "paper_yun2023_parameterized",
        ):
            raise ValueError(f"Unsupported reward profile: {self.reward_profile!r}.")
        if self.reward_profile == "paper_yun2023_parameterized" and (
            self.paper_reward_alpha is None
            or not math.isfinite(self.paper_reward_alpha)
            or self.paper_reward_alpha <= 0.0
        ):
            raise ValueError(
                "paper_yun2023_parameterized requires an explicit positive alpha; "
                "Yun 2023 does not report its numeric value."
            )
        for name in (
            "action_magnitude_coefficient",
            "action_change_coefficient",
            "concentration_safety_coefficient",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        threshold = self.propofol_ce_threshold_mg_per_l
        if threshold is not None and (not math.isfinite(threshold) or threshold <= 0.0):
            raise ValueError("propofol_ce_threshold_mg_per_l must be positive when set.")

    @property
    def history_steps(self) -> int:
        return int(self.history_window_seconds / self.action_interval_seconds)
