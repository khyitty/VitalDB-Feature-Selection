"""Gymnasium wrapper around the validated Module 4 PK-PD simulator."""

from __future__ import annotations

import math
from typing import Any, Mapping

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.pkpd.demographics import PatientDemographics
from src.pkpd.schedules import ConstantSchedule, RateSchedule
from src.pkpd.simulator import CombinedPatientState, PKPDSimulator
from src.pkpd.validation import SYNTHETIC_PATIENTS

from .action import validate_action
from .cohort import PatientCohort
from .config import EnvironmentConfig
from .history import HistoryBuffer
from .metrics import EpisodeMetricsCollector
from .observation import make_observation_space
from .reward import RewardCalculator
from .schedules import ConstantTargetSchedule, TargetSchedule
from .state_adapters import StateProfile, get_state_profile


class PropofolControlEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """Research-only continuous propofol-control environment.

    Propofol is the sole agent action. Remifentanil is an exogenous schedule.
    Every action is held for ten seconds while the simulator advances internally
    in one-second substeps.
    """

    metadata = {"render_modes": [], "render_fps": 0}

    def __init__(
        self,
        config: EnvironmentConfig | None = None,
        *,
        patient_profiles: Mapping[str, PatientDemographics] | None = None,
        cohort: PatientCohort | None = None,
        remifentanil_schedule: RateSchedule | None = None,
        target_schedule: TargetSchedule | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EnvironmentConfig()
        self.patient_profiles = dict(patient_profiles or SYNTHETIC_PATIENTS)
        self.cohort = cohort
        self._default_remifentanil_schedule = remifentanil_schedule or ConstantSchedule(0.0)
        self._default_target_schedule = target_schedule or ConstantTargetSchedule(
            self.config.target_bis
        )
        self._state_profile: StateProfile = get_state_profile(self.config.state_profile)
        bounds = self.config.action_bounds
        self.action_space = spaces.Box(
            low=np.asarray([bounds.low_mg_per_min], dtype=np.float32),
            high=np.asarray([bounds.high_mg_per_min], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = make_observation_space(
            self._state_profile, history_steps=self.config.history_steps
        )
        self._simulator: PKPDSimulator | None = None
        self._history = HistoryBuffer(
            history_steps=self.config.history_steps,
            action_interval_seconds=self.config.action_interval_seconds,
        )
        self._reward = RewardCalculator(self.config)
        self._metrics = EpisodeMetricsCollector(
            step_duration_seconds=self.config.action_interval_seconds,
            safe_bis_low=self.config.safe_bis_low,
            safe_bis_high=self.config.safe_bis_high,
            excessive_action_change_threshold_mg_per_min=(
                self.config.excessive_action_change_threshold_mg_per_min
            ),
            propofol_ce_threshold_mg_per_l=(
                self.config.propofol_ce_threshold_mg_per_l
            ),
        )
        self._patient: PatientDemographics | None = None
        self._patient_id = "uninitialized"
        self._patient_profile = "uninitialized"
        self._remifentanil_schedule: RateSchedule = self._default_remifentanil_schedule
        self._target_schedule: TargetSchedule = self._default_target_schedule
        self._episode_start_time = 0.0
        self._previous_action = 0.0
        self._below_safe_seconds = 0.0
        self._above_safe_seconds = 0.0
        self._done = False

    @property
    def simulator(self) -> PKPDSimulator:
        if self._simulator is None:
            raise RuntimeError("Environment must be reset before accessing simulator.")
        return self._simulator

    @property
    def state_profile(self) -> StateProfile:
        return self._state_profile

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        options = dict(options or {})
        simulator_seed = int(
            seed if seed is not None else self.np_random.integers(0, np.iinfo(np.int32).max)
        )
        self.action_space.seed(simulator_seed)
        self._patient, self._patient_id, self._patient_profile = self._resolve_patient(options)
        self._remifentanil_schedule = options.get(
            "remifentanil_schedule", self._default_remifentanil_schedule
        )
        self._target_schedule = options.get("target_schedule", self._default_target_schedule)
        self._validate_schedule(self._remifentanil_schedule, "remifentanil_schedule", "rate_at")
        self._validate_schedule(self._target_schedule, "target_schedule", "target_at")
        deterministic = bool(options.get("deterministic", self.config.deterministic))
        self._simulator = PKPDSimulator(
            internal_dt_seconds=self.config.internal_dt_seconds,
            deterministic=deterministic,
            integrator=self.config.integrator,
        )
        state = self._simulator.reset(
            self._patient,
            simulator_seed,
            initial_state=options.get("initial_state"),
        )
        self._ensure_finite_state(state)
        self._episode_start_time = state.time_seconds
        target = float(self._target_schedule.target_at(state.time_seconds))
        self._history.reset(state, target_bis=target)
        self._metrics.reset()
        self._previous_action = state.propofol_rate_mg_per_min
        self._below_safe_seconds = 0.0
        self._above_safe_seconds = 0.0
        self._done = False
        observation = self._observation(target)
        info = self._info(
            state,
            target_bis=target,
            requested_action=state.propofol_rate_mg_per_min,
            applied_action=state.propofol_rate_mg_per_min,
            action_clipped=False,
            reward_components={},
        )
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._simulator is None:
            raise RuntimeError("Environment must be reset before step().")
        if self._done:
            raise RuntimeError("step() cannot be called after episode completion; reset first.")
        validated = validate_action(
            action,
            bounds=self.config.action_bounds,
            mode=self.config.action_mode,
        )
        previous_action = self._previous_action
        try:
            state = self._simulator.advance(
                propofol_rate_mg_per_min=validated.applied_mg_per_min,
                remifentanil_schedule=self._remifentanil_schedule,
                duration_seconds=self.config.action_interval_seconds,
            )
            self._ensure_finite_state(state)
        except FloatingPointError as exc:
            self._metrics.mark_numerical_failure(type(exc).__name__)
            self._done = True
            raise FloatingPointError(
                f"PK-PD numerical failure during environment transition: {exc}"
            ) from exc
        target = float(self._target_schedule.target_at(state.time_seconds))
        self._history.append(state, target_bis=target)
        reward = self._reward.calculate(
            post_bis=state.observed_bis,
            target_bis=target,
            action_mg_per_min=validated.applied_mg_per_min,
            previous_action_mg_per_min=previous_action,
            propofol_ce_mg_per_l=state.propofol.ce,
        )
        if state.observed_bis < self.config.safe_bis_low:
            self._below_safe_seconds += self.config.action_interval_seconds
        if state.observed_bis > self.config.safe_bis_high:
            self._above_safe_seconds += self.config.action_interval_seconds
        elapsed = state.time_seconds - self._episode_start_time
        terminated = False
        truncated = math.isclose(
            elapsed, self.config.episode_duration_seconds, abs_tol=1e-9
        )
        if elapsed > self.config.episode_duration_seconds + 1e-9:
            raise RuntimeError("Environment advanced beyond the configured episode duration.")
        self._metrics.record(
            time_seconds=elapsed,
            bis=state.observed_bis,
            target_bis=target,
            propofol_rate_mg_per_min=validated.applied_mg_per_min,
            previous_action_mg_per_min=previous_action,
            propofol_cp_mg_per_l=state.propofol.cp,
            propofol_ce_mg_per_l=state.propofol.ce,
            remifentanil_cp_micrograms_per_l=state.remifentanil.cp,
            remifentanil_ce_micrograms_per_l=state.remifentanil.ce,
            reward=reward.total,
            reward_components=reward.components,
        )
        self._previous_action = validated.applied_mg_per_min
        self._done = terminated or truncated
        if truncated:
            self._metrics.set_terminated_reason("fixed_episode_duration")
        observation = self._observation(target)
        info = self._info(
            state,
            target_bis=target,
            requested_action=validated.requested_mg_per_min,
            applied_action=validated.applied_mg_per_min,
            action_clipped=validated.clipped,
            reward_components=reward.components,
        )
        if self._done:
            info["episode_metrics"] = self._metrics.summary()
        return observation, reward.total, terminated, truncated, info

    def close(self) -> None:
        self._simulator = None

    def episode_metrics(self) -> dict[str, Any]:
        return self._metrics.summary()

    def _observation(self, target_bis: float) -> dict[str, np.ndarray]:
        assert self._patient is not None
        observation = self._state_profile.observation(
            self._history, self._patient, target_bis
        )
        if not self.observation_space.contains(observation):
            raise FloatingPointError("Constructed observation violates observation_space.")
        return observation

    def _resolve_patient(
        self, options: dict[str, Any]
    ) -> tuple[PatientDemographics, str, str]:
        explicit = options.get("patient")
        patient_id = options.get("patient_id")
        profile_name = options.get("patient_profile", "middle_male")
        selected = sum(value is not None for value in (explicit, patient_id))
        if selected > 1:
            raise ValueError("Provide only one of patient or patient_id.")
        if explicit is not None:
            if not isinstance(explicit, PatientDemographics):
                raise ValueError("options['patient'] must be PatientDemographics.")
            return explicit, str(options.get("explicit_patient_id", "explicit")), "explicit"
        if patient_id is not None:
            if self.cohort is None:
                raise ValueError("patient_id requires a PatientCohort.")
            identifier = str(patient_id)
            split = self.cohort.manifest.split_for(identifier)
            return self.cohort.patient(identifier), identifier, f"cohort:{split}"
        try:
            return self.patient_profiles[str(profile_name)], str(profile_name), str(profile_name)
        except KeyError as exc:
            raise ValueError(
                f"Unknown synthetic patient profile {profile_name!r}; "
                f"choices={sorted(self.patient_profiles)}."
            ) from exc

    @staticmethod
    def _validate_schedule(schedule: Any, name: str, method: str) -> None:
        if not callable(getattr(schedule, method, None)):
            raise ValueError(f"{name} must implement {method}(time_seconds).")

    @staticmethod
    def _ensure_finite_state(state: CombinedPatientState) -> None:
        values = [
            state.time_seconds,
            state.raw_noiseless_bis,
            state.noiseless_bis,
            state.raw_observed_bis,
            state.observed_bis,
            state.propofol_rate_mg_per_min,
            state.remifentanil_rate_micrograms_per_min,
            state.propofol.x1,
            state.propofol.x2,
            state.propofol.x3,
            state.propofol.cp,
            state.propofol.ce,
            state.propofol.cumulative_dose,
            state.remifentanil.x1,
            state.remifentanil.x2,
            state.remifentanil.x3,
            state.remifentanil.cp,
            state.remifentanil.ce,
            state.remifentanil.cumulative_dose,
        ]
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("Simulator returned a non-finite state.")

    def _info(
        self,
        state: CombinedPatientState,
        *,
        target_bis: float,
        requested_action: float,
        applied_action: float,
        action_clipped: bool,
        reward_components: dict[str, float],
    ) -> dict[str, Any]:
        return {
            "simulation_time_seconds": state.time_seconds,
            "episode_elapsed_seconds": state.time_seconds - self._episode_start_time,
            "patient_id": self._patient_id,
            "patient_profile": self._patient_profile,
            "target_bis": target_bis,
            "bis": state.observed_bis,
            "raw_observed_bis": state.raw_observed_bis,
            "noiseless_bis": state.noiseless_bis,
            "bis_noise": state.bis_noise,
            "propofol_rate_mg_per_min": state.propofol_rate_mg_per_min,
            "propofol_cp_mg_per_l": state.propofol.cp,
            "propofol_ce_mg_per_l": state.propofol.ce,
            "propofol_cumulative_dose_mg": state.propofol.cumulative_dose,
            "remifentanil_rate_micrograms_per_min": (
                state.remifentanil_rate_micrograms_per_min
            ),
            "remifentanil_cp_micrograms_per_l": state.remifentanil.cp,
            "remifentanil_ce_micrograms_per_l": state.remifentanil.ce,
            "remifentanil_cumulative_dose_micrograms": (
                state.remifentanil.cumulative_dose
            ),
            "reward_components": dict(reward_components),
            "action_requested_mg_per_min": requested_action,
            "action_applied_mg_per_min": applied_action,
            "action_clipped": action_clipped,
            "bis_in_safe_range": (
                self.config.safe_bis_low
                <= state.observed_bis
                <= self.config.safe_bis_high
            ),
            "cumulative_bis_below_40_seconds": self._below_safe_seconds,
            "cumulative_bis_above_60_seconds": self._above_safe_seconds,
            "cumulative_bis_below_safe_seconds": self._below_safe_seconds,
            "cumulative_bis_above_safe_seconds": self._above_safe_seconds,
            "history_mask": self._history.mask.copy(),
            "state_profile": self._state_profile.name,
            "dynamic_feature_names": list(self._state_profile.dynamic_feature_names),
            "action_bounds_profile": self.config.action_bounds.profile_name,
        }
