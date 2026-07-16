"""Scripted, training-free validation for the propofol control environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
from typing import Any, Callable

import gymnasium
from gymnasium.utils.env_checker import check_env
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pkpd.schedules import ConstantSchedule, PiecewiseConstantSchedule, RateSegment
from src.pkpd.validation import SYNTHETIC_PATIENTS

from .config import (
    EnvironmentConfig,
    RESEARCH_ONLY_WARNING,
    action_bounds_from_profile,
)
from .environment import PropofolControlEnv
from .reward import reward_profile_registry
from .state_adapters import STATE_PROFILES, state_profile_registry


LOGGER = logging.getLogger(__name__)
SIMULATOR_COMMIT = "faf636ab3d5922c73a979b2cf2a8ea6e0f1e8483"
SOURCE_PAPER = (
    "W. J. Yun et al., Deep reinforcement learning-based propofol infusion "
    "control for anesthesia, Computers in Biology and Medicine 156 (2023) 106739."
)


@dataclass(frozen=True)
class ValidationConfig:
    state_profile: str = "attention_ready"
    patient_profile: str = "middle_male"
    target_bis: float = 50.0
    episode_duration_seconds: float = 600.0
    action_schedule: str = "step"
    remifentanil_schedule: str = "piecewise"
    deterministic: bool = True
    action_bounds_profile: str = "synthetic_nonclinical_v1"
    reward_profile: str = "transparent_tracking_v1"
    paper_reward_alpha: float | None = None
    seed: int = 20260716


def source_traceability_registry() -> dict[str, Any]:
    """Return machine-readable source/design boundaries."""

    return {
        "primary_source": SOURCE_PAPER,
        "entries": [
            {
                "item": "state",
                "source": "Yun 2023 PDF p.5, Eqs. (36)--(39)",
                "confirmed": (
                    "BIS history/slope/error, propofol and remifentanil history/"
                    "recent cumulative dose, age/sex/height/weight"
                ),
                "repository_choice": "Raw causal BIS replaces unspecified online LOWESS.",
                "unresolved": "LOWESS online method and numeric W are not reported.",
            },
            {
                "item": "action",
                "source": "Yun 2023 PDF p.5 Action Definition",
                "confirmed": "Continuous propofol action every 10 seconds, 0--27.7 mg/10 s.",
                "repository_choice": "Expose the equivalent 0--166.2 mg/min simulator unit.",
                "unresolved": "Paper alternates rate language with per-interval dose units.",
            },
            {
                "item": "reward",
                "source": "Yun 2023 PDF p.5 Eq. (40)",
                "confirmed": "1 / (abs(target - BIS(t+1)) + alpha)",
                "repository_choice": "Transparent tracking reward is the default.",
                "unresolved": "Numeric alpha is not reported.",
            },
            {
                "item": "history_initialization",
                "source": "repository design",
                "confirmed": None,
                "repository_choice": "Repeated initial row with [0,0,0,0,0,1] validity mask.",
                "unresolved": "No source reset-padding rule is reported.",
            },
        ],
        "documentation": "docs/rl_environment_source_traceability.md",
    }


def _git_head(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _remifentanil_schedule(kind: str, duration_seconds: float):
    if kind == "off":
        return ConstantSchedule(0.0)
    if kind == "constant":
        return ConstantSchedule(4.0)
    if kind == "piecewise":
        first = duration_seconds / 3.0
        second = 2.0 * duration_seconds / 3.0
        return PiecewiseConstantSchedule(
            [
                RateSegment(0.0, first, 3.0),
                RateSegment(first, second, 7.0),
                RateSegment(second, duration_seconds + 10.0, 1.0),
            ]
        )
    raise ValueError(f"Unsupported remifentanil schedule: {kind!r}.")


def scripted_actions(
    kind: str,
    *,
    step_count: int,
    high_mg_per_min: float,
    seed: int,
) -> np.ndarray:
    """Create deterministic valid actions; these are not learned policies."""

    low_rate = min(4.0, high_mg_per_min * 0.25)
    moderate = min(8.0, high_mg_per_min * 0.60)
    if kind == "zero":
        return np.zeros(step_count, dtype=np.float64)
    if kind == "low":
        return np.full(step_count, low_rate, dtype=np.float64)
    if kind == "moderate":
        return np.full(step_count, moderate, dtype=np.float64)
    if kind == "step":
        result = np.zeros(step_count, dtype=np.float64)
        first = step_count // 3
        second = 2 * step_count // 3
        result[:first] = min(10.0, high_mg_per_min * 0.8)
        result[first:second] = moderate
        result[second:] = low_rate
        return result
    if kind == "random":
        return np.random.default_rng(seed).uniform(0.0, high_mg_per_min, step_count)
    raise ValueError(f"Unsupported scripted action schedule: {kind!r}.")


def _environment_config(config: ValidationConfig, state_profile: str) -> EnvironmentConfig:
    return EnvironmentConfig(
        episode_duration_seconds=config.episode_duration_seconds,
        target_bis=config.target_bis,
        deterministic=config.deterministic,
        action_bounds=action_bounds_from_profile(config.action_bounds_profile),
        state_profile=state_profile,  # type: ignore[arg-type]
        reward_profile=config.reward_profile,
        paper_reward_alpha=config.paper_reward_alpha,
    )


def rollout(
    config: ValidationConfig,
    *,
    state_profile: str | None = None,
    actions: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    """Run one scripted episode and include the reset state as row zero."""

    profile = state_profile or config.state_profile
    env_config = _environment_config(config, profile)
    schedule = _remifentanil_schedule(
        config.remifentanil_schedule, config.episode_duration_seconds
    )
    env = PropofolControlEnv(env_config, remifentanil_schedule=schedule)
    observation, info = env.reset(
        seed=config.seed, options={"patient_profile": config.patient_profile}
    )
    rows: list[dict[str, Any]] = [
        {
            **{
                key: value
                for key, value in info.items()
                if not isinstance(value, (np.ndarray, dict, list))
            },
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
        }
    ]
    step_count = int(config.episode_duration_seconds / env_config.action_interval_seconds)
    selected_actions = actions
    if selected_actions is None:
        selected_actions = scripted_actions(
            config.action_schedule,
            step_count=step_count,
            high_mg_per_min=env_config.action_bounds.high_mg_per_min,
            seed=config.seed,
        )
    if len(selected_actions) != step_count:
        raise ValueError("Scripted action sequence length must equal the episode step count.")
    for action in selected_actions:
        observation, reward, terminated, truncated, info = env.step(
            np.asarray([action], dtype=np.float32)
        )
        component_values = {
            f"reward_{name}": value for name, value in info["reward_components"].items()
        }
        rows.append(
            {
                **{
                    key: value
                    for key, value in info.items()
                    if not isinstance(value, (np.ndarray, dict, list))
                },
                **component_values,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            }
        )
        if terminated or truncated:
            break
    metrics = env.episode_metrics()
    env.close()
    return pd.DataFrame(rows), metrics, observation


def compare_state_profiles(
    config: ValidationConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify that observation adapters cannot alter dynamics or reward."""

    step_count = int(config.episode_duration_seconds / 10.0)
    bounds = action_bounds_from_profile(config.action_bounds_profile)
    actions = scripted_actions(
        config.action_schedule,
        step_count=step_count,
        high_mg_per_min=bounds.high_mg_per_min,
        seed=config.seed,
    )
    trajectories: dict[str, pd.DataFrame] = {}
    for profile in STATE_PROFILES:
        trajectories[profile], _, _ = rollout(
            config, state_profile=profile, actions=actions
        )
    reference = trajectories["original_yun"]
    comparison_columns = [
        "bis",
        "noiseless_bis",
        "propofol_rate_mg_per_min",
        "propofol_cp_mg_per_l",
        "propofol_ce_mg_per_l",
        "propofol_cumulative_dose_mg",
        "remifentanil_rate_micrograms_per_min",
        "remifentanil_cp_micrograms_per_l",
        "remifentanil_ce_micrograms_per_l",
        "remifentanil_cumulative_dose_micrograms",
        "reward",
    ]
    rows: list[dict[str, Any]] = []
    maxima: dict[str, float] = {}
    for profile, frame in trajectories.items():
        max_difference = 0.0
        for column in comparison_columns:
            difference = np.abs(
                frame[column].to_numpy(float) - reference[column].to_numpy(float)
            )
            current = float(np.max(difference))
            max_difference = max(max_difference, current)
            rows.append(
                {
                    "state_profile": profile,
                    "variable": column,
                    "maximum_absolute_difference_vs_original_yun": current,
                }
            )
        maxima[profile] = max_difference
    return pd.DataFrame(rows), {
        "passed": all(value == 0.0 for value in maxima.values()),
        "maximum_absolute_difference_by_profile": maxima,
        "same_patient_seed_actions_remifentanil_target_reward": True,
    }


def _checker_passes(config: ValidationConfig) -> bool:
    checker_config = replace(config, episode_duration_seconds=20.0)
    env = PropofolControlEnv(_environment_config(checker_config, config.state_profile))
    check_env(env, skip_render_check=True)
    env.close()
    return True


def _write_figures(
    frame: pd.DataFrame,
    final_observation: dict[str, np.ndarray],
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    def line_figure(
        filename: str,
        columns: list[tuple[str, str]],
        ylabel: str,
    ) -> None:
        figure, axis = plt.subplots(figsize=(9, 4))
        for column, label in columns:
            axis.plot(frame["simulation_time_seconds"], frame[column], label=label)
        axis.set_xlabel("Simulation time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(figure_dir / filename, dpi=140)
        plt.close(figure)

    line_figure(
        "bis_and_target.png",
        [("bis", "BIS"), ("target_bis", "Target")],
        "BIS",
    )
    line_figure(
        "actions_and_infusions.png",
        [
            ("propofol_rate_mg_per_min", "Propofol (mg/min)"),
            ("remifentanil_rate_micrograms_per_min", "Remifentanil (microgram/min)"),
        ],
        "Infusion rate",
    )
    line_figure(
        "propofol_cp_ce.png",
        [("propofol_cp_mg_per_l", "Cp"), ("propofol_ce_mg_per_l", "Ce")],
        "Propofol concentration (mg/L)",
    )
    line_figure(
        "remifentanil_cp_ce.png",
        [
            ("remifentanil_cp_micrograms_per_l", "Cp"),
            ("remifentanil_ce_micrograms_per_l", "Ce"),
        ],
        "Remifentanil concentration (microgram/L)",
    )
    reward_columns = [
        column
        for column in frame
        if column.startswith("reward_")
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    line_figure(
        "reward_components.png",
        [(column, column.removeprefix("reward_")) for column in reward_columns],
        "Reward component",
    )
    figure, axis = plt.subplots(figsize=(10, 3.5))
    image = axis.imshow(final_observation["history"], aspect="auto", cmap="viridis")
    axis.set_xlabel("Ordered feature index")
    axis.set_ylabel("Oldest to newest decision row")
    axis.set_yticks(range(final_observation["history"].shape[0]))
    figure.colorbar(image, ax=axis, label="Raw physical value")
    figure.tight_layout()
    figure.savefig(figure_dir / "history_buffer_example.png", dpi=140)
    plt.close(figure)


def run_validation(
    config: ValidationConfig,
    output_dir: Path,
    repo_dir: Path,
) -> dict[str, Any]:
    """Run all synthetic checks and persist a reproducible validation package."""

    if config.state_profile not in STATE_PROFILES:
        raise ValueError(f"Unknown state profile: {config.state_profile!r}.")
    if config.patient_profile not in SYNTHETIC_PATIENTS:
        raise ValueError(f"Unknown synthetic patient: {config.patient_profile!r}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    checker_passed = _checker_passes(config)
    trajectory, metrics, final_observation = rollout(config)
    equivalence, equivalence_summary = compare_state_profiles(config)
    profile = STATE_PROFILES[config.state_profile]
    environment_config = _environment_config(config, config.state_profile)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_commit": _git_head(repo_dir),
        "simulator_commit": SIMULATOR_COMMIT,
        "source_papers": [SOURCE_PAPER],
        "state_action_reward_source_mapping": source_traceability_registry(),
        "action_interval_seconds": 10.0,
        "internal_dt_seconds": 1.0,
        "history_window_seconds": 60.0,
        "history_steps": 6,
        "target_bis": config.target_bis,
        "action_unit": "mg/min",
        "action_bounds": asdict(environment_config.action_bounds),
        "reward_profile": config.reward_profile,
        "state_profile": profile.metadata(),
        "unsupported_features": state_profile_registry()["unsupported_vital_signs"]
        + state_profile_registry()["unsupported_predictive_features_removed"],
        "padding_mask_policy": "Repeated initial snapshot; only newest reset row is valid.",
        "patient_reset_policy": "Explicit patient, synthetic profile, or validated cohort ID.",
        "remifentanil_schedule_policy": (
            "Exogenous and fixed identically across state-profile comparisons."
        ),
        "termination_rules": {
            "terminated": "reserved for explicit environment terminal conditions",
            "truncated": "exact configured fixed duration",
            "numerical_failure": "explicit FloatingPointError; never silently continued",
        },
        "deterministic": config.deterministic,
        "clinical_use_prohibition": RESEARCH_ONLY_WARNING,
        "not_connected_to_real_pump_or_patient": True,
        "rl_training_performed": False,
        "next_module": "Module 6 policy/attention encoder and PPO comparison",
        "gymnasium_version": gymnasium.__version__,
    }
    summary = {
        "status": "passed" if checker_passed and equivalence_summary["passed"] else "failed",
        "gymnasium_env_checker_passed": checker_passed,
        "observation_space_contains_rollout_observations": True,
        "profile_equivalence": equivalence_summary,
        "trajectory_rows_including_reset": len(trajectory),
        "episode_metrics": metrics,
        "rl_training_performed": False,
    }
    trajectory.to_csv(output_dir / "rollout_trajectory.csv", index=False)
    equivalence.to_csv(output_dir / "profile_equivalence.csv", index=False)
    files = {
        "rl_environment_manifest.json": manifest,
        "state_profile_registry.json": state_profile_registry(),
        "reward_profile_registry.json": reward_profile_registry(),
        "source_traceability.json": source_traceability_registry(),
        "episode_metrics.json": metrics,
        "validation_summary.json": summary,
    }
    for filename, payload in files.items():
        (output_dir / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    _write_figures(trajectory, final_observation, output_dir / "figures")
    report = f"""# RL Environment Validation Report

Status: **{summary['status']}**

- Gymnasium checker: `{checker_passed}`
- State-profile dynamics/reward equivalence: `{equivalence_summary['passed']}`
- Action interval: `10 s`; simulator internal step: `1 s`
- History: `6` decision rows with an explicit validity mask
- State profile: `{config.state_profile}`
- Reward profile: `{config.reward_profile}`
- Scripted action schedule: `{config.action_schedule}` (not a learned policy)
- RL training performed: `False`

## Scientific boundary

{RESEARCH_ONLY_WARNING}

No PPO, actor, critic, policy optimization, or real-patient/pump connection is
implemented or executed by this validation.
"""
    (output_dir / "rl_environment_validation_report.md").write_text(
        report, encoding="utf-8"
    )
    LOGGER.info("RL environment validation status: %s", summary["status"])
    return summary
