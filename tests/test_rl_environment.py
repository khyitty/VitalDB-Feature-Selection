"""Focused API, timing, integrity, and metric tests for Module 5."""

from __future__ import annotations

import ast
from dataclasses import replace
import json
import math
from pathlib import Path

from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
import numpy as np
import pandas as pd
import pytest

from src.pkpd import ConstantSchedule, PatientDemographics
from src.rl_env import (
    CohortManifest,
    EnvironmentConfig,
    PatientCohort,
    PropofolControlEnv,
    SYNTHETIC_NONCLINICAL_ACTION_BOUNDS,
    YUN_2023_CONVERTED_ACTION_BOUNDS,
)
from src.rl_env.action import validate_action
from src.rl_env.metrics import EpisodeMetricsCollector
from src.rl_env.reward import RewardCalculator, reward_profile_registry
from src.rl_env.state_adapters import (
    ALL_SUPPORTED_FEATURES,
    EXCLUDED_LATENT_STATES,
    ORIGINAL_YUN_FEATURES,
    SELECTED_CONTROL_AWARE_FEATURES,
    STATE_PROFILES,
    UNSUPPORTED_PREDICTIVE_FEATURES,
    UNSUPPORTED_VITAL_SIGNS,
    state_profile_registry,
)
from src.rl_env.validation import ValidationConfig, compare_state_profiles, run_validation


def _config(**kwargs: object) -> EnvironmentConfig:
    values: dict[str, object] = {
        "episode_duration_seconds": 60.0,
        "action_bounds": SYNTHETIC_NONCLINICAL_ACTION_BOUNDS,
    }
    values.update(kwargs)
    return EnvironmentConfig(**values)  # type: ignore[arg-type]


def _env(**kwargs: object) -> PropofolControlEnv:
    return PropofolControlEnv(_config(**kwargs))


def test_reset_returns_observation_and_info() -> None:
    observation, info = _env().reset(seed=7)
    assert set(observation) == {"history", "history_mask", "static", "target_bis"}
    assert info["simulation_time_seconds"] == 0.0


def test_step_returns_gymnasium_five_tuple() -> None:
    env = _env()
    env.reset(seed=7)
    result = env.step(np.asarray([6.0], dtype=np.float32))
    assert len(result) == 5
    assert isinstance(result[1], float)
    assert result[2:4] == (False, False)


@pytest.mark.parametrize("profile", tuple(STATE_PROFILES))
def test_observation_space_contains_every_profile_observation(profile: str) -> None:
    env = _env(state_profile=profile)
    observation, _ = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    for _ in range(2):
        observation, *_ = env.step(np.asarray([4.0], dtype=np.float32))
        assert env.observation_space.contains(observation)


def test_action_space_is_scalar_float32_box_with_traceable_bounds() -> None:
    env = _env()
    assert isinstance(env.action_space, spaces.Box)
    assert env.action_space.shape == (1,)
    assert env.action_space.dtype == np.float32
    assert float(env.action_space.low[0]) == 0.0
    assert float(env.action_space.high[0]) == 12.0


def test_yun_action_bound_is_converted_from_mg_per_ten_seconds() -> None:
    assert YUN_2023_CONVERTED_ACTION_BOUNDS.high_mg_per_min == pytest.approx(27.7 * 6.0)
    assert "converted" in YUN_2023_CONVERTED_ACTION_BOUNDS.profile_name


def test_gymnasium_env_checker_passes() -> None:
    check_env(_env(episode_duration_seconds=20.0), skip_render_check=True)


def test_seed_reproducibility_for_stochastic_simulator() -> None:
    def trajectory(seed: int) -> list[float]:
        env = _env(deterministic=False)
        _, info = env.reset(seed=seed)
        values = [info["raw_observed_bis"]]
        for _ in range(3):
            _, _, _, _, info = env.step(np.asarray([5.0], dtype=np.float32))
            values.append(info["raw_observed_bis"])
        return values

    assert trajectory(42) == trajectory(42)
    assert trajectory(42) != trajectory(43)


def test_one_step_advances_exactly_ten_seconds() -> None:
    env = _env()
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.asarray([2.0], dtype=np.float32))
    assert info["simulation_time_seconds"] == 10.0


def test_simulator_internal_step_is_fixed_at_one_second() -> None:
    env = _env()
    env.reset(seed=1)
    assert env.simulator.internal_dt_seconds == 1.0
    with pytest.raises(ValueError, match="internal_dt_seconds"):
        _config(internal_dt_seconds=2.0)


def test_six_history_steps_are_fixed_to_sixty_seconds() -> None:
    config = _config()
    assert config.history_steps == 6
    assert config.history_steps * config.action_interval_seconds == 60.0
    with pytest.raises(ValueError, match="history_window_seconds"):
        _config(history_window_seconds=30.0)


def test_action_post_bis_reward_alignment_has_no_off_by_one() -> None:
    env = _env()
    _, reset_info = env.reset(seed=1)
    _, reward, _, _, info = env.step(np.asarray([6.0], dtype=np.float32))
    assert info["bis"] != reset_info["bis"]
    expected = RewardCalculator(env.config).calculate(
        post_bis=info["bis"],
        target_bis=info["target_bis"],
        action_mg_per_min=6.0,
        previous_action_mg_per_min=0.0,
        propofol_ce_mg_per_l=info["propofol_ce_mg_per_l"],
    )
    assert reward == expected.total


def test_fixed_episode_duration_is_exact_and_truncated() -> None:
    env = _env(episode_duration_seconds=30.0)
    env.reset(seed=1)
    endings = [env.step(np.asarray([0.0], dtype=np.float32))[2:4] for _ in range(3)]
    assert endings == [(False, False), (False, False), (False, True)]
    assert env.simulator.snapshot().time_seconds == 30.0
    with pytest.raises(RuntimeError, match="after episode completion"):
        env.step(np.asarray([0.0], dtype=np.float32))


def test_original_reconstructed_ordered_schema() -> None:
    assert (
        STATE_PROFILES["original_reconstructed"].dynamic_feature_names
        == ORIGINAL_YUN_FEATURES
    )
    assert len(ORIGINAL_YUN_FEATURES) == 7


def test_all_supported_ordered_schema() -> None:
    assert STATE_PROFILES["all_supported"].dynamic_feature_names == ALL_SUPPORTED_FEATURES
    assert ALL_SUPPORTED_FEATURES[0:3] == ("bis", "bis_delta_10s", "bis_target_error")


def test_attention_ready_history_and_static_shapes() -> None:
    observation, _ = _env(state_profile="attention_ready").reset(seed=1)
    assert observation["history"].shape == (6, len(ALL_SUPPORTED_FEATURES))
    assert observation["static"].shape == (4,)


def test_legacy_control_aware_mapping_is_explicit() -> None:
    registry = state_profile_registry()
    assert STATE_PROFILES["legacy_control_aware"].dynamic_feature_names == (
        SELECTED_CONTROL_AWARE_FEATURES
    )
    assert registry["predictive_intersection"] == ["bis", "bis_slope", "ppf_rate", "ppf_cp"]
    assert set(registry["unsupported_predictive_features_removed"]) == set(
        UNSUPPORTED_PREDICTIVE_FEATURES
    )


def test_unsupported_vitals_and_bis_sqi_are_absent() -> None:
    all_names = {name.lower() for profile in STATE_PROFILES.values() for name in profile.dynamic_feature_names}
    assert not {name.lower() for name in UNSUPPORTED_VITAL_SIGNS}.intersection(all_names)
    assert "bis_sqi" not in all_names


def test_internal_x1_x2_x3_are_absent_from_observable_profiles() -> None:
    all_names = {name for profile in STATE_PROFILES.values() for name in profile.dynamic_feature_names}
    assert not set(EXCLUDED_LATENT_STATES).intersection(all_names)


def test_all_and_attention_ready_contain_identical_raw_information() -> None:
    assert STATE_PROFILES["all_supported"].dynamic_feature_names == (
        STATE_PROFILES["attention_ready"].dynamic_feature_names
    )
    left, _ = _env(state_profile="all_supported").reset(seed=1)
    right, _ = _env(state_profile="attention_ready").reset(seed=1)
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def test_state_switch_does_not_change_dynamics_or_reward() -> None:
    frame, summary = compare_state_profiles(
        ValidationConfig(episode_duration_seconds=30.0)
    )
    assert summary["passed"] is True
    assert frame["maximum_absolute_difference_vs_original_yun"].max() == 0.0


def test_target_bis_has_best_transparent_reward() -> None:
    calculator = RewardCalculator(_config())
    rewards = [
        calculator.calculate(
            post_bis=value,
            target_bis=50.0,
            action_mg_per_min=0.0,
            previous_action_mg_per_min=0.0,
            propofol_ce_mg_per_l=0.0,
        ).total
        for value in (40.0, 50.0, 60.0)
    ]
    assert rewards[1] > rewards[0] == rewards[2]


def test_reward_decreases_monotonically_with_distance_and_outside_range() -> None:
    calculator = RewardCalculator(_config())

    def value(bis: float) -> float:
        return calculator.calculate(
            post_bis=bis,
            target_bis=50.0,
            action_mg_per_min=0.0,
            previous_action_mg_per_min=0.0,
            propofol_ce_mg_per_l=0.0,
        ).total

    assert value(50) > value(55) > value(60) > value(70)
    assert value(30) < value(40) < value(50)


def test_reward_is_finite_and_component_sum_equals_total() -> None:
    result = RewardCalculator(_config()).calculate(
        post_bis=25.0,
        target_bis=50.0,
        action_mg_per_min=12.0,
        previous_action_mg_per_min=0.0,
        propofol_ce_mg_per_l=10.0,
    )
    assert math.isfinite(result.total)
    assert result.total == pytest.approx(sum(result.components.values()))


def test_action_penalties_only_apply_when_enabled() -> None:
    disabled = RewardCalculator(_config()).calculate(
        post_bis=50.0,
        target_bis=50.0,
        action_mg_per_min=12.0,
        previous_action_mg_per_min=0.0,
        propofol_ce_mg_per_l=0.0,
    )
    enabled = RewardCalculator(
        _config(action_magnitude_coefficient=1.0, action_change_coefficient=1.0)
    ).calculate(
        post_bis=50.0,
        target_bis=50.0,
        action_mg_per_min=12.0,
        previous_action_mg_per_min=0.0,
        propofol_ce_mg_per_l=0.0,
    )
    assert disabled.components["action_magnitude"] == 0.0
    assert disabled.components["action_change"] == 0.0
    assert enabled.total < disabled.total


def test_paper_reward_requires_alpha_and_is_traceable() -> None:
    with pytest.raises(ValueError, match="explicit positive alpha"):
        _config(reward_profile="paper_yun2023_parameterized")
    config = _config(
        reward_profile="paper_yun2023_parameterized", paper_reward_alpha=1.0
    )
    result = RewardCalculator(config).calculate(
        post_bis=49.0,
        target_bis=50.0,
        action_mg_per_min=0.0,
        previous_action_mg_per_min=0.0,
        propofol_ce_mg_per_l=0.0,
    )
    assert result.total == 0.5
    registry = reward_profile_registry()["profiles"]["paper_yun2023_parameterized"]
    assert registry["exact_reproduction"] is False
    assert "Eq. (40)" in registry["source"]


@pytest.mark.parametrize("bad_action", [np.nan, np.inf, -1.0])
def test_nonfinite_and_negative_actions_are_rejected(bad_action: float) -> None:
    with pytest.raises(ValueError):
        validate_action(
            bad_action,
            bounds=SYNTHETIC_NONCLINICAL_ACTION_BOUNDS,
            mode="strict",
        )


def test_out_of_bound_policy_is_strict_by_default_and_clip_is_explicit() -> None:
    with pytest.raises(ValueError, match="outside"):
        validate_action(13.0, bounds=SYNTHETIC_NONCLINICAL_ACTION_BOUNDS, mode="strict")
    clipped = validate_action(
        13.0, bounds=SYNTHETIC_NONCLINICAL_ACTION_BOUNDS, mode="clip"
    )
    assert clipped.applied_mg_per_min == 12.0
    assert clipped.clipped is True


def test_clipping_is_never_silent_in_environment_info() -> None:
    env = _env(action_mode="clip")
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.asarray([13.0], dtype=np.float32))
    assert info["action_requested_mg_per_min"] == 13.0
    assert info["action_applied_mg_per_min"] == 12.0
    assert info["action_clipped"] is True


def test_nonfinite_simulator_state_is_detected() -> None:
    env = _env()
    env.reset(seed=1)
    corrupted = replace(env.simulator.snapshot(), observed_bis=float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite"):
        env._ensure_finite_state(corrupted)


def test_patient_overlap_in_split_manifest_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        CohortManifest(("p1",), ("p1",), ("p2",))


def test_valid_cohort_patient_reset_reports_split() -> None:
    manifest = CohortManifest(("p1",), ("p2",), ("p3",))
    patients = {
        name: PatientDemographics(40, "male", 177, 77)
        for name in ("p1", "p2", "p3")
    }
    env = PropofolControlEnv(_config(), cohort=PatientCohort(patients, manifest))
    _, info = env.reset(seed=1, options={"patient_id": "p2"})
    assert info["patient_id"] == "p2"
    assert info["patient_profile"] == "cohort:validation"


def test_history_mask_and_causal_updates_do_not_use_future_values() -> None:
    env = _env()
    initial, _ = env.reset(seed=1)
    assert initial["history_mask"].tolist() == [0, 0, 0, 0, 0, 1]
    first, *_ = env.step(np.asarray([2.0], dtype=np.float32))
    frozen_first = first["history"].copy()
    second, *_ = env.step(np.asarray([10.0], dtype=np.float32))
    assert first["history_mask"].tolist() == [0, 0, 0, 0, 1, 1]
    np.testing.assert_array_equal(frozen_first[-1], second["history"][-2])


def test_recent_dose_uses_only_the_causal_sixty_second_window() -> None:
    env = _env(state_profile="all_supported", episode_duration_seconds=70.0)
    env.reset(seed=1)
    observation = None
    for _ in range(7):
        observation, *_ = env.step(np.asarray([6.0], dtype=np.float32))
    assert observation is not None
    feature_index = ALL_SUPPORTED_FEATURES.index("propofol_recent_dose_mg")
    assert observation["history"][-1, feature_index] == pytest.approx(6.0)


def test_identical_remifentanil_schedule_across_profiles() -> None:
    values = []
    for profile in STATE_PROFILES:
        env = PropofolControlEnv(
            _config(state_profile=profile), remifentanil_schedule=ConstantSchedule(5.0)
        )
        env.reset(seed=1)
        _, _, _, _, info = env.step(np.asarray([4.0], dtype=np.float32))
        values.append(
            (
                info["remifentanil_rate_micrograms_per_min"],
                info["remifentanil_ce_micrograms_per_l"],
            )
        )
    assert values.count(values[0]) == len(values)


def _collector() -> EpisodeMetricsCollector:
    collector = EpisodeMetricsCollector(
        step_duration_seconds=10.0,
        safe_bis_low=40.0,
        safe_bis_high=60.0,
        excessive_action_change_threshold_mg_per_min=4.0,
        propofol_ce_threshold_mg_per_l=None,
    )
    for index, (bis, action, previous) in enumerate(
        [(50.0, 6.0, 0.0), (30.0, 6.0, 6.0), (70.0, 6.0, 6.0)], start=1
    ):
        collector.record(
            time_seconds=index * 10.0,
            bis=bis,
            target_bis=50.0,
            propofol_rate_mg_per_min=action,
            previous_action_mg_per_min=previous,
            propofol_cp_mg_per_l=float(index),
            propofol_ce_mg_per_l=float(index),
            remifentanil_cp_micrograms_per_l=float(index),
            remifentanil_ce_micrograms_per_l=float(index),
            reward=-1.0,
            reward_components={"tracking": -1.0},
        )
    return collector


def test_time_in_range_and_unsafe_duration_metrics() -> None:
    summary = _collector().summary()
    assert summary["time_in_bis_40_60_seconds"] == 10.0
    assert summary["bis_below_40_duration_seconds"] == 10.0
    assert summary["bis_above_60_duration_seconds"] == 10.0


def test_bis_mae_and_rmse_metrics() -> None:
    summary = _collector().summary()
    assert summary["bis_target_mae"] == pytest.approx(40.0 / 3.0)
    assert summary["bis_target_rmse"] == pytest.approx(math.sqrt(800.0 / 3.0))


def test_action_smoothness_and_total_dose_metrics() -> None:
    summary = _collector().summary()
    assert summary["absolute_action_change_sum"] == 6.0
    assert summary["squared_action_change_sum"] == 36.0
    assert summary["excessive_action_change_count"] == 1
    assert summary["total_propofol_dose_mg"] == 3.0


def test_validation_writes_complete_training_free_package(tmp_path: Path) -> None:
    output = tmp_path / "rl_validation"
    summary = run_validation(
        ValidationConfig(episode_duration_seconds=30.0),
        output,
        Path(__file__).parents[1],
    )
    assert summary["status"] == "passed"
    required = {
        "rl_environment_manifest.json",
        "state_profile_registry.json",
        "reward_profile_registry.json",
        "source_traceability.json",
        "rollout_trajectory.csv",
        "episode_metrics.json",
        "profile_equivalence.csv",
        "validation_summary.json",
        "rl_environment_validation_report.md",
    }
    assert required.issubset({path.name for path in output.iterdir()})
    assert len(list((output / "figures").glob("*.png"))) == 6
    manifest = json.loads((output / "rl_environment_manifest.json").read_text())
    assert manifest["simulator_commit"].startswith("faf636a")
    assert manifest["rl_training_performed"] is False
    assert manifest["not_connected_to_real_pump_or_patient"] is True
    assert "not a medical device" in manifest["clinical_use_prohibition"]
    equivalence = pd.read_csv(output / "profile_equivalence.csv")
    assert equivalence["maximum_absolute_difference_vs_original_yun"].max() == 0.0


def test_colab_validation_notebook_is_clean_and_contains_no_training() -> None:
    path = Path(__file__).parents[1] / "notebooks/colab_rl_environment_validation.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{path}:cell-{index}")
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
    assert "validate_rl_environment.py" in source
    assert "drive.mount" not in source
    assert "stable_baselines" not in source.lower()
    assert "PPO(" not in source
    assert "train(" not in source
