"""Build identical-dynamics environments for every PPO condition."""

from __future__ import annotations

from src.rl_env import EnvironmentConfig, PropofolControlEnv, action_bounds_from_profile

from .action_wrapper import NormalizedPropofolActionWrapper
from .cohort import CohortBundle, CohortScenarioWrapper
from .config import PPOConfig, PolicyCondition, environment_profile_for_condition


def make_cohort_environment(
    *,
    condition: PolicyCondition,
    ppo: PPOConfig,
    cohort: CohortBundle,
    split: str,
    seed: int,
    cycle: bool = False,
) -> NormalizedPropofolActionWrapper:
    bounds = action_bounds_from_profile(ppo.action_bounds_profile)
    config = EnvironmentConfig(
        episode_duration_seconds=ppo.episode_duration_seconds,
        deterministic=ppo.deterministic_simulator,
        action_bounds=bounds,
        action_mode="strict",
        state_profile=environment_profile_for_condition(condition),  # type: ignore[arg-type]
        reward_profile=ppo.reward_profile,
        action_magnitude_coefficient=ppo.action_magnitude_coefficient,
        action_change_coefficient=ppo.action_change_coefficient,
    )
    base = PropofolControlEnv(config, cohort=cohort.cohort)
    scenario = CohortScenarioWrapper(
        base,
        bundle=cohort,
        split=split,
        base_seed=seed,
        episode_duration_seconds=ppo.episode_duration_seconds,
        cycle=cycle,
    )
    return NormalizedPropofolActionWrapper(scenario, bounds)
