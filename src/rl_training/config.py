"""Frozen policy-condition and PPO research configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal


PolicyCondition = Literal[
    "yun_reconstructed",
    "all_supported",
    "attention_supported",
    "selected_control_aware",
]
PrimaryStateProfile = Literal[
    "original_reconstructed",
    "all_supported",
    "prediction_minimal",
    "selected_control_core",
    "selected",
]
PRIMARY_STATE_PROFILES: tuple[PrimaryStateProfile, ...] = (
    "original_reconstructed",
    "all_supported",
    "prediction_minimal",
    "selected_control_core",
    "selected",
)

POLICY_CONDITIONS: tuple[PolicyCondition, ...] = (
    "yun_reconstructed",
    "all_supported",
    "attention_supported",
    "selected_control_aware",
)
EXPERIMENT_SEEDS = (7, 21, 42, 84, 123)


@dataclass(frozen=True)
class PPOConfig:
    """Prespecified SB3 PPO settings; these are repository design choices."""

    profile_name: str = "ppo_research_v1"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    latent_dim: int = 64
    policy_hidden_dim: int = 64
    total_timesteps: int = 1_024_000
    evaluation_frequency_timesteps: int = 51_200
    evaluation_episode_count: int = 15
    episode_duration_seconds: float = 1800.0
    action_bounds_profile: str = "synthetic_nonclinical_v1"
    reward_profile: str = "transparent_tracking_v1"
    action_magnitude_coefficient: float = 0.0
    action_change_coefficient: float = 0.0
    deterministic_simulator: bool = True

    def __post_init__(self) -> None:
        if self.profile_name not in (
            "ppo_research_v1",
            "ppo_primary_state_pilot_v1",
        ):
            raise ValueError(
                "PPO profile_name must identify the full or primary-state pilot protocol."
            )
        if self.n_steps <= 0 or self.batch_size <= 0 or self.n_epochs <= 0:
            raise ValueError("PPO rollout and optimization sizes must be positive.")
        if self.n_steps % self.batch_size != 0:
            raise ValueError("n_steps must be divisible by batch_size for one environment.")
        if self.total_timesteps <= 0 or self.evaluation_frequency_timesteps <= 0:
            raise ValueError("Training and evaluation intervals must be positive.")
        if self.total_timesteps % self.evaluation_frequency_timesteps != 0:
            raise ValueError("total_timesteps must be divisible by evaluation frequency.")
        if self.episode_duration_seconds % 10.0 != 0.0:
            raise ValueError("Episode duration must be an exact multiple of 10 seconds.")
        bounded = (
            self.learning_rate,
            self.gamma,
            self.gae_lambda,
            self.clip_range,
            self.max_grad_norm,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in bounded):
            raise ValueError("PPO floating-point hyperparameters must be finite and positive.")
        if self.reward_profile != "transparent_tracking_v1":
            raise ValueError(
                "The frozen main comparison uses transparent_tracking_v1; paper reward "
                "requires a separately versioned sensitivity protocol."
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def smoke_ppo_config(total_timesteps: int = 2048) -> PPOConfig:
    """Return a short CPU contract test, never a performance experiment."""

    evaluation_frequency = total_timesteps
    return PPOConfig(
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        total_timesteps=total_timesteps,
        evaluation_frequency_timesteps=evaluation_frequency,
        evaluation_episode_count=1,
        episode_duration_seconds=120.0,
    )


def primary_smoke_ppo_config(total_timesteps: int = 1_000) -> PPOConfig:
    """Return an exact-step common-MLP smoke configuration."""

    if total_timesteps <= 0 or total_timesteps % 100 != 0:
        raise ValueError("Primary smoke timesteps must be a positive multiple of 100.")
    return PPOConfig(
        n_steps=100,
        batch_size=20,
        n_epochs=2,
        total_timesteps=total_timesteps,
        evaluation_frequency_timesteps=total_timesteps,
        evaluation_episode_count=1,
        episode_duration_seconds=120.0,
    )


def environment_profile_for_condition(condition: PolicyCondition) -> str:
    return {
        "yun_reconstructed": "yun_reconstructed",
        "all_supported": "all_supported",
        "attention_supported": "attention_ready",
        "selected_control_aware": "selected_control_aware",
    }[condition]
