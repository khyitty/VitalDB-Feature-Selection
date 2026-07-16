"""Invertible normalized-policy to physical-propofol action mapping."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.rl_env.config import ActionBounds


@dataclass(frozen=True)
class ActionTransform:
    policy_action: float
    bounded_policy_action: float
    physical_action_mg_per_min: float
    applied_dose_mg_per_10s: float
    normalized_clipping_applied: bool
    lower_bound_clipping: bool = False
    upper_bound_clipping: bool = False


def policy_to_physical(action: Any, bounds: ActionBounds) -> ActionTransform:
    array = np.asarray(action, dtype=np.float64)
    if array.size != 1:
        raise ValueError("Normalized policy action must contain exactly one value.")
    raw = float(array.reshape(-1)[0])
    if not math.isfinite(raw):
        raise ValueError("Normalized policy action must be finite.")
    if raw < -1.0 or raw > 1.0:
        raise ValueError(f"Normalized policy action {raw} is outside [-1, 1].")
    physical = bounds.low_mg_per_min + (raw + 1.0) * 0.5 * (
        bounds.high_mg_per_min - bounds.low_mg_per_min
    )
    return ActionTransform(
        policy_action=raw,
        bounded_policy_action=raw,
        physical_action_mg_per_min=float(physical),
        applied_dose_mg_per_10s=float(physical * 10.0 / 60.0),
        normalized_clipping_applied=False,
    )


def physical_to_policy(physical_mg_per_min: float, bounds: ActionBounds) -> float:
    value = float(physical_mg_per_min)
    if not math.isfinite(value):
        raise ValueError("Physical action must be finite.")
    if value < bounds.low_mg_per_min or value > bounds.high_mg_per_min:
        raise ValueError("Physical action is outside the configured action bounds.")
    return float(
        2.0
        * (value - bounds.low_mg_per_min)
        / (bounds.high_mg_per_min - bounds.low_mg_per_min)
        - 1.0
    )


class NormalizedPropofolActionWrapper(gym.ActionWrapper):
    """Expose `[-1,1]` to PPO and pass only strict-valid mg/min to Module 5."""

    def __init__(self, env: gym.Env, bounds: ActionBounds) -> None:
        super().__init__(env)
        self.bounds = bounds
        self.action_space = spaces.Box(
            low=np.asarray([-1.0], dtype=np.float32),
            high=np.asarray([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.last_transform: ActionTransform | None = None

    def action(self, action: np.ndarray) -> np.ndarray:
        self.last_transform = policy_to_physical(action, self.bounds)
        return np.asarray(
            [self.last_transform.physical_action_mg_per_min], dtype=np.float32
        )

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = super().step(action)
        assert self.last_transform is not None
        info = dict(info)
        info.update(
            {
                "policy_raw_action": self.last_transform.policy_action,
                "policy_wrapper_received_action": self.last_transform.policy_action,
                "sb3_unbounded_raw_action_available_via_callback": True,
                "normalized_action": self.last_transform.bounded_policy_action,
                "action_before_wrapper_clipping": self.last_transform.policy_action,
                "action_after_wrapper_clipping": self.last_transform.bounded_policy_action,
                "policy_bounded_action": self.last_transform.bounded_policy_action,
                "scaled_environment_action_mg_per_min": (
                    self.last_transform.physical_action_mg_per_min
                ),
                "physical_action_mg_per_min": (
                    self.last_transform.physical_action_mg_per_min
                ),
                "applied_dose_mg_per_10s": self.last_transform.applied_dose_mg_per_10s,
                "normalized_clipping_applied": (
                    self.last_transform.normalized_clipping_applied
                ),
                "lower_bound_clipping": self.last_transform.lower_bound_clipping,
                "upper_bound_clipping": self.last_transform.upper_bound_clipping,
            }
        )
        return observation, reward, terminated, truncated, info

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        array = np.asarray(action, dtype=np.float64)
        if array.size != 1:
            raise ValueError("Physical action must contain exactly one value.")
        return np.asarray(
            [physical_to_policy(float(array.reshape(-1)[0]), self.bounds)],
            dtype=np.float32,
        )
