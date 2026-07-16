"""Gymnasium spaces and an optional deterministic flattening wrapper."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .state_adapters import StateProfile
from .state_manifests import fixed_policy_scale


def make_observation_space(
    profile: StateProfile,
    *,
    history_steps: int,
) -> spaces.Dict:
    """Create a raw-physical structured observation space."""

    return spaces.Dict(
        {
            "history": spaces.Box(
                low=np.float32(-1.0e9),
                high=np.float32(1.0e9),
                shape=(history_steps, len(profile.dynamic_feature_names)),
                dtype=np.float32,
            ),
            "history_mask": spaces.Box(
                low=0,
                high=1,
                shape=(history_steps,),
                dtype=np.int8,
            ),
            "static": spaces.Box(
                low=np.float32(-1.0e9),
                high=np.float32(1.0e9),
                shape=(len(profile.static_feature_names),),
                dtype=np.float32,
            ),
            "target_bis": spaces.Box(
                low=np.float32(0.0),
                high=np.float32(100.0),
                shape=(1,),
                dtype=np.float32,
            ),
        }
    )


class FlattenObservationAdapter(gym.ObservationWrapper):
    """Flatten a structured state without fitting normalization statistics."""

    def __init__(self, env: gym.Env[dict[str, np.ndarray], np.ndarray]) -> None:
        super().__init__(env)
        self.observation_space = spaces.flatten_space(env.observation_space)

    def observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray(
            spaces.flatten(self.env.observation_space, observation), dtype=np.float32
        )

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        return super().reset(**kwargs)


class ScaledFlattenObservationAdapter(gym.ObservationWrapper):
    """Apply declared fixed physical scales and flatten in a stable order."""

    def __init__(
        self,
        env: gym.Env[dict[str, np.ndarray], np.ndarray],
        profile: StateProfile,
    ) -> None:
        super().__init__(env)
        self.profile = profile
        self.dynamic_scales = np.asarray(
            [fixed_policy_scale(name) for name in profile.dynamic_feature_names],
            dtype=np.float32,
        )
        self.static_scales = np.asarray(
            [fixed_policy_scale(name) for name in profile.static_feature_names],
            dtype=np.float32,
        )
        dimension = profile.observation_dimension()
        self.observation_space = spaces.Box(
            low=np.full(dimension, -1.0e9, dtype=np.float32),
            high=np.full(dimension, 1.0e9, dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        history = observation["history"] / self.dynamic_scales
        static = observation["static"] / self.static_scales
        flattened = np.concatenate(
            (
                history.reshape(-1),
                observation["history_mask"].astype(np.float32),
                static,
                observation["target_bis"] / 100.0,
            )
        ).astype(np.float32)
        if flattened.shape != self.observation_space.shape or not np.isfinite(flattened).all():
            raise FloatingPointError("Scaled flattened observation violates its contract.")
        return flattened
