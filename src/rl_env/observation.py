"""Gymnasium spaces and an optional deterministic flattening wrapper."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .state_adapters import StateProfile


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
