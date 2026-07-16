"""Action clipping diagnostics and lightweight PPO progress collection."""

from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class PPOProgressCallback(BaseCallback):
    """Record SB3 raw-vs-clipped normalized actions and finite progress values."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.normalized_action_count = 0
        self.normalized_clipping_count = 0
        self.maximum_normalized_clipping = 0.0
        self.raw_action_minimum = np.inf
        self.raw_action_maximum = -np.inf
        self.bounded_action_minimum = np.inf
        self.bounded_action_maximum = -np.inf
        self.rollout_rows: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        raw = np.asarray(self.locals.get("actions", []), dtype=float)
        clipped = np.asarray(self.locals.get("clipped_actions", raw), dtype=float)
        if raw.size:
            difference = np.abs(raw - clipped)
            self.normalized_action_count += int(raw.size)
            self.normalized_clipping_count += int(np.count_nonzero(difference > 0.0))
            self.maximum_normalized_clipping = max(
                self.maximum_normalized_clipping, float(difference.max(initial=0.0))
            )
            self.raw_action_minimum = min(self.raw_action_minimum, float(raw.min()))
            self.raw_action_maximum = max(self.raw_action_maximum, float(raw.max()))
            self.bounded_action_minimum = min(
                self.bounded_action_minimum, float(clipped.min())
            )
            self.bounded_action_maximum = max(
                self.bounded_action_maximum, float(clipped.max())
            )
        return True

    def _on_rollout_end(self) -> None:
        values = self.model.logger.name_to_value
        self.rollout_rows.append(
            {
                "timesteps": self.num_timesteps,
                "rollout_mean_reward": float(values.get("rollout/ep_rew_mean", np.nan)),
                "train_loss": float(values.get("train/loss", np.nan)),
                "policy_gradient_loss": float(values.get("train/policy_gradient_loss", np.nan)),
                "value_loss": float(values.get("train/value_loss", np.nan)),
            }
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "normalized_action_count": self.normalized_action_count,
            "normalized_clipping_count": self.normalized_clipping_count,
            "normalized_clipping_fraction": (
                self.normalized_clipping_count / self.normalized_action_count
                if self.normalized_action_count
                else 0.0
            ),
            "maximum_normalized_clipping": self.maximum_normalized_clipping,
            "raw_normalized_action_minimum": (
                self.raw_action_minimum if self.normalized_action_count else None
            ),
            "raw_normalized_action_maximum": (
                self.raw_action_maximum if self.normalized_action_count else None
            ),
            "bounded_normalized_action_minimum": (
                self.bounded_action_minimum if self.normalized_action_count else None
            ),
            "bounded_normalized_action_maximum": (
                self.bounded_action_maximum if self.normalized_action_count else None
            ),
        }
