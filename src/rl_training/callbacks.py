"""Action clipping diagnostics and lightweight PPO progress collection."""

from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.rl_env.config import ActionBounds


class PPOProgressCallback(BaseCallback):
    """Record SB3 raw-vs-clipped normalized actions and finite progress values."""

    def __init__(self, bounds: ActionBounds | None = None) -> None:
        super().__init__(verbose=0)
        self.bounds = bounds
        self.normalized_action_count = 0
        self.normalized_clipping_count = 0
        self.maximum_normalized_clipping = 0.0
        self.raw_action_minimum = np.inf
        self.raw_action_maximum = -np.inf
        self.bounded_action_minimum = np.inf
        self.bounded_action_maximum = -np.inf
        self.lower_clipping_count = 0
        self.upper_clipping_count = 0
        self.near_lower_boundary_count = 0
        self.near_upper_boundary_count = 0
        self.raw_histogram_edges = np.linspace(-4.0, 4.0, 33)
        self.raw_histogram_counts = np.zeros(32, dtype=np.int64)
        self.raw_histogram_below = 0
        self.raw_histogram_above = 0
        self.episode_rows: list[dict[str, Any]] = []
        self._episode_actions = 0
        self._episode_lower = 0
        self._episode_upper = 0
        self._window_start_actions = 0
        self._window_start_clips = 0
        self.rollout_rows: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        raw = np.asarray(self.locals.get("actions", []), dtype=float)
        clipped = np.asarray(self.locals.get("clipped_actions", raw), dtype=float)
        if raw.size:
            difference = np.abs(raw - clipped)
            self.normalized_action_count += int(raw.size)
            self.normalized_clipping_count += int(np.count_nonzero(difference > 0.0))
            lower = raw < -1.0
            upper = raw > 1.0
            self.lower_clipping_count += int(np.count_nonzero(lower))
            self.upper_clipping_count += int(np.count_nonzero(upper))
            self.near_lower_boundary_count += int(np.count_nonzero(clipped <= -0.95))
            self.near_upper_boundary_count += int(np.count_nonzero(clipped >= 0.95))
            histogram, _ = np.histogram(raw, bins=self.raw_histogram_edges)
            self.raw_histogram_counts += histogram
            self.raw_histogram_below += int(np.count_nonzero(raw < self.raw_histogram_edges[0]))
            self.raw_histogram_above += int(np.count_nonzero(raw > self.raw_histogram_edges[-1]))
            self._episode_actions += int(raw.size)
            self._episode_lower += int(np.count_nonzero(lower))
            self._episode_upper += int(np.count_nonzero(upper))
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
        dones = np.asarray(self.locals.get("dones", []), dtype=bool)
        if dones.any() and self._episode_actions:
            clipped_count = self._episode_lower + self._episode_upper
            self.episode_rows.append(
                {
                    "episode_index": len(self.episode_rows),
                    "action_count": self._episode_actions,
                    "clipping_count": clipped_count,
                    "clipping_rate": clipped_count / self._episode_actions,
                    "lower_clipping_count": self._episode_lower,
                    "upper_clipping_count": self._episode_upper,
                }
            )
            self._episode_actions = 0
            self._episode_lower = 0
            self._episode_upper = 0
        return True

    def _on_rollout_end(self) -> None:
        values = self.model.logger.name_to_value
        episode_returns = [
            float(info["r"])
            for info in self.model.ep_info_buffer
            if "r" in info
        ]
        window_actions = self.normalized_action_count - self._window_start_actions
        window_clips = self.normalized_clipping_count - self._window_start_clips
        self.rollout_rows.append(
            {
                "timesteps": self.num_timesteps,
                "rollout_mean_reward": (
                    float(np.mean(episode_returns)) if episode_returns else np.nan
                ),
                "train_loss": float(values.get("train/loss", np.nan)),
                "policy_gradient_loss": float(values.get("train/policy_gradient_loss", np.nan)),
                "value_loss": float(values.get("train/value_loss", np.nan)),
                "action_count": window_actions,
                "clipping_count": window_clips,
                "clipping_rate": window_clips / window_actions if window_actions else 0.0,
            }
        )
        self._window_start_actions = self.normalized_action_count
        self._window_start_clips = self.normalized_clipping_count

    def diagnostics(self) -> dict[str, Any]:
        return {
            "normalized_action_count": self.normalized_action_count,
            "normalized_clipping_count": self.normalized_clipping_count,
            "lower_bound_clipping_count": self.lower_clipping_count,
            "upper_bound_clipping_count": self.upper_clipping_count,
            "normalized_clipping_fraction": (
                self.normalized_clipping_count / self.normalized_action_count
                if self.normalized_action_count
                else 0.0
            ),
            "maximum_normalized_clipping": self.maximum_normalized_clipping,
            "near_lower_boundary_fraction": (
                self.near_lower_boundary_count / self.normalized_action_count
                if self.normalized_action_count
                else 0.0
            ),
            "near_upper_boundary_fraction": (
                self.near_upper_boundary_count / self.normalized_action_count
                if self.normalized_action_count
                else 0.0
            ),
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
            "raw_action_histogram": {
                "edges": self.raw_histogram_edges.tolist(),
                "counts": self.raw_histogram_counts.tolist(),
                "below_range": self.raw_histogram_below,
                "above_range": self.raw_histogram_above,
            },
            "physical_action_bounds_mg_per_min": (
                [self.bounds.low_mg_per_min, self.bounds.high_mg_per_min]
                if self.bounds is not None
                else None
            ),
            "diagnosis_contract": (
                "SB3 raw Gaussian action is compared with its Box-clipped action before "
                "the wrapper performs one affine physical-unit conversion."
            ),
        }
