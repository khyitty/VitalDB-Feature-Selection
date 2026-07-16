"""Transparent post-transition BIS reward profiles."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .config import EnvironmentConfig


@dataclass(frozen=True)
class RewardResult:
    total: float
    components: dict[str, float]


class RewardCalculator:
    """Compute state-profile-invariant reward from the post-action snapshot."""

    def __init__(self, config: EnvironmentConfig) -> None:
        self.config = config

    def calculate(
        self,
        *,
        post_bis: float,
        target_bis: float,
        action_mg_per_min: float,
        previous_action_mg_per_min: float,
        propofol_ce_mg_per_l: float,
    ) -> RewardResult:
        values = (
            post_bis,
            target_bis,
            action_mg_per_min,
            previous_action_mg_per_min,
            propofol_ce_mg_per_l,
        )
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("Reward inputs must all be finite.")
        if self.config.reward_profile == "paper_yun2023_parameterized":
            assert self.config.paper_reward_alpha is not None
            components = {
                "paper_yun2023_equation40": 1.0
                / (abs(target_bis - post_bis) + self.config.paper_reward_alpha)
            }
        else:
            action_scale = self.config.action_bounds.high_mg_per_min
            tracking = -abs(post_bis - target_bis) / 50.0
            deep = -max(self.config.safe_bis_low - post_bis, 0.0) / 20.0
            inadequate = -max(post_bis - self.config.safe_bis_high, 0.0) / 20.0
            magnitude = -self.config.action_magnitude_coefficient * (
                action_mg_per_min / action_scale
            ) ** 2
            change = -self.config.action_change_coefficient * (
                (action_mg_per_min - previous_action_mg_per_min) / action_scale
            ) ** 2
            concentration = 0.0
            threshold = self.config.propofol_ce_threshold_mg_per_l
            if threshold is not None:
                concentration = -self.config.concentration_safety_coefficient * max(
                    propofol_ce_mg_per_l - threshold, 0.0
                ) ** 2
            components = {
                "target_tracking": tracking,
                "deep_hypnosis": deep,
                "inadequate_hypnosis": inadequate,
                "action_magnitude": magnitude,
                "action_change": change,
                "concentration_safety": concentration,
            }
        total = float(sum(components.values()))
        if not math.isfinite(total):
            raise FloatingPointError("Reward became non-finite.")
        return RewardResult(total=total, components=components)


def reward_profile_registry() -> dict[str, Any]:
    return {
        "reward_timing": "post-action BIS at t+1 for each 10-second transition",
        "profiles": {
            "transparent_tracking_v1": {
                "source": "repository design",
                "formula": (
                    "-|BIS-target|/50 - max(40-BIS,0)/20 - "
                    "max(BIS-60,0)/20 plus explicitly configured optional penalties"
                ),
                "default_optional_coefficients": 0.0,
            },
            "paper_yun2023_parameterized": {
                "source": "Yun 2023 PDF p.5 Eq. (40)",
                "formula": "1 / (abs(target - BIS(t+1)) + alpha)",
                "exact_reproduction": False,
                "reason": (
                    "The numeric alpha value is not reported; callers must provide "
                    "an explicit positive value."
                ),
            },
        },
    }
