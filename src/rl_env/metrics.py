"""Training-independent episode metrics for control validation."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping

import numpy as np


class EpisodeMetricsCollector:
    """Accumulate post-transition metrics at fixed-duration decision points."""

    def __init__(
        self,
        *,
        step_duration_seconds: float,
        safe_bis_low: float,
        safe_bis_high: float,
        excessive_action_change_threshold_mg_per_min: float,
        propofol_ce_threshold_mg_per_l: float | None,
    ) -> None:
        self.step_duration_seconds = float(step_duration_seconds)
        self.safe_bis_low = float(safe_bis_low)
        self.safe_bis_high = float(safe_bis_high)
        self.excessive_action_change_threshold_mg_per_min = float(
            excessive_action_change_threshold_mg_per_min
        )
        self.propofol_ce_threshold_mg_per_l = propofol_ce_threshold_mg_per_l
        self.reset()

    def reset(self) -> None:
        self._rows: list[dict[str, float]] = []
        self._reward_component_totals: defaultdict[str, float] = defaultdict(float)
        self._terminated_reason: str | None = None
        self._numerical_failures = 0

    def record(
        self,
        *,
        time_seconds: float,
        bis: float,
        target_bis: float,
        propofol_rate_mg_per_min: float,
        previous_action_mg_per_min: float,
        propofol_cp_mg_per_l: float,
        propofol_ce_mg_per_l: float,
        remifentanil_cp_micrograms_per_l: float,
        remifentanil_ce_micrograms_per_l: float,
        reward: float,
        reward_components: Mapping[str, float],
    ) -> None:
        row = {
            "time_seconds": float(time_seconds),
            "bis": float(bis),
            "target_bis": float(target_bis),
            "propofol_rate_mg_per_min": float(propofol_rate_mg_per_min),
            "absolute_action_change_mg_per_min": abs(
                float(propofol_rate_mg_per_min) - float(previous_action_mg_per_min)
            ),
            "squared_action_change": (
                float(propofol_rate_mg_per_min) - float(previous_action_mg_per_min)
            )
            ** 2,
            "propofol_cp_mg_per_l": float(propofol_cp_mg_per_l),
            "propofol_ce_mg_per_l": float(propofol_ce_mg_per_l),
            "remifentanil_cp_micrograms_per_l": float(
                remifentanil_cp_micrograms_per_l
            ),
            "remifentanil_ce_micrograms_per_l": float(
                remifentanil_ce_micrograms_per_l
            ),
            "reward": float(reward),
        }
        if not all(math.isfinite(value) for value in row.values()):
            raise FloatingPointError("Episode metric input became non-finite.")
        self._rows.append(row)
        for name, value in reward_components.items():
            self._reward_component_totals[name] += float(value)

    def mark_numerical_failure(self, reason: str) -> None:
        self._numerical_failures += 1
        self._terminated_reason = reason

    def set_terminated_reason(self, reason: str | None) -> None:
        self._terminated_reason = reason

    def summary(self) -> dict[str, Any]:
        if not self._rows:
            return {
                "step_count": 0,
                "duration_seconds": 0.0,
                "terminated_reason": self._terminated_reason,
                "numerical_failures": self._numerical_failures,
                "reward_total": 0.0,
                "reward_component_totals": {},
            }
        bis = np.asarray([row["bis"] for row in self._rows], dtype=np.float64)
        target = np.asarray([row["target_bis"] for row in self._rows], dtype=np.float64)
        rates = np.asarray(
            [row["propofol_rate_mg_per_min"] for row in self._rows], dtype=np.float64
        )
        absolute_changes = np.asarray(
            [row["absolute_action_change_mg_per_min"] for row in self._rows],
            dtype=np.float64,
        )
        squared_changes = np.asarray(
            [row["squared_action_change"] for row in self._rows], dtype=np.float64
        )
        errors = bis - target
        in_range = (bis >= self.safe_bis_low) & (bis <= self.safe_bis_high)
        below = bis < self.safe_bis_low
        above = bis > self.safe_bis_high
        reentry_time: float | None = None
        previously_outside = False
        for row, safe in zip(self._rows, in_range):
            if not safe:
                previously_outside = True
            elif previously_outside:
                reentry_time = float(row["time_seconds"])
                break
        propofol_ce = np.asarray(
            [row["propofol_ce_mg_per_l"] for row in self._rows], dtype=np.float64
        )
        threshold_duration: float | None = None
        if self.propofol_ce_threshold_mg_per_l is not None:
            threshold_duration = float(
                np.sum(propofol_ce > self.propofol_ce_threshold_mg_per_l)
                * self.step_duration_seconds
            )
        duration = len(self._rows) * self.step_duration_seconds
        return {
            "step_count": len(self._rows),
            "duration_seconds": duration,
            "time_in_bis_40_60_seconds": float(np.sum(in_range) * self.step_duration_seconds),
            "fraction_time_in_bis_40_60": float(np.mean(in_range)),
            "bis_target_mae": float(np.mean(np.abs(errors))),
            "bis_target_rmse": float(np.sqrt(np.mean(errors**2))),
            "bis_below_40_duration_seconds": float(np.sum(below) * self.step_duration_seconds),
            "bis_above_60_duration_seconds": float(np.sum(above) * self.step_duration_seconds),
            "maximum_absolute_bis_error": float(np.max(np.abs(errors))),
            "bis_overshoot_max": float(np.max(np.abs(errors))),
            "target_reentry_time_seconds": reentry_time,
            "bis_variability_standard_deviation": float(np.std(bis)),
            "total_propofol_dose_mg": float(np.sum(rates) * self.step_duration_seconds / 60.0),
            "mean_propofol_rate_mg_per_min": float(np.mean(rates)),
            "max_propofol_rate_mg_per_min": float(np.max(rates)),
            "absolute_action_change_sum": float(np.sum(absolute_changes)),
            "squared_action_change_sum": float(np.sum(squared_changes)),
            "excessive_action_change_count": int(
                np.sum(
                    absolute_changes
                    > self.excessive_action_change_threshold_mg_per_min
                )
            ),
            "action_smoothness_mean_absolute_change": float(np.mean(absolute_changes)),
            "propofol_cp_mean_mg_per_l": self._mean("propofol_cp_mg_per_l"),
            "propofol_cp_max_mg_per_l": self._max("propofol_cp_mg_per_l"),
            "propofol_ce_mean_mg_per_l": float(np.mean(propofol_ce)),
            "propofol_ce_max_mg_per_l": float(np.max(propofol_ce)),
            "remifentanil_cp_mean_micrograms_per_l": self._mean(
                "remifentanil_cp_micrograms_per_l"
            ),
            "remifentanil_cp_max_micrograms_per_l": self._max(
                "remifentanil_cp_micrograms_per_l"
            ),
            "remifentanil_ce_mean_micrograms_per_l": self._mean(
                "remifentanil_ce_micrograms_per_l"
            ),
            "remifentanil_ce_max_micrograms_per_l": self._max(
                "remifentanil_ce_micrograms_per_l"
            ),
            "propofol_ce_threshold_mg_per_l": self.propofol_ce_threshold_mg_per_l,
            "propofol_ce_above_threshold_duration_seconds": threshold_duration,
            "terminated_reason": self._terminated_reason,
            "numerical_failures": self._numerical_failures,
            "reward_total": self._sum("reward"),
            "reward_component_totals": dict(self._reward_component_totals),
        }

    def _mean(self, name: str) -> float:
        return float(np.mean([row[name] for row in self._rows]))

    def _max(self, name: str) -> float:
        return float(np.max([row[name] for row in self._rows]))

    def _sum(self, name: str) -> float:
        return float(np.sum([row[name] for row in self._rows]))
