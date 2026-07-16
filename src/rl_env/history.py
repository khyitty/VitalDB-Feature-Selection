"""Causal decision-point history with explicit reset padding."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields
import math

import numpy as np

from src.pkpd.simulator import CombinedPatientState


@dataclass(frozen=True)
class HistoryRecord:
    """Observable and causally derived values at one decision point."""

    time_seconds: float
    bis: float
    noiseless_bis: float
    bis_slope: float
    bis_target_error: float
    propofol_rate_mg_per_min: float
    propofol_interval_dose_mg: float
    propofol_recent_dose_mg: float
    propofol_cumulative_dose_mg: float
    propofol_cp_mg_per_l: float
    propofol_ce_mg_per_l: float
    remifentanil_rate_micrograms_per_min: float
    remifentanil_interval_dose_micrograms: float
    remifentanil_recent_dose_micrograms: float
    remifentanil_cumulative_dose_micrograms: float
    remifentanil_cp_micrograms_per_l: float
    remifentanil_ce_micrograms_per_l: float

    def value(self, name: str) -> float:
        if name not in {item.name for item in fields(self)}:
            raise KeyError(f"History feature {name!r} is unavailable.")
        return float(getattr(self, name))


class HistoryBuffer:
    """Store six ordered 10-second rows without creating a fictitious past."""

    def __init__(self, *, history_steps: int, action_interval_seconds: float) -> None:
        if history_steps <= 0:
            raise ValueError("history_steps must be positive.")
        if action_interval_seconds <= 0.0:
            raise ValueError("action_interval_seconds must be positive.")
        self.history_steps = int(history_steps)
        self.action_interval_seconds = float(action_interval_seconds)
        self._records: deque[HistoryRecord] = deque(maxlen=self.history_steps)
        self._mask: deque[int] = deque(maxlen=self.history_steps)
        self._actual: deque[HistoryRecord] = deque(maxlen=self.history_steps + 1)

    @property
    def records(self) -> tuple[HistoryRecord, ...]:
        if len(self._records) != self.history_steps:
            raise RuntimeError("HistoryBuffer must be reset before use.")
        return tuple(self._records)

    @property
    def mask(self) -> np.ndarray:
        if len(self._mask) != self.history_steps:
            raise RuntimeError("HistoryBuffer must be reset before use.")
        return np.asarray(self._mask, dtype=np.int8)

    def reset(self, state: CombinedPatientState, *, target_bis: float) -> None:
        record = self._record_from_state(
            state,
            target_bis=target_bis,
            previous=None,
            window_start=None,
        )
        self._records = deque([record] * self.history_steps, maxlen=self.history_steps)
        self._mask = deque(
            [0] * (self.history_steps - 1) + [1], maxlen=self.history_steps
        )
        self._actual = deque([record], maxlen=self.history_steps + 1)

    def append(self, state: CombinedPatientState, *, target_bis: float) -> HistoryRecord:
        if not self._actual:
            raise RuntimeError("HistoryBuffer must be reset before append().")
        previous = self._actual[-1]
        expected = previous.time_seconds + self.action_interval_seconds
        if not math.isclose(state.time_seconds, expected, abs_tol=1e-9):
            raise ValueError(
                "History rows must be appended at exact decision intervals: "
                f"expected {expected}, observed {state.time_seconds}."
            )
        window_start = (
            self._actual[1]
            if len(self._actual) == self.history_steps + 1
            else self._actual[0]
        )
        record = self._record_from_state(
            state,
            target_bis=target_bis,
            previous=previous,
            window_start=window_start,
        )
        self._actual.append(record)
        self._records.append(record)
        self._mask.append(1)
        return record

    def matrix(self, feature_names: tuple[str, ...]) -> np.ndarray:
        values = [
            [record.value(feature_name) for feature_name in feature_names]
            for record in self.records
        ]
        result = np.asarray(values, dtype=np.float32)
        if not np.isfinite(result).all():
            raise FloatingPointError("History observation contains NaN or infinity.")
        return result

    @staticmethod
    def _record_from_state(
        state: CombinedPatientState,
        *,
        target_bis: float,
        previous: HistoryRecord | None,
        window_start: HistoryRecord | None,
    ) -> HistoryRecord:
        if previous is None:
            bis_slope = 0.0
            propofol_interval = 0.0
            remifentanil_interval = 0.0
        else:
            bis_slope = state.observed_bis - previous.bis
            propofol_interval = (
                state.propofol.cumulative_dose - previous.propofol_cumulative_dose_mg
            )
            remifentanil_interval = (
                state.remifentanil.cumulative_dose
                - previous.remifentanil_cumulative_dose_micrograms
            )
        if window_start is None:
            propofol_recent = 0.0
            remifentanil_recent = 0.0
        else:
            propofol_recent = (
                state.propofol.cumulative_dose
                - window_start.propofol_cumulative_dose_mg
            )
            remifentanil_recent = (
                state.remifentanil.cumulative_dose
                - window_start.remifentanil_cumulative_dose_micrograms
            )
        values = np.asarray(
            [
                state.time_seconds,
                state.observed_bis,
                state.noiseless_bis,
                bis_slope,
                state.observed_bis - target_bis,
                state.propofol_rate_mg_per_min,
                propofol_interval,
                propofol_recent,
                state.propofol.cumulative_dose,
                state.propofol.cp,
                state.propofol.ce,
                state.remifentanil_rate_micrograms_per_min,
                remifentanil_interval,
                remifentanil_recent,
                state.remifentanil.cumulative_dose,
                state.remifentanil.cp,
                state.remifentanil.ce,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise FloatingPointError("Cannot append a non-finite simulator state to history.")
        return HistoryRecord(*map(float, values))
