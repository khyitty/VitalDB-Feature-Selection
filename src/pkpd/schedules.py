"""Exogenous remifentanil infusion schedules with explicit units."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Protocol, Sequence

import pandas as pd


REMIFENTANIL_RATE_UNIT = "microgram/min"


class RateSchedule(Protocol):
    unit: str

    def rate_at(self, time_seconds: float) -> float: ...


def _validate_rate(rate: float) -> float:
    value = float(rate)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"Infusion rate must be finite and non-negative: {rate}.")
    return value


def _validate_unit(unit: str) -> None:
    if unit != REMIFENTANIL_RATE_UNIT:
        raise ValueError(
            f"Remifentanil schedule unit must be {REMIFENTANIL_RATE_UNIT!r}; got {unit!r}."
        )


@dataclass(frozen=True)
class ConstantSchedule:
    rate_micrograms_per_min: float
    unit: str = REMIFENTANIL_RATE_UNIT

    def __post_init__(self) -> None:
        _validate_unit(self.unit)
        object.__setattr__(
            self, "rate_micrograms_per_min", _validate_rate(self.rate_micrograms_per_min)
        )

    def rate_at(self, time_seconds: float) -> float:
        if not math.isfinite(time_seconds) or time_seconds < 0.0:
            raise ValueError("Schedule time must be finite and non-negative.")
        return self.rate_micrograms_per_min


@dataclass(frozen=True)
class RateSegment:
    start_seconds: float
    end_seconds: float
    rate_micrograms_per_min: float

    def __post_init__(self) -> None:
        start = float(self.start_seconds)
        end = float(self.end_seconds)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end <= start:
            raise ValueError("Schedule segments require finite 0 <= start < end.")
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)
        object.__setattr__(
            self, "rate_micrograms_per_min", _validate_rate(self.rate_micrograms_per_min)
        )


@dataclass(frozen=True)
class PiecewiseConstantSchedule:
    segments: tuple[RateSegment, ...]
    unit: str = REMIFENTANIL_RATE_UNIT

    def __init__(
        self, segments: Sequence[RateSegment], unit: str = REMIFENTANIL_RATE_UNIT
    ) -> None:
        object.__setattr__(self, "segments", tuple(segments))
        object.__setattr__(self, "unit", unit)
        self.__post_init__()

    def __post_init__(self) -> None:
        _validate_unit(self.unit)
        if not self.segments:
            raise ValueError("Piecewise schedule requires at least one segment.")
        for previous, current in zip(self.segments, self.segments[1:]):
            if current.start_seconds < previous.end_seconds:
                raise ValueError("Piecewise schedule segments overlap or are unordered.")

    def rate_at(self, time_seconds: float) -> float:
        if not math.isfinite(time_seconds) or time_seconds < 0.0:
            raise ValueError("Schedule time must be finite and non-negative.")
        for segment in self.segments:
            if segment.start_seconds <= time_seconds < segment.end_seconds:
                return segment.rate_micrograms_per_min
        return 0.0


@dataclass(frozen=True)
class CallableSchedule:
    function: Callable[[float], float]
    unit: str = REMIFENTANIL_RATE_UNIT

    def __post_init__(self) -> None:
        _validate_unit(self.unit)

    def rate_at(self, time_seconds: float) -> float:
        if not math.isfinite(time_seconds) or time_seconds < 0.0:
            raise ValueError("Schedule time must be finite and non-negative.")
        return _validate_rate(self.function(float(time_seconds)))


@dataclass(frozen=True)
class CSVTrajectorySchedule:
    """Monotonic step-wise CSV adapter; no VitalDB trajectory is loaded automatically."""

    times_seconds: tuple[float, ...]
    rates_micrograms_per_min: tuple[float, ...]
    unit: str = REMIFENTANIL_RATE_UNIT

    @classmethod
    def from_csv(
        cls,
        path: Path,
        *,
        time_column: str = "time_seconds",
        rate_column: str = "remifentanil_rate_micrograms_per_min",
        unit: str = REMIFENTANIL_RATE_UNIT,
    ) -> "CSVTrajectorySchedule":
        frame = pd.read_csv(path)
        missing = [name for name in (time_column, rate_column) if name not in frame]
        if missing or frame.empty:
            raise ValueError(f"CSV schedule is empty or missing columns: {missing}")
        return cls(
            times_seconds=tuple(frame[time_column].to_numpy(float)),
            rates_micrograms_per_min=tuple(frame[rate_column].to_numpy(float)),
            unit=unit,
        )

    def __post_init__(self) -> None:
        _validate_unit(self.unit)
        if not self.times_seconds or len(self.times_seconds) != len(
            self.rates_micrograms_per_min
        ):
            raise ValueError("CSV schedule times/rates must be non-empty and equally sized.")
        if self.times_seconds[0] < 0.0 or any(
            not math.isfinite(value) for value in self.times_seconds
        ):
            raise ValueError("CSV schedule times must be finite and non-negative.")
        if any(right <= left for left, right in zip(self.times_seconds, self.times_seconds[1:])):
            raise ValueError("CSV schedule times must be strictly increasing.")
        for rate in self.rates_micrograms_per_min:
            _validate_rate(rate)

    def rate_at(self, time_seconds: float) -> float:
        if not math.isfinite(time_seconds) or time_seconds < 0.0:
            raise ValueError("Schedule time must be finite and non-negative.")
        selected = 0.0
        for start, rate in zip(self.times_seconds, self.rates_micrograms_per_min):
            if start > time_seconds:
                break
            selected = rate
        return float(selected)
