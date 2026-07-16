"""Target-BIS schedules and remifentanil schedule re-exports."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, Sequence

from src.pkpd.schedules import (
    CSVTrajectorySchedule,
    CallableSchedule,
    ConstantSchedule,
    PiecewiseConstantSchedule,
    RateSchedule,
    RateSegment,
)


class TargetSchedule(Protocol):
    def target_at(self, time_seconds: float) -> float: ...


def _validate_target(value: float) -> float:
    target = float(value)
    if not math.isfinite(target) or not 0.0 <= target <= 100.0:
        raise ValueError("Target BIS must be finite and within [0, 100].")
    return target


@dataclass(frozen=True)
class ConstantTargetSchedule:
    target_bis: float = 50.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_bis", _validate_target(self.target_bis))

    def target_at(self, time_seconds: float) -> float:
        if not math.isfinite(time_seconds) or time_seconds < 0.0:
            raise ValueError("Schedule time must be finite and non-negative.")
        return self.target_bis


@dataclass(frozen=True)
class TargetSegment:
    start_seconds: float
    end_seconds: float
    target_bis: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.start_seconds)
            or not math.isfinite(self.end_seconds)
            or self.start_seconds < 0.0
            or self.end_seconds <= self.start_seconds
        ):
            raise ValueError("Target segments require finite 0 <= start < end.")
        object.__setattr__(self, "target_bis", _validate_target(self.target_bis))


@dataclass(frozen=True, init=False)
class PiecewiseTargetSchedule:
    segments: tuple[TargetSegment, ...]
    fallback_target_bis: float

    def __init__(
        self,
        segments: Sequence[TargetSegment],
        fallback_target_bis: float = 50.0,
    ) -> None:
        object.__setattr__(self, "segments", tuple(segments))
        object.__setattr__(self, "fallback_target_bis", fallback_target_bis)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("Piecewise target schedule requires at least one segment.")
        object.__setattr__(
            self, "fallback_target_bis", _validate_target(self.fallback_target_bis)
        )
        for previous, current in zip(self.segments, self.segments[1:]):
            if current.start_seconds < previous.end_seconds:
                raise ValueError("Target schedule segments overlap or are unordered.")

    def target_at(self, time_seconds: float) -> float:
        if not math.isfinite(time_seconds) or time_seconds < 0.0:
            raise ValueError("Schedule time must be finite and non-negative.")
        for segment in self.segments:
            if segment.start_seconds <= time_seconds < segment.end_seconds:
                return segment.target_bis
        return self.fallback_target_bis


__all__ = [
    "CSVTrajectorySchedule",
    "CallableSchedule",
    "ConstantSchedule",
    "ConstantTargetSchedule",
    "PiecewiseConstantSchedule",
    "PiecewiseTargetSchedule",
    "RateSchedule",
    "RateSegment",
    "TargetSchedule",
    "TargetSegment",
]
