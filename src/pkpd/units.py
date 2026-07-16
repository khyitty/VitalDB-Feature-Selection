"""Explicit PK-PD unit conversions."""

from __future__ import annotations

import math


SECONDS_PER_MINUTE = 60.0
MICROGRAMS_PER_MG = 1000.0


def seconds_to_minutes(seconds: float) -> float:
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError(f"seconds must be finite and non-negative; received {seconds}.")
    return float(seconds) / SECONDS_PER_MINUTE


def milligrams_to_micrograms(milligrams: float) -> float:
    if not math.isfinite(milligrams):
        raise ValueError("milligrams must be finite.")
    return float(milligrams) * MICROGRAMS_PER_MG


def micrograms_to_milligrams(micrograms: float) -> float:
    if not math.isfinite(micrograms):
        raise ValueError("micrograms must be finite.")
    return float(micrograms) / MICROGRAMS_PER_MG


def propofol_mg_per_l_to_micrograms_per_ml(value: float) -> float:
    """These units are numerically equal."""

    return float(value)


def remifentanil_micrograms_per_l_to_ng_per_ml(value: float) -> float:
    """These units are numerically equal."""

    return float(value)
