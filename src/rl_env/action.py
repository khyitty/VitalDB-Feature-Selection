"""Strict, unit-explicit propofol action handling."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .config import ActionBounds, ActionMode


@dataclass(frozen=True)
class ValidatedAction:
    requested_mg_per_min: float
    applied_mg_per_min: float
    clipped: bool


def validate_action(
    action: Any,
    *,
    bounds: ActionBounds,
    mode: ActionMode,
) -> ValidatedAction:
    """Validate a scalar/one-element action without silent clipping."""

    array = np.asarray(action, dtype=np.float64)
    if array.size != 1:
        raise ValueError(f"Propofol action must contain exactly one value; shape={array.shape}.")
    requested = float(array.reshape(-1)[0])
    if not math.isfinite(requested):
        raise ValueError("Propofol action must be finite.")
    if requested < 0.0:
        raise ValueError("Propofol action must be non-negative.")
    outside = requested < bounds.low_mg_per_min or requested > bounds.high_mg_per_min
    if outside and mode == "strict":
        raise ValueError(
            f"Propofol action {requested} mg/min is outside "
            f"[{bounds.low_mg_per_min}, {bounds.high_mg_per_min}] mg/min."
        )
    applied = float(np.clip(requested, bounds.low_mg_per_min, bounds.high_mg_per_min))
    return ValidatedAction(requested, applied, applied != requested)
