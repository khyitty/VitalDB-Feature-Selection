"""Numerically stable linear three-compartment/effect-site dynamics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from .parameters import DrugPKParameters
from .units import seconds_to_minutes


IntegratorName = Literal["exact", "solve_ivp"]
NEGATIVE_ROUND_OFF_TOLERANCE = 1e-10


@dataclass(frozen=True)
class DrugCompartmentState:
    """Drug amount, concentration, dose, and elimination at one time point."""

    x1: float
    x2: float
    x3: float
    cp: float
    ce: float
    cumulative_dose: float
    cumulative_eliminated: float
    current_infusion_rate: float
    amount_unit: str
    concentration_unit: str
    infusion_rate_unit: str

    @property
    def total_compartment_amount(self) -> float:
        return self.x1 + self.x2 + self.x3

    def as_dict(self, prefix: str) -> dict[str, float | str]:
        return {
            f"{prefix}_x1": self.x1,
            f"{prefix}_x2": self.x2,
            f"{prefix}_x3": self.x3,
            f"{prefix}_cp": self.cp,
            f"{prefix}_ce": self.ce,
            f"{prefix}_cumulative_dose": self.cumulative_dose,
            f"{prefix}_cumulative_eliminated": self.cumulative_eliminated,
            f"{prefix}_rate": self.current_infusion_rate,
            f"{prefix}_amount_unit": self.amount_unit,
            f"{prefix}_concentration_unit": self.concentration_unit,
            f"{prefix}_rate_unit": self.infusion_rate_unit,
        }


def zero_state(parameters: DrugPKParameters) -> DrugCompartmentState:
    return DrugCompartmentState(
        x1=0.0,
        x2=0.0,
        x3=0.0,
        cp=0.0,
        ce=0.0,
        cumulative_dose=0.0,
        cumulative_eliminated=0.0,
        current_infusion_rate=0.0,
        amount_unit=parameters.amount_unit,
        concentration_unit=parameters.concentration_unit,
        infusion_rate_unit=f"{parameters.amount_unit}/min",
    )


def system_matrix(parameters: DrugPKParameters) -> tuple[np.ndarray, np.ndarray]:
    """Return A and B for `[x1,x2,x3,Ce,eliminated]' = A x + B u`."""

    k10 = parameters.k10_per_min
    k12 = parameters.k12_per_min
    k13 = parameters.k13_per_min
    k21 = parameters.k21_per_min
    k31 = parameters.k31_per_min
    ke0 = parameters.ke0_per_min
    matrix = np.asarray(
        [
            [-(k10 + k12 + k13), k21, k31, 0.0, 0.0],
            [k12, -k21, 0.0, 0.0, 0.0],
            [k13, 0.0, -k31, 0.0, 0.0],
            [ke0 / parameters.v1_l, 0.0, 0.0, -ke0, 0.0],
            [k10, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    infusion = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return matrix, infusion


def state_vector(state: DrugCompartmentState) -> np.ndarray:
    return np.asarray(
        [state.x1, state.x2, state.x3, state.ce, state.cumulative_eliminated],
        dtype=np.float64,
    )


def _clean_physical_vector(vector: np.ndarray) -> np.ndarray:
    if vector.shape != (5,) or not np.isfinite(vector).all():
        raise FloatingPointError(f"PK-PD integration produced invalid state: {vector}")
    if float(vector.min()) < -NEGATIVE_ROUND_OFF_TOLERANCE:
        raise FloatingPointError(
            "PK-PD integration produced a physically invalid negative state: "
            f"minimum={float(vector.min())}."
        )
    result = vector.copy()
    result[(result < 0.0) & (result >= -NEGATIVE_ROUND_OFF_TOLERANCE)] = 0.0
    return result


def exact_zero_order_hold_step(
    vector: np.ndarray,
    infusion_rate_per_min: float,
    duration_seconds: float,
    parameters: DrugPKParameters,
) -> np.ndarray:
    """Integrate one constant-rate interval using an augmented matrix exponential."""

    transition, control = exact_zero_order_hold_transition(parameters, duration_seconds)
    result = transition @ vector + control * infusion_rate_per_min
    return _clean_physical_vector(result)


def exact_zero_order_hold_transition(
    parameters: DrugPKParameters,
    duration_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reusable exact state and infusion transitions for one hold interval."""

    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be finite and positive.")
    duration_min = seconds_to_minutes(duration_seconds)
    matrix, infusion = system_matrix(parameters)
    augmented = np.zeros((6, 6), dtype=np.float64)
    augmented[:5, :5] = matrix
    augmented[:5, 5] = infusion
    augmented_transition = expm(augmented * duration_min)
    return augmented_transition[:5, :5], augmented_transition[:5, 5]


def solve_ivp_reference_step(
    vector: np.ndarray,
    infusion_rate_per_min: float,
    duration_seconds: float,
    parameters: DrugPKParameters,
    *,
    rtol: float = 1e-11,
    atol: float = 1e-13,
) -> np.ndarray:
    """Independently integrate one interval for validation."""

    duration_min = seconds_to_minutes(duration_seconds)
    matrix, infusion = system_matrix(parameters)

    def derivative(_time: float, state: np.ndarray) -> np.ndarray:
        return matrix @ state + infusion * infusion_rate_per_min

    result = solve_ivp(
        derivative,
        (0.0, duration_min),
        np.asarray(vector, dtype=np.float64),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        t_eval=[duration_min],
    )
    if not result.success:
        raise RuntimeError(f"solve_ivp failed: {result.message}")
    return _clean_physical_vector(result.y[:, -1])


def advance_compartment(
    state: DrugCompartmentState,
    parameters: DrugPKParameters,
    *,
    infusion_rate_per_min: float,
    duration_seconds: float,
    integrator: IntegratorName,
) -> DrugCompartmentState:
    """Advance one drug while preserving unit and cumulative-dose metadata."""

    if not math.isfinite(infusion_rate_per_min) or infusion_rate_per_min < 0.0:
        raise ValueError("infusion_rate_per_min must be finite and non-negative.")
    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive.")
    vector = state_vector(state)
    if integrator == "exact":
        updated = exact_zero_order_hold_step(
            vector, infusion_rate_per_min, duration_seconds, parameters
        )
    elif integrator == "solve_ivp":
        updated = solve_ivp_reference_step(
            vector, infusion_rate_per_min, duration_seconds, parameters
        )
    else:
        raise ValueError(f"Unsupported integrator: {integrator!r}.")
    cumulative_dose = state.cumulative_dose + infusion_rate_per_min * seconds_to_minutes(
        duration_seconds
    )
    return DrugCompartmentState(
        x1=float(updated[0]),
        x2=float(updated[1]),
        x3=float(updated[2]),
        cp=float(updated[0] / parameters.v1_l),
        ce=float(updated[3]),
        cumulative_dose=float(cumulative_dose),
        cumulative_eliminated=float(updated[4]),
        current_infusion_rate=float(infusion_rate_per_min),
        amount_unit=parameters.amount_unit,
        concentration_unit=parameters.concentration_unit,
        infusion_rate_unit=f"{parameters.amount_unit}/min",
    )


def mass_balance_residual(state: DrugCompartmentState) -> float:
    """Return dose minus amounts retained and eliminated for a zero initial state."""

    return state.cumulative_dose - (
        state.total_compartment_amount + state.cumulative_eliminated
    )
