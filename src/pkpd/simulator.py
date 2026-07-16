"""Pure PK-PD simulator API, intentionally independent of Gymnasium and RL."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .bis_response import BISResponse, evaluate_bis
from .compartment import (
    DrugCompartmentState,
    IntegratorName,
    advance_compartment,
    zero_state,
)
from .demographics import PatientDemographics
from .parameters import (
    DrugPKParameters,
    minto_remifentanil_parameters,
    schnider_propofol_parameters,
)
from .schedules import RateSchedule


@dataclass(frozen=True)
class CombinedPatientState:
    """Complete simulator-supported state at one instant."""

    time_seconds: float
    propofol: DrugCompartmentState
    remifentanil: DrugCompartmentState
    raw_noiseless_bis: float
    noiseless_bis: float
    bis_noise: float
    raw_observed_bis: float
    observed_bis: float
    bis_clipping_applied: bool
    demographics: PatientDemographics
    propofol_rate_mg_per_min: float
    remifentanil_rate_micrograms_per_min: float
    remifentanil_schedule_kind: str
    deterministic: bool
    integrator: IntegratorName

    def as_dict(self) -> dict[str, Any]:
        return {
            "time_seconds": self.time_seconds,
            **self.demographics.as_dict(),
            **self.propofol.as_dict("propofol"),
            **self.remifentanil.as_dict("remifentanil"),
            "raw_noiseless_bis": self.raw_noiseless_bis,
            "noiseless_bis": self.noiseless_bis,
            "bis_noise": self.bis_noise,
            "raw_observed_bis": self.raw_observed_bis,
            "observed_bis": self.observed_bis,
            "bis_clipping_applied": self.bis_clipping_applied,
            "propofol_rate_mg_per_min": self.propofol_rate_mg_per_min,
            "remifentanil_rate_micrograms_per_min": (
                self.remifentanil_rate_micrograms_per_min
            ),
            "remifentanil_schedule_kind": self.remifentanil_schedule_kind,
            "deterministic": self.deterministic,
            "integrator": self.integrator,
        }


class PKPDSimulator:
    """Research-only Schnider/Minto/Yun simulator with a pure advance API."""

    def __init__(
        self,
        *,
        internal_dt_seconds: float = 1.0,
        deterministic: bool = True,
        integrator: IntegratorName = "exact",
    ) -> None:
        if not math.isfinite(internal_dt_seconds) or internal_dt_seconds <= 0.0:
            raise ValueError("internal_dt_seconds must be finite and positive.")
        if integrator not in ("exact", "solve_ivp"):
            raise ValueError("integrator must be 'exact' or 'solve_ivp'.")
        self.internal_dt_seconds = float(internal_dt_seconds)
        self.deterministic = bool(deterministic)
        self.integrator = integrator
        self._rng = np.random.default_rng(0)
        self._patient: PatientDemographics | None = None
        self._propofol_parameters: DrugPKParameters | None = None
        self._remifentanil_parameters: DrugPKParameters | None = None
        self._state: CombinedPatientState | None = None

    @property
    def propofol_parameters(self) -> DrugPKParameters:
        if self._propofol_parameters is None:
            raise RuntimeError("Simulator must be reset before parameters are available.")
        return self._propofol_parameters

    @property
    def remifentanil_parameters(self) -> DrugPKParameters:
        if self._remifentanil_parameters is None:
            raise RuntimeError("Simulator must be reset before parameters are available.")
        return self._remifentanil_parameters

    def _bis(self, propofol: DrugCompartmentState, remifentanil: DrugCompartmentState) -> BISResponse:
        return evaluate_bis(
            propofol.ce,
            remifentanil.ce,
            deterministic=self.deterministic,
            rng=self._rng,
        )

    def reset(
        self,
        patient: PatientDemographics,
        seed: int,
        initial_state: CombinedPatientState | None = None,
    ) -> CombinedPatientState:
        """Reset drug amounts and reproducible stochastic state for one patient."""

        if not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer.")
        self._patient = patient
        self._propofol_parameters = schnider_propofol_parameters(patient)
        self._remifentanil_parameters = minto_remifentanil_parameters(patient)
        self._rng = np.random.default_rng(int(seed))
        if initial_state is None:
            time_seconds = 0.0
            propofol = zero_state(self.propofol_parameters)
            remifentanil = zero_state(self.remifentanil_parameters)
            propofol_rate = 0.0
            remifentanil_rate = 0.0
            schedule_kind = "constant"
        else:
            if initial_state.demographics != patient:
                raise ValueError("initial_state demographics do not match reset patient.")
            time_seconds = initial_state.time_seconds
            propofol = initial_state.propofol
            remifentanil = initial_state.remifentanil
            propofol_rate = initial_state.propofol_rate_mg_per_min
            remifentanil_rate = initial_state.remifentanil_rate_micrograms_per_min
            schedule_kind = initial_state.remifentanil_schedule_kind
        bis = self._bis(propofol, remifentanil)
        self._state = self._combine(
            time_seconds=time_seconds,
            propofol=propofol,
            remifentanil=remifentanil,
            bis=bis,
            propofol_rate=propofol_rate,
            remifentanil_rate=remifentanil_rate,
            schedule_kind=schedule_kind,
        )
        return self.snapshot()

    def _combine(
        self,
        *,
        time_seconds: float,
        propofol: DrugCompartmentState,
        remifentanil: DrugCompartmentState,
        bis: BISResponse,
        propofol_rate: float,
        remifentanil_rate: float,
        schedule_kind: str,
    ) -> CombinedPatientState:
        assert self._patient is not None
        return CombinedPatientState(
            time_seconds=float(time_seconds),
            propofol=propofol,
            remifentanil=remifentanil,
            raw_noiseless_bis=bis.raw_noiseless_bis,
            noiseless_bis=bis.noiseless_bis,
            bis_noise=bis.noise_bis_units,
            raw_observed_bis=bis.raw_observed_bis,
            observed_bis=bis.observed_bis,
            bis_clipping_applied=bis.clipping_applied,
            demographics=self._patient,
            propofol_rate_mg_per_min=float(propofol_rate),
            remifentanil_rate_micrograms_per_min=float(remifentanil_rate),
            remifentanil_schedule_kind=schedule_kind,
            deterministic=self.deterministic,
            integrator=self.integrator,
        )

    def advance(
        self,
        *,
        propofol_rate_mg_per_min: float,
        duration_seconds: float,
        remifentanil_rate_micrograms_per_min: float | None = None,
        remifentanil_schedule: RateSchedule | None = None,
    ) -> CombinedPatientState:
        """Advance a constant propofol action and exogenous remifentanil input."""

        if self._state is None:
            raise RuntimeError("Simulator must be reset before advance().")
        if not math.isfinite(propofol_rate_mg_per_min) or propofol_rate_mg_per_min < 0.0:
            raise ValueError("propofol_rate_mg_per_min must be finite and non-negative.")
        if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
            raise ValueError("duration_seconds must be finite and positive.")
        if remifentanil_rate_micrograms_per_min is not None and remifentanil_schedule is not None:
            raise ValueError("Provide a remifentanil rate or schedule, not both.")
        if remifentanil_rate_micrograms_per_min is not None and (
            not math.isfinite(remifentanil_rate_micrograms_per_min)
            or remifentanil_rate_micrograms_per_min < 0.0
        ):
            raise ValueError(
                "remifentanil_rate_micrograms_per_min must be finite and non-negative."
            )

        remaining = float(duration_seconds)
        state = self._state
        while remaining > 1e-12:
            step = min(self.internal_dt_seconds, remaining)
            if remifentanil_schedule is None:
                remifentanil_rate = float(remifentanil_rate_micrograms_per_min or 0.0)
                schedule_kind = "constant"
            else:
                remifentanil_rate = float(remifentanil_schedule.rate_at(state.time_seconds))
                schedule_kind = type(remifentanil_schedule).__name__
            propofol = advance_compartment(
                state.propofol,
                self.propofol_parameters,
                infusion_rate_per_min=float(propofol_rate_mg_per_min),
                duration_seconds=step,
                integrator=self.integrator,
            )
            remifentanil = advance_compartment(
                state.remifentanil,
                self.remifentanil_parameters,
                infusion_rate_per_min=remifentanil_rate,
                duration_seconds=step,
                integrator=self.integrator,
            )
            bis = self._bis(propofol, remifentanil)
            state = self._combine(
                time_seconds=state.time_seconds + step,
                propofol=propofol,
                remifentanil=remifentanil,
                bis=bis,
                propofol_rate=float(propofol_rate_mg_per_min),
                remifentanil_rate=remifentanil_rate,
                schedule_kind=schedule_kind,
            )
            remaining -= step
        self._state = state
        return self.snapshot()

    def snapshot(self) -> CombinedPatientState:
        if self._state is None:
            raise RuntimeError("Simulator must be reset before snapshot().")
        return copy.deepcopy(self._state)

    def clone(self) -> "PKPDSimulator":
        """Clone the complete state, including RNG state, for deterministic replay."""

        if self._state is None:
            raise RuntimeError("Simulator must be reset before clone().")
        return copy.deepcopy(self)
