"""Causal offline Cp/Ce reconstruction using the control simulator PK models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping
import warnings

import numpy as np
import pandas as pd

from .compartment import exact_zero_order_hold_transition
from .demographics import PatientDemographics
from .parameters import (
    DrugPKParameters,
    minto_remifentanil_parameters,
    schnider_propofol_parameters,
)


PROPOFOL_CONCENTRATION_MG_PER_ML = 20.0
REMIFENTANIL_CONCENTRATION_MICROGRAMS_PER_ML = 20.0
DEFAULT_MAX_RATE_HOLD_SECONDS = 10


@dataclass(frozen=True)
class DrugReconstructionSpec:
    """Source and output contract for one reconstructed drug trajectory."""

    drug_name: str
    rate_column: str
    volume_column: str
    recorded_cp_column: str
    recorded_ce_column: str
    output_cp_column: str
    output_ce_column: str
    recorded_output_cp_column: str
    recorded_output_ce_column: str
    concentration_per_ml: float


DRUG_SPECS = (
    DrugReconstructionSpec(
        drug_name="propofol",
        rate_column="Orchestra/PPF20_RATE",
        volume_column="Orchestra/PPF20_VOL",
        recorded_cp_column="Orchestra/PPF20_CP",
        recorded_ce_column="Orchestra/PPF20_CE",
        output_cp_column="propofol_cp_mg_per_l",
        output_ce_column="propofol_ce_mg_per_l",
        recorded_output_cp_column="__recorded_orchestra_propofol_cp_mg_per_l",
        recorded_output_ce_column="__recorded_orchestra_propofol_ce_mg_per_l",
        concentration_per_ml=PROPOFOL_CONCENTRATION_MG_PER_ML,
    ),
    DrugReconstructionSpec(
        drug_name="remifentanil",
        rate_column="Orchestra/RFTN20_RATE",
        volume_column="Orchestra/RFTN20_VOL",
        recorded_cp_column="Orchestra/RFTN20_CP",
        recorded_ce_column="Orchestra/RFTN20_CE",
        output_cp_column="remifentanil_cp_micrograms_per_l",
        output_ce_column="remifentanil_ce_micrograms_per_l",
        recorded_output_cp_column=(
            "__recorded_orchestra_remifentanil_cp_micrograms_per_l"
        ),
        recorded_output_ce_column=(
            "__recorded_orchestra_remifentanil_ce_micrograms_per_l"
        ),
        concentration_per_ml=REMIFENTANIL_CONCENTRATION_MICROGRAMS_PER_ML,
    ),
)

RAW_COLUMN_ALIASES: Mapping[str, str] = {
    "PPF_RATE": "Orchestra/PPF20_RATE",
    "PPF_VOL": "Orchestra/PPF20_VOL",
    "PPF_CP": "Orchestra/PPF20_CP",
    "PPF_CE": "Orchestra/PPF20_CE",
    "RFTN_RATE": "Orchestra/RFTN20_RATE",
    "RFTN_VOL": "Orchestra/RFTN20_VOL",
    "RFTN_CP": "Orchestra/RFTN20_CP",
    "RFTN_CE": "Orchestra/RFTN20_CE",
}


def _canonicalize_history_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for alias, canonical in RAW_COLUMN_ALIASES.items():
        if canonical not in result and alias in result:
            result[canonical] = result[alias]
    required = {
        "caseid",
        "time_sec",
        *(spec.rate_column for spec in DRUG_SPECS),
        *(spec.volume_column for spec in DRUG_SPECS),
    }
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(f"PK reconstruction history is missing columns: {missing}")
    for spec in DRUG_SPECS:
        for optional in (spec.recorded_cp_column, spec.recorded_ce_column):
            if optional not in result:
                result[optional] = np.nan
    result["caseid"] = pd.to_numeric(result["caseid"], errors="raise").astype(np.int64)
    result["time_sec"] = pd.to_numeric(result["time_sec"], errors="raise").astype(np.int64)
    if bool(result.duplicated(["caseid", "time_sec"]).any()):
        raise ValueError("PK reconstruction history contains duplicate case timestamps.")
    return result.sort_values(["caseid", "time_sec"], kind="stable").reset_index(drop=True)


def _demographics_by_case(frame: pd.DataFrame) -> dict[int, PatientDemographics]:
    aliases = {
        "age_years": "age",
        "height_cm": "height",
        "weight_kg": "weight",
    }
    work = frame.copy()
    for canonical, alias in aliases.items():
        if canonical not in work and alias in work:
            work[canonical] = work[alias]
    required = {"caseid", "age_years", "sex_male", "height_cm", "weight_kg"}
    missing = sorted(required - set(work.columns))
    if missing:
        raise ValueError(f"PK reconstruction demographics are missing columns: {missing}")
    result: dict[int, PatientDemographics] = {}
    for case_id, case in work.groupby("caseid", sort=False):
        values = case.loc[:, sorted(required - {"caseid"})].drop_duplicates()
        if len(values) != 1 or values.isna().any().any():
            raise ValueError(f"Case {int(case_id)} has missing or inconsistent demographics.")
        row = values.iloc[0]
        sex_value = float(row["sex_male"])
        if sex_value not in (0.0, 1.0):
            raise ValueError(f"Case {int(case_id)} has invalid sex_male={sex_value}.")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"Demographics are within simulator hard bounds.*"
            )
            result[int(case_id)] = PatientDemographics(
                age_years=float(row["age_years"]),
                sex="male" if sex_value == 1.0 else "female",
                height_cm=float(row["height_cm"]),
                weight_kg=float(row["weight_kg"]),
            )
    return result


def _parameters(spec: DrugReconstructionSpec, patient: PatientDemographics) -> DrugPKParameters:
    if spec.drug_name == "propofol":
        return schnider_propofol_parameters(patient)
    return minto_remifentanil_parameters(patient)


def _initial_state_is_unknown(case: pd.DataFrame, start_index: int, spec: DrugReconstructionSpec) -> bool:
    evidence_columns = (
        spec.volume_column,
        spec.recorded_cp_column,
        spec.recorded_ce_column,
    )
    prior = case.loc[:start_index, list(evidence_columns)]
    observed = prior.to_numpy(dtype=float)
    return bool(np.isfinite(observed).any() and np.nanmax(np.abs(observed)) > 1e-8)


def _reconstruct_drug(
    case: pd.DataFrame,
    spec: DrugReconstructionSpec,
    parameters: DrugPKParameters,
    *,
    max_rate_hold_seconds: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    count = len(case)
    cp = np.full(count, np.nan, dtype=np.float64)
    ce = np.full(count, np.nan, dtype=np.float64)
    rates = pd.to_numeric(case[spec.rate_column], errors="coerce").to_numpy(dtype=float)
    times = case["time_sec"].to_numpy(dtype=np.int64)
    observed_rate_indices = np.flatnonzero(np.isfinite(rates))
    audit: dict[str, Any] = {
        "status": "complete",
        "first_reconstructed_time_seconds": None,
        "first_invalid_time_seconds": None,
        "valid_rows": 0,
    }
    if not len(observed_rate_indices):
        audit["status"] = "missing_rate_track"
        return cp, ce, audit
    start = int(observed_rate_indices[0])
    if _initial_state_is_unknown(case, start, spec):
        audit["status"] = "unknown_nonzero_initial_pk_state"
        return cp, ce, audit

    vector = np.zeros(5, dtype=np.float64)
    last_rate = float(rates[start]) * spec.concentration_per_ml / 60.0
    if last_rate < 0.0:
        raise ValueError(f"Negative {spec.drug_name} rate at case time {times[start]}.")
    last_rate_time = int(times[start])
    cp[start] = 0.0
    ce[start] = 0.0
    audit["first_reconstructed_time_seconds"] = int(times[start])
    transition_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for index in range(start + 1, count):
        current_time = int(times[index])
        previous_time = int(times[index - 1])
        duration = current_time - previous_time
        if duration <= 0:
            raise ValueError("PK reconstruction timestamps must increase within each case.")
        if current_time - last_rate_time > max_rate_hold_seconds:
            audit["status"] = "unknown_after_stale_rate_gap"
            audit["first_invalid_time_seconds"] = current_time
            break
        transition = transition_cache.get(duration)
        if transition is None:
            transition = exact_zero_order_hold_transition(parameters, duration)
            transition_cache[duration] = transition
        vector = transition[0] @ vector + transition[1] * last_rate
        if not np.isfinite(vector).all() or float(vector.min()) < -1e-9:
            raise FloatingPointError(
                f"Invalid {spec.drug_name} reconstructed state at {current_time}: {vector}"
            )
        vector = np.maximum(vector, 0.0)
        cp[index] = float(vector[0] / parameters.v1_l)
        ce[index] = float(vector[3])
        if math.isfinite(rates[index]):
            rate = float(rates[index]) * spec.concentration_per_ml / 60.0
            if rate < 0.0:
                raise ValueError(
                    f"Negative {spec.drug_name} rate at case time {current_time}."
                )
            last_rate = rate
            last_rate_time = current_time
    audit["valid_rows"] = int(np.isfinite(cp).sum())
    return cp, ce, audit


def reconstruct_causal_pk_concentrations(
    history_frame: pd.DataFrame,
    demographics_frame: pd.DataFrame,
    *,
    max_rate_hold_seconds: int = DEFAULT_MAX_RATE_HOLD_SECONDS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconstruct model-defined Cp/Ce using only rate observations available by time t."""

    if max_rate_hold_seconds <= 0:
        raise ValueError("max_rate_hold_seconds must be positive.")
    history = _canonicalize_history_columns(history_frame)
    demographics = _demographics_by_case(demographics_frame)
    requested_cases = set(demographics)
    available_cases = set(history["caseid"].unique().astype(int))
    missing_cases = sorted(requested_cases - available_cases)
    if missing_cases:
        raise ValueError(f"Raw PK history is unavailable for cases: {missing_cases}")
    history = history[history["caseid"].isin(requested_cases)].copy()
    output = history.loc[:, ["caseid", "time_sec"]].copy()
    case_audits: dict[str, dict[str, Any]] = {}

    for spec in DRUG_SPECS:
        output[spec.output_cp_column] = np.nan
        output[spec.output_ce_column] = np.nan
        output[spec.recorded_output_cp_column] = pd.to_numeric(
            history[spec.recorded_cp_column], errors="coerce"
        )
        output[spec.recorded_output_ce_column] = pd.to_numeric(
            history[spec.recorded_ce_column], errors="coerce"
        )

    for case_id, case in history.groupby("caseid", sort=False):
        patient = demographics[int(case_id)]
        case_audits[str(int(case_id))] = {}
        for spec in DRUG_SPECS:
            cp, ce, drug_audit = _reconstruct_drug(
                case.reset_index(drop=True),
                spec,
                _parameters(spec, patient),
                max_rate_hold_seconds=max_rate_hold_seconds,
            )
            output.loc[case.index, spec.output_cp_column] = cp
            output.loc[case.index, spec.output_ce_column] = ce
            case_audits[str(int(case_id))][spec.drug_name] = drug_audit

    summary: dict[str, dict[str, int]] = {}
    for spec in DRUG_SPECS:
        statuses = [row[spec.drug_name]["status"] for row in case_audits.values()]
        summary[spec.drug_name] = {
            status: statuses.count(status) for status in sorted(set(statuses))
        }
    audit = {
        "schema_version": 1,
        "causal": True,
        "integrator": "repository exact zero-order-hold matrix exponential",
        "propofol_model": "Schnider",
        "remifentanil_model": "Minto",
        "initial_state_policy": (
            "Zero amount at first observed pump-rate timestamp only when volume and "
            "device-reported Cp/Ce through that timestamp are zero or absent; otherwise "
            "reconstruction remains unavailable for that drug/case."
        ),
        "missing_rate_policy": (
            f"Causal last observation carried for at most {max_rate_hold_seconds} seconds; "
            "after a longer gap the drug state remains unavailable."
        ),
        "rate_timing": (
            "The rate observed at t applies to (t, next timestamp]; state at t uses only "
            "rates observed before t."
        ),
        "target_concentration_used": False,
        "recorded_cp_ce_used_as_model_features": False,
        "case_count": len(case_audits),
        "status_counts": summary,
        "cases": case_audits,
    }
    return output, audit
