"""Causal offline Cp/Ce reconstruction and unit contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pkpd.demographics import PatientDemographics
from src.pkpd.reconstruction import (
    PROPOFOL_CONCENTRATION_MG_PER_ML,
    REMIFENTANIL_CONCENTRATION_MICROGRAMS_PER_ML,
    reconstruct_causal_pk_concentrations,
)
from src.pkpd.simulator import PKPDSimulator


def _demographics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "caseid": [1],
            "age": [40.0],
            "sex_male": [1.0],
            "height": [177.0],
            "weight": [77.0],
        }
    )


def _history(*, final_propofol_rate: float = 3.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "caseid": [1, 1, 1],
            "time_sec": [0, 10, 20],
            "PPF_RATE": [3.0, 3.0, final_propofol_rate],
            "PPF_VOL": [0.0, 0.2, 0.4],
            "PPF_CP": [0.0, 0.5, 0.8],
            "PPF_CE": [0.0, 0.1, 0.2],
            "RFTN_RATE": [6.0, 6.0, 6.0],
            "RFTN_VOL": [0.0, 0.1, 0.2],
            "RFTN_CP": [0.0, 1.0, 1.5],
            "RFTN_CE": [0.0, 0.2, 0.4],
            "Orchestra/PPF20_CT": [99.0, 99.0, 99.0],
        }
    )


def test_reconstruction_matches_same_schnider_minto_simulator_and_units() -> None:
    output, audit = reconstruct_causal_pk_concentrations(_history(), _demographics())
    patient = PatientDemographics(40.0, "male", 177.0, 77.0)
    simulator = PKPDSimulator(deterministic=True)
    simulator.reset(patient, seed=1)
    states = [
        simulator.advance(
            propofol_rate_mg_per_min=1.0,
            remifentanil_rate_micrograms_per_min=2.0,
            duration_seconds=10,
        )
        for _ in range(2)
    ]
    assert PROPOFOL_CONCENTRATION_MG_PER_ML == 20.0
    assert REMIFENTANIL_CONCENTRATION_MICROGRAMS_PER_ML == 20.0
    assert output.loc[1, "propofol_cp_mg_per_l"] == pytest.approx(states[0].propofol.cp)
    assert output.loc[2, "propofol_ce_mg_per_l"] == pytest.approx(states[1].propofol.ce)
    assert output.loc[1, "remifentanil_cp_micrograms_per_l"] == pytest.approx(
        states[0].remifentanil.cp
    )
    assert output.loc[2, "remifentanil_ce_micrograms_per_l"] == pytest.approx(
        states[1].remifentanil.ce
    )
    assert audit["propofol_model"] == "Schnider"
    assert audit["remifentanil_model"] == "Minto"
    assert audit["target_concentration_used"] is False


def test_rate_observed_at_t_cannot_change_concentration_at_same_t() -> None:
    left, _ = reconstruct_causal_pk_concentrations(_history(), _demographics())
    right, _ = reconstruct_causal_pk_concentrations(
        _history(final_propofol_rate=3000.0), _demographics()
    )
    columns = ["propofol_cp_mg_per_l", "propofol_ce_mg_per_l"]
    np.testing.assert_allclose(left.loc[2, columns], right.loc[2, columns])


def test_target_concentration_is_ignored() -> None:
    first = _history()
    second = _history()
    second["Orchestra/PPF20_CT"] = -12345.0
    left, _ = reconstruct_causal_pk_concentrations(first, _demographics())
    right, _ = reconstruct_causal_pk_concentrations(second, _demographics())
    canonical = [
        "propofol_cp_mg_per_l",
        "propofol_ce_mg_per_l",
        "remifentanil_cp_micrograms_per_l",
        "remifentanil_ce_micrograms_per_l",
    ]
    np.testing.assert_allclose(left[canonical], right[canonical])


def test_nonzero_initial_device_state_is_not_silently_relabelled() -> None:
    history = _history()
    history.loc[0, "PPF_CP"] = 1.0
    output, audit = reconstruct_causal_pk_concentrations(history, _demographics())
    assert output["propofol_cp_mg_per_l"].isna().all()
    assert output["propofol_ce_mg_per_l"].isna().all()
    assert (
        audit["cases"]["1"]["propofol"]["status"]
        == "unknown_nonzero_initial_pk_state"
    )


def test_stale_rate_gap_invalidates_future_state_without_fabrication() -> None:
    history = pd.concat(
        [_history().iloc[[0]]] * 12,
        ignore_index=True,
    )
    history["time_sec"] = np.arange(12)
    history.loc[1:, ["PPF_RATE", "RFTN_RATE"]] = np.nan
    output, audit = reconstruct_causal_pk_concentrations(history, _demographics())
    assert np.isfinite(output.loc[10, "propofol_cp_mg_per_l"])
    assert np.isnan(output.loc[11, "propofol_cp_mg_per_l"])
    assert audit["cases"]["1"]["propofol"]["first_invalid_time_seconds"] == 11
