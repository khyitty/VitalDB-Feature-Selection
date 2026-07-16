"""Focused scientific and numerical tests for the PK-PD simulator."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pkpd import (
    CSVTrajectorySchedule,
    CallableSchedule,
    ConstantSchedule,
    PKPDSimulator,
    PatientDemographics,
    PiecewiseConstantSchedule,
    RateSegment,
    calculate_lean_body_mass_kg,
    mass_balance_residual,
    minto_remifentanil_parameters,
    schnider_propofol_parameters,
)
from src.pkpd.compartment import (
    exact_zero_order_hold_step,
    solve_ivp_reference_step,
    state_vector,
    zero_state,
)
from src.pkpd.units import (
    micrograms_to_milligrams,
    milligrams_to_micrograms,
    propofol_mg_per_l_to_micrograms_per_ml,
    remifentanil_micrograms_per_l_to_ng_per_ml,
    seconds_to_minutes,
)
from src.pkpd.validation import (
    ValidationConfig,
    compare_integrators,
    run_validation,
    simulate_trajectory,
)


@pytest.fixture
def reference_patient() -> PatientDemographics:
    return PatientDemographics(40, "male", 177, 77)


@pytest.mark.parametrize(
    ("sex", "expected"),
    [
        ("male", 1.10 * 77 - 128 * (77 / 177) ** 2),
        ("female", 1.07 * 77 - 148 * (77 / 177) ** 2),
    ],
)
def test_james_lbm_uses_squared_weight_height_ratio(sex: str, expected: float) -> None:
    assert calculate_lean_body_mass_kg(
        sex=sex, height_cm=177, weight_kg=77  # type: ignore[arg-type]
    ) == pytest.approx(expected)


def test_schnider_reference_demographics_parameters(
    reference_patient: PatientDemographics,
) -> None:
    parameters = schnider_propofol_parameters(reference_patient)
    assert parameters.volumes_l == pytest.approx((4.27, 23.983, 238.0))
    assert parameters.clearances_l_per_min == pytest.approx(
        (1.7894807133965331, 1.602, 0.836)
    )
    assert parameters.ke0_per_min == 0.456


def test_minto_reference_demographics_parameters(
    reference_patient: PatientDemographics,
) -> None:
    parameters = minto_remifentanil_parameters(reference_patient)
    assert parameters.volumes_l == pytest.approx(
        (5.494275897730537, 10.411413846595806, 5.42)
    )
    assert parameters.clearances_l_per_min == pytest.approx(
        (2.7045926339812953, 2.05, 0.076)
    )
    assert parameters.ke0_per_min == 0.595


def test_clearances_convert_to_micro_rate_constants(
    reference_patient: PatientDemographics,
) -> None:
    parameters = schnider_propofol_parameters(reference_patient)
    assert parameters.k10_per_min == pytest.approx(parameters.cl1_l_per_min / parameters.v1_l)
    assert parameters.k12_per_min == pytest.approx(parameters.cl2_l_per_min / parameters.v1_l)
    assert parameters.k13_per_min == pytest.approx(parameters.cl3_l_per_min / parameters.v1_l)
    assert parameters.k21_per_min == pytest.approx(parameters.cl2_l_per_min / parameters.v2_l)
    assert parameters.k31_per_min == pytest.approx(parameters.cl3_l_per_min / parameters.v3_l)


def test_explicit_unit_conversions() -> None:
    assert seconds_to_minutes(90) == 1.5
    assert milligrams_to_micrograms(2.5) == 2500.0
    assert micrograms_to_milligrams(2500.0) == 2.5
    assert propofol_mg_per_l_to_micrograms_per_ml(4.47) == 4.47
    assert remifentanil_micrograms_per_l_to_ng_per_ml(19.3) == 19.3


def test_source_registry_covers_implemented_constants() -> None:
    path = Path(__file__).parents[1] / "outputs/pkpd_simulator_validation/source_equation_registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    entries = {entry["id"]: entry for entry in registry["entries"]}
    assert registry["blocked_missing_source_equation"] is False
    assert set(entries["schnider_h1_h17"]["variables"]) == {
        f"h{index}" for index in range(1, 18)
    }
    assert set(entries["minto_f1_f18"]["variables"]) == {
        f"f{index}" for index in range(1, 19)
    }
    assert all(entry["source"] and entry["implementation"] for entry in registry["entries"])


def test_zero_input_zero_initial_state_stays_zero(
    reference_patient: PatientDemographics,
) -> None:
    simulator = PKPDSimulator()
    simulator.reset(reference_patient, 1)
    state = simulator.advance(
        propofol_rate_mg_per_min=0.0,
        remifentanil_rate_micrograms_per_min=0.0,
        duration_seconds=600,
    )
    assert [state.propofol.x1, state.propofol.x2, state.propofol.x3, state.propofol.ce] == [0.0] * 4
    assert [state.remifentanil.x1, state.remifentanil.x2, state.remifentanil.x3, state.remifentanil.ce] == [0.0] * 4
    assert state.noiseless_bis == 98.0


def test_positive_infusion_is_finite_nonnegative_and_cp_leads_ce(
    reference_patient: PatientDemographics,
) -> None:
    simulator = PKPDSimulator()
    simulator.reset(reference_patient, 1)
    first = simulator.advance(
        propofol_rate_mg_per_min=12.0,
        remifentanil_rate_micrograms_per_min=6.0,
        duration_seconds=10,
    )
    values = np.asarray(
        [
            first.propofol.x1,
            first.propofol.x2,
            first.propofol.x3,
            first.propofol.cp,
            first.propofol.ce,
            first.remifentanil.x1,
            first.remifentanil.cp,
            first.remifentanil.ce,
        ]
    )
    assert np.isfinite(values).all() and (values >= 0.0).all()
    assert first.propofol.cp > first.propofol.ce > 0.0


def test_effect_site_rises_later_and_bis_decreases(
    reference_patient: PatientDemographics,
) -> None:
    simulator = PKPDSimulator()
    initial = simulator.reset(reference_patient, 1)
    early = simulator.advance(
        propofol_rate_mg_per_min=12,
        remifentanil_rate_micrograms_per_min=0,
        duration_seconds=10,
    )
    later = simulator.advance(
        propofol_rate_mg_per_min=12,
        remifentanil_rate_micrograms_per_min=0,
        duration_seconds=110,
    )
    assert later.propofol.ce > early.propofol.ce
    assert later.noiseless_bis < early.noiseless_bis < initial.noiseless_bis


def test_infusion_off_decreases_concentrations_and_recovers_bis(
    reference_patient: PatientDemographics,
) -> None:
    simulator = PKPDSimulator()
    simulator.reset(reference_patient, 1)
    infused = simulator.advance(
        propofol_rate_mg_per_min=10,
        remifentanil_rate_micrograms_per_min=4,
        duration_seconds=600,
    )
    recovered = simulator.advance(
        propofol_rate_mg_per_min=0,
        remifentanil_rate_micrograms_per_min=0,
        duration_seconds=1800,
    )
    assert recovered.propofol.cp < infused.propofol.cp
    assert recovered.propofol.ce < infused.propofol.ce
    assert recovered.noiseless_bis > infused.noiseless_bis


def test_remifentanil_lowers_combined_bis_direction(
    reference_patient: PatientDemographics,
) -> None:
    no_remi = PKPDSimulator()
    with_remi = PKPDSimulator()
    no_remi.reset(reference_patient, 1)
    with_remi.reset(reference_patient, 1)
    state_no_remi = no_remi.advance(
        propofol_rate_mg_per_min=7,
        remifentanil_rate_micrograms_per_min=0,
        duration_seconds=600,
    )
    state_with_remi = with_remi.advance(
        propofol_rate_mg_per_min=7,
        remifentanil_rate_micrograms_per_min=10,
        duration_seconds=600,
    )
    assert state_with_remi.noiseless_bis < state_no_remi.noiseless_bis


def _trajectory(patient: PatientDemographics, seed: int, deterministic: bool) -> list[tuple[float, float]]:
    simulator = PKPDSimulator(deterministic=deterministic)
    simulator.reset(patient, seed)
    return [
        (
            state.noiseless_bis,
            state.bis_noise,
        )
        for state in [
            simulator.advance(
                propofol_rate_mg_per_min=7,
                remifentanil_rate_micrograms_per_min=4,
                duration_seconds=10,
            )
            for _ in range(5)
        ]
    ]


def test_same_patient_seed_actions_replay_identically(
    reference_patient: PatientDemographics,
) -> None:
    assert _trajectory(reference_patient, 12, False) == _trajectory(reference_patient, 12, False)


def test_stochastic_seed_changes_noise_but_deterministic_seed_is_irrelevant(
    reference_patient: PatientDemographics,
) -> None:
    assert _trajectory(reference_patient, 1, False) != _trajectory(reference_patient, 2, False)
    assert _trajectory(reference_patient, 1, True) == _trajectory(reference_patient, 2, True)


def test_clone_preserves_replay_state(reference_patient: PatientDemographics) -> None:
    simulator = PKPDSimulator(deterministic=False)
    simulator.reset(reference_patient, 42)
    simulator.advance(
        propofol_rate_mg_per_min=7,
        remifentanil_rate_micrograms_per_min=4,
        duration_seconds=10,
    )
    clone = simulator.clone()
    left = simulator.advance(
        propofol_rate_mg_per_min=8,
        remifentanil_rate_micrograms_per_min=5,
        duration_seconds=10,
    )
    right = clone.advance(
        propofol_rate_mg_per_min=8,
        remifentanil_rate_micrograms_per_min=5,
        duration_seconds=10,
    )
    assert left == right


def test_ten_second_hold_matches_ten_one_second_advances(
    reference_patient: PatientDemographics,
) -> None:
    one = PKPDSimulator()
    ten = PKPDSimulator()
    one.reset(reference_patient, 1)
    ten.reset(reference_patient, 1)
    direct = ten.advance(
        propofol_rate_mg_per_min=9,
        remifentanil_rate_micrograms_per_min=5,
        duration_seconds=10,
    )
    repeated = None
    for _ in range(10):
        repeated = one.advance(
            propofol_rate_mg_per_min=9,
            remifentanil_rate_micrograms_per_min=5,
            duration_seconds=1,
        )
    assert repeated is not None
    assert repeated.propofol.ce == pytest.approx(direct.propofol.ce, abs=1e-14)
    assert repeated.remifentanil.ce == pytest.approx(direct.remifentanil.ce, abs=1e-14)


def test_matrix_exponential_matches_solve_ivp(reference_patient: PatientDemographics) -> None:
    parameters = schnider_propofol_parameters(reference_patient)
    initial = state_vector(zero_state(parameters))
    exact = exact_zero_order_hold_step(initial, 10.0, 120.0, parameters)
    reference = solve_ivp_reference_step(initial, 10.0, 120.0, parameters)
    assert exact == pytest.approx(reference, rel=1e-10, abs=1e-12)


def test_internal_dt_refinement_converges(reference_patient: PatientDemographics) -> None:
    coarse = PKPDSimulator(internal_dt_seconds=1.0)
    fine = PKPDSimulator(internal_dt_seconds=0.25)
    coarse.reset(reference_patient, 1)
    fine.reset(reference_patient, 1)
    left = coarse.advance(
        propofol_rate_mg_per_min=9,
        remifentanil_rate_micrograms_per_min=5,
        duration_seconds=10,
    )
    right = fine.advance(
        propofol_rate_mg_per_min=9,
        remifentanil_rate_micrograms_per_min=5,
        duration_seconds=10,
    )
    assert left.propofol.ce == pytest.approx(right.propofol.ce, abs=1e-13)
    assert left.noiseless_bis == pytest.approx(right.noiseless_bis, abs=1e-12)


def test_long_no_infusion_decay_and_mass_balance(reference_patient: PatientDemographics) -> None:
    simulator = PKPDSimulator(internal_dt_seconds=10)
    simulator.reset(reference_patient, 1)
    infused = simulator.advance(
        propofol_rate_mg_per_min=10,
        remifentanil_rate_micrograms_per_min=6,
        duration_seconds=300,
    )
    decayed = simulator.advance(
        propofol_rate_mg_per_min=0,
        remifentanil_rate_micrograms_per_min=0,
        duration_seconds=7200,
    )
    assert decayed.propofol.total_compartment_amount < infused.propofol.total_compartment_amount
    assert abs(mass_balance_residual(decayed.propofol)) < 1e-10
    assert abs(mass_balance_residual(decayed.remifentanil)) < 1e-10


def test_extreme_allowed_action_remains_finite(reference_patient: PatientDemographics) -> None:
    simulator = PKPDSimulator(internal_dt_seconds=10)
    simulator.reset(reference_patient, 1)
    state = simulator.advance(
        propofol_rate_mg_per_min=1000,
        remifentanil_rate_micrograms_per_min=1000,
        duration_seconds=600,
    )
    values = np.asarray(list(state.propofol.as_dict("p").values())[0:8], dtype=float)
    assert np.isfinite(values).all()
    assert 0.0 <= state.observed_bis <= 100.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"propofol_rate_mg_per_min": -1, "remifentanil_rate_micrograms_per_min": 0},
        {"propofol_rate_mg_per_min": 0, "remifentanil_rate_micrograms_per_min": -1},
    ],
)
def test_negative_actions_are_rejected(
    reference_patient: PatientDemographics, kwargs: dict[str, float]
) -> None:
    simulator = PKPDSimulator()
    simulator.reset(reference_patient, 1)
    with pytest.raises(ValueError, match="non-negative"):
        simulator.advance(**kwargs, duration_seconds=10)


@pytest.mark.parametrize(
    "args",
    [
        (17, "male", 177, 77),
        (40, "unknown", 177, 77),
        (40, "male", 100, 77),
        (40, "male", 177, -1),
    ],
)
def test_invalid_demographics_are_rejected(args: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        PatientDemographics(*args)  # type: ignore[arg-type]


def test_induction_maintenance_recovery_qualitative_trajectory() -> None:
    frame, _ = simulate_trajectory(
        ValidationConfig(duration_seconds=1800, internal_dt_seconds=10)
    )
    assert frame["noiseless_bis"].iloc[0] > 90
    assert frame["noiseless_bis"].min() < 60
    maintenance = frame.loc[frame["time_seconds"].between(600, 900)]
    assert maintenance["noiseless_bis"].median() == pytest.approx(50, abs=10)
    assert frame["noiseless_bis"].iloc[-1] > frame["noiseless_bis"].min()
    cp_peak_time = frame.loc[frame["propofol_cp"].idxmax(), "time_seconds"]
    ce_peak_time = frame.loc[frame["propofol_ce"].idxmax(), "time_seconds"]
    assert ce_peak_time > cp_peak_time


def test_schedule_types_and_validation(tmp_path: Path) -> None:
    assert ConstantSchedule(4).rate_at(100) == 4
    piecewise = PiecewiseConstantSchedule(
        [RateSegment(0, 10, 2), RateSegment(10, 20, 4)]
    )
    assert [piecewise.rate_at(value) for value in (5, 15, 25)] == [2, 4, 0]
    assert CallableSchedule(lambda time: time / 10).rate_at(20) == 2
    csv_path = tmp_path / "schedule.csv"
    pd.DataFrame(
        {
            "time_seconds": [0, 10, 20],
            "remifentanil_rate_micrograms_per_min": [1, 2, 3],
        }
    ).to_csv(csv_path, index=False)
    assert CSVTrajectorySchedule.from_csv(csv_path).rate_at(15) == 2
    with pytest.raises(ValueError, match="overlap"):
        PiecewiseConstantSchedule([RateSegment(0, 10, 1), RateSegment(9, 20, 2)])


def test_simulator_consumes_piecewise_exogenous_schedule(
    reference_patient: PatientDemographics,
) -> None:
    schedule = PiecewiseConstantSchedule(
        [RateSegment(0, 10, 2), RateSegment(10, 20, 4)]
    )
    simulator = PKPDSimulator(internal_dt_seconds=1)
    simulator.reset(reference_patient, 1)
    state = simulator.advance(
        propofol_rate_mg_per_min=0,
        remifentanil_schedule=schedule,
        duration_seconds=20,
    )
    assert state.remifentanil.cumulative_dose == pytest.approx(1.0)
    assert state.remifentanil_schedule_kind == "PiecewiseConstantSchedule"
    with pytest.raises(ValueError, match="not both"):
        simulator.advance(
            propofol_rate_mg_per_min=0,
            remifentanil_rate_micrograms_per_min=1,
            remifentanil_schedule=schedule,
            duration_seconds=1,
        )


def test_validation_outputs_and_no_unsupported_vitals(tmp_path: Path) -> None:
    repo_dir = Path(__file__).parents[1]
    output = tmp_path / "validation"
    summary = run_validation(
        ValidationConfig(duration_seconds=600, internal_dt_seconds=10), output, repo_dir
    )
    assert summary["status"] == "passed"
    required = {
        "simulator_manifest.json",
        "source_equation_registry.json",
        "reference_patient_parameters.csv",
        "trajectory.csv",
        "integrator_comparison.csv",
        "validation_summary.json",
        "pkpd_validation_report.md",
    }
    assert required.issubset({path.name for path in output.iterdir()})
    assert len(list((output / "figures").glob("*.png"))) == 5
    manifest = json.loads((output / "simulator_manifest.json").read_text())
    assert manifest["clinical_use_prohibition"] == (
        "This simulator is a research reconstruction of published PK–PD equations. "
        "It is not a medical device and must not be used for clinical dosing."
    )
    assert manifest["not_yet_connected_to_rl"] is True
    assert manifest["rl_training_performed"] is False
    assert manifest["unsupported_vital_signs"] == [
        "HR",
        "MBP",
        "SBP",
        "DBP",
        "SpO2",
        "ETCO2",
        "HRV",
    ]
    trajectory = pd.read_csv(output / "trajectory.csv")
    assert not {"hr", "mbp", "sbp", "dbp", "spo2", "etco2", "hrv"}.intersection(
        trajectory.columns
    )


def test_integrator_comparison_is_tight(reference_patient: PatientDemographics) -> None:
    comparison = compare_integrators(reference_patient, duration_seconds=30)
    assert comparison.filter(like="absolute_difference").to_numpy().max() < 1e-9


def test_standalone_colab_notebook_is_clean_and_contains_no_rl_training() -> None:
    path = Path(__file__).parents[1] / "notebooks/colab_pkpd_simulator_validation.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{path}:cell-{index}")
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
    assert "run_pkpd_simulator_validation.py" in source
    assert "drive.mount" not in source
    assert "gymnasium" not in source.lower()
    assert "stable_baselines" not in source.lower()
    assert "run_baselines.py" not in source
