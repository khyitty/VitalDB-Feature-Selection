"""Deterministic validation workflow for the research PK-PD simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .bis_response import (
    BIS_EXPONENT,
    PROPOFOL_HALF_EFFECT_MG_PER_L,
    REMIFENTANIL_HALF_EFFECT_MICROGRAMS_PER_L,
    STOCHASTIC_NOISE_MEAN,
    STOCHASTIC_NOISE_VARIANCE,
)
from .compartment import mass_balance_residual
from .demographics import PatientDemographics
from .parameters import (
    MINTO_CONSTANTS,
    SCHNIDER_CONSTANTS,
    minto_remifentanil_parameters,
    schnider_propofol_parameters,
)
from .simulator import PKPDSimulator


LOGGER = logging.getLogger(__name__)
CLINICAL_PROHIBITION = (
    "This simulator is a research reconstruction of published PK–PD equations. "
    "It is not a medical device and must not be used for clinical dosing."
)
UNSUPPORTED_VITAL_SIGNS = ["HR", "MBP", "SBP", "DBP", "SpO2", "ETCO2", "HRV"]
SUPPORTED_STATE_FEATURES = [
    "BIS",
    "propofol x1/x2/x3, Cp, Ce, rate, cumulative dose",
    "remifentanil x1/x2/x3, Cp, Ce, rate, cumulative dose",
    "age, source-model sex, height, weight, lean body mass",
    "simulation time and drug/BIS history derived by the caller",
]

SYNTHETIC_PATIENTS: Mapping[str, PatientDemographics] = {
    "young_female": PatientDemographics(28, "female", 165, 58),
    "middle_male": PatientDemographics(45, "male", 177, 77),
    "older_female": PatientDemographics(75, "female", 158, 62),
}


@dataclass(frozen=True)
class ValidationConfig:
    patient_name: str = "middle_male"
    duration_seconds: float = 1800.0
    internal_dt_seconds: float = 1.0
    sample_interval_seconds: float = 10.0
    seed: int = 20260716
    deterministic: bool = True
    integrator: str = "exact"
    propofol_schedule: str = "induction_maintenance_recovery"
    remifentanil_schedule: str = "piecewise"
    propofol_constant_rate_mg_per_min: float = 7.0
    remifentanil_constant_rate_micrograms_per_min: float = 6.0


def _git_head(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _propofol_rate(config: ValidationConfig, time_seconds: float) -> float:
    if config.propofol_schedule == "off":
        return 0.0
    if config.propofol_schedule == "constant":
        return config.propofol_constant_rate_mg_per_min
    if config.propofol_schedule == "induction_maintenance_recovery":
        if time_seconds < min(180.0, config.duration_seconds / 3.0):
            return 12.0
        if time_seconds < min(900.0, 2.0 * config.duration_seconds / 3.0):
            return 7.0
        return 0.0
    raise ValueError(f"Unsupported propofol schedule: {config.propofol_schedule}")


def _remifentanil_rate(config: ValidationConfig, time_seconds: float) -> float:
    if config.remifentanil_schedule == "off":
        return 0.0
    if config.remifentanil_schedule == "constant":
        return config.remifentanil_constant_rate_micrograms_per_min
    if config.remifentanil_schedule == "piecewise":
        if time_seconds < min(300.0, config.duration_seconds / 3.0):
            return 4.0
        if time_seconds < min(900.0, 2.0 * config.duration_seconds / 3.0):
            return 8.0
        return 0.0
    raise ValueError(f"Unsupported remifentanil schedule: {config.remifentanil_schedule}")


def simulate_trajectory(config: ValidationConfig) -> tuple[pd.DataFrame, PKPDSimulator]:
    """Run one synthetic schedule and collect stable decision-point snapshots."""

    if config.patient_name not in SYNTHETIC_PATIENTS:
        raise ValueError(f"Unknown synthetic patient: {config.patient_name}")
    simulator = PKPDSimulator(
        internal_dt_seconds=config.internal_dt_seconds,
        deterministic=config.deterministic,
        integrator=config.integrator,  # type: ignore[arg-type]
    )
    state = simulator.reset(SYNTHETIC_PATIENTS[config.patient_name], config.seed)
    rows = [state.as_dict()]
    while state.time_seconds < config.duration_seconds - 1e-12:
        duration = min(
            config.sample_interval_seconds,
            config.duration_seconds - state.time_seconds,
        )
        ppf_rate = _propofol_rate(config, state.time_seconds)
        remi_rate = _remifentanil_rate(config, state.time_seconds)
        state = simulator.advance(
            propofol_rate_mg_per_min=ppf_rate,
            remifentanil_rate_micrograms_per_min=remi_rate,
            duration_seconds=duration,
        )
        rows.append(state.as_dict())
    return pd.DataFrame(rows), simulator


def compare_integrators(
    patient: PatientDemographics,
    *,
    duration_seconds: int = 300,
    propofol_rate_mg_per_min: float = 10.0,
    remifentanil_rate_micrograms_per_min: float = 6.0,
) -> pd.DataFrame:
    """Compare exact ZOH and independent solve_ivp at one-second intervals."""

    exact = PKPDSimulator(internal_dt_seconds=1.0, deterministic=True, integrator="exact")
    reference = PKPDSimulator(
        internal_dt_seconds=1.0, deterministic=True, integrator="solve_ivp"
    )
    exact.reset(patient, 1)
    reference.reset(patient, 999)
    rows: list[dict[str, float]] = []
    for second in range(1, duration_seconds + 1):
        exact_state = exact.advance(
            propofol_rate_mg_per_min=propofol_rate_mg_per_min,
            remifentanil_rate_micrograms_per_min=remifentanil_rate_micrograms_per_min,
            duration_seconds=1.0,
        )
        reference_state = reference.advance(
            propofol_rate_mg_per_min=propofol_rate_mg_per_min,
            remifentanil_rate_micrograms_per_min=remifentanil_rate_micrograms_per_min,
            duration_seconds=1.0,
        )
        rows.append(
            {
                "time_seconds": float(second),
                "propofol_cp_exact": exact_state.propofol.cp,
                "propofol_cp_solve_ivp": reference_state.propofol.cp,
                "propofol_ce_exact": exact_state.propofol.ce,
                "propofol_ce_solve_ivp": reference_state.propofol.ce,
                "remifentanil_cp_exact": exact_state.remifentanil.cp,
                "remifentanil_cp_solve_ivp": reference_state.remifentanil.cp,
                "remifentanil_ce_exact": exact_state.remifentanil.ce,
                "remifentanil_ce_solve_ivp": reference_state.remifentanil.ce,
                "bis_exact": exact_state.noiseless_bis,
                "bis_solve_ivp": reference_state.noiseless_bis,
            }
        )
    frame = pd.DataFrame(rows)
    for name in ("propofol_cp", "propofol_ce", "remifentanil_cp", "remifentanil_ce", "bis"):
        frame[f"{name}_absolute_difference"] = np.abs(
            frame[f"{name}_exact"] - frame[f"{name}_solve_ivp"]
        )
    return frame


def reference_patient_parameters() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, patient in SYNTHETIC_PATIENTS.items():
        for parameters in (
            schnider_propofol_parameters(patient),
            minto_remifentanil_parameters(patient),
        ):
            rows.append(
                {
                    "patient": name,
                    **patient.as_dict(),
                    "drug": parameters.drug_name,
                    "source_model": parameters.source_model,
                    "v1_l": parameters.v1_l,
                    "v2_l": parameters.v2_l,
                    "v3_l": parameters.v3_l,
                    "cl1_l_per_min": parameters.cl1_l_per_min,
                    "cl2_l_per_min": parameters.cl2_l_per_min,
                    "cl3_l_per_min": parameters.cl3_l_per_min,
                    **{
                        f"{key}_per_min": value
                        for key, value in parameters.micro_rate_constants_per_min.items()
                    },
                    **parameters.unit_metadata,
                }
            )
    return pd.DataFrame(rows)


def _save_figure(path: Path, draw: Callable[[plt.Axes], None]) -> None:
    figure, axis = plt.subplots(figsize=(9, 4.5))
    draw(axis)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_figures(frame: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    def concentrations(axis: plt.Axes, prefix: str, title: str) -> None:
        axis.plot(frame["time_seconds"], frame[f"{prefix}_cp"], label="Cp")
        axis.plot(frame["time_seconds"], frame[f"{prefix}_ce"], label="Ce")
        axis.set(xlabel="Time (s)", ylabel="Concentration", title=title)
        axis.legend()

    _save_figure(
        figure_dir / "propofol_cp_ce.png",
        lambda axis: concentrations(axis, "propofol", "Propofol concentration"),
    )
    _save_figure(
        figure_dir / "remifentanil_cp_ce.png",
        lambda axis: concentrations(axis, "remifentanil", "Remifentanil concentration"),
    )

    def bis(axis: plt.Axes) -> None:
        axis.plot(frame["time_seconds"], frame["noiseless_bis"], label="Noiseless BIS")
        axis.plot(frame["time_seconds"], frame["observed_bis"], label="Observed BIS", alpha=0.7)
        axis.axhspan(40.0, 60.0, color="green", alpha=0.12)
        axis.set(xlabel="Time (s)", ylabel="BIS", ylim=(0, 100), title="BIS trajectory")
        axis.legend()

    _save_figure(figure_dir / "bis_trajectory.png", bis)

    def infusion(axis: plt.Axes) -> None:
        axis.plot(frame["time_seconds"], frame["propofol_rate_mg_per_min"], label="Propofol mg/min")
        axis.plot(
            frame["time_seconds"],
            frame["remifentanil_rate_micrograms_per_min"],
            label="Remifentanil microgram/min",
        )
        secondary = axis.twinx()
        secondary.plot(frame["time_seconds"], frame["noiseless_bis"], color="black", label="BIS")
        axis.set(xlabel="Time (s)", ylabel="Infusion rate", title="Infusion and BIS")
        secondary.set(ylabel="BIS", ylim=(0, 100))
        lines = axis.lines + secondary.lines
        axis.legend(lines, [line.get_label() for line in lines], loc="best")

    _save_figure(figure_dir / "infusion_and_bis.png", infusion)
    active = frame.loc[frame["propofol_rate_mg_per_min"].gt(0.0)]
    recovery = frame.loc[frame["time_seconds"].gt(float(active["time_seconds"].max()))]
    _save_figure(
        figure_dir / "recovery_trajectory.png",
        lambda axis: (
            axis.plot(recovery["time_seconds"], recovery["noiseless_bis"]),
            axis.set(xlabel="Time (s)", ylabel="BIS", ylim=(0, 100), title="Recovery after infusion off"),
        ),
    )


def _validation_summary(
    frame: pd.DataFrame,
    integrator_comparison: pd.DataFrame,
    simulator: PKPDSimulator,
) -> dict[str, Any]:
    numeric = frame.select_dtypes(include="number")
    maximum_integrator_difference = float(
        integrator_comparison.filter(like="absolute_difference").to_numpy().max()
    )
    final = simulator.snapshot()
    induction_rows = frame.loc[frame["propofol_rate_mg_per_min"].gt(0.0)]
    last_active_time = float(induction_rows["time_seconds"].max())
    recovery_rows = frame.loc[frame["time_seconds"].gt(last_active_time)]
    checks = {
        "all_numeric_outputs_finite": bool(np.isfinite(numeric.to_numpy()).all()),
        "all_amounts_non_negative": bool(
            (frame[["propofol_x1", "propofol_x2", "propofol_x3", "remifentanil_x1", "remifentanil_x2", "remifentanil_x3"]] >= 0.0)
            .all()
            .all()
        ),
        "integrator_max_abs_difference_below_1e_8": maximum_integrator_difference < 1e-8,
        "propofol_mass_balance_below_1e_10": abs(mass_balance_residual(final.propofol)) < 1e-10,
        "remifentanil_mass_balance_below_1e_10": abs(mass_balance_residual(final.remifentanil)) < 1e-10,
        "induction_bis_decreases": bool(induction_rows["noiseless_bis"].min() < frame["noiseless_bis"].iloc[0]),
        "recovery_direction_present": bool(
            len(recovery_rows) >= 2
            and recovery_rows["noiseless_bis"].iloc[-1] > recovery_rows["noiseless_bis"].iloc[0]
        ),
        "bis_within_physical_bounds": bool(frame["observed_bis"].between(0.0, 100.0).all()),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "maximum_integrator_absolute_difference": maximum_integrator_difference,
        "propofol_mass_balance_residual": mass_balance_residual(final.propofol),
        "remifentanil_mass_balance_residual": mass_balance_residual(final.remifentanil),
        "initial_bis": float(frame["noiseless_bis"].iloc[0]),
        "minimum_bis": float(frame["noiseless_bis"].min()),
        "final_bis": float(frame["noiseless_bis"].iloc[-1]),
        "paper_figure_reproduced": False,
        "paper_figure_comparison": "directional and approximate-range comparison only",
    }


def _manifest(config: ValidationConfig, repo_dir: Path) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_commit": _git_head(repo_dir),
        "source_papers": [
            {
                "citation": "Yun et al., Computers in Biology and Medicine 156 (2023) 106739",
                "equations": "(1)-(32)",
                "table": "Table A.4",
                "pdf_pages": [3, 4, 9],
            },
            {
                "citation": "Yun et al., IEEE TNNLS 35(2) (2024) 2510-2520",
                "equations": "(1)-(6), Appendix (21)-(23)",
                "table": "Table IV",
                "pdf_pages": [2, 3, 4, 10],
            },
            {"citation": "Schnider et al., Anesthesiology 88(5) (1998) 1170-1182"},
            {"citation": "Minto et al., Anesthesiology 86(1) (1997) 10-23"},
        ],
        "equation_registry": "source_equation_registry.json",
        "constants": {
            "schnider_h1_h17": dict(SCHNIDER_CONSTANTS),
            "minto_f1_f18": dict(MINTO_CONSTANTS),
            "propofol_bis_half_effect_mg_per_l": PROPOFOL_HALF_EFFECT_MG_PER_L,
            "remifentanil_bis_half_effect_micrograms_per_l": REMIFENTANIL_HALF_EFFECT_MICROGRAMS_PER_L,
            "bis_exponent": BIS_EXPONENT,
        },
        "internal_units": {
            "time": "minutes in PK matrices; seconds at public API",
            "propofol_amount": "mg",
            "propofol_concentration": "mg/L (microgram/mL)",
            "remifentanil_amount": "microgram",
            "remifentanil_concentration": "microgram/L (ng/mL)",
        },
        "external_input_units": {
            "propofol_rate_mg_per_min": "mg/min",
            "remifentanil_rate_micrograms_per_min": "microgram/min",
            "duration_seconds": "s",
        },
        "integrator": config.integrator,
        "internal_dt_seconds": config.internal_dt_seconds,
        "action_hold_seconds": 10,
        "deterministic": config.deterministic,
        "noise_policy": {
            "deterministic": "no noise",
            "experimental_stochastic": f"additive N({STOCHASTIC_NOISE_MEAN}, variance={STOCHASTIC_NOISE_VARIANCE}) BIS noise",
            "effect_site_drop_implemented": False,
        },
        "demographic_hard_bounds": {
            "age_years": [18, 90],
            "height_cm": [120, 220],
            "weight_kg": [35, 200],
        },
        "source_confirmed_assumptions": [
            "three-compartment PK with first-order effect site",
            "Schnider propofol and Minto remifentanil covariates",
            "additive Yun BIS interaction equation",
            "remifentanil is an exogenous infusion history",
        ],
        "unresolved_assumptions": [
            "Yun LBM typography omits the source-model square",
            "Yun remifentanil Cl1 prints undefined h18; resolved to Minto f18",
            "Yun tables give f12=0.030 while common Minto transcriptions give 0.0301",
            "N(10,0.4) second-parameter convention is not stated",
            "approximately 10% effect-site drug-drop update is underspecified and not implemented",
            "paper figure patient draws, initial conditions, and schedules are not fully disclosed",
        ],
        "supported_state_features": SUPPORTED_STATE_FEATURES,
        "unsupported_vital_signs": UNSUPPORTED_VITAL_SIGNS,
        "clinical_use_prohibition": CLINICAL_PROHIBITION,
        "not_externally_validated": True,
        "not_yet_connected_to_rl": True,
        "gymnasium_environment_implemented": False,
        "rl_training_performed": False,
        "actual_vitaldb_patients_simulated": False,
        "configuration": asdict(config),
    }


def _write_report(path: Path, config: ValidationConfig, summary: Mapping[str, Any]) -> None:
    checks = "\n".join(
        f"- `{name}`: {value}" for name, value in summary["checks"].items()
    )
    path.write_text(
        f"""# PK-PD simulator validation report

{CLINICAL_PROHIBITION}

## Configuration

```json
{json.dumps(asdict(config), indent=2)}
```

## Validation

- Status: **{summary['status']}**
- Initial/minimum/final noiseless BIS: {summary['initial_bis']:.3f} / {summary['minimum_bis']:.3f} / {summary['final_bis']:.3f}
- Maximum exact-vs-solve_ivp absolute difference: {summary['maximum_integrator_absolute_difference']:.3e}
- Propofol mass-balance residual: {summary['propofol_mass_balance_residual']:.3e}
- Remifentanil mass-balance residual: {summary['remifentanil_mass_balance_residual']:.3e}

{checks}

## Interpretation

The induction-like schedule lowers BIS, Cp leads Ce, and BIS recovers after infusion
is stopped. These are directional and approximate-range checks against the published
figures, not exact reproduction. The papers do not disclose every patient draw, initial
condition, remifentanil trajectory, or stochastic realization.

HR, MBP, SBP, DBP, SpO2, ETCO2, and HRV are unsupported. No Gymnasium environment,
PPO agent, actor/critic, or RL training is part of this validation.
""",
        encoding="utf-8",
    )


def run_validation(config: ValidationConfig, output_dir: Path, repo_dir: Path) -> dict[str, Any]:
    """Run deterministic synthetic validation and write auditable artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Running PK-PD validation for %s", config.patient_name)
    trajectory, simulator = simulate_trajectory(config)
    comparison = compare_integrators(SYNTHETIC_PATIENTS[config.patient_name])
    parameters = reference_patient_parameters()
    summary = _validation_summary(trajectory, comparison, simulator)
    trajectory.to_csv(output_dir / "trajectory.csv", index=False)
    comparison.to_csv(output_dir / "integrator_comparison.csv", index=False)
    parameters.to_csv(output_dir / "reference_patient_parameters.csv", index=False)
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    manifest = _manifest(config, repo_dir)
    (output_dir / "simulator_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    source_registry = repo_dir / "outputs/pkpd_simulator_validation/source_equation_registry.json"
    target_registry = output_dir / "source_equation_registry.json"
    if source_registry.resolve() != target_registry.resolve():
        target_registry.write_bytes(source_registry.read_bytes())
    write_figures(trajectory, output_dir / "figures")
    _write_report(output_dir / "pkpd_validation_report.md", config, summary)
    if summary["status"] != "passed":
        raise RuntimeError(f"PK-PD validation failed: {summary['checks']}")
    return summary
