"""Validation-only aggregation and exploratory reporting for the PPO state pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import atomic_write_dataframe, atomic_write_json, atomic_write_text
from .pilot_protocol import PILOT_PROFILES, PILOT_SEEDS, verify_pilot_protocol


PAIRED_METRICS = (
    "bis_target_mae",
    "bis_target_rmse",
    "integrated_absolute_bis_error",
    "fraction_time_in_bis_40_60",
    "bis_below_40_duration_seconds",
    "fraction_time_bis_below_40",
    "bis_above_60_duration_seconds",
    "fraction_time_bis_above_60",
    "bis_below_30_duration_seconds",
    "fraction_time_bis_below_30",
    "return",
    "total_propofol_dose_mg",
    "mean_propofol_rate_mg_per_min",
    "max_propofol_rate_mg_per_min",
    "absolute_action_change_sum",
    "action_smoothness_mean_absolute_change",
    "propofol_rate_standard_deviation_mg_per_min",
    "lower_action_saturation_fraction",
    "upper_action_saturation_fraction",
    "evaluation_action_clipping_fraction",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _concat(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    values = [frame for frame in frames if not frame.empty]
    return pd.concat(values, ignore_index=True) if values else pd.DataFrame()


def _markdown_table(frame: pd.DataFrame, *, maximum_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows available._"
    shown = frame.head(maximum_rows) if maximum_rows is not None else frame
    columns = [str(column) for column in shown.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in shown.iterrows():
        values = []
        for value in row:
            if pd.isna(value):
                text = ""
            elif isinstance(value, float):
                text = f"{value:.5g}"
            else:
                text = str(value)
            values.append(text.replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _validate_validation_frame(
    frame: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
    run_dir: Path,
) -> None:
    if set(frame["cohort_split"]) != {"validation"}:
        raise ValueError(f"Non-validation rows found in {run_dir}.")
    expected = list(protocol["cohort_contract"]["validation_scenario_ids"])
    observed = frame["scenario_id"].tolist()
    if observed != expected:
        raise ValueError(f"Paired validation scenario order changed in {run_dir}.")
    if frame["patient_id"].duplicated().any():
        raise ValueError(f"Validation patient rows are not one-to-one in {run_dir}.")


def collect_pilot_results(
    protocol: Mapping[str, Any], output_root: Path
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    """Load compatible completed/failed/pending artifacts without test access."""

    verify_pilot_protocol(protocol)
    run_rows: list[dict[str, Any]] = []
    evaluation_frames: list[pd.DataFrame] = []
    patient_frames: list[pd.DataFrame] = []
    learning_frames: list[pd.DataFrame] = []
    action_frames: list[pd.DataFrame] = []
    failed: list[dict[str, Any]] = []
    pending: list[str] = []
    for item in protocol["inventory"]:
        profile = str(item["state_profile"])
        seed = int(item["seed"])
        run_id = str(item["run_id"])
        run_dir = output_root / profile / f"seed_{seed}"
        completion = _load_json(run_dir / "completion.json")
        status = _load_json(run_dir / "run_status.json")
        if not completion:
            observed_status = status.get("status", "pending")
            run_rows.append(
                {
                    "run_id": run_id,
                    "state_profile": profile,
                    "seed": seed,
                    "status": observed_status,
                    "timesteps": status.get("resolved_config", {}).get(
                        "resume_timestep", 0
                    ),
                    "protocol_hash": protocol["protocol_hash"],
                }
            )
            if observed_status == "failed":
                failed.append(
                    {
                        "run_id": run_id,
                        "exception_type": status.get("exception_type"),
                        "exception_message": status.get("exception_message"),
                        "traceback": status.get("traceback"),
                        "last_checkpoint": status.get("last_checkpoint"),
                    }
                )
            else:
                pending.append(run_id)
            continue
        if completion.get("protocol_hash") != protocol["protocol_hash"]:
            raise ValueError(f"Completion protocol hash mismatch in {run_dir}.")
        if completion.get("test_cohort_accessed") is not False:
            raise ValueError(f"Test seal is not explicit in {run_dir}.")
        best = _load_json(run_dir / "best_checkpoint.json")
        validation = pd.read_csv(run_dir / str(best["validation_file"]))
        _validate_validation_frame(validation, protocol=protocol, run_dir=run_dir)
        validation["selected_checkpoint"] = True
        patient_frames.append(validation)
        evaluation = pd.read_csv(run_dir / "evaluation_progress.csv")
        evaluation_frames.append(evaluation)
        learning = pd.read_csv(run_dir / "training_progress.csv")
        learning_frames.append(learning)
        action = pd.read_csv(run_dir / "action_diagnostics.csv")
        action_columns = [
            "state_profile",
            "seed",
            "timesteps",
            "raw_normalized_action_minimum",
            "raw_normalized_action_maximum",
            "bounded_normalized_action_minimum",
            "bounded_normalized_action_maximum",
            "normalized_clipping_fraction",
            "near_lower_boundary_fraction",
            "near_upper_boundary_fraction",
        ]
        action = action.merge(
            learning[action_columns],
            on=["state_profile", "seed", "timesteps"],
            how="left",
            validate="one_to_one",
        )
        action_frames.append(action)
        counts = _load_json(run_dir / "parameter_counts.json")
        run_rows.append(
            {
                "run_id": run_id,
                "state_profile": profile,
                "seed": seed,
                "status": "complete",
                "timesteps": completion["timesteps"],
                "best_checkpoint_timesteps": best["timesteps"],
                "validation_bis_target_mae": best["validation_bis_target_mae"],
                "validation_fraction_time_in_bis_40_60": best[
                    "validation_fraction_time_in_bis_40_60"
                ],
                "validation_episode_return": best["mean_episode_return"],
                "validation_integrated_absolute_bis_error": float(
                    validation["integrated_absolute_bis_error"].mean()
                ),
                "validation_fraction_time_bis_below_40": float(
                    validation["fraction_time_bis_below_40"].mean()
                ),
                "validation_fraction_time_bis_above_60": float(
                    validation["fraction_time_bis_above_60"].mean()
                ),
                "validation_fraction_time_bis_below_30": float(
                    validation["fraction_time_bis_below_30"].mean()
                ),
                "validation_total_propofol_dose_mg": float(
                    validation["total_propofol_dose_mg"].mean()
                ),
                "validation_mean_propofol_rate_mg_per_min": float(
                    validation["mean_propofol_rate_mg_per_min"].mean()
                ),
                "validation_maximum_physical_action_mg_per_min": float(
                    validation["physical_action_maximum_mg_per_min"].max()
                ),
                "validation_raw_normalized_action_minimum": float(
                    validation["raw_normalized_action_minimum"].min()
                ),
                "validation_raw_normalized_action_maximum": float(
                    validation["raw_normalized_action_maximum"].max()
                ),
                "validation_action_clipping_fraction": best[
                    "mean_action_clipping_fraction"
                ],
                "validation_action_smoothness": float(
                    validation["action_smoothness_mean_absolute_change"].mean()
                ),
                "validation_large_rate_change_count": float(
                    validation["excessive_action_change_count"].mean()
                ),
                "validation_lower_action_saturation_fraction": float(
                    validation["lower_action_saturation_fraction"].mean()
                ),
                "validation_upper_action_saturation_fraction": float(
                    validation["upper_action_saturation_fraction"].mean()
                ),
                "validation_failure_episode_count": int(
                    (validation["numerical_failures"] > 0).sum()
                ),
                "observation_dimension": protocol["policy_contracts"][profile][
                    "observation_dimension"
                ],
                "policy_parameter_count": counts["total_policy_trainable_parameters"],
                "training_steps_per_second": float(
                    learning["training_steps_per_second"].mean()
                ),
                "training_elapsed_seconds": completion[
                    "total_training_elapsed_seconds"
                ],
                "protocol_hash": protocol["protocol_hash"],
                "cohort_fingerprint": protocol["cohort_contract"]["fingerprint"],
                "git_commit_hash": protocol["implementation_commit"],
                "device": protocol["execution_device"],
            }
        )
    return {
        "runs": pd.DataFrame(run_rows),
        "evaluations": _concat(evaluation_frames),
        "patients": _concat(patient_frames),
        "learning": _concat(learning_frames),
        "actions": _concat(action_frames),
        "failures": {"failed_runs": failed, "pending_runs": pending},
    }


def paired_patient_differences(frame: pd.DataFrame) -> pd.DataFrame:
    """Pair each candidate against original by seed, patient, and scenario identity."""

    if frame.empty:
        return pd.DataFrame()
    keys = ["training_seed", "scenario_id", "patient_id"]
    baseline = frame.loc[
        frame["state_profile"] == "original_reconstructed", keys + list(PAIRED_METRICS)
    ]
    rows: list[pd.DataFrame] = []
    for profile in PILOT_PROFILES[1:]:
        candidate = frame.loc[
            frame["state_profile"] == profile, keys + list(PAIRED_METRICS)
        ]
        if candidate.empty or baseline.empty:
            continue
        merged = candidate.merge(
            baseline,
            on=keys,
            suffixes=("_candidate", "_original"),
            validate="one_to_one",
        )
        for metric in PAIRED_METRICS:
            rows.append(
                pd.DataFrame(
                    {
                        **{key: merged[key] for key in keys},
                        "state_profile": profile,
                        "reference_profile": "original_reconstructed",
                        "metric": metric,
                        "candidate_value": merged[f"{metric}_candidate"],
                        "original_value": merged[f"{metric}_original"],
                        "difference_candidate_minus_original": (
                            merged[f"{metric}_candidate"]
                            - merged[f"{metric}_original"]
                        ),
                    }
                )
            )
    return _concat(rows)


def _save_line_plot(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    output: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if not frame.empty and y in frame:
        for (profile, seed), group in frame.groupby(["state_profile", "seed"]):
            ordered = group.sort_values(x)
            ax.plot(ordered[x], ordered[y], marker="o", label=f"{profile} / {seed}")
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    ax.grid(alpha=0.25)
    if ax.lines:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _save_paired_plot(frame: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    required = {
        "metric",
        "state_profile",
        "difference_candidate_minus_original",
    }
    selected = (
        frame.loc[frame["metric"] == "bis_target_mae"]
        if required.issubset(frame.columns)
        else pd.DataFrame(columns=sorted(required))
    )
    values = [
        selected.loc[
            selected["state_profile"] == profile,
            "difference_candidate_minus_original",
        ].to_numpy(float)
        for profile in PILOT_PROFILES[1:]
    ]
    nonempty = [(profile, value) for profile, value in zip(PILOT_PROFILES[1:], values) if len(value)]
    if nonempty:
        ax.boxplot([value for _, value in nonempty], tick_labels=[name for name, _ in nonempty])
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Patient-level paired BIS MAE difference")
    ax.set_ylabel("candidate minus original")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run_pilot_analysis(
    *,
    protocol: dict[str, Any],
    output_root: Path,
    analysis_dir: Path,
) -> dict[str, Any]:
    """Create all prespecified pilot tables, plots, and an explicitly exploratory report."""

    analysis_dir.mkdir(parents=True, exist_ok=True)
    collected = collect_pilot_results(protocol, output_root)
    runs = collected["runs"]
    evaluations = collected["evaluations"]
    patients = collected["patients"]
    learning = collected["learning"]
    actions = collected["actions"]
    assert isinstance(runs, pd.DataFrame)
    assert isinstance(evaluations, pd.DataFrame)
    assert isinstance(patients, pd.DataFrame)
    assert isinstance(learning, pd.DataFrame)
    assert isinstance(actions, pd.DataFrame)
    failures = collected["failures"]
    assert isinstance(failures, dict)

    paired = paired_patient_differences(patients)
    atomic_write_dataframe(analysis_dir / "run_level_summary.csv", runs)
    atomic_write_dataframe(
        analysis_dir / "evaluation_checkpoint_summary.csv", evaluations
    )
    atomic_write_dataframe(
        analysis_dir / "patient_level_paired_metrics.csv", paired
    )
    atomic_write_dataframe(analysis_dir / "action_diagnostics.csv", actions)
    atomic_write_dataframe(analysis_dir / "learning_curve.csv", learning)
    atomic_write_json(analysis_dir / "failed_runs_manifest.json", failures)

    completed = runs.loc[runs["status"] == "complete"].copy()
    metric_columns = [
        "validation_bis_target_mae",
        "validation_fraction_time_in_bis_40_60",
        "validation_episode_return",
        "validation_action_clipping_fraction",
        "validation_lower_action_saturation_fraction",
        "validation_upper_action_saturation_fraction",
        "validation_action_smoothness",
        "training_steps_per_second",
        "training_elapsed_seconds",
    ]
    if completed.empty:
        profile_summary = pd.DataFrame()
    else:
        profile_summary = completed.groupby("state_profile")[metric_columns].agg(
            ["mean", "std"]
        )
        profile_summary.columns = [f"{name}_{stat}" for name, stat in profile_summary.columns]
        profile_summary = profile_summary.reset_index()
    atomic_write_dataframe(analysis_dir / "profile_mean_sd.csv", profile_summary)

    figures = analysis_dir / "figures"
    figures.mkdir(exist_ok=True)
    _save_line_plot(
        evaluations,
        x="timesteps",
        y="mean_episode_return",
        output=figures / "learning_curve_return.png",
        title="Validation episode return across pilot checkpoints",
    )
    _save_line_plot(
        evaluations,
        x="timesteps",
        y="validation_bis_target_mae",
        output=figures / "validation_bis_mae.png",
        title="Validation BIS target MAE",
    )
    _save_line_plot(
        evaluations,
        x="timesteps",
        y="validation_fraction_time_in_bis_40_60",
        output=figures / "validation_time_in_range.png",
        title="Validation fraction of time in BIS 40-60",
    )
    _save_line_plot(
        actions,
        x="timesteps",
        y="clipping_rate",
        output=figures / "training_action_clipping.png",
        title="Training action clipping by PPO rollout",
    )
    smooth = patients[
        ["state_profile", "training_seed", "action_smoothness_mean_absolute_change"]
    ].rename(
        columns={
            "training_seed": "seed",
            "action_smoothness_mean_absolute_change": "action_smoothness",
        }
    ) if not patients.empty else pd.DataFrame()
    if not smooth.empty:
        smooth["patient_index"] = smooth.groupby(["state_profile", "seed"]).cumcount()
    _save_line_plot(
        smooth,
        x="patient_index",
        y="action_smoothness",
        output=figures / "validation_action_smoothness.png",
        title="Validation patient action smoothness",
    )
    _save_paired_plot(paired, figures / "patient_paired_bis_mae_difference.png")

    completion_table = runs[
        ["run_id", "state_profile", "seed", "status", "timesteps"]
    ]
    result_table = completed[
        [
            "state_profile",
            "seed",
            "validation_bis_target_mae",
            "validation_fraction_time_in_bis_40_60",
            "validation_episode_return",
            "validation_action_clipping_fraction",
            "validation_lower_action_saturation_fraction",
            "validation_upper_action_saturation_fraction",
            "validation_action_smoothness",
            "validation_failure_episode_count",
        ]
    ] if not completed.empty else pd.DataFrame()
    dimensions = completed[
        [
            "state_profile",
            "seed",
            "observation_dimension",
            "policy_parameter_count",
            "training_steps_per_second",
        ]
    ] if not completed.empty else pd.DataFrame()
    clipping = (
        actions.groupby(["state_profile", "seed", "training_phase"], as_index=False)[
            ["action_count", "clipping_count"]
        ].sum()
        if not actions.empty
        else pd.DataFrame()
    )
    if not clipping.empty:
        clipping["clipping_fraction"] = clipping["clipping_count"] / clipping["action_count"]
    saturation = completed[
        [
            "state_profile",
            "seed",
            "validation_raw_normalized_action_minimum",
            "validation_raw_normalized_action_maximum",
            "validation_action_clipping_fraction",
            "validation_lower_action_saturation_fraction",
            "validation_upper_action_saturation_fraction",
            "validation_action_smoothness",
        ]
    ] if not completed.empty else pd.DataFrame()

    report = f"""# Primary-State PPO Pilot Report

This report is an exploratory 102,400-step pilot. It does not select a final winner,
establish clinical performance, or justify profile-specific hyperparameter tuning.

## A. Protocol

| Field | Value |
| --- | --- |
| Protocol hash | `{protocol['protocol_hash']}` |
| Implementation commit | `{protocol['implementation_commit']}` |
| Cohort fingerprint | `{protocol['cohort_contract']['fingerprint']}` |
| Device | `{protocol['execution_device']}` |
| Profiles | {', '.join(protocol['profiles'])} |
| Seeds | {protocol['seeds']} |
| Timesteps per run | {protocol['ppo']['total_timesteps']} |
| Validation interval | {protocol['ppo']['evaluation_frequency_timesteps']} |
| Test trajectory/outcome access | prohibited |

## B. 12-Run Completion

{_markdown_table(completion_table)}

## C. Profile by Seed Final Metrics

{_markdown_table(result_table)}

## D. Profile Mean and SD

{_markdown_table(profile_summary)}

## E. Paired Differences Versus Original

{_markdown_table(paired.groupby(['state_profile', 'metric'], as_index=False)['difference_candidate_minus_original'].mean() if not paired.empty else pd.DataFrame(), maximum_rows=80)}

## F. Observation Dimension, Parameters, Throughput

{_markdown_table(dimensions)}

## G. Action Clipping by Training Phase

{_markdown_table(clipping)}

### Validation Clipping, Saturation, and Smoothness

{_markdown_table(saturation)}

## Interpretation Boundary

Only gross failure, non-finite behavior, severe saturation, consistently poor control
across all three seeds, full-run viability, runtime, and storage should be judged here.
Plots and differences are pilot diagnostics, not confirmatory statistical conclusions.
"""
    atomic_write_text(analysis_dir / "pilot_report.md", report)
    reproducibility = {
        "protocol_hash": protocol["protocol_hash"],
        "implementation_commit": protocol["implementation_commit"],
        "cohort_fingerprint": protocol["cohort_contract"]["fingerprint"],
        "execution_device": protocol["execution_device"],
        "expected_runs": 12,
        "completed_runs": int((runs["status"] == "complete").sum()),
        "failed_runs": len(failures["failed_runs"]),
        "pending_runs": len(failures["pending_runs"]),
        "paired_validation_scenario_ids": protocol["cohort_contract"][
            "validation_scenario_ids"
        ],
        "test_cohort_accessed": False,
        "final_winner_selected": False,
    }
    atomic_write_json(analysis_dir / "reproducibility_manifest.json", reproducibility)
    return reproducibility
