"""Final artifact audit and validation-only exploratory analysis for the PPO pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .cohort import CohortBundle
from .io import atomic_write_dataframe, atomic_write_json, atomic_write_text
from .pilot_analysis import PAIRED_METRICS, paired_patient_differences, run_pilot_analysis
from .pilot_protocol import verify_pilot_protocol


LOWER_IS_BETTER = {
    "bis_target_mae",
    "bis_target_rmse",
    "integrated_absolute_bis_error",
    "fraction_time_bis_below_40",
    "fraction_time_bis_above_60",
    "fraction_time_bis_below_30",
    "total_propofol_dose_mg",
    "mean_propofol_rate_mg_per_min",
    "max_propofol_rate_mg_per_min",
    "propofol_rate_standard_deviation_mg_per_min",
    "absolute_action_change_sum",
    "action_smoothness_mean_absolute_change",
    "evaluation_action_clipping_fraction",
    "lower_action_saturation_fraction",
    "upper_action_saturation_fraction",
}
HIGHER_IS_BETTER = {"fraction_time_in_bis_40_60", "return"}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required pilot artifact is missing: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Pilot JSON artifact root must be an object: {path}")
    return payload


def _assert_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"Pilot artifact {path} is missing columns: {missing}.")


def audit_pilot_artifacts(
    *, protocol: Mapping[str, Any], output_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate all 12 run identities and return audit and validation rows."""

    verify_pilot_protocol(protocol)
    expected_hash = str(protocol["protocol_hash"])
    expected_commit = str(protocol["implementation_commit"])
    expected_cohort = str(protocol["cohort_contract"]["fingerprint"])
    expected_scenarios = list(protocol["cohort_contract"]["validation_scenario_ids"])
    audit_rows: list[dict[str, Any]] = []
    validations: list[pd.DataFrame] = []
    for item in protocol["inventory"]:
        profile = str(item["state_profile"])
        seed = int(item["seed"])
        run_dir = output_root / profile / f"seed_{seed}"
        required = [
            "config.json",
            "protocol_snapshot.json",
            "run_status.json",
            "completion.json",
            "training_progress.csv",
            "action_diagnostics.csv",
            "evaluation_progress.csv",
            "checkpoint_51200.zip",
            "checkpoint_102400.zip",
            "validation_51200.csv",
            "validation_102400.csv",
            "best_model.zip",
            "best_checkpoint.json",
            "cohort_access_manifest.json",
        ]
        missing = [name for name in required if not (run_dir / name).is_file()]
        if missing:
            raise ValueError(f"Pilot run {item['run_id']} is incomplete: {missing}.")
        config = _load_json(run_dir / "config.json")
        status = _load_json(run_dir / "run_status.json")
        completion = _load_json(run_dir / "completion.json")
        snapshot = _load_json(run_dir / "protocol_snapshot.json")
        access = _load_json(run_dir / "cohort_access_manifest.json")
        observed = {
            "profile": config.get("state_profile") == profile,
            "seed": int(config.get("seed", -1)) == seed,
            "protocol": config.get("protocol_hash") == expected_hash,
            "commit": config.get("git_commit_hash") == expected_commit,
            "cohort": config.get("cohort_fingerprint") == expected_cohort,
            "snapshot_hash": snapshot.get("protocol_hash") == expected_hash,
            "status": status.get("status") == "complete",
            "completion": completion.get("status") == "complete",
            "timesteps": int(completion.get("timesteps", -1)) == 102_400,
            "test_config": config.get("test_cohort_accessed") is False,
            "test_status": status.get("test_cohort_accessed") is False,
            "test_completion": completion.get("test_cohort_accessed") is False,
            "test_trajectory": access.get("test_trajectory_loaded") is False,
            "test_outcomes": access.get("test_outcomes_evaluated") is False,
            "test_rollout": access.get("test_policy_rollout_performed") is False,
        }
        failed_checks = [key for key, valid in observed.items() if not valid]
        if failed_checks:
            raise ValueError(
                f"Pilot run {item['run_id']} failed compatibility checks: {failed_checks}."
            )
        training = pd.read_csv(run_dir / "training_progress.csv")
        action = pd.read_csv(run_dir / "action_diagnostics.csv")
        evaluation = pd.read_csv(run_dir / "evaluation_progress.csv")
        if training["timesteps"].astype(int).tolist() != list(range(2_048, 102_401, 2_048)):
            raise ValueError(f"Pilot training progress is misaligned: {item['run_id']}.")
        if action["timesteps"].astype(int).tolist() != list(range(2_048, 102_401, 2_048)):
            raise ValueError(f"Pilot action progress is misaligned: {item['run_id']}.")
        if evaluation["timesteps"].astype(int).tolist() != [51_200, 102_400]:
            raise ValueError(f"Pilot evaluation progress is misaligned: {item['run_id']}.")
        for step in (51_200, 102_400):
            path = run_dir / f"validation_{step}.csv"
            frame = pd.read_csv(path)
            _assert_columns(
                frame,
                {
                    "state_profile",
                    "training_seed",
                    "scenario_id",
                    "patient_id",
                    "cohort_split",
                    "bis_target_mae",
                    "fraction_time_in_bis_40_60",
                    "action_smoothness_mean_absolute_change",
                    "protocol_hash",
                    "cohort_fingerprint",
                },
                path,
            )
            if len(frame) != 15 or frame["scenario_id"].tolist() != expected_scenarios:
                raise ValueError(f"Pilot validation pairing changed: {item['run_id']} step {step}.")
            if set(frame["cohort_split"]) != {"validation"}:
                raise ValueError("Pilot analysis encountered a non-validation trajectory.")
            if set(frame["protocol_hash"]) != {expected_hash}:
                raise ValueError("Pilot validation protocol hash mismatch.")
            if set(frame["cohort_fingerprint"]) != {expected_cohort}:
                raise ValueError("Pilot validation cohort fingerprint mismatch.")
            frame.insert(0, "checkpoint_timesteps", step)
            validations.append(frame)
        audit_rows.append(
            {
                "run_id": item["run_id"],
                "state_profile": profile,
                "seed": seed,
                "status": "complete",
                "training_rows": len(training),
                "action_rows": len(action),
                "evaluation_rows": len(evaluation),
                "validation_rows_per_checkpoint": 15,
                "protocol_hash": expected_hash,
                "git_commit_hash": expected_commit,
                "cohort_fingerprint": expected_cohort,
                "test_trajectory_accessed": False,
                "test_outcomes_evaluated": False,
            }
        )
    return pd.DataFrame(audit_rows), pd.concat(validations, ignore_index=True)


def checkpoint_learning_change(validations: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired aggregate changes between the two pilot checkpoints."""

    metrics = {
        "bis_target_mae": "mean",
        "return": "mean",
        "fraction_time_in_bis_40_60": "mean",
        "evaluation_action_clipping_fraction": "mean",
        "action_smoothness_mean_absolute_change": "mean",
    }
    grouped = (
        validations.groupby(["state_profile", "training_seed", "checkpoint_timesteps"], sort=False)
        .agg(metrics)
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for (profile, seed), frame in grouped.groupby(["state_profile", "training_seed"], sort=False):
        by_step = frame.set_index("checkpoint_timesteps")
        row: dict[str, Any] = {"state_profile": profile, "seed": int(seed)}
        for metric in metrics:
            early = float(by_step.loc[51_200, metric])
            late = float(by_step.loc[102_400, metric])
            row[f"{metric}_51200"] = early
            row[f"{metric}_102400"] = late
            row[f"{metric}_change"] = late - early
        mae_change = row["bis_target_mae_change"]
        tir_change = row["fraction_time_in_bis_40_60_change"]
        if mae_change <= -0.02 and tir_change >= -0.01:
            trend = "improving"
        elif mae_change >= 0.02 or tir_change < -0.01:
            trend = "worsening"
        else:
            trend = "plateau_or_small_change"
        row["trend_rule"] = trend
        rows.append(row)
    return pd.DataFrame(rows)


def seed_rank_summary(validations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank profiles within each seed using the frozen lexicographic metrics."""

    final = validations[validations["checkpoint_timesteps"] == 102_400]
    summary = (
        final.groupby(["training_seed", "state_profile"], sort=False)
        .agg(
            bis_target_mae=("bis_target_mae", "mean"),
            fraction_time_in_bis_40_60=("fraction_time_in_bis_40_60", "mean"),
            action_smoothness=("action_smoothness_mean_absolute_change", "mean"),
            episode_return=("return", "mean"),
        )
        .reset_index()
    )
    summary = summary.sort_values(
        ["training_seed", "bis_target_mae", "fraction_time_in_bis_40_60", "action_smoothness"],
        ascending=[True, True, False, True],
    )
    summary["within_seed_rank"] = summary.groupby("training_seed").cumcount() + 1
    wins = (
        summary.groupby("state_profile", sort=False)
        .agg(
            seed_win_count=("within_seed_rank", lambda values: int((values == 1).sum())),
            mean_rank=("within_seed_rank", "mean"),
            bis_target_mae_mean=("bis_target_mae", "mean"),
            bis_target_mae_std=("bis_target_mae", "std"),
            action_smoothness_mean=("action_smoothness", "mean"),
            action_smoothness_std=("action_smoothness", "std"),
        )
        .reset_index()
        .sort_values(["seed_win_count", "mean_rank"], ascending=[False, True])
    )
    return summary, wins


def patient_difference_summary(validations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe patient-level final-checkpoint differences from original."""

    final = validations[validations["checkpoint_timesteps"] == 102_400]
    paired = paired_patient_differences(final)
    rows: list[dict[str, Any]] = []
    for (profile, metric), frame in paired.groupby(["state_profile", "metric"], sort=False):
        values = frame["difference_candidate_minus_original"].astype(float)
        if metric in LOWER_IS_BETTER:
            improved = values < 0
            worsened = values > 0
        elif metric in HIGHER_IS_BETTER:
            improved = values > 0
            worsened = values < 0
        else:
            improved = pd.Series(False, index=values.index)
            worsened = pd.Series(False, index=values.index)
        rows.append(
            {
                "state_profile": profile,
                "metric": metric,
                "paired_rows": len(values),
                "mean_difference": float(values.mean()),
                "median_difference": float(values.median()),
                "q025_difference": float(values.quantile(0.025)),
                "q975_difference": float(values.quantile(0.975)),
                "improved_count": int(improved.sum()),
                "worsened_count": int(worsened.sum()),
                "tied_count": int((~improved & ~worsened).sum()),
            }
        )
    return paired, pd.DataFrame(rows)


def demographic_subgroups(
    paired: pd.DataFrame, *, cohort: CohortBundle
) -> pd.DataFrame:
    """Compute explicitly exploratory validation-only demographic subgroup summaries."""

    validation_ids = set(cohort.cohort.manifest.validation_patient_ids)
    records = []
    for patient_id in sorted(validation_ids):
        patient = cohort.cohort.patients[patient_id]
        bmi = patient.weight_kg / (patient.height_cm / 100.0) ** 2
        records.append(
            {
                "patient_id": patient_id,
                "age_group": "age_65_plus" if patient.age_years >= 65 else "age_under_65",
                "sex_group": f"sex_{patient.sex}",
                "bmi_group": "bmi_30_plus" if bmi >= 30 else ("bmi_25_to_30" if bmi >= 25 else "bmi_under_25"),
            }
        )
    demographics = pd.DataFrame(records)
    selected = paired[paired["metric"].isin({"bis_target_mae", "action_smoothness_mean_absolute_change"})]
    selected = selected.copy()
    selected["patient_id"] = selected["patient_id"].astype(str)
    demographics["patient_id"] = demographics["patient_id"].astype(str)
    selected = selected.merge(demographics, on="patient_id", how="inner", validate="many_to_one")
    rows: list[dict[str, Any]] = []
    for group_type in ("age_group", "sex_group", "bmi_group"):
        for keys, frame in selected.groupby(["state_profile", "training_seed", "metric", group_type], sort=False):
            profile, seed, metric, label = keys
            values = frame["difference_candidate_minus_original"].astype(float)
            rows.append(
                {
                    "state_profile": profile,
                    "seed": int(seed),
                    "metric": metric,
                    "subgroup_type": group_type,
                    "subgroup": label,
                    "patient_count": frame["patient_id"].nunique(),
                    "mean_difference": float(values.mean()),
                    "median_difference": float(values.median()),
                    "exploratory_only": True,
                }
            )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, maximum_rows: int = 50) -> str:
    if frame.empty:
        return "_No rows._"
    shown = frame.head(maximum_rows).copy()
    columns = list(shown.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in shown.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if np.isnan(value) else f"{value:.5g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run_pilot_final_audit(
    *,
    protocol: dict[str, Any],
    output_root: Path,
    existing_analysis_dir: Path,
    audit_dir: Path,
    cohort: CohortBundle,
) -> dict[str, Any]:
    """Regenerate the base analysis, audit artifacts, and write new exploratory outputs."""

    base = run_pilot_analysis(
        protocol=protocol,
        output_root=output_root,
        analysis_dir=existing_analysis_dir,
    )
    if base["completed_runs"] != 12 or base["failed_runs"] or base["pending_runs"]:
        raise ValueError(f"Base pilot analysis is not complete: {base}.")
    audit, validations = audit_pilot_artifacts(protocol=protocol, output_root=output_root)
    learning = checkpoint_learning_change(validations)
    ranks, wins = seed_rank_summary(validations)
    paired, paired_summary = patient_difference_summary(validations)
    subgroups = demographic_subgroups(paired, cohort=cohort)
    selected_84 = paired[
        (paired["state_profile"] == "selected_control_core")
        & (paired["training_seed"] == 84)
    ].sort_values(["metric", "difference_candidate_minus_original"])
    audit_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pilot_artifact_audit.csv": audit,
        "checkpoint_learning_change.csv": learning,
        "seed_profile_ranks.csv": ranks,
        "seed_win_summary.csv": wins,
        "patient_level_paired_differences.csv": paired,
        "patient_level_difference_summary.csv": paired_summary,
        "demographic_subgroup_exploratory.csv": subgroups,
        "selected_control_core_seed84_patient_differences.csv": selected_84,
    }
    for name, frame in outputs.items():
        atomic_write_dataframe(audit_dir / name, frame)
    improving = learning.groupby("state_profile")["trend_rule"].value_counts().unstack(fill_value=0).reset_index()
    report = f"""# Final Primary-State PPO Pilot Audit

This is validation-only exploratory analysis. It does not select a final state or
establish clinical safety.

## Artifact Integrity

- Runs: 12 complete / 0 failed / 0 pending
- Protocol hash: `{protocol['protocol_hash']}`
- Implementation commit: `{protocol['implementation_commit']}`
- Cohort fingerprint: `{protocol['cohort_contract']['fingerprint']}`
- Test trajectories, outcomes, rollouts, and metrics: not accessed
- Each run: 50 training rows, 50 action rows, 2 evaluation rows, and 15 paired
  validation patients at both 51,200 and 102,400 steps

## Checkpoint Trend Counts

{_markdown_table(improving)}

## Seed Ranks and Wins

{_markdown_table(ranks)}

{_markdown_table(wins)}

## Patient-Level BIS MAE Differences Versus Original

{_markdown_table(paired_summary[paired_summary['metric'] == 'bis_target_mae'])}

## Interpretation

All four profiles completed every seed without numerical failure and remain eligible
for the full study. The 102,400-step checkpoint is too short for final state selection.
Patient and demographic subgroup summaries are descriptive diagnostics over only 15
validation patients; they must not be presented as subgroup efficacy claims. The
full protocol must start every run from a fresh random initialization and keep test
data sealed until the validation-selected state is frozen.
"""
    atomic_write_text(audit_dir / "pilot_final_audit.md", report)
    manifest = {
        "protocol_hash": protocol["protocol_hash"],
        "implementation_commit": protocol["implementation_commit"],
        "cohort_fingerprint": protocol["cohort_contract"]["fingerprint"],
        "completed_runs": len(audit),
        "failed_runs": 0,
        "pending_runs": 0,
        "test_trajectory_accessed": False,
        "test_outcomes_evaluated": False,
        "final_winner_selected": False,
        "output_files": sorted([*outputs, "pilot_final_audit.md"]),
    }
    atomic_write_json(audit_dir / "pilot_final_audit_manifest.json", manifest)
    return manifest
