"""Validation-only aggregation and hierarchical uncertainty for full PPO runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .full_protocol import FULL_PROFILES, verify_full_protocol
from .io import atomic_write_dataframe, atomic_write_json, atomic_write_text
from .pilot_analysis import PAIRED_METRICS, paired_patient_differences


PRIMARY_METRICS = (
    "bis_target_mae",
    "fraction_time_in_bis_40_60",
    "action_smoothness_mean_absolute_change",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Full artifact root must be an object: {path}")
    return payload


def _concat(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    items = [item for item in frames if not item.empty]
    return pd.concat(items, ignore_index=True) if items else pd.DataFrame()


def collect_full_results(
    *, protocol: Mapping[str, Any], output_root: Path
) -> dict[str, pd.DataFrame]:
    """Collect compatible complete runs while retaining pending/failed inventory rows."""

    verify_full_protocol(protocol)
    run_rows: list[dict[str, Any]] = []
    evaluation_frames: list[pd.DataFrame] = []
    validation_frames: list[pd.DataFrame] = []
    training_frames: list[pd.DataFrame] = []
    action_frames: list[pd.DataFrame] = []
    total = int(protocol["ppo"]["total_timesteps"])
    frequency = int(protocol["ppo"]["evaluation_frequency_timesteps"])
    expected_steps = list(range(frequency, total + 1, frequency))
    expected_scenarios = protocol["cohort_contract"]["validation_scenario_ids"]
    for item in protocol["inventory"]:
        profile = str(item["state_profile"])
        seed = int(item["seed"])
        run_dir = output_root / profile / f"seed_{seed}"
        status_path = run_dir / "run_status.json"
        completion_path = run_dir / "completion.json"
        status = _load_json(status_path) if status_path.is_file() else {}
        completion = _load_json(completion_path) if completion_path.is_file() else {}
        state = "complete" if completion.get("status") == "complete" else status.get("status", "pending")
        base_row = {
            "run_id": item["run_id"],
            "state_profile": profile,
            "seed": seed,
            "status": state,
            "timesteps": int(completion.get("timesteps", 0)),
            "protocol_hash": protocol["protocol_hash"],
            "cohort_fingerprint": protocol["cohort_contract"]["fingerprint"],
            "git_commit_hash": protocol["implementation_commit"],
            "device": protocol["execution_device"],
        }
        if state != "complete":
            run_rows.append(base_row)
            continue
        config = _load_json(run_dir / "config.json")
        if config.get("workflow") != "primary_state_ppo_full":
            raise ValueError(f"Completed full run has a non-full workflow: {item['run_id']}.")
        required_values = {
            "protocol_hash": protocol["protocol_hash"],
            "cohort_fingerprint": protocol["cohort_contract"]["fingerprint"],
            "git_commit_hash": protocol["implementation_commit"],
            "device": protocol["execution_device"],
            "initialization_source": "fresh_random",
            "pilot_checkpoint_used": False,
            "test_cohort_accessed": False,
        }
        mismatches = {
            key: (expected, config.get(key))
            for key, expected in required_values.items()
            if config.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Completed full run is incompatible: {item['run_id']}: {mismatches}.")
        if completion.get("test_cohort_accessed") is not False:
            raise ValueError("Full analysis detected test-cohort access.")
        expected_files = [
            run_dir / "best_model.zip",
            run_dir / "best_checkpoint.json",
            run_dir / "training_progress.csv",
            run_dir / "evaluation_progress.csv",
            run_dir / "action_diagnostics.csv",
        ]
        expected_files.extend(run_dir / f"checkpoint_{step}.zip" for step in expected_steps)
        expected_files.extend(run_dir / f"validation_{step}.csv" for step in expected_steps)
        missing = [str(path) for path in expected_files if not path.is_file()]
        if missing:
            raise ValueError(f"Completed full run has missing artifacts: {missing[:5]}.")
        evaluation = pd.read_csv(run_dir / "evaluation_progress.csv")
        if evaluation["timesteps"].astype(int).tolist() != expected_steps:
            raise ValueError(f"Full evaluation schedule mismatch: {item['run_id']}.")
        evaluation_frames.append(evaluation)
        best = _load_json(run_dir / "best_checkpoint.json")
        best_step = int(best["timesteps"])
        validation = pd.read_csv(run_dir / f"validation_{best_step}.csv")
        if len(validation) != 15 or validation["scenario_id"].tolist() != expected_scenarios:
            raise ValueError(f"Full paired validation identities changed: {item['run_id']}.")
        if set(validation["cohort_split"]) != {"validation"}:
            raise ValueError("Full analysis encountered a non-validation trajectory.")
        validation["best_checkpoint_timesteps"] = best_step
        validation_frames.append(validation)
        training = pd.read_csv(run_dir / "training_progress.csv")
        action = pd.read_csv(run_dir / "action_diagnostics.csv")
        training_frames.append(training)
        action_frames.append(action)
        run_rows.append(
            {
                **base_row,
                "best_checkpoint_timesteps": best_step,
                "validation_bis_target_mae": float(validation["bis_target_mae"].mean()),
                "validation_bis_target_rmse": float(validation["bis_target_rmse"].mean()),
                "validation_integrated_absolute_bis_error": float(validation["integrated_absolute_bis_error"].mean()),
                "validation_fraction_time_in_bis_40_60": float(validation["fraction_time_in_bis_40_60"].mean()),
                "validation_fraction_time_bis_above_60": float(validation["fraction_time_bis_above_60"].mean()),
                "validation_fraction_time_bis_below_40": float(validation["fraction_time_bis_below_40"].mean()),
                "validation_fraction_time_bis_below_30": float(validation["fraction_time_bis_below_30"].mean()),
                "validation_induction_settling_time_seconds": float(validation["induction_settling_time_seconds"].mean()),
                "validation_episode_return": float(validation["return"].mean()),
                "validation_total_propofol_dose_mg": float(validation["total_propofol_dose_mg"].mean()),
                "validation_mean_propofol_rate_mg_per_min": float(validation["mean_propofol_rate_mg_per_min"].mean()),
                "validation_max_propofol_rate_mg_per_min": float(validation["max_propofol_rate_mg_per_min"].max()),
                "validation_action_clipping_fraction": float(validation["evaluation_action_clipping_fraction"].mean()),
                "validation_lower_action_saturation_fraction": float(validation["lower_action_saturation_fraction"].mean()),
                "validation_upper_action_saturation_fraction": float(validation["upper_action_saturation_fraction"].mean()),
                "validation_mean_absolute_action_change": float(validation["action_smoothness_mean_absolute_change"].mean()),
                "validation_action_change_sum": float(validation["absolute_action_change_sum"].mean()),
                "validation_large_change_count": float(validation["excessive_action_change_count"].mean()),
                "validation_failure_episode_count": int((validation["numerical_failures"] > 0).sum()),
                "observation_dimension": int(config["observation_dimension"]),
                "policy_parameter_count": int(status["resolved_config"]["total_trainable_parameters"]),
                "training_elapsed_seconds": float(training["chunk_elapsed_seconds"].sum()),
                "training_steps_per_second": float(total / training["chunk_elapsed_seconds"].sum()),
            }
        )
    return {
        "run_level": pd.DataFrame(run_rows),
        "evaluation": _concat(evaluation_frames),
        "validation": _concat(validation_frames),
        "training": _concat(training_frames),
        "action": _concat(action_frames),
    }


def hierarchical_bootstrap_intervals(
    paired: pd.DataFrame,
    *,
    metrics: Iterable[str] = PRIMARY_METRICS,
    repeats: int = 5_000,
    random_seed: int = 20_260_719,
) -> pd.DataFrame:
    """Bootstrap training seeds, then validation patients within sampled seeds."""

    if repeats <= 0:
        raise ValueError("Bootstrap repeats must be positive.")
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []
    selected = paired[paired["metric"].isin(set(metrics))]
    for (profile, metric), frame in selected.groupby(["state_profile", "metric"], sort=False):
        seeds = np.array(sorted(frame["training_seed"].unique()), dtype=int)
        if len(seeds) < 2:
            continue
        distributions = []
        by_seed = {
            int(seed): group["difference_candidate_minus_original"].to_numpy(dtype=float)
            for seed, group in frame.groupby("training_seed")
        }
        for _ in range(repeats):
            sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            seed_means = []
            for sampled_seed in sampled_seeds:
                values = by_seed[int(sampled_seed)]
                seed_means.append(float(rng.choice(values, size=len(values), replace=True).mean()))
            distributions.append(float(np.mean(seed_means)))
        observed = float(frame.groupby("training_seed")["difference_candidate_minus_original"].mean().mean())
        rows.append(
            {
                "state_profile": profile,
                "reference_profile": "original_reconstructed",
                "metric": metric,
                "candidate_minus_original_difference": observed,
                "hierarchical_bootstrap_ci95_lower": float(np.quantile(distributions, 0.025)),
                "hierarchical_bootstrap_ci95_upper": float(np.quantile(distributions, 0.975)),
                "training_seed_count": len(seeds),
                "validation_patient_count_per_seed": int(frame.groupby("training_seed")["patient_id"].nunique().min()),
                "bootstrap_repeats": repeats,
                "p_value_reported": False,
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, maximum_rows: int = 40) -> str:
    if frame.empty:
        return "_No completed rows._"
    shown = frame.head(maximum_rows)
    columns = list(shown.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in shown.iterrows():
        values = [f"{value:.5g}" if isinstance(value, float) and np.isfinite(value) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run_full_analysis(
    *,
    protocol: dict[str, Any],
    output_root: Path,
    analysis_dir: Path,
    bootstrap_repeats: int = 5_000,
) -> dict[str, Any]:
    """Write partial-safe full-study validation outputs without touching test data."""

    collected = collect_full_results(protocol=protocol, output_root=output_root)
    runs = collected["run_level"]
    complete = runs[runs["status"] == "complete"]
    validation = collected["validation"]
    paired = paired_patient_differences(validation) if not validation.empty else pd.DataFrame()
    intervals = (
        hierarchical_bootstrap_intervals(paired, repeats=bootstrap_repeats)
        if not paired.empty
        else pd.DataFrame()
    )
    metric_columns = [
        "validation_bis_target_mae",
        "validation_fraction_time_in_bis_40_60",
        "validation_mean_absolute_action_change",
        "validation_episode_return",
        "validation_action_clipping_fraction",
        "training_steps_per_second",
    ]
    profile_summary = (
        complete.groupby("state_profile")[metric_columns].agg(["mean", "std"]).reset_index()
        if not complete.empty
        else pd.DataFrame()
    )
    if not profile_summary.empty:
        profile_summary.columns = [
            column if isinstance(column, str) else "_".join(part for part in column if part)
            for column in profile_summary.columns
        ]
    rank_rows = []
    if not complete.empty:
        ranked = complete.sort_values(
            ["seed", "validation_bis_target_mae", "validation_fraction_time_in_bis_40_60", "validation_mean_absolute_action_change"],
            ascending=[True, True, False, True],
        ).copy()
        ranked["within_seed_rank"] = ranked.groupby("seed").cumcount() + 1
        rank_rows = ranked[["seed", "state_profile", "within_seed_rank"]].to_dict("records")
    seed_ranks = pd.DataFrame(rank_rows)
    seed_wins = (
        seed_ranks.groupby("state_profile").agg(
            seed_win_count=("within_seed_rank", lambda values: int((values == 1).sum())),
            mean_rank=("within_seed_rank", "mean"),
        ).reset_index()
        if not seed_ranks.empty
        else pd.DataFrame()
    )
    learning = collected["training"]
    learning_summary = (
        learning.groupby(["state_profile", "timesteps"]).agg(
            episode_return_mean=("episode_return_mean", "mean"),
            episode_return_std=("episode_return_mean", "std"),
            approximate_kl_mean=("approximate_kl", "mean"),
            ppo_clip_fraction_mean=("ppo_clip_fraction", "mean"),
            explained_variance_mean=("explained_variance", "mean"),
        ).reset_index()
        if not learning.empty
        else pd.DataFrame()
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "run_level_summary.csv": runs,
        "evaluation_checkpoint_summary.csv": collected["evaluation"],
        "patient_level_best_checkpoint_metrics.csv": validation,
        "patient_level_paired_differences.csv": paired,
        "profile_five_seed_mean_sd.csv": profile_summary,
        "seed_profile_ranks.csv": seed_ranks,
        "seed_win_counts.csv": seed_wins,
        "hierarchical_bootstrap_intervals.csv": intervals,
        "learning_curve_mean_variability.csv": learning_summary,
        "action_diagnostics.csv": collected["action"],
    }
    for filename, frame in tables.items():
        atomic_write_dataframe(analysis_dir / filename, frame)
    failed = runs[runs["status"] == "failed"]["run_id"].tolist()
    pending = runs[~runs["status"].isin(["complete", "failed"])]["run_id"].tolist()
    manifest = {
        "protocol_hash": protocol["protocol_hash"],
        "implementation_commit": protocol["implementation_commit"],
        "cohort_fingerprint": protocol["cohort_contract"]["fingerprint"],
        "expected_runs": 20,
        "completed_runs": len(complete),
        "failed_runs": failed,
        "pending_runs": pending,
        "test_trajectory_accessed": False,
        "test_outcomes_evaluated": False,
        "final_state_selected": False,
    }
    atomic_write_json(analysis_dir / "full_analysis_manifest.json", manifest)
    report = f"""# Primary-State PPO Full Validation Analysis

This report uses validation trajectories only. Test trajectories, outcomes, rollouts,
and metrics remain sealed. Ranking is unavailable until all 20 runs are complete.

## Completion

- Complete: {len(complete)} / 20
- Failed: {len(failed)}
- Pending: {len(pending)}
- Protocol hash: `{protocol['protocol_hash']}`

## Profile Mean and Standard Deviation

{_markdown_table(profile_summary)}

## Seed Wins

{_markdown_table(seed_wins)}

## Hierarchical Bootstrap

{_markdown_table(intervals)}

Intervals resample training seeds first and validation patients within each sampled
seed. No p-values are reported. Final state selection requires all five seeds and
uses BIS target MAE, time in BIS 40-60, and action smoothness without hiding trade-offs.
"""
    atomic_write_text(analysis_dir / "full_validation_report.md", report)
    return manifest
