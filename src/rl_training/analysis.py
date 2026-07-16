"""Validation-only paired PPO analysis and explicit-attention stability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .attention_logging import verify_attention_checkpoint
from .config import EXPERIMENT_SEEDS, POLICY_CONDITIONS
from .io import atomic_write_dataframe, atomic_write_json


PRIMARY_METRICS = (
    "bis_target_mae",
    "bis_target_rmse",
    "fraction_time_in_bis_40_60",
    "bis_below_40_duration_seconds",
    "bis_above_60_duration_seconds",
    "total_propofol_dose_mg",
    "mean_propofol_rate_mg_per_min",
    "max_propofol_rate_mg_per_min",
    "absolute_action_change_sum",
    "squared_action_change_sum",
    "excessive_action_change_count",
    "return",
    "mean_inference_seconds_per_action",
)


def audit_run_inventory(output_root: Path) -> dict[str, Any]:
    expected = {
        (condition, seed) for condition in POLICY_CONDITIONS for seed in EXPERIMENT_SEEDS
    }
    completed: set[tuple[str, int]] = set()
    duplicates: list[str] = []
    for path in output_root.glob("*/seed_*/completion.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity = (str(payload["condition"]), int(payload["seed"]))
        if identity in completed:
            duplicates.append(f"{identity[0]}/seed_{identity[1]}")
        completed.add(identity)
    return {
        "expected_count": len(expected),
        "completed_count": len(completed),
        "missing": [f"{condition}/seed_{seed}" for condition, seed in sorted(expected - completed)],
        "unexpected": [f"{condition}/seed_{seed}" for condition, seed in sorted(completed - expected)],
        "duplicates": duplicates,
        "complete": completed == expected and not duplicates,
    }


def load_selected_validation(output_root: Path, *, require_complete: bool = True) -> pd.DataFrame:
    inventory = audit_run_inventory(output_root)
    if require_complete and not inventory["complete"]:
        raise ValueError(f"PPO validation inventory is incomplete: {inventory}")
    frames = []
    for completion_path in output_root.glob("*/seed_*/completion.json"):
        run_dir = completion_path.parent
        best = json.loads((run_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(run_dir / best["validation_file"])
        if set(frame["cohort_split"]) != {"validation"}:
            raise ValueError(f"Non-validation rows found in {run_dir}.")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        frame["condition"] = completion["condition"]
        frame["training_seed"] = int(completion["seed"])
        frame["training_timesteps"] = int(completion["timesteps"])
        counts = json.loads((run_dir / "parameter_counts.json").read_text(encoding="utf-8"))
        frame["total_policy_trainable_parameters"] = counts[
            "total_policy_trainable_parameters"
        ]
        progress_path = run_dir / "training_progress.csv"
        progress = pd.read_csv(progress_path)
        frame["final_train_loss"] = float(progress.iloc[-1]["train_loss"])
        frame["training_steps_per_second"] = float(
            progress["training_steps_per_second"].mean()
        )
        frames.append(frame)
    if not frames:
        raise ValueError("No completed PPO validation runs were found.")
    return pd.concat(frames, ignore_index=True)


def paired_contrast(
    frame: pd.DataFrame,
    *,
    left: str,
    right: str,
    metric: str,
) -> pd.DataFrame:
    keys = ["training_seed", "scenario_id", "patient_id"]
    left_frame = frame.loc[frame["condition"] == left, keys + [metric]].rename(
        columns={metric: "left_value"}
    )
    right_frame = frame.loc[frame["condition"] == right, keys + [metric]].rename(
        columns={metric: "right_value"}
    )
    paired = left_frame.merge(right_frame, on=keys, validate="one_to_one")
    paired["contrast"] = f"{left} - {right}"
    paired["metric"] = metric
    paired["difference"] = paired["left_value"] - paired["right_value"]
    return paired


def hierarchical_bootstrap(
    paired: pd.DataFrame,
    *,
    replicates: int = 10_000,
    seed: int = 20260716,
) -> dict[str, Any]:
    """Resample training seeds, then scenarios within each sampled seed."""

    if replicates <= 0:
        raise ValueError("Bootstrap replicates must be positive.")
    rng = np.random.default_rng(seed)
    seeds = np.asarray(sorted(paired["training_seed"].unique()))
    if len(seeds) < 2:
        raise ValueError("Hierarchical bootstrap requires at least two training seeds.")
    by_seed = {
        value: paired.loc[paired["training_seed"] == value, "difference"].to_numpy(float)
        for value in seeds
    }
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        cluster_means = []
        for sampled_seed in sampled_seeds:
            values = by_seed[int(sampled_seed)]
            cluster_means.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        draws[index] = float(np.mean(cluster_means))
    return {
        "observed_mean_difference": float(paired["difference"].mean()),
        "bootstrap_mean_difference": float(draws.mean()),
        "confidence_interval_95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "replicates": replicates,
        "bootstrap_seed": seed,
        "p_value_used_as_winner_rule": False,
    }


def analyze_attention_artifacts(paths: list[Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rankings: dict[str, np.ndarray] = {}
    rows = []
    top_sets: dict[str, set[str]] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            feature = archive["feature_attention"]
            temporal = archive["temporal_attention"]
            mask = archive["history_mask"].astype(bool)
            names = archive["feature_names"].astype(str)
            valid_weight = temporal[:, :, None] * feature
            marginal = valid_weight.sum(axis=1).mean(axis=0)
            marginal = marginal / marginal.sum()
            entropy = -np.sum(marginal[marginal > 0] * np.log(marginal[marginal > 0]))
            key = path.parent.parent.name
            rankings[key] = marginal
            top_sets[key] = set(names[np.argsort(-marginal)[:5]])
            for name, value in zip(names, marginal):
                rows.append({"artifact": key, "feature": name, "mean_attention": value})
            rows.append({"artifact": key, "feature": "__normalized_entropy__", "mean_attention": entropy / np.log(len(names))})
            if np.count_nonzero(feature[~mask]) or np.count_nonzero(temporal[~mask]):
                raise ValueError(f"Padded attention is nonzero in {path}.")
    correlations = []
    jaccards = []
    keys = sorted(rankings)
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1 :]:
            correlations.append(float(spearmanr(rankings[left], rankings[right]).statistic))
            union = top_sets[left] | top_sets[right]
            jaccards.append(len(top_sets[left] & top_sets[right]) / len(union))
    summary = {
        "artifact_count": len(paths),
        "mean_pairwise_spearman": float(np.mean(correlations)) if correlations else None,
        "mean_top5_jaccard": float(np.mean(jaccards)) if jaccards else None,
        "attention_is_causal_effect": False,
    }
    return pd.DataFrame(rows), summary


def attention_detail_tables(
    paths: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return lag, feature-by-lag, and BIS-region attention summaries."""

    temporal_rows = []
    heatmap_rows = []
    region_rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            feature = archive["feature_attention"]
            temporal = archive["temporal_attention"]
            combined = temporal[:, :, None] * feature
            names = archive["feature_names"].astype(str)
            lags = archive["lag_seconds"].astype(int)
            bis = archive["bis"].astype(float)
            artifact = path.parent.parent.name
            for lag_index, lag in enumerate(lags):
                temporal_rows.append(
                    {
                        "artifact": artifact,
                        "lag_seconds": lag,
                        "mean_temporal_attention": float(temporal[:, lag_index].mean()),
                    }
                )
                for feature_index, name in enumerate(names):
                    heatmap_rows.append(
                        {
                            "artifact": artifact,
                            "lag_seconds": lag,
                            "feature": name,
                            "mean_combined_attention": float(
                                combined[:, lag_index, feature_index].mean()
                            ),
                        }
                    )
            regions = {
                "bis_below_40": bis < 40.0,
                "bis_40_60": (bis >= 40.0) & (bis <= 60.0),
                "bis_above_60": bis > 60.0,
            }
            for region, selector in regions.items():
                if not selector.any():
                    continue
                marginal = combined[selector].sum(axis=1).mean(axis=0)
                marginal = marginal / marginal.sum()
                for name, value in zip(names, marginal):
                    region_rows.append(
                        {
                            "artifact": artifact,
                            "bis_region": region,
                            "feature": name,
                            "mean_attention": float(value),
                        }
                    )
    return pd.DataFrame(temporal_rows), pd.DataFrame(heatmap_rows), pd.DataFrame(region_rows)


def run_validation_analysis(
    output_root: Path,
    analysis_dir: Path,
    *,
    replicates: int = 10_000,
    bootstrap_seed: int = 20260716,
) -> dict[str, Any]:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    inventory = audit_run_inventory(output_root)
    frame = load_selected_validation(output_root, require_complete=True)
    atomic_write_dataframe(analysis_dir / "selected_validation_scenarios.csv", frame)
    contrasts = [
        ("attention_supported", "all_supported", "primary"),
        ("attention_supported", "yun_reconstructed", "secondary"),
        ("selected_control_aware", "all_supported", "secondary"),
    ]
    summaries = []
    paired_frames = []
    for left, right, role in contrasts:
        for metric in PRIMARY_METRICS:
            paired = paired_contrast(frame, left=left, right=right, metric=metric)
            paired_frames.append(paired)
            summary = hierarchical_bootstrap(
                paired, replicates=replicates, seed=bootstrap_seed
            )
            summaries.append({"contrast": f"{left} - {right}", "role": role, "metric": metric, **summary})
    atomic_write_dataframe(
        analysis_dir / "paired_scenario_contrasts.csv",
        pd.concat(paired_frames, ignore_index=True),
    )
    atomic_write_dataframe(
        analysis_dir / "hierarchical_bootstrap_summary.csv", pd.DataFrame(summaries)
    )
    attention_paths = []
    for best_path in output_root.glob("attention_supported/seed_*/best_checkpoint.json"):
        best = json.loads(best_path.read_text(encoding="utf-8"))
        artifact = best_path.parent / "attention_snapshots" / f"validation_{int(best['timesteps'])}.npz"
        if not artifact.exists():
            raise ValueError(f"Selected attention artifact is missing: {artifact}")
        verify_attention_checkpoint(artifact, best_path.parent / "best_model.zip")
        attention_paths.append(artifact)
    attention, attention_summary = analyze_attention_artifacts(attention_paths)
    atomic_write_dataframe(analysis_dir / "attention_feature_summary.csv", attention)
    temporal, heatmap, regions = attention_detail_tables(attention_paths)
    atomic_write_dataframe(analysis_dir / "attention_time_lag_summary.csv", temporal)
    atomic_write_dataframe(analysis_dir / "attention_feature_time_heatmap.csv", heatmap)
    atomic_write_dataframe(analysis_dir / "attention_bis_region_summary.csv", regions)
    atomic_write_json(
        analysis_dir / "attention_stability.json", attention_summary
    )
    result = {
        "inventory": inventory,
        "scenario_rows": len(frame),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": bootstrap_seed,
        "primary_contrast": "attention_supported - all_supported",
        "test_cohort_accessed": False,
        "protocol_changed": False,
        "winner_selected_from_p_value": False,
        "attention_is_causal_effect": False,
    }
    atomic_write_json(
        analysis_dir / "validation_analysis_summary.json", result
    )
    return result
