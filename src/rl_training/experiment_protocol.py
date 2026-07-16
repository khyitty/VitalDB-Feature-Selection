"""Resumable validation-selected full PPO experiment execution."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from .callbacks import PPOProgressCallback
from .cohort import CohortBundle, scenarios_for_split
from .config import PPOConfig, PolicyCondition
from .environment_factory import make_cohort_environment
from .evaluation import checkpoint_score, evaluate_scenarios
from .io import atomic_write_dataframe, atomic_write_json
from .manifests import canonical_json, verify_protocol
from .policy_registry import policy_contract
from .run_status import (
    begin_run_status,
    complete_run_status,
    fail_run_status,
    update_running_config,
)
from .training import create_ppo, parameter_counts
from src.rl_env.state_adapters import get_state_profile


def _write_json(path: Path, payload: Any) -> None:
    atomic_write_json(path, payload)


def _run_experiment_impl(
    *,
    protocol: dict[str, Any],
    condition: PolicyCondition,
    seed: int,
    cohort: CohortBundle,
    output_root: Path,
    device: str,
) -> dict[str, Any]:
    """Run/resume one inventory item; never reads the test cohort."""

    verify_protocol(protocol)
    identities = {(item["condition"], int(item["seed"])) for item in protocol["inventory"]}
    if (condition, seed) not in identities:
        raise ValueError(f"Run {(condition, seed)} is absent from the frozen inventory.")
    if protocol["cohort"]["fingerprint"] != cohort.fingerprint:
        raise ValueError("Cohort fingerprint differs from the frozen PPO protocol.")
    ppo = PPOConfig(**protocol["ppo"])
    run_dir = output_root / condition / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    completion_path = run_dir / "completion.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("protocol_hash") != protocol["protocol_hash"]:
            raise ValueError("Completed run protocol hash does not match the frozen protocol.")
        return {**completion, "skipped_complete": True}
    config_path = run_dir / "config.json"
    config_payload = {
        "condition": condition,
        "seed": seed,
        "protocol_hash": protocol["protocol_hash"],
        "device": device,
        "test_cohort_accessed": False,
    }
    if config_path.exists():
        observed = json.loads(config_path.read_text(encoding="utf-8"))
        if canonical_json(observed) != canonical_json(config_payload):
            raise ValueError("Partial run config is incompatible; refusing unsafe resume.")
    else:
        _write_json(config_path, config_payload)
        _write_json(run_dir / "protocol_snapshot.json", protocol)
        _write_json(
            run_dir / "normalization_state.json",
            {
                "kind": "fixed_unit_aware_scaling",
                "learned_statistics": False,
                "test_data_used": False,
            },
        )

    env = make_cohort_environment(
        condition=condition,
        ppo=ppo,
        cohort=cohort,
        split="train",
        seed=seed,
    )
    last_path = run_dir / "last_model.zip"
    if last_path.exists():
        model = PPO.load(last_path, env=env, device=device)
        resumed = True
    else:
        model = create_ppo(
            env, condition=condition, config=ppo, seed=seed, device=device, verbose=1
        )
        resumed = False
    counts = parameter_counts(model)
    _write_json(run_dir / "parameter_counts.json", counts)
    if (run_dir / "run_status.json").is_file():
        update_running_config(
            run_dir,
            updates={
                "total_trainable_parameters": counts[
                    "total_policy_trainable_parameters"
                ]
            },
        )
    validation_scenarios = scenarios_for_split(cohort, "validation", base_seed=100_000)
    progress_path = run_dir / "training_progress.csv"
    evaluation_progress_path = run_dir / "evaluation_progress.csv"
    progress = pd.read_csv(progress_path) if progress_path.exists() else pd.DataFrame()
    evaluation_progress = (
        pd.read_csv(evaluation_progress_path)
        if evaluation_progress_path.exists()
        else pd.DataFrame()
    )
    best_score = (np.inf, np.inf, np.inf)
    if not evaluation_progress.empty:
        best_row = evaluation_progress.sort_values(
            ["validation_bis_mae", "negative_time_in_range", "action_change_sum"]
        ).iloc[0]
        best_score = (
            float(best_row["validation_bis_mae"]),
            float(best_row["negative_time_in_range"]),
            float(best_row["action_change_sum"]),
        )
    evaluated_timesteps = (
        set(evaluation_progress["timesteps"].astype(int))
        if not evaluation_progress.empty
        else set()
    )
    while (
        model.num_timesteps < ppo.total_timesteps
        or model.num_timesteps not in evaluated_timesteps
    ):
        if model.num_timesteps < ppo.total_timesteps:
            remaining = ppo.total_timesteps - model.num_timesteps
            chunk = min(ppo.evaluation_frequency_timesteps, remaining)
            callback = PPOProgressCallback(bounds=env.bounds)
            chunk_started = time.perf_counter()
            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=(model.num_timesteps == 0),
                callback=callback,
                progress_bar=False,
            )
            chunk_elapsed = time.perf_counter() - chunk_started
            model.save(run_dir / "last_model")
            logger = model.logger.name_to_value
            row = {
                "timesteps": model.num_timesteps,
                "train_loss": float(logger.get("train/loss", np.nan)),
                "policy_gradient_loss": float(logger.get("train/policy_gradient_loss", np.nan)),
                "value_loss": float(logger.get("train/value_loss", np.nan)),
                "chunk_elapsed_seconds": chunk_elapsed,
                "training_steps_per_second": chunk / chunk_elapsed,
                **callback.diagnostics(),
            }
            progress = pd.concat((progress, pd.DataFrame([row])), ignore_index=True)
            atomic_write_dataframe(progress_path, progress)
            atomic_write_json(
                run_dir / "action_clipping_diagnostics.json",
                {
                    "latest_training_chunk": callback.diagnostics(),
                    "episode_clipping": callback.episode_rows,
                    "training_windows": callback.rollout_rows,
                    "action_bound_changed_after_audit": False,
                },
            )
        if model.num_timesteps in evaluated_timesteps:
            continue
        evaluation_path = run_dir / f"validation_{model.num_timesteps}.csv"
        attention_path = (
            run_dir / "attention_snapshots" / f"validation_{model.num_timesteps}.npz"
            if condition == "attention_supported"
            else None
        )
        validation = evaluate_scenarios(
            model,
            condition=condition,
            config=ppo,
            cohort=cohort,
            scenarios=validation_scenarios,
            training_seed=seed,
            checkpoint_path=last_path,
            attention_output_path=attention_path,
        )
        atomic_write_dataframe(evaluation_path, validation)
        score = checkpoint_score(validation)
        evaluation_row = {
            "timesteps": model.num_timesteps,
            "validation_bis_mae": score[0],
            "negative_time_in_range": score[1],
            "action_change_sum": score[2],
            "validation_file": evaluation_path.name,
            "selected_as_best": score < best_score,
        }
        if score < best_score:
            best_score = score
            shutil.copy2(last_path, run_dir / "best_model.zip")
            _write_json(
                run_dir / "best_checkpoint.json",
                {**evaluation_row, "selection_rule": protocol["checkpoint_selection"]},
            )
        evaluation_progress = pd.concat(
            (evaluation_progress, pd.DataFrame([evaluation_row])), ignore_index=True
        )
        atomic_write_dataframe(evaluation_progress_path, evaluation_progress)
        evaluated_timesteps.add(model.num_timesteps)

    env.close()
    completion = {
        "status": "complete",
        "condition": condition,
        "seed": seed,
        "timesteps": model.num_timesteps,
        "protocol_hash": protocol["protocol_hash"],
        "resumed_from_partial": resumed,
        "best_validation_score": list(best_score),
        "test_cohort_accessed": False,
        "replay_buffer_state": "not applicable: PPO is on-policy",
    }
    _write_json(run_dir / "run_manifest.json", {**config_payload, **counts})
    _write_json(completion_path, completion)
    return completion


def run_experiment(
    *,
    protocol: dict[str, Any],
    condition: PolicyCondition,
    seed: int,
    cohort: CohortBundle,
    output_root: Path,
    device: str,
) -> dict[str, Any]:
    """Run one frozen-v1 item with failure-safe status and legacy completion support."""

    run_dir = output_root / condition / f"seed_{seed}"
    completion_path = run_dir / "completion.json"
    if completion_path.is_file() and not (run_dir / "run_status.json").is_file():
        # Historical completed artifacts remain readable without retroactive mutation.
        return _run_experiment_impl(
            protocol=protocol,
            condition=condition,
            seed=seed,
            cohort=cohort,
            output_root=output_root,
            device=device,
        )

    ppo = PPOConfig(**protocol["ppo"])
    contract = policy_contract(condition, ppo.latent_dim)
    profile = get_state_profile(contract.environment_profile)
    resolved_config = {
        "workflow": "legacy_frozen_ppo_v1",
        "condition": condition,
        "seed": seed,
        "device": device,
        "state_profile": profile.name,
        "ordered_feature_names": list(profile.ordered_feature_names),
        "observation_dimension": profile.observation_dimension(),
        "policy_architecture": "SB3 MultiInputPolicy",
        "feature_extractor": contract.extractor_kind,
        "ppo": ppo.as_dict(),
        "protocol_hash": protocol.get("protocol_hash"),
    }
    begin_run_status(run_dir, resolved_config=resolved_config, repo_dir=Path(__file__).parents[2])
    try:
        result = _run_experiment_impl(
            protocol=protocol,
            condition=condition,
            seed=seed,
            cohort=cohort,
            output_root=output_root,
            device=device,
        )
        best = json.loads((run_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
        counts = json.loads((run_dir / "parameter_counts.json").read_text(encoding="utf-8"))
        complete_run_status(
            run_dir,
            final_checkpoint=run_dir / "best_model.zip",
            evaluation_artifacts=[
                run_dir / "best_checkpoint.json",
                run_dir / str(best["validation_file"]),
                run_dir / "training_progress.csv",
                run_dir / "evaluation_progress.csv",
                run_dir / "action_clipping_diagnostics.json",
            ],
            extra={"total_trainable_parameters": counts["total_policy_trainable_parameters"]},
        )
        return result
    except BaseException as exc:
        fail_run_status(run_dir, exc, last_checkpoint=run_dir / "last_model.zip")
        raise
