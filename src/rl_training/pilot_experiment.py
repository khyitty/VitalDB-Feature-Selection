"""Resumable non-smoke PPO pilot execution for primary state profiles."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import stable_baselines3
import torch

from .callbacks import PPOProgressCallback
from .cohort import (
    CohortBundle,
    CohortScenarioWrapper,
    EvaluationScenario,
    scenarios_for_split,
)
from .config import PPOConfig, PrimaryStateProfile
from .environment_factory import make_primary_state_environment
from .evaluation import checkpoint_score
from .io import atomic_write_dataframe, atomic_write_json
from .manifests import canonical_json
from .pilot_protocol import PILOT_PROFILES, resolve_execution_device, verify_pilot_protocol
from .policy_registry import primary_state_policy_contract
from .run_status import (
    begin_run_status,
    complete_run_status,
    fail_run_status,
    repository_commit,
    update_running_config,
)
from .training import create_primary_state_ppo, parameter_counts


def _atomic_model_save(model: PPO, path: Path) -> None:
    """Replace a PPO zip only after SB3 has completed serialization."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_base = path.with_name(f".{path.stem}.next")
    if temporary_base.exists():
        temporary_base.unlink()
    model.save(temporary_base)
    os.replace(temporary_base, path)


def _raw_deterministic_action(model: PPO, observation: np.ndarray) -> np.ndarray:
    """Return the policy mode before SB3 Box clipping for evaluation diagnostics."""

    tensor, _ = model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        action = model.policy._predict(tensor, deterministic=True)
    result = action.detach().cpu().numpy().reshape(-1)
    if result.size != 1 or not np.isfinite(result).all():
        raise FloatingPointError("Deterministic PPO action is not one finite scalar.")
    return result.astype(np.float32)


def _advance_training_sampler_for_resume(env: Any, timesteps: int) -> int:
    """Locate the authorized train wrapper and advance beyond the discarded episode."""

    current = env
    while current is not None:
        if isinstance(current, CohortScenarioWrapper):
            return current.advance_random_sampling_for_resume(timesteps)
        current = getattr(current, "env", None)
    raise TypeError("Primary pilot environment lacks CohortScenarioWrapper.")


def evaluate_primary_state_scenarios(
    model: PPO,
    *,
    state_profile: PrimaryStateProfile,
    config: PPOConfig,
    cohort: CohortBundle,
    scenarios: tuple[EvaluationScenario, ...],
    training_seed: int,
) -> pd.DataFrame:
    """Evaluate paired deterministic validation scenarios and preserve the test seal."""

    if not scenarios:
        raise ValueError("At least one validation scenario is required.")
    if any(scenario.split != "validation" for scenario in scenarios):
        raise ValueError("Primary-state pilot evaluation is validation-only; test is sealed.")
    env = make_primary_state_environment(
        state_profile=state_profile,
        ppo=config,
        seed=training_seed,
        cohort=cohort,
        split="validation",
    )
    rows: list[dict[str, Any]] = []
    try:
        for scenario in scenarios:
            observation, _ = env.reset(options={"scenario": scenario})
            done = False
            total_return = 0.0
            raw_actions: list[float] = []
            bounded_actions: list[float] = []
            physical_actions: list[float] = []
            rewards: list[float] = []
            lower_clips = 0
            upper_clips = 0
            final_info: dict[str, Any] = {}
            while not done:
                raw = float(_raw_deterministic_action(model, observation)[0])
                bounded = float(np.clip(raw, -1.0, 1.0))
                action = np.asarray([bounded], dtype=np.float32)
                observation, reward, terminated, truncated, final_info = env.step(action)
                raw_actions.append(raw)
                bounded_actions.append(bounded)
                physical_actions.append(float(final_info["physical_action_mg_per_min"]))
                rewards.append(float(reward))
                lower_clips += int(raw < -1.0)
                upper_clips += int(raw > 1.0)
                total_return += float(reward)
                done = terminated or truncated
            metrics = dict(final_info["episode_metrics"])
            components = metrics.pop("reward_component_totals", {})
            action_count = len(raw_actions)
            row = {
                "state_profile": state_profile,
                "training_seed": training_seed,
                "scenario_id": scenario.scenario_id,
                "patient_id": scenario.patient_id,
                "cohort_split": scenario.split,
                "scenario_seed": scenario.seed,
                "return": total_return,
                "raw_normalized_action_minimum": min(raw_actions),
                "raw_normalized_action_maximum": max(raw_actions),
                "bounded_normalized_action_minimum": min(bounded_actions),
                "bounded_normalized_action_maximum": max(bounded_actions),
                "evaluation_action_clipping_count": lower_clips + upper_clips,
                "evaluation_action_clipping_fraction": (
                    (lower_clips + upper_clips) / action_count
                ),
                "evaluation_lower_clipping_count": lower_clips,
                "evaluation_upper_clipping_count": upper_clips,
                "physical_action_minimum_mg_per_min": min(physical_actions),
                "physical_action_maximum_mg_per_min": max(physical_actions),
                "physical_action_standard_deviation_mg_per_min": float(
                    np.std(physical_actions)
                ),
                "mean_step_reward": float(np.mean(rewards)),
                **metrics,
                **{
                    f"reward_component_{name}": float(value)
                    for name, value in components.items()
                },
            }
            numeric = [
                value
                for value in row.values()
                if isinstance(value, (int, float, np.integer, np.floating))
                and value is not None
            ]
            if not all(math.isfinite(float(value)) for value in numeric):
                raise FloatingPointError(
                    f"Non-finite validation metric for {scenario.scenario_id}."
                )
            rows.append(row)
    finally:
        env.close()
    return pd.DataFrame(rows)


def _read_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _phase(timesteps: int, total_timesteps: int) -> str:
    fraction = timesteps / total_timesteps
    if fraction <= 1.0 / 3.0:
        return "early"
    if fraction <= 2.0 / 3.0:
        return "middle"
    return "late"


def next_evaluation_boundary(current: int, config: PPOConfig) -> int:
    """Return the next frozen validation boundary after a rollout-safe resume."""

    if current < 0 or current > config.total_timesteps:
        raise ValueError("Current timestep is outside the frozen training budget.")
    if current % config.n_steps:
        raise ValueError("Resume timestep is not a complete PPO rollout boundary.")
    if current == config.total_timesteps:
        return current
    boundary = (
        (current // config.evaluation_frequency_timesteps) + 1
    ) * config.evaluation_frequency_timesteps
    return min(boundary, config.total_timesteps)


def _assert_resume_frames(
    *,
    model_timesteps: int,
    training_progress: pd.DataFrame,
    evaluation_progress: pd.DataFrame,
) -> None:
    for name, frame in (
        ("training_progress", training_progress),
        ("evaluation_progress", evaluation_progress),
    ):
        if frame.empty:
            continue
        if "timesteps" not in frame or frame["timesteps"].duplicated().any():
            raise ValueError(f"Partial {name} has missing or duplicate timestep identities.")
        if int(frame["timesteps"].max()) > model_timesteps:
            raise ValueError(
                f"Partial {name} is ahead of resume checkpoint {model_timesteps}."
            )


def _recover_pending_rollout(
    *,
    pending_path: Path,
    model_timesteps: int,
    training_progress: pd.DataFrame,
    action_progress: pd.DataFrame,
    training_path: Path,
    action_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Finish or discard the tiny journal left around an atomic rollout save."""

    if not pending_path.is_file():
        return training_progress, action_progress
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    timestep = int(pending["timesteps"])
    if timestep > model_timesteps:
        pending_path.unlink()
        return training_progress, action_progress
    training_row = dict(pending["training_row"])
    if training_progress.empty or timestep not in set(
        training_progress["timesteps"].astype(int)
    ):
        training_progress = pd.concat(
            (training_progress, pd.DataFrame([training_row])), ignore_index=True
        )
        atomic_write_dataframe(training_path, training_progress)
    action_rows = list(pending.get("action_rows", []))
    if action_rows and (
        action_progress.empty
        or timestep not in set(action_progress["timesteps"].astype(int))
    ):
        action_progress = pd.concat(
            (action_progress, pd.DataFrame(action_rows)), ignore_index=True
        )
        atomic_write_dataframe(action_path, action_progress)
    pending_path.unlink()
    return training_progress, action_progress


def _completion_is_valid(
    completion: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    run_dir: Path,
    state_profile: str,
    seed: int,
    workflow: str,
) -> bool:
    ppo = protocol.get("ppo", {})
    total = int(ppo.get("total_timesteps", -1))
    frequency = int(ppo.get("evaluation_frequency_timesteps", -1))
    evaluation_steps = (
        tuple(range(frequency, total + 1, frequency))
        if frequency > 0 and total > 0 and total % frequency == 0
        else ()
    )
    required = [
        run_dir / "best_model.zip",
        run_dir / "best_checkpoint.json",
        run_dir / "training_progress.csv",
        run_dir / "evaluation_progress.csv",
        run_dir / "action_diagnostics.csv",
    ]
    required.extend(run_dir / f"checkpoint_{step}.zip" for step in evaluation_steps)
    required.extend(run_dir / f"validation_{step}.csv" for step in evaluation_steps)
    status_path = run_dir / "run_status.json"
    status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.is_file()
        else {}
    )
    return bool(
        completion.get("status") == "complete"
        and completion.get("protocol_hash") == protocol.get("protocol_hash")
        and completion.get("cohort_fingerprint")
        == protocol.get("cohort_contract", {}).get("fingerprint")
        and completion.get("state_profile") == state_profile
        and int(completion.get("seed", -1)) == seed
        and int(completion.get("timesteps", -1))
        == int(protocol.get("ppo", {}).get("total_timesteps", -2))
        and status.get("status") == "complete"
        and status.get("protocol_hash") == protocol.get("protocol_hash")
        and status.get("resolved_config", {}).get("workflow") == workflow
        and all(path.is_file() for path in required)
    )


def run_primary_state_experiment(
    *,
    protocol: dict[str, Any],
    state_profile: PrimaryStateProfile,
    seed: int,
    cohort: CohortBundle,
    output_root: Path,
    repo_dir: Path,
    device: str,
    protocol_verifier: Callable[[Mapping[str, Any]], None],
    allowed_profiles: Sequence[str],
    workflow: str,
    experiment_label: str,
    config_extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run or safely resume one frozen primary-state identity."""

    protocol_verifier(protocol)
    if state_profile not in allowed_profiles:
        raise ValueError(
            f"State profile {state_profile!r} is outside the {experiment_label}."
        )
    identities = {
        (item["state_profile"], int(item["seed"])) for item in protocol["inventory"]
    }
    if (state_profile, seed) not in identities:
        raise ValueError(f"Run {(state_profile, seed)} is absent from the frozen inventory.")
    resolved_device = resolve_execution_device(device)
    if resolved_device != protocol["execution_device"]:
        raise ValueError(
            f"Run device {resolved_device!r} differs from frozen "
            f"{protocol['execution_device']!r}."
        )
    if stable_baselines3.__version__ != protocol["library"]["stable_baselines3_required"]:
        raise ValueError(
            f"Stable-Baselines3 version differs from the frozen {experiment_label}."
        )
    if repository_commit(repo_dir) != protocol["implementation_commit"]:
        raise ValueError(
            f"Repository HEAD differs from the frozen {experiment_label} implementation commit."
        )
    if cohort.fingerprint != protocol["cohort_contract"]["fingerprint"]:
        raise ValueError("Cohort fingerprint differs from the frozen primary-state pilot.")

    ppo = PPOConfig(**protocol["ppo"])
    contract = primary_state_policy_contract(
        state_profile, hidden_dim=ppo.policy_hidden_dim
    )
    run_dir = output_root / state_profile / f"seed_{seed}"
    completion_path = run_dir / "completion.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if not _completion_is_valid(
            completion,
            protocol=protocol,
            run_dir=run_dir,
            state_profile=state_profile,
            seed=seed,
            workflow=workflow,
        ):
            raise ValueError(
                f"Completed {experiment_label} artifacts failed compatibility validation."
            )
        return {**completion, "skipped_complete": True}

    config_payload = {
        "workflow": workflow,
        "state_profile": state_profile,
        "seed": seed,
        "device": resolved_device,
        "protocol_hash": protocol["protocol_hash"],
        "cohort_fingerprint": cohort.fingerprint,
        "git_commit_hash": protocol["implementation_commit"],
        "ordered_feature_names": list(contract.ordered_feature_names),
        "observation_dimension": contract.observation_dimension,
        "policy_architecture": contract.policy_class,
        "feature_extractor": contract.feature_extractor,
        "hidden_layers": list(contract.hidden_layers),
        "activation": contract.activation,
        "optimizer": "Adam",
        "ppo": ppo.as_dict(),
        "test_cohort_accessed": False,
        **dict(config_extras or {}),
    }
    config_path = run_dir / "config.json"
    if config_path.is_file():
        observed = json.loads(config_path.read_text(encoding="utf-8"))
        if canonical_json(observed) != canonical_json(config_payload):
            raise ValueError(
                f"Partial {experiment_label} config is incompatible; refusing unsafe resume."
            )
    else:
        atomic_write_json(config_path, config_payload)
        atomic_write_json(run_dir / "protocol_snapshot.json", protocol)
        atomic_write_json(run_dir / "cohort_access_manifest.json", cohort.access_manifest)

    begin_run_status(run_dir, resolved_config=config_payload, repo_dir=repo_dir)
    resume_checkpoint = run_dir / "resume_model.zip"
    env = None
    try:
        env = make_primary_state_environment(
            state_profile=state_profile,
            ppo=ppo,
            seed=seed,
            cohort=cohort,
            split="train",
        )
        if resume_checkpoint.is_file():
            model = PPO.load(resume_checkpoint, env=env, device=resolved_device)
            resumed = True
            resume_patient_draws_skipped = _advance_training_sampler_for_resume(
                env, model.num_timesteps
            )
        else:
            model = create_primary_state_ppo(
                env,
                state_profile=state_profile,
                config=ppo,
                seed=seed,
                device=resolved_device,
                verbose=1,
            )
            resumed = False
            resume_patient_draws_skipped = 0
        if model.num_timesteps > ppo.total_timesteps or model.num_timesteps % ppo.n_steps:
            raise ValueError(
                f"Resume timestep {model.num_timesteps} is not a valid PPO rollout boundary."
            )
        counts = parameter_counts(model)
        update_running_config(
            run_dir,
            updates={
                "total_trainable_parameters": counts[
                    "total_policy_trainable_parameters"
                ],
                "resumed_from_partial": resumed,
                "resume_timestep": model.num_timesteps,
                "resume_patient_draws_skipped": resume_patient_draws_skipped,
                "partial_environment_episode_restored": False,
            },
        )
        atomic_write_json(run_dir / "parameter_counts.json", counts)

        training_path = run_dir / "training_progress.csv"
        evaluation_path = run_dir / "evaluation_progress.csv"
        action_path = run_dir / "action_diagnostics.csv"
        pending_rollout_path = run_dir / "pending_rollout.json"
        training_progress = _read_frame(training_path)
        evaluation_progress = _read_frame(evaluation_path)
        action_progress = _read_frame(action_path)
        training_progress, action_progress = _recover_pending_rollout(
            pending_path=pending_rollout_path,
            model_timesteps=model.num_timesteps,
            training_progress=training_progress,
            action_progress=action_progress,
            training_path=training_path,
            action_path=action_path,
        )
        _assert_resume_frames(
            model_timesteps=model.num_timesteps,
            training_progress=training_progress,
            evaluation_progress=evaluation_progress,
        )
        evaluated = (
            set(evaluation_progress["timesteps"].astype(int))
            if not evaluation_progress.empty
            else set()
        )
        best_score = (np.inf, np.inf, np.inf)
        if not evaluation_progress.empty:
            best = evaluation_progress.sort_values(
                ["validation_bis_target_mae", "negative_time_in_range", "action_change_sum"]
            ).iloc[0]
            best_score = (
                float(best["validation_bis_target_mae"]),
                float(best["negative_time_in_range"]),
                float(best["action_change_sum"]),
            )
        scenarios = scenarios_for_split(cohort, "validation", base_seed=100_000)
        if [item.scenario_id for item in scenarios] != protocol["cohort_contract"][
            "validation_scenario_ids"
        ]:
            raise ValueError("Paired validation scenario identities changed.")

        while model.num_timesteps < ppo.total_timesteps or model.num_timesteps not in evaluated:
            current = model.num_timesteps
            if current > 0 and current % ppo.evaluation_frequency_timesteps == 0 and current not in evaluated:
                checkpoint = run_dir / f"checkpoint_{current}.zip"
                _atomic_model_save(model, checkpoint)
                validation = evaluate_primary_state_scenarios(
                    model,
                    state_profile=state_profile,
                    config=ppo,
                    cohort=cohort,
                    scenarios=scenarios,
                    training_seed=seed,
                )
                validation_file = run_dir / f"validation_{current}.csv"
                validation.insert(0, "timesteps", current)
                validation["protocol_hash"] = protocol["protocol_hash"]
                validation["cohort_fingerprint"] = cohort.fingerprint
                atomic_write_dataframe(validation_file, validation)
                score = checkpoint_score(
                    validation.rename(columns={"state_profile": "condition"})
                )
                evaluation_row = {
                    "state_profile": state_profile,
                    "seed": seed,
                    "timesteps": current,
                    "checkpoint_path": checkpoint.name,
                    "validation_file": validation_file.name,
                    "validation_bis_target_mae": score[0],
                    "validation_bis_target_rmse": float(
                        validation["bis_target_rmse"].mean()
                    ),
                    "validation_fraction_time_in_bis_40_60": float(
                        validation["fraction_time_in_bis_40_60"].mean()
                    ),
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
                    "negative_time_in_range": score[1],
                    "action_change_sum": score[2],
                    "mean_episode_return": float(validation["return"].mean()),
                    "mean_total_propofol_dose_mg": float(
                        validation["total_propofol_dose_mg"].mean()
                    ),
                    "mean_propofol_rate_mg_per_min": float(
                        validation["mean_propofol_rate_mg_per_min"].mean()
                    ),
                    "maximum_physical_action_mg_per_min": float(
                        validation["physical_action_maximum_mg_per_min"].max()
                    ),
                    "raw_normalized_action_minimum": float(
                        validation["raw_normalized_action_minimum"].min()
                    ),
                    "raw_normalized_action_maximum": float(
                        validation["raw_normalized_action_maximum"].max()
                    ),
                    "mean_action_clipping_fraction": float(
                        validation["evaluation_action_clipping_fraction"].mean()
                    ),
                    "mean_action_smoothness": float(
                        validation["action_smoothness_mean_absolute_change"].mean()
                    ),
                    "mean_large_rate_change_count": float(
                        validation["excessive_action_change_count"].mean()
                    ),
                    "mean_lower_action_saturation_fraction": float(
                        validation["lower_action_saturation_fraction"].mean()
                    ),
                    "mean_upper_action_saturation_fraction": float(
                        validation["upper_action_saturation_fraction"].mean()
                    ),
                    "failure_episode_count": int(
                        (validation["numerical_failures"] > 0).sum()
                    ),
                    "selected_as_best": score < best_score,
                    "protocol_hash": protocol["protocol_hash"],
                    "cohort_fingerprint": cohort.fingerprint,
                }
                if score < best_score:
                    best_score = score
                    shutil.copy2(checkpoint, run_dir / "best_model.zip")
                    atomic_write_json(
                        run_dir / "best_checkpoint.json",
                        {**evaluation_row, "selection_rule": protocol["checkpoint_selection"]},
                    )
                evaluation_progress = pd.concat(
                    (evaluation_progress, pd.DataFrame([evaluation_row])), ignore_index=True
                )
                atomic_write_dataframe(evaluation_path, evaluation_progress)
                evaluated.add(current)
                if current >= ppo.total_timesteps:
                    break
            if model.num_timesteps >= ppo.total_timesteps:
                continue

            next_boundary = next_evaluation_boundary(model.num_timesteps, ppo)
            while model.num_timesteps < next_boundary:
                previous_timestep = model.num_timesteps
                callback = PPOProgressCallback(bounds=env.bounds)
                started = time.perf_counter()
                model.learn(
                    total_timesteps=ppo.n_steps,
                    reset_num_timesteps=(previous_timestep == 0),
                    callback=callback,
                    progress_bar=False,
                )
                elapsed = time.perf_counter() - started
                expected_timestep = previous_timestep + ppo.n_steps
                if model.num_timesteps != expected_timestep:
                    raise AssertionError(
                        f"PPO rollout boundary mismatch: expected {expected_timestep}, "
                        f"observed {model.num_timesteps}."
                    )
                logger = model.logger.name_to_value
                entropy_loss = float(logger.get("train/entropy_loss", np.nan))
                rollout_return = (
                    float(callback.rollout_rows[-1]["rollout_mean_reward"])
                    if callback.rollout_rows
                    else np.nan
                )
                row = {
                    "state_profile": state_profile,
                    "seed": seed,
                    "timesteps": model.num_timesteps,
                    "episode_return_mean": rollout_return,
                    "policy_loss": float(
                        logger.get("train/policy_gradient_loss", np.nan)
                    ),
                    "value_loss": float(logger.get("train/value_loss", np.nan)),
                    "entropy_loss": entropy_loss,
                    "policy_entropy": -entropy_loss,
                    "approximate_kl": float(logger.get("train/approx_kl", np.nan)),
                    "ppo_clip_fraction": float(
                        logger.get("train/clip_fraction", np.nan)
                    ),
                    "explained_variance": float(
                        logger.get("train/explained_variance", np.nan)
                    ),
                    "learning_rate": float(
                        logger.get("train/learning_rate", ppo.learning_rate)
                    ),
                    "fps": float(logger.get("time/fps", ppo.n_steps / elapsed)),
                    "chunk_elapsed_seconds": elapsed,
                    "training_steps_per_second": ppo.n_steps / elapsed,
                    "checkpoint_path": resume_checkpoint.name,
                    "observation_dimension": contract.observation_dimension,
                    "policy_parameter_count": counts[
                        "total_policy_trainable_parameters"
                    ],
                    "protocol_hash": protocol["protocol_hash"],
                    "cohort_fingerprint": cohort.fingerprint,
                    "git_commit_hash": protocol["implementation_commit"],
                    **callback.diagnostics(),
                }
                windows = pd.DataFrame(callback.rollout_rows)
                action_rows: list[dict[str, Any]] = []
                if not windows.empty:
                    windows.insert(0, "state_profile", state_profile)
                    windows.insert(1, "seed", seed)
                    windows["training_phase"] = windows["timesteps"].map(
                        lambda value: _phase(int(value), ppo.total_timesteps)
                    )
                    windows["protocol_hash"] = protocol["protocol_hash"]
                    action_rows = windows.to_dict("records")
                atomic_write_json(
                    pending_rollout_path,
                    {
                        "timesteps": model.num_timesteps,
                        "training_row": row,
                        "action_rows": action_rows,
                    },
                )
                _atomic_model_save(model, resume_checkpoint)
                training_progress = pd.concat(
                    (training_progress, pd.DataFrame([row])), ignore_index=True
                )
                atomic_write_dataframe(training_path, training_progress)
                if action_rows:
                    action_progress = pd.concat(
                        (action_progress, pd.DataFrame(action_rows)), ignore_index=True
                    )
                    atomic_write_dataframe(action_path, action_progress)
                pending_rollout_path.unlink()

        if env is not None:
            env.close()
            env = None
        best_payload = json.loads(
            (run_dir / "best_checkpoint.json").read_text(encoding="utf-8")
        )
        completion = {
            "status": "complete",
            "state_profile": state_profile,
            "seed": seed,
            "timesteps": model.num_timesteps,
            "protocol_hash": protocol["protocol_hash"],
            "cohort_fingerprint": cohort.fingerprint,
            "git_commit_hash": protocol["implementation_commit"],
            "device": resolved_device,
            "resumed_from_partial": resumed,
            "best_validation_score": list(best_score),
            "best_checkpoint": best_payload["checkpoint_path"],
            "total_training_elapsed_seconds": float(
                training_progress["chunk_elapsed_seconds"].sum()
            ),
            "test_cohort_accessed": False,
            "replay_buffer_state": "not applicable: PPO is on-policy",
        }
        atomic_write_json(run_dir / "run_manifest.json", {**config_payload, **counts})
        atomic_write_json(completion_path, completion)
        complete_run_status(
            run_dir,
            final_checkpoint=run_dir / "best_model.zip",
            evaluation_artifacts=[
                run_dir / "best_checkpoint.json",
                training_path,
                evaluation_path,
                action_path,
                completion_path,
            ],
            extra={
                "protocol_hash": protocol["protocol_hash"],
                "cohort_fingerprint": cohort.fingerprint,
                "test_cohort_accessed": False,
            },
        )
        return completion
    except BaseException as exc:
        if env is not None:
            env.close()
        fail_run_status(run_dir, exc, last_checkpoint=resume_checkpoint)
        raise


def run_primary_state_pilot(
    *,
    protocol: dict[str, Any],
    state_profile: PrimaryStateProfile,
    seed: int,
    cohort: CohortBundle,
    output_root: Path,
    repo_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Run or safely resume one frozen pilot identity; keep test trajectories sealed."""

    return run_primary_state_experiment(
        protocol=protocol,
        state_profile=state_profile,
        seed=seed,
        cohort=cohort,
        output_root=output_root,
        repo_dir=repo_dir,
        device=device,
        protocol_verifier=verify_pilot_protocol,
        allowed_profiles=PILOT_PROFILES,
        workflow="primary_state_ppo_pilot",
        experiment_label="primary-state pilot",
        config_extras=None,
    )
