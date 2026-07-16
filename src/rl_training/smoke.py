"""Short synthetic PPO contract smoke tests; never research comparisons."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from gymnasium.utils.env_checker import check_env

from src.rl_env import (
    EnvironmentConfig,
    PropofolControlEnv,
    SYNTHETIC_NONCLINICAL_ACTION_BOUNDS,
)

from .action_wrapper import NormalizedPropofolActionWrapper
from .callbacks import PPOProgressCallback
from .config import (
    POLICY_CONDITIONS,
    PolicyCondition,
    PrimaryStateProfile,
    environment_profile_for_condition,
    primary_smoke_ppo_config,
    smoke_ppo_config,
)
from .environment_factory import make_primary_state_environment
from .feature_extractors import FactorizedAttentionControlExtractor
from .io import atomic_write_dataframe, atomic_write_json
from .policy_registry import primary_state_policy_contract
from .run_status import (
    begin_run_status,
    complete_run_status,
    fail_run_status,
    update_running_config,
)
from .training import create_ppo, create_primary_state_ppo, parameter_counts


def make_synthetic_smoke_env(condition: PolicyCondition, total_timesteps: int = 2048):
    ppo = smoke_ppo_config(total_timesteps)
    config = EnvironmentConfig(
        episode_duration_seconds=ppo.episode_duration_seconds,
        action_bounds=SYNTHETIC_NONCLINICAL_ACTION_BOUNDS,
        action_mode="strict",
        state_profile=environment_profile_for_condition(condition),  # type: ignore[arg-type]
        reward_profile=ppo.reward_profile,
    )
    return NormalizedPropofolActionWrapper(
        PropofolControlEnv(config), SYNTHETIC_NONCLINICAL_ACTION_BOUNDS
    )


def _parameter_vector(model: PPO) -> torch.Tensor:
    return torch.cat(
        [parameter.detach().cpu().reshape(-1) for parameter in model.policy.parameters()]
    )


def run_condition_smoke(
    *,
    condition: PolicyCondition,
    seed: int,
    total_timesteps: int,
    output_dir: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train briefly, save/load/resume, and evaluate one synthetic episode."""

    output_dir.mkdir(parents=True, exist_ok=True)
    config = smoke_ppo_config(total_timesteps)
    env = make_synthetic_smoke_env(condition, total_timesteps)
    observation, _ = env.reset(seed=seed)
    model = create_ppo(
        env, condition=condition, config=config, seed=seed, device=device
    )
    initial_parameters = _parameter_vector(model).clone()
    initial_action, _ = model.predict(observation, deterministic=True)
    callback = PPOProgressCallback(bounds=SYNTHETIC_NONCLINICAL_ACTION_BOUNDS)
    started = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    elapsed = time.perf_counter() - started
    final_parameters = _parameter_vector(model)
    parameter_delta = float(torch.linalg.vector_norm(final_parameters - initial_parameters))
    loss = float(model.logger.name_to_value.get("train/loss", np.nan))
    if not np.isfinite(loss):
        raise FloatingPointError(f"PPO smoke loss is non-finite for {condition}: {loss}")
    if parameter_delta <= 0.0:
        raise AssertionError(f"PPO policy parameters did not update for {condition}.")
    checkpoint = output_dir / "last_model"
    model.save(checkpoint)
    checkpoint_zip = checkpoint.with_suffix(".zip")
    restored_env = make_synthetic_smoke_env(condition, total_timesteps)
    restored = PPO.load(checkpoint_zip, env=restored_env, device=device)
    restored_action, _ = restored.predict(observation, deterministic=True)
    if not np.allclose(model.predict(observation, deterministic=True)[0], restored_action):
        raise AssertionError("Saved/restored deterministic policy action changed.")

    eval_observation, _ = restored_env.reset(seed=seed + 1)
    done = False
    total_return = 0.0
    action_rows: list[dict[str, float]] = []
    feature_rows: list[np.ndarray] = []
    temporal_rows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    while not done:
        action, _ = restored.predict(eval_observation, deterministic=True)
        eval_observation, reward, terminated, truncated, info = restored_env.step(action)
        if not np.isfinite(action).all() or not np.isfinite(reward):
            raise FloatingPointError("Smoke evaluation produced non-finite action/reward.")
        total_return += float(reward)
        action_rows.append(
            {
                "policy_action": float(np.asarray(action).reshape(-1)[0]),
                "physical_action_mg_per_min": info["physical_action_mg_per_min"],
                "applied_dose_mg_per_10s": info["applied_dose_mg_per_10s"],
                "bis": info["bis"],
                "reward": reward,
            }
        )
        if condition == "attention_supported":
            extractor = restored.policy.features_extractor
            assert isinstance(extractor, FactorizedAttentionControlExtractor)
            assert extractor.last_attention is not None
            feature_rows.append(extractor.last_attention.feature_attention.cpu().numpy()[0])
            temporal_rows.append(extractor.last_attention.temporal_attention.cpu().numpy()[0])
            masks.append(np.asarray(info["history_mask"], dtype=bool))
        done = terminated or truncated
    atomic_write_dataframe(output_dir / "deterministic_evaluation.csv", pd.DataFrame(action_rows))
    if condition == "attention_supported":
        extractor = restored.policy.features_extractor
        assert isinstance(extractor, FactorizedAttentionControlExtractor)
        np.savez_compressed(
            output_dir / "smoke_attention.npz",
            feature_attention=np.stack(feature_rows),
            temporal_attention=np.stack(temporal_rows),
            history_mask=np.stack(masks),
            feature_names=np.asarray(extractor.feature_names),
            lag_seconds=np.asarray([-50, -40, -30, -20, -10, 0]),
            checkpoint_path=np.asarray(checkpoint_zip.name),
            predictive_checkpoint_transfer=np.asarray(False),
        )
    before_resume = restored.num_timesteps
    restored.learn(total_timesteps=config.n_steps, reset_num_timesteps=False, progress_bar=False)
    resume_advanced = restored.num_timesteps - before_resume
    restored.save(output_dir / "resumed_model")
    restored_env.close()
    env.close()
    summary = {
        "condition": condition,
        "seed": seed,
        "smoke_only_not_performance_result": True,
        "total_timesteps": total_timesteps,
        "elapsed_seconds": elapsed,
        "steps_per_second": total_timesteps / elapsed,
        "finite_loss": True,
        "final_train_loss": loss,
        "policy_parameter_delta_l2": parameter_delta,
        "parameter_update_detected": True,
        "checkpoint_save_load_action_equal": True,
        "resume_advanced_timesteps": resume_advanced,
        "deterministic_evaluation_return": total_return,
        "initial_deterministic_action": float(np.asarray(initial_action).reshape(-1)[0]),
        "parameter_counts": parameter_counts(model),
        "action_diagnostics": callback.diagnostics(),
        "full_training_performed": False,
        "test_cohort_accessed": False,
    }
    atomic_write_json(output_dir / "smoke_summary.json", summary)
    return summary


def run_all_smokes(
    output_root: Path,
    *,
    total_timesteps: int = 2048,
    seed: int = 42,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    summaries = [
        run_condition_smoke(
            condition=condition,
            seed=seed,
            total_timesteps=total_timesteps,
            output_dir=output_root / condition / f"seed_{seed}",
            device=device,
        )
        for condition in POLICY_CONDITIONS
    ]
    summary_frame = pd.DataFrame(
        [
            {
                "condition": row["condition"],
                "timesteps": row["total_timesteps"],
                "elapsed_seconds": row["elapsed_seconds"],
                "steps_per_second": row["steps_per_second"],
                "parameters": row["parameter_counts"]["total_policy_trainable_parameters"],
            }
            for row in summaries
        ]
    )
    atomic_write_dataframe(output_root / "smoke_summary.csv", summary_frame)
    return summaries


def run_primary_state_smoke(
    *,
    state_profile: PrimaryStateProfile,
    seed: int,
    total_timesteps: int,
    output_dir: Path,
    repo_dir: Path,
    device: str = "cpu",
    selected_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Run the exact-step common-MLP integration smoke; never a scientific result."""

    config = primary_smoke_ppo_config(total_timesteps)
    contract = primary_state_policy_contract(
        state_profile,
        selected_manifest_path=selected_manifest_path,
        hidden_dim=config.policy_hidden_dim,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        "workflow": "primary_state_smoke",
        "smoke_only_not_scientific_result": True,
        "seed": seed,
        "device": device,
        "state_profile": state_profile,
        "selected_manifest_path": (
            str(selected_manifest_path.resolve()) if selected_manifest_path else None
        ),
        "ordered_feature_names": list(contract.ordered_feature_names),
        "observation_dimension": contract.observation_dimension,
        "policy_architecture": contract.policy_class,
        "feature_extractor": contract.feature_extractor,
        "hidden_layers": list(contract.hidden_layers),
        "ppo": config.as_dict(),
    }
    begin_run_status(output_dir, resolved_config=resolved_config, repo_dir=repo_dir)
    checkpoint = output_dir / "final_model.zip"
    try:
        checker_env = make_primary_state_environment(
            state_profile=state_profile,
            ppo=config,
            seed=seed,
            selected_manifest_path=selected_manifest_path,
        )
        check_env(checker_env.unwrapped, skip_render_check=True)
        checker_env.close()

        random_env = make_primary_state_environment(
            state_profile=state_profile,
            ppo=config,
            seed=seed,
            selected_manifest_path=selected_manifest_path,
        )
        observation, _ = random_env.reset(seed=seed)
        random_rows = []
        done = False
        while not done:
            action = random_env.action_space.sample()
            observation, reward, terminated, truncated, info = random_env.step(action)
            random_rows.append(
                {
                    "step": len(random_rows) + 1,
                    "normalized_action": float(np.asarray(action).item()),
                    "physical_action_mg_per_min": info["physical_action_mg_per_min"],
                    "bis": info["bis"],
                    "reward": reward,
                }
            )
            done = terminated or truncated
        random_env.close()
        random_path = output_dir / "random_action_rollout.csv"
        atomic_write_dataframe(random_path, pd.DataFrame(random_rows))

        env = make_primary_state_environment(
            state_profile=state_profile,
            ppo=config,
            seed=seed,
            selected_manifest_path=selected_manifest_path,
        )
        model = create_primary_state_ppo(
            env,
            state_profile=state_profile,
            config=config,
            seed=seed,
            device=device,
        )
        counts = parameter_counts(model)
        update_running_config(
            output_dir,
            updates={
                "total_trainable_parameters": counts[
                    "total_policy_trainable_parameters"
                ]
            },
        )
        callback = PPOProgressCallback(bounds=SYNTHETIC_NONCLINICAL_ACTION_BOUNDS)
        started = time.perf_counter()
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,
        )
        elapsed = time.perf_counter() - started
        if model.num_timesteps != total_timesteps:
            raise AssertionError(
                f"Primary smoke requested {total_timesteps} exact steps, got {model.num_timesteps}."
            )
        model.save(checkpoint.with_suffix(""))
        env.close()

        restored_env = make_primary_state_environment(
            state_profile=state_profile,
            ppo=config,
            seed=seed + 1,
            selected_manifest_path=selected_manifest_path,
        )
        restored = PPO.load(checkpoint, env=restored_env, device=device)
        evaluation_observation, _ = restored_env.reset(seed=seed + 1)
        evaluation_rows = []
        done = False
        total_return = 0.0
        while not done:
            action, _ = restored.predict(evaluation_observation, deterministic=True)
            evaluation_observation, reward, terminated, truncated, info = restored_env.step(
                action
            )
            total_return += float(reward)
            evaluation_rows.append(
                {
                    "step": len(evaluation_rows) + 1,
                    "raw_policy_action_available": False,
                    "normalized_action": float(np.asarray(action).item()),
                    "scaled_environment_action_mg_per_min": info[
                        "scaled_environment_action_mg_per_min"
                    ],
                    "action_before_wrapper_clipping": info[
                        "action_before_wrapper_clipping"
                    ],
                    "action_after_wrapper_clipping": info["action_after_wrapper_clipping"],
                    "clipped": info["normalized_clipping_applied"],
                    "lower_bound_clipping": info["lower_bound_clipping"],
                    "upper_bound_clipping": info["upper_bound_clipping"],
                    "bis": info["bis"],
                    "reward": reward,
                }
            )
            done = terminated or truncated
        restored_env.close()
        evaluation_path = output_dir / "deterministic_evaluation.csv"
        atomic_write_dataframe(evaluation_path, pd.DataFrame(evaluation_rows))
        diagnostics_path = output_dir / "action_clipping_diagnostics.json"
        atomic_write_json(
            diagnostics_path,
            {
                "aggregate": callback.diagnostics(),
                "by_episode": callback.episode_rows,
                "by_training_window": callback.rollout_rows,
                "action_bound_changed_after_audit": False,
                "duplicate_scaling_detected": False,
                "interpretation": (
                    "SB3 samples an unbounded Gaussian policy action and clips it once to "
                    "the normalized Box before one affine wrapper conversion."
                ),
            },
        )
        summary_path = output_dir / "evaluation_summary.json"
        summary = {
            "status": "complete",
            "smoke_only_not_scientific_result": True,
            "gymnasium_check_env_passed": True,
            "random_action_rollout_steps": len(random_rows),
            "training_timesteps": model.num_timesteps,
            "model_reload_passed": True,
            "deterministic_evaluation_steps": len(evaluation_rows),
            "deterministic_evaluation_return": total_return,
            "elapsed_seconds": elapsed,
            "parameter_counts": counts,
            "action_clipping": callback.diagnostics(),
            "test_cohort_accessed": False,
        }
        atomic_write_json(summary_path, summary)
        complete_run_status(
            output_dir,
            final_checkpoint=checkpoint,
            evaluation_artifacts=[
                random_path,
                evaluation_path,
                diagnostics_path,
                summary_path,
            ],
            extra={"total_trainable_parameters": counts["total_policy_trainable_parameters"]},
        )
        return summary
    except BaseException as exc:
        fail_run_status(output_dir, exc, last_checkpoint=checkpoint)
        raise
