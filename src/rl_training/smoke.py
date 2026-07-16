"""Short synthetic PPO contract smoke tests; never research comparisons."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from src.rl_env import (
    EnvironmentConfig,
    PropofolControlEnv,
    SYNTHETIC_NONCLINICAL_ACTION_BOUNDS,
)

from .action_wrapper import NormalizedPropofolActionWrapper
from .callbacks import PPOProgressCallback
from .config import POLICY_CONDITIONS, PolicyCondition, environment_profile_for_condition, smoke_ppo_config
from .feature_extractors import FactorizedAttentionControlExtractor
from .training import create_ppo, parameter_counts


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
    callback = PPOProgressCallback()
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
    pd.DataFrame(action_rows).to_csv(output_dir / "deterministic_evaluation.csv", index=False)
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
    (output_dir / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
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
    pd.DataFrame(
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
    ).to_csv(output_root / "smoke_summary.csv", index=False)
    return summaries
