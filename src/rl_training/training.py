"""Stable-Baselines3 PPO construction and parameter accounting."""

from __future__ import annotations

from typing import Any

from stable_baselines3 import PPO

from .config import PPOConfig, PolicyCondition
from .policy_registry import sb3_policy_kwargs


def create_ppo(
    env: Any,
    *,
    condition: PolicyCondition,
    config: PPOConfig,
    seed: int,
    device: str,
    verbose: int = 0,
) -> PPO:
    """Construct PPO with identical optimizer/head settings across conditions."""

    return PPO(
        "MultiInputPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=sb3_policy_kwargs(condition, config.latent_dim),
        seed=seed,
        device=device,
        verbose=verbose,
    )


def parameter_counts(model: PPO) -> dict[str, int]:
    policy = model.policy
    return {
        "total_policy_trainable_parameters": sum(
            parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
        ),
        "extractor_trainable_parameters": sum(
            parameter.numel()
            for parameter in policy.features_extractor.parameters()
            if parameter.requires_grad
        ),
        "actor_head_trainable_parameters": sum(
            parameter.numel()
            for parameter in policy.action_net.parameters()
            if parameter.requires_grad
        ),
        "critic_head_trainable_parameters": sum(
            parameter.numel()
            for parameter in policy.value_net.parameters()
            if parameter.requires_grad
        ),
    }
