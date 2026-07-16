"""Policy-condition registry and fair SB3 policy construction metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn

from src.rl_env.state_adapters import get_state_profile

from .config import PolicyCondition, environment_profile_for_condition
from .feature_extractors import (
    FactorizedAttentionControlExtractor,
    GRUControlExtractor,
)


@dataclass(frozen=True)
class PolicyContract:
    condition: PolicyCondition
    environment_profile: str
    extractor_kind: str
    latent_dim: int
    feature_names: tuple[str, ...]
    main_comparison_role: str


def policy_contract(condition: PolicyCondition, latent_dim: int = 64) -> PolicyContract:
    environment_profile = environment_profile_for_condition(condition)
    profile = get_state_profile(environment_profile)  # type: ignore[arg-type]
    return PolicyContract(
        condition=condition,
        environment_profile=environment_profile,
        extractor_kind=(
            "factorized_feature_temporal_attention"
            if condition == "attention_supported"
            else "mask_aware_gru"
        ),
        latent_dim=latent_dim,
        feature_names=profile.dynamic_feature_names,
        main_comparison_role=(
            "primary_attention"
            if condition == "attention_supported"
            else "primary_nonattention"
            if condition == "all_supported"
            else "secondary"
        ),
    )


def sb3_policy_kwargs(condition: PolicyCondition, latent_dim: int = 64) -> dict[str, Any]:
    contract = policy_contract(condition, latent_dim)
    extractor_class = (
        FactorizedAttentionControlExtractor
        if condition == "attention_supported"
        else GRUControlExtractor
    )
    return {
        "features_extractor_class": extractor_class,
        "features_extractor_kwargs": {
            "feature_names": contract.feature_names,
            "latent_dim": latent_dim,
        },
        "net_arch": {"pi": [latent_dim], "vf": [latent_dim]},
        "activation_fn": nn.ReLU,
        "ortho_init": False,
    }


def encoder_contract_registry(latent_dim: int = 64) -> dict[str, Any]:
    return {
        condition: {
            **policy_contract(condition, latent_dim).__dict__,
            "feature_names": list(policy_contract(condition, latent_dim).feature_names),
            "actor_critic_heads": "identical SB3 MultiInputPolicy heads",
            "normalization": "fixed unit-aware scaling; no test-fitted statistics",
            "actor_critic_representation": (
                "shared explicit feature/time attention latent"
                if condition == "attention_supported"
                else "shared mask-aware GRU latent"
            ),
            "attention_output_contract": (
                {
                    "feature_attention": "[batch, history, features]",
                    "temporal_attention": "[batch, history]",
                    "padded_history_weight": 0,
                    "predictive_checkpoint_transfer": False,
                }
                if condition == "attention_supported"
                else None
            ),
        }
        for condition in (
            "yun_reconstructed",
            "all_supported",
            "attention_supported",
            "selected_control_aware",
        )
    }
