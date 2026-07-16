"""Policy-condition registry and fair SB3 policy construction metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn

from src.rl_env.state_adapters import get_state_profile

from .config import PolicyCondition, PrimaryStateProfile, environment_profile_for_condition
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


@dataclass(frozen=True)
class PrimaryStatePolicyContract:
    """Common downstream policy contract for the state-only comparison."""

    state_profile: PrimaryStateProfile
    policy_class: str
    feature_extractor: str
    hidden_layers: tuple[int, ...]
    activation: str
    ordered_feature_names: tuple[str, ...]
    observation_dimension: int

    @property
    def architecture_signature(self) -> tuple[Any, ...]:
        return (
            self.policy_class,
            self.feature_extractor,
            self.hidden_layers,
            self.activation,
        )


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
        main_comparison_role="legacy_secondary_architecture",
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


def primary_state_policy_contract(
    state_profile: PrimaryStateProfile,
    *,
    selected_manifest_path: Path | None = None,
    hidden_dim: int = 64,
) -> PrimaryStatePolicyContract:
    """Resolve one profile without changing the common MLP architecture."""

    profile = get_state_profile(
        state_profile, selected_manifest_path=selected_manifest_path
    )
    return PrimaryStatePolicyContract(
        state_profile=state_profile,
        policy_class="MlpPolicy",
        feature_extractor="stable_baselines3.common.torch_layers.FlattenExtractor",
        hidden_layers=(hidden_dim, hidden_dim),
        activation="Tanh",
        ordered_feature_names=profile.ordered_feature_names,
        observation_dimension=profile.observation_dimension(),
    )


def validate_state_only_comparison(
    contracts: list[PrimaryStatePolicyContract] | tuple[PrimaryStatePolicyContract, ...],
) -> None:
    """Fail if a purported state-only comparison changes policy architecture."""

    if len(contracts) < 2:
        raise ValueError("A state-only comparison requires at least two profiles.")
    signatures = {contract.architecture_signature for contract in contracts}
    if len(signatures) != 1:
        details = {
            contract.state_profile: contract.architecture_signature for contract in contracts
        }
        raise ValueError(
            "State-only comparison changes policy or feature-extractor architecture: "
            f"{details}"
        )


def primary_policy_kwargs(hidden_dim: int = 64) -> dict[str, Any]:
    """Return identical actor/critic MLP settings for every primary state."""

    return {"net_arch": {"pi": [hidden_dim, hidden_dim], "vf": [hidden_dim, hidden_dim]}}


def encoder_contract_registry(latent_dim: int = 64) -> dict[str, Any]:
    registry = {
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
    for value in registry.values():
        value["comparison_class"] = "legacy_or_secondary_architecture_experiment"
    return registry
