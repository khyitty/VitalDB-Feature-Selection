"""Fair non-attention and explicit-attention SB3 feature extractors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.rl_env.state_manifests import fixed_policy_scale


@dataclass(frozen=True)
class AttentionOutput:
    latent: torch.Tensor
    feature_attention: torch.Tensor
    temporal_attention: torch.Tensor
    combined_attention: torch.Tensor


def _feature_scales(feature_names: tuple[str, ...]) -> torch.Tensor:
    return torch.tensor(
        [fixed_policy_scale(name) for name in feature_names], dtype=torch.float32
    )


class _StructuredExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: Any,
        *,
        feature_names: tuple[str, ...],
        latent_dim: int,
    ) -> None:
        super().__init__(observation_space, features_dim=latent_dim)
        history_shape = observation_space["history"].shape
        if history_shape != (6, len(feature_names)):
            raise ValueError(
                f"Observation history shape {history_shape} does not match feature contract."
            )
        self.feature_names = feature_names
        self.history_steps = history_shape[0]
        self.feature_count = history_shape[1]
        self.static_count = observation_space["static"].shape[0]
        self.register_buffer("feature_scales", _feature_scales(feature_names))
        self.register_buffer(
            "static_scales", torch.tensor([100.0, 1.0, 220.0, 200.0])
        )

    def _inputs(
        self, observations: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history = observations["history"].float() / self.feature_scales
        history_mask = observations["history_mask"].bool()
        static = observations["static"].float() / self.static_scales
        target = observations["target_bis"].float() / 100.0
        static_target = torch.cat((static, target), dim=-1)
        if not torch.isfinite(history).all() or not torch.isfinite(static_target).all():
            raise FloatingPointError("Policy extractor received a non-finite observation.")
        if not history_mask.any(dim=1).all():
            raise ValueError("Every policy observation requires at least one valid history row.")
        return history, history_mask, static_target

    @staticmethod
    def _compact(history: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = mask.sum(dim=1).to(dtype=torch.long)
        compact = history.new_zeros(history.shape)
        for batch_index in range(history.shape[0]):
            count = int(lengths[batch_index].item())
            compact[batch_index, :count] = history[batch_index, mask[batch_index]]
        return compact, lengths


class GRUControlExtractor(_StructuredExtractor):
    """Mask-aware GRU baseline with no feature or temporal attention."""

    def __init__(
        self,
        observation_space: Any,
        *,
        feature_names: tuple[str, ...],
        latent_dim: int = 64,
        token_dim: int = 48,
        hidden_dim: int = 72,
        static_dim: int = 16,
    ) -> None:
        super().__init__(
            observation_space, feature_names=feature_names, latent_dim=latent_dim
        )
        self.dynamic_projection = nn.Sequential(
            nn.Linear(self.feature_count, token_dim), nn.ReLU()
        )
        self.gru = nn.GRU(token_dim, hidden_dim, batch_first=True)
        self.static_branch = nn.Sequential(
            nn.Linear(self.static_count + 1, static_dim),
            nn.ReLU(),
            nn.Linear(static_dim, static_dim),
            nn.ReLU(),
        )
        self.latent_projection = nn.Sequential(
            nn.Linear(hidden_dim + static_dim, latent_dim), nn.ReLU()
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        history, mask, static_target = self._inputs(observations)
        projected = self.dynamic_projection(history)
        compact, lengths = self._compact(projected, mask)
        packed = pack_padded_sequence(
            compact, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        static_context = self.static_branch(static_target)
        latent = self.latent_projection(torch.cat((hidden[-1], static_context), dim=-1))
        if not torch.isfinite(latent).all():
            raise FloatingPointError("GRU policy latent became non-finite.")
        return latent


class FactorizedAttentionControlExtractor(_StructuredExtractor):
    """Explicit feature weighting followed by mask-aware temporal weighting.

    The weights are model-internal importance values, not causal effects. They are
    learned end-to-end from RL reward and never initialized from predictive
    attention checkpoints.
    """

    predictive_checkpoint_transfer_supported = False
    actor_critic_attention_sharing = "shared common state representation"

    def __init__(
        self,
        observation_space: Any,
        *,
        feature_names: tuple[str, ...],
        latent_dim: int = 64,
        token_dim: int = 24,
        hidden_dim: int = 64,
        static_dim: int = 16,
    ) -> None:
        super().__init__(
            observation_space, feature_names=feature_names, latent_dim=latent_dim
        )
        self.token_dim = token_dim
        self.value_embedding = nn.Linear(1, token_dim)
        self.feature_embedding = nn.Embedding(self.feature_count, token_dim)
        self.time_embedding = nn.Embedding(self.history_steps, token_dim)
        self.token_norm = nn.LayerNorm(token_dim)
        self.static_branch = nn.Sequential(
            nn.Linear(self.static_count + 1, static_dim),
            nn.ReLU(),
            nn.Linear(static_dim, static_dim),
            nn.ReLU(),
        )
        score_dim = max(token_dim, static_dim)
        self.feature_scorer = nn.Sequential(
            nn.Linear(token_dim + static_dim, score_dim),
            nn.Tanh(),
            nn.Linear(score_dim, 1),
        )
        self.gru = nn.GRU(token_dim, hidden_dim, batch_first=True)
        self.temporal_scorer = nn.Sequential(
            nn.Linear(hidden_dim + static_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.latent_projection = nn.Sequential(
            nn.Linear(hidden_dim + static_dim, latent_dim), nn.ReLU()
        )
        self.last_attention: AttentionOutput | None = None

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        boolean_mask = mask.bool()
        masked = scores.masked_fill(~boolean_mask, -torch.inf)
        all_masked = ~boolean_mask.any(dim=dim, keepdim=True)
        masked = masked.masked_fill(all_masked, 0.0)
        weights = torch.softmax(masked, dim=dim).masked_fill(~boolean_mask, 0.0)
        return weights.masked_fill(all_masked, 0.0)

    def forward_with_attention(
        self, observations: dict[str, torch.Tensor]
    ) -> AttentionOutput:
        history, history_mask, static_target = self._inputs(observations)
        batch_size = history.shape[0]
        feature_ids = torch.arange(self.feature_count, device=history.device)
        time_ids = torch.arange(self.history_steps, device=history.device)
        tokens = (
            self.value_embedding(history.unsqueeze(-1))
            + self.feature_embedding(feature_ids).view(1, 1, self.feature_count, self.token_dim)
            + self.time_embedding(time_ids).view(1, self.history_steps, 1, self.token_dim)
        )
        tokens = torch.relu(self.token_norm(tokens))
        static_context = self.static_branch(static_target)
        expanded_static = static_context[:, None, None, :].expand(
            -1, self.history_steps, self.feature_count, -1
        )
        feature_scores = self.feature_scorer(
            torch.cat((tokens, expanded_static), dim=-1)
        ).squeeze(-1)
        feature_mask = history_mask.unsqueeze(-1).expand(-1, -1, self.feature_count)
        feature_attention = self._masked_softmax(feature_scores, feature_mask, dim=2)
        time_tokens = torch.sum(feature_attention.unsqueeze(-1) * tokens, dim=2)

        compact, lengths = self._compact(time_tokens, history_mask)
        packed = pack_padded_sequence(
            compact, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.gru(packed)
        gru_output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=self.history_steps
        )
        compact_mask = (
            torch.arange(self.history_steps, device=history.device)[None, :]
            < lengths[:, None]
        )
        temporal_static = static_context[:, None, :].expand(-1, self.history_steps, -1)
        temporal_scores = self.temporal_scorer(
            torch.cat((gru_output, temporal_static), dim=-1)
        ).squeeze(-1)
        temporal_compact = self._masked_softmax(temporal_scores, compact_mask, dim=1)
        temporal_attention = history.new_zeros((batch_size, self.history_steps))
        aligned_output = history.new_zeros(
            (batch_size, self.history_steps, gru_output.shape[-1])
        )
        for batch_index in range(batch_size):
            valid_positions = history_mask[batch_index]
            count = int(lengths[batch_index].item())
            temporal_attention[batch_index, valid_positions] = temporal_compact[
                batch_index, :count
            ]
            aligned_output[batch_index, valid_positions] = gru_output[batch_index, :count]
        temporal_context = torch.sum(
            temporal_attention.unsqueeze(-1) * aligned_output, dim=1
        )
        latent = self.latent_projection(
            torch.cat((temporal_context, static_context), dim=-1)
        )
        combined = temporal_attention.unsqueeze(-1) * feature_attention
        if not all(
            torch.isfinite(value).all()
            for value in (latent, feature_attention, temporal_attention, combined)
        ):
            raise FloatingPointError("Attention policy output became non-finite.")
        return AttentionOutput(latent, feature_attention, temporal_attention, combined)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.forward_with_attention(observations)
        self.last_attention = AttentionOutput(
            latent=output.latent.detach(),
            feature_attention=output.feature_attention.detach(),
            temporal_attention=output.temporal_attention.detach(),
            combined_attention=output.combined_attention.detach(),
        )
        return output.latent


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
