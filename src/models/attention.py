"""Explicit factorized feature and temporal attention for future-BIS prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FactorizedAttentionOutput:
    """Prediction and factorized model-importance weights from one forward pass."""

    prediction: torch.Tensor
    feature_attention: torch.Tensor
    temporal_attention: torch.Tensor
    combined_attention: torch.Tensor


class FactorizedAttentionGRU(nn.Module):
    """GRU regressor with explicit feature attention followed by temporal attention.

    ``combined_attention`` is a normalized factorized model-importance weight. It
    describes this model's weighting operation and must not be interpreted as a
    causal effect.
    """

    def __init__(
        self,
        dynamic_feature_count: int,
        static_feature_count: int,
        history_steps: int = 6,
        feature_token_embedding_dim: int = 16,
        static_context_dim: int = 16,
        hidden_size: int = 64,
        prediction_hidden_size: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dynamic_feature_count <= 0 or static_feature_count <= 0:
            raise ValueError("Feature counts must be positive.")
        if history_steps <= 0 or feature_token_embedding_dim <= 0:
            raise ValueError("History steps and token dimension must be positive.")

        self.dynamic_feature_count = dynamic_feature_count
        self.static_feature_count = static_feature_count
        self.history_steps = history_steps
        self.feature_token_embedding_dim = feature_token_embedding_dim

        self.value_embedding = nn.Linear(1, feature_token_embedding_dim)
        self.observation_embedding = nn.Linear(
            1, feature_token_embedding_dim, bias=False
        )
        self.feature_identity_embedding = nn.Embedding(
            dynamic_feature_count, feature_token_embedding_dim
        )
        self.time_lag_embedding = nn.Embedding(
            history_steps, feature_token_embedding_dim
        )
        self.token_normalization = nn.LayerNorm(feature_token_embedding_dim)
        self.token_activation = nn.ReLU()

        self.static_mlp = nn.Sequential(
            nn.Linear(static_feature_count, static_context_dim),
            nn.ReLU(),
            nn.Linear(static_context_dim, static_context_dim),
            nn.ReLU(),
        )
        feature_score_hidden = max(feature_token_embedding_dim, static_context_dim)
        self.feature_attention_scorer = nn.Sequential(
            nn.Linear(
                feature_token_embedding_dim + static_context_dim,
                feature_score_hidden,
            ),
            nn.Tanh(),
            nn.Linear(feature_score_hidden, 1),
        )
        self.gru = nn.GRU(
            input_size=feature_token_embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.temporal_attention_scorer = nn.Sequential(
            nn.Linear(hidden_size + static_context_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.prediction_mlp = nn.Sequential(
            nn.Linear(hidden_size + static_context_dim, prediction_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prediction_hidden_size, 1),
        )

    def _validate_inputs(
        self,
        X_dynamic: torch.Tensor,
        X_static: torch.Tensor,
        observation_mask: torch.Tensor,
    ) -> None:
        if X_dynamic.ndim != 3:
            raise ValueError("X_dynamic must have shape [B, L, P].")
        batch_size, history_steps, feature_count = X_dynamic.shape
        if (history_steps, feature_count) != (
            self.history_steps,
            self.dynamic_feature_count,
        ):
            raise ValueError(
                "X_dynamic history/feature shape does not match the configured model: "
                f"{(history_steps, feature_count)}."
            )
        if X_static.shape != (batch_size, self.static_feature_count):
            raise ValueError("X_static must have shape [B, Q] matching the model.")
        if observation_mask.shape != X_dynamic.shape:
            raise ValueError("observation_mask must have the same shape as X_dynamic.")

    def _feature_tokens(
        self, X_dynamic: torch.Tensor, observation_mask: torch.Tensor
    ) -> torch.Tensor:
        """Construct individual feature tokens with shape ``[B, L, P, D]``."""

        batch_size, history_steps, feature_count = X_dynamic.shape
        mask_values = observation_mask.to(dtype=X_dynamic.dtype).unsqueeze(-1)
        values = X_dynamic.unsqueeze(-1)
        feature_ids = torch.arange(feature_count, device=X_dynamic.device)
        time_ids = torch.arange(history_steps, device=X_dynamic.device)
        feature_identity = self.feature_identity_embedding(feature_ids).view(
            1, 1, feature_count, self.feature_token_embedding_dim
        )
        time_identity = self.time_lag_embedding(time_ids).view(
            1, history_steps, 1, self.feature_token_embedding_dim
        )
        tokens = (
            self.value_embedding(values)
            + self.observation_embedding(mask_values)
            + feature_identity
            + time_identity
        )
        tokens = tokens.expand(
            batch_size,
            history_steps,
            feature_count,
            self.feature_token_embedding_dim,
        )
        return self.token_activation(self.token_normalization(tokens))

    @staticmethod
    def _masked_feature_softmax(
        scores: torch.Tensor, observation_mask: torch.Tensor
    ) -> torch.Tensor:
        """Normalize over observed features and return exact zeros for missing ones."""

        mask = observation_mask.to(dtype=torch.bool)
        if not bool(mask.any(dim=-1).all()):
            invalid = (~mask.any(dim=-1)).nonzero(as_tuple=False).tolist()
            raise ValueError(
                "Every historical time step must contain at least one observed dynamic "
                f"feature; empty [batch, time] positions: {invalid}."
            )
        masked_scores = scores.masked_fill(~mask, -torch.inf)
        weights = torch.softmax(masked_scores, dim=-1)
        return weights.masked_fill(~mask, 0.0)

    def forward(
        self,
        X_dynamic: torch.Tensor,
        X_static: torch.Tensor,
        observation_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | FactorizedAttentionOutput:
        self._validate_inputs(X_dynamic, X_static, observation_mask)
        tokens = self._feature_tokens(X_dynamic, observation_mask)
        static_context = self.static_mlp(X_static)

        expanded_static = static_context[:, None, None, :].expand(
            -1, self.history_steps, self.dynamic_feature_count, -1
        )
        feature_scores = self.feature_attention_scorer(
            torch.cat((tokens, expanded_static), dim=-1)
        ).squeeze(-1)
        feature_attention = self._masked_feature_softmax(
            feature_scores, observation_mask
        )
        time_representation = torch.sum(
            feature_attention.unsqueeze(-1) * tokens, dim=2
        )

        gru_output, _ = self.gru(time_representation)
        temporal_static = static_context[:, None, :].expand(
            -1, self.history_steps, -1
        )
        temporal_scores = self.temporal_attention_scorer(
            torch.cat((gru_output, temporal_static), dim=-1)
        ).squeeze(-1)
        temporal_attention = torch.softmax(temporal_scores, dim=-1)
        temporal_context = torch.sum(
            temporal_attention.unsqueeze(-1) * gru_output, dim=1
        )
        prediction = self.prediction_mlp(
            torch.cat((temporal_context, static_context), dim=-1)
        ).squeeze(-1)

        if not return_attention:
            return prediction
        combined_attention = temporal_attention.unsqueeze(-1) * feature_attention
        return FactorizedAttentionOutput(
            prediction=prediction,
            feature_attention=feature_attention,
            temporal_attention=temporal_attention,
            combined_attention=combined_attention,
        )
