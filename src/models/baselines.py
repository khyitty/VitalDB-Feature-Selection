"""Persistence and compact non-attention GRU baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class PersistenceBaseline:
    """Predict future BIS from the most recent historical BIS."""

    bis_feature_index: int
    training_mean: float
    training_standard_deviation: float

    @classmethod
    def from_feature_metadata(
        cls,
        dynamic_feature_names: tuple[str, ...] | list[str],
        training_mean: float,
        training_standard_deviation: float,
    ) -> "PersistenceBaseline":
        if "bis" not in dynamic_feature_names:
            raise ValueError("Dynamic feature metadata does not contain 'bis'.")
        if training_standard_deviation <= 0.0:
            raise ValueError("BIS training standard deviation must be positive.")
        return cls(
            bis_feature_index=dynamic_feature_names.index("bis"),
            training_mean=training_mean,
            training_standard_deviation=training_standard_deviation,
        )

    def predict(self, X_dynamic: np.ndarray) -> np.ndarray:
        """Inverse-normalize final historical BIS without reading future targets."""

        normalized_bis = X_dynamic[:, -1, self.bis_feature_index]
        return (
            normalized_bis * self.training_standard_deviation + self.training_mean
        ).astype(np.float32)


class GRUBaseline(nn.Module):
    """Compact GRU regressor that uses values and observation masks, without attention."""

    def __init__(
        self,
        dynamic_feature_count: int,
        static_feature_count: int,
        hidden_size: int = 64,
        projection_size: int = 64,
        static_hidden_size: int = 16,
        prediction_hidden_size: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dynamic_feature_count = dynamic_feature_count
        self.static_feature_count = static_feature_count
        self.dynamic_projection = nn.Sequential(
            nn.Linear(dynamic_feature_count * 2, projection_size),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=projection_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.static_mlp = nn.Sequential(
            nn.Linear(static_feature_count, static_hidden_size),
            nn.ReLU(),
            nn.Linear(static_hidden_size, static_hidden_size),
            nn.ReLU(),
        )
        self.prediction_mlp = nn.Sequential(
            nn.Linear(hidden_size + static_hidden_size, prediction_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prediction_hidden_size, 1),
        )

    def forward(
        self,
        X_dynamic: torch.Tensor,
        X_static: torch.Tensor,
        observation_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = observation_mask.to(dtype=X_dynamic.dtype)
        dynamic_input = torch.cat((X_dynamic, mask), dim=-1)
        projected = self.dynamic_projection(dynamic_input)
        _, hidden = self.gru(projected)
        static_embedding = self.static_mlp(X_static)
        combined = torch.cat((hidden[-1], static_embedding), dim=-1)
        return self.prediction_mlp(combined).squeeze(-1)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable scalar parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

