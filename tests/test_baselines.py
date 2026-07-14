"""Tests for persistence and non-attention GRU baselines."""

import numpy as np
import torch

from src.models.baselines import GRUBaseline, PersistenceBaseline


def test_persistence_uses_named_final_bis_and_inverse_normalization_not_target() -> None:
    names = ["hr", "bis", "mbp"]
    X_dynamic = np.zeros((2, 6, 3), dtype=np.float32)
    X_dynamic[:, -1, 1] = np.array([1.0, -2.0])
    unrelated_future_target = np.array([99.0, 99.0])
    model = PersistenceBaseline.from_feature_metadata(names, 50.0, 10.0)

    prediction = model.predict(X_dynamic)

    assert np.array_equal(prediction, np.array([60.0, 30.0], dtype=np.float32))
    assert not np.array_equal(prediction, unrelated_future_target)


def test_gru_finite_forward_backward_with_all_missing_optional_features() -> None:
    torch.manual_seed(3)
    model = GRUBaseline(18, 6, hidden_size=16, projection_size=12)
    X_dynamic = torch.randn(4, 6, 18)
    X_static = torch.randn(4, 6)
    mask = torch.ones(4, 6, 18, dtype=torch.bool)
    mask[:, :, 12:16] = False

    prediction = model(X_dynamic, X_static, mask)
    loss = prediction.square().mean()
    loss.backward()

    assert prediction.shape == (4,)
    assert torch.isfinite(prediction).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_observation_mask_changes_gru_input_and_prediction() -> None:
    torch.manual_seed(4)
    model = GRUBaseline(3, 2, hidden_size=8, projection_size=6)
    model.eval()
    X_dynamic = torch.zeros(2, 6, 3)
    X_static = torch.zeros(2, 2)

    observed = model(X_dynamic, X_static, torch.ones_like(X_dynamic, dtype=torch.bool))
    missing = model(X_dynamic, X_static, torch.zeros_like(X_dynamic, dtype=torch.bool))

    assert not torch.equal(observed, missing)

