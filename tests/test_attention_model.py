"""Synthetic tests for explicit factorized feature and temporal attention."""

from pathlib import Path

import pytest
import torch

from src.models.attention import FactorizedAttentionGRU, FactorizedAttentionOutput


def _model(dropout: float = 0.0) -> FactorizedAttentionGRU:
    return FactorizedAttentionGRU(
        dynamic_feature_count=18,
        static_feature_count=6,
        history_steps=6,
        feature_token_embedding_dim=8,
        static_context_dim=4,
        hidden_size=12,
        prediction_hidden_size=6,
        dropout=dropout,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    dynamic = torch.randn(3, 6, 18, generator=generator)
    static = torch.randn(3, 6, generator=generator)
    mask = torch.ones(3, 6, 18, dtype=torch.bool)
    mask[0, :, 12:16] = False
    mask[1, 2, [3, 8, 17]] = False
    return dynamic, static, mask


def test_forward_attention_shapes_normalization_and_missing_zero() -> None:
    model = _model()
    dynamic, static, mask = _inputs()

    prediction = model(dynamic, static, mask)
    output = model(dynamic, static, mask, return_attention=True)

    assert isinstance(output, FactorizedAttentionOutput)
    assert prediction.shape == (3,)
    assert torch.isfinite(prediction).all()
    assert output.feature_attention.shape == (3, 6, 18)
    assert output.temporal_attention.shape == (3, 6)
    assert output.combined_attention.shape == (3, 6, 18)
    assert (output.feature_attention >= 0).all()
    assert (output.temporal_attention >= 0).all()
    assert torch.allclose(output.feature_attention.sum(dim=2), torch.ones(3, 6))
    assert torch.allclose(output.temporal_attention.sum(dim=1), torch.ones(3))
    assert torch.count_nonzero(output.feature_attention[~mask]) == 0
    expected_combined = (
        output.temporal_attention.unsqueeze(-1) * output.feature_attention
    )
    assert torch.equal(output.combined_attention, expected_combined)
    assert torch.allclose(
        output.combined_attention.sum(dim=(1, 2)), torch.ones(3)
    )
    assert all(
        torch.isfinite(tensor).all()
        for tensor in (
            output.feature_attention,
            output.temporal_attention,
            output.combined_attention,
        )
    )


def test_prediction_gradient_reaches_every_required_component() -> None:
    torch.manual_seed(23)
    model = _model()
    dynamic, static, mask = _inputs()

    output = model(dynamic, static, mask, return_attention=True)
    output.prediction.sum().backward()

    required_parameters = {
        "value_embedding": model.value_embedding.weight,
        "feature_attention_scorer": model.feature_attention_scorer[0].weight,
        "gru": model.gru.weight_ih_l0,
        "temporal_attention_scorer": model.temporal_attention_scorer[0].weight,
        "prediction_head": model.prediction_mlp[0].weight,
    }
    for name, parameter in required_parameters.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name


def test_entirely_missing_remifentanil_features_remain_safe_for_backward() -> None:
    model = _model()
    dynamic, static, mask = _inputs()
    remifentanil = [12, 13, 14, 15]
    mask[:, :, remifentanil] = False

    output = model(dynamic, static, mask, return_attention=True)
    output.prediction.square().mean().backward()

    assert torch.isfinite(output.prediction).all()
    assert torch.count_nonzero(output.feature_attention[:, :, remifentanil]) == 0
    assert model.gru.weight_ih_l0.grad is not None
    assert torch.isfinite(model.gru.weight_ih_l0.grad).all()


def test_time_step_with_no_observed_feature_fails_clearly() -> None:
    model = _model()
    dynamic, static, mask = _inputs()
    mask[1, 4, :] = False

    with pytest.raises(ValueError, match="at least one observed dynamic feature"):
        model(dynamic, static, mask)


def test_seventeen_feature_attention_forward_and_backward() -> None:
    torch.manual_seed(29)
    model = FactorizedAttentionGRU(
        dynamic_feature_count=17,
        static_feature_count=6,
        history_steps=6,
        feature_token_embedding_dim=8,
        static_context_dim=4,
        hidden_size=12,
        prediction_hidden_size=6,
    )
    dynamic = torch.randn(3, 6, 17)
    static = torch.randn(3, 6)
    mask = torch.ones(3, 6, 17, dtype=torch.bool)

    output = model(dynamic, static, mask, return_attention=True)
    assert isinstance(output, FactorizedAttentionOutput)
    output.prediction.square().mean().backward()

    assert output.feature_attention.shape == (3, 6, 17)
    assert output.combined_attention.shape == (3, 6, 17)
    assert torch.isfinite(output.prediction).all()
    assert model.gru.weight_ih_l0.grad is not None
    assert torch.isfinite(model.gru.weight_ih_l0.grad).all()


def test_eval_is_deterministic_and_checkpoint_restores_all_outputs(
    tmp_path: Path,
) -> None:
    torch.manual_seed(31)
    model = _model(dropout=0.2).eval()
    dynamic, static, mask = _inputs()
    first = model(dynamic, static, mask, return_attention=True)
    second = model(dynamic, static, mask, return_attention=True)
    assert isinstance(first, FactorizedAttentionOutput)
    assert isinstance(second, FactorizedAttentionOutput)
    for field in (
        "prediction",
        "feature_attention",
        "temporal_attention",
        "combined_attention",
    ):
        assert torch.equal(getattr(first, field), getattr(second, field))

    checkpoint = tmp_path / "attention.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = _model(dropout=0.2).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    restored_output = restored(dynamic, static, mask, return_attention=True)
    assert isinstance(restored_output, FactorizedAttentionOutput)
    for field in (
        "prediction",
        "feature_attention",
        "temporal_attention",
        "combined_attention",
    ):
        assert torch.equal(getattr(first, field), getattr(restored_output, field))
