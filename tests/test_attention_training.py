"""End-to-end smoke output tests for factorized-attention training."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.attention_training import (
    AttentionTrainingConfig,
    predict_and_extract_attention,
    run_attention_training,
)
from src.datasets import VitalBISDataset
from src.models.attention import FactorizedAttentionGRU


def test_attention_smoke_pipeline_saves_aligned_outputs(
    synthetic_modeling_dir: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "attention"
    result = run_attention_training(
        AttentionTrainingConfig(
            dataset_dir=synthetic_modeling_dir,
            output_dir=output_dir,
            seed=42,
            device="cpu",
            batch_size=4,
            max_epochs=1,
            patience=1,
            hidden_size=8,
            prediction_hidden_size=4,
            feature_token_embedding_dim=8,
            static_context_dim=4,
            smoke=True,
            exclude_dynamic_features=("bis_error",),
        )
    )

    expected_files = {
        "config.json",
        "best_model.pt",
        "last_model.pt",
        "training_history.csv",
        "val_predictions.csv",
        "val_metrics.json",
        "case_metrics.csv",
        "val_attention.npz",
        "attention_metadata.json",
    }
    assert expected_files.issubset(path.name for path in output_dir.iterdir())
    assert not (output_dir / "test_predictions.csv").exists()
    assert result["checkpoint_reload_predictions_identical"]
    assert result["checkpoint_reload_attention_identical"]

    predictions = pd.read_csv(output_dir / "val_predictions.csv")
    validation_metadata = pd.read_csv(
        synthetic_modeling_dir / "val_metadata.csv"
    )
    with np.load(output_dir / "val_attention.npz", allow_pickle=False) as attention:
        assert attention["feature_attention"].shape == (8, 6, 17)
        assert attention["temporal_attention"].shape == (8, 6)
        assert attention["combined_attention"].shape == (8, 6, 17)
        assert np.array_equal(attention["sample_index"], predictions["sample_index"])
        assert np.array_equal(attention["case_id"], predictions["case_id"])
        assert np.array_equal(attention["case_id"], validation_metadata["case_id"])
        assert np.isfinite(attention["feature_attention"]).all()
        assert np.isfinite(attention["temporal_attention"]).all()
        assert np.isfinite(attention["combined_attention"]).all()

    dataset_metadata = json.loads(
        (synthetic_modeling_dir / "dataset_metadata.json").read_text()
    )
    attention_metadata = json.loads(
        (output_dir / "attention_metadata.json").read_text()
    )
    expected_features = [
        name
        for name in dataset_metadata["dynamic_feature_names"]
        if name != "bis_error"
    ]
    assert attention_metadata["dynamic_feature_names"] == expected_features
    run_config = json.loads((output_dir / "config.json").read_text())
    assert run_config["dynamic_feature_names"] == expected_features
    assert attention_metadata["time_lags_seconds"] == [-50, -40, -30, -20, -10, 0]
    assert attention_metadata["maximum_missing_feature_attention_weight"] == 0.0
    assert attention_metadata["all_attention_values_finite"]
    assert attention_metadata["runtime_breakdown"][
        "repeated_final_dataset_pass_avoided"
    ]


def test_joint_evaluation_calls_attention_model_once_per_batch(
    synthetic_modeling_dir: Path,
) -> None:
    class CountingAttentionGRU(FactorizedAttentionGRU):
        call_count = 0

        def forward(self, *args: torch.Tensor, **kwargs: bool):  # type: ignore[no-untyped-def]
            self.call_count += 1
            return super().forward(*args, **kwargs)

    dataset = VitalBISDataset(synthetic_modeling_dir, "val")
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = CountingAttentionGRU(18, 6, hidden_size=8)

    predictions, attention, _ = predict_and_extract_attention(
        model, loader, nn.HuberLoss(), torch.device("cpu")
    )

    assert model.call_count == len(loader)
    assert len(predictions.y_pred) == len(dataset)
    assert len(attention.sample_indices) == len(dataset)
    assert np.array_equal(predictions.sample_indices, attention.sample_indices)
