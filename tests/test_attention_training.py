"""End-to-end smoke output tests for factorized-attention training."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.attention_training import AttentionTrainingConfig, run_attention_training


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
        assert attention["feature_attention"].shape == (8, 6, 18)
        assert attention["temporal_attention"].shape == (8, 6)
        assert attention["combined_attention"].shape == (8, 6, 18)
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
    assert attention_metadata["dynamic_feature_names"] == dataset_metadata[
        "dynamic_feature_names"
    ]
    assert attention_metadata["time_lags_seconds"] == [-50, -40, -30, -20, -10, 0]
    assert attention_metadata["maximum_missing_feature_attention_weight"] == 0.0
    assert attention_metadata["all_attention_values_finite"]
