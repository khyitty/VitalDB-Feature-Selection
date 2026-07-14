"""Tests for checkpointing and end-to-end GRU baseline output generation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.training import TrainingConfig, run_gru_training


def test_end_to_end_training_saves_and_reloads_identical_checkpoint_predictions(
    synthetic_modeling_dir: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "gru"
    result = run_gru_training(
        TrainingConfig(
            dataset_dir=synthetic_modeling_dir,
            output_dir=output_dir,
            seed=42,
            device="cpu",
            batch_size=4,
            max_epochs=1,
            patience=1,
            hidden_size=8,
            projection_size=8,
            static_hidden_size=4,
            prediction_hidden_size=4,
        )
    )

    expected_files = {
        "config.json",
        "best_model.pt",
        "last_model.pt",
        "training_history.csv",
        "val_predictions.csv",
        "test_predictions.csv",
        "val_metrics.json",
        "test_metrics.json",
        "case_metrics.csv",
    }
    assert expected_files.issubset(path.name for path in output_dir.iterdir())
    assert result["checkpoint_reload_predictions_identical"]
    test_metrics = json.loads((output_dir / "test_metrics.json").read_text())
    for case_id in ("97", "154"):
        diagnostic = test_metrics["entirely_missing_remifentanil_case_diagnostics"][case_id]
        assert diagnostic["included"]
        assert diagnostic["all_remifentanil_observation_masks_zero"]
        assert diagnostic["all_predictions_finite"]
        assert diagnostic["patient_metrics_reported"]
    predictions = pd.read_csv(output_dir / "test_predictions.csv")
    assert np.isfinite(predictions.predicted_future_bis).all()

