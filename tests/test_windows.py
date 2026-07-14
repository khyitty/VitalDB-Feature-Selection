"""Tests for exact-timestamp windows and end-to-end persistence."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.io import build_prediction_dataset_from_frame
from src.windows import build_windows


def _window_frame() -> pd.DataFrame:
    rows = []
    for case_id in (1, 2):
        for timestamp in range(0, 101, 10):
            target = 50.0
            if case_id == 1 and timestamp == 80:
                target = 61.0
            if case_id == 1 and timestamp == 90:
                target = 39.0
            rows.append(
                {
                    "caseid": case_id,
                    "timestamp": timestamp,
                    "bis": float(timestamp + case_id),
                    "target_bis": target,
                    "age": float(30 + case_id),
                    "__observed__bis": not (case_id == 1 and timestamp == 20),
                }
            )
    return pd.DataFrame(rows)


def test_windows_have_exact_history_targets_labels_and_shapes() -> None:
    dataset = build_windows(
        _window_frame(),
        dynamic_features=["bis"],
        static_features=["age"],
        history_steps=6,
        interval_seconds=10,
        horizon_seconds=30,
    )

    assert dataset.X_dynamic.shape == (6, 6, 1)
    assert dataset.X_static.shape == (6, 1)
    assert dataset.observation_mask.shape == dataset.X_dynamic.shape
    assert dataset.y_bis.shape == (6,)
    assert (dataset.metadata.target_timestamp - dataset.metadata.final_input_timestamp).eq(30).all()
    assert (dataset.metadata.final_input_timestamp - dataset.metadata.first_input_timestamp).eq(50).all()
    assert dataset.metadata.groupby("case_id").size().to_dict() == {1: 3, 2: 3}
    assert dataset.y_high_bis[:3].tolist() == [1, 0, 0]
    assert dataset.y_low_bis[:3].tolist() == [0, 1, 0]
    assert dataset.windows_removed_missing_future_bis == 6
    assert not dataset.observation_mask[0, 2, 0]


def test_missing_future_bis_is_excluded() -> None:
    frame = _window_frame()
    frame.loc[(frame.caseid == 1) & (frame.timestamp == 80), "target_bis"] = np.nan

    dataset = build_windows(
        frame[frame.caseid == 1],
        dynamic_features=["bis"],
        static_features=["age"],
        history_steps=6,
        interval_seconds=10,
        horizon_seconds=30,
    )

    assert 80 not in dataset.metadata.target_timestamp.tolist()
    assert len(dataset.y_bis) == 2


def _synthetic_raw_cases() -> pd.DataFrame:
    rows = []
    for case_id in (101, 102, 103):
        for timestamp in range(100):
            rows.append(
                {
                    "caseid": case_id,
                    "time_sec": timestamp,
                    "BIS": 35.0 + (case_id - 100) * 10.0 + timestamp / 20.0,
                    "HR": 60.0 + case_id % 3,
                    "PPF_RATE": float(timestamp // 20),
                    "PPF_VOL": float(timestamp),
                    "age": float(30 + case_id % 10),
                    "sex_male": case_id % 2,
                }
            )
    return pd.DataFrame(rows)


def test_end_to_end_synthetic_pipeline_saves_reloadable_ordered_arrays(tmp_path: Path) -> None:
    config = PipelineConfig(output_dir=tmp_path / "modeling")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = build_prediction_dataset_from_frame(
            _synthetic_raw_cases(), config, input_label="synthetic.csv"
        )

    assert result.case_counts == {"train": 1, "val": 1, "test": 1}
    for split_name in ("train", "val", "test"):
        with np.load(result.output_dir / f"{split_name}.npz") as arrays:
            assert arrays["X_dynamic"].shape[1] == 6
            assert arrays["observation_mask"].shape == arrays["X_dynamic"].shape
            assert arrays["y_bis"].shape[0] == arrays["X_dynamic"].shape[0]
        assert (result.output_dir / f"{split_name}_metadata.csv").exists()

    metadata = json.loads((result.output_dir / "dataset_metadata.json").read_text())
    assert metadata["dynamic_feature_names"] == list(result.dynamic_features)
    assert metadata["static_feature_names"] == list(result.static_features)
    assert metadata["history_steps"] == 6
    assert metadata["prediction_horizon_seconds"] == 30
    assert (result.output_dir / "preprocessing.pkl").exists()
    assert (result.output_dir / "feature_manifest.csv").exists()

