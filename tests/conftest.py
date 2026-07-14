"""Shared synthetic future-BIS artifact fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_modeling_dir(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "modeling"
    dataset_dir.mkdir()
    dynamic_names = [
        "bis",
        "bis_sqi",
        "hr",
        "mbp",
        "sbp",
        "dbp",
        "spo2",
        "etco2",
        "ppf_rate",
        "ppf_volume",
        "ppf_cp",
        "ppf_ce",
        "rftn_rate",
        "rftn_volume",
        "rftn_cp",
        "rftn_ce",
        "bis_slope",
        "bis_error",
    ]
    static_names = ["age", "sex_male", "height", "weight", "bmi", "asa"]
    metadata = {
        "dynamic_feature_names": dynamic_names,
        "static_feature_names": static_names,
        "history_steps": 6,
        "history_window_seconds": 60,
        "prediction_horizon_seconds": 30,
        "resampling_interval_seconds": 10,
        "split_seed": 42,
    }
    (dataset_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    statistics = []
    for name in [*dynamic_names, *static_names]:
        statistics.append(
            {
                "feature_name": name,
                "training_median": 0.0 if name != "bis" else 50.0,
                "training_mean": 0.0 if name != "bis" else 50.0,
                "training_standard_deviation": 1.0 if name != "bis" else 10.0,
                "imputation_value": 0.0,
                "normalization_scale": 1.0,
                "feature_type": "dynamic_continuous",
                "standardized": True,
            }
        )
    pd.DataFrame(statistics).to_csv(
        dataset_dir / "preprocessing_statistics.csv", index=False
    )

    split_cases = {
        "train": np.array([1] * 6 + [2] * 2),
        "val": np.array([3] * 4 + [4] * 4),
        "test": np.array([97] * 4 + [154] * 4),
    }
    split_targets = {
        "train": np.array([35, 45, 65, 50, 38, 70, 45, 55], dtype=np.float32),
        "val": np.array([35, 45, 65, 55, 30, 42, 70, 50], dtype=np.float32),
        "test": np.array([32, 48, 68, 55, 36, 44, 72, 52], dtype=np.float32),
    }
    rng = np.random.default_rng(7)
    remifentanil_indices = [dynamic_names.index(name) for name in dynamic_names if name.startswith("rftn_")]
    for split, case_ids in split_cases.items():
        n_samples = len(case_ids)
        X_dynamic = rng.normal(size=(n_samples, 6, 18)).astype(np.float32)
        X_static = rng.normal(size=(n_samples, 6)).astype(np.float32)
        observation_mask = np.ones((n_samples, 6, 18), dtype=bool)
        if split == "test":
            observation_mask[:, :, remifentanil_indices] = False
        y_bis = split_targets[split]
        np.savez_compressed(
            dataset_dir / f"{split}.npz",
            X_dynamic=X_dynamic,
            X_static=X_static,
            observation_mask=observation_mask,
            y_bis=y_bis,
            y_high_bis=(y_bis > 60).astype(np.int8),
            y_low_bis=(y_bis < 40).astype(np.int8),
        )
        within_case_index = pd.Series(case_ids).groupby(case_ids).cumcount().to_numpy()
        final_timestamps = 50 + within_case_index * 10
        pd.DataFrame(
            {
                "case_id": case_ids,
                "first_input_timestamp": final_timestamps - 50,
                "final_input_timestamp": final_timestamps,
                "target_timestamp": final_timestamps + 30,
            }
        ).to_csv(dataset_dir / f"{split}_metadata.csv", index=False)
    return dataset_dir

