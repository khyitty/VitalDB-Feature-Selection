"""Tests for model-ready NPZ loading and case-balanced sampling."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.datasets import VitalBISDataset, case_balanced_weights, make_case_balanced_sampler


def test_dataset_shapes_dtypes_metadata_and_labels(synthetic_modeling_dir: Path) -> None:
    dataset = VitalBISDataset(synthetic_modeling_dir, "train")
    item = dataset[0]

    assert len(dataset) == 8
    assert item["X_dynamic"].shape == (6, 18)
    assert item["X_dynamic"].dtype == torch.float32
    assert item["X_static"].shape == (6,)
    assert item["observation_mask"].shape == (6, 18)
    assert item["observation_mask"].dtype == torch.bool
    assert item["case_id"].item() == dataset.metadata.iloc[0].case_id
    assert item["target_timestamp"].item() == dataset.metadata.iloc[0].target_timestamp
    assert item["sample_index"].item() == 0
    assert item["y_high_bis"].item() == int(item["y_bis"].item() > 60)
    assert item["y_low_bis"].item() == int(item["y_bis"].item() < 40)


def test_missing_artifact_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing"):
        VitalBISDataset(tmp_path, "train")


def test_case_balanced_sampler_equalizes_case_probability_mass() -> None:
    case_ids = np.array([1] * 90 + [2] * 10)
    weights = case_balanced_weights(case_ids).numpy()

    assert weights[case_ids == 1].sum() == pytest.approx(
        weights[case_ids == 2].sum()
    )
    sampler = make_case_balanced_sampler(case_ids, seed=42, num_samples=20_000)
    sampled_cases = case_ids[np.fromiter(iter(sampler), dtype=np.int64)]
    case_one_fraction = np.mean(sampled_cases == 1)
    assert case_one_fraction == pytest.approx(0.5, abs=0.02)

