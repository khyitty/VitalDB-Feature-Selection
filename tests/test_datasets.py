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


def test_feature_subset_selects_named_values_and_matching_mask_columns(
    synthetic_modeling_dir: Path,
) -> None:
    full = VitalBISDataset(synthetic_modeling_dir, "train")
    reduced = VitalBISDataset(
        synthetic_modeling_dir,
        "train",
        exclude_dynamic_features=("bis_error",),
    )
    expected_names = tuple(name for name in full.dynamic_feature_names if name != "bis_error")
    expected_indices = [full.dynamic_feature_names.index(name) for name in expected_names]

    assert reduced.dynamic_feature_names == expected_names
    assert reduced.arrays["X_dynamic"].shape == (8, 6, 17)
    assert reduced.arrays["observation_mask"].shape == (8, 6, 17)
    assert np.array_equal(
        reduced.arrays["X_dynamic"], full.arrays["X_dynamic"][:, :, expected_indices]
    )
    assert np.array_equal(
        reduced.arrays["observation_mask"],
        full.arrays["observation_mask"][:, :, expected_indices],
    )


def test_explicit_feature_order_and_unknown_feature_validation(
    synthetic_modeling_dir: Path,
) -> None:
    selected = VitalBISDataset(
        synthetic_modeling_dir, "train", dynamic_features=("hr", "bis")
    )
    assert selected.dynamic_feature_names == ("hr", "bis")
    with pytest.raises(ValueError, match="Unknown dynamic feature"):
        VitalBISDataset(
            synthetic_modeling_dir,
            "train",
            exclude_dynamic_features=("not_a_track",),
        )


def test_no_feature_option_preserves_original_eighteen_feature_loading(
    synthetic_modeling_dir: Path,
) -> None:
    dataset = VitalBISDataset(synthetic_modeling_dir, "train")
    assert dataset.dynamic_feature_names == dataset.source_dynamic_feature_names
    assert dataset.dynamic_feature_indices == tuple(range(18))
    assert dataset.arrays["X_dynamic"].shape[-1] == 18


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
