"""Tests for validation-only persistence and train-only stability selection."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_selection.elastic_net_stability import (
    EXCLUDED_DUPLICATE_FEATURES,
    FORCED_STATIC_FEATURES,
    StabilitySelectionConfig,
    case_balanced_weights,
    group_selected,
    load_train_selection_data,
    patient_bootstrap_multiplicities,
    run_elastic_net_stability,
)
from src.persistence_validation import (
    PersistenceValidationConfig,
    run_validation_persistence,
)
from src.preprocessing import FeatureStatistics, PreprocessingArtifact
from src.rl_env.state_manifests import END_TO_END_DYNAMIC_FEATURES


def _write_split(
    dataset_dir: Path,
    split: str,
    dynamic: np.ndarray,
    static: np.ndarray,
    target: np.ndarray,
    case_ids: np.ndarray,
) -> None:
    np.savez_compressed(
        dataset_dir / f"{split}.npz",
        X_dynamic=dynamic.astype(np.float32),
        X_static=static.astype(np.float32),
        observation_mask=np.ones_like(dynamic, dtype=bool),
        y_bis=target.astype(np.float32),
        y_high_bis=(target > 60.0).astype(np.int8),
        y_low_bis=(target < 40.0).astype(np.int8),
    )
    within_case = pd.Series(case_ids).groupby(case_ids).cumcount().to_numpy()
    final_time = 50 + within_case * 10
    pd.DataFrame(
        {
            "case_id": case_ids,
            "first_input_timestamp": final_time - 50,
            "final_input_timestamp": final_time,
            "target_timestamp": final_time + 30,
        }
    ).to_csv(dataset_dir / f"{split}_metadata.csv", index=False)


@pytest.fixture
def canonical_selection_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "canonical"
    dataset_dir.mkdir()
    dynamic_names = tuple(END_TO_END_DYNAMIC_FEATURES)
    static_names = FORCED_STATIC_FEATURES
    metadata = {
        "dynamic_feature_names": list(dynamic_names),
        "static_feature_names": list(static_names),
        "history_steps": 6,
        "history_window_seconds": 60,
        "prediction_horizon_seconds": 30,
        "resampling_interval_seconds": 10,
    }
    (dataset_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    statistics = {
        name: FeatureStatistics(
            feature_name=name,
            training_median=0.0,
            training_mean=50.0 if name == "bis" else 0.0,
            training_standard_deviation=10.0 if name == "bis" else 1.0,
            imputation_value=0.0,
            normalization_scale=10.0 if name == "bis" else 1.0,
            feature_type="dynamic_continuous",
            standardized=name != "sex_male",
        )
        for name in (*dynamic_names, *static_names)
    }
    artifact = PreprocessingArtifact(statistics, dynamic_names, static_names)
    with (dataset_dir / "preprocessing.pkl").open("wb") as handle:
        pickle.dump(artifact, handle)

    rng = np.random.default_rng(31)
    train_cases = np.repeat(np.arange(1, 9), 5)
    train_dynamic = rng.normal(size=(len(train_cases), 6, len(dynamic_names)))
    train_static_by_case = rng.normal(size=(8, len(static_names)))
    train_static = np.repeat(train_static_by_case, 5, axis=0)
    bis_index = dynamic_names.index("bis")
    cp_index = dynamic_names.index("propofol_cp_mg_per_l")
    train_target = (
        50.0
        + 8.0 * train_dynamic[:, -1, bis_index]
        + 3.0 * train_dynamic[:, -2, cp_index]
        + 0.5 * train_static[:, 0]
        + rng.normal(scale=0.1, size=len(train_cases))
    )
    _write_split(
        dataset_dir,
        "train",
        train_dynamic,
        train_static,
        train_target,
        train_cases,
    )

    val_cases = np.repeat(np.array([101, 102]), 3)
    val_dynamic = np.zeros((len(val_cases), 6, len(dynamic_names)))
    val_dynamic[:, -1, bis_index] = np.array([-1.0, 0.0, 1.0, 2.0, -2.0, 0.5])
    val_static = np.zeros((len(val_cases), len(static_names)))
    val_target = np.array([38.0, 49.0, 57.0, 72.0, 28.0, 53.0])
    _write_split(
        dataset_dir, "val", val_dynamic, val_static, val_target, val_cases
    )

    (dataset_dir / "test.npz").write_bytes(b"must not be opened")
    (dataset_dir / "test_metadata.csv").write_text(
        "must not be opened", encoding="utf-8"
    )
    return dataset_dir


def test_persistence_inverse_transforms_latest_bis_without_test_access(
    canonical_selection_dataset: Path, tmp_path: Path
) -> None:
    output = tmp_path / "persistence"
    metrics = run_validation_persistence(
        PersistenceValidationConfig(canonical_selection_dataset, output)
    )
    predictions = pd.read_csv(output / "val_predictions.csv")
    expected = np.array([40.0, 50.0, 60.0, 70.0, 30.0, 55.0])

    assert np.allclose(predictions["predicted_future_bis"], expected)
    assert metrics["test_used"] is False
    assert set(path.name for path in output.iterdir()) == {
        "val_metrics.json",
        "val_predictions.csv",
        "case_metrics.csv",
        "run_status.json",
        "config.json",
    }
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert all(not name.startswith("test") for name in config["input_files_read"])


def test_persistence_requires_explicit_validation_only(
    canonical_selection_dataset: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="validation-only"):
        run_validation_persistence(
            PersistenceValidationConfig(
                canonical_selection_dataset,
                tmp_path / "unsafe",
                validation_only=False,
            )
        )


def test_selection_matrix_excludes_duplicate_and_groups_all_six_lags(
    canonical_selection_dataset: Path
) -> None:
    data = load_train_selection_data(canonical_selection_dataset)
    coefficients = np.zeros((12, 6))
    coefficients[4, 3] = 1e-3

    assert not set(EXCLUDED_DUPLICATE_FEATURES) & set(data.dynamic_feature_names)
    assert data.dynamic.shape[1] == 12 * 6
    assert data.lag_seconds == (50, 40, 30, 20, 10, 0)
    assert group_selected(coefficients, 1e-6).tolist() == [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_case_balanced_weights_give_every_patient_equal_total_weight() -> None:
    case_ids = np.array([1, 1, 1, 1, 2, 2, 3])
    weights = case_balanced_weights(case_ids)
    totals = [weights[case_ids == case_id].sum() for case_id in (1, 2, 3)]
    bootstrap_weights = case_balanced_weights(case_ids, {1: 2, 2: 1})
    bootstrap_totals = [
        bootstrap_weights[case_ids == case_id].sum() for case_id in (1, 2, 3)
    ]

    assert np.allclose(totals, totals[0])
    assert bootstrap_totals[0] == pytest.approx(2.0 * bootstrap_totals[1])
    assert bootstrap_totals[2] == 0.0


def test_patient_bootstrap_draws_cases_not_windows() -> None:
    case_ids = np.array([1] * 20 + [2] * 2 + [3] * 7 + [4])
    multiplicities = patient_bootstrap_multiplicities(
        case_ids, np.random.default_rng(9)
    )

    assert sum(multiplicities.values()) == 4
    assert set(multiplicities) <= {1, 2, 3, 4}


def _smoke_config(dataset: Path, output: Path) -> StabilitySelectionConfig:
    return StabilitySelectionConfig(
        dataset_dir=dataset,
        output_dir=output,
        seed=17,
        bootstrap_count=3,
        cv_folds=3,
        l1_ratios=(0.9,),
        alphas=(0.001, 0.01),
        coefficient_tolerance=1e-6,
        max_iter=5_000,
        optimization_tolerance=1e-5,
        smoke=True,
    )


def test_stability_smoke_is_reproducible_train_only_and_writes_thresholds(
    canonical_selection_dataset: Path, tmp_path: Path
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    result_a = run_elastic_net_stability(
        _smoke_config(canonical_selection_dataset, first)
    )
    result_b = run_elastic_net_stability(
        _smoke_config(canonical_selection_dataset, second)
    )

    assert result_a["status"] == result_b["status"] == "complete"
    pd.testing.assert_frame_equal(
        pd.read_csv(first / "bootstrap_selection_matrix.csv"),
        pd.read_csv(second / "bootstrap_selection_matrix.csv"),
    )
    summary = pd.read_csv(first / "stability_summary.csv")
    assert "bis_target_error" not in summary["feature_name"].tolist()
    assert not set(FORCED_STATIC_FEATURES) & set(summary["feature_name"])
    metadata = json.loads((first / "analysis_metadata.json").read_text())
    assert metadata["test_used"] is False
    assert metadata["selection_split"] == "train_only"
    assert metadata["bootstrap_unit"] == "patient"
    assert metadata["observation_mask_used_as_model_feature"] is False
    assert all(not name.startswith(("val", "test")) for name in metadata["input_files_read"])
    assert set(path.name for path in first.iterdir()) == {
        "stability_summary.csv",
        "lag_coefficient_summary.csv",
        "bootstrap_selection_matrix.csv",
        "bootstrap_run_summary.csv",
        "hyperparameter_cv_results.csv",
        "selected_frequency_080.json",
        "selected_frequency_060.json",
        "selected_frequency_040.json",
        "run_status.json",
        "config.json",
        "analysis_metadata.json",
    }
    for suffix in ("080", "060", "040"):
        selected = json.loads(
            (first / f"selected_frequency_{suffix}.json").read_text()
        )
        assert "bis" in selected["dynamic_features"]
        assert selected["forced_static_features"] == list(FORCED_STATIC_FEATURES)
        assert selected["final_selected_ppo_manifest"] is False
