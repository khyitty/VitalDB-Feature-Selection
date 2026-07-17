"""Validation-only GRU feature-group ablation tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.datasets import VitalBISDataset
from src.feature_selection.validation_group_ablation import (
    CANDIDATE_FEATURES,
    CORE_6,
    FULL_12_NO_DUPLICATE,
    NO_CPCE_8,
    PKPD_8,
    PKPD_CUMULATIVE_10,
    REFERENCE_CANDIDATE,
    STATIC_FEATURES,
    ValidationAblationConfig,
    run_validation_group_ablation,
)
from src.prediction_feature_profiles import (
    SIMULATOR_COMPATIBLE_PROFILE,
    get_prediction_feature_profile,
)
from src.rl_env.state_manifests import END_TO_END_DYNAMIC_FEATURES


EXPECTED_CANDIDATES = {
    "bis_only_2": ("bis", "bis_delta_10s"),
    "core_6": (
        "bis",
        "bis_delta_10s",
        "propofol_rate_mg_per_min",
        "propofol_cp_mg_per_l",
        "remifentanil_rate_micrograms_per_min",
        "remifentanil_cp_micrograms_per_l",
    ),
    "pkpd_8": (
        "bis",
        "bis_delta_10s",
        "propofol_rate_mg_per_min",
        "propofol_cp_mg_per_l",
        "remifentanil_rate_micrograms_per_min",
        "remifentanil_cp_micrograms_per_l",
        "propofol_ce_mg_per_l",
        "remifentanil_ce_micrograms_per_l",
    ),
    "pkpd_cumulative_10": (
        "bis",
        "bis_delta_10s",
        "propofol_rate_mg_per_min",
        "propofol_cp_mg_per_l",
        "remifentanil_rate_micrograms_per_min",
        "remifentanil_cp_micrograms_per_l",
        "propofol_ce_mg_per_l",
        "remifentanil_ce_micrograms_per_l",
        "propofol_cumulative_dose_mg",
        "remifentanil_cumulative_dose_micrograms",
    ),
    "full_12_no_duplicate": FULL_12_NO_DUPLICATE,
    "no_cpce_8": NO_CPCE_8,
}


def _write_split(
    dataset_dir: Path,
    split: str,
    case_ids: np.ndarray,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    count = len(case_ids)
    dynamic_count = len(END_TO_END_DYNAMIC_FEATURES)
    dynamic = rng.normal(size=(count, 6, dynamic_count)).astype(np.float32)
    static = rng.normal(size=(count, len(STATIC_FEATURES))).astype(np.float32)
    mask = np.ones_like(dynamic, dtype=bool)
    for feature_index in range(dynamic_count):
        mask[:, feature_index % 6, feature_index] = feature_index % 2 == 0
    targets = np.resize(
        np.array([35.0, 45.0, 65.0, 55.0, 30.0, 70.0], dtype=np.float32),
        count,
    )
    np.savez_compressed(
        dataset_dir / f"{split}.npz",
        X_dynamic=dynamic,
        X_static=static,
        observation_mask=mask,
        y_bis=targets,
        y_high_bis=(targets > 60).astype(np.int8),
        y_low_bis=(targets < 40).astype(np.int8),
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
def ablation_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    profile = get_prediction_feature_profile(SIMULATOR_COMPATIBLE_PROFILE)
    metadata = {
        **profile.as_metadata(),
        "preprocessing_fit_split": "train_only",
        "feature_selection_split_accessed": False,
        "test_results_inspected": False,
        "test_target_summary_sealed": True,
        "final_selected_feature_set_decided": False,
        "history_steps": 6,
        "history_window_seconds": 60,
        "prediction_horizon_seconds": 30,
        "resampling_interval_seconds": 10,
    }
    (dataset_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (dataset_dir / "pkpd_reconstruction_audit.json").write_text(
        json.dumps(
            {
                "causal": True,
                "target_concentration_used": False,
                "recorded_cp_ce_used_as_model_features": False,
            }
        ),
        encoding="utf-8",
    )
    _write_split(dataset_dir, "train", np.repeat(np.arange(1, 5), 2), 11)
    _write_split(dataset_dir, "val", np.repeat(np.arange(11, 14), 2), 12)
    (dataset_dir / "test.npz").write_bytes(b"sealed test must not be loaded")
    (dataset_dir / "test_metadata.csv").write_text(
        "sealed test must not be loaded", encoding="utf-8"
    )
    return dataset_dir


def test_candidate_feature_lists_are_exact_nested_contracts() -> None:
    assert dict(CANDIDATE_FEATURES) == EXPECTED_CANDIDATES
    assert CORE_6 == EXPECTED_CANDIDATES["core_6"]
    assert PKPD_8 == EXPECTED_CANDIDATES["pkpd_8"]
    assert PKPD_CUMULATIVE_10 == EXPECTED_CANDIDATES["pkpd_cumulative_10"]
    assert all("bis_target_error" not in features for features in CANDIDATE_FEATURES.values())
    assert all(len(features) == len(set(features)) for features in CANDIDATE_FEATURES.values())


def test_dataset_subset_keeps_six_lags_static_and_matching_mask_indices(
    ablation_dataset: Path,
) -> None:
    with np.load(ablation_dataset / "train.npz", allow_pickle=False) as source:
        source_dynamic = source["X_dynamic"]
        source_mask = source["observation_mask"]
    dataset = VitalBISDataset(
        ablation_dataset, "train", dynamic_features=CANDIDATE_FEATURES["core_6"]
    )
    indices = [END_TO_END_DYNAMIC_FEATURES.index(name) for name in CORE_6]

    assert dataset.arrays["X_dynamic"].shape == (8, 6, 6)
    assert dataset.arrays["X_static"].shape == (8, 4)
    assert dataset.static_feature_names == STATIC_FEATURES
    assert np.array_equal(dataset.arrays["X_dynamic"], source_dynamic[:, :, indices])
    assert np.array_equal(dataset.arrays["observation_mask"], source_mask[:, :, indices])


def test_all_candidate_smoke_seals_test_writes_summary_and_supports_skip(
    ablation_dataset: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "ablation"
    config = ValidationAblationConfig(
        dataset_dir=ablation_dataset,
        output_dir=output_dir,
        candidate="all",
        seed=42,
        device="cpu",
        validation_only=True,
        smoke=True,
    )
    first = run_validation_group_ablation(config)

    assert first["completed_candidates"] == list(CANDIDATE_FEATURES)
    assert first["test_used"] is False
    summary = pd.read_csv(output_dir / "ablation_summary.csv")
    assert summary["candidate_name"].tolist() == list(CANDIDATE_FEATURES)
    assert not summary["test_used"].astype(bool).any()
    reference = summary.loc[summary["candidate_name"] == REFERENCE_CANDIDATE].iloc[0]
    assert reference["delta_pooled_mae_vs_full12"] == pytest.approx(0.0)
    assert reference["delta_patient_mean_mae_vs_full12"] == pytest.approx(0.0)
    for _, row in summary.iterrows():
        assert row["delta_pooled_mae_vs_full12"] == pytest.approx(
            row["pooled_mae"] - reference["pooled_mae"]
        )
        assert row["delta_patient_mean_mae_vs_full12"] == pytest.approx(
            row["patient_mean_mae"] - reference["patient_mean_mae"]
        )
    analysis = json.loads((output_dir / "analysis_metadata.json").read_text())
    assert analysis["scientific_role"] == "validation_screening_before_final_state_freeze"
    assert analysis["source_selection_method"] == "train_only_patient_level_elastic_net"
    assert analysis["validation_used_for_candidate_comparison"] is True
    assert analysis["test_used"] is False
    assert analysis["final_selected_ppo_manifest_created"] is False
    assert analysis["all_static_features_forced"] is True
    for candidate_name, features in CANDIDATE_FEATURES.items():
        candidate_dir = output_dir / candidate_name
        subset = json.loads((candidate_dir / "feature_subset.json").read_text())
        run_status = json.loads((candidate_dir / "run_status.json").read_text())
        assert subset["train_tensor_shape"] == [8, 6, len(features)]
        assert subset["validation_tensor_shape"] == [6, 6, len(features)]
        assert subset["test_tensor_shape"] is None
        assert subset["static_features"] == list(STATIC_FEATURES)
        assert run_status["test_evaluated"] is False
        assert not (candidate_dir / "test_predictions.csv").exists()
        assert not (candidate_dir / "test_metrics.json").exists()

    mtimes = {
        name: (output_dir / name / "best_model.pt").stat().st_mtime_ns
        for name in CANDIDATE_FEATURES
    }
    second = run_validation_group_ablation(
        ValidationAblationConfig(
            **{
                **config.__dict__,
                "skip_completed": True,
            }
        )
    )
    assert second["completed_candidates"] == []
    assert second["skipped_candidates"] == list(CANDIDATE_FEATURES)
    assert mtimes == {
        name: (output_dir / name / "best_model.pt").stat().st_mtime_ns
        for name in CANDIDATE_FEATURES
    }


def test_individual_candidate_has_null_delta_until_full12_exists(
    ablation_dataset: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "single"
    result = run_validation_group_ablation(
        ValidationAblationConfig(
            dataset_dir=ablation_dataset,
            output_dir=output_dir,
            candidate="bis_only_2",
            validation_only=True,
            smoke=True,
        )
    )

    assert result["summary"][0]["delta_pooled_mae_vs_full12"] is None
    assert result["summary"][0]["delta_patient_mean_mae_vs_full12"] is None


def test_ablation_rejects_missing_validation_only_seal(
    ablation_dataset: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="validation-only"):
        run_validation_group_ablation(
            ValidationAblationConfig(
                dataset_dir=ablation_dataset,
                output_dir=tmp_path / "unsafe",
                validation_only=False,
                smoke=True,
            )
        )
