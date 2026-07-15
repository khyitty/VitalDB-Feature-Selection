"""Shared synthetic future-BIS artifact fixtures."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.frozen_candidate_retraining import (
    ANCHOR_MAPPING,
    FROZEN_CANDIDATES,
    MODELS,
    SEEDS,
    dataset_fingerprint,
    resolve_git_commit,
)
from src.group_retraining_analysis import EXPECTED_FEATURES
from src.redundancy_audit import REDUCED_FEATURES


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


def _write_json(payload: object, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _frozen_feature_sets() -> dict[str, list[str]]:
    return {
        "full17_reference": list(REDUCED_FEATURES),
        "no_respiratory_anchor": list(EXPECTED_FEATURES["no_respiratory"]),
        "compact11_anchor": list(
            EXPECTED_FEATURES["no_remifentanil_or_respiratory"]
        ),
        "strict_consensus": [
            "bis",
            "bis_sqi",
            "ppf_rate",
            "ppf_volume",
            "ppf_cp",
            "rftn_volume",
            "bis_slope",
        ],
        "compact_consensus": [
            "bis",
            "bis_sqi",
            "hr",
            "sbp",
            "spo2",
            "etco2",
            "ppf_rate",
            "ppf_volume",
            "ppf_cp",
            "rftn_ce",
            "bis_slope",
        ],
    }


def _patient_mae(predictions: pd.DataFrame) -> float:
    errors = np.abs(
        predictions["predicted_future_bis"] - predictions["observed_future_bis"]
    )
    return float(
        pd.DataFrame({"case_id": predictions["case_id"], "error": errors})
        .groupby("case_id")["error"]
        .mean()
        .mean()
    )


def _write_complete_run(
    run_dir: Path,
    *,
    model: str,
    seed: int,
    features: list[str],
    dataset_dir: Path,
    commit: str,
    error: float,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    targets = np.asarray([42.0, 44.0, 50.0, 52.0, 62.0, 64.0])
    case_ids = np.asarray([101, 101, 102, 102, 103, 103])
    predictions = pd.DataFrame(
        {
            "sample_index": np.arange(len(targets)),
            "case_id": case_ids,
            "target_timestamp": np.arange(100, 160, 10),
            "observed_future_bis": targets,
            "predicted_future_bis": targets + error,
        }
    )
    predictions["absolute_error"] = np.abs(
        predictions["predicted_future_bis"] - predictions["observed_future_bis"]
    )
    predictions.to_csv(run_dir / "val_predictions.csv", index=False)
    mae = _patient_mae(predictions)
    _write_json(
        {
            "patient_level": {"mae": {"mean": mae}},
            "pooled_window": {
                "regression": {"mae": mae, "rmse": mae, "r_squared": 0.5}
            },
        },
        run_dir / "val_metrics.json",
    )
    pd.DataFrame(
        {
            "epoch": [1, 2],
            "train_loss": [2.0, 1.0],
            "validation_patient_level_mae": [mae + 0.5, mae],
        }
    ).to_csv(run_dir / "training_history.csv", index=False)
    _write_json(
        {
            "seed": seed,
            "device": "cuda",
            "resolved_device": "cuda",
            "backend": "cuda",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 256,
            "max_epochs": 50,
            "patience": 8,
            "case_balanced_sampling": True,
            "num_workers": 0,
            "dynamic_feature_names": features,
            "static_feature_names": ["age", "sex_male"],
            "selected_training_cases": [1, 2],
            "selected_validation_cases": [101, 102, 103],
            "git_commit_hash": commit,
            "dataset_dir": str(dataset_dir.resolve()),
            "evaluate_test": False,
        },
        run_dir / "config.json",
    )
    _write_json(
        {
            "status": "complete",
            "seed": seed,
            "resolved_device": "cuda",
            "test_evaluated": False,
            "best_epoch": 2,
        },
        run_dir / "run_status.json",
    )
    pd.DataFrame(
        {"split": ["val"] * 3, "case_id": [101, 102, 103], "mae": [mae] * 3}
    ).to_csv(run_dir / "case_metrics.csv", index=False)
    (run_dir / "best_model.pt").write_bytes(b"best")
    (run_dir / "last_model.pt").write_bytes(b"last")
    if model == "gru":
        _write_json({"completed_epochs": 2}, run_dir / "runtime.json")
    else:
        np.savez_compressed(
            run_dir / "val_attention.npz",
            sample_index=np.arange(len(targets)),
            case_id=case_ids,
            feature_attention=np.ones((len(targets), 6, len(features))),
        )
        _write_json(
            {"runtime_breakdown": {"completed_epochs": 2}},
            run_dir / "attention_metadata.json",
        )


@pytest.fixture
def synthetic_frozen_candidate_workspace(tmp_path: Path) -> dict[str, object]:
    """Create source runs and frozen definitions without any held-out test data."""

    dataset_dir = tmp_path / "modeling" / "full"
    dataset_dir.mkdir(parents=True)
    _write_json(
        {
            "dynamic_feature_names": [*REDUCED_FEATURES, "bis_error"],
            "static_feature_names": ["age", "sex_male"],
            "history_steps": 6,
            "history_window_seconds": 60,
            "prediction_horizon_seconds": 30,
            "resampling_interval_seconds": 10,
        },
        dataset_dir / "dataset_metadata.json",
    )
    (dataset_dir / "preprocessing.pkl").write_bytes(b"train-fitted-preprocessing")
    pd.DataFrame({"feature_name": REDUCED_FEATURES}).to_csv(
        dataset_dir / "preprocessing_statistics.csv", index=False
    )
    pd.DataFrame({"case_id": [1, 2]}).to_csv(
        dataset_dir / "train_metadata.csv", index=False
    )
    pd.DataFrame(
        {
            "case_id": [101, 101, 102, 102, 103, 103],
            "target_timestamp": np.arange(100, 160, 10),
        }
    ).to_csv(dataset_dir / "val_metadata.csv", index=False)

    group_root = tmp_path / "group_retraining_validation_only"
    group_training_commit = resolve_git_commit(
        Path.cwd(), "3387a7e", label="synthetic group-training commit"
    )
    group_errors = {
        "full17": 1.0,
        "no_remifentanil": 0.8,
        "no_respiratory": 1.1,
        "no_remifentanil_or_respiratory": 0.7,
    }
    for condition, features in EXPECTED_FEATURES.items():
        for model in MODELS:
            for seed in SEEDS:
                _write_complete_run(
                    group_root / condition / model / f"seed_{seed}",
                    model=model,
                    seed=seed,
                    features=list(features),
                    dataset_dir=dataset_dir,
                    commit=group_training_commit,
                    error=(
                        group_errors[condition]
                        + 0.01 * SEEDS.index(seed)
                        - (0.1 if model == "attention" else 0.0)
                    ),
                )

    group_analysis_dir = group_root / "analysis"
    group_analysis_dir.mkdir()
    fingerprint = dataset_fingerprint(dataset_dir)
    _write_json(
        {
            "run_count": 40,
            "test_split_sealed": True,
            "input_fingerprints": [
                {
                    "path": str(dataset_dir / name),
                    "sha256": details["sha256"],
                    "size_bytes": details["size_bytes"],
                }
                for name, details in fingerprint["files"].items()
            ],
        },
        group_analysis_dir / "analysis_manifest.json",
    )

    feature_sets = _frozen_feature_sets()
    candidate_path = tmp_path / "predictive_feature_selection_30s" / "candidate_subsets.json"
    candidate_path.parent.mkdir()
    _write_json(
        {
            "frozen_retraining_candidates": list(FROZEN_CANDIDATES),
            "all_candidate_subsets": {
                name: {"features": features, "feature_count": len(features)}
                for name, features in feature_sets.items()
            },
        },
        candidate_path,
    )
    return {
        "dataset_dir": dataset_dir,
        "group_root": group_root,
        "group_analysis_dir": group_analysis_dir,
        "candidate_path": candidate_path,
        "output_root": tmp_path / "frozen_candidate_retraining_validation_only",
        "features": feature_sets,
        "write_complete_run": _write_complete_run,
        "anchor_mapping": ANCHOR_MAPPING,
        "copytree": shutil.copytree,
    }
