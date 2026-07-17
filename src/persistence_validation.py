"""Validation-only latest-BIS persistence evaluation."""

from __future__ import annotations

import json
import logging
import pickle
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.datasets import VitalBISDataset
from src.metrics import patient_level_evaluation, regression_metrics
from src.models.baselines import PersistenceBaseline
from src.preprocessing import PreprocessingArtifact


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PersistenceValidationConfig:
    """Configuration for a sealed-test persistence run."""

    dataset_dir: Path
    output_dir: Path
    validation_only: bool = True


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_bis_normalization(dataset_dir: Path) -> tuple[float, float]:
    path = dataset_dir / "preprocessing.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Training-only preprocessing artifact is missing: {path}")
    with path.open("rb") as handle:
        artifact = pickle.load(handle)
    if not isinstance(artifact, PreprocessingArtifact):
        raise TypeError(f"Unexpected preprocessing artifact type: {type(artifact)!r}")
    if "bis" not in artifact.statistics:
        raise ValueError("Training-only preprocessing artifact does not contain BIS.")
    statistics = artifact.statistics["bis"]
    if not statistics.standardized:
        raise ValueError("Persistence expects standardized BIS input.")
    return float(statistics.training_mean), float(statistics.normalization_scale)


def run_validation_persistence(
    config: PersistenceValidationConfig,
) -> dict[str, Any]:
    """Evaluate persistence on validation without opening train or test arrays."""

    if not config.validation_only:
        raise ValueError("Persistence evaluation requires --validation-only.")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    git_commit = _git_commit_hash()
    config_payload = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "baseline": "latest_observed_bis",
        "evaluation_split": "validation_only",
        "test_used": False,
        "input_files_read": [
            "dataset_metadata.json",
            "preprocessing.pkl",
            "val.npz",
            "val_metadata.csv",
        ],
        "git_commit_hash": git_commit,
    }
    _save_json(config_payload, config.output_dir / "config.json")
    _save_json(
        {"status": "running", "test_used": False, "git_commit_hash": git_commit},
        config.output_dir / "run_status.json",
    )

    try:
        training_mean, training_scale = _load_bis_normalization(config.dataset_dir)
        validation = VitalBISDataset(config.dataset_dir, "val")
        model = PersistenceBaseline.from_feature_metadata(
            validation.dynamic_feature_names,
            training_mean=training_mean,
            training_standard_deviation=training_scale,
        )
        observed = validation.arrays["y_bis"].astype(np.float32, copy=False)
        predicted = model.predict(validation.arrays["X_dynamic"])
        pooled = regression_metrics(observed, predicted)
        patient = patient_level_evaluation(observed, predicted, validation.case_ids)
        errors = predicted - observed
        predictions = pd.DataFrame(
            {
                "sample_index": np.arange(len(validation), dtype=np.int64),
                "case_id": validation.case_ids,
                "target_timestamp": validation.metadata[
                    "target_timestamp"
                ].to_numpy(dtype=np.int64, copy=False),
                "observed_future_bis": observed,
                "predicted_future_bis": predicted,
                "absolute_error": np.abs(errors),
                "squared_error": np.square(errors),
            }
        )
        predictions.to_csv(config.output_dir / "val_predictions.csv", index=False)
        patient.case_metrics.to_csv(config.output_dir / "case_metrics.csv", index=False)
        metrics = {
            "baseline": "latest_observed_bis",
            "evaluation_split": "validation_only",
            "test_used": False,
            "number_of_windows": len(validation),
            "number_of_cases": int(np.unique(validation.case_ids).size),
            "bis_feature_index_found_by_name": model.bis_feature_index,
            "inverse_normalization": {
                "training_mean": training_mean,
                "training_standard_deviation": training_scale,
                "source": "preprocessing.pkl fitted on training cases only",
            },
            "pooled_window": pooled,
            "patient_level": patient.summary,
        }
        _save_json(metrics, config.output_dir / "val_metrics.json")
        _save_json(
            {
                "status": "complete",
                "test_used": False,
                "evaluation_split": "validation_only",
                "number_of_windows": len(validation),
                "git_commit_hash": git_commit,
            },
            config.output_dir / "run_status.json",
        )
        LOGGER.info("Persistence validation MAE: %.4f", pooled["mae"])
        return metrics
    except Exception as error:
        _save_json(
            {
                "status": "failed",
                "test_used": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "git_commit_hash": git_commit,
            },
            config.output_dir / "run_status.json",
        )
        raise
