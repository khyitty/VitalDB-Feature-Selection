"""Guards and metadata for the new simulator-compatible prediction rerun."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

from src.prediction_feature_profiles import load_and_validate_dataset_feature_profile


def validate_main_prediction_run(
    dataset_dir: Path,
    *,
    validation_only: bool,
) -> dict[str, Any]:
    """Require the canonical dataset and keep held-out test outcomes sealed."""

    metadata = load_and_validate_dataset_feature_profile(dataset_dir)
    if not validation_only:
        raise ValueError(
            "Simulator-compatible feature selection is validation-only; the held-out "
            "test split must remain sealed."
        )
    required = {
        "preprocessing_fit_split": "train_only",
        "feature_selection_split_accessed": False,
        "test_results_inspected": False,
        "test_target_summary_sealed": True,
        "final_selected_feature_set_decided": False,
    }
    mismatches = {
        name: {"expected": expected, "observed": metadata.get(name)}
        for name, expected in required.items()
        if metadata.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"Dataset scientific guards are incompatible: {mismatches}")
    return metadata


def write_main_run_context(
    output_dir: Path,
    *,
    model_name: str,
    dataset_metadata: dict[str, Any],
) -> Path:
    """Write a run-level marker without changing the frozen training implementation."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = {
        "schema_version": 1,
        "scientific_role": "main_simulator_compatible_prediction_rerun",
        "feature_profile": dataset_metadata["feature_profile"],
        "feature_profile_version": dataset_metadata["feature_profile_version"],
        "dynamic_feature_names": dataset_metadata["dynamic_feature_names"],
        "static_feature_names": dataset_metadata["static_feature_names"],
        "model_name": model_name,
        "selection_status": "candidate_universe_only_not_final_selection",
        "evaluation_split": "validation_only",
        "test_evaluated": False,
        "legacy_results_used_for_selection": False,
        "git_commit": result.stdout.strip(),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = output_dir / "simulator_compatible_run_context.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
