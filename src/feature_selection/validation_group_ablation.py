"""Validation-only GRU screening of predefined simulator-compatible feature groups."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import pandas as pd

from src.simulator_compatible_training import validate_main_prediction_run
from src.training import TrainingConfig, run_gru_training


STATIC_FEATURES = ("age_years", "sex_male", "height_cm", "weight_kg")
DETERMINISTIC_DUPLICATE = "bis_target_error"
REFERENCE_CANDIDATE = "full_12_no_duplicate"

CORE_6 = (
    "bis",
    "bis_delta_10s",
    "propofol_rate_mg_per_min",
    "propofol_cp_mg_per_l",
    "remifentanil_rate_micrograms_per_min",
    "remifentanil_cp_micrograms_per_l",
)
PKPD_8 = (*CORE_6, "propofol_ce_mg_per_l", "remifentanil_ce_micrograms_per_l")
PKPD_CUMULATIVE_10 = (
    *PKPD_8,
    "propofol_cumulative_dose_mg",
    "remifentanil_cumulative_dose_micrograms",
)
FULL_12_NO_DUPLICATE = (
    "bis",
    "bis_delta_10s",
    "propofol_rate_mg_per_min",
    "propofol_recent_dose_mg",
    "propofol_cumulative_dose_mg",
    "propofol_cp_mg_per_l",
    "propofol_ce_mg_per_l",
    "remifentanil_rate_micrograms_per_min",
    "remifentanil_recent_dose_micrograms",
    "remifentanil_cumulative_dose_micrograms",
    "remifentanil_cp_micrograms_per_l",
    "remifentanil_ce_micrograms_per_l",
)
NO_CPCE_8 = (
    "bis",
    "bis_delta_10s",
    "propofol_rate_mg_per_min",
    "propofol_recent_dose_mg",
    "propofol_cumulative_dose_mg",
    "remifentanil_rate_micrograms_per_min",
    "remifentanil_recent_dose_micrograms",
    "remifentanil_cumulative_dose_micrograms",
)

CANDIDATE_FEATURES: Mapping[str, tuple[str, ...]] = {
    "bis_only_2": ("bis", "bis_delta_10s"),
    "core_6": CORE_6,
    "pkpd_8": PKPD_8,
    "pkpd_cumulative_10": PKPD_CUMULATIVE_10,
    REFERENCE_CANDIDATE: FULL_12_NO_DUPLICATE,
    "no_cpce_8": NO_CPCE_8,
}

CANDIDATE_REQUIRED_FILES = {
    "best_model.pt",
    "val_metrics.json",
    "val_predictions.csv",
    "case_metrics.csv",
    "training_history.csv",
    "config.json",
    "runtime.json",
    "run_status.json",
    "feature_subset.json",
}
FORBIDDEN_TEST_OUTPUTS = {"test_predictions.csv", "test_metrics.json"}


@dataclass(frozen=True)
class ValidationAblationConfig:
    """Locked configuration for one validation-only candidate screening run."""

    dataset_dir: Path
    output_dir: Path
    candidate: str = "all"
    seed: int = 42
    device: str = "cpu"
    validation_only: bool = True
    smoke: bool = False
    skip_completed: bool = False


def _save_json(payload: Any, path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_candidates(candidate: str) -> tuple[str, ...]:
    """Resolve all or one candidate while preserving the predefined order."""

    if candidate == "all":
        return tuple(CANDIDATE_FEATURES)
    if candidate not in CANDIDATE_FEATURES:
        raise ValueError(
            f"Unknown candidate {candidate!r}; expected all or {list(CANDIDATE_FEATURES)}"
        )
    return (candidate,)


def validate_candidate_contract(dataset_metadata: Mapping[str, Any]) -> None:
    """Require exact feature availability, six lags, and all four static covariates."""

    available = tuple(dataset_metadata["dynamic_feature_names"])
    static = tuple(dataset_metadata["static_feature_names"])
    if static != STATIC_FEATURES:
        raise ValueError(f"Expected static feature order {list(STATIC_FEATURES)}, got {list(static)}")
    if int(dataset_metadata["history_steps"]) != 6:
        raise ValueError("Validation group ablation requires exactly six causal lags.")
    unknown = sorted(
        set().union(*map(set, CANDIDATE_FEATURES.values())) - set(available)
    )
    if unknown:
        raise ValueError(f"Candidate features are absent from the dataset: {unknown}")
    for name, features in CANDIDATE_FEATURES.items():
        if DETERMINISTIC_DUPLICATE in features:
            raise ValueError(f"{name} includes deterministic duplicate {DETERMINISTIC_DUPLICATE}.")
        if len(features) != len(set(features)):
            raise ValueError(f"{name} contains duplicate feature names.")


def _candidate_is_complete(candidate_dir: Path, expected_features: Sequence[str]) -> bool:
    if not CANDIDATE_REQUIRED_FILES.issubset(path.name for path in candidate_dir.glob("*")):
        return False
    status = _load_json(candidate_dir / "run_status.json")
    config = _load_json(candidate_dir / "config.json")
    subset = _load_json(candidate_dir / "feature_subset.json")
    return (
        status.get("status") == "complete"
        and status.get("test_evaluated") is False
        and config.get("evaluate_test") is False
        and tuple(config.get("dynamic_feature_names", ())) == tuple(expected_features)
        and tuple(subset.get("dynamic_features", ())) == tuple(expected_features)
        and not any((candidate_dir / name).exists() for name in FORBIDDEN_TEST_OUTPUTS)
    )


def _feature_subset_payload(
    candidate_name: str,
    features: Sequence[str],
    dataset_metadata: Mapping[str, Any],
    training_result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "candidate_name": candidate_name,
        "scientific_role": "validation_screening_candidate_not_final_ppo_state",
        "dynamic_features": list(features),
        "dynamic_feature_count": len(features),
        "lags_retained_per_dynamic_feature": int(dataset_metadata["history_steps"]),
        "observation_mask_subset_with_same_dynamic_indices": True,
        "static_features": list(STATIC_FEATURES),
        "static_feature_count": len(STATIC_FEATURES),
        "train_tensor_shape": training_result["train_tensor_shape"],
        "validation_tensor_shape": training_result["validation_tensor_shape"],
        "test_tensor_shape": training_result["test_tensor_shape"],
        "parameter_count": training_result["parameter_count"],
        "bis_target_error_excluded_as_deterministic_duplicate": True,
        "test_used": False,
        "final_selected_ppo_manifest_created": False,
    }


def _training_config(
    config: ValidationAblationConfig,
    candidate_dir: Path,
    features: tuple[str, ...],
) -> TrainingConfig:
    return TrainingConfig(
        dataset_dir=config.dataset_dir,
        output_dir=candidate_dir,
        seed=config.seed,
        device=config.device,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        max_epochs=2 if config.smoke else 50,
        patience=8,
        gradient_clip_norm=1.0,
        hidden_size=64,
        dropout=0.0,
        case_balanced_sampling=True,
        smoke=config.smoke,
        evaluate_test=False,
        dynamic_features=features,
    )


def _summary_row(candidate_name: str, candidate_dir: Path) -> dict[str, Any]:
    config = _load_json(candidate_dir / "config.json")
    runtime = _load_json(candidate_dir / "runtime.json")
    metrics = _load_json(candidate_dir / "val_metrics.json")
    history = pd.read_csv(candidate_dir / "training_history.csv")
    subset = _load_json(candidate_dir / "feature_subset.json")
    pooled = metrics["pooled_window"]
    regression = pooled["regression"]
    patient_mae = metrics["patient_level"]["mae"]
    regions = pooled["bis_region_mae"]
    test_used = bool(config.get("evaluate_test")) or any(
        (candidate_dir / name).exists() for name in FORBIDDEN_TEST_OUTPUTS
    )
    return {
        "candidate_name": candidate_name,
        "dynamic_feature_count": len(subset["dynamic_features"]),
        "dynamic_features": json.dumps(subset["dynamic_features"]),
        "static_feature_count": len(subset["static_features"]),
        "parameter_count": int(config["model_parameter_count"]),
        "best_epoch": int(runtime["best_epoch"]),
        "completed_epochs": int(runtime["completed_epochs"]),
        "runtime_seconds": float(runtime["total_internal_runtime_seconds"]),
        "pooled_mae": float(regression["mae"]),
        "pooled_rmse": float(regression["rmse"]),
        "pooled_r_squared": regression["r_squared"],
        "patient_mean_mae": float(patient_mae["mean"]),
        "patient_median_mae": float(patient_mae["median"]),
        "bis_below_40_mae": regions["bis_below_40"],
        "bis_40_to_60_mae": regions["bis_40_to_60"],
        "bis_above_60_mae": regions["bis_above_60"],
        "test_used": test_used,
        "training_history_rows": len(history),
    }


def build_ablation_summary(output_dir: Path) -> list[dict[str, Any]]:
    """Collect completed candidates and calculate deltas against full12 when present."""

    rows = []
    for candidate_name, features in CANDIDATE_FEATURES.items():
        candidate_dir = output_dir / candidate_name
        if _candidate_is_complete(candidate_dir, features):
            rows.append(_summary_row(candidate_name, candidate_dir))
    reference = next(
        (row for row in rows if row["candidate_name"] == REFERENCE_CANDIDATE), None
    )
    for row in rows:
        row["delta_pooled_mae_vs_full12"] = (
            float(row["pooled_mae"] - reference["pooled_mae"])
            if reference is not None
            else None
        )
        row["delta_patient_mean_mae_vs_full12"] = (
            float(row["patient_mean_mae"] - reference["patient_mean_mae"])
            if reference is not None
            else None
        )
    return rows


def _write_summary(output_dir: Path) -> list[dict[str, Any]]:
    rows = build_ablation_summary(output_dir)
    pd.DataFrame(rows).to_csv(output_dir / "ablation_summary.csv", index=False)
    _save_json(rows, output_dir / "ablation_summary.json")
    return rows


def _analysis_metadata(
    config: ValidationAblationConfig,
    dataset_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scientific_role": "validation_screening_before_final_state_freeze",
        "source_selection_method": "train_only_patient_level_elastic_net",
        "source_selection_artifact": (
            "outputs/simulator_compatible_prediction_v2/elastic_net_stability"
        ),
        "candidate_sets_predefined_from_train_only_coefficient_ranking_and_physiological_groups": True,
        "validation_used_for_candidate_comparison": True,
        "test_used": False,
        "input_splits_loaded": ["train", "val"],
        "final_selected_ppo_manifest_created": False,
        "bis_target_error_excluded_as_deterministic_duplicate": True,
        "all_static_features_forced": True,
        "static_features": list(STATIC_FEATURES),
        "history_steps_retained": int(dataset_metadata["history_steps"]),
        "candidate_features": {
            name: list(features) for name, features in CANDIDATE_FEATURES.items()
        },
        "fixed_training_settings": {
            "seed": config.seed,
            "device": config.device,
            "max_epochs": 2 if config.smoke else 50,
            "patience": 8,
            "batch_size": 256,
            "hidden_size": 64,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "gradient_clip": 1.0,
            "dropout": 0.0,
            "case_balanced_sampling": True,
            "validation_only": True,
            "test_evaluation": False,
        },
        "smoke": config.smoke,
        "git_commit_hash": _git_commit_hash(),
    }


def _print_progress(message: str) -> None:
    print(message, flush=True)


def run_validation_group_ablation(
    config: ValidationAblationConfig,
) -> dict[str, Any]:
    """Run selected candidates sequentially and update the validation summary."""

    if not config.validation_only:
        raise ValueError("Validation group ablation requires --validation-only.")
    candidates = resolve_candidates(config.candidate)
    dataset_metadata = validate_main_prediction_run(
        config.dataset_dir, validation_only=True
    )
    validate_candidate_contract(dataset_metadata)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _analysis_metadata(config, dataset_metadata)
    _save_json(metadata, config.output_dir / "analysis_metadata.json")
    root_status = {
        "status": "running",
        "requested_candidates": list(candidates),
        "completed_candidates": [],
        "skipped_candidates": [],
        "test_used": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _save_json(root_status, config.output_dir / "run_status.json")

    try:
        for index, candidate_name in enumerate(candidates, start=1):
            features = CANDIDATE_FEATURES[candidate_name]
            candidate_dir = config.output_dir / candidate_name
            if config.skip_completed and _candidate_is_complete(candidate_dir, features):
                root_status["skipped_candidates"].append(candidate_name)
                _print_progress(
                    f"[{index}/{len(candidates)}] SKIP {candidate_name} (complete)"
                )
                continue
            forbidden = [
                name for name in FORBIDDEN_TEST_OUTPUTS if (candidate_dir / name).exists()
            ]
            if forbidden:
                raise ValueError(
                    f"Refusing candidate directory with test artifacts: {candidate_dir}: {forbidden}"
                )
            started = datetime.now().astimezone()
            timer = perf_counter()
            _print_progress(
                f"[{index}/{len(candidates)}] START {candidate_name} "
                f"dynamic_features={len(features)} at {started.isoformat(timespec='seconds')}"
            )
            result = run_gru_training(
                _training_config(config, candidate_dir, features)
            )
            _save_json(
                _feature_subset_payload(
                    candidate_name, features, dataset_metadata, result
                ),
                candidate_dir / "feature_subset.json",
            )
            elapsed = perf_counter() - timer
            row = _summary_row(candidate_name, candidate_dir)
            root_status["completed_candidates"].append(candidate_name)
            _write_summary(config.output_dir)
            _save_json(root_status, config.output_dir / "run_status.json")
            completed = datetime.now().astimezone()
            _print_progress(
                f"[{index}/{len(candidates)}] COMPLETE {candidate_name} "
                f"at {completed.isoformat(timespec='seconds')} runtime={elapsed:.1f}s "
                f"val_mae={row['pooled_mae']:.4f} "
                f"patient_mean_mae={row['patient_mean_mae']:.4f}"
            )
            if result.get("test_tensor_shape") is not None:
                raise AssertionError("Validation-only candidate unexpectedly evaluated test.")

        rows = _write_summary(config.output_dir)
        if any(row["test_used"] for row in rows):
            raise AssertionError("A completed candidate reports test usage.")
        root_status.update(
            {
                "status": "complete",
                "summary_candidate_count": len(rows),
                "test_used": False,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _save_json(root_status, config.output_dir / "run_status.json")
        return {
            "status": "complete",
            "requested_candidates": list(candidates),
            "completed_candidates": root_status["completed_candidates"],
            "skipped_candidates": root_status["skipped_candidates"],
            "summary": rows,
            "test_used": False,
        }
    except Exception as error:
        root_status.update(
            {
                "status": "failed",
                "test_used": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _save_json(root_status, config.output_dir / "run_status.json")
        raise
