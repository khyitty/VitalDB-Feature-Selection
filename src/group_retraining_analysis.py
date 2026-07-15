"""Validation-only aggregation for paired group-retraining experiments."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd

from src.redundancy_audit import REDUCED_FEATURES

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

LOGGER = logging.getLogger(__name__)

CONDITIONS = (
    "full17",
    "no_remifentanil",
    "no_respiratory",
    "no_remifentanil_or_respiratory",
)
MODELS = ("gru", "attention")
SEEDS = (7, 21, 42, 84, 123)
REFERENCE_CONDITION = "full17"
PRIMARY_METRIC = "validation_patient_level_mae"
BOOTSTRAP_SEED = 20260715
BOOTSTRAP_REPLICATES = 10_000
T_CRITICAL_95_DF4 = 2.7764451051977987

REMIFENTANIL_FEATURES = ("rftn_rate", "rftn_volume", "rftn_cp", "rftn_ce")
RESPIRATORY_FEATURES = ("spo2", "etco2")
EXCLUDED_FEATURES = {
    "full17": (),
    "no_remifentanil": REMIFENTANIL_FEATURES,
    "no_respiratory": RESPIRATORY_FEATURES,
    "no_remifentanil_or_respiratory": (
        *REMIFENTANIL_FEATURES,
        *RESPIRATORY_FEATURES,
    ),
}
EXPECTED_FEATURES = {
    condition: tuple(
        feature
        for feature in REDUCED_FEATURES
        if feature not in EXCLUDED_FEATURES[condition]
    )
    for condition in CONDITIONS
}

COMMON_REQUIRED_FILES = (
    "run_status.json",
    "config.json",
    "best_model.pt",
    "last_model.pt",
    "training_history.csv",
    "val_predictions.csv",
    "val_metrics.json",
    "case_metrics.csv",
)
MODEL_REQUIRED_FILES = {
    "gru": (*COMMON_REQUIRED_FILES, "runtime.json"),
    "attention": (
        *COMMON_REQUIRED_FILES,
        "val_attention.npz",
        "attention_metadata.json",
    ),
}
FORBIDDEN_TEST_ARTIFACTS = (
    "test_predictions.csv",
    "test_metrics.json",
    "test_attention.npz",
)
PREDICTION_COLUMNS = (
    "sample_index",
    "case_id",
    "target_timestamp",
    "observed_future_bis",
    "predicted_future_bis",
)
REQUIRED_CONFIG_FIELDS = (
    "seed",
    "device",
    "resolved_device",
    "backend",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "max_epochs",
    "patience",
    "case_balanced_sampling",
    "dynamic_feature_names",
    "selected_training_cases",
    "selected_validation_cases",
    "git_commit_hash",
    "dataset_dir",
    "evaluate_test",
)


@dataclass(frozen=True)
class ValidatedRun:
    """One complete validation-only run and its aligned artifacts."""

    condition: str
    model: str
    seed: int
    run_dir: Path
    config: dict[str, Any]
    status: dict[str, Any]
    metrics: dict[str, Any]
    history: pd.DataFrame
    predictions: pd.DataFrame


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object with a path-specific error."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read valid JSON from {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def dump_json(payload: Mapping[str, Any], path: Path) -> None:
    """Write strict JSON after creating only the requested analysis directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def file_fingerprint(path: Path) -> dict[str, Any]:
    """Return a SHA-256 fingerprint for a reproducibility-relevant input."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def expected_run_keys(
    conditions: Sequence[str] = CONDITIONS,
    models: Sequence[str] = MODELS,
    seeds: Sequence[int] = SEEDS,
) -> set[tuple[str, str, int]]:
    """Return the complete condition-by-model-by-seed experiment grid."""

    return {
        (condition, model, int(seed))
        for condition in conditions
        for model in models
        for seed in seeds
    }


def discover_run_directories(experiment_dir: Path) -> dict[tuple[str, str, int], Path]:
    """Discover exactly one config-bearing directory for every expected run."""

    expected = expected_run_keys()
    if len(expected) != 40:
        raise AssertionError(f"Expected experiment grid must contain 40 runs, got {len(expected)}.")
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")

    observed: dict[tuple[str, str, int], list[Path]] = {}
    unexpected: list[str] = []
    for condition in CONDITIONS:
        condition_dir = experiment_dir / condition
        if not condition_dir.is_dir():
            continue
        for config_path in condition_dir.rglob("config.json"):
            relative = config_path.relative_to(experiment_dir)
            if len(relative.parts) < 4:
                unexpected.append(str(relative))
                continue
            model = relative.parts[1]
            config = load_json(config_path)
            try:
                seed = int(config["seed"])
            except (KeyError, TypeError, ValueError):
                unexpected.append(f"{relative} has no valid seed")
                continue
            key = (condition, model, seed)
            if key not in expected:
                unexpected.append(f"{relative} resolves to unexpected key {key}")
                continue
            observed.setdefault(key, []).append(config_path.parent)

    duplicates = {key: paths for key, paths in observed.items() if len(paths) > 1}
    missing = sorted(expected - set(observed))
    if unexpected or duplicates or missing:
        parts = []
        if missing:
            parts.append(f"missing combinations={missing}")
        if duplicates:
            parts.append(
                "duplicate combinations="
                + repr({key: [str(path) for path in paths] for key, paths in duplicates.items()})
            )
        if unexpected:
            parts.append(f"unexpected configs={unexpected}")
        raise ValueError("Invalid 40-run inventory: " + "; ".join(parts))
    return {key: paths[0] for key, paths in observed.items()}


def _require_fields(payload: Mapping[str, Any], fields: Sequence[str], path: Path) -> None:
    missing = [field for field in fields if field not in payload]
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")


def _validate_prediction_frame(frame: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    missing = sorted(set(PREDICTION_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{run_dir} validation predictions lack columns: {missing}")
    if frame.empty:
        raise ValueError(f"{run_dir} validation predictions are empty.")
    numeric = frame.loc[:, PREDICTION_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{run_dir} validation predictions contain non-finite values.")
    if frame["sample_index"].duplicated().any():
        raise ValueError(f"{run_dir} has duplicate validation sample_index rows.")
    return frame.sort_values("sample_index", kind="stable").reset_index(drop=True)


def _validate_one_run(
    condition: str,
    model: str,
    seed: int,
    run_dir: Path,
) -> ValidatedRun:
    missing_files = [
        name for name in MODEL_REQUIRED_FILES[model] if not (run_dir / name).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Incomplete run {condition}/{model}/seed_{seed}: missing {missing_files}"
        )
    forbidden = [name for name in FORBIDDEN_TEST_ARTIFACTS if (run_dir / name).exists()]
    if forbidden:
        raise ValueError(
            f"Test split is not sealed in {condition}/{model}/seed_{seed}: {forbidden}"
        )

    config_path = run_dir / "config.json"
    status_path = run_dir / "run_status.json"
    config = load_json(config_path)
    status = load_json(status_path)
    _require_fields(config, REQUIRED_CONFIG_FIELDS, config_path)
    _require_fields(status, ("status", "test_evaluated", "resolved_device"), status_path)
    if status["status"] != "complete":
        raise ValueError(f"Incomplete run status in {status_path}: {status['status']!r}")
    if status["test_evaluated"] is not False or config["evaluate_test"] is not False:
        raise ValueError(f"Test evaluation was enabled in {run_dir}.")
    if int(config["seed"]) != seed or int(status.get("seed", seed)) != seed:
        raise ValueError(f"Seed metadata does not match seed_{seed} in {run_dir}.")
    if str(config["backend"]).lower() != "cuda":
        raise ValueError(f"Run did not record CUDA backend: {run_dir}")
    if not str(config["resolved_device"]).lower().startswith("cuda"):
        raise ValueError(f"Run did not resolve to a CUDA device: {run_dir}")
    if not str(status["resolved_device"]).lower().startswith("cuda"):
        raise ValueError(f"Run status did not resolve to CUDA: {run_dir}")

    features = tuple(config["dynamic_feature_names"])
    if features != EXPECTED_FEATURES[condition]:
        raise ValueError(
            f"Feature order mismatch in {run_dir}: expected "
            f"{list(EXPECTED_FEATURES[condition])}, got {list(features)}"
        )

    history = pd.read_csv(run_dir / "training_history.csv")
    required_history = {"epoch", "validation_patient_level_mae"}
    if history.empty or not required_history.issubset(history.columns):
        raise ValueError(f"Invalid training history in {run_dir}.")
    if not np.isfinite(history[list(required_history)].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite training history in {run_dir}.")
    predictions = _validate_prediction_frame(
        pd.read_csv(run_dir / "val_predictions.csv"), run_dir
    )
    metrics = load_json(run_dir / "val_metrics.json")
    try:
        patient_mae = float(metrics["patient_level"]["mae"]["mean"])
        pooled = metrics["pooled_window"]["regression"]
        pooled_mae = float(pooled["mae"])
        pooled_rmse = float(pooled["rmse"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid validation metric schema in {run_dir}: {error}") from error
    if not np.isfinite([patient_mae, pooled_mae, pooled_rmse]).all():
        raise ValueError(f"Non-finite validation metrics in {run_dir}.")

    best_history_row = history.loc[history["validation_patient_level_mae"].idxmin()]
    best_epoch = int(best_history_row["epoch"])
    recorded_best_epoch = int(status.get("best_epoch", best_epoch))
    if recorded_best_epoch != best_epoch:
        raise ValueError(
            f"Best epoch in {run_dir} was not selected by minimum validation patient MAE."
        )
    if not math.isclose(
        patient_mae,
        float(best_history_row["validation_patient_level_mae"]),
        rel_tol=1e-5,
        abs_tol=1e-5,
    ):
        raise ValueError(f"Best-checkpoint validation MAE disagrees with history in {run_dir}.")

    validation_cases = sorted(predictions["case_id"].astype(int).unique().tolist())
    if validation_cases != sorted(int(case) for case in config["selected_validation_cases"]):
        raise ValueError(f"Validation patient IDs disagree with config in {run_dir}.")
    return ValidatedRun(
        condition=condition,
        model=model,
        seed=seed,
        run_dir=run_dir,
        config=config,
        status=status,
        metrics=metrics,
        history=history,
        predictions=predictions,
    )


def _assert_prediction_alignment(reference: ValidatedRun, candidate: ValidatedRun) -> None:
    keys = ("sample_index", "case_id", "target_timestamp")
    for column in keys:
        if not np.array_equal(
            reference.predictions[column].to_numpy(),
            candidate.predictions[column].to_numpy(),
        ):
            raise ValueError(
                f"Mismatched validation {column} rows between {reference.run_dir} "
                f"and {candidate.run_dir}."
            )
    if not np.array_equal(
        reference.predictions["observed_future_bis"].to_numpy(dtype=float),
        candidate.predictions["observed_future_bis"].to_numpy(dtype=float),
    ):
        raise ValueError(
            f"Mismatched validation targets between {reference.run_dir} and "
            f"{candidate.run_dir}."
        )


def validate_experiment(experiment_dir: Path) -> list[ValidatedRun]:
    """Validate all 40 runs, the test seal, split identity, and row alignment."""

    directories = discover_run_directories(experiment_dir)
    runs = [
        _validate_one_run(condition, model, seed, directories[(condition, model, seed)])
        for condition, model, seed in sorted(directories)
    ]
    by_key = {(run.condition, run.model, run.seed): run for run in runs}

    training_case_sets = {
        tuple(sorted(int(case) for case in run.config["selected_training_cases"]))
        for run in runs
    }
    validation_case_sets = {
        tuple(sorted(int(case) for case in run.config["selected_validation_cases"]))
        for run in runs
    }
    dataset_dirs = {str(run.config["dataset_dir"]) for run in runs}
    static_features = {tuple(run.config.get("static_feature_names", ())) for run in runs}
    training_commits = {str(run.config["git_commit_hash"]) for run in runs}
    if len(training_case_sets) != 1 or len(validation_case_sets) != 1:
        raise ValueError("Runs do not share identical training and validation patient splits.")
    if len(dataset_dirs) != 1:
        raise ValueError(f"Runs reference different modeling datasets: {sorted(dataset_dirs)}")
    if len(static_features) != 1:
        raise ValueError("Runs do not share identical static feature definitions.")
    if len(training_commits) != 1:
        raise ValueError(
            "Runs contain multiple training commits. Code equivalence was not independently "
            f"established, so analysis is blocked: {sorted(training_commits)}"
        )
    for field in (
        "learning_rate",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "patience",
        "case_balanced_sampling",
    ):
        values = {run.config[field] for run in runs}
        if len(values) != 1:
            raise ValueError(f"Runs use incompatible {field} settings: {sorted(values)}")

    canonical = by_key[(REFERENCE_CONDITION, MODELS[0], SEEDS[0])]
    for run in runs:
        _assert_prediction_alignment(canonical, run)
    return runs


def build_run_level_summary(runs: Sequence[ValidatedRun]) -> pd.DataFrame:
    """Create one tidy validation-only row per condition/model/seed."""

    rows: list[dict[str, Any]] = []
    for run in runs:
        pooled = run.metrics["pooled_window"]["regression"]
        last_epoch = int(run.history["epoch"].max())
        best_epoch = int(run.status.get("best_epoch", run.history.loc[
            run.history["validation_patient_level_mae"].idxmin(), "epoch"
        ]))
        rows.append(
            {
                "condition": run.condition,
                "model": run.model,
                "seed": run.seed,
                "dynamic_feature_count": len(run.config["dynamic_feature_names"]),
                "dynamic_feature_names": json.dumps(run.config["dynamic_feature_names"]),
                "best_epoch": best_epoch,
                PRIMARY_METRIC: float(run.metrics["patient_level"]["mae"]["mean"]),
                "validation_pooled_mae": float(pooled["mae"]),
                "validation_pooled_rmse": float(pooled["rmse"]),
                "validation_pooled_r2": (
                    float(pooled["r_squared"])
                    if pooled.get("r_squared") is not None
                    else np.nan
                ),
                "training_epochs": len(run.history),
                "early_stopping": last_epoch < int(run.config["max_epochs"]),
                "git_commit": str(run.config["git_commit_hash"]),
                "device": str(run.config["resolved_device"]),
                "run_dir": str(run.run_dir),
            }
        )
    return pd.DataFrame(rows).sort_values(["condition", "model", "seed"]).reset_index(drop=True)


def aggregate_conditions(run_level: pd.DataFrame) -> pd.DataFrame:
    """Summarize five paired seeds without treating a p-value as a decision rule."""

    rows: list[dict[str, Any]] = []
    for (condition, model), group in run_level.groupby(["condition", "model"], sort=True):
        values = group[PRIMARY_METRIC].to_numpy(dtype=float)
        if len(values) != len(SEEDS):
            raise ValueError(f"{condition}/{model} does not contain exactly five seeds.")
        standard_deviation = float(np.std(values, ddof=1))
        standard_error = standard_deviation / math.sqrt(len(values))
        mean = float(np.mean(values))
        rows.append(
            {
                "condition": condition,
                "model": model,
                "seed_count": len(values),
                "dynamic_feature_count": int(group["dynamic_feature_count"].iloc[0]),
                "validation_patient_level_mae_mean": mean,
                "validation_patient_level_mae_standard_deviation": standard_deviation,
                "validation_patient_level_mae_median": float(np.median(values)),
                "validation_patient_level_mae_min": float(np.min(values)),
                "validation_patient_level_mae_max": float(np.max(values)),
                "validation_patient_level_mae_standard_error": standard_error,
                "descriptive_t_95_ci_lower": mean - T_CRITICAL_95_DF4 * standard_error,
                "descriptive_t_95_ci_upper": mean + T_CRITICAL_95_DF4 * standard_error,
                "descriptive_ci_note": "Reference-only t interval; 5 seeds only",
                "mean_best_epoch": float(group["best_epoch"].mean()),
            }
        )
    return pd.DataFrame(rows)


def paired_condition_comparisons(
    run_level: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair each group ablation with full17 by model and seed."""

    reference = run_level.loc[
        run_level["condition"] == REFERENCE_CONDITION,
        ["model", "seed", PRIMARY_METRIC],
    ].rename(columns={PRIMARY_METRIC: "reference_mae"})
    rows: list[pd.DataFrame] = []
    for candidate in CONDITIONS[1:]:
        candidate_rows = run_level.loc[
            run_level["condition"] == candidate,
            ["model", "seed", PRIMARY_METRIC],
        ].rename(columns={PRIMARY_METRIC: "candidate_mae"})
        paired = reference.merge(candidate_rows, on=["model", "seed"], validate="one_to_one")
        if len(paired) != len(MODELS) * len(SEEDS):
            raise ValueError(f"Incomplete paired condition rows for {candidate}.")
        paired.insert(1, "candidate_condition", candidate)
        paired["reference_condition"] = REFERENCE_CONDITION
        paired["paired_delta_candidate_minus_full17"] = (
            paired["candidate_mae"] - paired["reference_mae"]
        )
        paired["relative_percentage_change"] = (
            100.0 * paired["paired_delta_candidate_minus_full17"] / paired["reference_mae"]
        )
        rows.append(paired)
    deltas = pd.concat(rows, ignore_index=True)

    summaries: list[dict[str, Any]] = []
    for (model, candidate), group in deltas.groupby(
        ["model", "candidate_condition"], sort=True
    ):
        values = group["paired_delta_candidate_minus_full17"].to_numpy(dtype=float)
        summaries.append(
            {
                "model": model,
                "candidate_condition": candidate,
                "reference_condition": REFERENCE_CONDITION,
                "seed_count": len(values),
                "mean_delta": float(np.mean(values)),
                "delta_standard_deviation": float(np.std(values, ddof=1)),
                "median_delta": float(np.median(values)),
                "min_delta": float(np.min(values)),
                "max_delta": float(np.max(values)),
                "candidate_better_seed_count": int(np.sum(values < 0.0)),
                "mean_relative_percentage_change": float(
                    group["relative_percentage_change"].mean()
                ),
                "delta_direction": "negative favors candidate; positive favors full17",
            }
        )
    return deltas, pd.DataFrame(summaries)


def paired_model_comparisons(
    run_level: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair attention with GRU within each condition and seed."""

    gru = run_level.loc[
        run_level["model"] == "gru", ["condition", "seed", PRIMARY_METRIC]
    ].rename(columns={PRIMARY_METRIC: "gru_mae"})
    attention = run_level.loc[
        run_level["model"] == "attention", ["condition", "seed", PRIMARY_METRIC]
    ].rename(columns={PRIMARY_METRIC: "attention_mae"})
    deltas = gru.merge(attention, on=["condition", "seed"], validate="one_to_one")
    if len(deltas) != len(CONDITIONS) * len(SEEDS):
        raise ValueError("Incomplete GRU/attention paired rows.")
    deltas["paired_delta_attention_minus_gru"] = deltas["attention_mae"] - deltas["gru_mae"]
    summaries: list[dict[str, Any]] = []
    for condition, group in deltas.groupby("condition", sort=True):
        values = group["paired_delta_attention_minus_gru"].to_numpy(dtype=float)
        summaries.append(
            {
                "condition": condition,
                "seed_count": len(values),
                "mean_delta": float(np.mean(values)),
                "delta_standard_deviation": float(np.std(values, ddof=1)),
                "median_delta": float(np.median(values)),
                "min_delta": float(np.min(values)),
                "max_delta": float(np.max(values)),
                "attention_better_seed_count": int(np.sum(values < 0.0)),
                "delta_direction": "negative favors attention; positive favors GRU",
            }
        )
    return deltas, pd.DataFrame(summaries)


def patient_level_metrics(runs: Sequence[ValidatedRun]) -> pd.DataFrame:
    """Aggregate window absolute errors within patients for every run."""

    rows: list[pd.DataFrame] = []
    for run in runs:
        frame = run.predictions.copy()
        frame["absolute_error"] = np.abs(
            frame["predicted_future_bis"] - frame["observed_future_bis"]
        )
        patient = (
            frame.groupby("case_id", as_index=False)
            .agg(patient_mae=("absolute_error", "mean"), window_count=("absolute_error", "size"))
        )
        patient.insert(0, "seed", run.seed)
        patient.insert(0, "model", run.model)
        patient.insert(0, "condition", run.condition)
        rows.append(patient)
    result = pd.concat(rows, ignore_index=True)
    if not np.isfinite(result["patient_mae"].to_numpy(dtype=float)).all():
        raise ValueError("Patient-level MAE contains non-finite values.")
    return result.sort_values(["condition", "model", "seed", "case_id"]).reset_index(drop=True)


def hierarchical_paired_bootstrap(
    paired_patient_differences: pd.DataFrame,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Resample paired seeds and patients while preserving candidate/reference pairs."""

    required = {"seed", "case_id", "paired_delta"}
    if not required.issubset(paired_patient_differences.columns):
        raise ValueError(f"Bootstrap input lacks columns: {sorted(required - set(paired_patient_differences))}")
    if replicates <= 0:
        raise ValueError("Bootstrap replicates must be positive.")
    frame = paired_patient_differences.loc[:, ["seed", "case_id", "paired_delta"]].copy()
    if frame.duplicated(["seed", "case_id"]).any():
        raise ValueError("Bootstrap input has duplicate seed/patient pairs.")
    if not np.isfinite(frame["paired_delta"].to_numpy(dtype=float)).all():
        raise ValueError("Bootstrap paired differences must be finite.")
    seed_values = sorted(frame["seed"].astype(int).unique().tolist())
    if len(seed_values) != len(SEEDS):
        raise ValueError(f"Hierarchical bootstrap requires {len(SEEDS)} paired seeds.")
    patient_sets = {
        tuple(sorted(group["case_id"].astype(int).tolist()))
        for _, group in frame.groupby("seed")
    }
    if len(patient_sets) != 1:
        raise ValueError("Patient IDs are not aligned across seeds for hierarchical bootstrap.")
    patient_ids = next(iter(patient_sets))
    matrix = np.empty((len(seed_values), len(patient_ids)), dtype=float)
    for seed_index, seed_value in enumerate(seed_values):
        group = frame.loc[frame["seed"] == seed_value].set_index("case_id")
        matrix[seed_index] = group.loc[list(patient_ids), "paired_delta"].to_numpy(dtype=float)

    generator = np.random.default_rng(seed)
    bootstrap_means = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled_seed_indices = generator.integers(0, len(seed_values), size=len(seed_values))
        sampled_seed_means = []
        for seed_index in sampled_seed_indices:
            patient_indices = generator.integers(0, len(patient_ids), size=len(patient_ids))
            sampled_seed_means.append(float(matrix[seed_index, patient_indices].mean()))
        bootstrap_means[replicate] = float(np.mean(sampled_seed_means))
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return {
        "point_estimate_mean_delta": float(matrix.mean()),
        "percentile_95_ci_lower": float(lower),
        "percentile_95_ci_upper": float(upper),
        "seed_count": len(seed_values),
        "patient_count_per_seed": len(patient_ids),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "resampling_units": "paired seed and validation patient",
    }


def _paired_patient_condition_differences(
    patient: pd.DataFrame, model: str, candidate: str
) -> pd.DataFrame:
    reference = patient.loc[
        (patient["model"] == model) & (patient["condition"] == REFERENCE_CONDITION),
        ["seed", "case_id", "patient_mae"],
    ].rename(columns={"patient_mae": "reference_mae"})
    candidate_rows = patient.loc[
        (patient["model"] == model) & (patient["condition"] == candidate),
        ["seed", "case_id", "patient_mae"],
    ].rename(columns={"patient_mae": "candidate_mae"})
    paired = reference.merge(candidate_rows, on=["seed", "case_id"], validate="one_to_one")
    if len(paired) != len(reference) or len(paired) != len(candidate_rows):
        raise ValueError(f"Patient pairing failed for {model}: {candidate} versus full17.")
    paired["paired_delta"] = paired["candidate_mae"] - paired["reference_mae"]
    return paired


def _paired_patient_model_differences(patient: pd.DataFrame, condition: str) -> pd.DataFrame:
    gru = patient.loc[
        (patient["condition"] == condition) & (patient["model"] == "gru"),
        ["seed", "case_id", "patient_mae"],
    ].rename(columns={"patient_mae": "reference_mae"})
    attention = patient.loc[
        (patient["condition"] == condition) & (patient["model"] == "attention"),
        ["seed", "case_id", "patient_mae"],
    ].rename(columns={"patient_mae": "candidate_mae"})
    paired = gru.merge(attention, on=["seed", "case_id"], validate="one_to_one")
    if len(paired) != len(gru) or len(paired) != len(attention):
        raise ValueError(f"Patient pairing failed for attention versus GRU in {condition}.")
    paired["paired_delta"] = paired["candidate_mae"] - paired["reference_mae"]
    return paired


def bootstrap_all_contrasts(
    patient: pd.DataFrame,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Bootstrap all condition and model contrasts with deterministic sub-seeds."""

    rows: list[dict[str, Any]] = []
    contrast_index = 0
    for model in MODELS:
        for candidate in CONDITIONS[1:]:
            paired = _paired_patient_condition_differences(patient, model, candidate)
            result = hierarchical_paired_bootstrap(
                paired, replicates=replicates, seed=seed + contrast_index
            )
            rows.append(
                {
                    "comparison_type": "condition_vs_full17",
                    "model": model,
                    "condition": candidate,
                    "reference": REFERENCE_CONDITION,
                    "contrast": f"{candidate} - {REFERENCE_CONDITION}",
                    "negative_delta_favors": candidate,
                    **result,
                }
            )
            contrast_index += 1
    for condition in CONDITIONS:
        paired = _paired_patient_model_differences(patient, condition)
        result = hierarchical_paired_bootstrap(
            paired, replicates=replicates, seed=seed + contrast_index
        )
        rows.append(
            {
                "comparison_type": "attention_vs_gru",
                "model": "attention_minus_gru",
                "condition": condition,
                "reference": "gru",
                "contrast": "attention - gru",
                "negative_delta_favors": "attention",
                **result,
            }
        )
        contrast_index += 1
    return pd.DataFrame(rows)


def pareto_candidates(condition_aggregate: pd.DataFrame) -> pd.DataFrame:
    """Mark candidates dominated on both feature count and mean validation MAE."""

    rows: list[dict[str, Any]] = []
    for model, group in condition_aggregate.groupby("model", sort=True):
        records = group.to_dict(orient="records")
        best_condition = min(
            records, key=lambda row: row["validation_patient_level_mae_mean"]
        )["condition"]
        for row in records:
            dominators = [
                other["condition"]
                for other in records
                if other["condition"] != row["condition"]
                and other["dynamic_feature_count"] <= row["dynamic_feature_count"]
                and other["validation_patient_level_mae_mean"]
                <= row["validation_patient_level_mae_mean"]
                and (
                    other["dynamic_feature_count"] < row["dynamic_feature_count"]
                    or other["validation_patient_level_mae_mean"]
                    < row["validation_patient_level_mae_mean"]
                )
            ]
            rows.append(
                {
                    "model": model,
                    "condition": row["condition"],
                    "dynamic_feature_count": int(row["dynamic_feature_count"]),
                    "mean_validation_patient_level_mae": float(
                        row["validation_patient_level_mae_mean"]
                    ),
                    "dominated": bool(dominators),
                    "pareto_frontier": not dominators,
                    "dominated_by": ",".join(sorted(dominators)),
                    "best_mean_performance": row["condition"] == best_condition,
                }
            )
    result = pd.DataFrame(rows)
    result["simplest_non_dominated"] = False
    for model, group in result.loc[result["pareto_frontier"]].groupby("model"):
        minimum_features = int(group["dynamic_feature_count"].min())
        selector = (
            (result["model"] == model)
            & result["pareto_frontier"]
            & (result["dynamic_feature_count"] == minimum_features)
        )
        result.loc[selector, "simplest_non_dominated"] = True
    return result.sort_values(["model", "dynamic_feature_count", "condition"])


def _plot_seed_distribution(run_level: pd.DataFrame, path: Path) -> None:
    labels = [f"{condition}\n{model}" for model in MODELS for condition in CONDITIONS]
    groups = [
        run_level.loc[
            (run_level["condition"] == condition) & (run_level["model"] == model),
            PRIMARY_METRIC,
        ].to_numpy(dtype=float)
        for model in MODELS
        for condition in CONDITIONS
    ]
    figure, axis = plt.subplots(figsize=(13, 6))
    axis.boxplot(groups, positions=np.arange(len(groups)), widths=0.55, showfliers=False)
    for index, values in enumerate(groups):
        axis.scatter(np.full(len(values), index), values, color="#16697a", zorder=3)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    axis.set_ylabel("Validation patient-level MAE (BIS)")
    axis.set_title("Validation-only seed-level MAE by condition and model")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_condition_pairs(run_level: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    x = np.arange(len(CONDITIONS))
    for axis, model in zip(axes, MODELS, strict=True):
        model_rows = run_level.loc[run_level["model"] == model]
        for seed in SEEDS:
            seed_rows = model_rows.loc[model_rows["seed"] == seed].set_index("condition")
            values = seed_rows.loc[list(CONDITIONS), PRIMARY_METRIC].to_numpy(dtype=float)
            axis.plot(x, values, marker="o", alpha=0.75, label=f"seed {seed}")
        axis.set_xticks(x, CONDITIONS, rotation=25, ha="right")
        axis.set_title(f"{model}: validation-only paired conditions")
        axis.set_ylabel("Validation patient-level MAE (BIS)")
        axis.grid(axis="y", alpha=0.25)
    axes[-1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_bootstrap_forest(bootstrap: pd.DataFrame, path: Path) -> None:
    condition_rows = bootstrap.loc[
        bootstrap["comparison_type"] == "condition_vs_full17"
    ].reset_index(drop=True)
    y = np.arange(len(condition_rows))
    means = condition_rows["point_estimate_mean_delta"].to_numpy(dtype=float)
    lower = condition_rows["percentile_95_ci_lower"].to_numpy(dtype=float)
    upper = condition_rows["percentile_95_ci_upper"].to_numpy(dtype=float)
    labels = [
        f"{row.model}: {row.condition} - full17"
        for row in condition_rows.itertuples()
    ]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.errorbar(means, y, xerr=np.vstack((means - lower, upper - means)), fmt="o", capsize=4)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_yticks(y, labels)
    axis.set_xlabel("Paired validation patient MAE delta (negative favors candidate)")
    axis.set_title("Validation-only hierarchical paired bootstrap intervals")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_pareto(pareto: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, model in zip(axes, MODELS, strict=True):
        group = pareto.loc[pareto["model"] == model]
        for row in group.itertuples():
            color = "#2a9d8f" if row.pareto_frontier else "#9aa0a6"
            axis.scatter(row.dynamic_feature_count, row.mean_validation_patient_level_mae, color=color, s=60)
            axis.annotate(row.condition, (row.dynamic_feature_count, row.mean_validation_patient_level_mae), xytext=(4, 4), textcoords="offset points", fontsize=8)
        axis.set_xlabel("Dynamic feature count")
        axis.set_title(f"{model}: validation-only Pareto candidates")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean validation patient-level MAE (BIS)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_model_pairs(run_level: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10, 9), sharey=True)
    for axis, condition in zip(axes.flat, CONDITIONS, strict=True):
        group = run_level.loc[run_level["condition"] == condition]
        for seed in SEEDS:
            seed_rows = group.loc[group["seed"] == seed].set_index("model")
            values = seed_rows.loc[["gru", "attention"], PRIMARY_METRIC].to_numpy(dtype=float)
            axis.plot([0, 1], values, marker="o", alpha=0.75)
        axis.set_xticks([0, 1], ["GRU", "Attention"])
        axis.set_title(f"{condition}: validation-only")
        axis.set_ylabel("Patient-level MAE (BIS)")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_figures(
    run_level: pd.DataFrame,
    bootstrap: pd.DataFrame,
    pareto: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    """Save the five required validation-only figures with matplotlib."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        figures_dir / "validation_seed_mae_distribution.png",
        figures_dir / "validation_paired_condition_lines.png",
        figures_dir / "validation_paired_delta_bootstrap_forest.png",
        figures_dir / "validation_feature_count_pareto.png",
        figures_dir / "validation_gru_attention_pairs.png",
    ]
    _plot_seed_distribution(run_level, paths[0])
    _plot_condition_pairs(run_level, paths[1])
    _plot_bootstrap_forest(bootstrap, paths[2])
    _plot_pareto(pareto, paths[3])
    _plot_model_pairs(run_level, paths[4])
    return paths


def _frame_text(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    selected = frame.loc[:, list(columns)] if columns is not None else frame
    return "```text\n" + selected.to_string(index=False) + "\n```"


def build_markdown_report(
    condition_aggregate: pd.DataFrame,
    condition_contrasts: pd.DataFrame,
    model_contrasts: pd.DataFrame,
    bootstrap: pd.DataFrame,
    pareto: pd.DataFrame,
    training_commits: Sequence[str],
) -> str:
    """Build a cautious validation-only report from computed tables."""

    recommendations = []
    for model in MODELS:
        model_pareto = pareto.loc[pareto["model"] == model]
        best = model_pareto.loc[model_pareto["best_mean_performance"]].iloc[0]
        simplest = model_pareto.loc[model_pareto["simplest_non_dominated"]].sort_values(
            "mean_validation_patient_level_mae"
        ).iloc[0]
        for candidate_row in model_pareto.loc[model_pareto["pareto_frontier"]].itertuples():
            aggregate_row = condition_aggregate.loc[
                (condition_aggregate["model"] == model)
                & (condition_aggregate["condition"] == candidate_row.condition)
            ].iloc[0]
            if candidate_row.condition == REFERENCE_CONDITION:
                recommendations.append(
                    f"- {model}/{candidate_row.condition}: mean validation patient MAE "
                    f"{aggregate_row['validation_patient_level_mae_mean']:.4f}, SD "
                    f"{aggregate_row['validation_patient_level_mae_standard_deviation']:.4f}, "
                    f"{int(candidate_row.dynamic_feature_count)} features, non-dominated reference."
                )
                continue
            contrast_row = condition_contrasts.loc[
                (condition_contrasts["model"] == model)
                & (
                    condition_contrasts["candidate_condition"]
                    == candidate_row.condition
                )
            ].iloc[0]
            bootstrap_row = bootstrap.loc[
                (bootstrap["comparison_type"] == "condition_vs_full17")
                & (bootstrap["model"] == model)
                & (bootstrap["condition"] == candidate_row.condition)
            ].iloc[0]
            recommendations.append(
                f"- {model}/{candidate_row.condition}: mean validation patient MAE "
                f"{aggregate_row['validation_patient_level_mae_mean']:.4f}, SD "
                f"{aggregate_row['validation_patient_level_mae_standard_deviation']:.4f}, "
                f"mean paired delta vs full17 {contrast_row['mean_delta']:+.4f}, "
                f"hierarchical 95% CI "
                f"[{bootstrap_row['percentile_95_ci_lower']:+.4f}, "
                f"{bootstrap_row['percentile_95_ci_upper']:+.4f}], candidate better in "
                f"{int(contrast_row['candidate_better_seed_count'])}/5 seeds, "
                f"{int(candidate_row.dynamic_feature_count)} features, non-dominated."
            )
        recommendations.append(
            f"- {model} summary: lowest observed mean MAE candidate is `{best['condition']}`; "
            f"the simplest non-dominated candidate is `{simplest['condition']}` "
            f"({int(simplest['dynamic_feature_count'])} dynamic features). These are candidates "
            "for the next feature-selection stage, not final feature sets."
        )
    bootstrap_columns = (
        "comparison_type",
        "model",
        "condition",
        "point_estimate_mean_delta",
        "percentile_95_ci_lower",
        "percentile_95_ci_upper",
    )
    return f"""# Validation-Only Group Retraining Analysis

## 1. Analysis objective
This analysis compares physiological feature-group ablations as candidate RL state representations. Future-BIS prediction is an intermediate screening task, not the final propofol-control objective.

## 2. Experimental design
All 40 full runs were included: four conditions, two models, and five paired seeds (7, 21, 42, 84, 123). The primary metric is validation patient-level MAE, where lower is better. This is a group-ablation experiment, not completed individual feature selection. Training commit(s): {', '.join(training_commits)}.

## 3. Test seal
This report is validation-only. Every run recorded `test_evaluated=false` and `evaluate_test=false`, and no forbidden test prediction, metric, or attention artifact was present. The analysis did not load a test split.

## 4. Run completion and data integrity
All 40 runs were complete, used CUDA, shared the same training and validation patient sets and modeling dataset, and had exactly aligned validation patient IDs, timestamps, sample rows, and targets.

## 5. Results by condition
The t-based intervals below are descriptive reference intervals only; there are five seeds per cell.

{_frame_text(condition_aggregate)}

## 6. Paired contrasts against full17
Negative delta favors the candidate and positive delta favors full17. The contrasts are paired by seed. No p-value is used as an automatic winner rule.

{_frame_text(condition_contrasts)}

## 7. GRU versus Attention
Negative Attention-minus-GRU delta favors Attention. Model comparisons are reported separately from feature-condition comparisons.

{_frame_text(model_contrasts)}

## 8. Patient-level hierarchical bootstrap
Windows were first aggregated within each validation patient. Bootstrap replicates resampled paired seeds and patients, never independent windows. Percentile intervals reflect only five trained seeds and must be interpreted cautiously.

{_frame_text(bootstrap, bootstrap_columns)}

## 9. Pareto results
A condition is dominated only when another condition has no more features and no worse mean validation MAE, with at least one strict improvement. No non-inferiority margin was specified, so this report does not claim equivalence or non-inferiority.

{_frame_text(pareto)}

## 10. Candidates for the next stage
{chr(10).join(recommendations)}

Candidate decisions should consider mean patient-level MAE, paired delta versus full17, hierarchical bootstrap interval, five-seed consistency, variability, feature count, and Pareto status together. Small observed differences are not automatically clinically meaningful.

## 11. Interpretation limits
- Only five paired seeds were used; uncertainty estimates and seed consistency are more informative here than a single small-sample p-value.
- This experiment removes feature groups and does not establish causal importance for individual variables.
- A feature useful for future-BIS prediction is not guaranteed to improve RL control.
- Any selected state representation needs separate closed-loop control validation in the propofol RL framework.
- The test split remains unused until candidate decisions are frozen.
- The existing test split may not be a pristine external holdout. Final publication claims need separate unseen cases or another pre-specified evaluation design.
"""


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _validate_dataset_reference(runs: Sequence[ValidatedRun], dataset_dir: Path) -> list[dict[str, Any]]:
    configured = {
        Path(str(run.config["dataset_dir"])).resolve().as_posix() for run in runs
    }
    requested = dataset_dir.as_posix()
    if configured != {requested}:
        raise ValueError(
            f"Configured modeling dataset {sorted(configured)} does not match --dataset-dir {requested}."
        )
    required = (
        "dataset_metadata.json",
        "preprocessing.pkl",
        "preprocessing_statistics.csv",
        "train_metadata.csv",
        "val_metadata.csv",
    )
    missing = [name for name in required if not (dataset_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Modeling dataset lacks validation-safe artifacts: {missing}")
    validation_metadata = pd.read_csv(dataset_dir / "val_metadata.csv")
    metadata_columns = {"case_id", "target_timestamp"}
    if not metadata_columns.issubset(validation_metadata.columns):
        raise ValueError("Validation metadata lacks case_id or target_timestamp.")
    canonical = runs[0].predictions
    sample_indices = canonical["sample_index"].to_numpy(dtype=np.int64)
    if (
        sample_indices.min(initial=0) < 0
        or sample_indices.max(initial=-1) >= len(validation_metadata)
    ):
        raise ValueError("Validation prediction sample indices exceed val_metadata.csv.")
    aligned_metadata = validation_metadata.iloc[sample_indices].reset_index(drop=True)
    for column in sorted(metadata_columns):
        if not np.array_equal(
            canonical[column].to_numpy(), aligned_metadata[column].to_numpy()
        ):
            raise ValueError(
                f"Validation predictions do not align with val_metadata.csv for {column}."
            )
    return [file_fingerprint(dataset_dir / name) for name in required]


def run_group_retraining_analysis(
    experiment_dir: Path,
    dataset_dir: Path,
    output_dir: Path | None = None,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate, aggregate, plot, and report the 40-run experiment."""

    experiment_dir = experiment_dir.resolve()
    dataset_dir = dataset_dir.resolve()
    output_dir = (output_dir or experiment_dir / "analysis").resolve()
    if output_dir != experiment_dir / "analysis":
        raise ValueError("Analysis output must be exactly <EXPERIMENT_DIR>/analysis.")

    runs = validate_experiment(experiment_dir)
    dataset_fingerprints = _validate_dataset_reference(runs, dataset_dir)
    run_level = build_run_level_summary(runs)
    condition_aggregate = aggregate_conditions(run_level)
    condition_seed_deltas, condition_contrasts = paired_condition_comparisons(run_level)
    model_seed_deltas, model_contrasts = paired_model_comparisons(run_level)
    patient = patient_level_metrics(runs)
    bootstrap = bootstrap_all_contrasts(
        patient, replicates=bootstrap_replicates, seed=bootstrap_seed
    )
    pareto = pareto_candidates(condition_aggregate)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures = save_figures(run_level, bootstrap, pareto, output_dir / "figures")
    inventory = run_level.loc[
        :,
        [
            "condition",
            "model",
            "seed",
            "run_dir",
            "git_commit",
            "device",
            "dynamic_feature_count",
        ],
    ].copy()
    inventory["status"] = "complete"
    inventory["test_evaluated"] = False

    tables = {
        "validated_run_inventory.csv": inventory,
        "validation_run_level_summary.csv": run_level,
        "validation_condition_aggregate.csv": condition_aggregate,
        "paired_condition_seed_deltas.csv": condition_seed_deltas,
        "paired_condition_contrasts.csv": condition_contrasts,
        "paired_model_seed_deltas.csv": model_seed_deltas,
        "paired_model_contrasts.csv": model_contrasts,
        "patient_level_metrics.csv": patient,
        "hierarchical_bootstrap_contrasts.csv": bootstrap,
        "pareto_candidates.csv": pareto,
    }
    for name, frame in tables.items():
        frame.to_csv(output_dir / name, index=False)

    training_commits = sorted(run_level["git_commit"].unique().tolist())
    report = build_markdown_report(
        condition_aggregate,
        condition_contrasts,
        model_contrasts,
        bootstrap,
        pareto,
        training_commits,
    )
    report_path = output_dir / "validation_analysis_report.md"
    report_path.write_text(report, encoding="utf-8")

    input_fingerprints = dataset_fingerprints
    for run in runs:
        for name in (
            "config.json",
            "run_status.json",
            "training_history.csv",
            "val_metrics.json",
            "val_predictions.csv",
        ):
            input_fingerprints.append(file_fingerprint(run.run_dir / name))
    generated = sorted(
        [str(path.relative_to(output_dir)) for path in figures]
        + list(tables)
        + ["analysis_manifest.json", "validation_analysis_report.md"]
    )
    manifest = {
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_git_commit": _git_commit(Path(__file__).resolve().parents[1]),
        "training_git_commits": training_commits,
        "experiment_directory": str(experiment_dir),
        "dataset_directory": str(dataset_dir),
        "test_split_sealed": True,
        "test_split_read_by_analysis": False,
        "run_count": len(runs),
        "conditions": list(CONDITIONS),
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_direction": "lower is better",
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_resampling_units": "paired seed and validation patient",
        "input_fingerprints": input_fingerprints,
        "generated_outputs": generated,
    }
    dump_json(manifest, output_dir / "analysis_manifest.json")
    LOGGER.info("Validation-only analysis written to %s", output_dir)
    return {
        "output_dir": str(output_dir),
        "run_count": len(runs),
        "test_split_sealed": True,
        "condition_aggregate": condition_aggregate.to_dict(orient="records"),
        "paired_condition_contrasts": condition_contrasts.to_dict(orient="records"),
        "paired_model_contrasts": model_contrasts.to_dict(orient="records"),
        "pareto_candidates": pareto.to_dict(orient="records"),
    }
