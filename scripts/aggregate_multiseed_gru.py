"""Validate and aggregate fixed-seed non-attention GRU baseline runs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import patient_level_evaluation, pooled_evaluation  # noqa: E402

LOGGER = logging.getLogger(__name__)
EXPECTED_SEEDS = (7, 21, 42, 84, 123)
SPLITS = ("val", "test")
REQUIRED_ARTIFACTS = (
    "config.json",
    "best_model.pt",
    "last_model.pt",
    "training_history.csv",
    "val_predictions.csv",
    "test_predictions.csv",
    "val_metrics.json",
    "test_metrics.json",
    "case_metrics.csv",
)
CONFIG_FIELDS_FIXED_ACROSS_SEEDS = (
    "dataset_dir",
    "device",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "max_epochs",
    "patience",
    "gradient_clip_norm",
    "hidden_size",
    "projection_size",
    "static_hidden_size",
    "prediction_hidden_size",
    "dropout",
    "case_balanced_sampling",
    "num_workers",
    "smoke",
    "resume_checkpoint",
    "resolved_device",
    "model_parameter_count",
    "dynamic_feature_names",
    "static_feature_names",
    "selected_training_cases",
    "selected_validation_cases",
)
COMPARISON_METRICS = (
    "pooled_mae",
    "pooled_rmse",
    "patient_mean_mae",
    "bis_below_40_mae",
    "bis_40_to_60_mae",
    "bis_above_60_mae",
    "high_bis_auprc",
    "high_bis_auroc",
    "low_bis_auprc",
    "low_bis_auroc",
)
ERROR_METRICS = frozenset(
    {
        "pooled_mae",
        "pooled_rmse",
        "patient_mean_mae",
        "bis_below_40_mae",
        "bis_40_to_60_mae",
        "bis_above_60_mae",
    }
)
PREDICTION_KEYS = ("sample_index", "case_id", "target_timestamp")


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from ``path``."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(payload: dict[str, Any], path: Path) -> None:
    """Write strict, human-readable JSON."""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")


def discover_complete_seeds(gru_dir: Path) -> tuple[list[int], dict[int, list[str]]]:
    """Return complete numeric seed directories and missing artifacts by seed."""

    complete: list[int] = []
    incomplete: dict[int, list[str]] = {}
    for path in sorted(gru_dir.glob("seed_*")):
        try:
            seed = int(path.name.removeprefix("seed_"))
        except ValueError:
            continue
        missing = [name for name in REQUIRED_ARTIFACTS if not (path / name).is_file()]
        if missing:
            incomplete[seed] = missing
        else:
            complete.append(seed)
    return sorted(complete), incomplete


def summarize_numeric(values: Iterable[float]) -> dict[str, float | int]:
    """Summarize finite values using sample standard deviation."""

    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Metric aggregation requires a non-empty finite 1D sequence.")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def align_prediction_rows(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    split: str,
    candidate_name: str,
) -> None:
    """Require identical ordered rows, targets, and labels across predictions."""

    required = set(PREDICTION_KEYS) | {
        "observed_future_bis",
        "predicted_future_bis",
        "high_bis_label",
        "low_bis_label",
    }
    for name, frame in (("reference", reference), (candidate_name, candidate)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{split} {name} predictions are missing columns: {missing}")
    if len(reference) != len(candidate):
        raise ValueError(
            f"{split} {candidate_name} prediction row count {len(candidate)} does not "
            f"match reference count {len(reference)}."
        )
    if not reference[list(PREDICTION_KEYS)].equals(candidate[list(PREDICTION_KEYS)]):
        raise ValueError(
            f"{split} {candidate_name} prediction keys are missing, reordered, or inconsistent."
        )
    for column in ("observed_future_bis", "high_bis_label", "low_bis_label"):
        if not np.allclose(reference[column], candidate[column], equal_nan=False):
            raise ValueError(
                f"{split} {candidate_name} column {column!r} does not match the reference."
            )


def verify_metadata_alignment(frame: pd.DataFrame, metadata: pd.DataFrame, split: str) -> None:
    """Verify prediction sample indices resolve to matching dataset metadata rows."""

    indices = frame["sample_index"].to_numpy(dtype=int)
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{split} prediction sample indices are duplicated.")
    if len(indices) != len(metadata) or indices.min() < 0 or indices.max() >= len(metadata):
        raise ValueError(f"{split} prediction sample indices do not cover metadata exactly.")
    aligned = metadata.iloc[indices]
    for column in ("case_id", "target_timestamp"):
        if not np.array_equal(
            frame[column].to_numpy(dtype=int), aligned[column].to_numpy(dtype=int)
        ):
            raise ValueError(f"{split} prediction {column} is not aligned with metadata.")


def metric_values(frame: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    """Calculate all requested pooled and patient-level metrics from predictions."""

    observed = frame["observed_future_bis"].to_numpy(dtype=float)
    predicted = frame["predicted_future_bis"].to_numpy(dtype=float)
    cases = frame["case_id"].to_numpy(dtype=int)
    pooled = pooled_evaluation(observed, predicted)
    patient = patient_level_evaluation(observed, predicted, cases)
    values = {
        "pooled_mae": float(pooled["regression"]["mae"]),
        "pooled_rmse": float(pooled["regression"]["rmse"]),
        "r_squared": float(pooled["regression"]["r_squared"]),
        "patient_mean_mae": float(patient.summary["mae"]["mean"]),
        "patient_median_mae": float(patient.summary["mae"]["median"]),
        "patient_mean_rmse": float(patient.summary["rmse"]["mean"]),
        "bis_below_40_mae": float(pooled["bis_region_mae"]["bis_below_40"]),
        "bis_40_to_60_mae": float(pooled["bis_region_mae"]["bis_40_to_60"]),
        "bis_above_60_mae": float(pooled["bis_region_mae"]["bis_above_60"]),
        "high_bis_auprc": float(pooled["high_bis_classification"]["auprc"]),
        "high_bis_auroc": float(pooled["high_bis_classification"]["auroc"]),
        "low_bis_auprc": float(pooled["low_bis_classification"]["auprc"]),
        "low_bis_auroc": float(pooled["low_bis_classification"]["auroc"]),
    }
    return values, patient.case_metrics


def prediction_distribution(frame: pd.DataFrame) -> dict[str, float]:
    """Summarize prediction spread and agreement with observed future BIS."""

    observed = frame["observed_future_bis"].to_numpy(dtype=float)
    predicted = frame["predicted_future_bis"].to_numpy(dtype=float)
    return {
        "prediction_mean": float(predicted.mean()),
        "prediction_standard_deviation": float(predicted.std(ddof=0)),
        "prediction_minimum": float(predicted.min()),
        "prediction_maximum": float(predicted.max()),
        "observed_predicted_pearson_correlation": float(
            np.corrcoef(observed, predicted)[0, 1]
        ),
    }


def build_case_rows(
    gru: pd.DataFrame, persistence: pd.DataFrame, *, seed: int, split: str
) -> list[dict[str, Any]]:
    """Build one patient-level comparison row for a seed and split."""

    gru_values, gru_cases = metric_values(gru)
    del gru_values
    _, persistence_cases = metric_values(persistence)
    if set(gru_cases["case_id"]) != set(persistence_cases["case_id"]):
        raise ValueError(f"{split} seed {seed} patient sets do not align with persistence.")
    persistence_mae = persistence_cases.set_index("case_id")["mae"]
    rows: list[dict[str, Any]] = []
    for case_id, case in gru.groupby("case_id", sort=True):
        patient_metric = gru_cases.loc[gru_cases["case_id"] == case_id]
        if len(patient_metric) != 1:
            raise ValueError(f"{split} seed {seed} has inconsistent metrics for case {case_id}.")
        metric = patient_metric.iloc[0]
        p_mae = float(persistence_mae.loc[case_id])
        rows.append(
            {
                "split": split,
                "case_id": int(case_id),
                "seed": int(seed),
                "gru_mae": float(metric["mae"]),
                "persistence_mae": p_mae,
                "gru_minus_persistence_mae": float(metric["mae"] - p_mae),
                "gru_rmse": float(metric["rmse"]),
                "high_bis_prevalence": float(case["high_bis_label"].mean()),
                "low_bis_prevalence": float(case["low_bis_label"].mean()),
                "number_of_windows": int(len(case)),
            }
        )
    return rows


def validate_patient_seed_alignment(case_frame: pd.DataFrame, seeds: Iterable[int]) -> None:
    """Require every split/patient pair to have exactly one row per requested seed."""

    expected = set(int(seed) for seed in seeds)
    for (split, case_id), group in case_frame.groupby(["split", "case_id"]):
        found = set(group["seed"].astype(int))
        if len(group) != len(expected) or found != expected:
            raise ValueError(
                f"{split} case {case_id} has seeds {sorted(found)}; expected {sorted(expected)}."
            )
        if group["persistence_mae"].nunique() != 1:
            raise ValueError(f"{split} case {case_id} has inconsistent persistence MAE.")
        if group["number_of_windows"].nunique() != 1:
            raise ValueError(f"{split} case {case_id} has inconsistent window counts.")


def patient_bootstrap_mean_ci(
    patient_differences: Iterable[float],
    *,
    bootstrap_seed: int = 20260714,
    replicates: int = 20_000,
) -> dict[str, float | int | str]:
    """Bootstrap the mean by resampling one seed-averaged value per patient."""

    values = np.asarray(list(patient_differences), dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Patient bootstrap requires at least two finite patient values.")
    if replicates < 10_000:
        raise ValueError("At least 10,000 patient bootstrap replicates are required.")
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    bootstrap_means = values[indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return {
        "patient_count": int(len(values)),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(bootstrap_seed),
        "point_estimate": float(values.mean()),
        "percentile_95_ci_lower": float(lower),
        "percentile_95_ci_upper": float(upper),
        "resampling_unit": "patient",
        "interpretation": "uncertainty estimate, not proof of clinical superiority",
    }


def classify_multiseed_result(
    mean_difference: float,
    seed_improvement_count: int,
    patient_improvement_count: int,
    patient_count: int,
) -> tuple[str, str]:
    """Apply the prespecified descriptive multi-seed result categories."""

    distributed = patient_improvement_count >= max(3, (patient_count + 1) // 2)
    if (
        mean_difference <= -0.2
        and seed_improvement_count >= 4
        and patient_improvement_count >= 10
    ):
        return "ROBUST IMPROVEMENT", "Meets all prespecified robust-improvement criteria."
    if -0.2 < mean_difference < 0 and seed_improvement_count >= 4 and distributed:
        return (
            "SMALL BUT CONSISTENT IMPROVEMENT",
            "Mean improvement is below 0.2 BIS points but recurs across seeds and patients.",
        )
    if mean_difference > 0 and seed_improvement_count <= 2:
        return "UNDERPERFORMANCE", "Mean performance is worse and most seeds do not improve."
    return "UNSTABLE", "Improvement does not meet the cross-seed and patient-distribution rules."


def parse_runtime_seconds(value: str, seeds: Iterable[int]) -> dict[int, float]:
    """Parse ``seed=seconds`` comma-separated runtime measurements."""

    runtimes: dict[int, float] = {}
    try:
        for item in value.split(","):
            seed_text, seconds_text = item.split("=", maxsplit=1)
            runtimes[int(seed_text)] = float(seconds_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Runtime must be comma-separated seed=seconds pairs."
        ) from exc
    missing = sorted(set(seeds) - set(runtimes))
    if missing or any(seconds <= 0 for seconds in runtimes.values()):
        raise argparse.ArgumentTypeError(
            f"Positive runtime measurements are required for all seeds; missing {missing}."
        )
    return runtimes


def _across_seed_summary(rows: pd.DataFrame, excluded: set[str]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for split, split_frame in rows.groupby("split", sort=False):
        summaries[str(split)] = {
            column: summarize_numeric(split_frame[column])
            for column in split_frame.select_dtypes(include="number").columns
            if column not in excluded
        }
    return summaries


def _test_patient_stability(case_frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    test = case_frame.loc[case_frame["split"] == "test"]
    patient = (
        test.groupby("case_id", sort=True)
        .agg(
            mean_gru_mae=("gru_mae", "mean"),
            standard_deviation_gru_mae=("gru_mae", "std"),
            persistence_mae=("persistence_mae", "first"),
            mean_gru_minus_persistence_mae=("gru_minus_persistence_mae", "mean"),
            seeds_gru_beats_persistence=(
                "gru_minus_persistence_mae",
                lambda values: int((values < 0).sum()),
            ),
            minimum_difference=("gru_minus_persistence_mae", "min"),
            maximum_difference=("gru_minus_persistence_mae", "max"),
        )
        .reset_index()
    )
    patient["difference_sign_changes_across_seeds"] = (
        (patient["minimum_difference"] < 0) & (patient["maximum_difference"] > 0)
    )
    patient["highly_seed_sensitive"] = (
        (patient["standard_deviation_gru_mae"] > 0.2)
        | patient["difference_sign_changes_across_seeds"]
    )
    differences = patient["mean_gru_minus_persistence_mae"]
    improvements = patient.nsmallest(5, "mean_gru_minus_persistence_mae")
    deteriorations = patient.nlargest(5, "mean_gru_minus_persistence_mae")
    improved = int((differences < 0).sum())
    columns = ["case_id", "mean_gru_minus_persistence_mae"]
    summary = {
        "patient_count": int(len(patient)),
        "patients_seed_averaged_gru_mae_lower": improved,
        "percentage_patients_seed_averaged_gru_mae_lower": float(100 * improved / len(patient)),
        "median_patient_difference_gru_minus_persistence": float(differences.median()),
        "five_largest_improvements": improvements[columns].to_dict(orient="records"),
        "five_largest_deteriorations": deteriorations[columns].to_dict(orient="records"),
        "highly_seed_sensitive_definition": (
            "sample SD of GRU patient MAE > 0.2 BIS points, or difference sign changes across seeds"
        ),
        "highly_seed_sensitive_patients": patient.loc[
            patient["highly_seed_sensitive"],
            [
                "case_id",
                "standard_deviation_gru_mae",
                "difference_sign_changes_across_seeds",
            ],
        ].to_dict(orient="records"),
    }
    return summary, patient


def aggregate_multiseed(
    *,
    outputs_dir: Path,
    dataset_dir: Path,
    seeds: Iterable[int],
    runtime_seconds: dict[int, float],
    bootstrap_seed: int = 20260714,
    bootstrap_replicates: int = 20_000,
) -> dict[str, Any]:
    """Validate fixed GRU runs and write all requested multi-seed artifacts."""

    seeds = tuple(int(seed) for seed in seeds)
    gru_dir = outputs_dir / "gru"
    complete, incomplete = discover_complete_seeds(gru_dir)
    missing = sorted(set(seeds) - set(complete))
    if missing:
        details = {seed: incomplete.get(seed, list(REQUIRED_ARTIFACTS)) for seed in missing}
        raise FileNotFoundError(f"Requested GRU runs are incomplete: {details}")

    dataset_metadata = load_json(dataset_dir / "dataset_metadata.json")
    metadata = {
        split: pd.read_csv(dataset_dir / f"{split}_metadata.csv") for split in SPLITS
    }
    persistence = {
        split: pd.read_csv(outputs_dir / "persistence" / f"{split}_predictions.csv")
        for split in SPLITS
    }
    persistence_values = {split: metric_values(persistence[split])[0] for split in SPLITS}
    reference_config: dict[str, Any] | None = None
    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    integrity: dict[str, Any] = {}
    prediction_frames: dict[tuple[int, str], pd.DataFrame] = {}

    for seed in seeds:
        seed_dir = gru_dir / f"seed_{seed}"
        config = load_json(seed_dir / "config.json")
        if int(config["seed"]) != seed:
            raise ValueError(f"seed_{seed} config reports seed {config['seed']}.")
        if reference_config is None:
            reference_config = config
        else:
            inconsistent = [
                field
                for field in CONFIG_FIELDS_FIXED_ACROSS_SEEDS
                if config.get(field) != reference_config.get(field)
            ]
            if inconsistent:
                raise ValueError(f"seed {seed} fixed config differs in: {inconsistent}")
        if (
            config["dynamic_feature_names"] != dataset_metadata["dynamic_feature_names"]
            or config["static_feature_names"] != dataset_metadata["static_feature_names"]
        ):
            raise ValueError(f"seed {seed} feature ordering differs from dataset metadata.")

        history = pd.read_csv(seed_dir / "training_history.csv")
        best_checkpoint = torch.load(
            seed_dir / "best_model.pt", map_location="cpu", weights_only=False
        )
        last_checkpoint = torch.load(
            seed_dir / "last_model.pt", map_location="cpu", weights_only=False
        )
        best_epoch = int(best_checkpoint["epoch"])
        completed_epochs = int(len(history))
        minimum_epoch = int(
            history.loc[history["validation_patient_level_mae"].idxmin(), "epoch"]
        )
        if best_epoch != minimum_epoch:
            raise ValueError(
                f"seed {seed} best checkpoint epoch {best_epoch} is not validation "
                f"patient-MAE minimum epoch {minimum_epoch}."
            )
        if int(last_checkpoint["epoch"]) != int(history.iloc[-1]["epoch"]):
            raise ValueError(f"seed {seed} last checkpoint and history epochs disagree.")

        split_integrity: dict[str, Any] = {}
        for split in SPLITS:
            frame = pd.read_csv(seed_dir / f"{split}_predictions.csv")
            align_prediction_rows(
                persistence[split], frame, split=split, candidate_name=f"seed {seed}"
            )
            verify_metadata_alignment(frame, metadata[split], split)
            if not np.isfinite(frame.select_dtypes(include="number")).all().all():
                raise ValueError(f"{split} seed {seed} contains non-finite predictions.")
            case_count = int(frame["case_id"].nunique())
            if case_count != 15:
                raise ValueError(f"{split} seed {seed} contains {case_count} cases, not 15.")
            prediction_frames[(seed, split)] = frame
            values, _ = metric_values(frame)
            metrics_json = load_json(seed_dir / f"{split}_metrics.json")
            threshold_high = float(
                metrics_json["thresholds_selected_on_validation"]["high_bis_score"]
            )
            threshold_low = float(
                metrics_json["thresholds_selected_on_validation"]["low_bis_score"]
            )
            threshold_metrics = metrics_json["pooled_window"]
            row = {
                "seed": seed,
                "split": split,
                "runtime_seconds": float(runtime_seconds[seed]),
                "device": config["resolved_device"],
                "model_parameter_count": int(config["model_parameter_count"]),
                "best_epoch": best_epoch,
                "completed_epochs": completed_epochs,
                "early_stopping_epoch": int(last_checkpoint["epoch"]),
                "best_validation_patient_mae": float(
                    history.loc[history["epoch"] == best_epoch, "validation_patient_level_mae"].iloc[0]
                ),
                **values,
                "high_bis_validation_selected_threshold": threshold_high,
                "high_bis_f1": float(threshold_metrics["high_bis_threshold_metrics"]["f1"]),
                "high_bis_sensitivity": float(
                    threshold_metrics["high_bis_threshold_metrics"]["sensitivity"]
                ),
                "high_bis_specificity": float(
                    threshold_metrics["high_bis_threshold_metrics"]["specificity"]
                ),
                "low_bis_validation_selected_threshold": threshold_low,
                "low_bis_f1": float(threshold_metrics["low_bis_threshold_metrics"]["f1"]),
                "low_bis_sensitivity": float(
                    threshold_metrics["low_bis_threshold_metrics"]["sensitivity"]
                ),
                "low_bis_specificity": float(
                    threshold_metrics["low_bis_threshold_metrics"]["specificity"]
                ),
                **prediction_distribution(frame),
            }
            summary_rows.append(row)
            comparison_rows.append(
                {
                    "seed": seed,
                    "split": split,
                    **{
                        f"{metric}_difference_gru_minus_persistence": (
                            values[metric] - persistence_values[split][metric]
                        )
                        for metric in COMPARISON_METRICS
                    },
                }
            )
            case_rows.extend(build_case_rows(frame, persistence[split], seed=seed, split=split))
            split_integrity[split] = {
                "prediction_count": int(len(frame)),
                "case_count": case_count,
                "all_predictions_finite": True,
                "rows_align_with_metadata_and_persistence": True,
            }

        test_metrics = load_json(seed_dir / "test_metrics.json")
        if not bool(test_metrics.get("checkpoint_reload_predictions_identical")):
            raise ValueError(f"seed {seed} did not pass checkpoint reload verification.")
        test_cases = set(prediction_frames[(seed, "test")]["case_id"].astype(int))
        if not {97, 154}.issubset(test_cases):
            raise ValueError(f"seed {seed} test predictions omit case 97 or 154.")
        integrity[str(seed)] = {
            "all_required_artifacts_present": True,
            "fixed_configuration_matches": True,
            "feature_order_matches_dataset_metadata": True,
            "best_checkpoint_selected_by_validation_patient_mae_only": True,
            "test_not_used_for_checkpoint_selection": True,
            "checkpoint_reload_predictions_identical": True,
            "cases_97_and_154_included": True,
            "splits": split_integrity,
        }

    summary_frame = pd.DataFrame(summary_rows)
    comparison_frame = pd.DataFrame(comparison_rows)
    case_frame = pd.DataFrame(case_rows).sort_values(["split", "case_id", "seed"])
    validate_patient_seed_alignment(case_frame, seeds)

    summary_path = gru_dir / "multiseed_summary.csv"
    comparison_path = gru_dir / "multiseed_persistence_comparison.csv"
    case_path = gru_dir / "multiseed_case_metrics.csv"
    summary_frame.to_csv(summary_path, index=False)
    comparison_frame.to_csv(comparison_path, index=False)
    case_frame.to_csv(case_path, index=False)

    across_comparison: dict[str, Any] = {}
    for split, split_frame in comparison_frame.groupby("split", sort=False):
        across_comparison[str(split)] = {}
        for metric in COMPARISON_METRICS:
            column = f"{metric}_difference_gru_minus_persistence"
            differences = split_frame[column]
            improves = differences < 0 if metric in ERROR_METRICS else differences > 0
            best_index = differences.idxmin() if metric in ERROR_METRICS else differences.idxmax()
            worst_index = differences.idxmax() if metric in ERROR_METRICS else differences.idxmin()
            across_comparison[str(split)][metric] = {
                **summarize_numeric(differences),
                "direction_favoring_gru": "negative" if metric in ERROR_METRICS else "positive",
                "seeds_gru_improves": int(improves.sum()),
                "best_seed": int(comparison_frame.loc[best_index, "seed"]),
                "worst_seed": int(comparison_frame.loc[worst_index, "seed"]),
            }

    patient_stability, test_patient_frame = _test_patient_stability(case_frame)
    bootstrap = patient_bootstrap_mean_ci(
        test_patient_frame["mean_gru_minus_persistence_mae"],
        bootstrap_seed=bootstrap_seed,
        replicates=bootstrap_replicates,
    )
    test_comparison = comparison_frame.loc[comparison_frame["split"] == "test"]
    high_mae_difference = test_comparison["bis_above_60_mae_difference_gru_minus_persistence"]
    high_auprc_difference = test_comparison["high_bis_auprc_difference_gru_minus_persistence"]
    high_seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        frame = prediction_frames[(seed, "test")]
        high = frame.loc[frame["observed_future_bis"] > 60]
        high_seed_rows.append(
            {
                "seed": seed,
                "observed_mean": float(high["observed_future_bis"].mean()),
                "predicted_mean": float(high["predicted_future_bis"].mean()),
                "prediction_bias_predicted_minus_observed": float(
                    (high["predicted_future_bis"] - high["observed_future_bis"]).mean()
                ),
            }
        )
    high_seed_frame = pd.DataFrame(high_seed_rows)
    high_diagnostic = {
        "seeds_high_bis_mae_better_than_persistence": int((high_mae_difference < 0).sum()),
        "high_bis_mae_difference": summarize_numeric(high_mae_difference),
        "seeds_high_bis_auprc_better_than_persistence": int((high_auprc_difference > 0).sum()),
        "high_bis_auprc_difference": summarize_numeric(high_auprc_difference),
        "per_seed_subset_means_and_bias": high_seed_rows,
        "prediction_bias_predicted_minus_observed": summarize_numeric(
            high_seed_frame["prediction_bias_predicted_minus_observed"]
        ),
        "observed_bis_mean": summarize_numeric(high_seed_frame["observed_mean"]),
        "predicted_bis_mean": summarize_numeric(high_seed_frame["predicted_mean"]),
        "systematically_underpredicted": bool(
            (high_seed_frame["prediction_bias_predicted_minus_observed"] < 0).all()
        ),
    }

    missing_remifentanil: dict[str, Any] = {}
    for case_id in (97, 154):
        per_seed: list[dict[str, Any]] = []
        mask_checks: list[bool] = []
        finite_checks: list[bool] = []
        for seed in seeds:
            frame = prediction_frames[(seed, "test")]
            case = frame.loc[frame["case_id"] == case_id]
            persistence_case = persistence["test"].loc[
                persistence["test"]["case_id"] == case_id
            ]
            diagnostic = load_json(gru_dir / f"seed_{seed}" / "test_metrics.json")[
                "entirely_missing_remifentanil_case_diagnostics"
            ][str(case_id)]
            mask_checks.append(bool(diagnostic["all_remifentanil_observation_masks_zero"]))
            finite_checks.append(bool(diagnostic["all_predictions_finite"]))
            gru_mae = float(case["absolute_error"].mean())
            persistence_mae = float(persistence_case["absolute_error"].mean())
            per_seed.append(
                {
                    "seed": seed,
                    "gru_mae": gru_mae,
                    "persistence_mae": persistence_mae,
                    "gru_minus_persistence_mae": gru_mae - persistence_mae,
                    "prediction_mean": float(case["predicted_future_bis"].mean()),
                    "prediction_standard_deviation": float(
                        case["predicted_future_bis"].std(ddof=0)
                    ),
                }
            )
        case_seed = pd.DataFrame(per_seed)
        all_predictions = pd.concat(
            [
                prediction_frames[(seed, "test")].loc[
                    prediction_frames[(seed, "test")]["case_id"] == case_id,
                    "predicted_future_bis",
                ]
                for seed in seeds
            ],
            ignore_index=True,
        )
        missing_remifentanil[str(case_id)] = {
            "per_seed": per_seed,
            "gru_mae": summarize_numeric(case_seed["gru_mae"]),
            "persistence_mae": float(case_seed["persistence_mae"].iloc[0]),
            "gru_minus_persistence_mae": summarize_numeric(
                case_seed["gru_minus_persistence_mae"]
            ),
            "seeds_gru_improves": int((case_seed["gru_minus_persistence_mae"] < 0).sum()),
            "prediction_mean_across_seeds": summarize_numeric(case_seed["prediction_mean"]),
            "pooled_prediction_standard_deviation": float(all_predictions.std(ddof=0)),
            "remifentanil_masks_zero_for_every_seed": bool(all(mask_checks)),
            "all_predictions_finite_for_every_seed": bool(all(finite_checks)),
        }

    test_patient_diff = test_comparison[
        "patient_mean_mae_difference_gru_minus_persistence"
    ]
    seed_improvements = int((test_patient_diff < 0).sum())
    patient_improvements = int(
        patient_stability["patients_seed_averaged_gru_mae_lower"]
    )
    category, description = classify_multiseed_result(
        float(test_patient_diff.mean()),
        seed_improvements,
        patient_improvements,
        int(patient_stability["patient_count"]),
    )
    result_classification = {
        "category": category,
        "description": description,
        "mean_test_patient_mae_difference_gru_minus_persistence": float(
            test_patient_diff.mean()
        ),
        "seeds_gru_improves_patient_mean_mae": seed_improvements,
        "patients_seed_averaged_gru_improves": patient_improvements,
        "operational_description_not_statistical_significance": True,
        "test_results_not_used_for_tuning": True,
    }

    summary_json = {
        "experiment": {
            "model": "existing non-attention GRU baseline",
            "seeds": list(seeds),
            "standard_deviation_definition": "sample standard deviation across seeds (ddof=1)",
            "runtime_seconds_by_seed": {str(seed): runtime_seconds[seed] for seed in seeds},
        },
        "integrity": integrity,
        "seed_split_rows": summary_frame.to_dict(orient="records"),
        "across_seed_by_split": _across_seed_summary(
            summary_frame, {"seed", "model_parameter_count", "early_stopping_epoch"}
        ),
        "test_patient_stability": patient_stability,
        "test_patient_seed_averages": test_patient_frame.to_dict(orient="records"),
        "patient_level_bootstrap": bootstrap,
        "high_bis_diagnostic": high_diagnostic,
        "missing_remifentanil_test_cases": missing_remifentanil,
        "result_classification": result_classification,
    }
    comparison_json = {
        "difference_definition": "GRU metric minus deterministic persistence metric",
        "seed_split_rows": comparison_frame.to_dict(orient="records"),
        "across_seed_by_split": across_comparison,
    }
    dump_json(summary_json, gru_dir / "multiseed_summary.json")
    dump_json(comparison_json, gru_dir / "multiseed_persistence_comparison.json")
    LOGGER.info("Wrote %s, %s, and %s", summary_path, comparison_path, case_path)
    return summary_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/modeling/full"))
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in EXPECTED_SEEDS),
        help="Comma-separated completed seed list.",
    )
    parser.add_argument(
        "--runtime-seconds",
        required=True,
        help="Measured runtimes as comma-separated seed=seconds pairs.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260714)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    seeds = tuple(int(seed) for seed in args.seeds.split(","))
    runtimes = parse_runtime_seconds(args.runtime_seconds, seeds)
    result = aggregate_multiseed(
        outputs_dir=args.outputs_dir,
        dataset_dir=args.dataset_dir,
        seeds=seeds,
        runtime_seconds=runtimes,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(json.dumps(result["result_classification"], indent=2))
    print(json.dumps(result["patient_level_bootstrap"], indent=2))
    print(json.dumps(result["high_bis_diagnostic"], indent=2))


if __name__ == "__main__":
    main()
