"""Patient-level Elastic Net stability selection for future-BIS features."""

from __future__ import annotations

import json
import logging
import subprocess
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold

from src.datasets import VitalBISDataset


LOGGER = logging.getLogger(__name__)
EXCLUDED_DUPLICATE_FEATURES = ("bis_target_error",)
FORCED_STATIC_FEATURES = ("age_years", "sex_male", "height_cm", "weight_kg")
SELECTION_THRESHOLDS = (0.80, 0.60, 0.40)


@dataclass(frozen=True)
class StabilitySelectionConfig:
    """Configuration for train-only patient bootstrap stability selection."""

    dataset_dir: Path
    output_dir: Path
    seed: int = 42
    bootstrap_count: int = 100
    cv_folds: int = 5
    l1_ratios: tuple[float, ...] = (0.1, 0.5, 0.9, 1.0)
    alphas: tuple[float, ...] = tuple(np.logspace(-4, 0, 9).tolist())
    coefficient_tolerance: float = 1e-6
    max_iter: int = 20_000
    optimization_tolerance: float = 1e-5
    smoke: bool = False


@dataclass(frozen=True)
class SelectionData:
    """Train-only matrix with dynamic lag groups and forced static covariates."""

    dynamic: np.ndarray
    static: np.ndarray
    target: np.ndarray
    case_ids: np.ndarray
    dynamic_feature_names: tuple[str, ...]
    static_feature_names: tuple[str, ...]
    lag_seconds: tuple[int, ...]


@dataclass(frozen=True)
class FittedElasticNet:
    """Elastic Net dynamic coefficients with unpenalized static coefficients."""

    dynamic_coefficients: np.ndarray
    static_coefficients_with_intercept: np.ndarray
    iterations: int
    convergence_warning: str | None

    def predict(self, dynamic: np.ndarray, static: np.ndarray) -> np.ndarray:
        static_design = np.column_stack((np.ones(len(static)), static))
        return (
            dynamic @ self.dynamic_coefficients
            + static_design @ self.static_coefficients_with_intercept
        )


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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


def load_train_selection_data(dataset_dir: Path) -> SelectionData:
    """Load train artifacts only and flatten six lags within each feature group."""

    dataset = VitalBISDataset(dataset_dir, "train")
    names = tuple(
        name
        for name in dataset.dynamic_feature_names
        if name not in EXCLUDED_DUPLICATE_FEATURES
    )
    if len(names) != 12:
        raise ValueError(
            "Canonical stability selection expects 12 dynamic groups after excluding "
            f"{list(EXCLUDED_DUPLICATE_FEATURES)}; observed {list(names)}."
        )
    if dataset.static_feature_names != FORCED_STATIC_FEATURES:
        raise ValueError(
            "Canonical forced static features do not match the dataset: "
            f"{list(dataset.static_feature_names)}"
        )
    indices = [dataset.dynamic_feature_names.index(name) for name in names]
    grouped = np.take(dataset.arrays["X_dynamic"], indices, axis=2).transpose(0, 2, 1)
    dynamic = grouped.reshape(len(dataset), -1).astype(np.float64, copy=False)
    static = dataset.arrays["X_static"].astype(np.float64, copy=False)
    target = dataset.arrays["y_bis"].astype(np.float64, copy=False)
    interval = int(dataset.dataset_metadata["resampling_interval_seconds"])
    steps = int(dataset.dataset_metadata["history_steps"])
    lag_seconds = tuple((steps - 1 - index) * interval for index in range(steps))
    if steps != 6:
        raise ValueError(f"Canonical stability selection expects 6 lags, observed {steps}.")
    return SelectionData(
        dynamic=dynamic,
        static=static,
        target=target,
        case_ids=dataset.case_ids.astype(np.int64, copy=False),
        dynamic_feature_names=names,
        static_feature_names=dataset.static_feature_names,
        lag_seconds=lag_seconds,
    )


def case_balanced_weights(
    case_ids: Sequence[int] | np.ndarray,
    multiplicities: dict[int, int] | None = None,
) -> np.ndarray:
    """Give each patient equal total weight, scaled by bootstrap multiplicity."""

    values = np.asarray(case_ids, dtype=np.int64)
    if values.size == 0:
        raise ValueError("Cannot weight an empty patient sample.")
    unique, counts = np.unique(values, return_counts=True)
    requested = multiplicities or {int(case_id): 1 for case_id in unique}
    unknown = sorted(set(requested) - set(unique.astype(int).tolist()))
    if unknown:
        raise ValueError(f"Bootstrap multiplicities include absent cases: {unknown}")
    weights = np.zeros(len(values), dtype=np.float64)
    for case_id, count in zip(unique, counts, strict=True):
        multiplier = int(requested.get(int(case_id), 0))
        if multiplier < 0:
            raise ValueError("Bootstrap multiplicities must be non-negative.")
        weights[values == case_id] = multiplier / int(count)
    if not np.any(weights > 0.0):
        raise ValueError("At least one patient must have positive weight.")
    return weights * (len(weights) / weights.sum())


def patient_bootstrap_multiplicities(
    case_ids: Sequence[int] | np.ndarray,
    rng: np.random.Generator,
) -> dict[int, int]:
    """Sample patients with replacement and return case-level multiplicities."""

    unique = np.unique(np.asarray(case_ids, dtype=np.int64))
    if not len(unique):
        raise ValueError("Cannot bootstrap an empty patient list.")
    sampled = rng.choice(unique, size=len(unique), replace=True)
    return {int(case_id): int(count) for case_id, count in Counter(sampled).items()}


def group_selected(
    coefficients: np.ndarray,
    coefficient_tolerance: float,
) -> np.ndarray:
    """Select a feature group when any of its lag coefficients exceeds tolerance."""

    if coefficients.ndim != 2:
        raise ValueError("Grouped coefficients must have shape [features, lags].")
    return np.any(np.abs(coefficients) > coefficient_tolerance, axis=1)


def _weighted_lstsq(
    design: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    root_weight = np.sqrt(weights)
    weighted_design = design * root_weight[:, None]
    weighted_target = target * (
        root_weight if target.ndim == 1 else root_weight[:, None]
    )
    return np.linalg.lstsq(weighted_design, weighted_target, rcond=None)[0]


def fit_weighted_elastic_net(
    dynamic: np.ndarray,
    static: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    optimization_tolerance: float,
    seed: int,
) -> FittedElasticNet:
    """Fit penalized dynamic lags while keeping static covariates unpenalized."""

    residual_dynamic, residual_target = _project_out_static(
        dynamic, static, target, weights
    )
    return _fit_projected_elastic_net(
        dynamic,
        static,
        target,
        weights,
        residual_dynamic,
        residual_target,
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        optimization_tolerance=optimization_tolerance,
        seed=seed,
    )


def _project_out_static(
    dynamic: np.ndarray,
    static: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project dynamic lags and target off the unpenalized static design."""

    static_design = np.column_stack((np.ones(len(static)), static))
    dynamic_on_static = _weighted_lstsq(static_design, dynamic, weights)
    target_on_static = _weighted_lstsq(static_design, target, weights)
    return (
        dynamic - static_design @ dynamic_on_static,
        target - static_design @ target_on_static,
    )


def _fit_projected_elastic_net(
    dynamic: np.ndarray,
    static: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    residual_dynamic: np.ndarray,
    residual_target: np.ndarray,
    *,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    optimization_tolerance: float,
    seed: int,
) -> FittedElasticNet:
    """Fit one grid point using a fold's cached static projection."""

    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=False,
        max_iter=max_iter,
        tol=optimization_tolerance,
        selection="cyclic",
        random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(residual_dynamic, residual_target, sample_weight=weights)
    warning_messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    dynamic_coefficients = model.coef_.astype(np.float64, copy=True)
    static_design = np.column_stack((np.ones(len(static)), static))
    static_coefficients = _weighted_lstsq(
        static_design,
        target - dynamic @ dynamic_coefficients,
        weights,
    )
    return FittedElasticNet(
        dynamic_coefficients=dynamic_coefficients,
        static_coefficients_with_intercept=static_coefficients,
        iterations=int(model.n_iter_),
        convergence_warning=" | ".join(warning_messages) or None,
    )


def _weighted_mse(observed: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(np.square(predicted - observed), weights=weights))


def select_hyperparameters(
    data: SelectionData,
    config: StabilitySelectionConfig,
) -> tuple[float, float, pd.DataFrame]:
    """Select alpha and l1_ratio using patient-level GroupKFold on train only."""

    unique_cases = np.unique(data.case_ids)
    fold_count = min(config.cv_folds, len(unique_cases))
    if fold_count < 2:
        raise ValueError("Patient-level CV requires at least two training cases.")
    splitter = GroupKFold(n_splits=fold_count)
    folds: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for train_indices, validation_indices in splitter.split(
        data.dynamic, data.target, groups=data.case_ids
    ):
        train_weights = case_balanced_weights(data.case_ids[train_indices])
        residual_dynamic, residual_target = _project_out_static(
            data.dynamic[train_indices],
            data.static[train_indices],
            data.target[train_indices],
            train_weights,
        )
        folds.append(
            (
                train_indices,
                validation_indices,
                train_weights,
                residual_dynamic,
                residual_target,
            )
        )
    rows: list[dict[str, Any]] = []
    for l1_ratio in config.l1_ratios:
        for alpha in config.alphas:
            fold_losses: list[float] = []
            warning_count = 0
            max_iterations = 0
            for fold_index, (
                train_indices,
                validation_indices,
                train_weights,
                residual_dynamic,
                residual_target,
            ) in enumerate(
                folds, start=1
            ):
                fitted = _fit_projected_elastic_net(
                    data.dynamic[train_indices],
                    data.static[train_indices],
                    data.target[train_indices],
                    train_weights,
                    residual_dynamic,
                    residual_target,
                    alpha=float(alpha),
                    l1_ratio=float(l1_ratio),
                    max_iter=config.max_iter,
                    optimization_tolerance=config.optimization_tolerance,
                    seed=config.seed + fold_index,
                )
                validation_weights = case_balanced_weights(data.case_ids[validation_indices])
                predicted = fitted.predict(
                    data.dynamic[validation_indices], data.static[validation_indices]
                )
                fold_losses.append(
                    _weighted_mse(
                        data.target[validation_indices], predicted, validation_weights
                    )
                )
                warning_count += int(fitted.convergence_warning is not None)
                max_iterations = max(max_iterations, fitted.iterations)
            row: dict[str, Any] = {
                "l1_ratio": float(l1_ratio),
                "alpha": float(alpha),
                "mean_case_balanced_mse": float(np.mean(fold_losses)),
                "standard_deviation_case_balanced_mse": float(np.std(fold_losses)),
                "fold_count": fold_count,
                "convergence_warning_count": warning_count,
                "maximum_iterations": max_iterations,
            }
            row.update(
                {f"fold_{index}_case_balanced_mse": loss for index, loss in enumerate(fold_losses, 1)}
            )
            rows.append(row)
    results = pd.DataFrame(rows).sort_values(
        ["mean_case_balanced_mse", "alpha", "l1_ratio"], kind="stable"
    ).reset_index(drop=True)
    best = results.iloc[0]
    return float(best["alpha"]), float(best["l1_ratio"]), results


def _rank_groups(group_scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-group_scores, kind="stable")
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def _threshold_payload(
    threshold: float,
    summary: pd.DataFrame,
    dynamic_feature_order: tuple[str, ...],
    static_features: tuple[str, ...],
) -> dict[str, Any]:
    selected = summary.loc[
        summary["selection_frequency"] >= threshold, "feature_name"
    ].tolist()
    if "bis" not in selected:
        selected.insert(0, "bis")
    ordered = [name for name in dynamic_feature_order if name in set(selected)]
    return {
        "schema_version": 1,
        "scientific_role": "candidate_subset_before_validation_group_ablation",
        "selection_method": "patient_level_elastic_net_stability_selection",
        "selection_frequency_threshold": threshold,
        "dynamic_features": ordered,
        "forced_dynamic_anchor_features": ["bis"],
        "forced_static_features": list(static_features),
        "excluded_duplicate_features": list(EXCLUDED_DUPLICATE_FEATURES),
        "final_selected_ppo_manifest": False,
        "test_used": False,
    }


def _run_bootstraps(
    data: SelectionData,
    config: StabilitySelectionConfig,
    alpha: float,
    l1_ratio: float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(config.seed)
    feature_count = len(data.dynamic_feature_names)
    lag_count = len(data.lag_seconds)
    coefficients = np.zeros((config.bootstrap_count, feature_count, lag_count))
    selections = np.zeros((config.bootstrap_count, feature_count), dtype=bool)
    rows: list[dict[str, Any]] = []
    for bootstrap_index in range(config.bootstrap_count):
        multiplicities = patient_bootstrap_multiplicities(data.case_ids, rng)
        included = np.isin(data.case_ids, list(multiplicities))
        weights = case_balanced_weights(data.case_ids[included], multiplicities)
        fitted = fit_weighted_elastic_net(
            data.dynamic[included],
            data.static[included],
            data.target[included],
            weights,
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=config.max_iter,
            optimization_tolerance=config.optimization_tolerance,
            seed=config.seed + bootstrap_index + 10_000,
        )
        grouped = fitted.dynamic_coefficients.reshape(feature_count, lag_count)
        selected = group_selected(grouped, config.coefficient_tolerance)
        coefficients[bootstrap_index] = grouped
        selections[bootstrap_index] = selected
        rows.append(
            {
                "bootstrap_index": bootstrap_index,
                "drawn_patient_count": int(sum(multiplicities.values())),
                "unique_patient_count": len(multiplicities),
                "selected_feature_group_count": int(selected.sum()),
                "iterations": fitted.iterations,
                "convergence_warning": fitted.convergence_warning,
                "patient_multiplicities": json.dumps(multiplicities, sort_keys=True),
            }
        )
    return coefficients, selections, pd.DataFrame(rows)


def _summaries(
    data: SelectionData,
    coefficients: np.ndarray,
    selections: np.ndarray,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranks = np.stack(
        [_rank_groups(np.mean(np.abs(replicate), axis=1)) for replicate in coefficients]
    )
    summary_rows: list[dict[str, Any]] = []
    lag_rows: list[dict[str, Any]] = []
    for feature_index, feature_name in enumerate(data.dynamic_feature_names):
        values = coefficients[:, feature_index, :]
        nonzero = values[np.abs(values) > tolerance]
        mean_signed = float(np.mean(values))
        dominant_sign = "positive" if mean_signed > 0 else "negative" if mean_signed < 0 else "zero"
        dominant_value = 1.0 if dominant_sign == "positive" else -1.0 if dominant_sign == "negative" else 0.0
        sign_consistency = (
            float(np.mean(np.sign(nonzero) == dominant_value)) if len(nonzero) else 0.0
        )
        summary_rows.append(
            {
                "feature_name": feature_name,
                "selection_frequency": float(np.mean(selections[:, feature_index])),
                "selected_count": int(selections[:, feature_index].sum()),
                "number_of_bootstraps": len(selections),
                "mean_absolute_coefficient_across_lags": float(np.mean(np.abs(values))),
                "median_absolute_coefficient_across_lags": float(np.median(np.abs(values))),
                "maximum_mean_absolute_lag_coefficient": float(
                    np.max(np.mean(np.abs(values), axis=0))
                ),
                "dominant_sign": dominant_sign,
                "sign_consistency": sign_consistency,
                "mean_rank": float(np.mean(ranks[:, feature_index])),
                "standard_deviation_rank": float(np.std(ranks[:, feature_index])),
            }
        )
        for lag_index, lag_seconds in enumerate(data.lag_seconds):
            lag_values = values[:, lag_index]
            lag_rows.append(
                {
                    "feature_name": feature_name,
                    "lag_seconds": lag_seconds,
                    "mean_coefficient": float(np.mean(lag_values)),
                    "mean_absolute_coefficient": float(np.mean(np.abs(lag_values))),
                    "selection_frequency_at_lag": float(
                        np.mean(np.abs(lag_values) > tolerance)
                    ),
                }
            )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["selection_frequency", "mean_rank"], ascending=[False, True], kind="stable"
    )
    return summary.reset_index(drop=True), pd.DataFrame(lag_rows)


def run_elastic_net_stability(config: StabilitySelectionConfig) -> dict[str, Any]:
    """Run train-only CV and patient bootstrap stability selection."""

    if config.bootstrap_count <= 0 or config.cv_folds < 2:
        raise ValueError("bootstrap_count must be positive and cv_folds at least two.")
    if config.coefficient_tolerance < 0.0:
        raise ValueError("coefficient_tolerance must be non-negative.")
    if not config.l1_ratios or not config.alphas:
        raise ValueError("Hyperparameter grids must not be empty.")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    git_commit = _git_commit_hash()
    config_payload = {
        **{
            key: str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value
            for key, value in asdict(config).items()
        },
        "selection_split": "train_only",
        "test_used": False,
        "git_commit_hash": git_commit,
    }
    _save_json(config_payload, config.output_dir / "config.json")
    _save_json(
        {"status": "running", "selection_split": "train_only", "test_used": False},
        config.output_dir / "run_status.json",
    )
    try:
        data = load_train_selection_data(config.dataset_dir)
        alpha, l1_ratio, cv_results = select_hyperparameters(data, config)
        cv_results.to_csv(config.output_dir / "hyperparameter_cv_results.csv", index=False)
        coefficients, selections, bootstrap_runs = _run_bootstraps(
            data, config, alpha, l1_ratio
        )
        summary, lag_summary = _summaries(
            data, coefficients, selections, config.coefficient_tolerance
        )
        summary.to_csv(config.output_dir / "stability_summary.csv", index=False)
        lag_summary.to_csv(config.output_dir / "lag_coefficient_summary.csv", index=False)
        selection_frame = pd.DataFrame(
            selections.astype(np.int8), columns=data.dynamic_feature_names
        )
        selection_frame.insert(0, "bootstrap_index", np.arange(config.bootstrap_count))
        selection_frame.to_csv(
            config.output_dir / "bootstrap_selection_matrix.csv", index=False
        )
        bootstrap_runs.to_csv(config.output_dir / "bootstrap_run_summary.csv", index=False)
        for threshold in SELECTION_THRESHOLDS:
            suffix = f"{int(round(threshold * 100)):03d}"
            _save_json(
                _threshold_payload(
                    threshold,
                    summary,
                    data.dynamic_feature_names,
                    data.static_feature_names,
                ),
                config.output_dir / f"selected_frequency_{suffix}.json",
            )
        metadata = {
            "schema_version": 1,
            "scientific_role": "primary_feature_selection_before_validation_ablation",
            "selection_method": "patient_level_elastic_net_stability_selection",
            "attention_role": "preserved_auxiliary_analysis",
            "test_used": False,
            "selection_split": "train_only",
            "input_files_read": [
                "dataset_metadata.json",
                "train.npz",
                "train_metadata.csv",
            ],
            "bootstrap_unit": "patient",
            "feature_group_definition": {
                "dynamic_group_count": len(data.dynamic_feature_names),
                "lags_per_group": len(data.lag_seconds),
                "lag_seconds": list(data.lag_seconds),
                "selected_when": (
                    "any absolute lag coefficient exceeds coefficient_tolerance"
                ),
            },
            "excluded_duplicate_features": list(EXCLUDED_DUPLICATE_FEATURES),
            "forced_static_features": list(data.static_feature_names),
            "forced_static_penalty": "unpenalized weighted projection in every fit",
            "observation_mask_used_as_model_feature": False,
            "preprocessing": "arrays already transformed with train-only preprocessing.pkl",
            "selected_hyperparameters": {"alpha": alpha, "l1_ratio": l1_ratio},
            "seed": config.seed,
            "bootstrap_count": config.bootstrap_count,
            "coefficient_tolerance": config.coefficient_tolerance,
            "case_balancing_method": (
                "each patient has equal total sample weight; bootstrap draws multiply "
                "that patient's total weight"
            ),
            "validation_used": False,
            "final_selected_ppo_manifest_created": False,
            "git_commit_hash": git_commit,
        }
        _save_json(metadata, config.output_dir / "analysis_metadata.json")
        result = {
            "status": "complete",
            "test_used": False,
            "selection_split": "train_only",
            "selected_alpha": alpha,
            "selected_l1_ratio": l1_ratio,
            "bootstrap_count": config.bootstrap_count,
            "top_features": summary.head(5).loc[
                :, ["feature_name", "selection_frequency"]
            ].to_dict("records"),
            "convergence_warning_count": int(
                bootstrap_runs["convergence_warning"].notna().sum()
            ),
            "git_commit_hash": git_commit,
        }
        _save_json(result, config.output_dir / "run_status.json")
        LOGGER.info("Elastic Net stability selection completed with alpha=%g, l1_ratio=%g", alpha, l1_ratio)
        return result
    except Exception as error:
        _save_json(
            {
                "status": "failed",
                "selection_split": "train_only",
                "test_used": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "git_commit_hash": git_commit,
            },
            config.output_dir / "run_status.json",
        )
        raise
