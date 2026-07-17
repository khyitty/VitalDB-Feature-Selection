"""Legacy train-only selector for the physiological-inclusive feature universe."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import pickle
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from src.preprocessing import PreprocessingArtifact
from src.prediction_feature_profiles import SIMULATOR_COMPATIBLE_PROFILE
from src.redundancy_audit import FEATURE_GROUPS, REDUCED_FEATURES

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

LOGGER = logging.getLogger(__name__)

DEFAULT_RANDOM_SEED = 20260716
DEFAULT_STABILITY_ITERATIONS = 100
DEFAULT_STABILITY_THRESHOLDS = (0.6, 0.7, 0.8, 0.9)
DEFAULT_STABLE_THRESHOLD = 0.7
DEFAULT_SUBSAMPLE_FRACTION = 0.5
DEFAULT_CORRELATION_THRESHOLD = 0.8
MAX_FROZEN_CANDIDATES = 6
DETERMINISTIC_DUPLICATE = "bis_error"
REQUIRED_GROUP_ANALYSIS_FILES = (
    "analysis_manifest.json",
    "validation_condition_aggregate.csv",
    "hierarchical_bootstrap_contrasts.csv",
)
TRAIN_ONLY_FILES = (
    "train.npz",
    "train_metadata.csv",
    "feature_manifest.csv",
    "dataset_metadata.json",
    "preprocessing.pkl",
    "preprocessing_statistics.csv",
)

ELASTIC_GRID = (
    {"alpha": 0.001, "l1_ratio": 0.1},
    {"alpha": 0.001, "l1_ratio": 0.5},
    {"alpha": 0.001, "l1_ratio": 0.9},
    {"alpha": 0.01, "l1_ratio": 0.1},
    {"alpha": 0.01, "l1_ratio": 0.5},
    {"alpha": 0.01, "l1_ratio": 0.9},
    {"alpha": 0.1, "l1_ratio": 0.1},
    {"alpha": 0.1, "l1_ratio": 0.5},
    {"alpha": 0.1, "l1_ratio": 0.9},
)
TREE_GRID = (
    {"max_depth": 3, "min_child_weight": 5.0, "learning_rate": 0.05},
    {"max_depth": 5, "min_child_weight": 5.0, "learning_rate": 0.05},
    {"max_depth": 3, "min_child_weight": 10.0, "learning_rate": 0.1},
)


class Regressor(Protocol):
    """Minimal estimator protocol shared by XGBoost and synthetic test estimators."""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> Any: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


TreeFactory = Callable[[Mapping[str, Any], int, str], Regressor]


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration for train-only predictive feature selection."""

    dataset_dir: Path
    group_analysis_dir: Path
    output_dir: Path
    random_seed: int = DEFAULT_RANDOM_SEED
    internal_folds: int = 3
    stability_iterations: int = DEFAULT_STABILITY_ITERATIONS
    subsample_fraction: float = DEFAULT_SUBSAMPLE_FRACTION
    stable_threshold: float = DEFAULT_STABLE_THRESHOLD
    stability_thresholds: tuple[float, ...] = DEFAULT_STABILITY_THRESHOLDS
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD
    tree_permutation_repeats: int = 2
    tree_estimators: int = 150
    tree_device: str = "cpu"
    compute_shap: bool = False
    shap_max_windows: int = 5000
    protected_control_features: tuple[str, ...] = ()
    max_frozen_candidates: int = MAX_FROZEN_CANDIDATES


@dataclass(frozen=True)
class TrainSelectionData:
    """Train-only arrays and aligned patient metadata used by selectors."""

    X_dynamic: np.ndarray
    X_static: np.ndarray
    y_bis: np.ndarray
    patient_ids: np.ndarray
    target_timestamps: np.ndarray
    dynamic_features: tuple[str, ...]
    static_features: tuple[str, ...]
    time_lags_seconds: tuple[int, ...]
    feature_manifest: pd.DataFrame


@dataclass(frozen=True)
class DesignMatrix:
    """Flattened lag-aware design with dynamic groups and static covariates."""

    X: np.ndarray
    column_names: tuple[str, ...]
    dynamic_column_indices: dict[str, tuple[int, ...]]
    column_feature: tuple[str, ...]
    column_lag_seconds: tuple[int | None, ...]
    dynamic_column_count: int


def dump_json(payload: Mapping[str, Any], path: Path) -> None:
    """Write a strict JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON object with a path-specific error."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read valid JSON from {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def file_fingerprint(path: Path) -> dict[str, Any]:
    """Return the SHA-256 and size of one input artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


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


def _feature_group(feature: str) -> str:
    for group, members in FEATURE_GROUPS.items():
        if feature in members:
            return group
    return "ungrouped"


def _validate_config(config: SelectionConfig) -> None:
    if config.internal_folds < 2:
        raise ValueError("At least two patient-grouped internal folds are required.")
    if config.stability_iterations < 1:
        raise ValueError("Stability iterations must be positive.")
    if not 0.0 < config.subsample_fraction < 1.0:
        raise ValueError("Patient subsample fraction must be between zero and one.")
    if not 0.0 <= config.stable_threshold <= 1.0:
        raise ValueError("Stable selection threshold must be between zero and one.")
    if config.max_frozen_candidates < 1 or config.max_frozen_candidates > 6:
        raise ValueError("Frozen retraining candidates must be limited to at most six.")
    if config.tree_device not in {"cpu", "cuda"}:
        raise ValueError("tree_device must be 'cpu' or 'cuda'.")
    if config.tree_permutation_repeats < 1:
        raise ValueError("Tree permutation repeats must be positive.")


def load_train_selection_data(dataset_dir: Path) -> TrainSelectionData:
    """Load only train arrays and verify the deterministic BIS-error duplicate."""

    dataset_dir = dataset_dir.resolve()
    missing = [name for name in TRAIN_ONLY_FILES if not (dataset_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Train-only modeling artifacts are missing: {missing}")
    metadata = load_json(dataset_dir / "dataset_metadata.json")
    if metadata.get("feature_profile") == SIMULATOR_COMPATIBLE_PROFILE:
        raise ValueError(
            "This Elastic Net/XGBoost selector is legacy exploratory and cannot select "
            "the final simulator-compatible state. Use the new explicit-attention rerun."
        )
    dynamic_names = tuple(metadata["dynamic_feature_names"])
    static_names = tuple(metadata["static_feature_names"])
    if DETERMINISTIC_DUPLICATE not in dynamic_names:
        raise ValueError("dataset_metadata.json does not contain bis_error for redundancy audit.")
    selected_names = tuple(name for name in dynamic_names if name != DETERMINISTIC_DUPLICATE)
    if selected_names != REDUCED_FEATURES:
        raise ValueError(
            f"Expected exact full17 order {list(REDUCED_FEATURES)}, got {list(selected_names)}"
        )

    with np.load(dataset_dir / "train.npz", allow_pickle=False) as archive:
        required_arrays = {"X_dynamic", "X_static", "y_bis"}
        if not required_arrays.issubset(archive.files):
            raise ValueError(
                f"train.npz lacks arrays: {sorted(required_arrays - set(archive.files))}"
            )
        full_dynamic = archive["X_dynamic"].astype(np.float32, copy=False)
        X_static = archive["X_static"].astype(np.float32, copy=False)
        y_bis = archive["y_bis"].astype(np.float32, copy=False)
    train_metadata = pd.read_csv(dataset_dir / "train_metadata.csv")
    required_metadata = {"case_id", "target_timestamp"}
    if not required_metadata.issubset(train_metadata.columns):
        raise ValueError(
            f"train_metadata.csv lacks columns: {sorted(required_metadata - set(train_metadata.columns))}"
        )
    if not (len(full_dynamic) == len(X_static) == len(y_bis) == len(train_metadata)):
        raise ValueError("Train arrays and metadata row counts are not aligned.")
    if full_dynamic.shape[2] != len(dynamic_names):
        raise ValueError("Dynamic train tensor does not match dataset feature names.")
    if X_static.shape[1] != len(static_names):
        raise ValueError("Static train tensor does not match dataset feature names.")
    if not np.isfinite(full_dynamic).all() or not np.isfinite(X_static).all() or not np.isfinite(y_bis).all():
        raise ValueError("Train-only selector inputs contain non-finite values.")

    artifact = pickle.loads((dataset_dir / "preprocessing.pkl").read_bytes())
    if not isinstance(artifact, PreprocessingArtifact):
        raise ValueError("preprocessing.pkl is not a PreprocessingArtifact.")
    bis_index = dynamic_names.index("bis")
    error_index = dynamic_names.index(DETERMINISTIC_DUPLICATE)
    bis_stats = artifact.statistics["bis"]
    error_stats = artifact.statistics[DETERMINISTIC_DUPLICATE]
    bis_original = (
        full_dynamic[:, :, bis_index] * bis_stats.normalization_scale
        + bis_stats.training_mean
    )
    error_original = (
        full_dynamic[:, :, error_index] * error_stats.normalization_scale
        + error_stats.training_mean
    )
    if not np.allclose(error_original, bis_original - 50.0, atol=1e-4, rtol=0.0):
        raise ValueError("Train data do not verify the deterministic bis_error == bis - 50 relation.")

    keep_indices = [dynamic_names.index(name) for name in selected_names]
    history_steps = int(metadata["history_steps"])
    interval = int(metadata["resampling_interval_seconds"])
    lags = tuple(-(history_steps - 1 - index) * interval for index in range(history_steps))
    manifest = pd.read_csv(dataset_dir / "feature_manifest.csv")
    return TrainSelectionData(
        X_dynamic=full_dynamic[:, :, keep_indices],
        X_static=X_static,
        y_bis=y_bis,
        patient_ids=train_metadata["case_id"].to_numpy(dtype=np.int64),
        target_timestamps=train_metadata["target_timestamp"].to_numpy(dtype=np.int64),
        dynamic_features=selected_names,
        static_features=static_names,
        time_lags_seconds=lags,
        feature_manifest=manifest,
    )


def build_feature_inventory(data: TrainSelectionData) -> pd.DataFrame:
    """Describe dynamic candidates and static adjustment covariates separately."""

    manifest = data.feature_manifest.set_index("standardized_feature_name")
    rows = []
    for role, names in (
        ("dynamic_candidate", data.dynamic_features),
        ("static_adjustment_only", data.static_features),
    ):
        for name in names:
            source = manifest.loc[name] if name in manifest.index else {}
            rows.append(
                {
                    "feature": name,
                    "selection_role": role,
                    "feature_group": _feature_group(name),
                    "aggregation_rule": source.get("aggregation_rule"),
                    "missing_before_imputation_percent": source.get(
                        "percentage_missing_before_imputation"
                    ),
                    "eligible_for_predictive_selection": role == "dynamic_candidate",
                    "deterministic_duplicate_excluded": False,
                }
            )
    rows.append(
        {
            "feature": DETERMINISTIC_DUPLICATE,
            "selection_role": "excluded_deterministic_duplicate",
            "feature_group": "current_bis",
            "aggregation_rule": "derived_after_resampling",
            "missing_before_imputation_percent": 0.0,
            "eligible_for_predictive_selection": False,
            "deterministic_duplicate_excluded": True,
        }
    )
    return pd.DataFrame(rows)


def build_design_matrix(data: TrainSelectionData) -> DesignMatrix:
    """Flatten time lags while retaining dynamic-feature column groups."""

    windows, history_steps, feature_count = data.X_dynamic.shape
    dynamic_flat = data.X_dynamic.reshape(windows, history_steps * feature_count)
    X = np.concatenate((dynamic_flat, data.X_static), axis=1).astype(np.float32, copy=False)
    names: list[str] = []
    column_feature: list[str] = []
    column_lags: list[int | None] = []
    groups: dict[str, list[int]] = {feature: [] for feature in data.dynamic_features}
    for time_index, lag in enumerate(data.time_lags_seconds):
        for feature_index, feature in enumerate(data.dynamic_features):
            column_index = time_index * feature_count + feature_index
            names.append(f"{feature}@{lag}s")
            column_feature.append(feature)
            column_lags.append(lag)
            groups[feature].append(column_index)
    for static_name in data.static_features:
        names.append(f"static:{static_name}")
        column_feature.append(static_name)
        column_lags.append(None)
    return DesignMatrix(
        X=X,
        column_names=tuple(names),
        dynamic_column_indices={name: tuple(indices) for name, indices in groups.items()},
        column_feature=tuple(column_feature),
        column_lag_seconds=tuple(column_lags),
        dynamic_column_count=history_steps * feature_count,
    )


def patient_grouped_folds(
    patient_ids: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic folds with every patient's windows in exactly one fold."""

    patients = np.unique(np.asarray(patient_ids, dtype=np.int64))
    if n_splits < 2 or n_splits > len(patients):
        raise ValueError("Grouped fold count must be between two and the patient count.")
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(patients)
    validation_groups = np.array_split(shuffled, n_splits)
    folds = []
    for validation_patients in validation_groups:
        validation_mask = np.isin(patient_ids, validation_patients)
        train_indices = np.flatnonzero(~validation_mask)
        validation_indices = np.flatnonzero(validation_mask)
        if set(patient_ids[train_indices]) & set(patient_ids[validation_indices]):
            raise AssertionError("A patient appears in both grouped fold partitions.")
        folds.append((train_indices, validation_indices))
    return folds


def patient_subsample(
    patient_ids: np.ndarray,
    fraction: float,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample patients without replacement and return complete patient blocks."""

    patients = np.unique(np.asarray(patient_ids, dtype=np.int64))
    sample_count = max(1, min(len(patients) - 1, int(round(len(patients) * fraction))))
    selected_patients = np.sort(generator.choice(patients, size=sample_count, replace=False))
    selected_mask = np.isin(patient_ids, selected_patients)
    selected_indices = np.flatnonzero(selected_mask)
    heldout_indices = np.flatnonzero(~selected_mask)
    if set(patient_ids[selected_indices]) & set(patient_ids[heldout_indices]):
        raise AssertionError("Patient subsampling split a patient block.")
    return selected_indices, heldout_indices, selected_patients


def patient_balanced_weights(patient_ids: np.ndarray) -> np.ndarray:
    """Give every patient equal total fitting weight regardless of window count."""

    ids = np.asarray(patient_ids, dtype=np.int64)
    patients, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(float)
    return weights * (len(weights) / len(patients))


def patient_balanced_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: np.ndarray,
) -> float:
    """Average window MAE within patients and then equally across patients."""

    frame = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "absolute_error": np.abs(np.asarray(y_pred) - np.asarray(y_true)),
        }
    )
    return float(frame.groupby("patient_id")["absolute_error"].mean().mean())


def _fold_manifest_row(
    method: str,
    parameter_id: int,
    fold: int,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    patient_ids: np.ndarray,
    metric: float,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    train_patients = sorted(np.unique(patient_ids[train_indices]).astype(int).tolist())
    evaluation_patients = sorted(
        np.unique(patient_ids[evaluation_indices]).astype(int).tolist()
    )
    return {
        "method": method,
        "parameter_id": parameter_id,
        "fold": fold,
        "train_patient_count": len(train_patients),
        "evaluation_patient_count": len(evaluation_patients),
        "train_window_count": len(train_indices),
        "evaluation_window_count": len(evaluation_indices),
        "train_patient_ids": json.dumps(train_patients),
        "evaluation_patient_ids": json.dumps(evaluation_patients),
        "patient_overlap_count": len(set(train_patients) & set(evaluation_patients)),
        "patient_balanced_mae": metric,
        "parameters": json.dumps(dict(parameters), sort_keys=True),
        "selection_data_scope": "train split internal patient-grouped fold",
    }


def elastic_net_grouped_cv(
    data: TrainSelectionData,
    design: DesignMatrix,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    parameter_grid: Sequence[Mapping[str, float]] = ELASTIC_GRID,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Tune Elastic Net with fold-local scaling and patient-balanced validation MAE."""

    manifest_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    parameter_scores: list[float] = []
    for parameter_id, parameters in enumerate(parameter_grid):
        fold_scores = []
        for fold_id, (train_indices, evaluation_indices) in enumerate(folds):
            scaler = StandardScaler().fit(design.X[train_indices])
            X_train = scaler.transform(design.X[train_indices])
            X_evaluation = scaler.transform(design.X[evaluation_indices])
            model = ElasticNet(
                alpha=float(parameters["alpha"]),
                l1_ratio=float(parameters["l1_ratio"]),
                max_iter=10_000,
                selection="cyclic",
            )
            model.fit(
                X_train,
                data.y_bis[train_indices],
                sample_weight=patient_balanced_weights(data.patient_ids[train_indices]),
            )
            prediction = model.predict(X_evaluation)
            score = patient_balanced_mae(
                data.y_bis[evaluation_indices],
                prediction,
                data.patient_ids[evaluation_indices],
            )
            fold_scores.append(score)
            manifest_rows.append(
                _fold_manifest_row(
                    "elastic_net",
                    parameter_id,
                    fold_id,
                    train_indices,
                    evaluation_indices,
                    data.patient_ids,
                    score,
                    parameters,
                )
            )
            for column_index, coefficient in enumerate(model.coef_):
                coefficient_rows.append(
                    {
                        "stage": "grouped_cv_path",
                        "parameter_id": parameter_id,
                        "fold": fold_id,
                        "iteration": np.nan,
                        "column": design.column_names[column_index],
                        "feature": design.column_feature[column_index],
                        "lag_seconds": design.column_lag_seconds[column_index],
                        "dynamic_candidate": column_index < design.dynamic_column_count,
                        "standardized_coefficient": float(coefficient),
                        "selected_nonzero": bool(abs(coefficient) > 1e-10),
                    }
                )
        parameter_scores.append(float(np.mean(fold_scores)))
    best_id = int(np.argmin(parameter_scores))
    best_parameters = {
        "alpha": float(parameter_grid[best_id]["alpha"]),
        "l1_ratio": float(parameter_grid[best_id]["l1_ratio"]),
        "parameter_id": best_id,
        "mean_internal_patient_mae": parameter_scores[best_id],
    }
    return best_parameters, pd.DataFrame(manifest_rows), pd.DataFrame(coefficient_rows)


def elastic_net_stability_selection(
    data: TrainSelectionData,
    design: DesignMatrix,
    parameters: Mapping[str, float],
    *,
    iterations: int,
    subsample_fraction: float,
    seed: int,
    thresholds: Sequence[float] = DEFAULT_STABILITY_THRESHOLDS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Repeat Elastic Net on complete patient subsamples and summarize stability."""

    generator = np.random.default_rng(seed)
    feature_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    subsample_rows: list[dict[str, Any]] = []
    for iteration in range(iterations):
        selected_indices, _, selected_patients = patient_subsample(
            data.patient_ids, subsample_fraction, generator
        )
        scaler = StandardScaler().fit(design.X[selected_indices])
        X_selected = scaler.transform(design.X[selected_indices])
        model = ElasticNet(
            alpha=float(parameters["alpha"]),
            l1_ratio=float(parameters["l1_ratio"]),
            max_iter=10_000,
            selection="cyclic",
        )
        model.fit(
            X_selected,
            data.y_bis[selected_indices],
            sample_weight=patient_balanced_weights(data.patient_ids[selected_indices]),
        )
        subsample_rows.append(
            {
                "method": "elastic_net_stability",
                "iteration": iteration,
                "selected_patient_count": len(selected_patients),
                "selected_window_count": len(selected_indices),
                "selected_patient_ids": json.dumps(selected_patients.astype(int).tolist()),
                "patient_subsample_without_replacement": True,
            }
        )
        for column_index, coefficient in enumerate(model.coef_):
            coefficient_rows.append(
                {
                    "stage": "patient_stability",
                    "parameter_id": int(parameters["parameter_id"]),
                    "fold": np.nan,
                    "iteration": iteration,
                    "column": design.column_names[column_index],
                    "feature": design.column_feature[column_index],
                    "lag_seconds": design.column_lag_seconds[column_index],
                    "dynamic_candidate": column_index < design.dynamic_column_count,
                    "standardized_coefficient": float(coefficient),
                    "selected_nonzero": bool(abs(coefficient) > 1e-10),
                }
            )
        for feature in data.dynamic_features:
            indices = np.asarray(design.dynamic_column_indices[feature], dtype=int)
            coefficients = model.coef_[indices]
            strongest = float(coefficients[np.argmax(np.abs(coefficients))])
            feature_rows.append(
                {
                    "iteration": iteration,
                    "feature": feature,
                    "selected": bool(np.any(np.abs(coefficients) > 1e-10)),
                    "strongest_standardized_coefficient": strongest,
                    "sum_absolute_standardized_coefficient": float(
                        np.sum(np.abs(coefficients))
                    ),
                }
            )
    detail = pd.DataFrame(feature_rows)
    summary_rows = []
    for feature, group in detail.groupby("feature", sort=False):
        selected_coefficients = group.loc[
            group["selected"], "strongest_standardized_coefficient"
        ].to_numpy(dtype=float)
        selection_probability = float(group["selected"].mean())
        positive = int(np.sum(selected_coefficients > 0.0))
        negative = int(np.sum(selected_coefficients < 0.0))
        nonzero = len(selected_coefficients)
        row = {
            "feature": feature,
            "selection_probability": selection_probability,
            "median_standardized_coefficient": (
                float(np.median(selected_coefficients)) if nonzero else 0.0
            ),
            "median_absolute_standardized_coefficient": (
                float(np.median(np.abs(selected_coefficients))) if nonzero else 0.0
            ),
            "coefficient_sign_consistency": (
                max(positive, negative) / nonzero if nonzero else 0.0
            ),
            "dominant_coefficient_sign": (
                "positive" if positive > negative else "negative" if negative > positive else "mixed_or_zero"
            ),
            "stability_iterations": iterations,
        }
        for threshold in thresholds:
            row[f"selected_at_{threshold:.1f}"] = selection_probability >= threshold
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(coefficient_rows), pd.DataFrame(subsample_rows)


def make_xgboost_regressor(
    parameters: Mapping[str, Any], seed: int, device: str
) -> Regressor:
    """Construct one deterministic XGBoost regressor with optional CUDA hist training."""

    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise RuntimeError(
            "XGBoost is required for the tree selector. Install requirements-colab.txt."
        ) from error
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(parameters.get("n_estimators", 150)),
        max_depth=int(parameters["max_depth"]),
        min_child_weight=float(parameters["min_child_weight"]),
        learning_rate=float(parameters["learning_rate"]),
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        device=device,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


def patient_block_permute_feature(
    X: np.ndarray,
    patient_ids: np.ndarray,
    timestamps: np.ndarray,
    column_indices: Sequence[int],
    generator: np.random.Generator,
) -> np.ndarray:
    """Reassign complete lag blocks between patients while preserving source order."""

    result = np.asarray(X).copy()
    patients = np.unique(patient_ids)
    if len(patients) < 2:
        raise ValueError("Patient-block permutation requires at least two patients.")
    destination_order = generator.permutation(patients)
    shift = int(generator.integers(1, len(patients)))
    source_order = np.roll(destination_order, shift)
    columns = np.asarray(column_indices, dtype=int)
    for destination, source in zip(destination_order, source_order, strict=True):
        destination_rows = np.flatnonzero(patient_ids == destination)
        source_rows = np.flatnonzero(patient_ids == source)
        destination_rows = destination_rows[np.argsort(timestamps[destination_rows], kind="stable")]
        source_rows = source_rows[np.argsort(timestamps[source_rows], kind="stable")]
        source_positions = np.rint(
            np.linspace(0, len(source_rows) - 1, len(destination_rows))
        ).astype(int)
        result[np.ix_(destination_rows, columns)] = X[
            np.ix_(source_rows[source_positions], columns)
        ]
    return result


def grouped_permutation_importance(
    model: Regressor,
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    timestamps: np.ndarray,
    feature_columns: Mapping[str, Sequence[int]],
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Measure patient-balanced MAE increase after joint lag-block permutation."""

    baseline = patient_balanced_mae(y, model.predict(X), patient_ids)
    generator = np.random.default_rng(seed)
    rows = []
    for feature, columns in feature_columns.items():
        for repeat in range(repeats):
            permuted = patient_block_permute_feature(
                X, patient_ids, timestamps, columns, generator
            )
            permuted_mae = patient_balanced_mae(y, model.predict(permuted), patient_ids)
            rows.append(
                {
                    "feature": feature,
                    "repeat": repeat,
                    "baseline_patient_balanced_mae": baseline,
                    "permuted_patient_balanced_mae": permuted_mae,
                    "mae_increase": permuted_mae - baseline,
                    "permutation_unit": "complete feature lag block reassigned by patient",
                }
            )
    return pd.DataFrame(rows)


def tree_grouped_cv(
    data: TrainSelectionData,
    design: DesignMatrix,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    parameter_grid: Sequence[Mapping[str, Any]] = TREE_GRID,
    n_estimators: int,
    device: str,
    permutation_repeats: int,
    seed: int,
    estimator_factory: TreeFactory = make_xgboost_regressor,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Tune XGBoost on grouped folds and compute grouped permutation importance."""

    manifest_rows = []
    scores = []
    for parameter_id, raw_parameters in enumerate(parameter_grid):
        parameters = {**raw_parameters, "n_estimators": n_estimators}
        fold_scores = []
        for fold_id, (train_indices, evaluation_indices) in enumerate(folds):
            model = estimator_factory(parameters, seed + fold_id, device)
            model.fit(
                design.X[train_indices],
                data.y_bis[train_indices],
                sample_weight=patient_balanced_weights(data.patient_ids[train_indices]),
            )
            score = patient_balanced_mae(
                data.y_bis[evaluation_indices],
                model.predict(design.X[evaluation_indices]),
                data.patient_ids[evaluation_indices],
            )
            fold_scores.append(score)
            manifest_rows.append(
                _fold_manifest_row(
                    "xgboost",
                    parameter_id,
                    fold_id,
                    train_indices,
                    evaluation_indices,
                    data.patient_ids,
                    score,
                    parameters,
                )
            )
        scores.append(float(np.mean(fold_scores)))
    best_id = int(np.argmin(scores))
    best_parameters = {
        **parameter_grid[best_id],
        "n_estimators": n_estimators,
        "parameter_id": best_id,
        "mean_internal_patient_mae": scores[best_id],
    }

    importance_rows = []
    for fold_id, (train_indices, evaluation_indices) in enumerate(folds):
        model = estimator_factory(best_parameters, seed + 100 + fold_id, device)
        model.fit(
            design.X[train_indices],
            data.y_bis[train_indices],
            sample_weight=patient_balanced_weights(data.patient_ids[train_indices]),
        )
        fold_importance = grouped_permutation_importance(
            model,
            design.X[evaluation_indices],
            data.y_bis[evaluation_indices],
            data.patient_ids[evaluation_indices],
            data.target_timestamps[evaluation_indices],
            design.dynamic_column_indices,
            repeats=permutation_repeats,
            seed=seed + 1000 + fold_id,
        )
        fold_importance.insert(0, "fold", fold_id)
        importance_rows.append(fold_importance)
    detail = pd.concat(importance_rows, ignore_index=True)
    summary = (
        detail.groupby("feature", sort=False)["mae_increase"]
        .agg(
            grouped_permutation_importance_mean="mean",
            grouped_permutation_importance_standard_deviation="std",
            grouped_permutation_importance_median="median",
            grouped_permutation_observations="size",
        )
        .reset_index()
    )
    return best_parameters, pd.DataFrame(manifest_rows), summary


def tree_stability_selection(
    data: TrainSelectionData,
    design: DesignMatrix,
    parameters: Mapping[str, Any],
    *,
    iterations: int,
    subsample_fraction: float,
    device: str,
    seed: int,
    thresholds: Sequence[float] = DEFAULT_STABILITY_THRESHOLDS,
    estimator_factory: TreeFactory = make_xgboost_regressor,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat XGBoost fit/permutation on complementary patient blocks."""

    generator = np.random.default_rng(seed)
    detail_rows = []
    subsample_rows = []
    for iteration in range(iterations):
        train_indices, evaluation_indices, selected_patients = patient_subsample(
            data.patient_ids, subsample_fraction, generator
        )
        model = estimator_factory(parameters, seed + iteration, device)
        model.fit(
            design.X[train_indices],
            data.y_bis[train_indices],
            sample_weight=patient_balanced_weights(data.patient_ids[train_indices]),
        )
        importance = grouped_permutation_importance(
            model,
            design.X[evaluation_indices],
            data.y_bis[evaluation_indices],
            data.patient_ids[evaluation_indices],
            data.target_timestamps[evaluation_indices],
            design.dynamic_column_indices,
            repeats=1,
            seed=seed + 10_000 + iteration,
        )
        importance.insert(0, "iteration", iteration)
        detail_rows.append(importance)
        subsample_rows.append(
            {
                "method": "xgboost_stability",
                "iteration": iteration,
                "selected_patient_count": len(selected_patients),
                "evaluation_patient_count": len(
                    np.unique(data.patient_ids[evaluation_indices])
                ),
                "selected_window_count": len(train_indices),
                "evaluation_window_count": len(evaluation_indices),
                "selected_patient_ids": json.dumps(selected_patients.astype(int).tolist()),
                "patient_subsample_without_replacement": True,
            }
        )
    detail = pd.concat(detail_rows, ignore_index=True)
    detail["selected_positive_importance"] = detail["mae_increase"] > 0.0
    summary_rows = []
    for feature, group in detail.groupby("feature", sort=False):
        probability = float(group["selected_positive_importance"].mean())
        row = {
            "feature": feature,
            "selection_probability": probability,
            "grouped_permutation_importance_mean": float(group["mae_increase"].mean()),
            "grouped_permutation_importance_standard_deviation": float(
                group["mae_increase"].std(ddof=1)
            ),
            "grouped_permutation_importance_median": float(
                group["mae_increase"].median()
            ),
            "stability_iterations": iterations,
            "selection_rule": "held-out patient-block permutation MAE increase > 0",
        }
        for threshold in thresholds:
            row[f"selected_at_{threshold:.1f}"] = probability >= threshold
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(subsample_rows)


def optional_xgboost_shap_importance(
    data: TrainSelectionData,
    design: DesignMatrix,
    parameters: Mapping[str, Any],
    *,
    device: str,
    seed: int,
    max_windows: int,
) -> pd.DataFrame:
    """Compute optional in-tree SHAP contributions as auxiliary train-only evidence."""

    try:
        import xgboost as xgb
    except ImportError as error:
        raise RuntimeError("XGBoost is required for optional SHAP contributions.") from error
    model = make_xgboost_regressor(parameters, seed, device)
    model.fit(
        design.X,
        data.y_bis,
        sample_weight=patient_balanced_weights(data.patient_ids),
    )
    generator = np.random.default_rng(seed)
    sample_size = min(max_windows, len(design.X))
    sample_indices = np.sort(generator.choice(len(design.X), size=sample_size, replace=False))
    matrix = xgb.DMatrix(design.X[sample_indices], feature_names=list(design.column_names))
    contributions = model.get_booster().predict(matrix, pred_contribs=True)[:, :-1]  # type: ignore[attr-defined]
    rows = []
    for feature in data.dynamic_features:
        columns = np.asarray(design.dynamic_column_indices[feature], dtype=int)
        rows.append(
            {
                "feature": feature,
                "shap_mean_absolute_importance": float(
                    np.mean(np.sum(np.abs(contributions[:, columns]), axis=1))
                ),
                "shap_role": "auxiliary only; not a sole selection criterion",
                "shap_train_window_count": sample_size,
            }
        )
    return pd.DataFrame(rows)


def train_only_correlations(
    data: TrainSelectionData,
    *,
    cluster_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute decision-point and patient-mean Pearson/Spearman correlations."""

    current = pd.DataFrame(data.X_dynamic[:, -1, :], columns=data.dynamic_features)
    current.insert(0, "patient_id", data.patient_ids)
    patient_means = current.groupby("patient_id")[list(data.dynamic_features)].mean()
    window_values = current.loc[:, data.dynamic_features]
    matrices = {
        "pearson": (
            window_values.corr(method="pearson"),
            patient_means.corr(method="pearson"),
        ),
        "spearman": (
            window_values.corr(method="spearman"),
            patient_means.corr(method="spearman"),
        ),
    }
    outputs = {}
    for method, (window_matrix, patient_matrix) in matrices.items():
        rows = []
        for first_index, first in enumerate(data.dynamic_features):
            for second in data.dynamic_features[first_index + 1 :]:
                rows.append(
                    {
                        "feature_a": first,
                        "feature_b": second,
                        "window_level_correlation": float(window_matrix.loc[first, second]),
                        "patient_mean_correlation": float(patient_matrix.loc[first, second]),
                        "correlation_method": method,
                        "window_level_note": "repeated train windows; not independent observations",
                    }
                )
        outputs[method] = pd.DataFrame(rows)

    parent = {feature: feature for feature in data.dynamic_features}

    def find(feature: str) -> str:
        while parent[feature] != feature:
            parent[feature] = parent[parent[feature]]
            feature = parent[feature]
        return feature

    def union(first: str, second: str) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    combined = outputs["pearson"].merge(
        outputs["spearman"],
        on=["feature_a", "feature_b"],
        suffixes=("_pearson", "_spearman"),
    )
    for row in combined.itertuples():
        maximum = max(
            abs(row.window_level_correlation_pearson),
            abs(row.patient_mean_correlation_pearson),
            abs(row.window_level_correlation_spearman),
            abs(row.patient_mean_correlation_spearman),
        )
        if maximum >= cluster_threshold:
            union(row.feature_a, row.feature_b)
    clusters: dict[str, list[str]] = {}
    for feature in data.dynamic_features:
        clusters.setdefault(find(feature), []).append(feature)
    ordered_clusters = sorted(clusters.values(), key=lambda members: data.dynamic_features.index(members[0]))
    cluster_rows = []
    for cluster_index, members in enumerate(ordered_clusters, start=1):
        for feature in members:
            cluster_rows.append(
                {
                    "feature": feature,
                    "correlation_cluster": f"cluster_{cluster_index:02d}",
                    "cluster_members": json.dumps(members),
                    "cluster_size": len(members),
                    "absolute_correlation_threshold": cluster_threshold,
                    "feature_group": _feature_group(feature),
                    "removal_note": "correlation/VIF alone is not an automatic removal rule",
                }
            )
    return outputs["pearson"], outputs["spearman"], pd.DataFrame(cluster_rows)


def build_consensus_table(
    elastic_stability: pd.DataFrame,
    tree_importance: pd.DataFrame,
    tree_stability: pd.DataFrame,
    clusters: pd.DataFrame,
    shap_importance: pd.DataFrame | None = None,
    *,
    stable_threshold: float,
) -> pd.DataFrame:
    """Combine methods without collapsing them into one automatic winner score."""

    table = elastic_stability.rename(
        columns={
            "selection_probability": "elastic_net_selection_probability",
            "median_standardized_coefficient": "elastic_net_median_standardized_coefficient",
            "median_absolute_standardized_coefficient": "elastic_net_median_absolute_coefficient",
            "coefficient_sign_consistency": "elastic_net_sign_consistency",
        }
    )
    table = table.merge(tree_importance, on="feature", validate="one_to_one")
    tree_columns = tree_stability.rename(
        columns={
            "selection_probability": "tree_selection_probability",
            "grouped_permutation_importance_mean": "tree_stability_importance_mean",
            "grouped_permutation_importance_standard_deviation": "tree_stability_importance_standard_deviation",
        }
    )
    table = table.merge(tree_columns, on="feature", validate="one_to_one")
    table = table.merge(
        clusters[["feature", "correlation_cluster", "cluster_members", "feature_group"]],
        on="feature",
        validate="one_to_one",
    )
    if shap_importance is not None:
        table = table.merge(shap_importance, on="feature", validate="one_to_one")
    else:
        table["shap_mean_absolute_importance"] = np.nan
        table["shap_role"] = "not computed"
        table["shap_train_window_count"] = np.nan
    table["elastic_net_stable"] = (
        table["elastic_net_selection_probability"] >= stable_threshold
    )
    table["tree_stable"] = table["tree_selection_probability"] >= stable_threshold
    table["method_agreement_count"] = (
        table["elastic_net_stable"].astype(int) + table["tree_stable"].astype(int)
    )
    table["group_ablation_context"] = table["feature_group"].map(
        {
            "respiratory": "prior no_respiratory group ablation",
            "remifentanil": "prior remifentanil group ablation; retain control-design caution",
        }
    ).fillna("not directly isolated by prior group ablation")
    table["automatic_final_selection"] = False
    return table.sort_values(
        ["method_agreement_count", "elastic_net_selection_probability", "tree_selection_probability"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _compact_redundancy_aware_subset(consensus: pd.DataFrame) -> list[str]:
    candidates = consensus.loc[
        consensus["elastic_net_stable"] | consensus["tree_stable"]
    ].copy()
    if candidates.empty:
        return []
    candidates["combined_rank_key"] = (
        candidates["method_agreement_count"] * 10.0
        + candidates["elastic_net_selection_probability"]
        + candidates["tree_selection_probability"]
    )
    selected = (
        candidates.sort_values(
            ["combined_rank_key", "tree_stability_importance_mean", "elastic_net_median_absolute_coefficient"],
            ascending=False,
        )
        .groupby("correlation_cluster", sort=False)
        .head(1)["feature"]
        .tolist()
    )
    return [feature for feature in REDUCED_FEATURES if feature in selected]


def build_candidate_subsets(
    consensus: pd.DataFrame,
    *,
    protected_control_features: Sequence[str] = (),
    max_frozen_candidates: int = MAX_FROZEN_CANDIDATES,
) -> dict[str, Any]:
    """Create required discovery subsets and at most six frozen retraining candidates."""

    protected = tuple(dict.fromkeys(protected_control_features))
    unknown = sorted(set(protected) - set(REDUCED_FEATURES))
    if unknown:
        raise ValueError(f"Protected control features are not in full17: {unknown}")
    elastic = set(consensus.loc[consensus["elastic_net_stable"], "feature"])
    tree = set(consensus.loc[consensus["tree_stable"], "feature"])
    compact = _compact_redundancy_aware_subset(consensus)

    def ordered(features: Sequence[str] | set[str]) -> list[str]:
        return [feature for feature in REDUCED_FEATURES if feature in set(features)]

    all_candidates: dict[str, dict[str, Any]] = {
        "elastic_net_stable": {
            "features": ordered(elastic),
            "rule": "Elastic Net patient stability probability >= configured threshold",
            "candidate_role": "predictive-only discovery",
        },
        "tree_stable": {
            "features": ordered(tree),
            "rule": "XGBoost held-out patient permutation stability >= configured threshold",
            "candidate_role": "predictive-only discovery",
        },
        "strict_consensus": {
            "features": ordered(elastic & tree),
            "rule": "stable under both primary selector methods",
            "candidate_role": "predictive-only candidate",
        },
        "consensus_union": {
            "features": ordered(elastic | tree),
            "rule": "stable under at least one primary selector method",
            "candidate_role": "predictive-only candidate",
        },
        "compact_consensus": {
            "features": compact,
            "rule": "one strongest stable feature per empirical correlation cluster",
            "candidate_role": "predictive-only compact candidate",
        },
        "no_respiratory_anchor": {
            "features": ordered(set(REDUCED_FEATURES) - {"spo2", "etco2"}),
            "rule": "pre-specified 15-feature anchor from prior group ablation",
            "candidate_role": "group-ablation anchor",
        },
        "compact11_anchor": {
            "features": ordered(
                set(REDUCED_FEATURES)
                - {"spo2", "etco2", "rftn_rate", "rftn_volume", "rftn_cp", "rftn_ce"}
            ),
            "rule": "pre-specified 11-feature compact group-ablation anchor",
            "candidate_role": "group-ablation anchor; remifentanil control caution required",
        },
        "full17_reference": {
            "features": list(REDUCED_FEATURES),
            "rule": "reference set after deterministic bis_error removal",
            "candidate_role": "reference",
        },
    }
    if protected:
        all_candidates["control_aware_consensus"] = {
            "features": ordered(set(compact) | set(protected)),
            "rule": "compact predictive consensus plus explicitly supplied protected control features",
            "candidate_role": "control-aware candidate",
        }
    for candidate in all_candidates.values():
        candidate["feature_count"] = len(candidate["features"])
        candidate["eligible_for_retraining"] = len(candidate["features"]) > 0

    priority = [
        "full17_reference",
        "no_respiratory_anchor",
        "compact11_anchor",
        "strict_consensus",
        "compact_consensus",
        "control_aware_consensus" if protected else "consensus_union",
    ]
    frozen = []
    seen_feature_sets: set[tuple[str, ...]] = set()
    for name in priority:
        candidate = all_candidates[name]
        feature_key = tuple(candidate["features"])
        if candidate["eligible_for_retraining"] and feature_key not in seen_feature_sets:
            frozen.append(name)
            seen_feature_sets.add(feature_key)
        if len(frozen) == max_frozen_candidates:
            break
    if len(frozen) > max_frozen_candidates:
        raise AssertionError("Frozen candidate limit was exceeded.")
    return {
        "all_candidate_subsets": all_candidates,
        "frozen_retraining_candidates": frozen,
        "maximum_frozen_candidates": max_frozen_candidates,
        "protected_control_features": list(protected),
        "control_protection_policy": (
            "explicit user-supplied features only; repository contains no RL state implementation"
        ),
        "selection_warning": (
            "Predictive subsets are not final RL states. Remifentanil observations may need "
            "control-aware protection after the external RL state is inspected."
        ),
    }


def _plot_bar(
    frame: pd.DataFrame,
    value: str,
    title: str,
    xlabel: str,
    path: Path,
    error: str | None = None,
) -> None:
    ordered = frame.sort_values(value)
    figure, axis = plt.subplots(figsize=(9, 6))
    errors = ordered[error].to_numpy(dtype=float) if error else None
    axis.barh(ordered["feature"], ordered[value], xerr=errors, color="#2a9d8f")
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_selection_figures(
    elastic: pd.DataFrame,
    tree_importance: pd.DataFrame,
    tree_stability: pd.DataFrame,
    consensus: pd.DataFrame,
    pearson: pd.DataFrame,
    candidates: Mapping[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Save required train-only feature-selection figures."""

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "elastic_probability": figures_dir / "elastic_net_selection_probability.png",
        "coefficient": figures_dir / "elastic_net_coefficient_stability.png",
        "tree": figures_dir / "tree_grouped_permutation_importance.png",
        "agreement": figures_dir / "multi_method_agreement.png",
        "correlation": figures_dir / "train_feature_correlation_heatmap.png",
        "threshold": figures_dir / "feature_stability_threshold_sensitivity.png",
        "candidate_matrix": figures_dir / "candidate_subset_feature_matrix.png",
    }
    _plot_bar(
        elastic,
        "selection_probability",
        "Train-only Elastic Net patient stability",
        "Selection probability",
        paths["elastic_probability"],
    )
    coefficient_frame = elastic.copy()
    coefficient_frame["absolute_coefficient"] = coefficient_frame[
        "median_standardized_coefficient"
    ].abs()
    _plot_bar(
        coefficient_frame,
        "absolute_coefficient",
        "Train-only standardized coefficient stability",
        "Median absolute standardized coefficient",
        paths["coefficient"],
    )
    _plot_bar(
        tree_importance,
        "grouped_permutation_importance_mean",
        "Train-only XGBoost grouped permutation importance",
        "Patient-balanced MAE increase",
        paths["tree"],
        error="grouped_permutation_importance_standard_deviation",
    )
    _plot_bar(
        consensus,
        "method_agreement_count",
        "Train-only multi-method agreement",
        "Stable primary methods (0-2)",
        paths["agreement"],
    )

    matrix = pd.DataFrame(np.eye(len(REDUCED_FEATURES)), index=REDUCED_FEATURES, columns=REDUCED_FEATURES)
    for row in pearson.itertuples():
        matrix.loc[row.feature_a, row.feature_b] = row.window_level_correlation
        matrix.loc[row.feature_b, row.feature_a] = row.window_level_correlation
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(matrix)), matrix.columns, rotation=90)
    axis.set_yticks(range(len(matrix)), matrix.index)
    axis.set_title("Train-only decision-point Pearson correlation")
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    figure.tight_layout()
    figure.savefig(paths["correlation"], dpi=160)
    plt.close(figure)

    thresholds = DEFAULT_STABILITY_THRESHOLDS
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        thresholds,
        [sum(elastic["selection_probability"] >= threshold) for threshold in thresholds],
        marker="o",
        label="Elastic Net",
    )
    axis.plot(
        thresholds,
        [sum(tree_stability["selection_probability"] >= threshold) for threshold in thresholds],
        marker="o",
        label="XGBoost permutation",
    )
    axis.set_xlabel("Selection probability threshold")
    axis.set_ylabel("Selected dynamic feature count")
    axis.set_title("Train-only stability threshold sensitivity")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(paths["threshold"], dpi=160)
    plt.close(figure)

    candidate_map = candidates["all_candidate_subsets"]
    candidate_names = list(candidate_map)
    values = np.asarray(
        [
            [feature in candidate_map[name]["features"] for feature in REDUCED_FEATURES]
            for name in candidate_names
        ],
        dtype=int,
    )
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.imshow(values, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    axis.set_xticks(range(len(REDUCED_FEATURES)), REDUCED_FEATURES, rotation=90)
    axis.set_yticks(range(len(candidate_names)), candidate_names)
    axis.set_title("Frozen/discovery candidate dynamic-feature matrix")
    figure.tight_layout()
    figure.savefig(paths["candidate_matrix"], dpi=160)
    plt.close(figure)
    return list(paths.values())


def _frame_text(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    return "```text\n" + frame.loc[:, list(columns)].to_string(index=False) + "\n```"


def build_selection_report(
    config: SelectionConfig,
    elastic: pd.DataFrame,
    tree: pd.DataFrame,
    consensus: pd.DataFrame,
    candidates: Mapping[str, Any],
    prior_group_aggregate: pd.DataFrame,
    prior_group_bootstrap: pd.DataFrame,
) -> str:
    """Build a train-only methodological report without fabricated results."""

    candidate_rows = [
        {
            "candidate": name,
            "feature_count": payload["feature_count"],
            "role": payload["candidate_role"],
            "frozen": name in candidates["frozen_retraining_candidates"],
            "features": ",".join(payload["features"]),
        }
        for name, payload in candidates["all_candidate_subsets"].items()
    ]
    return f"""# Train-Only Predictive Feature Selection for 30-Second Future BIS

## Scope and data boundary
All selector fitting, tuning, correlations, stability subsampling, permutation importance, and optional SHAP contributions use only the pre-existing train split. Validation labels are not loaded or used. Held-out test data and test artifacts are not loaded. Static covariates are adjustment inputs only; only the 17 dynamic features are eligible for selection.

## Read-only prior group-ablation context
The completed validation-only group-ablation summaries are shown below only to document the pre-specified anchors. They are loaded after selector computation and do not tune feature rankings, thresholds, or stability models.

```text
{prior_group_aggregate.to_string(index=False)}
```

```text
{prior_group_bootstrap.to_string(index=False)}
```

## Patient grouping
Internal CV assigns complete patients to folds. Stability selection samples approximately {config.subsample_fraction:.0%} of train patients without replacement and retains every selected patient's windows together. Fitting weights give each patient equal total mass. Tree permutation reassigns each feature's six-lag block by patient while preserving source trajectory order.

## Elastic Net
Alpha and l1 ratio are selected by train-internal patient-grouped CV with a scaler fit separately in every training fold. Stability probabilities come from {config.stability_iterations} patient-level subsamples. Correlated coefficients may be shared or exchanged, so coefficient absence is not interpreted as causal irrelevance.

{_frame_text(elastic, ('feature', 'selection_probability', 'median_standardized_coefficient', 'coefficient_sign_consistency'))}

## XGBoost grouped permutation
XGBoost hyperparameters are selected by train-internal grouped CV. Primary interpretation is held-out patient-block grouped permutation, not gain importance. Optional SHAP contributions are auxiliary and never the sole selection criterion.

{_frame_text(tree, ('feature', 'grouped_permutation_importance_mean', 'grouped_permutation_importance_standard_deviation'))}

## Correlation, redundancy, and consensus
Window-level correlations are reported separately from patient-mean correlations. Repeated windows are not treated as independent evidence, and correlation clusters or VIF-like redundancy are not automatic removal rules. Multi-method evidence remains in separate columns rather than one winner score.

{_frame_text(consensus, ('feature', 'elastic_net_selection_probability', 'tree_selection_probability', 'correlation_cluster', 'feature_group', 'method_agreement_count'))}

## Candidate subsets
The workflow generates all requested discovery and anchor subsets but freezes at most {config.max_frozen_candidates} unique subsets for GRU/Attention retraining. Frozen names: {', '.join(candidates['frozen_retraining_candidates'])}.

{_frame_text(pd.DataFrame(candidate_rows), ('candidate', 'feature_count', 'role', 'frozen', 'features'))}

## Stability-threshold sensitivity
Selection probabilities are reported at thresholds 0.6, 0.7, 0.8, and 0.9. The configured threshold ({config.stable_threshold:.1f}) is an operational candidate rule, not an absolute truth. No false-selection-control guarantee is claimed because the correlated repeated-measure design does not establish the assumptions needed for such a bound.

## Control-design protection
Protected control features: {', '.join(config.protected_control_features) if config.protected_control_features else 'none supplied'}. The repository contains no professor RL environment or baseline state implementation, so no clinical control feature was protected by invention. Remifentanil rate, concentration, or effect-site observations may still be necessary to observe external drug administration even when predictive importance is low. A control-aware candidate is generated only from explicitly supplied protected features.

## Interpretation limits and next step
- This is predictive feature selection, not a causal analysis or final RL state declaration.
- The prior group-ablation results are read-only context and are not used to fit or tune selectors.
- Predictive utility for future BIS does not guarantee improved closed-loop propofol control.
- Frozen candidates must be retrained with the existing GRU and explicit feature-attention model before validation comparison.
- The held-out test split remains sealed until candidate decisions are frozen.
"""


def _validate_group_analysis(
    group_analysis_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    missing = [
        name for name in REQUIRED_GROUP_ANALYSIS_FILES if not (group_analysis_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Prior group-analysis context is missing: {missing}")
    manifest = load_json(group_analysis_dir / "analysis_manifest.json")
    if manifest.get("test_split_sealed") is not True:
        raise ValueError("Prior group analysis does not confirm a sealed test split.")
    fingerprints = [
        file_fingerprint(group_analysis_dir / name) for name in REQUIRED_GROUP_ANALYSIS_FILES
    ]
    aggregate = pd.read_csv(group_analysis_dir / "validation_condition_aggregate.csv")
    bootstrap = pd.read_csv(group_analysis_dir / "hierarchical_bootstrap_contrasts.csv")
    if aggregate.empty or bootstrap.empty:
        raise ValueError("Prior group-analysis context tables must not be empty.")
    return manifest, fingerprints, aggregate, bootstrap


def run_predictive_feature_selection(
    config: SelectionConfig,
    *,
    estimator_factory: TreeFactory = make_xgboost_regressor,
) -> dict[str, Any]:
    """Run the complete train-only selection workflow and write reproducible outputs."""

    _validate_config(config)
    dataset_dir = config.dataset_dir.resolve()
    group_analysis_dir = config.group_analysis_dir.resolve()
    output_dir = config.output_dir.resolve()
    if output_dir == dataset_dir or dataset_dir in output_dir.parents:
        raise ValueError("Selection outputs must not be written inside the modeling dataset.")
    if output_dir == group_analysis_dir or group_analysis_dir in output_dir.parents:
        raise ValueError("Selection outputs must not modify prior group-analysis results.")
    (
        group_manifest,
        group_fingerprints,
        prior_group_aggregate,
        prior_group_bootstrap,
    ) = _validate_group_analysis(group_analysis_dir)
    data = load_train_selection_data(dataset_dir)
    design = build_design_matrix(data)
    inventory = build_feature_inventory(data)
    folds = patient_grouped_folds(data.patient_ids, config.internal_folds, config.random_seed)

    elastic_parameters, elastic_cv, elastic_cv_coefficients = elastic_net_grouped_cv(
        data, design, folds
    )
    elastic_stability, elastic_stability_coefficients, elastic_subsamples = (
        elastic_net_stability_selection(
            data,
            design,
            elastic_parameters,
            iterations=config.stability_iterations,
            subsample_fraction=config.subsample_fraction,
            seed=config.random_seed + 100,
            thresholds=config.stability_thresholds,
        )
    )
    tree_parameters, tree_cv, tree_importance = tree_grouped_cv(
        data,
        design,
        folds,
        n_estimators=config.tree_estimators,
        device=config.tree_device,
        permutation_repeats=config.tree_permutation_repeats,
        seed=config.random_seed + 200,
        estimator_factory=estimator_factory,
    )
    tree_stability, tree_subsamples = tree_stability_selection(
        data,
        design,
        tree_parameters,
        iterations=config.stability_iterations,
        subsample_fraction=config.subsample_fraction,
        device=config.tree_device,
        seed=config.random_seed + 300,
        thresholds=config.stability_thresholds,
        estimator_factory=estimator_factory,
    )
    shap = None
    if config.compute_shap:
        if estimator_factory is not make_xgboost_regressor:
            raise ValueError("Optional SHAP is supported only by the production XGBoost estimator.")
        shap = optional_xgboost_shap_importance(
            data,
            design,
            tree_parameters,
            device=config.tree_device,
            seed=config.random_seed + 400,
            max_windows=config.shap_max_windows,
        )
    pearson, spearman, clusters = train_only_correlations(
        data, cluster_threshold=config.correlation_threshold
    )
    consensus = build_consensus_table(
        elastic_stability,
        tree_importance,
        tree_stability,
        clusters,
        shap,
        stable_threshold=config.stable_threshold,
    )
    candidates = build_candidate_subsets(
        consensus,
        protected_control_features=config.protected_control_features,
        max_frozen_candidates=config.max_frozen_candidates,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output_dir / "feature_inventory.csv", index=False)
    cv_manifest = pd.concat((elastic_cv, tree_cv), ignore_index=True)
    cv_manifest.to_csv(output_dir / "train_internal_cv_manifest.csv", index=False)
    pd.concat(
        (elastic_cv_coefficients, elastic_stability_coefficients), ignore_index=True
    ).to_csv(output_dir / "elastic_net_coefficients.csv", index=False)
    elastic_stability.to_csv(output_dir / "elastic_net_stability.csv", index=False)
    tree_importance.to_csv(output_dir / "tree_permutation_importance.csv", index=False)
    tree_stability.to_csv(output_dir / "tree_stability.csv", index=False)
    pearson.to_csv(output_dir / "feature_correlations_pearson.csv", index=False)
    spearman.to_csv(output_dir / "feature_correlations_spearman.csv", index=False)
    clusters.to_csv(output_dir / "correlation_clusters.csv", index=False)
    consensus.to_csv(output_dir / "consensus_feature_table.csv", index=False)
    pd.concat((elastic_subsamples, tree_subsamples), ignore_index=True).to_csv(
        output_dir / "patient_subsampling_manifest.csv", index=False
    )
    dump_json(candidates, output_dir / "candidate_subsets.json")
    figures = save_selection_figures(
        elastic_stability,
        tree_importance,
        tree_stability,
        consensus,
        pearson,
        candidates,
        output_dir,
    )
    report = build_selection_report(
        config,
        elastic_stability,
        tree_importance,
        consensus,
        candidates,
        prior_group_aggregate,
        prior_group_bootstrap,
    )
    (output_dir / "predictive_feature_selection_report.md").write_text(
        report, encoding="utf-8"
    )

    input_fingerprints = [
        file_fingerprint(dataset_dir / name) for name in TRAIN_ONLY_FILES
    ] + group_fingerprints
    generated_outputs = sorted(
        [path.name for path in output_dir.iterdir() if path.is_file()]
        + [str(path.relative_to(output_dir)) for path in figures]
        + ["selection_manifest.json"]
    )
    manifest = {
        "scientific_role": "legacy_physiological_exploratory_not_final_selection",
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_git_commit": _git_commit(Path(__file__).resolve().parents[1]),
        "configuration": {
            **asdict(config),
            "dataset_dir": str(dataset_dir),
            "group_analysis_dir": str(group_analysis_dir),
            "output_dir": str(output_dir),
            "protected_control_features": list(config.protected_control_features),
            "stability_thresholds": list(config.stability_thresholds),
        },
        "data_scope": "train split only",
        "validation_loaded": False,
        "test_loaded": False,
        "patient_count": int(np.unique(data.patient_ids).size),
        "window_count": len(data.y_bis),
        "dynamic_features": list(data.dynamic_features),
        "static_adjustment_features": list(data.static_features),
        "target": "30-second future BIS",
        "internal_grouping_unit": "patient case_id",
        "elastic_net_best_parameters": elastic_parameters,
        "xgboost_best_parameters": tree_parameters,
        "tree_interpretation_primary": "held-out patient-block grouped permutation importance",
        "shap_role": "auxiliary only" if config.compute_shap else "not computed",
        "prior_group_analysis_manifest_fingerprint": file_fingerprint(
            group_analysis_dir / "analysis_manifest.json"
        ),
        "prior_group_analysis_training_commits": group_manifest.get(
            "training_git_commits", []
        ),
        "prior_group_context_used_for_selector_tuning": False,
        "prior_group_aggregate_row_count": len(prior_group_aggregate),
        "prior_group_bootstrap_row_count": len(prior_group_bootstrap),
        "candidate_generation_rules": {
            name: payload["rule"]
            for name, payload in candidates["all_candidate_subsets"].items()
        },
        "frozen_retraining_candidates": candidates["frozen_retraining_candidates"],
        "input_fingerprints": input_fingerprints,
        "package_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "generated_outputs": generated_outputs,
        "methodological_warnings": [
            "predictive importance is not causal importance",
            "predictive selection does not establish RL control utility",
            "correlation clusters are not automatic removal rules",
            "stability thresholds are sensitivity settings, not absolute truth",
            "remifentanil observations may require control-aware protection",
        ],
    }
    dump_json(manifest, output_dir / "selection_manifest.json")
    LOGGER.info("Train-only predictive feature selection written to %s", output_dir)
    return {
        "output_dir": str(output_dir),
        "train_only": True,
        "validation_loaded": False,
        "test_loaded": False,
        "dynamic_feature_count": len(data.dynamic_features),
        "frozen_retraining_candidates": candidates["frozen_retraining_candidates"],
        "candidate_subsets": candidates,
    }
