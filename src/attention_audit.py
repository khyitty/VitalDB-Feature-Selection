"""Post-hoc audit utilities for one completed factorized-attention run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from src.attention_training import predict_and_extract_attention
from src.datasets import VitalBISDataset
from src.metrics import patient_level_evaluation, pooled_evaluation
from src.models.attention import FactorizedAttentionGRU
from src.models.baselines import GRUBaseline, PersistenceBaseline
from src.training import make_data_loader, predict_model

SPLITS = ("val", "test")
TIME_LAGS = (-50, -40, -30, -20, -10, 0)
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
STATIC_CONTEXT_NOTE = (
    "Static features (age, sex_male, height, weight, bmi, asa) are context inputs "
    "without explicit attention weights; their importance requires later ablation "
    "or another explicit mechanism."
)
DIAGNOSTIC_LABEL = "Single-seed diagnostic; not final feature-selection evidence."


@dataclass(frozen=True)
class SplitAttentionData:
    """Aligned predictions, masks, and attention arrays for one split."""

    split: str
    predictions: pd.DataFrame
    sample_indices: np.ndarray
    case_ids: np.ndarray
    observation_mask: np.ndarray
    feature_attention: np.ndarray
    temporal_attention: np.ndarray
    combined_attention: np.ndarray


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")


def align_prediction_rows(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    split: str,
    candidate_name: str,
) -> None:
    """Require identical ordered sample keys, targets, and labels."""

    keys = ["sample_index", "case_id", "target_timestamp"]
    required = set(keys) | {
        "observed_future_bis",
        "predicted_future_bis",
        "high_bis_label",
        "low_bis_label",
    }
    for name, frame in (("reference", reference), (candidate_name, candidate)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{split} {name} predictions lack columns: {missing}")
    if len(reference) != len(candidate):
        raise ValueError(
            f"{split} {candidate_name} row count {len(candidate)} does not match "
            f"reference count {len(reference)}."
        )
    if not reference[keys].equals(candidate[keys]):
        raise ValueError(f"{split} {candidate_name} prediction rows are misaligned.")
    for column in ("observed_future_bis", "high_bis_label", "low_bis_label"):
        if not np.array_equal(reference[column].to_numpy(), candidate[column].to_numpy()):
            raise ValueError(f"{split} {candidate_name} {column} differs from reference.")


def load_split_attention(
    run_dir: Path, dataset_dir: Path, split: str
) -> SplitAttentionData:
    """Load and strictly align one saved attention split."""

    predictions = pd.read_csv(run_dir / f"{split}_predictions.csv")
    metadata = pd.read_csv(dataset_dir / f"{split}_metadata.csv")
    with np.load(run_dir / f"{split}_attention.npz", allow_pickle=False) as archive:
        required = {
            "sample_index",
            "case_id",
            "feature_attention",
            "temporal_attention",
            "combined_attention",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{split} attention archive lacks arrays: {missing}")
        sample_indices = archive["sample_index"].astype(np.int64, copy=False)
        case_ids = archive["case_id"].astype(np.int64, copy=False)
        feature = archive["feature_attention"]
        temporal = archive["temporal_attention"]
        combined = archive["combined_attention"]
    if len(predictions) != len(sample_indices):
        raise ValueError(f"{split} prediction and attention row counts differ.")
    if not np.array_equal(predictions["sample_index"], sample_indices):
        raise ValueError(f"{split} attention sample indices do not align with predictions.")
    if not np.array_equal(predictions["case_id"], case_ids):
        raise ValueError(f"{split} attention case IDs do not align with predictions.")
    aligned_metadata = metadata.iloc[sample_indices]
    for column in ("case_id", "target_timestamp"):
        if not np.array_equal(predictions[column], aligned_metadata[column]):
            raise ValueError(f"{split} predictions do not align with metadata {column}.")
    with np.load(dataset_dir / f"{split}.npz", allow_pickle=False) as dataset_archive:
        observation_mask = dataset_archive["observation_mask"][sample_indices]
    expected_shape = observation_mask.shape
    if feature.shape != expected_shape or combined.shape != expected_shape:
        raise ValueError(f"{split} feature/combined attention shape is inconsistent.")
    if temporal.shape != expected_shape[:2]:
        raise ValueError(f"{split} temporal attention shape is inconsistent.")
    arrays = (
        predictions.select_dtypes(include="number").to_numpy(),
        feature,
        temporal,
        combined,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError(f"{split} prediction or attention values are non-finite.")
    return SplitAttentionData(
        split=split,
        predictions=predictions,
        sample_indices=sample_indices,
        case_ids=case_ids,
        observation_mask=observation_mask.astype(bool, copy=False),
        feature_attention=feature,
        temporal_attention=temporal,
        combined_attention=combined,
    )


def normalized_feature_entropy(
    feature_attention: np.ndarray, observation_mask: np.ndarray
) -> np.ndarray:
    """Return entropy normalized by log of each time step's observed count."""

    weights = np.asarray(feature_attention, dtype=float)
    mask = np.asarray(observation_mask, dtype=bool)
    if weights.shape != mask.shape or weights.ndim != 3:
        raise ValueError("Feature attention and mask must share shape [N, L, P].")
    observed_count = mask.sum(axis=2)
    if np.any(observed_count == 0):
        raise ValueError("Entropy is undefined for a time step with no observed feature.")
    terms = np.zeros_like(weights)
    positive = weights > 0
    terms[positive] = -weights[positive] * np.log(weights[positive])
    entropy = terms.sum(axis=2)
    denominator = np.where(observed_count > 1, np.log(observed_count), 1.0)
    normalized = entropy / denominator
    normalized[observed_count == 1] = 0.0
    return normalized


def attention_normalization_audit(
    data: SplitAttentionData, tolerance: float = 1e-5
) -> dict[str, Any]:
    """Audit non-negativity, masks, normalization, and factorization."""

    feature_error = np.abs(data.feature_attention.sum(axis=2) - 1.0)
    temporal_error = np.abs(data.temporal_attention.sum(axis=1) - 1.0)
    definition_error = np.abs(
        data.combined_attention
        - data.temporal_attention[:, :, None] * data.feature_attention
    )
    combined_error = np.abs(data.combined_attention.sum(axis=(1, 2)) - 1.0)
    missing_weights = data.feature_attention[~data.observation_mask]
    return {
        "feature_attention": {
            "minimum_weight": float(data.feature_attention.min()),
            "non_negative": bool((data.feature_attention >= 0).all()),
            "maximum_normalization_error": float(feature_error.max()),
            "time_steps_violating_tolerance_1e_5": int(
                (feature_error > tolerance).sum()
            ),
            "maximum_unobserved_feature_weight": float(
                np.max(np.abs(missing_weights), initial=0.0)
            ),
            "unobserved_feature_nonzero_count": int(
                np.count_nonzero(missing_weights)
            ),
        },
        "temporal_attention": {
            "minimum_weight": float(data.temporal_attention.min()),
            "non_negative": bool((data.temporal_attention >= 0).all()),
            "maximum_normalization_error": float(temporal_error.max()),
            "rows_violating_tolerance_1e_5": int(
                (temporal_error > tolerance).sum()
            ),
        },
        "combined_attention": {
            "minimum_weight": float(data.combined_attention.min()),
            "non_negative": bool((data.combined_attention >= 0).all()),
            "maximum_definition_error": float(definition_error.max()),
            "maximum_normalization_error": float(combined_error.max()),
            "rows_violating_tolerance_1e_5": int(
                (combined_error > tolerance).sum()
            ),
        },
    }


def _case_level_array(
    values: np.ndarray, case_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    cases = np.unique(case_ids)
    return cases, np.stack([values[case_ids == case].mean(axis=0) for case in cases])


def case_balanced_feature_summary(
    data: SplitAttentionData,
    feature_names: Iterable[str],
    validation_ranks: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Summarize features after averaging windows within case, then across cases."""

    names = list(feature_names)
    _, case_feature = _case_level_array(
        data.feature_attention.mean(axis=1), data.case_ids
    )
    _, case_combined = _case_level_array(
        data.combined_attention.sum(axis=1), data.case_ids
    )
    rows = []
    for index, name in enumerate(names):
        values = case_feature[:, index]
        rows.append(
            {
                "split": data.split,
                "feature": name,
                "mean_feature_attention": float(values.mean()),
                "standard_deviation_across_cases": float(
                    values.std(ddof=1) if len(values) > 1 else 0.0
                ),
                "median_across_cases": float(np.median(values)),
                "minimum_case_attention": float(values.min()),
                "maximum_case_attention": float(values.max()),
                "mean_combined_attention_summed_across_time": float(
                    case_combined[:, index].mean()
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if validation_ranks is None:
        ordered = frame.sort_values(
            "mean_feature_attention", ascending=False
        )["feature"].tolist()
        validation_ranks = {name: rank for rank, name in enumerate(ordered, start=1)}
    frame["validation_rank"] = frame["feature"].map(validation_ranks).astype(int)
    return frame.sort_values("validation_rank").reset_index(drop=True)


def case_balanced_time_summary(
    data: SplitAttentionData,
    time_lags: Iterable[int],
    validation_ranks: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Summarize temporal attention with equal case weighting."""

    lags = list(time_lags)
    _, case_time = _case_level_array(data.temporal_attention, data.case_ids)
    rows = []
    for index, lag in enumerate(lags):
        values = case_time[:, index]
        rows.append(
            {
                "split": data.split,
                "time_lag_seconds": int(lag),
                "mean_temporal_attention": float(values.mean()),
                "standard_deviation_across_cases": float(
                    values.std(ddof=1) if len(values) > 1 else 0.0
                ),
                "median_across_cases": float(np.median(values)),
                "minimum_case_attention": float(values.min()),
                "maximum_case_attention": float(values.max()),
            }
        )
    frame = pd.DataFrame(rows)
    if validation_ranks is None:
        ordered = frame.sort_values(
            "mean_temporal_attention", ascending=False
        )["time_lag_seconds"].tolist()
        validation_ranks = {lag: rank for rank, lag in enumerate(ordered, start=1)}
    frame["validation_rank"] = (
        frame["time_lag_seconds"].map(validation_ranks).astype(int)
    )
    return frame.sort_values("validation_rank").reset_index(drop=True)


def attention_concentration(data: SplitAttentionData) -> dict[str, Any]:
    """Calculate entropy and top-mass diagnostics without scientific interpretation."""

    feature_entropy = normalized_feature_entropy(
        data.feature_attention, data.observation_mask
    )
    sorted_feature = np.sort(data.feature_attention, axis=2)
    temporal_terms = np.zeros_like(data.temporal_attention, dtype=float)
    positive_temporal = data.temporal_attention > 0
    temporal_terms[positive_temporal] = -data.temporal_attention[
        positive_temporal
    ] * np.log(data.temporal_attention[positive_temporal])
    temporal_entropy = temporal_terms.sum(axis=1) / np.log(
        data.temporal_attention.shape[1]
    )
    temporal_max = data.temporal_attention.max(axis=1)
    flat_combined = data.combined_attention.reshape(len(data.case_ids), -1)
    sorted_combined = np.sort(flat_combined, axis=1)
    feature_top1 = sorted_feature[:, :, -1]
    feature_top3 = sorted_feature[:, :, -3:].sum(axis=2)
    combined_top5 = sorted_combined[:, -5:].sum(axis=1)
    collapse = {
        "feature_over_0_9_large_majority": bool((feature_top1 > 0.9).mean() >= 0.8),
        "temporal_over_0_9_large_majority": bool((temporal_max > 0.9).mean() >= 0.8),
        "feature_entropy_near_zero_large_majority": bool(
            (feature_entropy < 0.1).mean() >= 0.8
        ),
        "temporal_entropy_near_zero_large_majority": bool(
            (temporal_entropy < 0.1).mean() >= 0.8
        ),
    }
    return {
        "feature_attention": {
            "mean_normalized_entropy": float(feature_entropy.mean()),
            "median_normalized_entropy": float(np.median(feature_entropy)),
            "mean_top_1_weight": float(feature_top1.mean()),
            "median_top_1_weight": float(np.median(feature_top1)),
            "mean_top_3_cumulative_weight": float(feature_top3.mean()),
            "proportion_time_steps_top_1_over_0_5": float((feature_top1 > 0.5).mean()),
            "proportion_time_steps_top_1_over_0_75": float((feature_top1 > 0.75).mean()),
            "proportion_time_steps_top_1_over_0_9": float((feature_top1 > 0.9).mean()),
        },
        "temporal_attention": {
            "mean_normalized_entropy": float(temporal_entropy.mean()),
            "median_normalized_entropy": float(np.median(temporal_entropy)),
            "mean_maximum_time_weight": float(temporal_max.mean()),
            "proportion_samples_max_over_0_5": float((temporal_max > 0.5).mean()),
            "proportion_samples_max_over_0_75": float((temporal_max > 0.75).mean()),
            "proportion_samples_max_over_0_9": float((temporal_max > 0.9).mean()),
        },
        "combined_attention": {
            "mean_maximum_feature_lag_cell_weight": float(flat_combined.max(axis=1).mean()),
            "median_maximum_feature_lag_cell_weight": float(
                np.median(flat_combined.max(axis=1))
            ),
            "mean_top_5_cumulative_weight": float(combined_top5.mean()),
        },
        "collapse_rules": collapse,
        "possible_numerical_collapse": bool(any(collapse.values())),
        "large_majority_definition": "at least 80%",
    }


def metric_values(frame: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    observed = frame["observed_future_bis"].to_numpy(dtype=float)
    predicted = frame["predicted_future_bis"].to_numpy(dtype=float)
    cases = frame["case_id"].to_numpy(dtype=int)
    pooled = pooled_evaluation(observed, predicted)
    patient = patient_level_evaluation(observed, predicted, cases)
    return {
        "pooled_mae": float(pooled["regression"]["mae"]),
        "pooled_rmse": float(pooled["regression"]["rmse"]),
        "r_squared": float(pooled["regression"]["r_squared"]),
        "patient_mean_mae": float(patient.summary["mae"]["mean"]),
        "patient_median_mae": float(patient.summary["mae"]["median"]),
        "patient_mae_standard_deviation": float(
            patient.summary["mae"]["standard_deviation"]
        ),
        "patient_mae_interquartile_range": float(
            patient.summary["mae"]["interquartile_range"]
        ),
        "patient_mean_rmse": float(patient.summary["rmse"]["mean"]),
        "bis_below_40_mae": float(pooled["bis_region_mae"]["bis_below_40"]),
        "bis_40_to_60_mae": float(pooled["bis_region_mae"]["bis_40_to_60"]),
        "bis_above_60_mae": float(pooled["bis_region_mae"]["bis_above_60"]),
        "high_bis_auprc": float(pooled["high_bis_classification"]["auprc"]),
        "high_bis_auroc": float(pooled["high_bis_classification"]["auroc"]),
        "low_bis_auprc": float(pooled["low_bis_classification"]["auprc"]),
        "low_bis_auroc": float(pooled["low_bis_classification"]["auroc"]),
    }, patient.case_metrics


def prediction_distribution(frame: pd.DataFrame) -> dict[str, float]:
    observed = frame["observed_future_bis"].to_numpy(dtype=float)
    predicted = frame["predicted_future_bis"].to_numpy(dtype=float)
    return {
        "observed_mean": float(observed.mean()),
        "observed_standard_deviation": float(observed.std(ddof=0)),
        "predicted_mean": float(predicted.mean()),
        "predicted_standard_deviation": float(predicted.std(ddof=0)),
        "predicted_minimum": float(predicted.min()),
        "predicted_maximum": float(predicted.max()),
        "pearson_correlation": float(np.corrcoef(observed, predicted)[0, 1]),
    }


def high_bis_bias(frame: pd.DataFrame) -> dict[str, float | int]:
    high = frame.loc[frame["observed_future_bis"] > 60]
    return {
        "window_count": int(len(high)),
        "mean_prediction_bias_predicted_minus_observed": float(
            (high["predicted_future_bis"] - high["observed_future_bis"]).mean()
        ),
        "observed_bis_mean": float(high["observed_future_bis"].mean()),
        "predicted_bis_mean": float(high["predicted_future_bis"].mean()),
    }


def _build_model_from_config(
    model_name: str, config: dict[str, Any]
) -> nn.Module:
    if model_name == "attention":
        return FactorizedAttentionGRU(
            dynamic_feature_count=len(config["dynamic_feature_names"]),
            static_feature_count=len(config["static_feature_names"]),
            history_steps=6,
            feature_token_embedding_dim=int(config["feature_token_embedding_dim"]),
            static_context_dim=int(config["static_context_dim"]),
            hidden_size=int(config["hidden_size"]),
            prediction_hidden_size=int(config["prediction_hidden_size"]),
            dropout=float(config["dropout"]),
        )
    return GRUBaseline(
        dynamic_feature_count=len(config["dynamic_feature_names"]),
        static_feature_count=len(config["static_feature_names"]),
        hidden_size=int(config["hidden_size"]),
        projection_size=int(config["projection_size"]),
        static_hidden_size=int(config["static_hidden_size"]),
        prediction_hidden_size=int(config["prediction_hidden_size"]),
        dropout=float(config["dropout"]),
    )


def benchmark_inference(
    dataset_dir: Path,
    attention_run_dir: Path,
    gru_run_dir: Path,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Time one exhaustive CPU inference pass per model and split."""

    device = torch.device("cpu")
    criterion = nn.HuberLoss(delta=1.0)
    attention_config = load_json(attention_run_dir / "config.json")
    gru_config = load_json(gru_run_dir / "config.json")
    attention = _build_model_from_config("attention", attention_config).to(device)
    gru = _build_model_from_config("gru", gru_config).to(device)
    attention.load_state_dict(
        torch.load(
            attention_run_dir / "best_model.pt", map_location=device, weights_only=False
        )["model_state_dict"]
    )
    gru.load_state_dict(
        torch.load(
            gru_run_dir / "best_model.pt", map_location=device, weights_only=False
        )["model_state_dict"]
    )
    stats = pd.read_csv(dataset_dir / "preprocessing_statistics.csv").set_index(
        "feature_name"
    )
    persistence: PersistenceBaseline | None = None
    results: dict[str, Any] = {}
    for split in SPLITS:
        dataset = VitalBISDataset(dataset_dir, split)
        if persistence is None:
            persistence = PersistenceBaseline.from_feature_metadata(
                dataset.dynamic_feature_names,
                float(stats.loc["bis", "training_mean"]),
                float(stats.loc["bis", "training_standard_deviation"]),
            )
        indices = np.arange(len(dataset), dtype=np.int64)
        loader = make_data_loader(
            dataset,
            indices,
            batch_size,
            seed=42,
            training=False,
            case_balanced=False,
        )
        started = perf_counter()
        persistence.predict(dataset.arrays["X_dynamic"])
        persistence_seconds = perf_counter() - started
        started = perf_counter()
        predict_model(gru, loader, criterion, device)
        gru_seconds = perf_counter() - started
        started = perf_counter()
        predict_model(attention, loader, criterion, device)
        attention_prediction_seconds = perf_counter() - started
        _, _, joint_timing = predict_and_extract_attention(
            attention, loader, criterion, device
        )
        results[split] = {
            "persistence_prediction_seconds": persistence_seconds,
            "gru_prediction_seconds": gru_seconds,
            "attention_prediction_seconds": attention_prediction_seconds,
            "attention_prediction_including_extraction_seconds": (
                joint_timing.total_seconds
            ),
            "attention_shared_forward_with_extraction_seconds": (
                joint_timing.shared_model_forward_seconds
            ),
            "single_pass_diagnostic": True,
            "batch_size": batch_size,
            "window_count": len(dataset),
        }
    return results


def _region_summaries(
    validation: SplitAttentionData,
    feature_names: list[str],
    overall_feature: pd.DataFrame,
    overall_time: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    observed = validation.predictions["observed_future_bis"].to_numpy()
    regions = {
        "bis_below_40": observed < 40,
        "bis_40_to_60": (observed >= 40) & (observed <= 60),
        "bis_above_60": observed > 60,
    }
    feature_rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {}
    overall_feature_map = overall_feature.set_index("feature")[
        "mean_feature_attention"
    ]
    overall_time_map = overall_time.set_index("time_lag_seconds")[
        "mean_temporal_attention"
    ]
    for region, selector in regions.items():
        subset = SplitAttentionData(
            split="val",
            predictions=validation.predictions.loc[selector].reset_index(drop=True),
            sample_indices=validation.sample_indices[selector],
            case_ids=validation.case_ids[selector],
            observation_mask=validation.observation_mask[selector],
            feature_attention=validation.feature_attention[selector],
            temporal_attention=validation.temporal_attention[selector],
            combined_attention=validation.combined_attention[selector],
        )
        feature = case_balanced_feature_summary(subset, feature_names)
        time = case_balanced_time_summary(subset, TIME_LAGS)
        feature["region"] = region
        feature["window_count"] = int(selector.sum())
        feature["case_count"] = int(np.unique(subset.case_ids).size)
        feature["difference_from_overall_validation"] = feature.apply(
            lambda row: row["mean_feature_attention"]
            - overall_feature_map.loc[row["feature"]],
            axis=1,
        )
        time["region"] = region
        time["window_count"] = int(selector.sum())
        time["case_count"] = int(np.unique(subset.case_ids).size)
        time["difference_from_overall_validation"] = time.apply(
            lambda row: row["mean_temporal_attention"]
            - overall_time_map.loc[row["time_lag_seconds"]],
            axis=1,
        )
        feature_rows.extend(feature.to_dict(orient="records"))
        time_rows.extend(time.to_dict(orient="records"))
        largest = feature.reindex(
            feature["difference_from_overall_validation"].abs().sort_values(
                ascending=False
            ).index
        ).head(5)
        payload[region] = {
            "window_count": int(selector.sum()),
            "case_count": int(np.unique(subset.case_ids).size),
            "five_largest_absolute_feature_attention_differences": largest[
                ["feature", "difference_from_overall_validation"]
            ].to_dict(orient="records"),
        }
    return pd.DataFrame(feature_rows), pd.DataFrame(time_rows), payload


def _case_comparison(
    attention_cases: pd.DataFrame,
    gru_cases: pd.DataFrame,
    persistence_cases: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    paired = attention_cases[["case_id", "mae"]].rename(
        columns={"mae": "attention_mae"}
    )
    paired = paired.merge(
        gru_cases[["case_id", "mae"]].rename(columns={"mae": "gru_mae"}),
        on="case_id",
        validate="one_to_one",
    ).merge(
        persistence_cases[["case_id", "mae"]].rename(
            columns={"mae": "persistence_mae"}
        ),
        on="case_id",
        validate="one_to_one",
    )
    paired["attention_minus_gru_mae"] = paired["attention_mae"] - paired["gru_mae"]
    paired["attention_minus_persistence_mae"] = (
        paired["attention_mae"] - paired["persistence_mae"]
    )
    columns = ["case_id", "attention_minus_gru_mae"]
    return {
        "case_count": int(len(paired)),
        "patients_attention_mae_lower_than_gru": int(
            (paired["attention_minus_gru_mae"] < 0).sum()
        ),
        "patients_attention_mae_lower_than_persistence": int(
            (paired["attention_minus_persistence_mae"] < 0).sum()
        ),
        "median_attention_minus_gru_mae": float(
            paired["attention_minus_gru_mae"].median()
        ),
        "five_largest_improvements_relative_to_gru": paired.nsmallest(
            5, "attention_minus_gru_mae"
        )[columns].to_dict(orient="records"),
        "five_largest_deteriorations_relative_to_gru": paired.nlargest(
            5, "attention_minus_gru_mae"
        )[columns].to_dict(orient="records"),
        "cases_97_and_154": paired.loc[
            paired["case_id"].isin([97, 154])
        ].to_dict(orient="records"),
    }, paired


def _plot_diagnostics(
    output_dir: Path,
    feature_summary: pd.DataFrame,
    time_summary: pd.DataFrame,
    heatmap: pd.DataFrame,
) -> None:
    validation_features = feature_summary.loc[feature_summary["split"] == "val"].sort_values(
        "mean_feature_attention"
    )
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.barh(
        validation_features["feature"],
        validation_features["mean_feature_attention"],
    )
    axis.set_xlabel("Case-balanced mean feature attention")
    axis.set_title(f"Validation feature attention\n{DIAGNOSTIC_LABEL}")
    fig.tight_layout()
    fig.savefig(output_dir / "validation_feature_attention.png", dpi=150)
    plt.close(fig)

    validation_time = time_summary.loc[time_summary["split"] == "val"].sort_values(
        "time_lag_seconds"
    )
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(
        validation_time["time_lag_seconds"].astype(str),
        validation_time["mean_temporal_attention"],
    )
    axis.set_xlabel("Time lag (seconds)")
    axis.set_ylabel("Case-balanced mean temporal attention")
    axis.set_title(f"Validation temporal attention\n{DIAGNOSTIC_LABEL}")
    fig.tight_layout()
    fig.savefig(output_dir / "validation_temporal_attention.png", dpi=150)
    plt.close(fig)

    matrix = heatmap.pivot(
        index="feature", columns="time_lag_seconds", values="mean_combined_attention"
    )
    fig, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns)
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set_xlabel("Time lag (seconds)")
    axis.set_title(f"Validation combined attention\n{DIAGNOSTIC_LABEL}")
    fig.colorbar(image, ax=axis, label="Case-balanced mean combined attention")
    fig.tight_layout()
    fig.savefig(output_dir / "validation_combined_attention_heatmap.png", dpi=150)
    plt.close(fig)


def run_attention_audit(
    *,
    run_dir: Path,
    dataset_dir: Path,
    baselines_dir: Path,
    command_wall_seconds: float,
) -> dict[str, Any]:
    """Create the complete one-seed attention audit and source tables."""

    run_config = load_json(run_dir / "config.json")
    dataset_metadata = load_json(dataset_dir / "dataset_metadata.json")
    attention_metadata = load_json(run_dir / "attention_metadata.json")
    feature_names = list(dataset_metadata["dynamic_feature_names"])
    if run_config["dynamic_feature_names"] != feature_names:
        raise ValueError("Run feature order differs from dataset metadata.")
    if attention_metadata["dynamic_feature_names"] != feature_names:
        raise ValueError("Attention metadata feature order differs from dataset metadata.")
    if attention_metadata["time_lags_seconds"] != list(TIME_LAGS):
        raise ValueError("Attention time-lag order is not -50,-40,-30,-20,-10,0.")

    data = {
        split: load_split_attention(run_dir, dataset_dir, split) for split in SPLITS
    }
    persistence_predictions = {
        split: pd.read_csv(baselines_dir / "persistence" / f"{split}_predictions.csv")
        for split in SPLITS
    }
    gru_dir = baselines_dir / "gru" / "seed_42"
    gru_predictions = {
        split: pd.read_csv(gru_dir / f"{split}_predictions.csv") for split in SPLITS
    }
    for split in SPLITS:
        align_prediction_rows(
            data[split].predictions,
            persistence_predictions[split],
            split=split,
            candidate_name="persistence",
        )
        align_prediction_rows(
            data[split].predictions,
            gru_predictions[split],
            split=split,
            candidate_name="GRU seed 42",
        )

    normalization = {
        split: attention_normalization_audit(data[split]) for split in SPLITS
    }
    concentration = {split: attention_concentration(data[split]) for split in SPLITS}
    validation_feature = case_balanced_feature_summary(data["val"], feature_names)
    feature_ranks = dict(
        zip(
            validation_feature["feature"],
            validation_feature["validation_rank"],
            strict=True,
        )
    )
    test_feature = case_balanced_feature_summary(
        data["test"], feature_names, feature_ranks
    )
    feature_summary = pd.concat((validation_feature, test_feature), ignore_index=True)
    validation_time = case_balanced_time_summary(data["val"], TIME_LAGS)
    time_ranks = dict(
        zip(
            validation_time["time_lag_seconds"],
            validation_time["validation_rank"],
            strict=True,
        )
    )
    test_time = case_balanced_time_summary(data["test"], TIME_LAGS, time_ranks)
    time_summary = pd.concat((validation_time, test_time), ignore_index=True)
    feature_summary.to_csv(run_dir / "feature_attention_summary.csv", index=False)
    time_summary.to_csv(run_dir / "time_attention_summary.csv", index=False)

    _, case_combined = _case_level_array(
        data["val"].combined_attention, data["val"].case_ids
    )
    mean_heatmap = case_combined.mean(axis=0)
    heatmap_rows = [
        {
            "feature": feature,
            "time_lag_seconds": lag,
            "mean_combined_attention": float(mean_heatmap[lag_index, feature_index]),
            "diagnostic_label": DIAGNOSTIC_LABEL,
        }
        for feature_index, feature in enumerate(feature_names)
        for lag_index, lag in enumerate(TIME_LAGS)
    ]
    heatmap = pd.DataFrame(heatmap_rows)
    heatmap.to_csv(
        run_dir / "validation_combined_attention_heatmap.csv", index=False
    )
    region_feature, region_time, region_payload = _region_summaries(
        data["val"], feature_names, validation_feature, validation_time
    )
    region_feature.to_csv(
        run_dir / "validation_bis_region_feature_attention.csv", index=False
    )
    region_time.to_csv(
        run_dir / "validation_bis_region_time_attention.csv", index=False
    )
    _plot_diagnostics(run_dir, feature_summary, time_summary, heatmap)

    inference = benchmark_inference(dataset_dir, run_dir, gru_dir)
    parameter_counts = {
        "persistence": 0,
        "gru_seed_42": int(load_json(gru_dir / "config.json")["model_parameter_count"]),
        "factorized_attention_gru_seed_42": int(run_config["model_parameter_count"]),
    }
    comparison_rows: list[dict[str, Any]] = []
    comparison_payload: dict[str, Any] = {}
    patient_payload: dict[str, Any] = {}
    performance: dict[str, Any] = {}
    for split in SPLITS:
        frames = {
            "persistence": persistence_predictions[split],
            "gru_seed_42": gru_predictions[split],
            "factorized_attention_gru_seed_42": data[split].predictions,
        }
        values: dict[str, dict[str, float]] = {}
        cases: dict[str, pd.DataFrame] = {}
        for model_name, frame in frames.items():
            values[model_name], cases[model_name] = metric_values(frame)
            timing_key = {
                "persistence": "persistence_prediction_seconds",
                "gru_seed_42": "gru_prediction_seconds",
                "factorized_attention_gru_seed_42": "attention_prediction_seconds",
            }[model_name]
            row = {
                "split": split,
                "model": model_name,
                **values[model_name],
                "parameter_count": parameter_counts[model_name],
                "prediction_inference_seconds": inference[split][timing_key],
                "inference_including_attention_seconds": (
                    inference[split][
                        "attention_prediction_including_extraction_seconds"
                    ]
                    if model_name == "factorized_attention_gru_seed_42"
                    else None
                ),
            }
            comparison_rows.append(row)
        differences: dict[str, Any] = {}
        for reference in ("gru_seed_42", "persistence"):
            name = f"attention_minus_{reference}"
            difference = {
                metric: values["factorized_attention_gru_seed_42"][metric]
                - values[reference][metric]
                for metric in COMPARISON_METRICS
            }
            differences[name] = difference
            comparison_rows.append(
                {
                    "split": split,
                    "model": name,
                    **difference,
                    "parameter_count": (
                        parameter_counts["factorized_attention_gru_seed_42"]
                        - parameter_counts[reference]
                    ),
                    "prediction_inference_seconds": (
                        inference[split]["attention_prediction_seconds"]
                        - inference[split][
                            "gru_prediction_seconds"
                            if reference == "gru_seed_42"
                            else "persistence_prediction_seconds"
                        ]
                    ),
                    "inference_including_attention_seconds": None,
                }
            )
        comparison_payload[split] = {
            "models": values,
            "differences": differences,
            "difference_direction": {
                metric: "negative favors attention"
                if metric in ERROR_METRICS
                else "positive favors attention"
                for metric in COMPARISON_METRICS
            },
            "inference_timing": inference[split],
        }
        case_summary, paired = _case_comparison(
            cases["factorized_attention_gru_seed_42"],
            cases["gru_seed_42"],
            cases["persistence"],
        )
        patient_payload[split] = case_summary
        if split == "test":
            paired.to_csv(run_dir / "test_patient_model_comparison.csv", index=False)
        performance[split] = {
            "metrics": values["factorized_attention_gru_seed_42"],
            "prediction_distribution": prediction_distribution(data[split].predictions),
            "high_bis_bias": high_bis_bias(data[split].predictions),
            "threshold_metrics": load_json(run_dir / f"{split}_metrics.json")[
                "pooled_window"
            ],
        }
    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_frame.to_csv(run_dir / "model_comparison.csv", index=False)
    model_comparison_json = {
        "models": ["persistence", "gru_seed_42", "factorized_attention_gru_seed_42"],
        "row_alignment_verified": True,
        "splits": comparison_payload,
        "test_patient_comparison": patient_payload["test"],
        "no_significance_test_performed": True,
    }
    dump_json(model_comparison_json, run_dir / "model_comparison.json")

    remifentanil_indices = [feature_names.index(name) for name in feature_names if name.startswith("rftn_")]
    missing_remifentanil: dict[str, Any] = {}
    paired_special_cases = {
        int(row["case_id"]): row
        for row in patient_payload["test"]["cases_97_and_154"]
    }
    for case_id in (97, 154):
        selector = data["test"].case_ids == case_id
        feature = data["test"].feature_attention[selector]
        combined = data["test"].combined_attention[selector]
        predictions = data["test"].predictions.loc[selector, "predicted_future_bis"]
        missing_remifentanil[str(case_id)] = {
            "window_count": int(selector.sum()),
            "remifentanil_feature_attention_always_zero": bool(
                np.count_nonzero(feature[:, :, remifentanil_indices]) == 0
            ),
            "predictions_finite": bool(np.isfinite(predictions).all()),
            "remaining_feature_attention_finite": bool(np.isfinite(feature).all()),
            "combined_attention_finite": bool(np.isfinite(combined).all()),
            **paired_special_cases[case_id],
        }

    history = pd.read_csv(run_dir / "training_history.csv")
    best_index = history["validation_patient_level_mae"].idxmin()
    best_epoch = int(history.loc[best_index, "epoch"])
    last = history.iloc[-1]
    best = history.loc[best_index]
    prediction_collapsed = any(
        performance[split]["prediction_distribution"]["predicted_standard_deviation"]
        < 0.5
        * performance[split]["prediction_distribution"]["observed_standard_deviation"]
        for split in SPLITS
    )
    normalization_valid = all(
        normalization[split][section].get(
            "time_steps_violating_tolerance_1e_5",
            normalization[split][section].get("rows_violating_tolerance_1e_5", 0),
        )
        == 0
        for split in SPLITS
        for section in (
            "feature_attention",
            "temporal_attention",
            "combined_attention",
        )
    )
    attention_degenerate = any(
        concentration[split]["possible_numerical_collapse"] for split in SPLITS
    )
    validation_difference = comparison_payload["val"]["differences"][
        "attention_minus_gru_seed_42"
    ]["patient_mean_mae"]
    proceed = (
        validation_difference <= 0.2
        and not prediction_collapsed
        and normalization_valid
        and not attention_degenerate
    )
    runtime = attention_metadata["runtime_breakdown"]
    runtime["total_command_wall_clock_seconds"] = command_wall_seconds
    categories = {
        "training": float(runtime["training_time_seconds"]),
        "per_epoch_validation": float(
            sum(runtime["validation_evaluation_time_per_epoch_seconds"])
        ),
        "final_joint_prediction_attention": float(
            sum(
                row["total_seconds"]
                for row in runtime["final_joint_prediction_attention_passes"].values()
            )
        ),
        "checkpoint_save": float(runtime["checkpoint_save_time_seconds"]),
        "checkpoint_load_reload": float(
            runtime["checkpoint_load_and_reload_verification_time_seconds"]
        ),
        "serialization": float(runtime["serialization_time_seconds"]),
    }
    runtime["dominant_measured_cost"] = max(categories, key=categories.get)
    runtime["measured_costs_seconds"] = categories

    audit = {
        "scope": {
            "seed": 42,
            "single_seed_diagnostic": True,
            "attention_not_causal": True,
            "test_attention_not_for_feature_selection_or_tuning": True,
            "static_context_note": STATIC_CONTEXT_NOTE,
        },
        "integrity": {
            "required_artifacts_present": True,
            "best_checkpoint_selected_by_validation_patient_mae_only": True,
            "test_not_used_for_checkpoint_selection": True,
            "validation_case_count": int(np.unique(data["val"].case_ids).size),
            "test_case_count": int(np.unique(data["test"].case_ids).size),
            "prediction_attention_metadata_alignment_verified": True,
            "feature_order_matches_dataset_metadata": True,
            "time_lag_order_seconds": list(TIME_LAGS),
            "checkpoint_prediction_reload_identical": bool(
                load_json(run_dir / "test_metrics.json")[
                    "checkpoint_reload_predictions_identical"
                ]
            ),
            "checkpoint_attention_reload_identical": bool(
                load_json(run_dir / "test_metrics.json")[
                    "checkpoint_reload_attention_identical"
                ]
            ),
        },
        "runtime": runtime,
        "training_diagnostics": {
            "epochs": history.to_dict(orient="records"),
            "completed_epochs": int(len(history)),
            "best_epoch": best_epoch,
            "early_stopping_epoch": int(last["epoch"]),
            "plateau_after_best": bool(int(last["epoch"]) > best_epoch),
            "training_loss_decreased_after_best": bool(last["train_loss"] < best["train_loss"]),
            "validation_patient_mae_worsened_after_best": bool(
                last["validation_patient_level_mae"]
                > best["validation_patient_level_mae"]
            ),
            "prediction_collapsed": prediction_collapsed,
            "attention_numerically_unstable": not normalization_valid,
        },
        "performance": performance,
        "comparison": model_comparison_json,
        "attention_normalization_and_mask_audit": normalization,
        "attention_concentration_and_collapse": concentration,
        "case_balanced_validation_top_features": validation_feature.head(18).to_dict(
            orient="records"
        ),
        "case_balanced_validation_time_lags": validation_time.to_dict(orient="records"),
        "validation_bis_region_attention": region_payload,
        "missing_remifentanil_test_cases": missing_remifentanil,
        "diagnostic_figures": {
            "label": DIAGNOSTIC_LABEL,
            "feature_bar_plot": "validation_feature_attention.png",
            "temporal_bar_plot": "validation_temporal_attention.png",
            "combined_heatmap": "validation_combined_attention_heatmap.png",
            "source_tables": [
                "feature_attention_summary.csv",
                "time_attention_summary.csv",
                "validation_combined_attention_heatmap.csv",
                "validation_bis_region_feature_attention.csv",
                "validation_bis_region_time_attention.csv",
            ],
        },
        "result_classification": {
            "category": (
                "PROCEED TO MULTI-SEED ATTENTION"
                if proceed
                else "REVIEW MODEL BEFORE MULTI-SEED"
            ),
            "validation_patient_mae_attention_minus_gru": float(
                validation_difference
            ),
            "validation_within_0_2_bis_points_of_gru": bool(
                validation_difference <= 0.2
            ),
            "prediction_distribution_not_collapsed": not prediction_collapsed,
            "attention_normalization_valid": normalization_valid,
            "attention_not_numerically_degenerate": not attention_degenerate,
            "test_did_not_determine_classification_or_tuning": True,
        },
    }
    dump_json(audit, run_dir / "attention_audit.json")
    return audit
