"""Validation-only predictive-contribution and attention-faithfulness audit."""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.attention_audit import case_balanced_feature_summary, load_split_attention
from src.datasets import VitalBISDataset
from src.metrics import bis_region_mae, regression_metrics
from src.models.attention import FactorizedAttentionGRU
from src.models.baselines import GRUBaseline
from src.multiseed_attention_audit import (
    DEFAULT_SEEDS,
    PairedRun,
    discover_complete_paired_runs,
)
from src.redundancy_audit import FEATURE_GROUPS, REDUCED_FEATURES

LOGGER = logging.getLogger(__name__)
BOOTSTRAP_SEED = 20260716
BOOTSTRAP_REPLICATES = 20_000
PERMUTATION_REPETITIONS = 10
NUMERICAL_NOISE_ATOL = 1e-6


def dump_json(payload: Mapping[str, Any], path: Path) -> None:
    """Write strict, deterministic JSON."""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")


def ablate_feature_arrays(
    values: np.ndarray, masks: np.ndarray, feature_indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return feature-unavailable arrays with normalized values and masks set to zero."""

    indices = np.asarray(feature_indices, dtype=np.int64)
    if values.shape != masks.shape or values.ndim != 3:
        raise ValueError("Values and masks must share shape [N, L, P].")
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= values.shape[2]):
        raise ValueError("Feature indices must be non-empty and in range.")
    ablated_values = values.copy()
    ablated_masks = masks.copy()
    ablated_values[:, :, indices] = 0
    ablated_masks[:, :, indices] = 0
    return ablated_values, ablated_masks


def ablate_named_group(
    values: np.ndarray,
    masks: np.ndarray,
    feature_names: Sequence[str],
    group_members: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply feature-unavailable ablation to an exact named feature group."""

    unknown = sorted(set(group_members) - set(feature_names))
    if unknown:
        raise ValueError(f"Unknown group members: {unknown}")
    return ablate_feature_arrays(
        values, masks, [feature_names.index(name) for name in group_members]
    )


def within_patient_circular_permutation(
    values: np.ndarray,
    masks: np.ndarray,
    case_ids: np.ndarray,
    feature_indices: Sequence[int],
    repetition: int,
    permutation_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Circularly reassign complete trajectories within each patient only."""

    if values.shape != masks.shape or len(values) != len(case_ids):
        raise ValueError("Permutation arrays and case IDs are not aligned.")
    permuted_values = values.copy()
    permuted_masks = masks.copy()
    feature_indices = tuple(int(index) for index in feature_indices)
    shifts: dict[int, int] = {}
    for case_id in np.unique(case_ids):
        rows = np.flatnonzero(case_ids == case_id)
        if len(rows) <= 1:
            shifts[int(case_id)] = 0
            continue
        shift = 1 + (
            permutation_seed + repetition * 104729 + int(case_id) * 1009
        ) % (len(rows) - 1)
        source_rows = np.roll(rows, int(shift))
        permuted_values[np.ix_(rows, np.arange(values.shape[1]), feature_indices)] = (
            values[np.ix_(source_rows, np.arange(values.shape[1]), feature_indices)]
        )
        permuted_masks[np.ix_(rows, np.arange(masks.shape[1]), feature_indices)] = (
            masks[np.ix_(source_rows, np.arange(masks.shape[1]), feature_indices)]
        )
        shifts[int(case_id)] = int(shift)
    return permuted_values, permuted_masks, shifts


def patient_equal_mae(
    y_true: np.ndarray, y_pred: np.ndarray, case_ids: np.ndarray
) -> float:
    """Return MAE calculated per patient and then averaged equally."""

    return float(
        np.mean(
            [
                np.mean(np.abs(y_pred[case_ids == case] - y_true[case_ids == case]))
                for case in np.unique(case_ids)
            ]
        )
    )


def patient_mae_differences(
    y_true: np.ndarray,
    original_prediction: np.ndarray,
    perturbed_prediction: np.ndarray,
    case_ids: np.ndarray,
) -> pd.DataFrame:
    """Return one paired perturbed-minus-original MAE difference per patient."""

    rows = []
    for case_id in np.unique(case_ids):
        selected = case_ids == case_id
        original = np.mean(np.abs(original_prediction[selected] - y_true[selected]))
        perturbed = np.mean(np.abs(perturbed_prediction[selected] - y_true[selected]))
        rows.append(
            {
                "case_id": int(case_id),
                "original_patient_mae": float(original),
                "perturbed_patient_mae": float(perturbed),
                "delta_patient_mae": float(perturbed - original),
            }
        )
    return pd.DataFrame(rows)


def patient_bootstrap_interval(
    patient_differences: np.ndarray,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Bootstrap a mean paired difference by resampling patients."""

    differences = np.asarray(patient_differences, dtype=float)
    if differences.ndim != 1 or not len(differences):
        raise ValueError("Patient differences must be a non-empty vector.")
    generator = np.random.default_rng(seed)
    draw = generator.integers(0, len(differences), size=(replicates, len(differences)))
    means = differences[draw].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return {
        "patient_count": int(len(differences)),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "mean_delta_patient_mae": float(differences.mean()),
        "percentile_95_ci_lower": float(lower),
        "percentile_95_ci_upper": float(upper),
    }


def spearman_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    """Calculate Spearman correlation using average ranks for ties."""

    first_rank = pd.Series(first, dtype=float).rank(method="average")
    second_rank = pd.Series(second, dtype=float).rank(method="average")
    return float(first_rank.corr(second_rank, method="pearson"))


def align_attention_contribution(
    attention: pd.DataFrame,
    contribution: pd.DataFrame,
    item_column: str,
) -> pd.DataFrame:
    """One-to-one align attention and contribution rows by seed and item."""

    required_attention = {"seed", item_column, "attention"}
    required_contribution = {"seed", item_column, "delta_patient_mae"}
    if not required_attention.issubset(attention) or not required_contribution.issubset(
        contribution
    ):
        raise ValueError("Attention or contribution columns are incomplete.")
    aligned = attention.merge(
        contribution,
        on=["seed", item_column],
        how="inner",
        validate="one_to_one",
    )
    expected = len(attention)
    if len(aligned) != expected or len(contribution) != expected:
        raise ValueError("Attention and contribution rows are not exactly aligned.")
    return aligned


def _prediction_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, case_ids: np.ndarray
) -> dict[str, float]:
    regression = regression_metrics(y_true, y_pred)
    regions = bis_region_mae(y_true, y_pred)
    return {
        "pooled_mae": float(regression["mae"]),
        "patient_mae": patient_equal_mae(y_true, y_pred, case_ids),
        "rmse": float(regression["rmse"]),
        "bis_below_40_mae": float(regions["bis_below_40"]),
        "bis_40_to_60_mae": float(regions["bis_40_to_60"]),
        "bis_above_60_mae": float(regions["bis_above_60"]),
    }


def _delta_metrics(original: Mapping[str, float], perturbed: Mapping[str, float]) -> dict[str, float]:
    return {
        f"delta_{name}": float(perturbed[name] - original[name])
        for name in original
    }


def _build_model(model_name: str, config: Mapping[str, Any]) -> torch.nn.Module:
    dynamic_count = len(config["dynamic_feature_names"])
    static_count = len(config["static_feature_names"])
    if model_name == "gru":
        return GRUBaseline(
            dynamic_feature_count=dynamic_count,
            static_feature_count=static_count,
            hidden_size=int(config["hidden_size"]),
            projection_size=int(config["projection_size"]),
            static_hidden_size=int(config["static_hidden_size"]),
            prediction_hidden_size=int(config["prediction_hidden_size"]),
            dropout=float(config["dropout"]),
        )
    if model_name == "attention":
        return FactorizedAttentionGRU(
            dynamic_feature_count=dynamic_count,
            static_feature_count=static_count,
            history_steps=6,
            feature_token_embedding_dim=int(config["feature_token_embedding_dim"]),
            static_context_dim=int(config["static_context_dim"]),
            hidden_size=int(config["hidden_size"]),
            prediction_hidden_size=int(config["prediction_hidden_size"]),
            dropout=float(config["dropout"]),
        )
    raise ValueError(f"Unsupported model: {model_name}")


@torch.no_grad()
def predict_arrays(
    model: torch.nn.Module,
    values: np.ndarray,
    static: np.ndarray,
    masks: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """Run exhaustive CPU inference over aligned in-memory arrays."""

    model.eval()
    predictions: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        output = model(
            torch.from_numpy(values[start:stop]).to(dtype=torch.float32),
            torch.from_numpy(static[start:stop]).to(dtype=torch.float32),
            torch.from_numpy(masks[start:stop]),
        )
        if not isinstance(output, torch.Tensor):
            output = output.prediction
        predictions.append(output.detach().cpu().numpy())
    return np.concatenate(predictions).astype(np.float32, copy=False)


def _load_model(model_name: str, run_dir: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    with (run_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    model = _build_model(model_name, config)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def _validate_baseline(
    prediction: np.ndarray,
    dataset: VitalBISDataset,
    run_dir: Path,
) -> dict[str, Any]:
    frame = pd.read_csv(run_dir / "val_predictions.csv")
    if not np.array_equal(frame["sample_index"].to_numpy(), np.arange(len(dataset))):
        raise ValueError(f"{run_dir} validation prediction alignment is not exhaustive.")
    expected_timestamp = dataset.metadata["target_timestamp"].to_numpy(dtype=np.int64)
    if not np.array_equal(frame["case_id"].to_numpy(dtype=np.int64), dataset.case_ids):
        raise ValueError(f"{run_dir} validation case IDs do not match the dataset.")
    if not np.array_equal(
        frame["target_timestamp"].to_numpy(dtype=np.int64), expected_timestamp
    ):
        raise ValueError(f"{run_dir} validation timestamps do not match the dataset.")
    if not np.allclose(
        frame["observed_future_bis"].to_numpy(),
        dataset.arrays["y_bis"],
        rtol=0.0,
        atol=5e-6,
    ):
        raise ValueError(f"{run_dir} validation targets do not match the dataset.")
    if not np.allclose(
        prediction, frame["predicted_future_bis"].to_numpy(), rtol=0.0, atol=1e-5
    ):
        raise AssertionError(f"{run_dir} checkpoint predictions were not reproduced.")
    calculated = _prediction_metrics(dataset.arrays["y_bis"], prediction, dataset.case_ids)
    with (run_dir / "val_metrics.json").open("r", encoding="utf-8") as handle:
        stored = json.load(handle)
    expected = {
        "pooled_mae": stored["pooled_window"]["regression"]["mae"],
        "patient_mae": stored["patient_level"]["mae"]["mean"],
        "rmse": stored["pooled_window"]["regression"]["rmse"],
    }
    for metric, value in expected.items():
        if not np.isclose(calculated[metric], value, rtol=0.0, atol=1e-5):
            raise AssertionError(f"{run_dir} reproduced {metric} differs from stored value.")
    return {
        "prediction_rows_aligned": True,
        "metadata_alignment_exact": True,
        "checkpoint_predictions_reproduced": True,
        "stored_metrics_reproduced": True,
        **calculated,
    }


def _add_ranks(frame: pd.DataFrame, item_column: str) -> pd.DataFrame:
    frame = frame.copy()
    if "repetition" in frame:
        seed_means = (
            frame.groupby(["model", "seed", item_column], as_index=False)[
                "delta_patient_mae"
            ].mean()
        )
        seed_means["contribution_rank"] = seed_means.groupby(
            ["model", "seed"]
        )["delta_patient_mae"].rank(method="average", ascending=False)
        return frame.merge(
            seed_means[["model", "seed", item_column, "contribution_rank"]],
            on=["model", "seed", item_column],
            validate="many_to_one",
        )
    frame["contribution_rank"] = frame.groupby(["model", "seed"])[
        "delta_patient_mae"
    ].rank(method="average", ascending=False)
    return frame


def contribution_stability_summary(
    frame: pd.DataFrame, analysis: str, item_column: str
) -> pd.DataFrame:
    """Summarize case-balanced contribution and rank stability across seeds."""

    seed_frame = (
        frame.groupby(["model", "seed", item_column], as_index=False)
        .agg(delta_patient_mae=("delta_patient_mae", "mean"))
    )
    seed_frame["rank"] = seed_frame.groupby(["model", "seed"])[
        "delta_patient_mae"
    ].rank(method="average", ascending=False)
    rows: list[dict[str, Any]] = []
    for model, model_frame in seed_frame.groupby("model"):
        pivot = model_frame.pivot(index=item_column, columns="seed", values="rank")
        correlations = [
            spearman_correlation(pivot[first], pivot[second])
            for first, second in combinations(pivot.columns, 2)
        ]
        mean_pairwise = float(np.mean(correlations))
        for item, item_frame in model_frame.groupby(item_column, sort=False):
            delta = item_frame["delta_patient_mae"].to_numpy(dtype=float)
            ranks = item_frame["rank"].to_numpy(dtype=float)
            positive = int((delta > 0).sum())
            mean_delta = float(delta.mean())
            rows.append(
                {
                    "analysis": analysis,
                    "model": model,
                    "item": item,
                    "mean_delta_patient_mae": mean_delta,
                    "sample_standard_deviation": float(delta.std(ddof=1)),
                    "minimum": float(delta.min()),
                    "maximum": float(delta.max()),
                    "positive_seed_count": positive,
                    "mean_rank": float(ranks.mean()),
                    "minimum_rank": float(ranks.min()),
                    "maximum_rank": float(ranks.max()),
                    "rank_range": float(ranks.max() - ranks.min()),
                    "mean_pairwise_spearman_rank_correlation": mean_pairwise,
                    "descriptively_stable": bool(
                        positive >= 4
                        and mean_delta > NUMERICAL_NOISE_ATOL
                    ),
                }
            )
    return pd.DataFrame(rows)


def _attention_faithfulness(
    attention_rows: pd.DataFrame,
    individual: pd.DataFrame,
    group: pd.DataFrame,
    permutation: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    permutation_means = permutation.groupby(
        ["model", "seed", "group"], as_index=False
    )["delta_patient_mae"].mean()
    for contribution_model in ("gru", "attention"):
        feature_contribution = individual.loc[
            individual["model"] == contribution_model,
            ["seed", "feature", "delta_patient_mae"],
        ]
        group_contribution = group.loc[
            group["model"] == contribution_model,
            ["seed", "group", "delta_patient_mae"],
        ]
        permutation_contribution = permutation_means.loc[
            permutation_means["model"] == contribution_model,
            ["seed", "group", "delta_patient_mae"],
        ]
        for seed in sorted(attention_rows["seed"].unique()):
            feature_attention = attention_rows.loc[
                (attention_rows["seed"] == seed) & (attention_rows["level"] == "feature"),
                ["seed", "item", "attention"],
            ].rename(columns={"item": "feature"})
            aligned_feature = align_attention_contribution(
                feature_attention,
                feature_contribution[feature_contribution["seed"] == seed],
                "feature",
            )
            attention_order = aligned_feature.nlargest(10, "attention")["feature"].tolist()
            contribution_order = aligned_feature.nlargest(10, "delta_patient_mae")[
                "feature"
            ].tolist()
            top5_intersection = len(set(attention_order[:5]) & set(contribution_order[:5]))
            top10_intersection = len(set(attention_order) & set(contribution_order))
            rows.append(
                {
                    "seed": int(seed),
                    "contribution_model": contribution_model,
                    "comparison": "feature_attention_vs_feature_unavailable_ablation",
                    "spearman_correlation": spearman_correlation(
                        aligned_feature["attention"], aligned_feature["delta_patient_mae"]
                    ),
                    "top_5_overlap_count": top5_intersection,
                    "top_5_jaccard": top5_intersection / (10 - top5_intersection),
                    "top_10_overlap_count": top10_intersection,
                    "top_10_jaccard": top10_intersection / (20 - top10_intersection),
                }
            )
            group_attention = attention_rows.loc[
                (attention_rows["seed"] == seed) & (attention_rows["level"] == "group"),
                ["seed", "item", "attention"],
            ].rename(columns={"item": "group"})
            for comparison, contribution in (
                ("group_attention_vs_group_unavailable_ablation", group_contribution),
                ("group_attention_vs_within_patient_permutation", permutation_contribution),
            ):
                aligned_group = align_attention_contribution(
                    group_attention,
                    contribution[contribution["seed"] == seed],
                    "group",
                )
                rows.append(
                    {
                        "seed": int(seed),
                        "contribution_model": contribution_model,
                        "comparison": comparison,
                        "spearman_correlation": spearman_correlation(
                            aligned_group["attention"],
                            aligned_group["delta_patient_mae"],
                        ),
                        "top_5_overlap_count": None,
                        "top_5_jaccard": None,
                        "top_10_overlap_count": None,
                        "top_10_jaccard": None,
                    }
                )
    result = pd.DataFrame(rows)
    result["mean_correlation_across_seeds"] = result.groupby(
        ["contribution_model", "comparison"]
    )["spearman_correlation"].transform("mean")
    result["sample_sd_correlation_across_seeds"] = result.groupby(
        ["contribution_model", "comparison"]
    )["spearman_correlation"].transform("std")
    result["positive_correlation_seed_count"] = result.groupby(
        ["contribution_model", "comparison"]
    )["spearman_correlation"].transform(lambda values: int((values > 0).sum()))
    return result


def _attention_rows(pairs: Sequence[PairedRun], dataset_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        attention = load_split_attention(
            pair.attention_dir, dataset_dir, "val", REDUCED_FEATURES
        )
        feature = case_balanced_feature_summary(attention, REDUCED_FEATURES)
        feature_values = feature.set_index("feature")["mean_feature_attention"]
        for name, value in feature_values.items():
            rows.append(
                {"seed": pair.seed, "level": "feature", "item": name, "attention": value}
            )
        for group, members in FEATURE_GROUPS.items():
            rows.append(
                {
                    "seed": pair.seed,
                    "level": "group",
                    "item": group,
                    "attention": float(feature_values[list(members)].sum()),
                }
            )
    return pd.DataFrame(rows)


def run_faithfulness_audit(
    root_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    batch_size: int = 512,
    permutation_repetitions: int = PERMUTATION_REPETITIONS,
) -> dict[str, Any]:
    """Run the complete validation-only audit using existing best checkpoints."""

    total_started = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = discover_complete_paired_runs(root_dir, seeds)
    dataset = VitalBISDataset(dataset_dir, "val", dynamic_features=REDUCED_FEATURES)
    expected_cases = sorted(dataset.metadata["case_id"].unique().astype(int).tolist())
    if len(expected_cases) != 15:
        raise ValueError(f"Expected 15 validation patients, found {len(expected_cases)}.")
    values = dataset.arrays["X_dynamic"]
    static = dataset.arrays["X_static"]
    masks = dataset.arrays["observation_mask"]
    y_true = dataset.arrays["y_bis"]
    case_ids = dataset.case_ids
    individual_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    patient_rows: list[pd.DataFrame] = []
    integrity: dict[str, Any] = {}
    timing = {"individual_ablation_seconds": 0.0, "group_ablation_seconds": 0.0,
              "group_permutation_seconds": 0.0}

    for pair in pairs:
        integrity[str(pair.seed)] = {}
        for model_name, run_dir in (("gru", pair.gru_dir), ("attention", pair.attention_dir)):
            LOGGER.info("Auditing %s seed %d", model_name, pair.seed)
            model, config = _load_model(model_name, run_dir)
            if tuple(config["dynamic_feature_names"]) != REDUCED_FEATURES:
                raise ValueError(f"{run_dir} does not contain the fixed 17 features.")
            if sorted(config["selected_validation_cases"]) != expected_cases:
                raise ValueError(f"{run_dir} validation patient set differs from the dataset.")
            original_prediction = predict_arrays(model, values, static, masks, batch_size)
            original_metrics = _prediction_metrics(y_true, original_prediction, case_ids)
            integrity[str(pair.seed)][model_name] = {
                **_validate_baseline(original_prediction, dataset, run_dir),
                "feature_order_exact": True,
                "bis_error_absent": True,
                "checkpoint_reloaded_once_for_analysis_block": True,
            }

            started = perf_counter()
            for feature_index, feature in enumerate(REDUCED_FEATURES):
                changed_values, changed_masks = ablate_feature_arrays(
                    values, masks, [feature_index]
                )
                prediction = predict_arrays(model, changed_values, static, changed_masks, batch_size)
                changed_metrics = _prediction_metrics(y_true, prediction, case_ids)
                individual_rows.append(
                    {
                        "model": model_name,
                        "seed": pair.seed,
                        "feature": feature,
                        "perturbation": "feature_unavailable_value_zero_mask_zero_all_six_steps",
                        **{f"original_{key}": value for key, value in original_metrics.items()},
                        **{f"perturbed_{key}": value for key, value in changed_metrics.items()},
                        **_delta_metrics(original_metrics, changed_metrics),
                    }
                )
            timing["individual_ablation_seconds"] += perf_counter() - started

            started = perf_counter()
            for group, members in FEATURE_GROUPS.items():
                changed_values, changed_masks = ablate_named_group(
                    values, masks, REDUCED_FEATURES, members
                )
                prediction = predict_arrays(model, changed_values, static, changed_masks, batch_size)
                changed_metrics = _prediction_metrics(y_true, prediction, case_ids)
                group_rows.append(
                    {
                        "model": model_name,
                        "seed": pair.seed,
                        "group": group,
                        "member_features": ",".join(members),
                        "perturbation": "group_unavailable_value_zero_mask_zero_all_six_steps",
                        **{f"original_{key}": value for key, value in original_metrics.items()},
                        **{f"perturbed_{key}": value for key, value in changed_metrics.items()},
                        **_delta_metrics(original_metrics, changed_metrics),
                    }
                )
                patient = patient_mae_differences(
                    y_true, original_prediction, prediction, case_ids
                )
                patient["analysis"] = "group_ablation"
                patient["model"] = model_name
                patient["seed"] = pair.seed
                patient["group"] = group
                patient["repetition"] = -1
                patient_rows.append(patient)
            timing["group_ablation_seconds"] += perf_counter() - started

            started = perf_counter()
            for group_index, (group, members) in enumerate(FEATURE_GROUPS.items()):
                indices = [REDUCED_FEATURES.index(name) for name in members]
                group_repetition_rows: list[dict[str, Any]] = []
                for repetition in range(permutation_repetitions):
                    changed_values, changed_masks, shifts = within_patient_circular_permutation(
                        values,
                        masks,
                        case_ids,
                        indices,
                        repetition,
                        permutation_seed=pair.seed * 100 + group_index,
                    )
                    prediction = predict_arrays(
                        model, changed_values, static, changed_masks, batch_size
                    )
                    changed_metrics = _prediction_metrics(y_true, prediction, case_ids)
                    row = {
                        "model": model_name,
                        "seed": pair.seed,
                        "group": group,
                        "member_features": ",".join(members),
                        "repetition": repetition,
                        "permutation": "within_patient_nonzero_circular_shift_complete_trajectory_and_mask",
                        "all_multiwindow_patient_shifts_nonzero": bool(
                            all(shift > 0 for shift in shifts.values())
                        ),
                        **_delta_metrics(original_metrics, changed_metrics),
                    }
                    group_repetition_rows.append(row)
                    patient = patient_mae_differences(
                        y_true, original_prediction, prediction, case_ids
                    )
                    patient["analysis"] = "group_permutation"
                    patient["model"] = model_name
                    patient["seed"] = pair.seed
                    patient["group"] = group
                    patient["repetition"] = repetition
                    patient_rows.append(patient)
                delta = np.asarray(
                    [row["delta_patient_mae"] for row in group_repetition_rows]
                )
                for row in group_repetition_rows:
                    row.update(
                        {
                            "seed_group_mean_delta_patient_mae": float(delta.mean()),
                            "seed_group_sample_sd_delta_patient_mae": float(delta.std(ddof=1)),
                            "seed_group_minimum_delta_patient_mae": float(delta.min()),
                            "seed_group_maximum_delta_patient_mae": float(delta.max()),
                        }
                    )
                permutation_rows.extend(group_repetition_rows)
            timing["group_permutation_seconds"] += perf_counter() - started

    aggregation_started = perf_counter()
    individual = _add_ranks(pd.DataFrame(individual_rows), "feature")
    group = _add_ranks(pd.DataFrame(group_rows), "group")
    permutation = _add_ranks(pd.DataFrame(permutation_rows), "group")
    summaries = pd.concat(
        (
            contribution_stability_summary(individual, "individual_feature_ablation", "feature"),
            contribution_stability_summary(group, "group_ablation", "group"),
            contribution_stability_summary(permutation, "group_permutation", "group"),
        ),
        ignore_index=True,
    )
    attention_rows = _attention_rows(pairs, dataset_dir)
    faithfulness = _attention_faithfulness(attention_rows, individual, group, permutation)
    patient_frame = pd.concat(patient_rows, ignore_index=True)
    bootstrap_rows = []
    for (analysis, model, group_name), frame in patient_frame.groupby(
        ["analysis", "model", "group"]
    ):
        patient_mean = frame.groupby("case_id")["delta_patient_mae"].mean()
        bootstrap_rows.append(
            {
                "analysis": analysis,
                "model": model,
                "group": group_name,
                "resampling_unit": "validation_patient",
                **patient_bootstrap_interval(patient_mean.to_numpy()),
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)

    stable_non_bis = summaries[
        summaries["analysis"].isin(["group_ablation", "group_permutation"])
        & (summaries["item"] != "current_bis")
        & summaries["descriptively_stable"]
    ]
    own_attention = faithfulness[faithfulness["contribution_model"] == "attention"]
    consistently_positive = bool(
        (own_attention["positive_correlation_seed_count"] >= 4).all()
    )
    group_rank_ranges = summaries.loc[
        summaries["analysis"].isin(["group_ablation", "group_permutation"]),
        "rank_range",
    ]
    if consistently_positive and bool((group_rank_ranges == 0).all()):
        classification = "ATTENTION SUPPORTED FOR FEATURE SELECTION"
    elif not stable_non_bis.empty:
        classification = "USE CONTRIBUTION-BASED SELECTION, NOT RAW ATTENTION"
    else:
        classification = "NO STABLE FEATURE REDUCTION EVIDENCE"
    timing["faithfulness_aggregation_seconds"] = perf_counter() - aggregation_started
    timing["total_faithfulness_audit_seconds"] = perf_counter() - total_started

    individual.to_csv(output_dir / "individual_feature_ablation.csv", index=False)
    group.to_csv(output_dir / "group_ablation.csv", index=False)
    permutation.to_csv(output_dir / "group_permutation.csv", index=False)
    summaries.to_csv(output_dir / "contribution_summary.csv", index=False)
    faithfulness.to_csv(output_dir / "attention_faithfulness.csv", index=False)
    bootstrap.to_csv(output_dir / "patient_bootstrap_summary.csv", index=False)
    audit = {
        "scope": {
            "validation_only": True,
            "test_labels_or_test_importance_accessed": False,
            "model_training_or_fine_tuning_performed": False,
            "top_k_selection_performed": False,
            "interpretation": "Predictive association diagnostic; not causal evidence.",
        },
        "configuration": {
            "seeds": list(seeds),
            "dynamic_features": list(REDUCED_FEATURES),
            "feature_groups": {key: list(value) for key, value in FEATURE_GROUPS.items()},
            "validation_patient_ids": expected_cases,
            "validation_window_count": len(dataset),
            "permutation_repetitions": permutation_repetitions,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "inference_batch_size": batch_size,
        },
        "run_integrity": integrity,
        "timing": timing,
        "attention_faithfulness_summary": own_attention.groupby("comparison")[
            "spearman_correlation"
        ].agg(["mean", "std", "min", "max"]).to_dict(orient="index"),
        "descriptively_stable_non_bis_contributions": stable_non_bis.to_dict(orient="records"),
        "evidence_classification": classification,
        "classification_note": (
            "Descriptive stability uses the requested positive-in-four-of-five rule and "
            "a 1e-6 numerical-equality tolerance, not a significance threshold. The "
            "attention-supported branch requires identical group ranks. No selected "
            "feature set was created."
        ),
    }
    dump_json(audit, output_dir / "faithfulness_audit.json")
    return audit
