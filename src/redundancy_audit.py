"""Legacy exploratory diagnostic for the physiological-inclusive feature universe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.attention_audit import (
    COMPARISON_METRICS,
    SPLITS,
    align_prediction_rows,
    attention_concentration,
    attention_normalization_audit,
    case_balanced_feature_summary,
    case_balanced_time_summary,
    dump_json,
    high_bis_bias,
    load_json,
    load_split_attention,
    metric_values,
)
from src.datasets import VitalBISDataset

REDUCED_FEATURES = (
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
)
FEATURE_GROUPS = {
    "current_bis": ("bis",),
    "bis_quality_and_dynamics": ("bis_sqi", "bis_slope"),
    "propofol": ("ppf_rate", "ppf_volume", "ppf_cp", "ppf_ce"),
    "remifentanil": ("rftn_rate", "rftn_volume", "rftn_cp", "rftn_ce"),
    "hemodynamic": ("hr", "mbp", "sbp", "dbp"),
    "respiratory": ("spo2", "etco2"),
}
MODEL_NAMES = (
    "original_gru_18",
    "original_attention_18",
    "reduced_gru_17",
    "reduced_attention_17",
)
SCIENTIFIC_ROLE = "legacy_physiological_exploratory_not_final_selection"


def verify_bis_error_redundancy(dataset_dir: Path) -> dict[str, Any]:
    """Inverse-transform BIS columns and verify ``bis_error == bis - 50``."""

    metadata = load_json(dataset_dir / "dataset_metadata.json")
    names = list(metadata["dynamic_feature_names"])
    bis_index = names.index("bis")
    error_index = names.index("bis_error")
    statistics = pd.read_csv(
        dataset_dir / "preprocessing_statistics.csv"
    ).set_index("feature_name")
    bis_mean = float(statistics.loc["bis", "training_mean"])
    bis_scale = float(statistics.loc["bis", "normalization_scale"])
    error_mean = float(statistics.loc["bis_error", "training_mean"])
    error_scale = float(statistics.loc["bis_error", "normalization_scale"])
    all_bis: list[np.ndarray] = []
    all_error: list[np.ndarray] = []
    split_results: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        with np.load(dataset_dir / f"{split}.npz", allow_pickle=False) as archive:
            dynamic = archive["X_dynamic"]
        bis = dynamic[..., bis_index].astype(np.float64) * bis_scale + bis_mean
        bis_error = (
            dynamic[..., error_index].astype(np.float64) * error_scale + error_mean
        )
        direct_error = np.abs(bis_error - (bis - 50.0))
        split_results[split] = {
            "maximum_absolute_reconstruction_error": float(direct_error.max()),
            "correlation_original_unit_bis_and_bis_error": float(
                np.corrcoef(bis.ravel(), bis_error.ravel())[0, 1]
            ),
            "value_count": int(bis.size),
        }
        all_bis.append(bis.ravel())
        all_error.append(bis_error.ravel())
    combined_bis = np.concatenate(all_bis)
    combined_error = np.concatenate(all_error)
    maximum_error = max(
        row["maximum_absolute_reconstruction_error"]
        for row in split_results.values()
    )
    return {
        "construction_equation_in_original_units": "bis_error = bis - 50.0",
        "verification_method": (
            "inverse transform with saved training means/scales, then direct equation"
        ),
        "splits": split_results,
        "maximum_absolute_reconstruction_error": float(maximum_error),
        "correlation_original_unit_bis_and_bis_error": float(
            np.corrcoef(combined_bis, combined_error)[0, 1]
        ),
        "deterministic_within_numerical_precision": bool(maximum_error <= 1e-10),
        "correlation_not_used_as_sole_proof": True,
    }


def _training_runtime(run_dir: Path) -> tuple[float | None, str]:
    runtime_path = run_dir / "runtime.json"
    if runtime_path.exists():
        return (
            float(load_json(runtime_path)["total_internal_runtime_seconds"]),
            "runtime.json total_internal_runtime_seconds",
        )
    attention_path = run_dir / "attention_metadata.json"
    if attention_path.exists():
        runtime = load_json(attention_path)["runtime_breakdown"]
        return (
            float(runtime["total_internal_runtime_seconds"]),
            "attention_metadata.json total_internal_runtime_seconds",
        )
    return None, "not recorded by the original training implementation"


def _case_balanced_group_summary(
    feature_attention: np.ndarray,
    case_ids: np.ndarray,
    feature_names: list[str],
    model: str,
) -> pd.DataFrame:
    sample_feature = feature_attention.mean(axis=1)
    rows: list[dict[str, Any]] = []
    for group, members in FEATURE_GROUPS.items():
        indices = [feature_names.index(name) for name in members]
        sample_group = sample_feature[:, indices].sum(axis=1)
        case_values = np.asarray(
            [sample_group[case_ids == case].mean() for case in np.unique(case_ids)]
        )
        rows.append(
            {
                "model": model,
                "group": group,
                "member_features": ",".join(members),
                "case_balanced_mean_attention": float(case_values.mean()),
                "standard_deviation_across_cases": float(
                    case_values.std(ddof=1) if len(case_values) > 1 else 0.0
                ),
                "median_across_cases": float(np.median(case_values)),
            }
        )
    return pd.DataFrame(rows)


def _paired_case_comparison(
    first: pd.DataFrame,
    second: pd.DataFrame,
    first_name: str,
    second_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    paired = first[["case_id", "mae"]].rename(columns={"mae": f"{first_name}_mae"})
    paired = paired.merge(
        second[["case_id", "mae"]].rename(
            columns={"mae": f"{second_name}_mae"}
        ),
        on="case_id",
        validate="one_to_one",
    )
    difference_name = f"{first_name}_minus_{second_name}_mae"
    paired[difference_name] = paired[f"{first_name}_mae"] - paired[f"{second_name}_mae"]
    return {
        "case_count": int(len(paired)),
        "first_model_win_count": int((paired[difference_name] < 0).sum()),
        "median_paired_mae_difference": float(paired[difference_name].median()),
        "five_largest_improvements": paired.nsmallest(5, difference_name).to_dict(
            orient="records"
        ),
        "five_largest_deteriorations": paired.nlargest(5, difference_name).to_dict(
            orient="records"
        ),
    }, paired


def _bias_change_label(original_bias: float, reduced_bias: float) -> str:
    change = abs(reduced_bias) - abs(original_bias)
    if change < -0.05:
        return "improves"
    if change > 0.05:
        return "worsens"
    return "leaves essentially unchanged"


def run_bis_error_ablation_audit(
    *,
    dataset_dir: Path,
    output_dir: Path,
    original_gru_dir: Path,
    original_attention_dir: Path,
    reduced_gru_dir: Path,
    reduced_attention_dir: Path,
) -> dict[str, Any]:
    """Audit four aligned models and attention redistribution after exclusion."""

    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = {
        "original_gru_18": original_gru_dir,
        "original_attention_18": original_attention_dir,
        "reduced_gru_17": reduced_gru_dir,
        "reduced_attention_17": reduced_attention_dir,
    }
    configs = {name: load_json(path / "config.json") for name, path in run_dirs.items()}
    if tuple(configs["reduced_gru_17"]["dynamic_feature_names"]) != REDUCED_FEATURES:
        raise ValueError("Reduced GRU feature order does not match the required 17 features.")
    if tuple(configs["reduced_attention_17"]["dynamic_feature_names"]) != REDUCED_FEATURES:
        raise ValueError(
            "Reduced attention feature order does not match the required 17 features."
        )

    predictions = {
        split: {
            name: pd.read_csv(path / f"{split}_predictions.csv")
            for name, path in run_dirs.items()
        }
        for split in SPLITS
    }
    for split in SPLITS:
        reference = predictions[split]["original_gru_18"]
        for name in MODEL_NAMES[1:]:
            align_prediction_rows(reference, predictions[split][name], split=split, candidate_name=name)

    runtimes = {name: _training_runtime(path) for name, path in run_dirs.items()}
    comparison_rows: list[dict[str, Any]] = []
    split_payload: dict[str, Any] = {}
    case_metrics: dict[str, dict[str, pd.DataFrame]] = {}
    contrasts = {
        "reduced_gru_minus_original_gru": ("reduced_gru_17", "original_gru_18"),
        "reduced_attention_minus_original_attention": (
            "reduced_attention_17",
            "original_attention_18",
        ),
        "reduced_attention_minus_reduced_gru": (
            "reduced_attention_17",
            "reduced_gru_17",
        ),
    }
    for split in SPLITS:
        values: dict[str, dict[str, float]] = {}
        case_metrics[split] = {}
        for name in MODEL_NAMES:
            values[name], case_metrics[split][name] = metric_values(
                predictions[split][name]
            )
            runtime, runtime_source = runtimes[name]
            comparison_rows.append(
                {
                    "split": split,
                    "model": name,
                    **{metric: values[name][metric] for metric in COMPARISON_METRICS},
                    "parameter_count": int(configs[name]["model_parameter_count"]),
                    "training_runtime_seconds": runtime,
                    "runtime_source": runtime_source,
                }
            )
        differences: dict[str, Any] = {}
        for contrast, (first, second) in contrasts.items():
            difference = {
                metric: values[first][metric] - values[second][metric]
                for metric in COMPARISON_METRICS
            }
            differences[contrast] = difference
            comparison_rows.append(
                {
                    "split": split,
                    "model": contrast,
                    **difference,
                    "parameter_count": (
                        int(configs[first]["model_parameter_count"])
                        - int(configs[second]["model_parameter_count"])
                    ),
                    "training_runtime_seconds": None,
                    "runtime_source": "difference row",
                }
            )
        split_payload[split] = {"models": values, "contrasts": differences}

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "comparison.csv", index=False)

    paired_payload: dict[str, Any] = {}
    for split in SPLITS:
        reduced_vs_gru, paired_gru = _paired_case_comparison(
            case_metrics[split]["reduced_attention_17"],
            case_metrics[split]["reduced_gru_17"],
            "reduced_attention",
            "reduced_gru",
        )
        reduced_vs_original, paired_original = _paired_case_comparison(
            case_metrics[split]["reduced_attention_17"],
            case_metrics[split]["original_attention_18"],
            "reduced_attention",
            "original_attention",
        )
        paired_payload[split] = {
            "reduced_attention_versus_reduced_gru": reduced_vs_gru,
            "reduced_attention_versus_original_attention": reduced_vs_original,
        }
        if split == "test":
            paired_gru.merge(paired_original, on="case_id").to_csv(
                output_dir / "test_patient_comparison.csv", index=False
            )

    original_attention = load_split_attention(
        original_attention_dir,
        dataset_dir,
        "val",
        configs["original_attention_18"]["dynamic_feature_names"],
    )
    reduced_attention = load_split_attention(
        reduced_attention_dir, dataset_dir, "val", REDUCED_FEATURES
    )
    original_feature = case_balanced_feature_summary(
        original_attention, configs["original_attention_18"]["dynamic_feature_names"]
    )
    reduced_feature = case_balanced_feature_summary(reduced_attention, REDUCED_FEATURES)
    reduced_time = case_balanced_time_summary(reduced_attention, (-50, -40, -30, -20, -10, 0))
    reduced_feature.to_csv(output_dir / "validation_feature_attention_17.csv", index=False)
    reduced_time.to_csv(output_dir / "validation_time_attention_17.csv", index=False)

    original_groups = _case_balanced_group_summary(
        original_attention.feature_attention,
        original_attention.case_ids,
        list(configs["original_attention_18"]["dynamic_feature_names"]),
        "original_attention_18",
    )
    reduced_groups = _case_balanced_group_summary(
        reduced_attention.feature_attention,
        reduced_attention.case_ids,
        list(REDUCED_FEATURES),
        "reduced_attention_17",
    )
    group_summary = pd.concat((original_groups, reduced_groups), ignore_index=True)
    group_summary.to_csv(output_dir / "validation_feature_group_attention.csv", index=False)
    group_pivot = group_summary.pivot(
        index="group", columns="model", values="case_balanced_mean_attention"
    )
    group_pivot["reduced_minus_original"] = (
        group_pivot["reduced_attention_17"] - group_pivot["original_attention_18"]
    )

    original_weights = original_feature.set_index("feature")["mean_feature_attention"]
    reduced_weights = reduced_feature.set_index("feature")["mean_feature_attention"]
    concentration = {
        "original_attention_18": attention_concentration(original_attention),
        "reduced_attention_17": attention_concentration(reduced_attention),
    }
    normalization = attention_normalization_audit(reduced_attention)
    attention_valid = bool(
        all(
            section["non_negative"]
            for section in normalization.values()
        )
        and normalization["feature_attention"]["time_steps_violating_tolerance_1e_5"] == 0
        and normalization["temporal_attention"]["rows_violating_tolerance_1e_5"] == 0
        and normalization["combined_attention"]["rows_violating_tolerance_1e_5"] == 0
    )
    redistribution = {
        "current_bis_change": float(reduced_weights["bis"] - original_weights["bis"]),
        "bis_slope_change": float(
            reduced_weights["bis_slope"] - original_weights["bis_slope"]
        ),
        "group_changes": group_pivot["reduced_minus_original"].to_dict(),
        "vital_sign_change_hemodynamic_plus_respiratory": float(
            group_pivot.loc["hemodynamic", "reduced_minus_original"]
            + group_pivot.loc["respiratory", "reduced_minus_original"]
        ),
        "attention_is_descriptive_not_causal": True,
    }

    high_bis: dict[str, Any] = {}
    for reduced_name, original_name in (
        ("reduced_gru_17", "original_gru_18"),
        ("reduced_attention_17", "original_attention_18"),
    ):
        split_diagnostics: dict[str, Any] = {}
        for split in SPLITS:
            reduced_bias = high_bis_bias(predictions[split][reduced_name])
            original_bias = high_bis_bias(predictions[split][original_name])
            split_diagnostics[split] = {
                **reduced_bias,
                "bis_above_60_mae": split_payload[split]["models"][reduced_name][
                    "bis_above_60_mae"
                ],
                "high_bis_auprc": split_payload[split]["models"][reduced_name][
                    "high_bis_auprc"
                ],
                "high_bis_auroc": split_payload[split]["models"][reduced_name][
                    "high_bis_auroc"
                ],
                "original_model_bias": original_bias,
                "underprediction_assessment": _bias_change_label(
                    float(
                        original_bias[
                            "mean_prediction_bias_predicted_minus_observed"
                        ]
                    ),
                    float(
                        reduced_bias[
                            "mean_prediction_bias_predicted_minus_observed"
                        ]
                    ),
                ),
            }
        high_bis[reduced_name] = split_diagnostics

    reduced_test_dataset = VitalBISDataset(
        dataset_dir, "test", dynamic_features=REDUCED_FEATURES
    )
    remifentanil_indices = [
        REDUCED_FEATURES.index(name)
        for name in REDUCED_FEATURES
        if name.startswith("rftn_")
    ]
    reduced_test_attention = load_split_attention(
        reduced_attention_dir, dataset_dir, "test", REDUCED_FEATURES
    )
    special_cases: dict[str, Any] = {}
    for case_id in (97, 154):
        sample_indices = reduced_test_dataset.indices_for_cases([case_id])
        attention_selector = reduced_test_attention.case_ids == case_id
        model_rows: dict[str, Any] = {}
        for name in MODEL_NAMES:
            case_row = case_metrics["test"][name].loc[
                case_metrics["test"][name]["case_id"] == case_id
            ]
            model_rows[name] = {
                "mae": float(case_row.iloc[0]["mae"]),
                "predictions_finite": bool(
                    np.isfinite(
                        predictions["test"][name].loc[
                            predictions["test"][name]["case_id"] == case_id,
                            "predicted_future_bis",
                        ]
                    ).all()
                ),
                "attention_values_applicable": "attention" in name,
            }
        special_cases[str(case_id)] = {
            "window_count": int(len(sample_indices)),
            "all_remifentanil_masks_zero": bool(
                ~reduced_test_dataset.arrays["observation_mask"][
                    sample_indices, :, :
                ][:, :, remifentanil_indices].any()
            ),
            "reduced_attention_values_finite": bool(
                np.isfinite(
                    reduced_test_attention.feature_attention[attention_selector]
                ).all()
                and np.isfinite(
                    reduced_test_attention.temporal_attention[attention_selector]
                ).all()
                and np.isfinite(
                    reduced_test_attention.combined_attention[attention_selector]
                ).all()
            ),
            "models": model_rows,
        }

    gru_validation_delta = split_payload["val"]["contrasts"][
        "reduced_gru_minus_original_gru"
    ]["patient_mean_mae"]
    attention_validation_delta = split_payload["val"]["contrasts"][
        "reduced_attention_minus_original_attention"
    ]["patient_mean_mae"]
    permanently_remove = bool(
        gru_validation_delta <= 0.05
        and attention_validation_delta <= 0.05
        and attention_valid
    )
    decision = {
        "operational_threshold_bis_points": 0.05,
        "reduced_gru_validation_patient_mae_change": float(gru_validation_delta),
        "reduced_attention_validation_patient_mae_change": float(
            attention_validation_delta
        ),
        "reduced_attention_outputs_valid": attention_valid,
        "recommend_permanently_removing_bis_error_from_prediction_features": (
            permanently_remove
        ),
        "bis_error_may_be_reintroduced_to_rl_policy_separately": permanently_remove,
        "bis_error_must_not_count_as_independently_selected_predictive_feature": True,
        "criterion_is_operational_not_inferential": True,
    }
    comparison_json = {
        "models": list(MODEL_NAMES),
        "row_alignment_verified": True,
        "splits": split_payload,
        "patient_level_comparisons": paired_payload,
        "parameter_counts": {
            name: int(config["model_parameter_count"])
            for name, config in configs.items()
        },
        "training_runtimes": {
            name: {"seconds": runtime, "source": source}
            for name, (runtime, source) in runtimes.items()
        },
        "difference_direction": (
            "negative favors the first named model for errors; positive favors it "
            "for discrimination"
        ),
        "no_inferential_significance_testing": True,
    }
    dump_json(comparison_json, output_dir / "comparison.json")
    audit = {
        "redundancy_verification": verify_bis_error_redundancy(dataset_dir),
        "comparison": comparison_json,
        "reduced_validation_feature_ranking": reduced_feature.to_dict(
            orient="records"
        ),
        "reduced_validation_temporal_ranking": reduced_time.to_dict(
            orient="records"
        ),
        "validation_group_attention": group_summary.to_dict(orient="records"),
        "attention_redistribution": redistribution,
        "attention_concentration": concentration,
        "reduced_attention_normalization": normalization,
        "high_bis_diagnostic": high_bis,
        "cases_97_and_154": special_cases,
        "decision": decision,
    }
    dump_json(audit, output_dir / "redundancy_audit.json")
    return audit
