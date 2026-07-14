"""Audit saved future-BIS dataset artifacts without modifying model arrays."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io import load_cleaned_data  # noqa: E402
from src.preprocessing import (  # noqa: E402
    add_derived_features,
    resample_cases,
    resolve_feature_specs,
)

LOGGER = logging.getLogger(__name__)
SPLIT_NAMES = ("train", "val", "test")


def _float(value: float | np.floating[Any]) -> float:
    return float(value)


def _numeric_summary(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    q1, median, q3 = np.percentile(array, [25, 50, 75])
    return {
        "count": int(array.size),
        "mean": _float(array.mean()),
        "standard_deviation": _float(array.std(ddof=0)),
        "median": _float(median),
        "q1": _float(q1),
        "q3": _float(q3),
        "interquartile_range": _float(q3 - q1),
        "minimum": _float(array.min()),
        "maximum": _float(array.max()),
    }


def _target_summary(targets: np.ndarray) -> dict[str, float | int]:
    summary = _numeric_summary(targets)
    summary.update(
        {
            "percentage_bis_below_40": _float(np.mean(targets < 40.0) * 100.0),
            "percentage_bis_40_to_60": _float(
                np.mean((targets >= 40.0) & (targets <= 60.0)) * 100.0
            ),
            "percentage_bis_above_60": _float(np.mean(targets > 60.0) * 100.0),
        }
    )
    return summary


def _window_candidate_counts(
    case_frame: pd.DataFrame,
    history_steps: int,
    interval_seconds: int,
    horizon_seconds: int,
) -> dict[str, int]:
    """Classify nominal endpoints after enough history exists in a case."""

    timestamps = set(int(value) for value in case_frame["timestamp"])
    bis_by_time = case_frame.set_index("timestamp")["bis"]
    first_candidate = min(timestamps) + interval_seconds * (history_steps - 1)
    candidate_endpoints = sorted(value for value in timestamps if value >= first_candidate)
    gap_excluded = 0
    future_excluded = 0
    included = 0
    for final_time in candidate_endpoints:
        history_times = [
            final_time - interval_seconds * offset for offset in range(history_steps)
        ]
        if not all(timestamp in timestamps for timestamp in history_times):
            gap_excluded += 1
            continue
        target_time = final_time + horizon_seconds
        if target_time not in timestamps or pd.isna(bis_by_time.loc[target_time]):
            future_excluded += 1
            continue
        included += 1
    return {
        "candidate_windows": len(candidate_endpoints),
        "included_windows": included,
        "excluded_history_gap": gap_excluded,
        "excluded_unavailable_future_bis": future_excluded,
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dump_json(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def _split_case_ids(dataset_dir: Path) -> dict[str, set[int]]:
    return {
        split: set(
            pd.read_csv(dataset_dir / "splits" / f"{split}_cases.csv")["caseid"].astype(int)
        )
        for split in SPLIT_NAMES
    }


def _case_level_summary(
    split: str,
    case_ids: set[int],
    metadata: pd.DataFrame,
    targets: np.ndarray,
) -> pd.DataFrame:
    target_frame = metadata.copy()
    target_frame["target_bis"] = targets
    grouped = target_frame.groupby("case_id", sort=True)
    summary = grouped.agg(
        number_of_windows=("target_bis", "size"),
        mean_target_bis=("target_bis", "mean"),
        first_target_timestamp=("target_timestamp", "min"),
        last_target_timestamp=("target_timestamp", "max"),
    )
    summary["bis_below_40_prevalence"] = grouped["target_bis"].apply(
        lambda values: float((values < 40.0).mean())
    )
    summary["bis_40_to_60_prevalence"] = grouped["target_bis"].apply(
        lambda values: float(((values >= 40.0) & (values <= 60.0)).mean())
    )
    summary["bis_above_60_prevalence"] = grouped["target_bis"].apply(
        lambda values: float((values > 60.0).mean())
    )
    summary = summary.reindex(sorted(case_ids))
    if summary["number_of_windows"].isna().any():
        missing = summary.index[summary["number_of_windows"].isna()].tolist()
        raise AssertionError(f"Cases have no saved windows in {split}: {missing}")
    summary.insert(0, "split", split)
    summary.index.name = "case_id"
    return summary.reset_index()


def _case_event_coverage(case_summary: pd.DataFrame) -> dict[str, float | int]:
    n_cases = len(case_summary)
    high = case_summary["bis_above_60_prevalence"] > 0.0
    low = case_summary["bis_below_40_prevalence"] > 0.0
    return {
        "case_count": n_cases,
        "cases_with_bis_above_60": int(high.sum()),
        "percentage_cases_with_bis_above_60": _float(high.mean() * 100.0),
        "cases_with_bis_below_40": int(low.sum()),
        "percentage_cases_with_bis_below_40": _float(low.mean() * 100.0),
        "cases_without_bis_above_60": int((~high).sum()),
        "cases_without_bis_below_40": int((~low).sum()),
        "median_per_case_bis_above_60_prevalence": _float(
            case_summary["bis_above_60_prevalence"].median()
        ),
        "median_per_case_bis_below_40_prevalence": _float(
            case_summary["bis_below_40_prevalence"].median()
        ),
    }


def _smd(first: np.ndarray, second: np.ndarray) -> float:
    pooled = np.sqrt((first.var(ddof=0) + second.var(ddof=0)) / 2.0)
    return _float((first.mean() - second.mean()) / pooled) if pooled > 0.0 else 0.0


def _feature_values(
    frame: pd.DataFrame, feature_name: str, feature_class: str
) -> pd.Series:
    if feature_class == "static":
        return frame[["caseid", feature_name]].drop_duplicates("caseid")[feature_name]
    return frame[feature_name]


def _missingness_audit(
    resampled: pd.DataFrame,
    split_ids: dict[str, set[int]],
    feature_manifest: pd.DataFrame,
    preprocessing_stats: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stats_by_feature = preprocessing_stats.set_index("feature_name")
    features: dict[str, Any] = {}
    flags: dict[str, list[Any]] = {
        "over_30_percent_missing_in_any_split": [],
        "entirely_missing_in_validation_or_test_cases": [],
        "near_zero_training_variance": [],
        "validation_or_test_far_outside_training_distribution": [],
    }

    included = feature_manifest[feature_manifest["included"].astype(bool)]
    for row in included.itertuples(index=False):
        name = row.standardized_feature_name
        feature_class = row.dynamic_or_static
        train_frame = resampled[resampled["caseid"].isin(split_ids["train"])]
        train_values = pd.to_numeric(
            _feature_values(train_frame, name, feature_class), errors="coerce"
        ).dropna()
        train_min = _float(train_values.min())
        train_max = _float(train_values.max())
        stats = stats_by_feature.loc[name]
        train_mean = _float(stats.training_mean)
        train_std = _float(stats.training_standard_deviation)
        near_zero = train_std < 1e-8
        if near_zero:
            flags["near_zero_training_variance"].append(name)

        split_details: dict[str, Any] = {}
        for split, ids in split_ids.items():
            split_frame = resampled[resampled["caseid"].isin(ids)]
            values = pd.to_numeric(
                _feature_values(split_frame, name, feature_class), errors="coerce"
            )
            entirely_missing = [
                int(case_id)
                for case_id, case_frame in split_frame.groupby("caseid", sort=True)
                if not case_frame[name].notna().any()
            ]
            observed = values.dropna().to_numpy(dtype=float)
            outside_range = (observed < train_min) | (observed > train_max)
            if near_zero:
                far_outside = np.abs(observed - train_mean) > 1e-8
                max_abs_z = None
            else:
                z_scores = np.abs((observed - train_mean) / train_std)
                far_outside = z_scores > 5.0
                max_abs_z = _float(z_scores.max()) if len(z_scores) else None
            details = {
                "pre_imputation_missing_percentage": _float(values.isna().mean() * 100.0),
                "entirely_missing_case_count": len(entirely_missing),
                "entirely_missing_case_ids": entirely_missing,
                "observed_minimum": _float(observed.min()) if len(observed) else None,
                "observed_maximum": _float(observed.max()) if len(observed) else None,
                "percentage_outside_training_observed_range": (
                    _float(outside_range.mean() * 100.0) if len(observed) else None
                ),
                "far_outside_training_distribution_count": int(far_outside.sum()),
                "far_outside_training_distribution_percentage": (
                    _float(far_outside.mean() * 100.0) if len(observed) else None
                ),
                "maximum_absolute_training_z_score": max_abs_z,
            }
            split_details[split] = details
            if details["pre_imputation_missing_percentage"] > 30.0:
                flags["over_30_percent_missing_in_any_split"].append(
                    {"feature": name, "split": split, "percentage": details["pre_imputation_missing_percentage"]}
                )
            if split in {"val", "test"} and entirely_missing:
                flags["entirely_missing_in_validation_or_test_cases"].append(
                    {"feature": name, "split": split, "case_ids": entirely_missing}
                )
            if split in {"val", "test"} and int(far_outside.sum()) > 0:
                flags["validation_or_test_far_outside_training_distribution"].append(
                    {
                        "feature": name,
                        "split": split,
                        "count": int(far_outside.sum()),
                        "percentage": details["far_outside_training_distribution_percentage"],
                        "maximum_absolute_training_z_score": max_abs_z,
                    }
                )

        features[name] = {
            "feature_class": feature_class,
            "training_median_used_for_imputation": _float(stats.training_median),
            "training_mean_used_for_normalization": train_mean,
            "training_standard_deviation_used_for_normalization": train_std,
            "training_observed_minimum": train_min,
            "training_observed_maximum": train_max,
            "splits": split_details,
        }
    return features, flags


def audit_dataset(
    dataset_dir: Path,
    input_path: Path,
    build_runtime_seconds: float | None,
) -> dict[str, Any]:
    """Run the complete full-dataset audit and write its requested artifacts."""

    required_paths = [
        dataset_dir / f"{split}.npz" for split in SPLIT_NAMES
    ] + [
        dataset_dir / "splits" / f"{split}_cases.csv" for split in SPLIT_NAMES
    ] + [
        dataset_dir / f"{split}_metadata.csv" for split in SPLIT_NAMES
    ] + [
        dataset_dir / "dataset_metadata.json",
        dataset_dir / "dataset_report.json",
        dataset_dir / "feature_manifest.csv",
        dataset_dir / "preprocessing.pkl",
        dataset_dir / "preprocessing_statistics.csv",
    ]
    artifact_existence = {str(path): path.exists() for path in required_paths}
    missing_artifacts = [path for path, exists in artifact_existence.items() if not exists]
    if missing_artifacts:
        raise FileNotFoundError(f"Required full artifacts are missing: {missing_artifacts}")

    metadata_json = _load_json(dataset_dir / "dataset_metadata.json")
    report_json = _load_json(dataset_dir / "dataset_report.json")
    feature_manifest = pd.read_csv(dataset_dir / "feature_manifest.csv")
    preprocessing_stats = pd.read_csv(dataset_dir / "preprocessing_statistics.csv")
    with (dataset_dir / "preprocessing.pkl").open("rb") as handle:
        preprocessing_artifact = pickle.load(handle)
    split_ids = _split_case_ids(dataset_dir)

    overlap = {
        "train_val": sorted(split_ids["train"] & split_ids["val"]),
        "train_test": sorted(split_ids["train"] & split_ids["test"]),
        "val_test": sorted(split_ids["val"] & split_ids["test"]),
    }
    all_case_ids = set().union(*split_ids.values())
    case_membership_counts = {
        case_id: sum(case_id in ids for ids in split_ids.values()) for case_id in all_case_ids
    }

    arrays: dict[str, dict[str, np.ndarray]] = {}
    split_metadata: dict[str, pd.DataFrame] = {}
    integrity_by_split: dict[str, Any] = {}
    case_summaries: list[pd.DataFrame] = []
    targets_by_split: dict[str, np.ndarray] = {}
    for split in SPLIT_NAMES:
        with np.load(dataset_dir / f"{split}.npz") as loaded:
            arrays[split] = {name: loaded[name].copy() for name in loaded.files}
        split_metadata[split] = pd.read_csv(dataset_dir / f"{split}_metadata.csv")
        targets = arrays[split]["y_bis"].astype(float)
        targets_by_split[split] = targets
        first_dimensions = {
            name: int(value.shape[0]) for name, value in arrays[split].items()
        }
        first_dimensions["metadata"] = len(split_metadata[split])
        integrity_by_split[split] = {
            "all_arrays_finite": all(
                bool(np.isfinite(value).all()) for value in arrays[split].values()
            ),
            "observation_mask_is_boolean": arrays[split]["observation_mask"].dtype == np.bool_,
            "consistent_first_dimensions": len(set(first_dimensions.values())) == 1,
            "first_dimensions": first_dimensions,
            "X_dynamic_shape": list(arrays[split]["X_dynamic"].shape),
            "X_static_shape": list(arrays[split]["X_static"].shape),
            "target_offsets_exactly_30_seconds": bool(
                (
                    split_metadata[split]["target_timestamp"]
                    - split_metadata[split]["final_input_timestamp"]
                    == 30
                ).all()
            ),
            "history_span_exactly_50_seconds": bool(
                (
                    split_metadata[split]["final_input_timestamp"]
                    - split_metadata[split]["first_input_timestamp"]
                    == 50
                ).all()
            ),
            "metadata_cases_belong_to_split": set(
                split_metadata[split]["case_id"].astype(int)
            ).issubset(split_ids[split]),
        }
        case_summaries.append(
            _case_level_summary(split, split_ids[split], split_metadata[split], targets)
        )

    dynamic_names = metadata_json["dynamic_feature_names"]
    static_names = metadata_json["static_feature_names"]
    expected_shapes = all(
        arrays[split]["X_dynamic"].shape[1:] == (6, len(dynamic_names))
        and arrays[split]["X_static"].shape[1:] == (len(static_names),)
        for split in SPLIT_NAMES
    )

    raw = load_cleaned_data(input_path)
    included_specs, _ = resolve_feature_specs(raw)
    resampled = add_derived_features(
        resample_cases(
            raw,
            [spec for spec in included_specs if not spec.derived],
            metadata_json["resampling_interval_seconds"],
        ),
        metadata_json["resampling_interval_seconds"],
    )
    resampled = resampled[resampled["caseid"].isin(all_case_ids)].copy()

    histories_within_cases = True
    targets_match_resampled_bis = True
    case_time_frames = {
        int(case_id): frame.set_index("timestamp")
        for case_id, frame in resampled.groupby("caseid", sort=False)
    }
    for split in SPLIT_NAMES:
        for row, target in zip(
            split_metadata[split].itertuples(index=False), targets_by_split[split], strict=True
        ):
            case_frame = case_time_frames[int(row.case_id)]
            needed = range(int(row.first_input_timestamp), int(row.final_input_timestamp) + 1, 10)
            if not all(timestamp in case_frame.index for timestamp in needed):
                histories_within_cases = False
                break
            source_target = case_frame.at[int(row.target_timestamp), "bis"]
            if not np.isclose(source_target, target, rtol=1e-5, atol=1e-5):
                targets_match_resampled_bis = False
                break

    case_summary = pd.concat(case_summaries, ignore_index=True)
    case_summary.to_csv(dataset_dir / "case_level_target_summary.csv", index=False)

    split_sizes: dict[str, Any] = {}
    target_distributions: dict[str, Any] = {}
    event_coverage: dict[str, Any] = {}
    total_cases = len(all_case_ids)
    total_windows = sum(len(targets) for targets in targets_by_split.values())
    for split in SPLIT_NAMES:
        split_cases = case_summary[case_summary["split"] == split]
        windows_per_case = split_cases["number_of_windows"].to_numpy(dtype=float)
        size_stats = _numeric_summary(windows_per_case)
        sorted_windows = np.sort(windows_per_case)[::-1]
        split_sizes[split] = {
            "case_count": len(split_ids[split]),
            "window_count": len(targets_by_split[split]),
            "percentage_of_total_cases": _float(len(split_ids[split]) / total_cases * 100.0),
            "percentage_of_total_windows": _float(
                len(targets_by_split[split]) / total_windows * 100.0
            ),
            "windows_per_case": size_stats,
            "largest_case_window_share": _float(sorted_windows[0] / sorted_windows.sum()),
            "largest_two_cases_window_share": _float(
                sorted_windows[:2].sum() / sorted_windows.sum()
            ),
        }
        target_distributions[split] = _target_summary(targets_by_split[split])
        event_coverage[split] = _case_event_coverage(split_cases)

    train = targets_by_split["train"]
    comparisons = {
        "absolute_bis_above_60_prevalence_difference_train_val": _float(
            abs(np.mean(train > 60.0) - np.mean(targets_by_split["val"] > 60.0))
        ),
        "absolute_bis_above_60_prevalence_difference_train_test": _float(
            abs(np.mean(train > 60.0) - np.mean(targets_by_split["test"] > 60.0))
        ),
        "absolute_bis_below_40_prevalence_difference_train_val": _float(
            abs(np.mean(train < 40.0) - np.mean(targets_by_split["val"] < 40.0))
        ),
        "absolute_bis_below_40_prevalence_difference_train_test": _float(
            abs(np.mean(train < 40.0) - np.mean(targets_by_split["test"] < 40.0))
        ),
        "target_bis_standardized_mean_difference_train_val": _smd(
            train, targets_by_split["val"]
        ),
        "target_bis_standardized_mean_difference_train_test": _smd(
            train, targets_by_split["test"]
        ),
        "mean_windows_per_case": {
            split: _float(split_sizes[split]["window_count"] / split_sizes[split]["case_count"])
            for split in SPLIT_NAMES
        },
        "mean_windows_per_case_ratios": {
            "train_to_val": _float(
                (split_sizes["train"]["window_count"] / split_sizes["train"]["case_count"])
                / (split_sizes["val"]["window_count"] / split_sizes["val"]["case_count"])
            ),
            "train_to_test": _float(
                (split_sizes["train"]["window_count"] / split_sizes["train"]["case_count"])
                / (split_sizes["test"]["window_count"] / split_sizes["test"]["case_count"])
            ),
            "val_to_test": _float(
                (split_sizes["val"]["window_count"] / split_sizes["val"]["case_count"])
                / (split_sizes["test"]["window_count"] / split_sizes["test"]["case_count"])
            ),
        },
    }

    missingness, missingness_flags = _missingness_audit(
        resampled, split_ids, feature_manifest, preprocessing_stats
    )

    timestamp_audit: dict[str, Any] = {}
    timestamp_audit["candidate_window_definition"] = (
        "each observed resampled endpoint at least history_span seconds after the "
        "case's first resampled timestamp; initial endpoints without nominal history "
        "are not candidates"
    )
    case_exclusion_rows: list[dict[str, Any]] = []
    for split, ids in split_ids.items():
        raw_split = raw[raw["caseid"].isin(ids)]
        resampled_split = resampled[resampled["caseid"].isin(ids)]
        raw_gaps_by_case = (
            raw_split.groupby("caseid", sort=True)["time_sec"]
            .apply(lambda values: int((values.diff().dropna() != 1).sum()))
            .to_dict()
        )
        resampled_gaps_by_case = (
            resampled_split.groupby("caseid", sort=True)["timestamp"]
            .apply(lambda values: int((values.diff().dropna() != 10).sum()))
            .to_dict()
        )
        for case_id, case_frame in resampled_split.groupby("caseid", sort=True):
            counts = _window_candidate_counts(case_frame, 6, 10, 30)
            excluded = counts["excluded_history_gap"] + counts["excluded_unavailable_future_bis"]
            case_exclusion_rows.append(
                {
                    "case_id": int(case_id),
                    "split": split,
                    **counts,
                    "excluded_total": excluded,
                    "excluded_percentage": _float(
                        excluded / counts["candidate_windows"] * 100.0
                    ),
                    "raw_irregular_gap_count": raw_gaps_by_case[int(case_id)],
                    "resampled_irregular_gap_count": resampled_gaps_by_case[int(case_id)],
                }
            )

        split_exclusions = [row for row in case_exclusion_rows if row["split"] == split]
        candidates = sum(row["candidate_windows"] for row in split_exclusions)
        gap_excluded = sum(row["excluded_history_gap"] for row in split_exclusions)
        future_excluded = sum(
            row["excluded_unavailable_future_bis"] for row in split_exclusions
        )
        included = sum(row["included_windows"] for row in split_exclusions)
        timestamp_audit[split] = {
            "raw_irregular_gap_count": sum(raw_gaps_by_case.values()),
            "resampled_irregular_gap_count": sum(resampled_gaps_by_case.values()),
            "candidate_window_count": candidates,
            "included_window_count": included,
            "excluded_history_gap_count": gap_excluded,
            "excluded_history_gap_percentage": _float(gap_excluded / candidates * 100.0),
            "excluded_unavailable_future_bis_count": future_excluded,
            "excluded_unavailable_future_bis_percentage": _float(
                future_excluded / candidates * 100.0
            ),
        }
        if included != len(targets_by_split[split]):
            raise AssertionError(
                f"Audit window count differs from saved {split} windows: "
                f"{included} != {len(targets_by_split[split])}"
            )

    exclusion_frame = pd.DataFrame(case_exclusion_rows)
    q1, q3 = np.percentile(exclusion_frame["excluded_percentage"], [25, 75])
    disproportionate_threshold = _float(q3 + 1.5 * (q3 - q1))
    disproportionately_affected = exclusion_frame[
        exclusion_frame["excluded_percentage"] > disproportionate_threshold
    ].sort_values("excluded_percentage", ascending=False)
    timestamp_audit["disproportionately_affected_definition"] = (
        "case exclusion percentage above Q3 + 1.5*IQR across all cases"
    )
    timestamp_audit["disproportionately_affected_threshold_percentage"] = (
        disproportionate_threshold
    )
    timestamp_audit["disproportionately_affected_cases"] = disproportionately_affected.to_dict(
        orient="records"
    )
    timestamp_audit["top_10_cases_by_exclusion_percentage"] = exclusion_frame.nlargest(
        10, "excluded_percentage"
    ).to_dict(orient="records")

    no_dominant_split = all(
        split_sizes[split]["largest_two_cases_window_share"] < 0.50
        for split in SPLIT_NAMES
    )
    multiple_event_cases = all(
        event_coverage[split]["cases_with_bis_above_60"] >= 2
        and event_coverage[split]["cases_with_bis_below_40"] >= 2
        for split in ("val", "test")
    )
    target_differences_not_extreme = (
        comparisons["absolute_bis_above_60_prevalence_difference_train_val"] <= 0.10
        and comparisons["absolute_bis_above_60_prevalence_difference_train_test"] <= 0.10
        and comparisons["absolute_bis_below_40_prevalence_difference_train_val"] <= 0.10
        and comparisons["absolute_bis_below_40_prevalence_difference_train_test"] <= 0.10
        and abs(comparisons["target_bis_standardized_mean_difference_train_val"]) <= 0.50
        and abs(comparisons["target_bis_standardized_mean_difference_train_test"]) <= 0.50
    )
    systematic_unavailability = bool(
        missingness_flags["over_30_percent_missing_in_any_split"]
    )
    decision_checks = {
        "validation_and_test_each_have_at_least_10_cases": all(
            len(split_ids[split]) >= 10 for split in ("val", "test")
        ),
        "validation_and_test_have_multiple_high_and_low_event_cases": multiple_event_cases,
        "no_split_dominated_by_largest_two_cases": no_dominant_split,
        "target_distribution_differences_not_extreme": target_differences_not_extreme,
        "no_critical_feature_systematically_unavailable": not systematic_unavailability,
    }
    provisionally_acceptable = all(decision_checks.values())

    audit = {
        "audit_scope": {
            "dataset_directory": str(dataset_dir),
            "input_file": str(input_path),
            "audit_generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "build_runtime_seconds": build_runtime_seconds,
            "arrays_modified": False,
            "far_outside_training_definition": (
                "observed value farther than 5 training standard deviations from the "
                "training mean; any differing value when training variance is near zero"
            ),
            "near_zero_variance_threshold": 1e-8,
            "no_window_level_significance_tests_performed": True,
        },
        "basic_integrity": {
            "split_case_id_overlaps": overlap,
            "case_count_in_union": len(all_case_ids),
            "every_case_in_exactly_one_split": all(
                count == 1 for count in case_membership_counts.values()
            ),
            "all_cases_from_build_report_covered": len(all_case_ids)
            == int(report_json["included_case_count"]),
            "feature_ordering_identical_across_splits": expected_shapes,
            "dynamic_feature_order": dynamic_names,
            "static_feature_order": static_names,
            "histories_resolve_within_metadata_case": histories_within_cases,
            "saved_targets_match_resampled_future_bis": targets_match_resampled_bis,
            "by_split": integrity_by_split,
        },
        "split_sizes": split_sizes,
        "target_distributions": target_distributions,
        "case_level_event_coverage": event_coverage,
        "split_comparability": comparisons,
        "missingness_and_preprocessing": {
            "basis": (
                "dynamic missingness uses resampled rows; static missingness uses one value per case"
            ),
            "features": missingness,
            "flags": missingness_flags,
        },
        "timestamp_gaps_and_window_exclusions": timestamp_audit,
        "output_validation": {
            "required_artifacts_exist": artifact_existence,
            "all_required_artifacts_exist": all(artifact_existence.values()),
            "npz_files_reloaded": True,
            "metadata_csv_files_reloaded": True,
            "preprocessing_pickle_reloaded": preprocessing_artifact is not None,
            "dataset_metadata_reloaded": bool(metadata_json),
            "dataset_report_reloaded": bool(report_json),
        },
        "split_acceptability": {
            "decision_rule_checks": decision_checks,
            "provisionally_acceptable_for_initial_modeling": provisionally_acceptable,
            "recommend_revising_split_strategy_next_task": not provisionally_acceptable,
        },
    }
    _dump_json(audit, dataset_dir / "full_dataset_audit.json")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/modeling/full"))
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/processed/vitaldb_clean_100cases.csv"),
    )
    parser.add_argument("--build-runtime-seconds", type=float)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    audit = audit_dataset(args.dataset_dir, args.input_file, args.build_runtime_seconds)
    print(json.dumps(audit["split_sizes"], indent=2))
    print(json.dumps(audit["case_level_event_coverage"], indent=2))
    print(json.dumps(audit["split_acceptability"], indent=2))
    print(f"Audit JSON: {(args.dataset_dir / 'full_dataset_audit.json').resolve()}")
    print(
        "Case summary CSV: "
        f"{(args.dataset_dir / 'case_level_target_summary.csv').resolve()}"
    )


if __name__ == "__main__":
    main()
