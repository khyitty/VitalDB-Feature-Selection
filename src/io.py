"""End-to-end orchestration and persistence for prediction datasets."""

from __future__ import annotations

import json
import logging
import pickle
import random
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.preprocessing import (
    add_derived_features,
    apply_preprocessor,
    build_feature_manifest,
    fit_preprocessor,
    feature_specs_for_profile,
    resample_cases,
    resolve_feature_specs,
)
from src.prediction_feature_profiles import (
    SIMULATOR_COMPATIBLE_PROFILE,
    get_prediction_feature_profile,
    prediction_rl_definition_rows,
    validate_simulator_compatible_features,
)
from src.splits import load_case_splits, save_case_splits, split_case_ids
from src.windows import WindowDataset, build_windows, eligible_case_ids

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildResult:
    """Summary returned after a complete dataset build."""

    output_dir: Path
    dynamic_features: tuple[str, ...]
    static_features: tuple[str, ...]
    case_counts: dict[str, int]
    window_counts: dict[str, int]
    tensor_shapes: dict[str, dict[str, tuple[int, ...]]]
    prevalence: dict[str, dict[str, float]]


def load_cleaned_data(path: Path) -> pd.DataFrame:
    """Load and validate the cleaned, mostly 1-second VitalDB table."""

    if not path.exists():
        raise FileNotFoundError(f"Cleaned VitalDB input does not exist: {path}")
    frame = pd.read_csv(path, low_memory=False)
    required = {"caseid", "time_sec", "BIS"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Input dataset is empty.")
    if frame[["caseid", "time_sec"]].isna().any().any():
        raise ValueError("caseid and time_sec must not contain missing values.")

    frame["caseid"] = pd.to_numeric(frame["caseid"], errors="raise").astype(np.int64)
    frame["time_sec"] = pd.to_numeric(frame["time_sec"], errors="raise").astype(np.int64)
    duplicate_count = int(frame.duplicated(["caseid", "time_sec"]).sum())
    if duplicate_count:
        raise ValueError(f"Input contains {duplicate_count} duplicate case timestamps.")
    if not frame[["caseid", "time_sec"]].equals(
        frame[["caseid", "time_sec"]]
        .sort_values(["caseid", "time_sec"], kind="stable")
        .reset_index(drop=True)
    ):
        LOGGER.warning("Input rows were not sorted by caseid/time_sec; sorting before processing.")
        frame = frame.sort_values(["caseid", "time_sec"], kind="stable").reset_index(drop=True)
    nonpositive = frame.groupby("caseid", sort=False)["time_sec"].diff().le(0).sum()
    if nonpositive:
        raise ValueError(f"Input contains {int(nonpositive)} non-increasing case timestamps.")
    return frame


def _count_irregular_gaps(
    frame: pd.DataFrame, timestamp_column: str, expected_interval: int
) -> int:
    differences = frame.groupby("caseid", sort=False)[timestamp_column].diff().dropna()
    return int((differences != expected_interval).sum())


def _save_window_dataset(dataset: WindowDataset, output_dir: Path, split_name: str) -> None:
    np.savez_compressed(
        output_dir / f"{split_name}.npz",
        X_dynamic=dataset.X_dynamic,
        X_static=dataset.X_static,
        observation_mask=dataset.observation_mask,
        y_bis=dataset.y_bis,
        y_high_bis=dataset.y_high_bis,
        y_low_bis=dataset.y_low_bis,
    )
    dataset.metadata.to_csv(output_dir / f"{split_name}_metadata.csv", index=False)


def _target_summary(dataset: WindowDataset) -> dict[str, float | int | None]:
    targets = dataset.y_bis.astype(float)
    if not len(targets):
        return {
            "count": 0,
            "mean": None,
            "standard_deviation": None,
            "minimum": None,
            "maximum": None,
            "percentage_bis_below_40": None,
            "percentage_bis_40_to_60": None,
            "percentage_bis_above_60": None,
        }
    return {
        "count": int(len(targets)),
        "mean": float(np.mean(targets)),
        "standard_deviation": float(np.std(targets)),
        "minimum": float(np.min(targets)),
        "maximum": float(np.max(targets)),
        "percentage_bis_below_40": float(np.mean(targets < 40.0) * 100.0),
        "percentage_bis_40_to_60": float(
            np.mean((targets >= 40.0) & (targets <= 60.0)) * 100.0
        ),
        "percentage_bis_above_60": float(np.mean(targets > 60.0) * 100.0),
    }


def _warn_empty_extreme_classes(split_name: str, dataset: WindowDataset) -> None:
    if split_name not in {"val", "test"} or not len(dataset.y_bis):
        return
    if int(dataset.y_high_bis.sum()) == 0:
        warnings.warn(
            f"{split_name} split has zero future BIS > 60 targets.", RuntimeWarning, stacklevel=2
        )
    if int(dataset.y_low_bis.sum()) == 0:
        warnings.warn(
            f"{split_name} split has zero future BIS < 40 targets.", RuntimeWarning, stacklevel=2
        )


def _json_dump(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def build_prediction_dataset_from_frame(
    input_frame: pd.DataFrame,
    config: PipelineConfig,
    max_cases: int | None = None,
    input_label: str | None = None,
) -> BuildResult:
    """Build and save a split-safe future-BIS dataset from an in-memory table."""

    random.seed(config.seed)
    np.random.seed(config.seed)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = get_prediction_feature_profile(config.feature_profile)

    frame = input_frame.copy()
    input_rows = int(len(frame))
    input_cases = int(frame["caseid"].nunique())
    input_irregular_gaps = _count_irregular_gaps(frame, "time_sec", expected_interval=1)

    feature_specs = feature_specs_for_profile(config.feature_profile)
    included_specs, exclusions = resolve_feature_specs(frame, feature_specs)
    source_specs = [spec for spec in included_specs if not spec.derived]
    resampled_all = add_derived_features(
        resample_cases(frame, source_specs, config.resampling_interval_seconds),
        config.resampling_interval_seconds,
        config.feature_profile,
    )
    eligible_ids = eligible_case_ids(
        resampled_all,
        config.history_steps,
        config.resampling_interval_seconds,
        config.prediction_horizon_seconds,
    )
    if max_cases is not None:
        if max_cases < 3:
            raise ValueError("max_cases must be at least 3 for train/val/test splits.")
        selected_ids = eligible_ids[:max_cases]
    else:
        selected_ids = eligible_ids
    if len(selected_ids) < 3:
        raise ValueError(
            f"Only {len(selected_ids)} eligible cases can produce exact history/target windows."
        )

    selected = resampled_all[resampled_all["caseid"].isin(selected_ids)].copy()
    selected = selected.sort_values(["caseid", "timestamp"], kind="stable").reset_index(drop=True)
    dynamic_specs = [
        spec
        for spec in included_specs
        if spec.feature_class == "dynamic" and spec.include_in_model
    ]
    static_specs = [
        spec
        for spec in included_specs
        if spec.feature_class == "static" and spec.include_in_model
    ]
    dynamic_features = tuple(spec.name for spec in dynamic_specs)
    static_features = tuple(spec.name for spec in static_specs)

    splits = (
        load_case_splits(config.split_reference_dir, selected_ids)
        if config.split_reference_dir is not None
        else split_case_ids(
            selected_ids,
            seed=config.seed,
            fractions=(config.train_fraction, config.val_fraction, config.test_fraction),
        )
    )
    save_case_splits(splits, output_dir)
    LOGGER.info(
        "Case counts: train=%d val=%d test=%d",
        len(splits.train),
        len(splits.val),
        len(splits.test),
    )

    split_frames = {
        name: selected[selected["caseid"].isin(case_ids)].copy()
        for name, case_ids in splits.as_dict().items()
    }
    manifest = build_feature_manifest(
        feature_specs, included_specs, exclusions, split_frames["train"]
    )
    manifest.to_csv(output_dir / "feature_manifest.csv", index=False)
    artifact = fit_preprocessor(split_frames["train"], dynamic_specs, static_specs)
    with (output_dir / "preprocessing.pkl").open("wb") as handle:
        pickle.dump(artifact, handle)
    artifact.statistics_frame().to_csv(
        output_dir / "preprocessing_statistics.csv", index=False
    )

    datasets: dict[str, WindowDataset] = {}
    missingness: dict[str, dict[str, float]] = {}
    resampled_irregular_gaps: dict[str, int] = {}
    for split_name, split_frame in split_frames.items():
        split_frame["target_bis"] = split_frame["bis"]
        feature_names = [*dynamic_features, *static_features]
        missingness[split_name] = {
            name: float(split_frame[name].isna().mean() * 100.0) for name in feature_names
        }
        resampled_irregular_gaps[split_name] = _count_irregular_gaps(
            split_frame, "timestamp", config.resampling_interval_seconds
        )
        transformed = apply_preprocessor(split_frame, artifact)
        dataset = build_windows(
            transformed,
            dynamic_features,
            static_features,
            config.history_steps,
            config.resampling_interval_seconds,
            config.prediction_horizon_seconds,
            config.high_bis_threshold,
            config.low_bis_threshold,
        )
        datasets[split_name] = dataset
        _save_window_dataset(dataset, output_dir, split_name)
        if not (
            config.feature_profile == SIMULATOR_COMPATIBLE_PROFILE
            and split_name == "test"
        ):
            _warn_empty_extreme_classes(split_name, dataset)

    case_counts = {name: len(ids) for name, ids in splits.as_dict().items()}
    window_counts = {name: len(dataset.y_bis) for name, dataset in datasets.items()}
    summary_splits = (
        ("train", "val")
        if config.feature_profile == SIMULATOR_COMPATIBLE_PROFILE
        else ("train", "val", "test")
    )
    target_summaries = {
        name: _target_summary(datasets[name]) for name in summary_splits
    }
    if config.feature_profile != SIMULATOR_COMPATIBLE_PROFILE:
        all_targets = np.concatenate([dataset.y_bis for dataset in datasets.values()])
        combined_stub = WindowDataset(
            X_dynamic=np.empty((0, 0, 0)),
            X_static=np.empty((0, 0)),
            observation_mask=np.empty((0, 0, 0), dtype=bool),
            y_bis=all_targets,
            y_high_bis=(all_targets > config.high_bis_threshold).astype(np.int8),
            y_low_bis=(all_targets < config.low_bis_threshold).astype(np.int8),
            metadata=pd.DataFrame(),
            windows_removed_missing_future_bis=0,
        )
        target_summaries["all"] = _target_summary(combined_stub)

    generation_timestamp = datetime.now(timezone.utc).isoformat()
    input_name = input_label or str(config.input_path)
    if config.feature_profile == SIMULATOR_COMPATIBLE_PROFILE:
        validate_simulator_compatible_features(dynamic_features, static_features)
    dataset_metadata = {
        **profile.as_metadata(),
        "dynamic_feature_names": list(dynamic_features),
        "static_feature_names": list(static_features),
        "history_window_seconds": config.history_window_seconds,
        "history_steps": config.history_steps,
        "window_convention": "t-50,t-40,t-30,t-20,t-10,t for the 60-second default",
        "prediction_horizon_seconds": config.prediction_horizon_seconds,
        "resampling_interval_seconds": config.resampling_interval_seconds,
        "split_seed": config.seed,
        "input_file": input_name,
        "generation_timestamp_utc": generation_timestamp,
        "split_reused_from": (
            str(config.split_reference_dir.resolve())
            if config.split_reference_dir is not None
            else None
        ),
        "split_generated_from_seed": config.split_reference_dir is None,
        "preprocessing_fit_split": "train_only",
        "feature_selection_split_accessed": False,
        "test_results_inspected": False,
        "test_target_summary_sealed": (
            config.feature_profile == SIMULATOR_COMPATIBLE_PROFILE
        ),
        "legacy_physiological_outputs_overwritten": False,
        "vitaldb_pump_source_contract": (
            {
                "propofol": {
                    "track": "Orchestra/PPF20_RATE and Orchestra/PPF20_VOL",
                    "source_rate_unit": "mL/hr",
                    "concentration": "20 mg/mL",
                    "rate_conversion": "mL/hr * 20 mg/mL / 60 = mg/min",
                },
                "remifentanil": {
                    "track": "Orchestra/RFTN20_RATE and Orchestra/RFTN20_VOL",
                    "source_rate_unit": "mL/hr",
                    "concentration": "20 microgram/mL",
                    "rate_conversion": (
                        "mL/hr * 20 microgram/mL / 60 = microgram/min"
                    ),
                },
                "official_metadata_url": (
                    "https://vitaldb.net/dataset/?query=overview"
                ),
                "cumulative_dose_policy": (
                    "Within each case, sum non-negative recorded pump-volume increments. "
                    "A volume decrease starts a new pump segment and its current volume "
                    "is added as post-reset delivered volume. Missing readings remain missing."
                ),
                "pump_reset_counts": {
                    "propofol": int(selected["__propofol_pump_reset"].sum()),
                    "remifentanil": int(selected["__remifentanil_pump_reset"].sum()),
                },
            }
            if config.feature_profile == SIMULATOR_COMPATIBLE_PROFILE
            else None
        ),
        "prediction_rl_feature_definitions": (
            prediction_rl_definition_rows()
            if config.feature_profile == SIMULATOR_COMPATIBLE_PROFILE
            else None
        ),
        "excluded_from_main_profile": (
            {
                "unsupported_physiology": [
                    "HR/PLETH_HR",
                    "blood_pressure",
                    "SpO2",
                    "ETCO2",
                    "respiratory_variables",
                    "HRV",
                    "PLETH_waveform_features",
                    "BIS_SQI",
                ],
                "recorded_pkpd_concentrations": (
                    "PPF_CP/PPF_CE/RFTN_CP/RFTN_CE are not reconstructed by the "
                    "repository PK-PD simulator in prediction preprocessing."
                ),
                "legacy_static_covariates": ["bmi", "asa"],
            }
            if config.feature_profile == SIMULATOR_COMPATIBLE_PROFILE
            else None
        ),
        "cases_per_split": case_counts,
        "windows_per_split": window_counts,
    }
    _json_dump(dataset_metadata, output_dir / "dataset_metadata.json")

    report = {
        "feature_profile": config.feature_profile,
        "scientific_role": profile.scientific_role,
        "input_row_count": input_rows,
        "input_case_count": input_cases,
        "eligible_case_count": len(eligible_ids),
        "included_case_count": len(selected_ids),
        "excluded_case_count": input_cases - len(selected_ids),
        "ineligible_case_count": input_cases - len(eligible_ids),
        "cases_not_selected_due_to_mode": len(eligible_ids) - len(selected_ids),
        "cases_per_split": case_counts,
        "windows_per_split": window_counts,
        "bis_target_statistics": target_summaries,
        "missingness_percentage_by_feature_and_split": {
            name: missingness[name] for name in summary_splits
        },
        "windows_removed_due_to_missing_future_bis": {
            name: dataset.windows_removed_missing_future_bis
            for name, dataset in datasets.items()
        },
        "number_of_irregular_timestamp_gaps_detected": input_irregular_gaps,
        "input_irregular_one_second_gaps": input_irregular_gaps,
        "resampled_irregular_ten_second_gaps_by_split": resampled_irregular_gaps,
        "prior_physiological_inclusive_results_are_legacy": (
            config.feature_profile == SIMULATOR_COMPATIBLE_PROFILE
        ),
        "final_selected_feature_set_decided": False,
        "pump_reset_counts": (
            {
                "propofol": int(selected["__propofol_pump_reset"].sum()),
                "remifentanil": int(selected["__remifentanil_pump_reset"].sum()),
            }
            if config.feature_profile == SIMULATOR_COMPATIBLE_PROFILE
            else None
        ),
    }
    _json_dump(report, output_dir / "dataset_report.json")

    tensor_shapes = {
        name: {
            "X_dynamic": dataset.X_dynamic.shape,
            "X_static": dataset.X_static.shape,
            "observation_mask": dataset.observation_mask.shape,
            "y_bis": dataset.y_bis.shape,
        }
        for name, dataset in datasets.items()
    }
    prevalence = {
        name: {
            "high_bis": float(dataset.y_high_bis.mean()) if len(dataset.y_bis) else float("nan"),
            "low_bis": float(dataset.y_low_bis.mean()) if len(dataset.y_bis) else float("nan"),
            "bis_40_to_60": (
                float(np.mean((dataset.y_bis >= 40.0) & (dataset.y_bis <= 60.0)))
                if len(dataset.y_bis)
                else float("nan")
            ),
        }
        for name, dataset in datasets.items()
        if name in summary_splits
    }
    return BuildResult(
        output_dir=output_dir,
        dynamic_features=dynamic_features,
        static_features=static_features,
        case_counts=case_counts,
        window_counts=window_counts,
        tensor_shapes=tensor_shapes,
        prevalence=prevalence,
    )


def build_prediction_dataset(
    config: PipelineConfig, max_cases: int | None = None
) -> BuildResult:
    """Load the configured CSV and build all modeling-data artifacts."""

    frame = load_cleaned_data(config.input_path)
    return build_prediction_dataset_from_frame(
        frame, config, max_cases=max_cases, input_label=str(config.input_path)
    )
