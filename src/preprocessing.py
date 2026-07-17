"""Per-case resampling and train-only preprocessing utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.pkpd.reconstruction import (
    PROPOFOL_CONCENTRATION_MG_PER_ML,
    REMIFENTANIL_CONCENTRATION_MICROGRAMS_PER_ML,
)
from src.prediction_feature_profiles import (
    LEGACY_PHYSIOLOGICAL_PROFILE,
    SIMULATOR_COMPATIBLE_PROFILE,
)
from src.rl_env.state_manifests import FEATURE_REGISTRY


@dataclass(frozen=True)
class FeatureSpec:
    """Meaning and aggregation behavior for one candidate input feature."""

    original_name: str
    name: str
    feature_class: str
    aggregation: str
    categorical: bool = False
    required: bool = False
    derived: bool = False
    include_in_model: bool = True


LEGACY_PHYSIOLOGICAL_FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec("BIS", "bis", "dynamic", "median", required=True),
    FeatureSpec("SQI", "bis_sqi", "dynamic", "median"),
    FeatureSpec("HR", "hr", "dynamic", "median"),
    FeatureSpec("MBP", "mbp", "dynamic", "median"),
    FeatureSpec("SBP", "sbp", "dynamic", "median"),
    FeatureSpec("DBP", "dbp", "dynamic", "median"),
    FeatureSpec("SPO2", "spo2", "dynamic", "median"),
    FeatureSpec("ETCO2", "etco2", "dynamic", "median"),
    FeatureSpec("PPF_RATE", "ppf_rate", "dynamic", "last"),
    FeatureSpec("PPF_VOL", "ppf_volume", "dynamic", "last"),
    FeatureSpec("PPF_CP", "ppf_cp", "dynamic", "median"),
    FeatureSpec("PPF_CE", "ppf_ce", "dynamic", "median"),
    FeatureSpec("RFTN_RATE", "rftn_rate", "dynamic", "last"),
    FeatureSpec("RFTN_VOL", "rftn_volume", "dynamic", "last"),
    FeatureSpec("RFTN_CP", "rftn_cp", "dynamic", "median"),
    FeatureSpec("RFTN_CE", "rftn_ce", "dynamic", "median"),
    FeatureSpec("age", "age", "static", "constant"),
    FeatureSpec("sex_male", "sex_male", "static", "constant", categorical=True),
    FeatureSpec("height", "height", "static", "constant"),
    FeatureSpec("weight", "weight", "static", "constant"),
    FeatureSpec("bmi", "bmi", "static", "constant"),
    FeatureSpec("asa", "asa", "static", "constant"),
    FeatureSpec("BIS", "bis_slope", "dynamic", "derived_after_resampling", derived=True),
    FeatureSpec("BIS", "bis_error", "dynamic", "derived_after_resampling", derived=True),
)

# Backward-compatible name for immutable physiological-inclusive exploratory code.
FEATURE_SPECS = LEGACY_PHYSIOLOGICAL_FEATURE_SPECS

SIMULATOR_COMPATIBLE_FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec("BIS", "bis", "dynamic", "median", required=True),
    FeatureSpec(
        "PPF_RATE",
        "__ppf20_rate_ml_per_hour",
        "source_only",
        "last",
        required=True,
        include_in_model=False,
    ),
    FeatureSpec(
        "PPF_VOL",
        "__ppf20_volume_ml",
        "source_only",
        "last",
        required=True,
        include_in_model=False,
    ),
    FeatureSpec(
        "RFTN_RATE",
        "__rftn20_rate_ml_per_hour",
        "source_only",
        "last",
        required=True,
        include_in_model=False,
    ),
    FeatureSpec(
        "RFTN_VOL",
        "__rftn20_volume_ml",
        "source_only",
        "last",
        required=True,
        include_in_model=False,
    ),
    FeatureSpec("age", "age_years", "static", "constant", required=True),
    FeatureSpec(
        "sex_male",
        "sex_male",
        "static",
        "constant",
        categorical=True,
        required=True,
    ),
    FeatureSpec("height", "height_cm", "static", "constant", required=True),
    FeatureSpec("weight", "weight_kg", "static", "constant", required=True),
    FeatureSpec("BIS", "bis_delta_10s", "dynamic", "derived_after_resampling", derived=True),
    FeatureSpec("BIS", "bis_target_error", "dynamic", "derived_after_resampling", derived=True),
    FeatureSpec(
        "PPF_RATE",
        "propofol_rate_mg_per_min",
        "dynamic",
        "derived_unit_conversion",
        derived=True,
    ),
    FeatureSpec(
        "PPF_VOL",
        "propofol_recent_dose_mg",
        "dynamic",
        "derived_causal_60s_difference",
        derived=True,
    ),
    FeatureSpec(
        "PPF_VOL",
        "propofol_cumulative_dose_mg",
        "dynamic",
        "derived_from_profile_start",
        derived=True,
    ),
    FeatureSpec(
        "propofol_cp_mg_per_l",
        "propofol_cp_mg_per_l",
        "dynamic",
        "last",
        required=True,
    ),
    FeatureSpec(
        "propofol_ce_mg_per_l",
        "propofol_ce_mg_per_l",
        "dynamic",
        "last",
        required=True,
    ),
    FeatureSpec(
        "RFTN_RATE",
        "remifentanil_rate_micrograms_per_min",
        "dynamic",
        "derived_unit_conversion",
        derived=True,
    ),
    FeatureSpec(
        "RFTN_VOL",
        "remifentanil_recent_dose_micrograms",
        "dynamic",
        "derived_causal_60s_difference",
        derived=True,
    ),
    FeatureSpec(
        "RFTN_VOL",
        "remifentanil_cumulative_dose_micrograms",
        "dynamic",
        "derived_from_profile_start",
        derived=True,
    ),
    FeatureSpec(
        "remifentanil_cp_micrograms_per_l",
        "remifentanil_cp_micrograms_per_l",
        "dynamic",
        "last",
        required=True,
    ),
    FeatureSpec(
        "remifentanil_ce_micrograms_per_l",
        "remifentanil_ce_micrograms_per_l",
        "dynamic",
        "last",
        required=True,
    ),
    FeatureSpec(
        "__recorded_orchestra_propofol_cp_mg_per_l",
        "__recorded_orchestra_propofol_cp_mg_per_l",
        "source_only",
        "last",
        include_in_model=False,
    ),
    FeatureSpec(
        "__recorded_orchestra_propofol_ce_mg_per_l",
        "__recorded_orchestra_propofol_ce_mg_per_l",
        "source_only",
        "last",
        include_in_model=False,
    ),
    FeatureSpec(
        "__recorded_orchestra_remifentanil_cp_micrograms_per_l",
        "__recorded_orchestra_remifentanil_cp_micrograms_per_l",
        "source_only",
        "last",
        include_in_model=False,
    ),
    FeatureSpec(
        "__recorded_orchestra_remifentanil_ce_micrograms_per_l",
        "__recorded_orchestra_remifentanil_ce_micrograms_per_l",
        "source_only",
        "last",
        include_in_model=False,
    ),
)

PROPOFOL_20_CONCENTRATION_MG_PER_ML = PROPOFOL_CONCENTRATION_MG_PER_ML
REMIFENTANIL_20_CONCENTRATION_MICROGRAMS_PER_ML = (
    REMIFENTANIL_CONCENTRATION_MICROGRAMS_PER_ML
)


def feature_specs_for_profile(profile_name: str) -> tuple[FeatureSpec, ...]:
    """Return the exact immutable feature specs for one dataset profile."""

    if profile_name == SIMULATOR_COMPATIBLE_PROFILE:
        return SIMULATOR_COMPATIBLE_FEATURE_SPECS
    if profile_name == LEGACY_PHYSIOLOGICAL_PROFILE:
        return LEGACY_PHYSIOLOGICAL_FEATURE_SPECS
    raise ValueError(f"Unknown prediction feature profile: {profile_name!r}")


@dataclass(frozen=True)
class FeatureStatistics:
    """Training-only imputation and normalization values for one feature."""

    feature_name: str
    training_median: float
    training_mean: float
    training_standard_deviation: float
    imputation_value: float
    normalization_scale: float
    feature_type: str
    standardized: bool


@dataclass(frozen=True)
class PreprocessingArtifact:
    """Serializable preprocessing state fitted using training cases only."""

    statistics: dict[str, FeatureStatistics]
    dynamic_features: tuple[str, ...]
    static_features: tuple[str, ...]

    def statistics_frame(self) -> pd.DataFrame:
        """Return human-readable training preprocessing statistics."""

        return pd.DataFrame(asdict(value) for value in self.statistics.values())


def resolve_feature_specs(
    frame: pd.DataFrame,
    specs: Iterable[FeatureSpec] = FEATURE_SPECS,
) -> tuple[list[FeatureSpec], dict[str, str]]:
    """Include available candidate features and explain every exclusion."""

    included: list[FeatureSpec] = []
    exclusions: dict[str, str] = {}
    for spec in specs:
        if spec.derived:
            included.append(spec)
            continue
        if spec.original_name not in frame.columns:
            if spec.required:
                raise ValueError(f"Required source column {spec.original_name!r} is unavailable.")
            exclusions[spec.name] = "source column unavailable"
            continue
        if not frame[spec.original_name].notna().any():
            if spec.required:
                raise ValueError(f"Required source column {spec.original_name!r} has no observations.")
            exclusions[spec.name] = "source column contains no observed values"
            continue
        included.append(spec)
    return included, exclusions


def resample_cases(
    frame: pd.DataFrame,
    specs: Iterable[FeatureSpec],
    interval_seconds: int,
) -> pd.DataFrame:
    """Aggregate every case independently into fixed-width time bins."""

    specs = [spec for spec in specs if not spec.derived]
    required = {"caseid", "time_sec", *(spec.original_name for spec in specs)}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Cannot resample; missing columns: {missing}")

    static_specs = [spec for spec in specs if spec.feature_class == "static"]
    for spec in static_specs:
        inconsistent = frame.groupby("caseid")[spec.original_name].nunique(dropna=False) > 1
        if inconsistent.any():
            bad_cases = inconsistent[inconsistent].index.tolist()[:5]
            raise ValueError(
                f"Static feature {spec.original_name!r} varies within cases: {bad_cases}"
            )

    work = frame.loc[:, sorted(required)].copy()
    work["timestamp"] = (
        pd.to_numeric(work["time_sec"], errors="raise").astype(np.int64)
        // interval_seconds
        * interval_seconds
    )
    aggregation = {
        spec.original_name: ("first" if spec.aggregation == "constant" else spec.aggregation)
        for spec in specs
    }
    resampled = (
        work.groupby(["caseid", "timestamp"], sort=True, as_index=False)
        .agg(aggregation)
        .rename(columns={spec.original_name: spec.name for spec in specs})
        .sort_values(["caseid", "timestamp"], kind="stable")
        .reset_index(drop=True)
    )
    return resampled


def _reset_aware_cumulative(
    frame: pd.DataFrame,
    source_name: str,
    concentration: float,
    output_name: str,
) -> tuple[pd.Series, pd.Series]:
    """Accumulate non-negative volume increments across explicit causal pump resets."""

    cumulative = pd.Series(np.nan, index=frame.index, dtype=float)
    reset = pd.Series(False, index=frame.index, dtype=bool)
    for _, case in frame.groupby("caseid", sort=False):
        total_volume = 0.0
        previous_volume: float | None = None
        for index, raw_value in case[source_name].items():
            if pd.isna(raw_value):
                continue
            value = float(raw_value)
            if value < 0.0:
                raise ValueError(f"Negative cumulative pump volume is unsupported: {source_name}")
            if previous_volume is None:
                increment = 0.0
            elif value + 1e-6 >= previous_volume:
                increment = max(value - previous_volume, 0.0)
            else:
                reset.loc[index] = True
                increment = value
            total_volume += increment
            cumulative.loc[index] = total_volume * concentration
            previous_volume = value
    if bool((cumulative.groupby(frame["caseid"], sort=False).diff().dropna() < -1e-6).any()):
        raise AssertionError(f"Reset-aware cumulative reconstruction decreased: {output_name}")
    return cumulative, reset


def _causal_recent_difference(
    frame: pd.DataFrame,
    cumulative_name: str,
    *,
    window_seconds: int,
) -> pd.Series:
    """Difference cumulative dose from exactly 60 seconds ago or the case start."""

    recent = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, case in frame.groupby("caseid", sort=False):
        timestamps = case["timestamp"].to_numpy(dtype=np.int64)
        case_start = int(timestamps[0])
        reference_timestamps = np.maximum(timestamps - window_seconds, case_start)
        cumulative = case[cumulative_name]
        by_timestamp = pd.Series(cumulative.to_numpy(), index=timestamps)
        reference = by_timestamp.reindex(reference_timestamps).to_numpy(dtype=float)
        recent.loc[case.index] = cumulative.to_numpy(dtype=float) - reference
    return recent


def add_derived_features(
    frame: pd.DataFrame,
    interval_seconds: int,
    feature_profile: str = LEGACY_PHYSIOLOGICAL_PROFILE,
) -> pd.DataFrame:
    """Add causal profile-specific transformations after fixed-interval resampling."""

    result = frame.copy()
    grouped = result.groupby("caseid", sort=False)
    previous_gap = grouped["timestamp"].diff()
    bis_difference = grouped["bis"].diff()
    if feature_profile == LEGACY_PHYSIOLOGICAL_PROFILE:
        result["bis_slope"] = (bis_difference / interval_seconds).where(
            previous_gap == interval_seconds
        )
        result["bis_error"] = result["bis"] - 50.0
        return result
    if feature_profile != SIMULATOR_COMPATIBLE_PROFILE:
        raise ValueError(f"Unknown prediction feature profile: {feature_profile!r}")
    if interval_seconds != 10:
        raise ValueError(
            "The simulator-compatible profile requires a 10-second sampling interval."
        )

    result["bis_delta_10s"] = bis_difference.where(previous_gap == interval_seconds)
    result["bis_target_error"] = result["bis"] - 50.0
    for source in ("__ppf20_rate_ml_per_hour", "__rftn20_rate_ml_per_hour"):
        if bool((result[source].dropna() < 0.0).any()):
            raise ValueError(f"Negative pump rates are unsupported: {source}")
    result["propofol_rate_mg_per_min"] = (
        result["__ppf20_rate_ml_per_hour"]
        * PROPOFOL_20_CONCENTRATION_MG_PER_ML
        / 60.0
    )
    result["remifentanil_rate_micrograms_per_min"] = (
        result["__rftn20_rate_ml_per_hour"]
        * REMIFENTANIL_20_CONCENTRATION_MICROGRAMS_PER_ML
        / 60.0
    )
    (
        result["propofol_cumulative_dose_mg"],
        result["__propofol_pump_reset"],
    ) = _reset_aware_cumulative(
        result,
        "__ppf20_volume_ml",
        PROPOFOL_20_CONCENTRATION_MG_PER_ML,
        "propofol_cumulative_dose_mg",
    )
    (
        result["remifentanil_cumulative_dose_micrograms"],
        result["__remifentanil_pump_reset"],
    ) = _reset_aware_cumulative(
        result,
        "__rftn20_volume_ml",
        REMIFENTANIL_20_CONCENTRATION_MICROGRAMS_PER_ML,
        "remifentanil_cumulative_dose_micrograms",
    )
    for cumulative_name, recent_name in (
        ("propofol_cumulative_dose_mg", "propofol_recent_dose_mg"),
        (
            "remifentanil_cumulative_dose_micrograms",
            "remifentanil_recent_dose_micrograms",
        ),
    ):
        result[recent_name] = _causal_recent_difference(
            result,
            cumulative_name,
            window_seconds=60,
        )
    return result


def build_feature_manifest(
    specs: Iterable[FeatureSpec],
    included_specs: Iterable[FeatureSpec],
    exclusions: dict[str, str],
    resampled: pd.DataFrame,
) -> pd.DataFrame:
    """Describe feature provenance, aggregation, inclusion, and missingness."""

    included_names = {spec.name for spec in included_specs}
    rows: list[dict[str, object]] = []
    for spec in specs:
        included = spec.name in included_names
        missing_pct = (
            float(resampled[spec.name].isna().mean() * 100.0)
            if included and spec.name in resampled
            else 100.0
        )
        rows.append(
            {
                "original_column_name": spec.original_name,
                "standardized_feature_name": spec.name,
                "dynamic_or_static": spec.feature_class,
                "aggregation_rule": spec.aggregation,
                "percentage_missing_before_imputation": missing_pct,
                "included": included,
                "included_in_model": included and spec.include_in_model,
                "exclusion_reason": "" if included else exclusions.get(spec.name, "excluded"),
                "units": (
                    FEATURE_REGISTRY[spec.name].units
                    if spec.name in FEATURE_REGISTRY
                    else "legacy_or_source_unit"
                ),
                "temporal_window_seconds": (
                    FEATURE_REGISTRY[spec.name].temporal_window_seconds
                    if spec.name in FEATURE_REGISTRY
                    else None
                ),
                "simulator_supported": (
                    FEATURE_REGISTRY[spec.name].simulator_supported
                    if spec.name in FEATURE_REGISTRY
                    else False
                ),
                "end_to_end_eligible": (
                    FEATURE_REGISTRY[spec.name].end_to_end_eligible
                    if spec.name in FEATURE_REGISTRY
                    else False
                ),
            }
        )
    return pd.DataFrame(rows)


def fit_preprocessor(
    train_frame: pd.DataFrame,
    dynamic_specs: Iterable[FeatureSpec],
    static_specs: Iterable[FeatureSpec],
) -> PreprocessingArtifact:
    """Fit medians, means, and standard deviations on training cases only."""

    dynamic_specs = list(dynamic_specs)
    static_specs = list(static_specs)
    statistics: dict[str, FeatureStatistics] = {}

    for spec in [*dynamic_specs, *static_specs]:
        if spec.feature_class == "static":
            values = train_frame[["caseid", spec.name]].drop_duplicates("caseid")[spec.name]
        else:
            values = train_frame[spec.name]
        numeric = pd.to_numeric(values, errors="coerce")
        observed = numeric.dropna()
        if observed.empty:
            raise ValueError(
                f"Feature {spec.name!r} has no observed values in training cases; "
                "train-only imputation cannot be fitted."
            )

        median = float(observed.median())
        if spec.categorical:
            modes = observed.mode()
            imputation_value = float(modes.iloc[0])
        else:
            imputation_value = median
        imputed = numeric.fillna(imputation_value).astype(float)
        mean = float(imputed.mean())
        standard_deviation = float(imputed.std(ddof=0))
        standardized = not spec.categorical
        scale = standard_deviation if standard_deviation > 0.0 else 1.0
        feature_type = f"{spec.feature_class}_{'categorical' if spec.categorical else 'continuous'}"
        statistics[spec.name] = FeatureStatistics(
            feature_name=spec.name,
            training_median=median,
            training_mean=mean,
            training_standard_deviation=standard_deviation,
            imputation_value=imputation_value,
            normalization_scale=scale,
            feature_type=feature_type,
            standardized=standardized,
        )

    return PreprocessingArtifact(
        statistics=statistics,
        dynamic_features=tuple(spec.name for spec in dynamic_specs),
        static_features=tuple(spec.name for spec in static_specs),
    )


def apply_preprocessor(
    frame: pd.DataFrame,
    artifact: PreprocessingArtifact,
) -> pd.DataFrame:
    """Apply training-only imputation and normalization to one split."""

    result = frame.copy()
    for feature_name, stats in artifact.statistics.items():
        if feature_name in artifact.dynamic_features:
            result[f"__observed__{feature_name}"] = result[feature_name].notna()
        values = pd.to_numeric(result[feature_name], errors="coerce").fillna(
            stats.imputation_value
        )
        if stats.standardized:
            values = (values - stats.training_mean) / stats.normalization_scale
        result[feature_name] = values.astype(float)
    return result
