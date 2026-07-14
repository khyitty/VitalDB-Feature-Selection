"""Per-case resampling and train-only preprocessing utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd


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


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
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


def add_derived_features(frame: pd.DataFrame, interval_seconds: int) -> pd.DataFrame:
    """Add causal BIS slope and BIS error after resampling."""

    result = frame.copy()
    grouped = result.groupby("caseid", sort=False)
    previous_gap = grouped["timestamp"].diff()
    bis_difference = grouped["bis"].diff()
    result["bis_slope"] = (bis_difference / interval_seconds).where(
        previous_gap == interval_seconds
    )
    result["bis_error"] = result["bis"] - 50.0
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
                "exclusion_reason": "" if included else exclusions.get(spec.name, "excluded"),
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

