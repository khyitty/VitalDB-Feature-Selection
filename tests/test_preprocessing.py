"""Tests for per-case aggregation and train-only preprocessing."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    FeatureSpec,
    add_derived_features,
    apply_preprocessor,
    fit_preprocessor,
    resample_cases,
)
from src.prediction_feature_profiles import SIMULATOR_COMPATIBLE_PROFILE


def test_resampling_never_combines_cases_and_uses_feature_aggregation_rules() -> None:
    frame = pd.DataFrame(
        {
            "caseid": [1] * 10 + [2] * 10,
            "time_sec": list(range(10)) * 2,
            "BIS": list(range(40, 50)) + list(range(60, 70)),
            "PPF_RATE": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, np.nan, 9.0]
            + [10.0] * 10,
            "PPF_VOL": list(range(10)) + list(range(100, 110)),
        }
    )
    specs = [
        FeatureSpec("BIS", "bis", "dynamic", "median", required=True),
        FeatureSpec("PPF_RATE", "ppf_rate", "dynamic", "last"),
        FeatureSpec("PPF_VOL", "ppf_volume", "dynamic", "last"),
    ]

    result = resample_cases(frame, specs, interval_seconds=10)

    assert len(result) == 2
    assert result.loc[result.caseid == 1, "bis"].item() == 44.5
    assert result.loc[result.caseid == 2, "bis"].item() == 64.5
    assert result.loc[result.caseid == 1, "ppf_rate"].item() == 9.0
    assert result.loc[result.caseid == 1, "ppf_volume"].item() == 9.0
    assert result.loc[result.caseid == 2, "ppf_volume"].item() == 109.0


def test_train_only_statistics_mask_and_constant_feature_handling() -> None:
    train = pd.DataFrame(
        {
            "caseid": [1, 1, 1],
            "timestamp": [0, 10, 20],
            "signal": [1.0, np.nan, 3.0],
            "constant": [5.0, 5.0, 5.0],
            "age": [40.0, 40.0, 40.0],
            "sex_male": [1, 1, 1],
        }
    )
    validation = pd.DataFrame(
        {
            "caseid": [2],
            "timestamp": [0],
            "signal": [1000.0],
            "constant": [5.0],
            "age": [90.0],
            "sex_male": [0],
        }
    )
    dynamic_specs = [
        FeatureSpec("signal", "signal", "dynamic", "median"),
        FeatureSpec("constant", "constant", "dynamic", "median"),
    ]
    static_specs = [
        FeatureSpec("age", "age", "static", "constant"),
        FeatureSpec("sex_male", "sex_male", "static", "constant", categorical=True),
    ]

    artifact = fit_preprocessor(train, dynamic_specs, static_specs)
    transformed_train = apply_preprocessor(train, artifact)
    transformed_validation = apply_preprocessor(validation, artifact)

    assert artifact.statistics["signal"].training_median == 2.0
    assert artifact.statistics["signal"].training_mean == 2.0
    assert artifact.statistics["age"].training_mean == 40.0
    assert artifact.statistics["constant"].training_standard_deviation == 0.0
    assert transformed_train["__observed__signal"].tolist() == [True, False, True]
    assert np.isfinite(transformed_train["constant"]).all()
    assert transformed_train["constant"].eq(0.0).all()
    assert transformed_validation["signal"].item() > 100.0
    assert transformed_validation["sex_male"].item() == 0.0


def test_bis_slope_does_not_bridge_irregular_gaps() -> None:
    frame = pd.DataFrame(
        {"caseid": [1, 1, 1], "timestamp": [0, 10, 30], "bis": [40.0, 50.0, 80.0]}
    )

    result = add_derived_features(frame, interval_seconds=10)

    assert np.isnan(result.loc[0, "bis_slope"])
    assert result.loc[1, "bis_slope"] == 1.0
    assert np.isnan(result.loc[2, "bis_slope"])
    assert result["bis_error"].tolist() == [-10.0, 0.0, 30.0]


def test_simulator_compatible_derivations_use_explicit_units_and_60s_window() -> None:
    frame = pd.DataFrame(
        {
            "caseid": [1] * 8,
            "timestamp": list(range(0, 80, 10)),
            "bis": np.arange(40.0, 56.0, 2.0),
            "__ppf20_rate_ml_per_hour": [3.0] * 8,
            "__ppf20_volume_ml": np.arange(8.0),
            "__rftn20_rate_ml_per_hour": [6.0] * 8,
            "__rftn20_volume_ml": np.arange(8.0) * 0.5,
        }
    )

    result = add_derived_features(
        frame, interval_seconds=10, feature_profile=SIMULATOR_COMPATIBLE_PROFILE
    )

    assert result.loc[1, "bis_delta_10s"] == 2.0
    assert result.loc[1, "bis_target_error"] == -8.0
    assert result["propofol_rate_mg_per_min"].eq(1.0).all()
    assert result["remifentanil_rate_micrograms_per_min"].eq(2.0).all()
    assert result.loc[0, "propofol_recent_dose_mg"] == 0.0
    assert result.loc[1, "propofol_recent_dose_mg"] == 20.0
    assert result.loc[6, "propofol_cumulative_dose_mg"] == 120.0
    assert result.loc[6, "propofol_recent_dose_mg"] == 120.0
    assert result.loc[6, "remifentanil_recent_dose_micrograms"] == 60.0


def test_simulator_compatible_profile_rejects_non_decision_interval() -> None:
    frame = pd.DataFrame(
        {
            "caseid": [1],
            "timestamp": [0],
            "bis": [50.0],
        }
    )
    with pytest.raises(ValueError, match="10-second sampling interval"):
        add_derived_features(
            frame,
            interval_seconds=5,
            feature_profile=SIMULATOR_COMPATIBLE_PROFILE,
        )


def test_simulator_compatible_cumulative_dose_records_and_handles_pump_resets() -> None:
    frame = pd.DataFrame(
        {
            "caseid": [1, 1, 1],
            "timestamp": [0, 10, 20],
            "bis": [50.0, 49.0, 48.0],
            "__ppf20_rate_ml_per_hour": [3.0, 3.0, 3.0],
            "__ppf20_volume_ml": [1.0, 2.0, 0.0],
            "__rftn20_rate_ml_per_hour": [1.0, 1.0, 1.0],
            "__rftn20_volume_ml": [0.0, 0.1, 0.2],
        }
    )
    result = add_derived_features(
        frame, 10, feature_profile=SIMULATOR_COMPATIBLE_PROFILE
    )
    assert result["__propofol_pump_reset"].tolist() == [False, False, True]
    assert result["propofol_cumulative_dose_mg"].tolist() == [0.0, 20.0, 20.0]
