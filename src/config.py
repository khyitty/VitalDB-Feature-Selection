"""Configuration for constructing future-BIS prediction datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Lightweight configuration for the modeling-data pipeline.

    A history window includes its endpoint. With the defaults, the six input
    observations are t-50, t-40, t-30, t-20, t-10, and t seconds, and the
    prediction target is the unnormalized BIS observed at t+30 seconds.
    """

    input_path: Path = Path("data/processed/vitaldb_clean_100cases.csv")
    output_dir: Path = Path("data/modeling/pilot")
    seed: int = 42
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    resampling_interval_seconds: int = 10
    history_window_seconds: int = 60
    prediction_horizon_seconds: int = 30
    high_bis_threshold: float = 60.0
    low_bis_threshold: float = 40.0

    def __post_init__(self) -> None:
        fractions = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(fractions - 1.0) > 1e-9:
            raise ValueError("Train, validation, and test fractions must sum to 1.")
        if self.resampling_interval_seconds <= 0:
            raise ValueError("Resampling interval must be positive.")
        if self.history_window_seconds % self.resampling_interval_seconds != 0:
            raise ValueError("History window must be divisible by the sampling interval.")
        if self.prediction_horizon_seconds % self.resampling_interval_seconds != 0:
            raise ValueError("Prediction horizon must be divisible by the sampling interval.")

    @property
    def history_steps(self) -> int:
        """Return the number of observations in each history window."""

        return self.history_window_seconds // self.resampling_interval_seconds

