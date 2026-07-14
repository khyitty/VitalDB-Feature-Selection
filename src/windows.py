"""Exact-timestamp history window and future-target construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowDataset:
    """Model arrays, metadata, and construction diagnostics for one split."""

    X_dynamic: np.ndarray
    X_static: np.ndarray
    observation_mask: np.ndarray
    y_bis: np.ndarray
    y_high_bis: np.ndarray
    y_low_bis: np.ndarray
    metadata: pd.DataFrame
    windows_removed_missing_future_bis: int


def case_has_valid_window(
    case_frame: pd.DataFrame,
    history_steps: int,
    interval_seconds: int,
    horizon_seconds: int,
) -> bool:
    """Return whether a case can supply at least one exact input/target window."""

    timestamps = set(int(value) for value in case_frame["timestamp"])
    bis_by_time = case_frame.set_index("timestamp")["bis"]
    for final_time in timestamps:
        history = [final_time - interval_seconds * offset for offset in range(history_steps)]
        target_time = final_time + horizon_seconds
        if all(value in timestamps for value in history) and target_time in timestamps:
            if pd.notna(bis_by_time.loc[target_time]):
                return True
    return False


def eligible_case_ids(
    resampled: pd.DataFrame,
    history_steps: int,
    interval_seconds: int,
    horizon_seconds: int,
) -> list[int]:
    """Return sorted cases that can produce at least one valid window."""

    return [
        int(case_id)
        for case_id, case_frame in resampled.groupby("caseid", sort=True)
        if case_has_valid_window(
            case_frame, history_steps, interval_seconds, horizon_seconds
        )
    ]


def build_windows(
    frame: pd.DataFrame,
    dynamic_features: Sequence[str],
    static_features: Sequence[str],
    history_steps: int,
    interval_seconds: int,
    horizon_seconds: int,
    high_bis_threshold: float = 60.0,
    low_bis_threshold: float = 40.0,
) -> WindowDataset:
    """Construct windows without crossing cases or accepting irregular timestamps.

    For six 10-second steps, each input consists of t-50 through t in ascending
    order. A target is retained only when BIS exists exactly at t+horizon.
    """

    dynamic_rows: list[np.ndarray] = []
    static_rows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    targets: list[float] = []
    metadata_rows: list[dict[str, int]] = []
    removed_missing_future = 0

    for case_id, case_frame in frame.groupby("caseid", sort=True):
        case_frame = case_frame.sort_values("timestamp", kind="stable").set_index("timestamp")
        timestamps = set(int(value) for value in case_frame.index)
        for final_time in sorted(timestamps):
            first_time = final_time - interval_seconds * (history_steps - 1)
            history_times = [first_time + interval_seconds * index for index in range(history_steps)]
            if not all(timestamp in timestamps for timestamp in history_times):
                continue

            target_time = final_time + horizon_seconds
            if target_time not in timestamps or pd.isna(case_frame.at[target_time, "target_bis"]):
                removed_missing_future += 1
                continue

            history = case_frame.loc[history_times]
            dynamic_rows.append(history.loc[:, dynamic_features].to_numpy(dtype=np.float32))
            static_rows.append(
                case_frame.loc[final_time, static_features].to_numpy(dtype=np.float32)
                if static_features
                else np.empty((0,), dtype=np.float32)
            )
            masks.append(
                history.loc[:, [f"__observed__{name}" for name in dynamic_features]]
                .to_numpy(dtype=bool)
            )
            targets.append(float(case_frame.at[target_time, "target_bis"]))
            metadata_rows.append(
                {
                    "case_id": int(case_id),
                    "first_input_timestamp": int(first_time),
                    "final_input_timestamp": int(final_time),
                    "target_timestamp": int(target_time),
                }
            )

    n_windows = len(targets)
    n_dynamic = len(dynamic_features)
    n_static = len(static_features)
    X_dynamic = (
        np.stack(dynamic_rows)
        if dynamic_rows
        else np.empty((0, history_steps, n_dynamic), dtype=np.float32)
    )
    X_static = (
        np.stack(static_rows)
        if static_rows
        else np.empty((0, n_static), dtype=np.float32)
    )
    observation_mask = (
        np.stack(masks)
        if masks
        else np.empty((0, history_steps, n_dynamic), dtype=bool)
    )
    y_bis = np.asarray(targets, dtype=np.float32)
    return WindowDataset(
        X_dynamic=X_dynamic,
        X_static=X_static,
        observation_mask=observation_mask,
        y_bis=y_bis,
        y_high_bis=(y_bis > high_bis_threshold).astype(np.int8),
        y_low_bis=(y_bis < low_bis_threshold).astype(np.int8),
        metadata=pd.DataFrame(
            metadata_rows,
            columns=[
                "case_id",
                "first_input_timestamp",
                "final_input_timestamp",
                "target_timestamp",
            ],
        ),
        windows_removed_missing_future_bis=removed_missing_future,
    )

