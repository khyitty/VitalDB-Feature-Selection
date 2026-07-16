"""Checkpoint-bound control-attention artifact persistence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_attention_artifact(
    path: Path,
    *,
    feature_attention: np.ndarray,
    temporal_attention: np.ndarray,
    history_mask: np.ndarray,
    feature_names: tuple[str, ...],
    scenario_ids: np.ndarray,
    bis: np.ndarray,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Save explicit weights with feature/lag order and exact checkpoint hash."""

    feature = np.asarray(feature_attention, dtype=np.float32)
    temporal = np.asarray(temporal_attention, dtype=np.float32)
    mask = np.asarray(history_mask, dtype=bool)
    if feature.ndim != 3 or temporal.shape != feature.shape[:2] or mask.shape != temporal.shape:
        raise ValueError("Attention arrays require [N,L,P], [N,L], and [N,L] shapes.")
    if feature.shape[2] != len(feature_names):
        raise ValueError("Feature-name order does not match attention width.")
    if not np.isfinite(feature).all() or not np.isfinite(temporal).all():
        raise FloatingPointError("Attention artifact contains non-finite weights.")
    if np.max(np.abs(feature.sum(axis=2)[mask] - 1.0), initial=0.0) > 1e-5:
        raise ValueError("Feature attention is not normalized at valid history rows.")
    if np.max(np.abs(temporal.sum(axis=1) - 1.0), initial=0.0) > 1e-5:
        raise ValueError("Temporal attention is not normalized over valid lags.")
    if np.count_nonzero(feature[~mask]) or np.count_nonzero(temporal[~mask]):
        raise ValueError("Padded history positions must have exactly zero attention.")
    checkpoint_hash = file_sha256(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        feature_attention=feature,
        temporal_attention=temporal,
        combined_attention=temporal[:, :, None] * feature,
        history_mask=mask,
        feature_names=np.asarray(feature_names),
        lag_seconds=np.asarray([-50, -40, -30, -20, -10, 0], dtype=np.int32),
        scenario_ids=np.asarray(scenario_ids),
        bis=np.asarray(bis, dtype=np.float32),
        checkpoint_sha256=np.asarray(checkpoint_hash),
        interpretation=np.asarray("model-internal control attention; not causal effect"),
    )
    metadata = {
        "path": path.as_posix(),
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": checkpoint_hash,
        "feature_names": list(feature_names),
        "lag_seconds": [-50, -40, -30, -20, -10, 0],
        "rows": int(feature.shape[0]),
        "predictive_checkpoint_transfer": False,
        "causal_effect_claim": False,
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def verify_attention_checkpoint(artifact_path: Path, checkpoint_path: Path) -> None:
    with np.load(artifact_path, allow_pickle=False) as archive:
        observed = str(archive["checkpoint_sha256"].item())
    expected = file_sha256(checkpoint_path)
    if observed != expected:
        raise ValueError(
            f"Attention/checkpoint hash mismatch: observed={observed}, expected={expected}."
        )
