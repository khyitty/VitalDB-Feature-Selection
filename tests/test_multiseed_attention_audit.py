"""Focused tests for paired multiseed attention aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.multiseed_attention_audit import (
    ATTENTION_REQUIRED,
    GRU_REQUIRED,
    case_balanced_group_attention,
    discover_complete_paired_runs,
    paired_patient_bootstrap,
    pairwise_vector_stability,
    top_k_jaccard,
    validate_seed_feature_alignment,
    validate_temporal_lag_alignment,
)
from src.redundancy_audit import REDUCED_FEATURES


def _fake_complete_pair(root: Path, seed: int, features: list[str]) -> None:
    for model, required in (("gru", GRU_REQUIRED), ("attention", ATTENTION_REQUIRED)):
        run_dir = root / model / f"seed_{seed}"
        run_dir.mkdir(parents=True)
        for name in required:
            (run_dir / name).touch()
        (run_dir / "config.json").write_text(
            json.dumps({"seed": seed, "dynamic_feature_names": features}),
            encoding="utf-8",
        )


def test_discovers_only_complete_paired_runs_with_exact_feature_order(
    tmp_path: Path,
) -> None:
    _fake_complete_pair(tmp_path, 7, list(REDUCED_FEATURES))

    pairs = discover_complete_paired_runs(tmp_path, (7,))

    assert len(pairs) == 1
    assert pairs[0].seed == 7


def test_rejects_incomplete_and_eighteen_feature_runs(tmp_path: Path) -> None:
    _fake_complete_pair(tmp_path, 7, [*REDUCED_FEATURES, "bis_error"])
    with pytest.raises(ValueError, match="exact 17-feature order"):
        discover_complete_paired_runs(tmp_path, (7,))

    incomplete_root = tmp_path / "incomplete"
    _fake_complete_pair(incomplete_root, 21, list(REDUCED_FEATURES))
    (incomplete_root / "attention" / "seed_21" / "test_attention.npz").unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete attention run"):
        discover_complete_paired_runs(incomplete_root, (21,))


def test_seed_feature_and_temporal_alignment() -> None:
    seeds = (7, 21)
    feature = pd.DataFrame(
        [(seed, name) for seed in seeds for name in REDUCED_FEATURES],
        columns=["seed", "feature"],
    )
    validate_seed_feature_alignment(feature, seeds)
    with pytest.raises(ValueError, match="misaligned"):
        validate_seed_feature_alignment(feature.iloc[:-1], seeds)

    lags = (-50, -40, -30, -20, -10, 0)
    temporal = pd.DataFrame(
        [(seed, lag) for seed in seeds for lag in lags],
        columns=["seed", "time_lag_seconds"],
    )
    validate_temporal_lag_alignment(temporal, seeds)
    with pytest.raises(ValueError, match="misaligned"):
        validate_temporal_lag_alignment(temporal.iloc[:-1], seeds)


def test_pairwise_rank_correlation_and_top_k_jaccard() -> None:
    vectors = {
        7: np.array([0.5, 0.3, 0.2, 0.0]),
        21: np.array([0.4, 0.35, 0.1, 0.15]),
        42: np.array([0.1, 0.2, 0.3, 0.4]),
    }
    pairwise = pairwise_vector_stability(vectors, top_ks=(2,))

    assert len(pairwise) == 3
    assert np.isfinite(pairwise["spearman_rank_correlation"]).all()
    assert np.isfinite(pairwise["cosine_similarity"]).all()
    assert top_k_jaccard(vectors[7], vectors[21], 2) == 1.0
    assert top_k_jaccard(vectors[7], vectors[42], 2) == 0.0


def test_group_sum_occurs_before_equal_case_averaging() -> None:
    attention = np.array(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ]
    )
    case_ids = np.array([1, 1, 2])
    means, case_values = case_balanced_group_attention(
        attention,
        case_ids,
        ["a", "b"],
        {"first": ("a",), "both": ("a", "b")},
    )

    assert means["first"] == pytest.approx(0.5)
    assert means["first"] != pytest.approx(2 / 3)
    assert np.array_equal(case_values["first"], np.array([1.0, 0.0]))
    assert means["both"] == pytest.approx(1.0)


def test_patient_bootstrap_is_paired_deterministic_and_patient_level() -> None:
    differences = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    first = paired_patient_bootstrap(differences, replicates=2_000, seed=17)
    second = paired_patient_bootstrap(differences, replicates=2_000, seed=17)

    assert first == second
    assert first["point_estimate_mean_attention_minus_gru_mae"] == pytest.approx(0.0)
    assert first["percentile_95_ci_lower"] < 0
    assert first["percentile_95_ci_upper"] > 0
    assert first["resampling_unit"] == "test patient"
