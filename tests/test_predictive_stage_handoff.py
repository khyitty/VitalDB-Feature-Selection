"""Tests for the compact, reference-only predictive handoff package."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.frozen_predictive_test_evaluation import (
    CANDIDATES,
    MODELS,
    PRIMARY_CANDIDATE,
    PRIMARY_FEATURES,
    REFERENCE_CANDIDATE,
    REFERENCE_FEATURES,
    SEEDS,
)
from src.predictive_stage_handoff import build_predictive_stage_handoff


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _decision(path: Path) -> None:
    _write_json(
        path,
        {
            "decision_timestamp_utc": "2026-01-01T00:00:00+00:00",
            "decision_code_commit": "a" * 40,
            "source_analysis_manifest_sha256": "b" * 64,
            "primary_candidate": PRIMARY_CANDIDATE,
            "reference_candidate": REFERENCE_CANDIDATE,
            "secondary_validation_only_candidate": "compact_consensus",
            "test_evaluation_candidates": list(CANDIDATES),
            "primary_dynamic_feature_names": list(PRIMARY_FEATURES),
            "reference_dynamic_feature_names": list(REFERENCE_FEATURES),
            "pre_test_freeze": True,
            "candidate_changes_after_test_prohibited": True,
        },
    )


def _inventory(path: Path) -> None:
    rows = []
    for candidate in CANDIDATES:
        features = PRIMARY_FEATURES if candidate == PRIMARY_CANDIDATE else REFERENCE_FEATURES
        for model in MODELS:
            for seed in SEEDS:
                rows.append(
                    {
                        "candidate": candidate,
                        "model": model,
                        "seed": seed,
                        "source_run_directory": f"external/{candidate}/{model}/seed_{seed}",
                        "checkpoint_path": f"external/{candidate}/{model}/seed_{seed}/best_model.pt",
                        "checkpoint_name": "best_model.pt",
                        "checkpoint_sha256": f"{len(rows) + 1:064x}",
                        "config_sha256": f"{len(rows) + 101:064x}",
                        "training_git_commit": "c" * 40,
                        "dynamic_feature_names": json.dumps(list(features), separators=(",", ":")),
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def handoff_inputs(tmp_path: Path) -> dict[str, Path]:
    dataset = tmp_path / "data"
    dataset.mkdir()
    _write_json(
        dataset / "dataset_metadata.json",
        {
            "static_feature_names": ["age", "sex_male", "height", "weight", "bmi", "asa"],
            "resampling_interval_seconds": 10,
            "history_window_seconds": 60,
            "history_steps": 6,
            "prediction_horizon_seconds": 30,
        },
    )
    (dataset / "preprocessing.pkl").write_bytes(b"train-fitted")
    decision = tmp_path / "decision.json"
    inventory = tmp_path / "inventory.csv"
    _decision(decision)
    _inventory(inventory)
    return {
        "dataset": dataset,
        "decision": decision,
        "inventory": inventory,
        "output": tmp_path / "handoff",
    }


def test_handoff_records_hashes_lineage_and_feature_order_without_large_files(
    handoff_inputs: dict[str, Path],
) -> None:
    result = build_predictive_stage_handoff(
        repo_dir=Path.cwd(),
        dataset_dir=handoff_inputs["dataset"],
        decision_path=handoff_inputs["decision"],
        checkpoint_inventory_path=handoff_inputs["inventory"],
        output_dir=handoff_inputs["output"],
    )
    output = handoff_inputs["output"]
    expected = {
        "predictive_stage_manifest.json",
        "predictive_state_contract.json",
        "frozen_candidate_definition.json",
        "checkpoint_inventory.csv",
        "experiment_lineage.csv",
        "reproducibility_checklist.md",
        "predictive_stage_summary.md",
        "external_rl_assets_request.md",
    }
    assert {path.name for path in output.iterdir()} == expected
    manifest = json.loads((output / "predictive_stage_manifest.json").read_text())
    state = json.loads((output / "predictive_state_contract.json").read_text())
    lineage = pd.read_csv(output / "experiment_lineage.csv")
    inventory = pd.read_csv(output / "checkpoint_inventory.csv")
    assert result["checkpoint_hashes_complete"] is True
    assert manifest["large_artifacts_copied"] is False
    assert len(manifest["checkpoint_fingerprints"]) == 20
    assert all(
        len(item["checkpoint_sha256"]) == 64
        for item in manifest["checkpoint_fingerprints"]
    )
    assert state["dynamic_feature_names_ordered"] == list(PRIMARY_FEATURES)
    assert state["dynamic_tensor_shape"] == ["batch", 6, 7]
    assert len(inventory) == 20
    assert inventory["large_artifact_copied"].eq(False).all()
    assert "frozen_predictive_decision" in set(lineage["stage"])
    assert "rl_audit_blocked" in set(lineage["stage"])
    assert not any(path.suffix in {".pt", ".npz", ".pkl"} for path in output.iterdir())


def test_handoff_without_drive_inventory_reports_pending_hashes(
    handoff_inputs: dict[str, Path],
) -> None:
    result = build_predictive_stage_handoff(
        repo_dir=Path.cwd(),
        dataset_dir=handoff_inputs["dataset"],
        decision_path=handoff_inputs["decision"],
        checkpoint_inventory_path=None,
        output_dir=handoff_inputs["output"],
    )
    inventory = pd.read_csv(handoff_inputs["output"] / "checkpoint_inventory.csv")
    assert result["checkpoint_hashes_complete"] is False
    assert set(inventory["hash_status"]) == {"pending_drive_preflight"}


def test_handoff_rejects_changed_feature_order(handoff_inputs: dict[str, Path]) -> None:
    inventory = pd.read_csv(handoff_inputs["inventory"])
    inventory.loc[0, "dynamic_feature_names"] = json.dumps(list(reversed(PRIMARY_FEATURES)))
    inventory.to_csv(handoff_inputs["inventory"], index=False)
    with pytest.raises(ValueError, match="Feature order"):
        build_predictive_stage_handoff(
            repo_dir=Path.cwd(),
            dataset_dir=handoff_inputs["dataset"],
            decision_path=handoff_inputs["decision"],
            checkpoint_inventory_path=handoff_inputs["inventory"],
            output_dir=handoff_inputs["output"],
        )
