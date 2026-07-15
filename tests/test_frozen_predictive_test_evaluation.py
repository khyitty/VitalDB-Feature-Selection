"""Synthetic tests for the one-time frozen predictive test workflow."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.frozen_candidate_retraining import MODELS, SEEDS, sha256_file
from src.frozen_predictive_test_evaluation import (
    CANDIDATES,
    CONFIRMATION_PHRASE,
    PRIMARY_CANDIDATE,
    PRIMARY_FEATURES,
    REFERENCE_CANDIDATE,
    REFERENCE_FEATURES,
    build_checkpoint_inventory,
    paired_test_statistics,
    prepare_test_preflight,
    run_frozen_predictive_test_evaluation,
    validate_frozen_decision,
)
from src.rl_handoff import REQUIRED_EXTERNAL_INPUTS, run_rl_handoff_audit


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _add_test_split(dataset_dir: Path) -> None:
    targets = np.asarray([42.0, 44.0, 50.0, 52.0, 62.0, 64.0], dtype=np.float32)
    np.savez_compressed(
        dataset_dir / "test.npz",
        X_dynamic=np.zeros((6, 6, 18), dtype=np.float32),
        X_static=np.zeros((6, 2), dtype=np.float32),
        observation_mask=np.ones((6, 6, 18), dtype=bool),
        y_bis=targets,
        y_high_bis=(targets > 60).astype(np.int8),
        y_low_bis=(targets < 40).astype(np.int8),
    )
    pd.DataFrame(
        {
            "case_id": [201, 201, 202, 202, 203, 203],
            "target_timestamp": np.arange(100, 160, 10),
        }
    ).to_csv(dataset_dir / "test_metadata.csv", index=False)


@pytest.fixture
def frozen_test_workspace(
    synthetic_frozen_candidate_workspace: dict[str, object], tmp_path: Path
) -> dict[str, Any]:
    workspace = synthetic_frozen_candidate_workspace
    dataset_dir = Path(workspace["dataset_dir"])
    _add_test_split(dataset_dir)
    strict_root = Path(workspace["output_root"]) / PRIMARY_CANDIDATE
    write_complete_run = workspace["write_complete_run"]
    features = workspace["features"]
    for model in MODELS:
        for seed in SEEDS:
            run_dir = strict_root / model / f"seed_{seed}"
            write_complete_run(
                run_dir,
                model=model,
                seed=seed,
                features=list(features[PRIMARY_CANDIDATE]),
                dataset_dir=dataset_dir,
                commit="00d27c03fd9cfd79e99a7a474631b85ea6d02735",
                error=0.8 + 0.01 * SEEDS.index(seed),
            )
            if model == "attention":
                config_path = run_dir / "config.json"
                config = json.loads(config_path.read_text(encoding="utf-8"))
                config["model_name"] = "FactorizedAttentionGRU"
                _write_json(config_path, config)
    for seed in SEEDS:
        config_path = (
            Path(workspace["group_root"])
            / "full17"
            / "attention"
            / f"seed_{seed}"
            / "config.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["model_name"] = "FactorizedAttentionGRU"
        _write_json(config_path, config)

    analysis_manifest = Path(workspace["group_analysis_dir"]) / "analysis_manifest.json"
    template_dir = tmp_path / "decision_template"
    decision = {
        "decision_timestamp_utc": "2026-07-15T00:00:00+00:00",
        "decision_code_commit": "00d27c03fd9cfd79e99a7a474631b85ea6d02735",
        "source_analysis_manifest_sha256": sha256_file(analysis_manifest),
        "primary_candidate": PRIMARY_CANDIDATE,
        "reference_candidate": REFERENCE_CANDIDATE,
        "secondary_validation_only_candidate": "compact_consensus",
        "test_evaluation_candidates": list(CANDIDATES),
        "primary_dynamic_feature_names": list(PRIMARY_FEATURES),
        "reference_dynamic_feature_names": list(REFERENCE_FEATURES),
        "pre_test_freeze": True,
        "candidate_changes_after_test_prohibited": True,
    }
    _write_json(template_dir / "frozen_predictive_decision.json", decision)
    (template_dir / "frozen_predictive_decision.md").write_text(
        "# Synthetic pre-test freeze\n", encoding="utf-8"
    )
    workspace.update(
        {
            "strict_root": strict_root,
            "full17_root": Path(workspace["group_root"]) / "full17",
            "analysis_manifest": analysis_manifest,
            "template_dir": template_dir,
            "decision_dir": tmp_path / "frozen_predictive_decision_30s",
            "test_output": tmp_path / "frozen_predictive_test_evaluation_30s",
        }
    )
    return workspace


def _prediction_callback(
    record: dict[str, Any], dataset_dir: Path, device: str, batch_size: int
) -> pd.DataFrame:
    del dataset_dir, device, batch_size
    targets = np.asarray([42.0, 44.0, 50.0, 52.0, 62.0, 64.0])
    candidate_error = 0.8 if record["candidate"] == PRIMARY_CANDIDATE else 1.0
    model_error = -0.05 if record["model"] == "attention" else 0.0
    seed_error = 0.01 * SEEDS.index(int(record["seed"]))
    return pd.DataFrame(
        {
            "sample_index": np.arange(6),
            "case_id": [201, 201, 202, 202, 203, 203],
            "target_timestamp": np.arange(100, 160, 10),
            "observed_future_bis": targets,
            "predicted_future_bis": targets
            + candidate_error
            + model_error
            + seed_error,
        }
    )


def _kwargs(workspace: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_template_dir": Path(workspace["template_dir"]),
        "decision_dir": Path(workspace["decision_dir"]),
        "source_analysis_manifest": Path(workspace["analysis_manifest"]),
        "dataset_dir": Path(workspace["dataset_dir"]),
        "strict_root": Path(workspace["strict_root"]),
        "full17_root": Path(workspace["full17_root"]),
        "output_dir": Path(workspace["test_output"]),
    }


def test_preflight_freezes_decision_before_test_and_lists_exactly_20(
    frozen_test_workspace: dict[str, Any],
) -> None:
    result = prepare_test_preflight(**_kwargs(frozen_test_workspace))
    output = Path(frozen_test_workspace["test_output"])
    inventory = pd.read_csv(output / "evaluated_checkpoint_inventory.csv")
    decision = json.loads((output / "frozen_decision_snapshot.json").read_text())

    assert result["test_split_opened"] is False
    assert decision["primary_candidate"] == PRIMARY_CANDIDATE
    assert decision["reference_candidate"] == REFERENCE_CANDIDATE
    assert decision["test_evaluation_candidates"] == list(CANDIDATES)
    assert "compact_consensus" not in set(inventory["candidate"])
    assert len(inventory) == 20
    assert set(inventory["checkpoint_name"]) == {"best_model.pt"}
    assert inventory["checkpoint_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert inventory["dataset_fingerprint"].nunique() == 1
    assert not (output / "test_evaluation_contract.json").exists()


def test_decision_rejects_candidate_or_feature_changes(
    frozen_test_workspace: dict[str, Any],
) -> None:
    decision = json.loads(
        (Path(frozen_test_workspace["template_dir"]) / "frozen_predictive_decision.json").read_text()
    )
    decision["primary_candidate"] = "compact_consensus"
    with pytest.raises(ValueError, match="strict_consensus"):
        validate_frozen_decision(decision)


def test_exact_confirmation_is_required(frozen_test_workspace: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=CONFIRMATION_PHRASE):
        run_frozen_predictive_test_evaluation(
            **_kwargs(frozen_test_workspace),
            confirmation="yes",
            inference_fn=_prediction_callback,
            bootstrap_replicates=20,
        )
    assert not Path(frozen_test_workspace["test_output"]).exists()


def test_full_synthetic_evaluation_is_patient_grouped_paired_and_source_immutable(
    frozen_test_workspace: dict[str, Any],
) -> None:
    inventory_before = build_checkpoint_inventory(
        Path(frozen_test_workspace["dataset_dir"]),
        Path(frozen_test_workspace["strict_root"]),
        Path(frozen_test_workspace["full17_root"]),
    )
    result = run_frozen_predictive_test_evaluation(
        **_kwargs(frozen_test_workspace),
        confirmation=CONFIRMATION_PHRASE,
        inference_fn=_prediction_callback,
        bootstrap_replicates=40,
        bootstrap_seed=123,
    )
    output = Path(frozen_test_workspace["test_output"])
    run_level = pd.read_csv(output / "test_run_level_metrics.csv")
    patients = pd.read_csv(output / "patient_level_test_metrics.csv")
    deltas = pd.read_csv(output / "paired_test_seed_deltas.csv")
    bootstrap = pd.read_csv(output / "hierarchical_bootstrap_test_contrasts.csv")
    manifest = json.loads((output / "test_evaluation_manifest.json").read_text())
    inventory_after = build_checkpoint_inventory(
        Path(frozen_test_workspace["dataset_dir"]),
        Path(frozen_test_workspace["strict_root"]),
        Path(frozen_test_workspace["full17_root"]),
    )

    assert result == {"status": "complete", "skipped": False, "run_count": 20}
    assert len(run_level) == 20
    assert len(patients) == 20 * 3
    assert len(deltas) == 4 * 5
    assert len(bootstrap) == 4
    assert manifest["training_performed"] is False
    assert manifest["checkpoint_reselection_performed"] is False
    assert manifest["compact_consensus_tested"] is False
    assert manifest["primary_candidate_remains_frozen"] == PRIMARY_CANDIDATE
    pd.testing.assert_frame_equal(inventory_before, inventory_after)


def test_partial_resume_runs_only_missing_inference(
    frozen_test_workspace: dict[str, Any],
) -> None:
    calls: list[tuple[str, str, int]] = []

    def interrupted(record: dict[str, Any], *args: Any) -> pd.DataFrame:
        key = (str(record["candidate"]), str(record["model"]), int(record["seed"]))
        calls.append(key)
        if len(calls) == 4:
            raise RuntimeError("synthetic interruption")
        return _prediction_callback(record, *args)

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_frozen_predictive_test_evaluation(
            **_kwargs(frozen_test_workspace),
            confirmation=CONFIRMATION_PHRASE,
            inference_fn=interrupted,
            bootstrap_replicates=20,
        )
    resumed: list[tuple[str, str, int]] = []

    def resume(record: dict[str, Any], *args: Any) -> pd.DataFrame:
        resumed.append(
            (str(record["candidate"]), str(record["model"]), int(record["seed"]))
        )
        return _prediction_callback(record, *args)

    result = run_frozen_predictive_test_evaluation(
        **_kwargs(frozen_test_workspace),
        confirmation=CONFIRMATION_PHRASE,
        inference_fn=resume,
        bootstrap_replicates=20,
    )
    assert result["status"] == "complete"
    assert len(resumed) == 17


def test_complete_compatible_evaluation_skips_all_inference(
    frozen_test_workspace: dict[str, Any],
) -> None:
    common = {
        **_kwargs(frozen_test_workspace),
        "confirmation": CONFIRMATION_PHRASE,
        "inference_fn": _prediction_callback,
        "bootstrap_replicates": 20,
    }
    run_frozen_predictive_test_evaluation(**common)

    def forbidden(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise AssertionError("inference should have been skipped")

    common["inference_fn"] = forbidden
    result = run_frozen_predictive_test_evaluation(**common)
    assert result == {"status": "complete", "skipped": True, "run_count": 20}


def test_changed_checkpoint_rejects_incompatible_rerun(
    frozen_test_workspace: dict[str, Any],
) -> None:
    prepare_test_preflight(**_kwargs(frozen_test_workspace))
    checkpoint = (
        Path(frozen_test_workspace["strict_root"]) / "gru" / "seed_7" / "best_model.pt"
    )
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="overwrite incompatible|inventory changed"):
        run_frozen_predictive_test_evaluation(
            **_kwargs(frozen_test_workspace),
            confirmation=CONFIRMATION_PHRASE,
            inference_fn=_prediction_callback,
            bootstrap_replicates=20,
        )


def test_hierarchical_bootstrap_is_reproducible() -> None:
    run_rows = []
    patient_rows = []
    for candidate in CANDIDATES:
        for model in MODELS:
            for seed in SEEDS:
                value = 1.0 + (0.2 if candidate == REFERENCE_CANDIDATE else 0.0)
                value += 0.1 if model == "gru" else 0.0
                run_rows.append(
                    {
                        "candidate": candidate,
                        "model": model,
                        "seed": seed,
                        "test_patient_level_mae": value,
                    }
                )
                for case_id in (1, 2, 3):
                    patient_rows.append(
                        {
                            "candidate": candidate,
                            "model": model,
                            "seed": seed,
                            "case_id": case_id,
                            "mae": value + case_id * 0.01,
                        }
                    )
    args = (pd.DataFrame(run_rows), pd.DataFrame(patient_rows))
    first = paired_test_statistics(*args, replicates=100, seed=42)
    second = paired_test_statistics(*args, replicates=100, seed=42)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])


def test_colab_notebook_has_lock_and_no_training_command() -> None:
    path = Path("notebooks/colab_frozen_predictive_test.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
    assert CONFIRMATION_PHRASE in code
    assert "--preflight-only" in code
    assert "evaluated_checkpoint_inventory.csv" in code
    assert "run_baselines.py" not in code
    assert "run_attention.py" not in code
    assert "--max-epochs" not in code


def test_rl_audit_blocks_without_external_implementation(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    dataset = repo / "data" / "modeling" / "full"
    dataset.mkdir(parents=True)
    _write_json(
        dataset / "dataset_metadata.json",
        {"static_feature_names": ["age", "sex_male"]},
    )
    (dataset / "preprocessing.pkl").write_bytes(b"train-only")
    (repo / "main.py").write_text(
        "def crop_to_propofol_period(frame):\n    return frame\n", encoding="utf-8"
    )
    output = repo / "outputs" / "rl_handoff"
    result = run_rl_handoff_audit(repo, dataset, output)
    audit = json.loads((output / "rl_repository_audit.json").read_text())
    contract = json.loads((output / "rl_state_contract.json").read_text())
    mapping = pd.read_csv(output / "predictive_to_control_feature_mapping.csv")

    assert result["blocked_missing_external_rl_implementation"] is True
    assert result["rl_training_started"] is False
    assert audit["professor_rl_implementation_found"] is False
    assert len(audit["required_external_inputs"]) == len(REQUIRED_EXTERNAL_INPUTS)
    assert contract["predictive_compact_state"]["dynamic_feature_names_ordered"] == list(
        PRIMARY_FEATURES
    )
    assert contract["control_aware_state"][
        "must_not_replace_baseline_with_predictive_subset"
    ] is True
    assert mapping["predictive_feature"].tolist() == list(PRIMARY_FEATURES)
