"""Contracts for the primary-state PPO full study and device benchmark."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_ppo_state_full import validate_confirmation
from src.rl_training.device_benchmark import analyze_device_benchmarks
from src.rl_training.full_analysis import (
    hierarchical_bootstrap_intervals,
    run_full_analysis,
)
from src.rl_training.full_experiment import _assert_fresh_or_full_resume
from src.rl_training.full_protocol import (
    FULL_CONFIRMATION,
    FULL_PROFILES,
    FULL_SEEDS,
    build_full_protocol,
    freeze_full_protocol,
    full_protocol_hash,
    load_full_source,
    select_full_inventory,
    source_config_sha256,
    verify_full_protocol,
)
from src.rl_training.pilot_experiment import _completion_is_valid


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "configs/ppo_primary_state_full.json"
NOTEBOOK = ROOT / "notebooks/colab_ppo_primary_state_full_training.ipynb"


@pytest.fixture(scope="module")
def full_cohort(ppo_test_cohort):
    return ppo_test_cohort


@pytest.fixture(scope="module")
def full_protocol(full_cohort):
    from src.rl_training.run_status import repository_commit

    decision = {
        "implementation_commit": repository_commit(ROOT),
        "source_config_sha256": source_config_sha256(SOURCE),
        "cohort_fingerprint": full_cohort.fingerprint,
        "selected_backend": "cpu",
        "scientific_metrics_used_for_backend_selection": False,
    }
    return build_full_protocol(
        source_path=SOURCE,
        repo_dir=ROOT,
        cohort=full_cohort,
        execution_device="cpu",
        backend_decision=decision,
    )


def _make_path_resolver_inputs(root: Path) -> tuple[Path, Path]:
    dataset_dir = root / "data/modeling/full"
    dataset_dir.mkdir(parents=True)
    demographics_csv = root / "data/raw/clinical.csv"
    demographics_csv.parent.mkdir(parents=True)
    demographics_csv.write_text("caseid,age,sex,height,weight\n", encoding="utf-8")
    return dataset_dir, demographics_csv


def test_ppo_test_paths_use_dataset_environment_override(
    tmp_path: Path, ppo_test_path_resolver
) -> None:
    _, fallback_demographics = _make_path_resolver_inputs(tmp_path / "repo")
    external_dataset = tmp_path / "drive/modeling/full"
    external_dataset.mkdir(parents=True)
    resolved = ppo_test_path_resolver(
        repository_root=tmp_path / "repo",
        environ={"VITALDB_MODELING_DATASET_DIR": str(external_dataset)},
    )
    assert resolved.dataset_dir == external_dataset.resolve()
    assert resolved.demographics_csv == fallback_demographics.resolve()


def test_ppo_test_paths_use_demographics_environment_override(
    tmp_path: Path, ppo_test_path_resolver
) -> None:
    fallback_dataset, _ = _make_path_resolver_inputs(tmp_path / "repo")
    external_demographics = tmp_path / "drive/raw/clinical.csv"
    external_demographics.parent.mkdir(parents=True)
    external_demographics.write_text(
        "caseid,age,sex,height,weight\n", encoding="utf-8"
    )
    resolved = ppo_test_path_resolver(
        repository_root=tmp_path / "repo",
        environ={"VITALDB_DEMOGRAPHICS_CSV": str(external_demographics)},
    )
    assert resolved.dataset_dir == fallback_dataset.resolve()
    assert resolved.demographics_csv == external_demographics.resolve()


def test_ppo_test_paths_fall_back_to_repository_local_inputs(
    tmp_path: Path, ppo_test_path_resolver
) -> None:
    dataset_dir, demographics_csv = _make_path_resolver_inputs(tmp_path / "repo")
    resolved = ppo_test_path_resolver(
        repository_root=tmp_path / "repo", environ={}
    )
    assert resolved.dataset_dir == dataset_dir.resolve()
    assert resolved.demographics_csv == demographics_csv.resolve()
    assert resolved.project_data_root is None


def test_ppo_test_paths_support_project_data_root(
    tmp_path: Path, ppo_test_path_resolver
) -> None:
    project_data_root = tmp_path / "drive/data"
    dataset_dir = project_data_root / "modeling/full"
    dataset_dir.mkdir(parents=True)
    demographics_csv = project_data_root / "raw/clinical.csv"
    demographics_csv.parent.mkdir(parents=True)
    demographics_csv.write_text(
        "caseid,age,sex,height,weight\n", encoding="utf-8"
    )
    resolved = ppo_test_path_resolver(
        repository_root=tmp_path / "repo",
        environ={"VITALDB_PROJECT_DATA_ROOT": str(project_data_root)},
    )
    assert resolved.dataset_dir == dataset_dir.resolve()
    assert resolved.demographics_csv == demographics_csv.resolve()
    assert resolved.project_data_root == project_data_root.resolve()


def test_ppo_test_paths_fail_clearly_for_invalid_environment_override(
    tmp_path: Path, ppo_test_path_resolver
) -> None:
    _, demographics_csv = _make_path_resolver_inputs(tmp_path / "repo")
    missing = tmp_path / "missing/modeling/full"
    with pytest.raises(FileNotFoundError) as error:
        ppo_test_path_resolver(
            repository_root=tmp_path / "repo",
            environ={
                "VITALDB_MODELING_DATASET_DIR": str(missing),
                "VITALDB_DEMOGRAPHICS_CSV": str(demographics_csv),
            },
        )
    message = str(error.value)
    assert "VITALDB_MODELING_DATASET_DIR" in message
    assert str(missing.resolve()) in message
    assert "Repository-local defaults" in message


def test_full_source_is_exact_non_smoke_inventory() -> None:
    source = load_full_source(SOURCE)
    assert tuple(source["profiles"]) == FULL_PROFILES
    assert tuple(source["seeds"]) == FULL_SEEDS
    assert source["ppo"]["total_timesteps"] == 1_024_000
    assert source["ppo"]["evaluation_frequency_timesteps"] == 51_200
    assert source["ppo"]["n_steps"] == 2_048
    assert source["initialization"] == {
        "mode": "fresh_random",
        "pilot_checkpoint_reuse": False,
        "pilot_output_import": False,
    }


def test_full_protocol_hash_inventory_pairing_and_test_seal(full_protocol) -> None:
    verify_full_protocol(full_protocol)
    assert full_protocol["protocol_hash"] == full_protocol_hash(full_protocol)
    assert full_protocol["inventory_count"] == 20
    assert len(full_protocol["cohort_contract"]["validation_scenario_ids"]) == 15
    assert full_protocol["cohort_contract"]["case_counts"] == {
        "train": 68,
        "validation": 15,
        "test": 15,
    }
    assert full_protocol["test_seal"]["test_trajectory_loaded"] is False
    assert full_protocol["test_seal"]["test_outcomes_evaluated"] is False
    signatures = {
        json.dumps(contract["architecture_signature"], sort_keys=True)
        for contract in full_protocol["policy_contracts"].values()
    }
    assert len(signatures) == 1


def test_full_protocol_mutation_and_pilot_directory_are_rejected(
    tmp_path: Path, full_protocol
) -> None:
    corrupted = json.loads(json.dumps(full_protocol))
    corrupted["ppo"]["gamma"] = 0.9
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_full_protocol(corrupted)
    with pytest.raises(ValueError, match="separate"):
        freeze_full_protocol(
            full_protocol,
            tmp_path / "ppo_primary_state_pilot/protocol",
            run_output_root=tmp_path / "full/runs",
        )


def test_full_inventory_subsets_and_confirmation(full_protocol) -> None:
    subset = select_full_inventory(
        full_protocol,
        profiles=["prediction_minimal"],
        seeds=[7, 123],
    )
    assert [(row["state_profile"], row["seed"]) for row in subset] == [
        ("prediction_minimal", 7),
        ("prediction_minimal", 123),
    ]
    with pytest.raises(ValueError, match="outside the full"):
        select_full_inventory(full_protocol, seeds=[999])
    with pytest.raises(ValueError, match=FULL_CONFIRMATION):
        validate_confirmation(None)
    validate_confirmation(FULL_CONFIRMATION)


def test_fresh_initialization_rejects_pilot_or_unidentified_checkpoint(
    tmp_path: Path, full_protocol
) -> None:
    run_dir = tmp_path / "all_supported/seed_7"
    run_dir.mkdir(parents=True)
    (run_dir / "resume_model.zip").write_bytes(b"pilot")
    with pytest.raises(ValueError, match="pilot checkpoint reuse"):
        _assert_fresh_or_full_resume(
            run_dir,
            protocol_hash=full_protocol["protocol_hash"],
            state_profile="all_supported",
            seed=7,
        )
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "workflow": "primary_state_ppo_pilot",
                "protocol_hash": full_protocol["protocol_hash"],
                "state_profile": "all_supported",
                "seed": 7,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pilot reuse is forbidden"):
        _assert_fresh_or_full_resume(
            run_dir,
            protocol_hash=full_protocol["protocol_hash"],
            state_profile="all_supported",
            seed=7,
        )


def test_compatible_full_resume_config_is_accepted(tmp_path: Path, full_protocol) -> None:
    run_dir = tmp_path / "all_supported/seed_7"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "workflow": "primary_state_ppo_full",
                "protocol_hash": full_protocol["protocol_hash"],
                "state_profile": "all_supported",
                "seed": 7,
                "initialization_source": "fresh_random",
                "pilot_checkpoint_used": False,
            }
        ),
        encoding="utf-8",
    )
    _assert_fresh_or_full_resume(
        run_dir,
        protocol_hash=full_protocol["protocol_hash"],
        state_profile="all_supported",
        seed=7,
    )


def test_completed_full_skip_requires_all_twenty_evaluations(
    tmp_path: Path, full_protocol
) -> None:
    run_dir = tmp_path / "all_supported/seed_7"
    run_dir.mkdir(parents=True)
    for name in (
        "best_model.zip",
        "best_checkpoint.json",
        "training_progress.csv",
        "evaluation_progress.csv",
        "action_diagnostics.csv",
    ):
        (run_dir / name).write_text("x", encoding="utf-8")
    steps = range(51_200, 1_024_001, 51_200)
    for step in steps:
        (run_dir / f"checkpoint_{step}.zip").write_text("x", encoding="utf-8")
        (run_dir / f"validation_{step}.csv").write_text("x", encoding="utf-8")
    (run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "protocol_hash": full_protocol["protocol_hash"],
                "resolved_config": {"workflow": "primary_state_ppo_full"},
            }
        ),
        encoding="utf-8",
    )
    completion = {
        "status": "complete",
        "protocol_hash": full_protocol["protocol_hash"],
        "cohort_fingerprint": full_protocol["cohort_contract"]["fingerprint"],
        "state_profile": "all_supported",
        "seed": 7,
        "timesteps": 1_024_000,
    }
    assert _completion_is_valid(
        completion,
        protocol=full_protocol,
        run_dir=run_dir,
        state_profile="all_supported",
        seed=7,
        workflow="primary_state_ppo_full",
    )
    (run_dir / "checkpoint_51200.zip").unlink()
    assert not _completion_is_valid(
        completion,
        protocol=full_protocol,
        run_dir=run_dir,
        state_profile="all_supported",
        seed=7,
        workflow="primary_state_ppo_full",
    )


def test_hierarchical_bootstrap_preserves_seed_patient_structure() -> None:
    rows = []
    for seed in FULL_SEEDS:
        for patient in range(15):
            rows.append(
                {
                    "state_profile": "all_supported",
                    "reference_profile": "original_reconstructed",
                    "training_seed": seed,
                    "patient_id": str(patient),
                    "metric": "bis_target_mae",
                    "difference_candidate_minus_original": -0.1 - seed / 100_000,
                }
            )
    result = hierarchical_bootstrap_intervals(
        pd.DataFrame(rows), repeats=200, random_seed=7
    )
    assert len(result) == 1
    assert result.iloc[0]["training_seed_count"] == 5
    assert result.iloc[0]["validation_patient_count_per_seed"] == 15
    assert result.iloc[0]["p_value_reported"] == False  # noqa: E712
    assert result.iloc[0]["hierarchical_bootstrap_ci95_upper"] < 0


def _fake_benchmark_rows(device: str, seconds: float) -> pd.DataFrame:
    rows = []
    for profile in ("all_supported", "selected_control_core"):
        for repeat in range(1, 4):
            rows.append(
                {
                    "schema_version": 1,
                    "implementation_commit": "a" * 40,
                    "source_config_sha256": "b" * 64,
                    "cohort_fingerprint": "c" * 64,
                    "device": device,
                    "state_profile": profile,
                    "repeat_index": repeat,
                    "training_wall_seconds": seconds + repeat,
                    "training_steps_per_second": 20_480 / (seconds + repeat),
                    "resume_verified": True,
                    "metric_schema_verified": True,
                    "numerical_failure_count": 0,
                }
            )
    return pd.DataFrame(rows)


def test_benchmark_schema_and_backend_rule(tmp_path: Path) -> None:
    cpu = tmp_path / "cpu.csv"
    cuda = tmp_path / "cuda.csv"
    _fake_benchmark_rows("cpu", 100.0).to_csv(cpu, index=False)
    _fake_benchmark_rows("cuda", 60.0).to_csv(cuda, index=False)
    decision = analyze_device_benchmarks(
        result_files=[cpu, cuda], output_dir=tmp_path / "analysis"
    )
    assert decision["selected_backend"] == "cuda"
    assert decision["cuda_qualifies"] is True
    assert decision["scientific_metrics_used_for_backend_selection"] is False
    slower = tmp_path / "cuda_slower.csv"
    _fake_benchmark_rows("cuda", 80.0).to_csv(slower, index=False)
    decision = analyze_device_benchmarks(
        result_files=[cpu, slower], output_dir=tmp_path / "analysis_slower"
    )
    assert decision["selected_backend"] == "cpu"


def test_pending_full_analysis_writes_schema(tmp_path: Path, full_protocol) -> None:
    result = run_full_analysis(
        protocol=full_protocol,
        output_root=tmp_path / "runs",
        analysis_dir=tmp_path / "analysis",
        bootstrap_repeats=20,
    )
    assert result["completed_runs"] == 0
    assert len(result["pending_runs"]) == 20
    expected = {
        "run_level_summary.csv",
        "evaluation_checkpoint_summary.csv",
        "patient_level_paired_differences.csv",
        "profile_five_seed_mean_sd.csv",
        "hierarchical_bootstrap_intervals.csv",
        "full_analysis_manifest.json",
        "full_validation_report.md",
    }
    assert expected.issubset(path.name for path in (tmp_path / "analysis").iterdir())


def test_colab_notebook_is_clean_json_and_python_ast() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
    for cell in code_cells:
        ast.parse("".join(cell["source"]))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "RUN_FULL_TRAINING = False" in source
    assert "RUN_20_PRIMARY_STATE_FULL_RUNS" in source
    assert "pilot_checkpoint" not in source.lower()
    assert "test.npz" in source
    assert "np.load" not in source
    assert 'env["VITALDB_MODELING_DATASET_DIR"]' in source
    assert 'env["VITALDB_DEMOGRAPHICS_CSV"]' in source
    assert 'env["VITALDB_PROJECT_DATA_ROOT"]' in source
    assert 'env["VITALDB_EXPECTED_COHORT_FINGERPRINT"]' in source
