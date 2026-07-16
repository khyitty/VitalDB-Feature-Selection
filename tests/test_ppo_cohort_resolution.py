"""Colab-safe PPO cohort resolution and frozen-output safety tests."""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_ppo_experiment
from src.rl_training.cohort import CohortPreflightError, load_vitaldb_virtual_cohort
from src.rl_training.manifests import (
    build_frozen_protocol,
    freeze_protocol,
    protocol_hash,
)


ROOT = Path(__file__).parents[1]


def _rows() -> list[dict[str, object]]:
    return [
        {"caseid": 1, "age": 30, "sex_male": 1, "height": 170, "weight": 70},
        {"caseid": 2, "age": 40, "sex_male": 0, "height": 165, "weight": 65},
        {"caseid": 3, "age": 50, "sex_male": 1, "height": 175, "weight": 80},
        {"caseid": 4, "age": 60, "sex_male": 0, "height": 160, "weight": 60},
    ]


def _fixture_dataset(
    tmp_path: Path,
    *,
    embedded: bool = False,
    source_rows: list[dict[str, object]] | None = None,
) -> tuple[Path, Path, Path]:
    data_root = tmp_path / "drive_project/data"
    dataset = data_root / "modeling/full"
    split_dir = dataset / "splits"
    split_dir.mkdir(parents=True)
    split_values = {"train": [1, 2], "val": [3], "test": [4]}
    for split, caseids in split_values.items():
        pd.DataFrame({"caseid": caseids}).to_csv(
            split_dir / f"{split}_cases.csv", index=False
        )
    rows = source_rows if source_rows is not None else _rows()
    by_case = {int(row["caseid"]): row for row in rows if "caseid" in row}
    for split, caseids in split_values.items():
        if embedded:
            metadata = pd.DataFrame([by_case[caseid] for caseid in caseids])
        else:
            metadata = pd.DataFrame(
                {
                    "case_id": caseids,
                    "first_input_timestamp": [0] * len(caseids),
                    "final_input_timestamp": [50] * len(caseids),
                    "target_timestamp": [80] * len(caseids),
                }
            )
        metadata.to_csv(dataset / f"{split}_metadata.csv", index=False)
    source = data_root / "processed/cohort.csv"
    source.parent.mkdir(parents=True)
    if not embedded:
        pd.DataFrame(rows).to_csv(source, index=False)
    (dataset / "dataset_metadata.json").write_text(
        json.dumps(
            {
                "input_file": r"C:\stale\Windows\data\processed\cohort.csv",
                "cases_per_split": {"train": 2, "val": 1, "test": 1},
            }
        ),
        encoding="utf-8",
    )
    (dataset / "test.npz").write_bytes(b"must-not-open")
    return dataset, data_root, source


def test_loader_has_no_repository_root_or_cwd_data_fallback(tmp_path: Path) -> None:
    dataset, data_root, _ = _fixture_dataset(tmp_path)
    assert "repo_dir" not in inspect.signature(load_vitaldb_virtual_cohort).parameters
    bundle = load_vitaldb_virtual_cohort(dataset, project_data_root=data_root)
    assert bundle.demographics_source == "data/processed/cohort.csv"


def test_embedded_modeling_metadata_is_used_before_external_sources(tmp_path: Path) -> None:
    dataset, _, source = _fixture_dataset(tmp_path, embedded=True)
    assert not source.exists()
    bundle = load_vitaldb_virtual_cohort(dataset)
    assert bundle.demographics_source_kind == "modeling_metadata"
    assert len(bundle.patient_records) == 4


def test_dataset_metadata_json_demographics_are_supported(tmp_path: Path) -> None:
    dataset, _, source = _fixture_dataset(tmp_path)
    source.unlink()
    metadata = json.loads((dataset / "dataset_metadata.json").read_text())
    metadata["case_demographics"] = _rows()
    (dataset / "dataset_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    bundle = load_vitaldb_virtual_cohort(dataset)
    assert bundle.demographics_source_kind == "dataset_metadata_json"


def test_explicit_demographics_csv_is_used(tmp_path: Path) -> None:
    dataset, _, source = _fixture_dataset(tmp_path)
    bundle = load_vitaldb_virtual_cohort(dataset, demographics_csv=source)
    assert bundle.demographics_source_kind == "explicit_csv"
    assert bundle.access_manifest["selected_demographics_path"] == str(source.resolve())


def test_project_data_root_resolves_windows_metadata_path_by_filename(tmp_path: Path) -> None:
    dataset, data_root, source = _fixture_dataset(tmp_path)
    bundle = load_vitaldb_virtual_cohort(dataset, project_data_root=data_root)
    assert Path(bundle.access_manifest["selected_demographics_path"]) == source.resolve()


def test_missing_source_preflight_lists_paths_columns_and_required_input(tmp_path: Path) -> None:
    dataset, data_root, source = _fixture_dataset(tmp_path)
    source.unlink()
    with pytest.raises(CohortPreflightError) as caught:
        load_vitaldb_virtual_cohort(dataset, project_data_root=data_root)
    message = str(caught.value)
    assert "Required canonical columns" in message
    assert "Modeling metadata columns" in message
    assert "Searched candidate paths" in message
    assert "--demographics-csv" in message


def test_required_demographic_column_missing_is_reported(tmp_path: Path) -> None:
    rows = _rows()
    for row in rows:
        row.pop("weight")
    dataset, _, source = _fixture_dataset(tmp_path, source_rows=rows)
    with pytest.raises(CohortPreflightError, match="weight"):
        load_vitaldb_virtual_cohort(dataset, demographics_csv=source)


def test_every_split_case_requires_demographics(tmp_path: Path) -> None:
    dataset, _, source = _fixture_dataset(tmp_path, source_rows=_rows()[:-1])
    with pytest.raises(CohortPreflightError, match="missing split caseid"):
        load_vitaldb_virtual_cohort(dataset, demographics_csv=source)


def test_split_overlap_is_rejected(tmp_path: Path) -> None:
    dataset, data_root, _ = _fixture_dataset(tmp_path)
    pd.DataFrame({"caseid": [2]}).to_csv(
        dataset / "splits/val_cases.csv", index=False
    )
    with pytest.raises(ValueError, match="overlap"):
        load_vitaldb_virtual_cohort(dataset, project_data_root=data_root)


def test_duplicate_demographics_are_collapsed_but_conflicts_fail(tmp_path: Path) -> None:
    rows = _rows()
    dataset, _, source = _fixture_dataset(tmp_path, source_rows=rows + [dict(rows[0])])
    bundle = load_vitaldb_virtual_cohort(dataset, demographics_csv=source)
    assert bundle.access_manifest["duplicate_demographics_rows_collapsed"] == 1
    conflicting = dict(rows[0])
    conflicting["weight"] = 90
    pd.DataFrame(rows + [conflicting]).to_csv(source, index=False)
    with pytest.raises(CohortPreflightError, match="Conflicting demographic rows"):
        load_vitaldb_virtual_cohort(dataset, demographics_csv=source)


def test_train_only_imputation_never_uses_test_statistics(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["weight"] = None
    rows[2]["age"] = None
    rows[3]["age"] = None
    dataset, _, source = _fixture_dataset(tmp_path, source_rows=rows)
    bundle = load_vitaldb_virtual_cohort(
        dataset, demographics_csv=source, missing_policy="train_impute"
    )
    assert bundle.imputation_statistics["age"] == 35.0
    assert bundle.imputation_statistics["weight"] == 65.0
    assert bundle.access_manifest["test_data_used_for_imputation"] is False
    assert bundle.cohort.patient("4").age_years == 35.0


def test_initialize_never_reads_test_trajectory_or_evaluates_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, data_root, _ = _fixture_dataset(tmp_path)
    monkeypatch.setattr(
        "src.rl_training.cohort.np.load",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("trajectory opened")),
        raising=False,
    )
    bundle = load_vitaldb_virtual_cohort(dataset, project_data_root=data_root)
    assert bundle.access_manifest["test_split_membership_loaded"] is True
    assert bundle.access_manifest["test_demographics_loaded"] is True
    assert bundle.access_manifest["test_trajectory_loaded"] is False
    assert bundle.access_manifest["test_outcomes_evaluated"] is False
    assert bundle.access_manifest["test_policy_rollout_performed"] is False


def test_cohort_fingerprint_is_deterministic(tmp_path: Path) -> None:
    dataset, data_root, _ = _fixture_dataset(tmp_path)
    first = load_vitaldb_virtual_cohort(dataset, project_data_root=data_root)
    second = load_vitaldb_virtual_cohort(dataset, project_data_root=data_root)
    assert first.fingerprint == second.fingerprint
    assert first.demographics_source_fingerprint == second.demographics_source_fingerprint


def test_matching_existing_protocol_is_reused_across_implementation_commit(
    tmp_path: Path,
) -> None:
    dataset, data_root, _ = _fixture_dataset(tmp_path)
    bundle = load_vitaldb_virtual_cohort(dataset, project_data_root=data_root)
    requested = build_frozen_protocol(repo_dir=ROOT, cohort=bundle)
    observed = copy.deepcopy(requested)
    observed["implementation_commit_at_creation"] = "0" * 40
    observed["protocol_hash"] = protocol_hash(observed)
    protocol_dir = tmp_path / "protocol"
    freeze_protocol(observed, protocol_dir, run_output_root=tmp_path / "runs")
    reused = freeze_protocol(requested, protocol_dir, run_output_root=tmp_path / "runs")
    assert reused["protocol_hash"] == observed["protocol_hash"]


def test_mismatching_protocol_is_rejected_without_overwrite(tmp_path: Path) -> None:
    dataset, data_root, _ = _fixture_dataset(tmp_path)
    bundle = load_vitaldb_virtual_cohort(dataset, project_data_root=data_root)
    protocol = build_frozen_protocol(repo_dir=ROOT, cohort=bundle)
    protocol_dir = tmp_path / "protocol"
    freeze_protocol(protocol, protocol_dir, run_output_root=tmp_path / "runs")
    before = (protocol_dir / "frozen_ppo_protocol.json").read_bytes()
    changed = copy.deepcopy(protocol)
    changed["ppo"]["gamma"] = 0.5
    changed["protocol_hash"] = protocol_hash(changed)
    with pytest.raises(ValueError, match="was not deleted or modified"):
        freeze_protocol(changed, protocol_dir, run_output_root=tmp_path / "runs")
    assert (protocol_dir / "frozen_ppo_protocol.json").read_bytes() == before


def test_partial_protocol_is_atomically_repaired_only_without_run_output(tmp_path: Path) -> None:
    dataset, data_root, _ = _fixture_dataset(tmp_path)
    bundle = load_vitaldb_virtual_cohort(dataset, project_data_root=data_root)
    protocol = build_frozen_protocol(repo_dir=ROOT, cohort=bundle)
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    protocol_path = protocol_dir / "frozen_ppo_protocol.json"
    protocol_path.write_text("{partial", encoding="utf-8")
    repaired = freeze_protocol(protocol, protocol_dir, run_output_root=tmp_path / "runs")
    assert repaired["protocol_hash"] == protocol["protocol_hash"]
    assert not list(protocol_dir.glob("*.tmp"))


def test_partial_protocol_with_run_output_is_preserved_and_rejected(tmp_path: Path) -> None:
    dataset, data_root, _ = _fixture_dataset(tmp_path)
    bundle = load_vitaldb_virtual_cohort(dataset, project_data_root=data_root)
    protocol = build_frozen_protocol(repo_dir=ROOT, cohort=bundle)
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    protocol_path = protocol_dir / "frozen_ppo_protocol.json"
    protocol_path.write_text("{partial", encoding="utf-8")
    run_file = tmp_path / "runs/all_supported/seed_7/config.json"
    run_file.parent.mkdir(parents=True)
    run_file.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing repair"):
        freeze_protocol(protocol, protocol_dir, run_output_root=tmp_path / "runs")
    assert protocol_path.read_text(encoding="utf-8") == "{partial"


def test_initialize_only_fixture_succeeds_and_calls_training_zero_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, data_root, _ = _fixture_dataset(tmp_path)
    calls: list[object] = []
    monkeypatch.setattr(run_ppo_experiment, "run_experiment", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(run_ppo_experiment, "write_policy_contract_artifacts", lambda **kwargs: None)
    run_ppo_experiment.main(
        [
            "--dataset-dir",
            str(dataset),
            "--project-data-root",
            str(data_root),
            "--protocol-dir",
            str(tmp_path / "protocol"),
            "--output-root",
            str(tmp_path / "runs"),
            "--initialize-only",
        ]
    )
    assert calls == []
    protocol = json.loads((tmp_path / "protocol/frozen_ppo_protocol.json").read_text())
    assert protocol["inventory_count"] == 20


def test_confirmation_failure_calls_training_zero_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, data_root, _ = _fixture_dataset(tmp_path)
    calls: list[object] = []
    monkeypatch.setattr(run_ppo_experiment, "run_experiment", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(run_ppo_experiment, "write_policy_contract_artifacts", lambda **kwargs: None)
    with pytest.raises(ValueError, match="Full training remains locked"):
        run_ppo_experiment.main(
            [
                "--dataset-dir",
                str(dataset),
                "--project-data-root",
                str(data_root),
                "--protocol-dir",
                str(tmp_path / "protocol"),
                "--output-root",
                str(tmp_path / "runs"),
                "--confirmation",
                "wrong",
            ]
        )
    assert calls == []


def test_colab_notebooks_use_drive_paths_and_robust_error_reporting() -> None:
    full = (ROOT / "notebooks/colab_ppo_full_training.ipynb").read_text(encoding="utf-8")
    validation = (ROOT / "notebooks/colab_ppo_validation_analysis.ipynb").read_text(
        encoding="utf-8"
    )
    for source in (full, validation):
        assert "data/modeling/full" in source
        assert "project_data_root" in source
        assert "run_command" in source
        assert "Last 200 lines" in source
        assert "test.npz" not in source
    assert "repo / 'data/modeling/full'" not in full
    assert "--project-data-root" in full
    assert "cohort_access_manifest.json" in validation
