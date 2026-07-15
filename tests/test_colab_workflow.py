"""Focused tests for the reproducible Google Colab GPU workflow."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.run_attention import parse_args as parse_attention_args
from scripts.run_baselines import parse_args as parse_baseline_args
from src.colab_workflow import (
    ATTENTION_SMOKE_REQUIRED,
    GPU_BENCHMARK_COLUMNS,
    GRU_SMOKE_REQUIRED,
    audit_colab_environment,
    inspect_run_completion,
    validate_colab_requirements,
    validate_gpu_benchmark_schema,
    validate_modeling_artifacts,
    validate_pip_install_plan,
)
from src.redundancy_audit import REDUCED_FEATURES
from src.training import TrainingConfig, resolve_device, run_gru_training


def test_colab_dependencies_never_request_pytorch_replacement(tmp_path: Path) -> None:
    project_requirements = Path("requirements-colab.txt")
    requirements = validate_colab_requirements(project_requirements)
    assert requirements
    assert all("torch" not in line.lower() for line in requirements)

    unsafe = tmp_path / "unsafe.txt"
    unsafe.write_text("numpy\ntorch==2.0+cpu\n", encoding="utf-8")
    with pytest.raises(ValueError, match="retain preinstalled CUDA PyTorch"):
        validate_colab_requirements(unsafe)

    with pytest.raises(RuntimeError, match="installation aborted"):
        validate_pip_install_plan(
            {"install": [{"metadata": {"name": "torchvision"}}]}
        )


def test_training_clis_keep_explicit_colab_output_directories() -> None:
    gru = parse_baseline_args(
        [
            "gru",
            "--output-dir",
            "/content/drive/MyDrive/project/outputs/gru",
            "--validation-only",
        ]
    )
    attention = parse_attention_args(
        [
            "--output-dir",
            "/content/drive/MyDrive/project/outputs/attention",
            "--validation-only",
        ]
    )
    assert gru.output_dir.parts[-2:] == ("outputs", "gru")
    assert attention.output_dir.parts[-2:] == ("outputs", "attention")
    assert gru.validation_only is True
    assert attention.validation_only is True


def test_explicit_cuda_request_raises_instead_of_falling_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Runtime > Change runtime type > GPU"):
        resolve_device("cuda")
    assert resolve_device("auto") == torch.device("cpu")


def test_environment_audit_serializes_gpu_not_assigned_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr("src.colab_workflow._nvidia_smi", lambda: {"path": None, "output": None, "return_code": None})
    monkeypatch.setattr("src.colab_workflow._git_commit", lambda _: "abc123")
    output = tmp_path / "environment_audit.json"

    payload = audit_colab_environment(output, tmp_path, require_cuda=False)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert payload == saved
    assert saved["classification"] == "B. COLAB GPU NOT ASSIGNED"
    assert saved["git_commit"] == "abc123"


def test_modeling_artifact_validation_and_fixed_feature_order(
    synthetic_modeling_dir: Path,
) -> None:
    with (synthetic_modeling_dir / "preprocessing.pkl").open("wb") as handle:
        pickle.dump({"fitted_on": "train"}, handle)
    pd.DataFrame(
        {"feature_name": [*REDUCED_FEATURES, "bis_error"]}
    ).to_csv(synthetic_modeling_dir / "feature_manifest.csv", index=False)

    audit = validate_modeling_artifacts(synthetic_modeling_dir)
    assert audit["resolved_dynamic_features"] == list(REDUCED_FEATURES)
    assert audit["static_feature_count"] == 6
    assert audit["case_level_split_integrity"] is True
    assert all(split["all_arrays_finite"] for split in audit["splits"].values())
    assert all(split["metadata_row_alignment"] for split in audit["splits"].values())


def test_complete_and_incomplete_smoke_directories_are_distinguished(tmp_path: Path) -> None:
    run_dir = tmp_path / "gru"
    run_dir.mkdir()
    for name in GRU_SMOKE_REQUIRED:
        (run_dir / name).write_text("{}", encoding="utf-8")
    (run_dir / "run_status.json").write_text(
        json.dumps({"status": "running"}), encoding="utf-8"
    )
    assert inspect_run_completion(run_dir, "gru")["complete"] is False

    (run_dir / "run_status.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    assert inspect_run_completion(run_dir, "gru")["complete"] is True
    (run_dir / "best_model.pt").unlink()
    status = inspect_run_completion(run_dir, "gru")
    assert status["complete"] is False
    assert status["missing_artifacts"] == ["best_model.pt"]


def test_gpu_benchmark_schema_accepts_success_and_rejects_missing_columns() -> None:
    row = {column: 1 for column in GPU_BENCHMARK_COLUMNS}
    row.update(
        {
            "model": "gru",
            "device": "cuda",
            "gpu_name": "test-gpu",
            "status": "ok",
        }
    )
    frame = pd.DataFrame([row])
    validate_gpu_benchmark_schema(frame)
    with pytest.raises(ValueError, match="missing columns"):
        validate_gpu_benchmark_schema(frame.drop(columns="gpu_name"))


def test_gru_smoke_is_validation_only_and_marks_complete(
    synthetic_modeling_dir: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "explicit-drive-output" / "gru"
    result = run_gru_training(
        TrainingConfig(
            dataset_dir=synthetic_modeling_dir,
            output_dir=output_dir,
            seed=42,
            device="cpu",
            batch_size=4,
            max_epochs=1,
            patience=1,
            hidden_size=8,
            projection_size=8,
            static_hidden_size=4,
            prediction_hidden_size=4,
            smoke=True,
            exclude_dynamic_features=("bis_error",),
        )
    )
    status = json.loads((output_dir / "run_status.json").read_text(encoding="utf-8"))
    assert result["test_tensor_shape"] is None
    assert status["status"] == "complete"
    assert status["test_evaluated"] is False
    assert not (output_dir / "test_predictions.csv").exists()
    assert not (output_dir / "test_metrics.json").exists()
    assert tuple(json.loads((output_dir / "config.json").read_text())["dynamic_feature_names"]) == REDUCED_FEATURES


def test_full_gru_can_keep_test_split_sealed(
    synthetic_modeling_dir: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "validation-only-full" / "gru"
    result = run_gru_training(
        TrainingConfig(
            dataset_dir=synthetic_modeling_dir,
            output_dir=output_dir,
            seed=42,
            device="cpu",
            batch_size=4,
            max_epochs=1,
            patience=1,
            hidden_size=8,
            projection_size=8,
            static_hidden_size=4,
            prediction_hidden_size=4,
            evaluate_test=False,
            exclude_dynamic_features=("bis_error",),
        )
    )

    status = json.loads((output_dir / "run_status.json").read_text(encoding="utf-8"))
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["smoke"] is False
    assert config["evaluate_test"] is False
    assert status["test_evaluated"] is False
    assert result["test_tensor_shape"] is None
    assert not (output_dir / "test_predictions.csv").exists()
    assert not (output_dir / "test_metrics.json").exists()


def test_notebook_is_valid_json_with_generic_drive_placeholders() -> None:
    notebook_path = Path("notebooks/colab_gpu_setup.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert notebook["nbformat"] == 4
    assert "/content/drive/MyDrive/VitalDB-Feature-Selection" in source
    assert "--device', 'cuda" in source
    assert "group-retraining" not in source.lower()
    assert set(ATTENTION_SMOKE_REQUIRED) >= {"val_attention.npz", "run_status.json"}


def test_full_training_notebook_is_locked_and_validation_only() -> None:
    notebook_path = Path("notebooks/colab_full_training.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    source = "\n".join(code_cells)

    assert notebook["nbformat"] == 4
    assert "RUN_FULL_TRAINING = False" in source
    assert "RUN_40_VALIDATION_ONLY_RUNS" in source
    assert "--validation-only" in source
    assert "--device', 'cuda'" in source
    assert "torch.cuda.is_available()" in source
    assert "--smoke" not in source
    assert "EXPECTED_RUN_COUNT == 40" in source
    assert "no_remifentanil_or_respiratory" in source
    assert "test_evaluated') is not False" in source
    for index, cell_source in enumerate(code_cells):
        compile(cell_source, f"colab_full_training_cell_{index}", "exec")
