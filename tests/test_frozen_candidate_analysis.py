"""Synthetic tests for validation-only frozen-candidate comparison."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.frozen_candidate_analysis import (
    PRIMARY_METRIC,
    aggregate_candidates,
    candidate_pareto,
    load_registry,
    paired_candidate_statistics,
    patient_metrics,
    run_frozen_candidate_analysis,
    validate_candidate_inventory,
)
from src.frozen_candidate_retraining import (
    MODELS,
    SEEDS,
    build_retraining_plan,
    dump_json,
)


def _complete_inventory(workspace: dict[str, Any]) -> tuple[Path, Path]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = build_retraining_plan(
        candidate_path=workspace["candidate_path"],
        dataset_dir=workspace["dataset_dir"],
        group_root=workspace["group_root"],
        group_analysis_dir=workspace["group_analysis_dir"],
        output_root=workspace["output_root"],
        repo_dir=Path.cwd(),
        active_commit=commit,
    )
    errors = {"strict_consensus": 0.6, "compact_consensus": 0.75}
    for record in result["new_runs"]:
        workspace["write_complete_run"](
            Path(record["source_run_directory"]),
            model=record["model"],
            seed=record["seed"],
            features=record["feature_names"],
            dataset_dir=workspace["dataset_dir"],
            commit=commit,
            error=(
                errors[record["candidate"]]
                + 0.01 * SEEDS.index(record["seed"])
                - (0.1 if record["model"] == "attention" else 0.0)
            ),
        )
        record["completion_status"] = "complete"
    registry_path = workspace["output_root"] / "candidate_source_registry.json"
    dump_json(result["registry"], registry_path)
    return registry_path, workspace["output_root"] / "analysis"


def _hash_registered_runs(registry_path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for row in load_registry(registry_path):
        run_dir = Path(row["source_run_directory"])
        for path in run_dir.rglob("*"):
            if path.is_file():
                key = f"{row['candidate']}/{row['model']}/{row['seed']}/{path.name}"
                hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def test_exact_50_run_inventory_integrates_30_reused_and_20_new(
    synthetic_frozen_candidate_workspace: dict[str, Any],
) -> None:
    workspace = synthetic_frozen_candidate_workspace
    registry_path, _ = _complete_inventory(workspace)
    summary, predictions = validate_candidate_inventory(
        registry_path, workspace["candidate_path"], workspace["dataset_dir"]
    )
    assert len(summary) == 50
    assert len(predictions) == 50
    assert (summary["source_type"] == "reused_prior").sum() == 30
    assert (summary["source_type"] == "newly_trained").sum() == 20
    assert summary["test_evaluated"].eq(False).all()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_source"])
def test_registry_missing_duplicate_and_wrong_source_are_rejected(
    synthetic_frozen_candidate_workspace: dict[str, Any], mutation: str
) -> None:
    workspace = synthetic_frozen_candidate_workspace
    registry_path, _ = _complete_inventory(workspace)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        registry.pop()
    elif mutation == "duplicate":
        registry.append(dict(registry[0]))
    else:
        registry[0]["source_type"] = "newly_trained"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="50 unique|source type|30 reused"):
        load_registry(registry_path)


def test_prediction_alignment_mismatch_is_rejected(
    synthetic_frozen_candidate_workspace: dict[str, Any],
) -> None:
    workspace = synthetic_frozen_candidate_workspace
    registry_path, _ = _complete_inventory(workspace)
    registry = load_registry(registry_path)
    run_dir = Path(registry[-1]["source_run_directory"])
    path = run_dir / "val_predictions.csv"
    predictions = pd.read_csv(path)
    predictions.loc[0, "target_timestamp"] += 10
    predictions.to_csv(path, index=False)
    with pytest.raises(ValueError, match="alignment mismatch"):
        validate_candidate_inventory(
            registry_path, workspace["candidate_path"], workspace["dataset_dir"]
        )


def test_paired_delta_patient_aggregation_and_bootstrap_are_reproducible(
    synthetic_frozen_candidate_workspace: dict[str, Any],
) -> None:
    workspace = synthetic_frozen_candidate_workspace
    registry_path, _ = _complete_inventory(workspace)
    run_summary, predictions = validate_candidate_inventory(
        registry_path, workspace["candidate_path"], workspace["dataset_dir"]
    )
    patients = patient_metrics(predictions)
    assert len(patients) == 50 * 3
    first = paired_candidate_statistics(run_summary, patients, replicates=100, seed=123)
    second = paired_candidate_statistics(run_summary, patients, replicates=100, seed=123)
    deltas, contrasts, bootstrap = first
    selected = deltas.query(
        "model == 'gru' and candidate == 'strict_consensus' and reference == 'full17_reference'"
    )
    assert len(selected) == 5
    assert np.allclose(selected["delta"], -0.4)
    assert contrasts.query(
        "model == 'gru' and candidate == 'strict_consensus' and reference == 'full17_reference'"
    )["candidate_better_seed_count"].item() == 5
    pd.testing.assert_frame_equal(bootstrap, second[2])


def test_pareto_marks_distinct_decision_aids() -> None:
    rows = []
    candidate_values = {
        "full17_reference": (17, 1.0),
        "no_respiratory_anchor": (15, 1.1),
        "compact11_anchor": (11, 0.9),
        "strict_consensus": (7, 0.95),
        "compact_consensus": (11, 0.85),
    }
    for model in MODELS:
        for candidate, (count, value) in candidate_values.items():
            for seed in SEEDS:
                rows.append(
                    {
                        "candidate": candidate,
                        "model": model,
                        "seed": seed,
                        "feature_count": count,
                        PRIMARY_METRIC: value,
                    }
                )
    summary = pd.DataFrame(rows)
    aggregate = aggregate_candidates(summary)
    contrast_rows = [
        {
            "model": model,
            "candidate": candidate,
            "reference": "full17_reference",
            "candidate_better_seed_count": 5 if candidate == "compact_consensus" else 3,
            "mean_delta": value - 1.0,
        }
        for model in MODELS
        for candidate, (_, value) in candidate_values.items()
        if candidate != "full17_reference"
    ]
    pareto = candidate_pareto(aggregate, pd.DataFrame(contrast_rows))
    for model in MODELS:
        selected = pareto[pareto["model"] == model].set_index("candidate")
        assert bool(selected.loc["compact_consensus", "best_observed_validation_candidate"])
        assert bool(selected.loc["strict_consensus", "simplest_non_dominated_candidate"])
        assert bool(selected.loc["compact_consensus", "most_seed_consistent_candidate"])
        assert bool(selected.loc["full17_reference", "dominated"])


def test_full_analysis_writes_outputs_without_modifying_source_runs(
    synthetic_frozen_candidate_workspace: dict[str, Any],
) -> None:
    workspace = synthetic_frozen_candidate_workspace
    registry_path, analysis_dir = _complete_inventory(workspace)
    before = _hash_registered_runs(registry_path)
    result = run_frozen_candidate_analysis(
        registry_path,
        workspace["candidate_path"],
        workspace["dataset_dir"],
        analysis_dir,
        bootstrap_replicates=50,
        bootstrap_seed=123,
    )
    after = _hash_registered_runs(registry_path)
    assert before == after
    assert result == {
        "output_dir": str(analysis_dir),
        "run_count": 50,
        "test_split_sealed": True,
    }
    manifest = json.loads((analysis_dir / "analysis_manifest.json").read_text())
    report = (analysis_dir / "frozen_candidate_validation_report.md").read_text()
    assert manifest["run_count"] == 50
    assert manifest["adaptive_validation_warning"] is True
    assert manifest["test_split_read_by_analysis"] is False
    assert len(manifest["input_fingerprints"]) == 2 + 5 + 50 * 5
    assert "Adaptive validation warning" in report
    assert "test split remains sealed" in report
    assert "non-inferiority" in report
    assert len(list((analysis_dir / "figures").glob("*.png"))) == 6


def test_analysis_notebook_is_cpu_only_valid_and_has_no_training_commands() -> None:
    notebook_path = Path("notebooks/colab_frozen_candidate_analysis.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    source = "\n".join(code_cells)
    script = Path("scripts/analyze_frozen_candidates.py").read_text(encoding="utf-8")
    assert notebook["nbformat"] == 4
    assert "scripts/analyze_frozen_candidates.py" in source
    assert "run_baselines.py" not in source
    assert "run_attention.py" not in source
    assert "torch.cuda" not in source
    assert "test.npz" not in source
    assert "test.npz" not in script
    for index, cell_source in enumerate(code_cells):
        compile(cell_source, f"frozen_analysis_cell_{index}", "exec")
