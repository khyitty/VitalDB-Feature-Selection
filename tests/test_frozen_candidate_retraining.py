"""Synthetic tests for frozen-candidate validation-only retraining planning."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zlib
from pathlib import Path
from typing import Any

import pytest

from src.frozen_candidate_retraining import (
    ANCHOR_MAPPING,
    FIXED_SETTINGS,
    FROZEN_CANDIDATES,
    NEW_CANDIDATES,
    build_retraining_plan,
    build_training_command,
    load_frozen_candidates,
    require_same_git_commit,
    resolve_git_commit,
    validate_resume_compatibility,
    validate_run_directory,
    verify_training_source_compatibility,
)


def _full_commit(reference: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", reference],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_loose_git_object(repo_dir: Path, payload: bytes) -> str:
    raw = b"commit " + str(len(payload)).encode("ascii") + b"\0" + payload
    object_id = hashlib.sha1(raw).hexdigest()
    path = repo_dir / ".git" / "objects" / object_id[:2] / object_id[2:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zlib.compress(raw))
    return object_id


def _ambiguous_commit_prefix(repo_dir: Path) -> str:
    subprocess.run(["git", "init", "--quiet", str(repo_dir)], check=True)
    seen: dict[str, tuple[str, bytes]] = {}
    for index in range(10_000):
        payload = (
            "tree 4b825dc642cb6eb9a060e54bf8d69288fbee4904\n"
            "author Synthetic <synthetic@example.com> 1700000000 +0000\n"
            "committer Synthetic <synthetic@example.com> 1700000000 +0000\n"
            f"\ncollision {index}\n"
        ).encode("ascii")
        raw = b"commit " + str(len(payload)).encode("ascii") + b"\0" + payload
        object_id = hashlib.sha1(raw).hexdigest()
        prefix = object_id[:4]
        previous = seen.get(prefix)
        if previous is not None and previous[0] != object_id:
            _write_loose_git_object(repo_dir, previous[1])
            _write_loose_git_object(repo_dir, payload)
            return prefix
        seen[prefix] = (object_id, payload)
    raise AssertionError("Could not construct an ambiguous four-character commit prefix.")


def _plan(workspace: dict[str, Any], *, write_outputs: bool = True) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return build_retraining_plan(
        candidate_path=workspace["candidate_path"],
        dataset_dir=workspace["dataset_dir"],
        group_root=workspace["group_root"],
        group_analysis_dir=workspace["group_analysis_dir"],
        output_root=workspace["output_root"],
        repo_dir=Path.cwd(),
        active_commit=commit,
        write_outputs=write_outputs,
    )


def test_candidate_artifact_names_counts_and_exact_features(
    synthetic_frozen_candidate_workspace: dict[str, Any],
) -> None:
    workspace = synthetic_frozen_candidate_workspace
    candidates = load_frozen_candidates(workspace["candidate_path"])
    assert tuple(candidates.features) == FROZEN_CANDIDATES
    assert candidates.features == {
        name: tuple(features) for name, features in workspace["features"].items()
    }
    assert [len(candidates.features[name]) for name in FROZEN_CANDIDATES] == [
        17,
        15,
        11,
        7,
        11,
    ]


def test_candidate_artifact_rejects_changed_discovery_features(
    synthetic_frozen_candidate_workspace: dict[str, Any],
) -> None:
    path = synthetic_frozen_candidate_workspace["candidate_path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["all_candidate_subsets"]["strict_consensus"]["features"][0] = "hr"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="feature fingerprint mismatch|feature order"):
        load_frozen_candidates(path)


def test_anchor_mapping_and_training_sources_are_compatible() -> None:
    assert ANCHOR_MAPPING == {
        "full17_reference": "full17",
        "no_respiratory_anchor": "no_respiratory",
        "compact11_anchor": "no_remifentanil_or_respiratory",
    }
    hashes = verify_training_source_compatibility(Path.cwd())
    assert len(hashes) == 8
    assert all(len(digest) == 64 for digest in hashes.values())


@pytest.mark.parametrize(
    ("expected_form", "observed_form"),
    (("short", "full"), ("full", "short"), ("full", "full")),
)
def test_equivalent_short_and_full_commit_forms_are_accepted(
    expected_form: str, observed_form: str
) -> None:
    short = "3387a7e"
    full = _full_commit(short)
    expected = short if expected_form == "short" else full
    observed = short if observed_form == "short" else full
    assert require_same_git_commit(
        Path.cwd(), expected, observed, context="synthetic prior anchor"
    ) == full


def test_different_git_commits_are_rejected_with_both_canonical_shas() -> None:
    expected = _full_commit("3387a7e")
    observed = _full_commit("3387a7e^")
    with pytest.raises(ValueError, match=f"full={expected}.*full={observed}"):
        require_same_git_commit(
            Path.cwd(), "3387a7e", observed, context="synthetic prior anchor"
        )


def test_nonexistent_and_missing_git_commits_are_rejected() -> None:
    with pytest.raises(ValueError, match="Could not uniquely resolve observed"):
        require_same_git_commit(
            Path.cwd(), "3387a7e", "f" * 40, context="synthetic prior anchor"
        )
    with pytest.raises(ValueError, match="Missing observed training commit"):
        require_same_git_commit(
            Path.cwd(), "3387a7e", None, context="synthetic prior anchor"
        )


def test_ambiguous_short_commit_is_rejected(tmp_path: Path) -> None:
    repo_dir = tmp_path / "ambiguous_repo"
    prefix = _ambiguous_commit_prefix(repo_dir)
    probe = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "--verify", f"{prefix}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode != 0
    assert "ambiguous" in probe.stderr.lower()
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_git_commit(repo_dir, prefix, label="ambiguous test commit")


def test_plan_reuses_30_by_reference_and_plans_only_20_new_runs(
    synthetic_frozen_candidate_workspace: dict[str, Any],
) -> None:
    workspace = synthetic_frozen_candidate_workspace
    result = _plan(workspace)
    registry = result["registry"]
    reused = [row for row in registry if row["source_type"] == "reused_prior"]
    new = [row for row in registry if row["source_type"] == "newly_trained"]
    assert len(registry) == 50
    assert len(reused) == 30
    assert len(new) == 20
    assert {row["candidate"] for row in new} == set(NEW_CANDIDATES)
    assert all(Path(row["source_run_directory"]).is_relative_to(workspace["group_root"]) for row in reused)
    assert all(Path(row["source_run_directory"]).is_relative_to(workspace["output_root"]) for row in new)
    assert not any((workspace["output_root"] / name).exists() for name in ANCHOR_MAPPING)
    assert (workspace["output_root"] / "experiment_plan.json").is_file()
    assert (workspace["output_root"] / "candidate_source_registry.json").is_file()
    expected_full = _full_commit("3387a7e")
    assert result["plan"]["expected_group_training_commit"] == {
        "configured": "3387a7e",
        "canonical": expected_full,
    }
    assert all(len(row["training_commit"]) == 40 for row in registry)
    assert all(row["training_commit"] == expected_full for row in reused)


def test_incompatible_prior_run_is_rejected(
    synthetic_frozen_candidate_workspace: dict[str, Any],
) -> None:
    workspace = synthetic_frozen_candidate_workspace
    config_path = workspace["group_root"] / "full17" / "gru" / "seed_7" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["batch_size"] = 128
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible batch_size"):
        _plan(workspace, write_outputs=False)


def test_training_command_uses_exact_inclusion_list_cuda_and_validation_only(
    synthetic_frozen_candidate_workspace: dict[str, Any],
) -> None:
    workspace = synthetic_frozen_candidate_workspace
    record = _plan(workspace, write_outputs=False)["new_runs"][0]
    command = build_training_command(record, workspace["dataset_dir"])
    feature_argument = command[command.index("--dynamic-features") + 1]
    assert feature_argument.split(",") == record["feature_names"]
    assert command[command.index("--device") + 1] == "cuda"
    assert "--validation-only" in command
    assert "--exclude-dynamic-features" not in command
    assert not any(
        forbidden in argument
        for argument in command
        for forbidden in ("test.npz", "test_metrics.json", "test_predictions.csv")
    )


def test_test_seal_and_resume_compatibility(
    synthetic_frozen_candidate_workspace: dict[str, Any],
) -> None:
    workspace = synthetic_frozen_candidate_workspace
    result = _plan(workspace, write_outputs=False)
    reused = result["registry"][0]
    run_dir = Path(reused["source_run_directory"])
    (run_dir / "test_metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Forbidden test artifacts"):
        validate_run_directory(
            run_dir,
            reused["candidate"],
            reused["model"],
            reused["seed"],
            reused["feature_names"],
            workspace["dataset_dir"],
        )
    (run_dir / "test_metrics.json").unlink()

    record = result["new_runs"][0]
    resume_dir = Path(record["source_run_directory"])
    resume_dir.mkdir(parents=True)
    (resume_dir / "last_model.pt").write_bytes(b"checkpoint")
    commit = result["plan"]["active_training_commit"]
    config = {
        "seed": record["seed"],
        "dynamic_feature_names": record["feature_names"],
        "dataset_dir": str(workspace["dataset_dir"]),
        "output_dir": str(resume_dir),
        "git_commit_hash": commit,
        "device": "cuda",
        "resolved_device": "cuda",
        "backend": "cuda",
        "smoke": False,
        **FIXED_SETTINGS,
    }
    (resume_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert validate_resume_compatibility(
        resume_dir, record, workspace["dataset_dir"], commit
    ) == resume_dir / "last_model.pt"
    config["learning_rate"] = 0.01
    (resume_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Incompatible resume"):
        validate_resume_compatibility(resume_dir, record, workspace["dataset_dir"], commit)


def test_retraining_notebook_is_locked_cuda_only_and_valid() -> None:
    notebook_path = Path("notebooks/colab_frozen_candidate_retraining.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    source = "\n".join(code_cells)
    assert notebook["nbformat"] == 4
    assert "RUN_FULL_TRAINING=False" in source
    assert "CONFIRMATION_TEXT=''" in source
    assert "RUN_20_FROZEN_CANDIDATE_CUDA_RUNS" in source
    assert "torch.cuda.is_available()" in source
    assert "source_type']=='newly_trained'" in source
    assert "test.npz" not in source
    assert "run_baselines.py" not in source
    assert "run_attention.py" not in source
    for index, cell_source in enumerate(code_cells):
        compile(cell_source, f"frozen_retraining_cell_{index}", "exec")
