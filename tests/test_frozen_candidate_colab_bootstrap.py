"""Tests for self-contained frozen-candidate Colab repository bootstrap cells."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

RETRAINING_NOTEBOOK = Path("notebooks/colab_frozen_candidate_retraining.ipynb")
ANALYSIS_NOTEBOOK = Path("notebooks/colab_frozen_candidate_analysis.ipynb")


@pytest.fixture(autouse=True)
def restore_working_directory() -> None:
    """Keep notebook-style chdir calls isolated from the pytest process."""

    previous = Path.cwd()
    try:
        yield
    finally:
        os.chdir(previous)


def _notebook(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap_source(path: Path) -> str:
    cells = [
        cell
        for cell in _notebook(path)["cells"]
        if "repository-bootstrap" in cell.get("metadata", {}).get("tags", [])
    ]
    assert len(cells) == 1
    return "".join(cells[0]["source"])


def _bootstrap_functions(path: Path = RETRAINING_NOTEBOOK) -> dict[str, Any]:
    tree = ast.parse(_bootstrap_source(path))
    definitions = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        ],
        type_ignores=[],
    )
    namespace: dict[str, Any] = {"Path": Path}
    exec(compile(definitions, str(path), "exec"), namespace)
    return namespace


def _git(*arguments: object, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *(str(argument) for argument in arguments)],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _create_remote(tmp_path: Path) -> tuple[Path, Path, str]:
    source = tmp_path / "source"
    remote = tmp_path / "remote.git"
    source.mkdir()
    _git("init", "--initial-branch=main", cwd=source)
    _git("config", "user.email", "synthetic@example.com", cwd=source)
    _git("config", "user.name", "Synthetic", cwd=source)
    (source / "version.txt").write_text("one\n", encoding="utf-8")
    _git("add", "version.txt", cwd=source)
    _git("commit", "-m", "initial", cwd=source)
    first_commit = _git("rev-parse", "HEAD", cwd=source)
    _git("clone", "--bare", source, remote, cwd=tmp_path)
    _git("remote", "add", "origin", remote, cwd=source)
    return source, remote, first_commit


def _synchronize(
    namespace: dict[str, Any], repo: Path, remote: Path, ancestor: str
) -> dict[str, str]:
    return namespace["synchronize_repository"](
        repo, str(remote), ancestor, allowed_parent=repo.parent
    )


def test_clone_succeeds_when_repository_is_absent(tmp_path: Path) -> None:
    namespace = _bootstrap_functions()
    _, remote, first_commit = _create_remote(tmp_path)
    repo = tmp_path / "checkout"
    state = _synchronize(namespace, repo, remote, first_commit)
    assert (repo / ".git").is_dir()
    assert state["head"] == first_commit
    assert state["head"] == state["origin_main"]


def test_sync_succeeds_when_current_directory_is_inside_stale_repo(
    tmp_path: Path,
) -> None:
    namespace = _bootstrap_functions()
    source, remote, first_commit = _create_remote(tmp_path)
    repo = tmp_path / "checkout"
    _synchronize(namespace, repo, remote, first_commit)

    (source / "version.txt").write_text("two\n", encoding="utf-8")
    _git("add", "version.txt", cwd=source)
    _git("commit", "-m", "update", cwd=source)
    _git("push", "origin", "main", cwd=source)
    latest = _git("rev-parse", "HEAD", cwd=source)
    (repo / "nested").mkdir()
    previous_cwd = Path.cwd()
    try:
        os.chdir(repo / "nested")
        state = _synchronize(namespace, repo, remote, first_commit)
    finally:
        os.chdir(previous_cwd)
    assert state["head"] == latest
    assert (repo / "version.txt").read_text(encoding="utf-8") == "two\n"


def test_non_git_temporary_path_is_safely_replaced(tmp_path: Path) -> None:
    namespace = _bootstrap_functions()
    _, remote, first_commit = _create_remote(tmp_path)
    repo = tmp_path / "checkout"
    repo.mkdir()
    (repo / "stale.txt").write_text("stale", encoding="utf-8")
    _synchronize(namespace, repo, remote, first_commit)
    assert (repo / ".git").is_dir()
    assert not (repo / "stale.txt").exists()


def test_wrong_origin_url_is_corrected_before_fetch(tmp_path: Path) -> None:
    namespace = _bootstrap_functions()
    _, remote, first_commit = _create_remote(tmp_path)
    repo = tmp_path / "checkout"
    _synchronize(namespace, repo, remote, first_commit)
    _git("remote", "set-url", "origin", tmp_path / "wrong.git", cwd=repo)
    state = _synchronize(namespace, repo, remote, first_commit)
    assert state["origin_url"] == str(remote)


def test_git_failure_exposes_command_stdout_and_stderr(tmp_path: Path) -> None:
    run_command = _bootstrap_functions()["run_command"]
    with pytest.raises(RuntimeError) as captured:
        run_command(["git", "-C", tmp_path / "missing", "rev-parse", "HEAD"])
    message = str(captured.value)
    assert "git -C" in message
    assert "stdout:" in message
    assert "stderr:" in message
    assert "fatal" in message.lower()


def test_head_origin_mismatch_is_detected(tmp_path: Path) -> None:
    namespace = _bootstrap_functions()
    _, remote, first_commit = _create_remote(tmp_path)
    repo = tmp_path / "checkout"
    _synchronize(namespace, repo, remote, first_commit)
    _git("config", "user.email", "synthetic@example.com", cwd=repo)
    _git("config", "user.name", "Synthetic", cwd=repo)
    (repo / "local.txt").write_text("local\n", encoding="utf-8")
    _git("add", "local.txt", cwd=repo)
    _git("commit", "-m", "local divergence", cwd=repo)
    with pytest.raises(RuntimeError, match="not synchronized"):
        namespace["verify_repository_state"](repo, first_commit)


def test_missing_required_ancestor_is_detected(tmp_path: Path) -> None:
    namespace = _bootstrap_functions()
    _, remote, first_commit = _create_remote(tmp_path)
    repo = tmp_path / "checkout"
    _synchronize(namespace, repo, remote, first_commit)
    with pytest.raises(RuntimeError, match="not an ancestor"):
        namespace["verify_repository_state"](repo, "f" * 40)


def test_stale_project_modules_are_purged_and_repo_is_first_on_path(
    tmp_path: Path,
) -> None:
    namespace = _bootstrap_functions()
    prefix = "synthetic_colab_project"
    module_name = prefix + ".old"
    previous_path = list(sys.path)
    sys.modules[prefix] = types.ModuleType(prefix)
    sys.modules[module_name] = types.ModuleType(module_name)
    try:
        stale = namespace["prepare_project_imports"](tmp_path, prefixes=(prefix,))
        assert set(stale) == {prefix, module_name}
        assert prefix not in sys.modules
        assert module_name not in sys.modules
        assert Path(sys.path[0]).resolve() == tmp_path.resolve()
    finally:
        sys.modules.pop(prefix, None)
        sys.modules.pop(module_name, None)
        sys.path[:] = previous_path


def test_notebooks_share_bootstrap_and_have_safe_cell_order() -> None:
    retraining = _notebook(RETRAINING_NOTEBOOK)
    analysis = _notebook(ANALYSIS_NOTEBOOK)
    assert _bootstrap_source(RETRAINING_NOTEBOOK) == _bootstrap_source(ANALYSIS_NOTEBOOK)
    for path, notebook in ((RETRAINING_NOTEBOOK, retraining), (ANALYSIS_NOTEBOOK, analysis)):
        code_cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        source = "\n".join(code_cells)
        bootstrap_index = next(
            index
            for index, cell in enumerate(notebook["cells"])
            if "repository-bootstrap" in cell.get("metadata", {}).get("tags", [])
        )
        before_bootstrap = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"][:bootstrap_index]
        )
        assert "from src" not in before_bootstrap
        assert "import src" not in before_bootstrap
        assert source.index("synchronize_repository(") < source.index(
            "scripts/install_colab_dependencies.py"
        )
        assert "git', '-C', repo_dir, 'fetch', 'origin', 'main', '--prune'" in source
        assert "git', '-C', repo_dir, 'reset', '--hard', 'origin/main'" in source
        assert "merge-base', '--is-ancestor', required_ancestor, 'HEAD'" in source
        assert "shutil.rmtree(repo_dir)" in source
        assert "shutil.rmtree(ROOT" not in source
        assert "test.npz" not in source
        for index, cell_source in enumerate(code_cells):
            compile(cell_source, f"{path.name}:cell-{index}", "exec")
        assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")
        assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_retraining_notebook_keeps_preflight_and_interactive_lock() -> None:
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in _notebook(RETRAINING_NOTEBOOK)["cells"]
    )
    assert "REQUIRED_WORKFLOW_COMMIT = '8df2802'" in source
    assert "RUN_FULL_TRAINING = False" in source
    assert "PREFLIGHT_PASSED = False" in source
    assert "input(" in source
    assert "RUN_20_FROZEN_CANDIDATE_CUDA_RUNS" in source
    assert source.index("PREFLIGHT_PASSED = True") < source.index("CONFIRMATION_TEXT = input")
    assert "(len(reused), len(new), len(REGISTRY)) != (30, 20, 50)" in source
    assert "command[command.index('--device') + 1] != 'cuda'" in source
    assert "'--validation-only' not in command" in source
    assert "run_command(command, cwd=REPO_DIR, check=False, stream=True)" in source
    assert "anchor_before != anchor_after" in source
