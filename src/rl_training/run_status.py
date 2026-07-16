"""Failure-safe status records shared by PPO full and smoke workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess
import traceback
from typing import Any, Mapping

from .io import atomic_write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_versions() -> dict[str, str]:
    """Return the compact environment fingerprint needed to audit a run."""

    result: dict[str, str] = {}
    for package in ("python", "numpy", "torch", "gymnasium", "stable-baselines3"):
        if package == "python":
            import platform

            result[package] = platform.python_version()
            continue
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def repository_commit(repo_dir: Path) -> str:
    """Resolve the exact implementation commit or fail explicitly."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Unable to resolve repository commit in {repo_dir}.") from exc
    commit = completed.stdout.strip()
    if len(commit) != 40:
        raise RuntimeError(f"Resolved repository commit is not canonical: {commit!r}")
    return commit


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Run status is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Run status root must be an object: {path}")
    return payload


def begin_run_status(
    run_dir: Path,
    *,
    resolved_config: Mapping[str, Any],
    repo_dir: Path,
) -> dict[str, Any]:
    """Mark a new or resumed attempt running before any training begins."""

    path = run_dir / "run_status.json"
    previous = _read(path)
    now = utc_now()
    payload = {
        "status_schema_version": 1,
        "status": "running",
        "start_time_utc": previous.get("start_time_utc", now),
        "attempt_start_time_utc": now,
        "previous_status": previous.get("status"),
        "resolved_config": dict(resolved_config),
        "seed": resolved_config.get("seed"),
        "state_profile": resolved_config.get("state_profile"),
        "ordered_feature_names": list(resolved_config.get("ordered_feature_names", [])),
        "git_commit": repository_commit(repo_dir),
        "package_versions": package_versions(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return payload


def update_running_config(
    run_dir: Path,
    *,
    updates: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically add values resolved after model initialization but before training."""

    path = run_dir / "run_status.json"
    payload = _read(path)
    if payload.get("status") != "running":
        raise ValueError(f"Cannot update config for a non-running PPO run: {path}")
    resolved_config = payload.get("resolved_config")
    if not isinstance(resolved_config, dict):
        raise ValueError(f"Running status has no resolved_config object: {path}")
    resolved_config.update(dict(updates))
    payload["resolved_config"] = resolved_config
    atomic_write_json(path, payload)
    return payload


def complete_run_status(
    run_dir: Path,
    *,
    final_checkpoint: Path,
    evaluation_artifacts: list[Path],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Mark complete only after checkpoint and evaluation serialization."""

    path = run_dir / "run_status.json"
    payload = _read(path)
    if not final_checkpoint.is_file():
        raise FileNotFoundError(f"Final checkpoint is missing: {final_checkpoint}")
    missing = [str(item) for item in evaluation_artifacts if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Evaluation artifacts are missing: {missing}")
    payload.update(
        {
            "status": "complete",
            "completion_time_utc": utc_now(),
            "final_checkpoint": str(final_checkpoint.resolve()),
            "evaluation_artifact_paths": [
                str(item.resolve()) for item in evaluation_artifacts
            ],
        }
    )
    if extra:
        payload.update(dict(extra))
    atomic_write_json(path, payload)
    return payload


def fail_run_status(
    run_dir: Path,
    exception: BaseException,
    *,
    last_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Persist the original exception and traceback without hiding the failure."""

    path = run_dir / "run_status.json"
    payload = _read(path)
    payload.update(
        {
            "status": "failed",
            "failure_time_utc": utc_now(),
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "traceback": "".join(
                traceback.format_exception(type(exception), exception, exception.__traceback__)
            ),
            "last_checkpoint": (
                str(last_checkpoint.resolve())
                if last_checkpoint is not None and last_checkpoint.is_file()
                else None
            ),
        }
    )
    atomic_write_json(path, payload)
    return payload
