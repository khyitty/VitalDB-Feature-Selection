"""Static, non-executing intake validation for an external RL package."""

from __future__ import annotations

import ast
import csv
import json
import logging
import re
import shutil
import stat
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Sequence

from src.frozen_candidate_retraining import dump_json

LOGGER = logging.getLogger(__name__)

MAX_ARCHIVE_MEMBERS = 10_000
MAX_ARCHIVE_BYTES = 500 * 1024 * 1024
TEXT_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".toml", ".md", ".txt"}
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".zip"}
REQUIREMENTS = (
    "environment_class",
    "reset_method",
    "step_method",
    "observation_state_schema",
    "action_space",
    "action_unit",
    "propofol_infusion_range",
    "reward_function",
    "termination_truncation",
    "pk_pd_simulator",
    "patient_case_split",
    "train_evaluation_entry_point",
    "rl_algorithm",
    "policy_or_value_network",
    "replay_or_rollout_storage",
    "seed_setting",
    "baseline_config",
    "checkpoint",
    "evaluation_metrics",
)
CORE_REQUIREMENTS = {"environment_class", "reset_method", "step_method"}


@dataclass(frozen=True)
class Evidence:
    """One statically verified interface or configuration location."""

    requirement: str
    path: str
    line: int | None
    symbol: str
    details: str


def _safe_relative_member(name: str) -> Path:
    normalized = name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"Unsafe archive member path: {name!r}")
    if re.match(r"^[A-Za-z]:", normalized):
        raise ValueError(f"Unsafe archive drive path: {name!r}")
    return Path(*pure.parts)


def _extract_zip(source: Path, destination: Path) -> None:
    total = 0
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError("Archive contains too many members.")
        for member in members:
            relative = _safe_relative_member(member.filename)
            mode = (member.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise ValueError(f"Archive symlinks are prohibited: {member.filename}")
            total += member.file_size
            if total > MAX_ARCHIVE_BYTES:
                raise ValueError("Archive exceeds the static intake size limit.")
            target = destination / relative
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source_handle, target.open("wb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle)


def _extract_tar(source: Path, destination: Path) -> None:
    total = 0
    with tarfile.open(source, mode="r:*") as archive:
        members = archive.getmembers()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError("Archive contains too many members.")
        for member in members:
            relative = _safe_relative_member(member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"Archive links/devices are prohibited: {member.name}")
            total += max(member.size, 0)
            if total > MAX_ARCHIVE_BYTES:
                raise ValueError("Archive exceeds the static intake size limit.")
            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source_handle = archive.extractfile(member)
            if source_handle is None:
                raise ValueError(f"Could not read archive member: {member.name}")
            with source_handle, target.open("wb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle)


@contextmanager
def materialize_external_package(source: Path) -> Iterator[Path]:
    """Yield a source directory or a safe temporary archive extraction."""

    source = source.resolve()
    if source.is_dir():
        yield source
        return
    if not source.is_file():
        raise FileNotFoundError(f"External RL package does not exist: {source}")
    with tempfile.TemporaryDirectory(prefix="rl-intake-") as temporary:
        destination = Path(temporary) / "package"
        destination.mkdir()
        lower = source.name.lower()
        if lower.endswith(".zip"):
            _extract_zip(source, destination)
        elif lower.endswith((".tar.gz", ".tgz", ".tar")):
            _extract_tar(source, destination)
        else:
            raise ValueError("Supported external packages are directories, .zip, or tar archives.")
        yield destination


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return _name(node.func)
    if isinstance(node, ast.Subscript):
        return _name(node.value)
    return ""


def _add(
    evidence: list[Evidence],
    requirement: str,
    path: Path,
    root: Path,
    node: ast.AST | None,
    symbol: str,
    details: str,
) -> None:
    item = Evidence(
        requirement=requirement,
        path=path.relative_to(root).as_posix(),
        line=getattr(node, "lineno", None),
        symbol=symbol,
        details=details,
    )
    if item not in evidence:
        evidence.append(item)


def _scan_python(path: Path, root: Path, evidence: list[Evidence], errors: list[str]) -> None:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError) as error:
        errors.append(f"{path.relative_to(root).as_posix()}: {error}")
        return
    lowered = source.lower()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = {
                child.name: child
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            bases = " ".join(_name(base) for base in node.bases)
            is_environment = (
                {"reset", "step"}.issubset(methods)
                and (
                    re.search(r"(?:env|environment)$", node.name, re.IGNORECASE)
                    or re.search(r"(?:^|\.)Env$", bases)
                )
            )
            if is_environment:
                _add(evidence, "environment_class", path, root, node, node.name, f"bases={bases or '<none>'}")
                _add(evidence, "reset_method", path, root, methods["reset"], f"{node.name}.reset", ast.unparse(methods["reset"].args))
                _add(evidence, "step_method", path, root, methods["step"], f"{node.name}.step", ast.unparse(methods["step"].args))
            class_lower = node.name.lower()
            if any(term in class_lower for term in ("pkpd", "simulator", "schnider", "marsh", "minto")):
                _add(evidence, "pk_pd_simulator", path, root, node, node.name, "simulator-like class")
            if any(term in class_lower for term in ("actor", "critic", "policy", "value", "network")):
                _add(evidence, "policy_or_value_network", path, root, node, node.name, "policy/value network class")
            if any(term in class_lower for term in ("replay", "rollout", "buffer", "storage")):
                _add(evidence, "replay_or_rollout_storage", path, root, node, node.name, "experience storage class")
            if any(term in class_lower for term in ("ppo", "sac", "td3", "ddpg", "dqn", "agent", "algorithm")):
                _add(evidence, "rl_algorithm", path, root, node, node.name, "RL algorithm class")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name.lower()
            if "reward" in name:
                _add(evidence, "reward_function", path, root, node, node.name, ast.unparse(node.args))
            if name in {"main", "train", "evaluate", "evaluation", "train_agent", "evaluate_policy"}:
                _add(evidence, "train_evaluation_entry_point", path, root, node, node.name, ast.unparse(node.args))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [_name(target).lower() for target in targets]
            for name in names:
                if name.endswith("observation_space") or "state_schema" in name or "observation_schema" in name:
                    _add(evidence, "observation_state_schema", path, root, node, name, "static assignment")
                if name.endswith("action_space"):
                    value = node.value
                    _add(evidence, "action_space", path, root, node, name, ast.unparse(value) if value else "assignment")
                    value_text = ast.unparse(value).lower() if value else ""
                    if "low" in value_text and "high" in value_text:
                        _add(evidence, "propofol_infusion_range", path, root, node, name, value_text[:300])
                if any(term in name for term in ("train_cases", "val_cases", "test_cases", "patient_split", "case_split")):
                    _add(evidence, "patient_case_split", path, root, node, name, "patient/case split assignment")
                if "seed" in name:
                    _add(evidence, "seed_setting", path, root, node, name, "seed assignment")
        elif isinstance(node, ast.Call):
            call_name = _name(node.func).lower()
            if any(term in call_name for term in ("manual_seed", "seed_everything", "set_seed")):
                _add(evidence, "seed_setting", path, root, node, call_name, ast.unparse(node))
            if any(term in call_name for term in ("ppo", "sac", "td3", "ddpg", "dqn")):
                _add(evidence, "rl_algorithm", path, root, node, call_name, ast.unparse(node)[:300])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip()
            if re.search(r"(?:mcg|ug|mg|ml)(?:/kg)?(?:/min|/h|/hr)?", value, re.IGNORECASE):
                _add(evidence, "action_unit", path, root, node, "string_literal", value[:300])

    if "reward" in lowered and "def step" in lowered:
        step_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "step"]
        for node in step_nodes:
            _add(evidence, "reward_function", path, root, node, "step reward", "reward computed or returned in step")
    if "terminated" in lowered or "truncated" in lowered or re.search(r"\bdone\b", lowered):
        node = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "step"), tree)
        _add(evidence, "termination_truncation", path, root, node, "episode termination", "terminated/truncated/done evidence")
    if any(term in lowered for term in ("propofol", "infusion")) and any(item.requirement == "action_space" and item.path == path.relative_to(root).as_posix() for item in evidence):
        node = next((node for node in ast.walk(tree) if isinstance(node, ast.Assign)), tree)
        _add(evidence, "propofol_infusion_range", path, root, node, "propofol action", "propofol/infusion action-space evidence")
    if any(term in lowered for term in ("mae", "rmse", "bis_in_range", "episode_return", "control_error")):
        _add(evidence, "evaluation_metrics", path, root, tree, "metric identifiers", "static metric names found")


def inspect_external_package(root: Path) -> tuple[list[Evidence], list[str]]:
    """Inspect interfaces and artifacts without importing any external module."""

    evidence: list[Evidence] = []
    errors: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if path.suffix.lower() == ".py" and path.stat().st_size <= 2 * 1024 * 1024:
            _scan_python(path, root, evidence, errors)
        lower_name = path.name.lower()
        if path.suffix.lower() in CHECKPOINT_SUFFIXES and any(
            term in lower_name for term in ("policy", "actor", "agent", "baseline", "checkpoint", "model")
        ):
            _add(evidence, "checkpoint", path, root, None, relative.as_posix(), "checkpoint file present")
        if path.suffix.lower() in {".json", ".yaml", ".yml", ".toml"} and any(
            term in lower_name for term in ("baseline", "config", "hyperparam")
        ):
            _add(evidence, "baseline_config", path, root, None, relative.as_posix(), "configuration file present")
        if any(term in lower_name for term in ("split", "train_cases", "test_cases")) and path.suffix.lower() in TEXT_SUFFIXES | {".csv"}:
            _add(evidence, "patient_case_split", path, root, None, relative.as_posix(), "split artifact present")
    return sorted(evidence, key=lambda item: (item.requirement, item.path, item.line or 0)), errors


def _status(found: set[str]) -> str:
    missing = set(REQUIREMENTS) - found
    if not missing:
        return "ready_for_adapter"
    if CORE_REQUIREMENTS.issubset(found):
        return "partially_ready"
    return "blocked_missing_components"


def _write_reports(
    output_dir: Path,
    source: Path,
    evidence: Sequence[Evidence],
    parse_errors: Sequence[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    found = {item.requirement for item in evidence}
    missing = [name for name in REQUIREMENTS if name not in found]
    status_value = _status(found)
    evidence_rows = [asdict(item) for item in evidence]
    with (output_dir / "discovered_interfaces.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Evidence.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(evidence_rows)
    with (output_dir / "missing_requirements.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["requirement", "blocking"])
        writer.writeheader()
        writer.writerows(
            {"requirement": name, "blocking": name in CORE_REQUIREMENTS}
            for name in missing
        )
    adapter = {
        "status": status_value,
        "ready_for_adapter": status_value == "ready_for_adapter",
        "static_analysis_only": True,
        "external_code_imported": False,
        "external_code_executed": False,
        "missing_requirements": missing,
        "next_action": (
            "Build an adapter only after manually confirming all discovered contracts."
            if status_value == "ready_for_adapter"
            else "Request the missing external RL components before adapter or training work."
        ),
    }
    dump_json(adapter, output_dir / "adapter_feasibility.json")
    report = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "status": status_value,
        "requirements": list(REQUIREMENTS),
        "found_requirements": sorted(found),
        "missing_requirements": missing,
        "parse_errors": list(parse_errors),
        "discovered_interface_count": len(evidence),
        "static_analysis_only": True,
        "external_code_imported": False,
        "external_code_executed": False,
        "rl_training_started": False,
    }
    dump_json(report, output_dir / "rl_intake_report.json")
    lines = [
        "# External RL Intake Report",
        "",
        f"Status: `{status_value}`",
        "",
        "The package was inspected statically. No external module was imported or executed, and no RL training was started.",
        "",
        "## Discovered",
        "",
        *(f"- `{name}`" for name in sorted(found)),
        "",
        "## Missing",
        "",
        *(f"- `{name}`" for name in missing),
    ]
    (output_dir / "rl_intake_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def validate_external_rl_package(source: Path, output_dir: Path) -> dict[str, Any]:
    """Safely materialize and statically inspect an external RL package."""

    with materialize_external_package(source) as root:
        evidence, parse_errors = inspect_external_package(root)
    report = _write_reports(output_dir, source, evidence, parse_errors)
    LOGGER.info("External RL intake status: %s", report["status"])
    return report
