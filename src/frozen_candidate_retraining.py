"""Plan and validate validation-only retraining of frozen feature candidates."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.colab_workflow import inspect_run_completion
from src.group_retraining_analysis import load_json, validate_experiment
from src.redundancy_audit import REDUCED_FEATURES

LOGGER = logging.getLogger(__name__)

SEEDS = (7, 21, 42, 84, 123)
MODELS = ("gru", "attention")
FROZEN_CANDIDATES = (
    "full17_reference",
    "no_respiratory_anchor",
    "compact11_anchor",
    "strict_consensus",
    "compact_consensus",
)
ANCHOR_MAPPING = {
    "full17_reference": "full17",
    "no_respiratory_anchor": "no_respiratory",
    "compact11_anchor": "no_remifentanil_or_respiratory",
}
NEW_CANDIDATES = ("strict_consensus", "compact_consensus")
EXPECTED_FEATURE_COUNTS = {
    "full17_reference": 17,
    "no_respiratory_anchor": 15,
    "compact11_anchor": 11,
    "strict_consensus": 7,
    "compact_consensus": 11,
}
EXPECTED_DISCOVERY_HASHES = {
    "strict_consensus": "83e2eb15fdf8b272348ddad674fd95505722b99683697e5c9bb7a0ecf926af6d",
    "compact_consensus": "f6ee4bc37e3926493497636ca670bfb15b79843d4d6bd9537f594e8c96899ed9",
}
# Keep the historical human-readable abbreviation, but never compare or persist it
# before resolving it to its unique commit object in the active repository.
TRAINING_COMMIT = "3387a7e"
TRAINING_SOURCE_FILES = (
    "src/training.py",
    "src/attention_training.py",
    "src/datasets.py",
    "src/models/baselines.py",
    "src/models/attention.py",
    "src/metrics.py",
    "scripts/run_baselines.py",
    "scripts/run_attention.py",
)
DATASET_FINGERPRINT_FILES = (
    "dataset_metadata.json",
    "preprocessing.pkl",
    "preprocessing_statistics.csv",
    "train_metadata.csv",
    "val_metadata.csv",
)
FORBIDDEN_TEST_ARTIFACTS = (
    "test_predictions.csv",
    "test_metrics.json",
    "test_attention.npz",
)
FIXED_SETTINGS = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "max_epochs": 50,
    "patience": 8,
    "case_balanced_sampling": True,
    "num_workers": 0,
    "evaluate_test": False,
}


@dataclass(frozen=True)
class FrozenCandidateSet:
    """Validated candidate definitions loaded from the selector artifact."""

    features: dict[str, tuple[str, ...]]
    source_path: Path
    source_sha256: str


def dump_json(payload: Mapping[str, Any] | Sequence[Any], path: Path) -> None:
    """Write a strict JSON artifact under an explicitly requested path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_list_hash(features: Sequence[str]) -> str:
    canonical = json.dumps(list(features), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_frozen_candidates(candidate_path: Path) -> FrozenCandidateSet:
    """Load the five frozen sets from the selector artifact and verify identities."""

    payload = load_json(candidate_path)
    frozen = tuple(payload.get("frozen_retraining_candidates", ()))
    if frozen != FROZEN_CANDIDATES:
        raise ValueError(f"Frozen candidate names/order mismatch: {list(frozen)}")
    definitions = payload.get("all_candidate_subsets")
    if not isinstance(definitions, dict):
        raise ValueError("candidate_subsets.json lacks all_candidate_subsets.")
    features: dict[str, tuple[str, ...]] = {}
    for candidate in FROZEN_CANDIDATES:
        definition = definitions.get(candidate)
        if not isinstance(definition, dict) or not isinstance(
            definition.get("features"), list
        ):
            raise ValueError(f"Missing feature definition for {candidate}.")
        names = tuple(definition["features"])
        if len(names) != EXPECTED_FEATURE_COUNTS[candidate] or len(set(names)) != len(names):
            raise ValueError(f"Invalid feature count or duplicate in {candidate}: {list(names)}")
        if any(name not in REDUCED_FEATURES for name in names):
            raise ValueError(f"{candidate} contains a feature outside full17.")
        if tuple(name for name in REDUCED_FEATURES if name in names) != names:
            raise ValueError(f"{candidate} feature order differs from the source manifest.")
        features[candidate] = names
    expected_anchors = {
        "full17_reference": REDUCED_FEATURES,
        "no_respiratory_anchor": tuple(
            name for name in REDUCED_FEATURES if name not in {"spo2", "etco2"}
        ),
        "compact11_anchor": tuple(
            name
            for name in REDUCED_FEATURES
            if name
            not in {
                "spo2",
                "etco2",
                "rftn_rate",
                "rftn_volume",
                "rftn_cp",
                "rftn_ce",
            }
        ),
    }
    for candidate, expected in expected_anchors.items():
        if features[candidate] != expected:
            raise ValueError(f"Anchor feature mismatch for {candidate}.")
    for candidate, expected_hash in EXPECTED_DISCOVERY_HASHES.items():
        if feature_list_hash(features[candidate]) != expected_hash:
            raise ValueError(f"Predictive-selection feature fingerprint mismatch for {candidate}.")
    return FrozenCandidateSet(features, candidate_path, sha256_file(candidate_path))


def dataset_fingerprint(dataset_dir: Path) -> dict[str, Any]:
    """Fingerprint validation-safe dataset and preprocessing artifacts."""

    missing = [name for name in DATASET_FINGERPRINT_FILES if not (dataset_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Dataset fingerprint inputs are missing: {missing}")
    files = {
        name: {"sha256": sha256_file(dataset_dir / name), "size_bytes": (dataset_dir / name).stat().st_size}
        for name in DATASET_FINGERPRINT_FILES
    }
    combined = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = load_json(dataset_dir / "dataset_metadata.json")
    expected_timing = {
        "history_window_seconds": 60,
        "prediction_horizon_seconds": 30,
        "resampling_interval_seconds": 10,
    }
    mismatches = {
        key: (expected, metadata.get(key))
        for key, expected in expected_timing.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Modeling dataset timing differs from the frozen pilot: {mismatches}")
    return {
        "combined_sha256": combined,
        "files": files,
        "history_window_seconds": metadata["history_window_seconds"],
        "prediction_horizon_seconds": metadata["prediction_horizon_seconds"],
        "resampling_interval_seconds": metadata["resampling_interval_seconds"],
    }


def resolve_git_commit(repo_dir: Path, commit: object, *, label: str) -> str:
    """Resolve a SHA abbreviation or full SHA to one canonical commit object ID."""

    if commit is None:
        raise ValueError(f"Missing {label}.")
    value = str(commit).strip()
    if not re.fullmatch(r"[0-9a-fA-F]{4,40}", value):
        raise ValueError(
            f"Invalid {label} {value!r}; expected a 4- to 40-character hexadecimal Git SHA."
        )
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_dir),
                "rev-parse",
                "--verify",
                f"{value}^{{commit}}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise ValueError(
            f"Could not resolve {label} {value!r} in {repo_dir}: {error}"
        ) from error
    resolved = result.stdout.strip()
    if result.returncode != 0 or not re.fullmatch(r"[0-9a-fA-F]{40}", resolved):
        detail = result.stderr.strip() or result.stdout.strip() or "no Git diagnostic"
        raise ValueError(
            f"Could not uniquely resolve {label} {value!r} in {repo_dir}. "
            f"The SHA may be missing, invalid, or ambiguous. Git reported: {detail}"
        )
    return resolved.lower()


def require_same_git_commit(
    repo_dir: Path,
    expected_commit: object,
    observed_commit: object,
    *,
    context: str,
) -> str:
    """Require two SHA spellings to resolve to the same Git commit."""

    expected_full = resolve_git_commit(
        repo_dir, expected_commit, label=f"expected training commit for {context}"
    )
    observed_full = resolve_git_commit(
        repo_dir, observed_commit, label=f"observed training commit for {context}"
    )
    return _require_resolved_git_commit_match(
        expected_commit,
        expected_full,
        observed_commit,
        observed_full,
        context=context,
    )


def _require_resolved_git_commit_match(
    expected_commit: object,
    expected_full: str,
    observed_commit: object,
    observed_full: str,
    *,
    context: str,
) -> str:
    """Compare already resolved IDs while preserving both raw values in errors."""

    if observed_full != expected_full:
        raise ValueError(
            f"{context} training commit mismatch: "
            f"expected raw={expected_commit!r}, full={expected_full}; "
            f"observed raw={observed_commit!r}, full={observed_full}."
        )
    return expected_full


def verify_training_source_compatibility(
    repo_dir: Path, training_commit: object = TRAINING_COMMIT
) -> dict[str, str]:
    """Require no Git-visible training-source changes since training commit 3387a7e."""

    canonical_commit = resolve_git_commit(
        repo_dir, training_commit, label="training-source reference commit"
    )
    comparison = subprocess.run(
        ["git", "diff", "--quiet", canonical_commit, "--", *TRAINING_SOURCE_FILES],
        cwd=repo_dir,
        check=False,
    )
    if comparison.returncode != 0:
        changed = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                canonical_commit,
                "--",
                *TRAINING_SOURCE_FILES,
            ],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        raise ValueError(
            f"Training sources changed meaningfully since "
            f"{training_commit!r} ({canonical_commit}): {changed}"
        )
    hashes = {}
    for relative in TRAINING_SOURCE_FILES:
        result = subprocess.run(
            ["git", "show", f"{canonical_commit}:{relative}"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )
        hashes[relative] = hashlib.sha256(result.stdout).hexdigest()
    return hashes


def _validate_config_settings(config: Mapping[str, Any], run_dir: Path) -> None:
    for key, expected in FIXED_SETTINGS.items():
        if config.get(key) != expected:
            raise ValueError(f"{run_dir} has incompatible {key}: {config.get(key)!r}")
    if str(config.get("backend", "")).lower() != "cuda":
        raise ValueError(f"{run_dir} was not trained with CUDA backend.")
    if not str(config.get("resolved_device", "")).lower().startswith("cuda"):
        raise ValueError(f"{run_dir} did not resolve to CUDA.")


def validate_run_directory(
    run_dir: Path,
    candidate: str,
    model: str,
    seed: int,
    features: Sequence[str],
    dataset_dir: Path,
) -> dict[str, Any]:
    """Validate one complete run without opening held-out test artifacts."""

    completion = inspect_run_completion(run_dir, model)
    if not completion["complete"]:
        raise ValueError(f"Incomplete {candidate}/{model}/seed_{seed}: {completion}")
    forbidden = [name for name in FORBIDDEN_TEST_ARTIFACTS if (run_dir / name).exists()]
    if forbidden:
        raise ValueError(f"Forbidden test artifacts in {run_dir}: {forbidden}")
    config = load_json(run_dir / "config.json")
    status = load_json(run_dir / "run_status.json")
    if int(config.get("seed", -1)) != seed or status.get("status") != "complete":
        raise ValueError(f"Seed/status mismatch in {run_dir}.")
    if config.get("evaluate_test") is not False or status.get("test_evaluated") is not False:
        raise ValueError(f"Test evaluation was enabled in {run_dir}.")
    if tuple(config.get("dynamic_feature_names", ())) != tuple(features):
        raise ValueError(f"Frozen feature mismatch in {run_dir}.")
    if Path(str(config.get("dataset_dir"))).resolve() != dataset_dir.resolve():
        raise ValueError(f"Dataset path mismatch in {run_dir}.")
    _validate_config_settings(config, run_dir)
    predictions = pd.read_csv(run_dir / "val_predictions.csv")
    required = {"sample_index", "case_id", "target_timestamp", "observed_future_bis"}
    if not required.issubset(predictions.columns) or predictions.empty:
        raise ValueError(f"Invalid validation predictions in {run_dir}.")
    return {
        "config": config,
        "status": status,
        "prediction_rows": len(predictions),
        "validation_patients": sorted(predictions["case_id"].astype(int).unique().tolist()),
    }


def _verify_prior_dataset_fingerprint(
    fingerprint: Mapping[str, Any], group_analysis_dir: Path
) -> None:
    manifest = load_json(group_analysis_dir / "analysis_manifest.json")
    prior = {
        Path(item["path"]).name: item["sha256"]
        for item in manifest.get("input_fingerprints", [])
        if Path(item.get("path", "")).name in DATASET_FINGERPRINT_FILES
    }
    current = {name: details["sha256"] for name, details in fingerprint["files"].items()}
    if prior != current:
        raise ValueError("Current dataset/preprocessing fingerprint differs from prior analysis.")


def build_retraining_plan(
    *,
    candidate_path: Path,
    dataset_dir: Path,
    group_root: Path,
    group_analysis_dir: Path,
    output_root: Path,
    repo_dir: Path,
    active_commit: str,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Validate 30 reused anchors and define exactly 20 new CUDA runs."""

    candidates = load_frozen_candidates(candidate_path)
    expected_training_commit = resolve_git_commit(
        repo_dir, TRAINING_COMMIT, label="expected group-training commit"
    )
    active_training_commit = resolve_git_commit(
        repo_dir, active_commit, label="active training commit"
    )
    source_hashes = verify_training_source_compatibility(
        repo_dir, expected_training_commit
    )
    fingerprint = dataset_fingerprint(dataset_dir)
    _verify_prior_dataset_fingerprint(fingerprint, group_analysis_dir)
    prior_runs = validate_experiment(group_root)
    prior_by_key = {(run.condition, run.model, run.seed): run for run in prior_runs}
    registry = []
    observed_commit_cache: dict[str, str] = {}
    reference_split: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    for candidate, condition in ANCHOR_MAPPING.items():
        for model in MODELS:
            for seed in SEEDS:
                run = prior_by_key[(condition, model, seed)]
                validated = validate_run_directory(
                    run.run_dir,
                    candidate,
                    model,
                    seed,
                    candidates.features[candidate],
                    dataset_dir,
                )
                observed_training_commit = validated["config"].get("git_commit_hash")
                observed_cache_key = (
                    str(observed_training_commit).strip()
                    if observed_training_commit is not None
                    else "<missing>"
                )
                if observed_cache_key not in observed_commit_cache:
                    observed_commit_cache[observed_cache_key] = resolve_git_commit(
                        repo_dir,
                        observed_training_commit,
                        label=f"observed training commit for Prior anchor {run.run_dir}",
                    )
                canonical_training_commit = _require_resolved_git_commit_match(
                    TRAINING_COMMIT,
                    expected_training_commit,
                    observed_training_commit,
                    observed_commit_cache[observed_cache_key],
                    context=f"Prior anchor {run.run_dir}",
                )
                split = (
                    tuple(validated["config"]["selected_training_cases"]),
                    tuple(validated["config"]["selected_validation_cases"]),
                )
                if reference_split is None:
                    reference_split = split
                elif split != reference_split:
                    raise ValueError("Prior anchor runs do not share the same patient split.")
                registry.append(
                    {
                        "candidate": candidate,
                        "model": model,
                        "seed": seed,
                        "source_type": "reused_prior",
                        "source_run_directory": str(run.run_dir),
                        "feature_names": list(candidates.features[candidate]),
                        "feature_count": len(candidates.features[candidate]),
                        "training_commit": canonical_training_commit,
                        "dataset_fingerprint": fingerprint["combined_sha256"],
                        "test_evaluated": False,
                        "completion_status": "complete",
                    }
                )
    new_runs = []
    for candidate in NEW_CANDIDATES:
        for model in MODELS:
            for seed in SEEDS:
                run_dir = output_root / candidate / model / f"seed_{seed}"
                record = {
                    "candidate": candidate,
                    "model": model,
                    "seed": seed,
                    "source_type": "newly_trained",
                    "source_run_directory": str(run_dir),
                    "feature_names": list(candidates.features[candidate]),
                    "feature_count": len(candidates.features[candidate]),
                    "training_commit": active_training_commit,
                    "dataset_fingerprint": fingerprint["combined_sha256"],
                    "test_evaluated": False,
                    "completion_status": "planned",
                }
                new_runs.append(record)
                registry.append(record)
    if len(registry) != 50 or len(new_runs) != 20:
        raise AssertionError("Expected 30 reused + 20 new = 50 registry rows.")
    plan = {
        "candidate_source_sha256": candidates.source_sha256,
        "frozen_candidates": {name: list(candidates.features[name]) for name in FROZEN_CANDIDATES},
        "reused_run_count": 30,
        "new_run_count": 20,
        "comparison_run_count": 50,
        "new_candidates": list(NEW_CANDIDATES),
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "settings": {**FIXED_SETTINGS, "device": "cuda", "num_workers": 0},
        "dataset_fingerprint": fingerprint,
        "training_source_hashes_at_3387a7e": source_hashes,
        "expected_group_training_commit": {
            "configured": TRAINING_COMMIT,
            "canonical": expected_training_commit,
        },
        "active_training_commit": active_training_commit,
        "test_split_policy": "not loaded or evaluated",
    }
    if write_outputs:
        output_root.mkdir(parents=True, exist_ok=True)
        dump_json(plan, output_root / "experiment_plan.json")
        dump_json(registry, output_root / "candidate_source_registry.json")
    LOGGER.info(
        "Validated 30 reusable anchors and planned 20 new CUDA validation-only runs."
    )
    return {"plan": plan, "registry": registry, "new_runs": new_runs}


def build_training_command(
    record: Mapping[str, Any], dataset_dir: Path, resume: Path | None = None
) -> list[str]:
    """Build one explicit CUDA validation-only command from registry features."""

    model = str(record["model"])
    script = "scripts/run_baselines.py" if model == "gru" else "scripts/run_attention.py"
    command = [sys.executable, script]
    if model == "gru":
        command.append("gru")
    command.extend(
        [
            "--dataset-dir", str(dataset_dir),
            "--output-dir", str(record["source_run_directory"]),
            "--seed", str(record["seed"]),
            "--device", "cuda",
            "--learning-rate", "0.001",
            "--weight-decay", "0.0001",
            "--batch-size", "256",
            "--max-epochs", "50",
            "--patience", "8",
            "--num-workers", "0",
            "--dynamic-features", ",".join(record["feature_names"]),
            "--validation-only",
        ]
    )
    if resume is not None:
        command.extend(["--resume", str(resume)])
    return command


def validate_resume_compatibility(
    run_dir: Path, record: Mapping[str, Any], dataset_dir: Path, active_commit: str
) -> Path | None:
    """Allow resume only when an interrupted config exactly matches the plan."""

    last = run_dir / "last_model.pt"
    config_path = run_dir / "config.json"
    if not last.exists() and not config_path.exists():
        return None
    if not last.exists() or not config_path.exists():
        raise ValueError(f"Partial resume artifacts in {run_dir}.")
    config = load_json(config_path)
    expected = {
        "seed": record["seed"],
        "dynamic_feature_names": record["feature_names"],
        "dataset_dir": str(dataset_dir),
        "output_dir": str(run_dir),
        "git_commit_hash": active_commit,
        "device": "cuda",
        "resolved_device": "cuda",
        "backend": "cuda",
        "smoke": False,
        **FIXED_SETTINGS,
    }
    mismatches = {key: (expected_value, config.get(key)) for key, expected_value in expected.items() if config.get(key) != expected_value}
    if mismatches:
        raise ValueError(f"Incompatible resume in {run_dir}: {mismatches}")
    if record["model"] == "attention" and config.get("model_name") != "FactorizedAttentionGRU":
        raise ValueError(f"Incompatible attention model identity in {run_dir}.")
    if record["model"] == "gru" and config.get("model_name") is not None:
        raise ValueError(f"Incompatible GRU model identity in {run_dir}.")
    return last


def update_registry_after_run(
    registry: list[dict[str, Any]], record: Mapping[str, Any], dataset_dir: Path
) -> None:
    validated = validate_run_directory(
        Path(str(record["source_run_directory"])),
        str(record["candidate"]),
        str(record["model"]),
        int(record["seed"]),
        record["feature_names"],
        dataset_dir,
    )
    for row in registry:
        if all(row[key] == record[key] for key in ("candidate", "model", "seed")):
            row["completion_status"] = "complete"
            row["training_commit"] = validated["config"]["git_commit_hash"]
            row["test_evaluated"] = False
            return
    raise KeyError("New run record was not found in registry.")
