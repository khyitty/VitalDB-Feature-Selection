"""Package the completed predictive stage without copying large artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.frozen_candidate_retraining import dump_json, sha256_file
from src.frozen_predictive_test_evaluation import (
    CANDIDATES,
    MODELS,
    PRIMARY_CANDIDATE,
    PRIMARY_FEATURES,
    REFERENCE_CANDIDATE,
    REFERENCE_FEATURES,
    SEEDS,
    validate_frozen_decision,
)

LOGGER = logging.getLogger(__name__)

OUTPUT_NAMES = (
    "predictive_state_contract.json",
    "frozen_candidate_definition.json",
    "checkpoint_inventory.csv",
    "experiment_lineage.csv",
    "reproducibility_checklist.md",
    "predictive_stage_summary.md",
    "external_rl_assets_request.md",
)

LINEAGE = (
    ("data_preparation", "Leakage-safe patient split, train-fitted preprocessing, and 10-second windows"),
    ("gpu_smoke", "GRU and explicit-attention Colab GPU smoke verification"),
    ("group_retraining_40_runs", "Four feature groups by two models by five seeds"),
    ("group_analysis", "Validation-only grouped comparison and paired statistics"),
    ("train_only_predictive_feature_selection", "Train-only predictive candidate construction"),
    ("frozen_candidate_retraining", "Twenty new and thirty reused validation-only runs"),
    ("frozen_candidate_analysis_50_runs", "Five candidates by two models by five seeds"),
    ("frozen_predictive_decision", "Pre-test strict_consensus primary freeze"),
    ("one_time_test_workflow", "Guarded inference-only internal held-out evaluation"),
    ("rl_audit_blocked", "External professor RL implementation not present"),
)


def _git_head(repo_dir: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def expected_checkpoint_inventory() -> pd.DataFrame:
    """Return the exact expected 20-run inventory with unresolved external hashes."""

    rows = []
    for candidate in CANDIDATES:
        for model in MODELS:
            for seed in SEEDS:
                if candidate == PRIMARY_CANDIDATE:
                    source = (
                        "outputs/frozen_candidate_retraining_validation_only/"
                        f"strict_consensus/{model}/seed_{seed}"
                    )
                else:
                    source = (
                        "outputs/group_retraining_validation_only/"
                        f"full17/{model}/seed_{seed}"
                    )
                features = (
                    PRIMARY_FEATURES
                    if candidate == PRIMARY_CANDIDATE
                    else REFERENCE_FEATURES
                )
                rows.append(
                    {
                        "candidate": candidate,
                        "model": model,
                        "seed": seed,
                        "source_run_directory": source,
                        "checkpoint_path": f"{source}/best_model.pt",
                        "checkpoint_name": "best_model.pt",
                        "checkpoint_sha256": "",
                        "config_sha256": "",
                        "training_git_commit": "",
                        "dynamic_feature_names": json.dumps(
                            list(features), separators=(",", ":")
                        ),
                        "hash_status": "pending_drive_preflight",
                        "large_artifact_copied": False,
                    }
                )
    return pd.DataFrame(rows)


def validate_checkpoint_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate exact candidate/model/seed identities and best-checkpoint policy."""

    required = {
        "candidate",
        "model",
        "seed",
        "checkpoint_path",
        "checkpoint_name",
        "checkpoint_sha256",
        "training_git_commit",
        "dynamic_feature_names",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Checkpoint inventory lacks required columns: {missing}")
    expected = {
        (candidate, model, seed)
        for candidate in CANDIDATES
        for model in MODELS
        for seed in SEEDS
    }
    observed = {
        (str(row.candidate), str(row.model), int(row.seed))
        for row in frame.itertuples()
    }
    if len(frame) != 20 or len(observed) != 20 or observed != expected:
        raise ValueError("Predictive handoff requires exactly 20 unique frozen checkpoints.")
    if set(frame["checkpoint_name"]) != {"best_model.pt"}:
        raise ValueError("Predictive handoff may reference only best_model.pt.")
    for row in frame.itertuples():
        expected_features = (
            PRIMARY_FEATURES
            if row.candidate == PRIMARY_CANDIDATE
            else REFERENCE_FEATURES
        )
        if tuple(json.loads(row.dynamic_feature_names)) != expected_features:
            raise ValueError(f"Feature order mismatch in checkpoint inventory: {row}")
    result = frame.copy()
    if "large_artifact_copied" not in result:
        result["large_artifact_copied"] = False
    if result["large_artifact_copied"].astype(bool).any():
        raise ValueError("Large checkpoints must be referenced, not copied.")
    return result.sort_values(["candidate", "model", "seed"]).reset_index(drop=True)


def _state_contract(dataset_dir: Path) -> dict[str, Any]:
    metadata_path = dataset_dir / "dataset_metadata.json"
    preprocessing_path = dataset_dir / "preprocessing.pkl"
    if not metadata_path.is_file() or not preprocessing_path.is_file():
        raise FileNotFoundError("Dataset metadata and train-fitted preprocessing are required.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("resampling_interval_seconds") != 10
        or metadata.get("history_window_seconds") != 60
        or metadata.get("history_steps") != 6
        or metadata.get("prediction_horizon_seconds") != 30
    ):
        raise ValueError("Dataset timing differs from the frozen 10/60/30-second contract.")
    static = list(metadata.get("static_feature_names", []))
    return {
        "role": "predictive-only future-BIS state",
        "primary_candidate": PRIMARY_CANDIDATE,
        "dynamic_feature_names_ordered": list(PRIMARY_FEATURES),
        "static_covariate_names_ordered": static,
        "sampling_interval_seconds": 10,
        "action_interval_seconds_for_later_rl": 10,
        "history_window_seconds": 60,
        "history_steps": 6,
        "prediction_horizon_seconds": 30,
        "dynamic_tensor_shape": ["batch", 6, len(PRIMARY_FEATURES)],
        "static_tensor_shape": ["batch", len(static)],
        "observation_mask_shape": ["batch", 6, len(PRIMARY_FEATURES)],
        "preprocessing_artifact": preprocessing_path.as_posix(),
        "preprocessing_sha256": sha256_file(preprocessing_path),
        "missing_value_policy": (
            "Apply only train-fitted imputation and normalization; retain the aligned "
            "observation mask and never fabricate unavailable tracks."
        ),
        "control_warning": (
            "Do not use this predictive subset directly as a control-aware state. "
            "Preserve the external baseline action, target, drug-history, remifentanil, "
            "and exogenous variables before adding predictive features."
        ),
    }


def _lineage(commit: str | None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage_order": index,
                "stage": stage,
                "description": description,
                "recorded_by_commit": commit,
                "status": "complete" if stage != "rl_audit_blocked" else "blocked_external",
            }
            for index, (stage, description) in enumerate(LINEAGE, start=1)
        ]
    )


def _reproducibility_checklist(inventory_complete: bool) -> str:
    inventory_mark = "x" if inventory_complete else " "
    return f"""# Predictive Stage Reproducibility Checklist

- [x] Patient-level train/validation/test split established before windows.
- [x] Imputation and normalization fitted on training cases only.
- [x] Sampling interval 10 seconds, history 60 seconds, horizon 30 seconds.
- [x] Primary frozen before held-out test access.
- [x] Primary is `strict_consensus`; reference is `full17_reference`.
- [x] `compact_consensus` excluded from held-out test.
- [{inventory_mark}] All 20 Drive checkpoint SHA256 values recorded by frozen-test preflight.
- [x] Only validation-selected `best_model.pt` is permitted.
- [x] Predictive state is explicitly not an RL-optimality claim.
- [x] No checkpoint, NPZ, or other large artifact copied into this package.
"""


def build_predictive_stage_handoff(
    *,
    repo_dir: Path,
    dataset_dir: Path,
    decision_path: Path,
    output_dir: Path,
    checkpoint_inventory_path: Path | None = None,
) -> dict[str, Any]:
    """Write a hash-addressed predictive-stage package containing references only."""

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    validate_frozen_decision(decision)
    state = _state_contract(dataset_dir)
    if checkpoint_inventory_path is None:
        inventory = expected_checkpoint_inventory()
    else:
        inventory = pd.read_csv(checkpoint_inventory_path)
        inventory["hash_status"] = "verified_by_frozen_test_preflight"
        inventory["large_artifact_copied"] = False
    inventory = validate_checkpoint_inventory(inventory)
    hashes_complete = bool(
        inventory["checkpoint_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all()
    )
    commit = _git_head(repo_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_definition = {
        "decision_sha256": sha256_file(decision_path),
        "primary": {
            "name": PRIMARY_CANDIDATE,
            "features_ordered": list(PRIMARY_FEATURES),
            "role": "frozen primary predictive candidate",
        },
        "reference": {
            "name": REFERENCE_CANDIDATE,
            "features_ordered": list(REFERENCE_FEATURES),
            "role": "frozen predictive reference",
        },
        "excluded_test_candidate": "compact_consensus",
        "candidate_changes_after_test_prohibited": True,
    }
    dump_json(state, output_dir / "predictive_state_contract.json")
    dump_json(candidate_definition, output_dir / "frozen_candidate_definition.json")
    inventory.to_csv(output_dir / "checkpoint_inventory.csv", index=False)
    _lineage(commit).to_csv(output_dir / "experiment_lineage.csv", index=False)
    (output_dir / "reproducibility_checklist.md").write_text(
        _reproducibility_checklist(hashes_complete), encoding="utf-8"
    )
    (output_dir / "predictive_stage_summary.md").write_text(
        "# Predictive Stage Summary\n\n"
        "The predictive stage is frozen with `strict_consensus` as the seven-feature "
        "primary and `full17_reference` as reference. The one-time internal held-out "
        "workflow is inference-only and cannot change the candidate. This package "
        "contains paths, hashes, schemas, feature order, and lineage only. It does not "
        "contain checkpoints or datasets. Predictive utility does not guarantee "
        "closed-loop control utility or external validity.\n",
        encoding="utf-8",
    )
    (output_dir / "external_rl_assets_request.md").write_text(
        "# External RL Assets Request\n\n"
        "Provide the professor RL repository or archive, environment class, reset/step "
        "contract, ordered baseline state, propofol action range and unit, reward, "
        "termination rules, PK-PD simulator, patient split, algorithm configuration, "
        "baseline checkpoint, evaluation metrics, and random seeds. The external "
        "package will be inspected statically and will not be executed during intake.\n",
        encoding="utf-8",
    )

    output_fingerprints = [
        {
            "path": name,
            "sha256": sha256_file(output_dir / name),
            "size_bytes": (output_dir / name).stat().st_size,
        }
        for name in OUTPUT_NAMES
    ]
    checkpoint_fingerprints = [
        {
            "candidate": str(row.candidate),
            "model": str(row.model),
            "seed": int(row.seed),
            "checkpoint_path": str(row.checkpoint_path),
            "checkpoint_sha256": str(row.checkpoint_sha256),
            "config_sha256": str(row.config_sha256),
            "hash_status": str(row.hash_status),
        }
        for row in inventory.itertuples()
    ]
    manifest = {
        "package_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "package_git_commit": commit,
        "decision_path": decision_path.as_posix(),
        "decision_sha256": sha256_file(decision_path),
        "dataset_metadata_sha256": sha256_file(dataset_dir / "dataset_metadata.json"),
        "preprocessing_sha256": state["preprocessing_sha256"],
        "checkpoint_count": 20,
        "checkpoint_hashes_complete": hashes_complete,
        "checkpoint_fingerprints": checkpoint_fingerprints,
        "large_artifacts_copied": False,
        "predictive_control_distinction_recorded": True,
        "state_contract_sha256": _canonical_hash(state),
        "output_fingerprints": output_fingerprints,
    }
    dump_json(manifest, output_dir / "predictive_stage_manifest.json")
    forbidden = [
        path.name
        for path in output_dir.iterdir()
        if path.suffix.lower() in {".pt", ".pth", ".npz", ".pkl", ".zip"}
    ]
    if forbidden:
        raise ValueError(f"Large artifacts were copied into the handoff package: {forbidden}")
    LOGGER.info("Wrote predictive-stage handoff package to %s", output_dir)
    return {
        "output_dir": str(output_dir),
        "checkpoint_count": 20,
        "checkpoint_hashes_complete": hashes_complete,
        "large_artifacts_copied": False,
    }
