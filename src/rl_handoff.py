"""Repository audit and state contract for the predictive-to-RL handoff."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from src.frozen_candidate_retraining import dump_json, sha256_file
from src.frozen_predictive_test_evaluation import PRIMARY_FEATURES

LOGGER = logging.getLogger(__name__)

SEARCH_TERMS = (
    "gymnasium",
    "gym.Env",
    "reset(",
    "step(",
    "action_space",
    "observation_space",
    "propofol",
    "pk-pd",
    "pkpd",
    "Schnider",
    "Marsh",
    "Minto",
    "reward",
    "actor",
    "critic",
    "replay_buffer",
    "target_bis",
    "bis_target",
    "simulator",
    "controller",
    "reinforcement",
    "stable_baselines",
    "SAC",
    "TD3",
    "PPO",
    "DDPG",
)
TEXT_SUFFIXES = {".py", ".ipynb", ".md", ".json", ".yaml", ".yml", ".toml", ".txt"}
EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__", "data"}
EXCLUDED_AUDIT_FILES = {
    "docs/rl_handoff_requirements.md",
    "scripts/audit_rl_handoff.py",
    "src/rl_handoff.py",
    "tests/test_frozen_predictive_test_evaluation.py",
}
REQUIRED_EXTERNAL_INPUTS = (
    "professor RL repository or attached implementation",
    "environment class and module path",
    "reset() and step() API contracts",
    "ordered baseline state schema and history construction",
    "action range, unit, clipping, and 10-second hold semantics",
    "reward definition and all coefficients",
    "episode termination and truncation rules",
    "patient-level train/validation/test split identifiers",
    "RL algorithm and hyperparameters",
    "baseline policy checkpoint and compatibility metadata",
    "closed-loop evaluation metrics and protocol",
    "random seeds and determinism settings",
)


def _iter_text_files(repo_dir: Path) -> Iterable[Path]:
    for path in repo_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(repo_dir)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if relative.as_posix() in EXCLUDED_AUDIT_FILES:
            continue
        if relative.as_posix().startswith("outputs/rl_handoff/"):
            continue
        yield path


def _search_repository(repo_dir: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    patterns = []
    for term in SEARCH_TERMS:
        escaped = re.escape(term)
        if term.replace("_", "").isalnum():
            escaped = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
        patterns.append((term, re.compile(escaped, re.IGNORECASE)))
    for path in _iter_text_files(repo_dir):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as error:
            LOGGER.warning("Could not inspect %s: %s", path, error)
            continue
        for number, line in enumerate(lines, start=1):
            matched = sorted({term for term, pattern in patterns if pattern.search(line)})
            if matched:
                findings.append(
                    {
                        "path": path.relative_to(repo_dir).as_posix(),
                        "line": number,
                        "matched_terms": matched,
                        "text": line.strip()[:300],
                    }
                )
    return findings


def _classify_rl_implementation(findings: list[dict[str, Any]]) -> dict[str, bool]:
    by_path: dict[str, str] = {}
    for item in findings:
        by_path.setdefault(item["path"], "")
        by_path[item["path"]] += "\n" + item["text"]
    environment = any(
        re.search(r"class\s+\w*(?:Env|Environment)\b", text, re.IGNORECASE)
        and "reset(" in text
        and "step(" in text
        for text in by_path.values()
    )
    simulator = any(
        re.search(r"class\s+\w*(?:Simulator|PKPD|Patient)\b", text, re.IGNORECASE)
        for text in by_path.values()
    )
    algorithm = any(
        re.search(r"class\s+\w*(?:Actor|Critic|Agent|Policy)\b", text, re.IGNORECASE)
        for text in by_path.values()
    )
    return {
        "gymnasium_environment_found": environment,
        "pk_pd_simulator_found": simulator,
        "rl_agent_or_algorithm_found": algorithm,
        "baseline_policy_checkpoint_found": False,
    }


def build_rl_repository_audit(repo_dir: Path) -> dict[str, Any]:
    """Audit local code without inferring an unavailable professor implementation."""

    findings = _search_repository(repo_dir)
    implementation = _classify_rl_implementation(findings)
    found = any(implementation.values())
    return {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repo_dir.as_posix(),
        "search_terms": list(SEARCH_TERMS),
        "implementation_status": implementation,
        "professor_rl_implementation_found": found,
        "blocked_missing_external_rl_implementation": not found,
        "rl_training_started": False,
        "finding_count": len(findings),
        "findings": findings,
        "conclusion": (
            "No Gymnasium environment, PK-PD simulator, RL agent, reward, action contract, "
            "or baseline policy checkpoint is implemented in this repository. The propofol "
            "references in main.py crop observational data and are not a control environment."
            if not found
            else "Potential RL implementation signatures were found and require manual contract review."
        ),
        "required_external_inputs": list(REQUIRED_EXTERNAL_INPUTS),
    }


def build_rl_state_contract(dataset_dir: Path) -> dict[str, Any]:
    """Build the predictive contract while leaving the control-aware state unresolved."""

    metadata_path = dataset_dir / "dataset_metadata.json"
    metadata: Mapping[str, Any] = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    preprocessing_path = dataset_dir / "preprocessing.pkl"
    return {
        "contract_version": 1,
        "predictive_compact_state": {
            "dynamic_feature_names_ordered": list(PRIMARY_FEATURES),
            "dynamic_feature_count": len(PRIMARY_FEATURES),
            "static_covariate_names_ordered": list(
                metadata.get("static_feature_names", [])
            ),
            "action_interval_seconds": 10,
            "sampling_interval_seconds": 10,
            "history_window_seconds": 60,
            "history_steps": 6,
            "prediction_horizon_seconds": 30,
            "history_order": "oldest to newest: t-50,t-40,t-30,t-20,t-10,t",
            "preprocessing_artifact": preprocessing_path.as_posix(),
            "preprocessing_sha256": (
                sha256_file(preprocessing_path) if preprocessing_path.is_file() else None
            ),
            "missing_data_policy": (
                "use the train-fitted imputation/normalization artifact and pass the aligned "
                "observation mask; unsupported tracks must remain explicitly missing"
            ),
            "predictive_only": True,
            "not_an_rl_optimality_claim": True,
        },
        "control_aware_state": {
            "status": "blocked_pending_external_baseline_contract",
            "blocked_missing_external_rl_implementation": True,
            "must_begin_from_external_baseline_state": True,
            "must_preserve_required_baseline_variables": [
                "current and recent actions",
                "propofol delivery and drug-history variables",
                "BIS target or target error variables",
                "remifentanil variables",
                "exogenous disturbances and patient context",
            ],
            "must_not_replace_baseline_with_predictive_subset": True,
            "candidate_augmentation": list(PRIMARY_FEATURES),
        },
        "action_contract": {
            "action_interval_seconds": 10,
            "range": None,
            "unit": None,
            "blocked_reason": "External professor RL environment is unavailable.",
        },
        "reward_contract": None,
        "termination_contract": None,
    }


def predictive_to_control_mapping() -> pd.DataFrame:
    """Map predictive features to control concerns without inventing baseline fields."""

    roles = {
        "bis": ("feedback", True, "Current controlled physiological output."),
        "bis_sqi": ("quality", True, "Quality signal for BIS reliability."),
        "ppf_rate": ("action_or_exposure", True, "May overlap the propofol action; use external action semantics."),
        "ppf_volume": ("drug_history", True, "Cumulative exposure is not a substitute for the full action history."),
        "ppf_cp": ("pk_state", True, "Observed/model-derived concentration provenance must match the simulator."),
        "rftn_volume": ("co_medication_history", True, "Only retained remifentanil feature; external baseline remifentanil state must remain."),
        "bis_slope": ("derived_feedback", True, "Derivation must use only causal 10-second history."),
    }
    return pd.DataFrame(
        [
            {
                "predictive_feature": feature,
                "predictive_order": index,
                "predictive_role": roles[feature][0],
                "candidate_control_input": roles[feature][1],
                "required_by_external_baseline": "unknown_pending_external_contract",
                "control_caution": roles[feature][2],
            }
            for index, feature in enumerate(PRIMARY_FEATURES, start=1)
        ]
    )


def run_rl_handoff_audit(
    repo_dir: Path, dataset_dir: Path, output_dir: Path
) -> dict[str, Any]:
    """Write the audit, state contract, and feature mapping without RL training."""

    audit = build_rl_repository_audit(repo_dir)
    contract = build_rl_state_contract(dataset_dir)
    mapping = predictive_to_control_mapping()
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(audit, output_dir / "rl_repository_audit.json")
    dump_json(contract, output_dir / "rl_state_contract.json")
    mapping.to_csv(output_dir / "predictive_to_control_feature_mapping.csv", index=False)
    return {
        "blocked_missing_external_rl_implementation": audit[
            "blocked_missing_external_rl_implementation"
        ],
        "rl_training_started": False,
        "output_dir": str(output_dir),
    }
