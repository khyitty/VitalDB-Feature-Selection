"""Frozen protocol construction for the 20-run primary-state PPO full study."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import stable_baselines3
import torch

from src.rl_env.config import RESEARCH_ONLY_WARNING, action_bounds_from_profile
from src.rl_env.reward import reward_profile_registry

from .cohort import CohortBundle, scenarios_for_split
from .config import PPOConfig, PrimaryStateProfile
from .io import atomic_write_json, atomic_write_text
from .manifests import canonical_json
from .pilot_protocol import resolve_execution_device
from .policy_registry import primary_state_policy_contract, validate_state_only_comparison
from .run_status import package_versions, repository_commit


FULL_PROFILES = (
    "original_reconstructed",
    "all_supported",
    "prediction_minimal",
    "selected_control_core",
)
FULL_SEEDS = (7, 21, 42, 84, 123)
FULL_PROTOCOL_FILENAME = "frozen_primary_state_full_protocol.json"
FULL_CONFIRMATION = "RUN_20_PRIMARY_STATE_FULL_RUNS"

_HASH_KEYS = (
    "protocol_version",
    "source_config_sha256",
    "implementation_commit",
    "execution_device",
    "profiles",
    "seeds",
    "inventory",
    "policy_contracts",
    "ppo",
    "observation",
    "reward",
    "action",
    "environment",
    "cohort_contract",
    "evaluation",
    "checkpoint_selection",
    "resume",
    "initialization",
    "backend_selection",
    "backend_decision",
    "test_seal",
    "interpretation",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def source_config_sha256(path: Path) -> str:
    """Hash the exact committed full source configuration bytes."""

    return _sha256_bytes(path.read_bytes())


def load_full_source(path: Path) -> dict[str, Any]:
    """Load and strictly validate the full-study source configuration."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Full source configuration is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Full source configuration is invalid JSON: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Full source configuration requires an object with schema_version=1.")
    if payload.get("protocol_version") != "ppo_primary_state_full_v1":
        raise ValueError("Unexpected primary-state full protocol_version.")
    if tuple(payload.get("profiles", ())) != FULL_PROFILES:
        raise ValueError(f"Full profiles must be exactly {FULL_PROFILES}.")
    if tuple(payload.get("seeds", ())) != FULL_SEEDS:
        raise ValueError(f"Full seeds must be exactly {FULL_SEEDS}.")
    if payload.get("library", {}).get("stable_baselines3") != "2.9.0":
        raise ValueError("Full source must pin stable_baselines3 2.9.0.")
    ppo = PPOConfig(**payload.get("ppo", {}))
    if ppo.profile_name != "ppo_primary_state_full_v1":
        raise ValueError("Full PPO profile_name is not ppo_primary_state_full_v1.")
    if ppo.total_timesteps != 1_024_000:
        raise ValueError("Full training budget must remain 1024000 steps per run.")
    if ppo.evaluation_frequency_timesteps != 51_200:
        raise ValueError("Full validation interval must remain 51200 steps.")
    if ppo.total_timesteps % ppo.n_steps or ppo.evaluation_frequency_timesteps % ppo.n_steps:
        raise ValueError("Full training and evaluation boundaries must align to PPO rollouts.")
    initialization = payload.get("initialization", {})
    if initialization != {
        "mode": "fresh_random",
        "pilot_checkpoint_reuse": False,
        "pilot_output_import": False,
    }:
        raise ValueError("Full runs must use fresh random initialization without pilot reuse.")
    cohort = payload.get("cohort", {})
    if cohort.get("test_trajectory_access") is not False:
        raise ValueError("Full source must prohibit test trajectory access.")
    if cohort.get("test_outcome_access") is not False:
        raise ValueError("Full source must prohibit test outcome access.")
    contracts = [
        primary_state_policy_contract(cast(PrimaryStateProfile, profile))
        for profile in FULL_PROFILES
    ]
    validate_state_only_comparison(contracts)
    expected_policy = {
        "class": "MlpPolicy",
        "feature_extractor": "stable_baselines3.common.torch_layers.FlattenExtractor",
        "hidden_layers": [64, 64],
        "activation": "Tanh",
        "optimizer": "Adam",
    }
    if payload.get("policy") != expected_policy:
        raise ValueError("Full common-policy contract changed.")
    return payload


def full_protocol_hash(payload: Mapping[str, Any]) -> str:
    """Hash all scientific, cohort, implementation, and backend compatibility fields."""

    selected = {key: payload.get(key) for key in _HASH_KEYS}
    return _sha256_bytes(canonical_json(selected).encode("utf-8"))


def build_full_protocol(
    *,
    source_path: Path,
    repo_dir: Path,
    cohort: CohortBundle,
    execution_device: str,
    backend_decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the full source to one exact commit, cohort, and selected backend."""

    source = load_full_source(source_path)
    required_sb3 = source["library"]["stable_baselines3"]
    if stable_baselines3.__version__ != required_sb3:
        raise ValueError(
            "Observed Stable-Baselines3 differs from the full source: "
            f"required={required_sb3}, observed={stable_baselines3.__version__}."
        )
    device = resolve_execution_device(execution_device)
    commit = repository_commit(repo_dir)
    expected_decision = {
        "implementation_commit": commit,
        "source_config_sha256": source_config_sha256(source_path),
        "cohort_fingerprint": cohort.fingerprint,
        "selected_backend": device,
        "scientific_metrics_used_for_backend_selection": False,
    }
    decision_mismatches = {
        key: {"expected": value, "observed": backend_decision.get(key)}
        for key, value in expected_decision.items()
        if backend_decision.get(key) != value
    }
    if decision_mismatches:
        raise ValueError(
            "Backend decision is incompatible with the requested full protocol: "
            f"{decision_mismatches}."
        )
    ppo = PPOConfig(**source["ppo"])
    profiles = tuple(cast(PrimaryStateProfile, item) for item in source["profiles"])
    contracts = [primary_state_policy_contract(profile) for profile in profiles]
    validate_state_only_comparison(contracts)
    scenarios = scenarios_for_split(cohort, "validation", base_seed=100_000)
    if len(scenarios) != ppo.evaluation_episode_count:
        raise ValueError(
            "Full validation episode count differs from the validation cohort: "
            f"config={ppo.evaluation_episode_count}, cohort={len(scenarios)}."
        )
    bounds = action_bounds_from_profile(ppo.action_bounds_profile)
    inventory = [
        {"state_profile": profile, "seed": seed, "run_id": f"{profile}/seed_{seed}"}
        for profile in profiles
        for seed in source["seeds"]
    ]
    manifest = cohort.cohort.manifest
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": source["protocol_version"],
        "source_config_path": str(source_path.resolve()),
        "source_config_sha256": source_config_sha256(source_path),
        "implementation_commit": commit,
        "execution_device": device,
        "runtime_at_creation": {
            "packages": package_versions(),
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "library": {
            "stable_baselines3_required": required_sb3,
            "stable_baselines3_observed": stable_baselines3.__version__,
        },
        "profiles": list(profiles),
        "seeds": list(source["seeds"]),
        "inventory": inventory,
        "inventory_count": len(inventory),
        "confirmation_text": FULL_CONFIRMATION,
        "policy_contracts": {
            contract.state_profile: {
                "policy_class": contract.policy_class,
                "feature_extractor": contract.feature_extractor,
                "hidden_layers": list(contract.hidden_layers),
                "activation": contract.activation,
                "optimizer": source["policy"]["optimizer"],
                "ordered_feature_names": list(contract.ordered_feature_names),
                "observation_dimension": contract.observation_dimension,
                "architecture_signature": list(contract.architecture_signature),
            }
            for contract in contracts
        },
        "ppo": ppo.as_dict(),
        "observation": source["observation"],
        "reward": {
            **source["reward"],
            "profile": ppo.reward_profile,
            "profile_registry": reward_profile_registry(),
        },
        "action": {
            **asdict(bounds),
            "policy_space": [-1.0, 1.0],
            "raw_policy_distribution": "unbounded diagonal Gaussian",
            "sb3_transform": "single clip to normalized Box [-1,1]",
            "physical_transform": "low + (bounded + 1) * (high-low) / 2",
            "dose_transform": "physical_mg_per_min * 10/60",
        },
        "environment": {
            "action_interval_seconds": 10.0,
            "internal_dt_seconds": 1.0,
            "history_window_seconds": 60.0,
            "episode_duration_seconds": ppo.episode_duration_seconds,
            "deterministic_simulator": ppo.deterministic_simulator,
            "target_bis": 50.0,
            "initial_drug_state": "zero",
            "remifentanil_schedule": "scenario-ID-derived piecewise schedule",
        },
        "cohort_contract": {
            "fingerprint": cohort.fingerprint,
            "case_counts": {
                "train": len(manifest.train_patient_ids),
                "validation": len(manifest.validation_patient_ids),
                "test": len(manifest.test_patient_ids),
            },
            "split_patient_ids": {
                "train": list(manifest.train_patient_ids),
                "validation": list(manifest.validation_patient_ids),
                "test": list(manifest.test_patient_ids),
            },
            "patient_overlap": False,
            "training_sampling": source["cohort"]["training_sampling"],
            "validation_sampling": source["cohort"]["validation_sampling"],
            "validation_scenario_ids": [item.scenario_id for item in scenarios],
        },
        "cohort_creation_provenance": {
            "demographics_source": cohort.demographics_source,
            "demographics_source_kind": cohort.demographics_source_kind,
            "demographics_source_fingerprint": cohort.demographics_source_fingerprint,
            "split_source": cohort.split_source,
            "access_manifest": cohort.access_manifest,
        },
        "evaluation": source["evaluation"],
        "checkpoint_selection": {
            "split": "validation",
            "primary": "mean patient/scenario BIS target MAE ascending",
            "tie_breaker_1": "mean fraction time in BIS 40-60 descending",
            "tie_breaker_2": "mean absolute action change sum ascending",
        },
        "resume": source["resume"],
        "initialization": source["initialization"],
        "backend_selection": source["backend_selection"],
        "backend_decision": dict(backend_decision),
        "test_seal": {
            "test_split_membership_loaded": True,
            "test_demographics_loaded": True,
            "test_trajectory_loaded": False,
            "test_outcomes_evaluated": False,
            "test_policy_rollout_performed": False,
            "test_checkpoint_selection": False,
        },
        "interpretation": source["interpretation"],
        "research_warning": RESEARCH_ONLY_WARNING,
    }
    payload["protocol_hash"] = full_protocol_hash(payload)
    verify_full_protocol(payload)
    return payload


def verify_full_protocol(payload: Mapping[str, Any]) -> None:
    """Reject mutation or any mismatch in the frozen full protocol."""

    if payload.get("protocol_hash") != full_protocol_hash(payload):
        raise ValueError("Primary-state full protocol hash mismatch.")
    if tuple(payload.get("profiles", ())) != FULL_PROFILES:
        raise ValueError("Frozen full profile inventory changed.")
    if tuple(payload.get("seeds", ())) != FULL_SEEDS:
        raise ValueError("Frozen full seed inventory changed.")
    inventory = payload.get("inventory", ())
    if payload.get("inventory_count") != 20 or len(inventory) != 20:
        raise ValueError("Frozen full protocol requires four profiles by five seeds.")
    expected = {(profile, seed) for profile in FULL_PROFILES for seed in FULL_SEEDS}
    observed = {(item.get("state_profile"), int(item.get("seed"))) for item in inventory}
    if observed != expected:
        raise ValueError("Frozen full inventory identities changed.")
    if payload.get("execution_device") not in {"cpu", "cuda"}:
        raise ValueError("Frozen full execution_device is invalid.")
    library = payload.get("library", {})
    if library.get("stable_baselines3_required") != "2.9.0":
        raise ValueError("Frozen full protocol no longer pins stable_baselines3 2.9.0.")
    if library.get("stable_baselines3_observed") != library.get("stable_baselines3_required"):
        raise ValueError("Frozen full protocol was created with incompatible SB3.")
    ppo = PPOConfig(**cast(Mapping[str, Any], payload.get("ppo", {})))
    if ppo.total_timesteps != 1_024_000 or ppo.evaluation_frequency_timesteps != 51_200:
        raise ValueError("Frozen full training or evaluation budget changed.")
    initialization = payload.get("initialization", {})
    if initialization.get("mode") != "fresh_random":
        raise ValueError("Frozen full initialization is not fresh_random.")
    if initialization.get("pilot_checkpoint_reuse") is not False:
        raise ValueError("Frozen full protocol permits pilot checkpoint reuse.")
    decision = cast(Mapping[str, Any], payload.get("backend_decision", {}))
    if decision.get("selected_backend") != payload.get("execution_device"):
        raise ValueError("Frozen full backend decision differs from execution_device.")
    if decision.get("implementation_commit") != payload.get("implementation_commit"):
        raise ValueError("Frozen full backend decision differs from implementation commit.")
    if decision.get("source_config_sha256") != payload.get("source_config_sha256"):
        raise ValueError("Frozen full backend decision differs from source config hash.")
    if decision.get("cohort_fingerprint") != payload.get("cohort_contract", {}).get("fingerprint"):
        raise ValueError("Frozen full backend decision differs from cohort fingerprint.")
    if decision.get("scientific_metrics_used_for_backend_selection") is not False:
        raise ValueError("Frozen full backend decision used scientific metrics.")
    seals = cast(Mapping[str, Any], payload.get("test_seal", {}))
    forbidden = (
        "test_trajectory_loaded",
        "test_outcomes_evaluated",
        "test_policy_rollout_performed",
        "test_checkpoint_selection",
    )
    if any(seals.get(key) is not False for key in forbidden):
        raise ValueError("Frozen full test-cohort seal is not intact.")
    contracts = list(cast(Mapping[str, Any], payload["policy_contracts"]).values())
    signatures = {canonical_json(item["architecture_signature"]) for item in contracts}
    if len(signatures) != 1:
        raise ValueError("Frozen full protocol changes architecture between profiles.")


def _protocol_markdown(payload: Mapping[str, Any]) -> str:
    return f"""# Frozen Primary-State PPO Full Protocol

- Protocol: `{payload['protocol_version']}`
- Hash: `{payload['protocol_hash']}`
- Implementation: `{payload['implementation_commit']}`
- Backend: `{payload['execution_device']}`
- Profiles: `{', '.join(payload['profiles'])}`
- Seeds: `{payload['seeds']}`
- Runs: `{payload['inventory_count']}`
- Timesteps per run: `{payload['ppo']['total_timesteps']}`
- Validation interval: `{payload['ppo']['evaluation_frequency_timesteps']}`
- Initialization: fresh random; pilot checkpoints forbidden
- Cohort fingerprint: `{payload['cohort_contract']['fingerprint']}`
- Test trajectories/outcomes/policy rollouts: sealed until final state freeze
"""


def freeze_full_protocol(
    payload: dict[str, Any],
    output_dir: Path,
    *,
    run_output_root: Path | None = None,
) -> dict[str, Any]:
    """Create once and refuse incompatible or pilot-contaminated output roots."""

    verify_full_protocol(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / FULL_PROTOCOL_FILENAME
    run_files = (
        sorted(item for item in run_output_root.rglob("*") if item.is_file())
        if run_output_root is not None and run_output_root.exists()
        else []
    )
    if any("ppo_primary_state_pilot" in str(item).lower() for item in (output_dir, run_output_root)):
        raise ValueError("Pilot and full protocol/output directories must remain separate.")
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        verify_full_protocol(observed)
        if observed["protocol_hash"] != payload["protocol_hash"]:
            raise ValueError(
                "Existing full protocol differs from requested commit, cohort, backend, "
                "or scientific configuration; existing outputs were preserved."
            )
    else:
        if run_files:
            raise ValueError(
                "Full run output exists without its protocol; refusing recovery by guessing."
            )
        atomic_write_json(path, payload)
        observed = payload
    atomic_write_text(output_dir / "frozen_primary_state_full_protocol.md", _protocol_markdown(observed))
    return observed


def load_frozen_full_protocol(path: Path) -> dict[str, Any]:
    """Load and verify one frozen full protocol."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Frozen full protocol root must be an object.")
    verify_full_protocol(payload)
    return payload


def select_full_inventory(
    protocol: Mapping[str, Any],
    *,
    profiles: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Select only identities already present in the frozen 20-run inventory."""

    verify_full_protocol(protocol)
    selected_profiles = set(profiles or FULL_PROFILES)
    selected_seeds = {int(seed) for seed in (seeds or FULL_SEEDS)}
    unknown_profiles = selected_profiles - set(FULL_PROFILES)
    unknown_seeds = selected_seeds - set(FULL_SEEDS)
    if unknown_profiles or unknown_seeds:
        raise ValueError(
            f"Requested identities are outside the full protocol: "
            f"profiles={sorted(unknown_profiles)}, seeds={sorted(unknown_seeds)}."
        )
    return [
        dict(item)
        for item in protocol["inventory"]
        if item["state_profile"] in selected_profiles and int(item["seed"]) in selected_seeds
    ]
