"""Frozen protocol construction for the common-MLP primary-state PPO pilot."""

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
from .policy_registry import primary_state_policy_contract, validate_state_only_comparison
from .run_status import package_versions, repository_commit


PILOT_PROFILES = (
    "original_reconstructed",
    "all_supported",
    "prediction_minimal",
    "selected_control_core",
)
PILOT_SEEDS = (7, 42, 84)
PILOT_PROTOCOL_FILENAME = "frozen_primary_state_pilot_protocol.json"

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
    "test_seal",
    "interpretation",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_pilot_source(path: Path) -> dict[str, Any]:
    """Load and strictly validate the committed pilot source configuration."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Pilot source configuration is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Pilot source configuration is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Pilot source configuration root must be a JSON object.")
    if payload.get("schema_version") != 1:
        raise ValueError("Pilot source configuration requires schema_version=1.")
    if payload.get("protocol_version") != "ppo_primary_state_pilot_v1":
        raise ValueError("Unexpected primary-state pilot protocol_version.")
    if tuple(payload.get("profiles", ())) != PILOT_PROFILES:
        raise ValueError(f"Pilot profiles must be exactly {PILOT_PROFILES}.")
    if tuple(payload.get("seeds", ())) != PILOT_SEEDS:
        raise ValueError(f"Pilot seeds must be exactly {PILOT_SEEDS}.")
    if payload.get("library", {}).get("stable_baselines3") != "2.9.0":
        raise ValueError("Pilot source must pin stable_baselines3 2.9.0.")
    ppo = PPOConfig(**payload.get("ppo", {}))
    if ppo.profile_name != "ppo_primary_state_pilot_v1":
        raise ValueError("Pilot PPO profile_name is not ppo_primary_state_pilot_v1.")
    if ppo.total_timesteps != 102_400 or ppo.evaluation_frequency_timesteps != 51_200:
        raise ValueError("Pilot budget must remain 102400 with evaluation every 51200 steps.")
    if ppo.total_timesteps % ppo.n_steps or ppo.evaluation_frequency_timesteps % ppo.n_steps:
        raise ValueError("Pilot training and evaluation boundaries must align to PPO rollouts.")
    if payload.get("cohort", {}).get("test_trajectory_access") is not False:
        raise ValueError("Pilot source must prohibit test trajectory access.")
    if payload.get("cohort", {}).get("test_outcome_access") is not False:
        raise ValueError("Pilot source must prohibit test outcome access.")
    if payload.get("evaluation", {}).get("test_split_used_for_selection") is not False:
        raise ValueError("Pilot checkpoint selection must keep the test split sealed.")
    contracts = [
        primary_state_policy_contract(cast(PrimaryStateProfile, profile))
        for profile in PILOT_PROFILES
    ]
    validate_state_only_comparison(contracts)
    expected_policy = {
        "class": "MlpPolicy",
        "feature_extractor": (
            "stable_baselines3.common.torch_layers.FlattenExtractor"
        ),
        "hidden_layers": [64, 64],
        "activation": "Tanh",
        "optimizer": "Adam",
    }
    if payload.get("policy") != expected_policy:
        raise ValueError(f"Pilot common-policy contract changed: {payload.get('policy')}")
    return payload


def source_config_sha256(path: Path) -> str:
    """Hash the exact committed source bytes, including feature order."""

    return _sha256_bytes(path.read_bytes())


def resolve_execution_device(requested: str) -> str:
    """Resolve one backend for the entire inventory without assuming CUDA is faster."""

    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Execution device must be one of auto, cpu, or cuda.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def pilot_protocol_hash(payload: Mapping[str, Any]) -> str:
    """Hash only scientific, cohort, implementation, and backend compatibility fields."""

    selected = {key: payload.get(key) for key in _HASH_KEYS}
    return _sha256_bytes(canonical_json(selected).encode("utf-8"))


def build_pilot_protocol(
    *,
    source_path: Path,
    repo_dir: Path,
    cohort: CohortBundle,
    execution_device: str,
) -> dict[str, Any]:
    """Bind the committed pilot source to one cohort, commit, and execution backend."""

    source = load_pilot_source(source_path)
    required_sb3 = source["library"]["stable_baselines3"]
    if stable_baselines3.__version__ != required_sb3:
        raise ValueError(
            "Observed Stable-Baselines3 differs from the committed pilot source: "
            f"required={required_sb3}, observed={stable_baselines3.__version__}."
        )
    device = resolve_execution_device(execution_device)
    ppo = PPOConfig(**source["ppo"])
    profiles = tuple(cast(PrimaryStateProfile, name) for name in source["profiles"])
    contracts = [primary_state_policy_contract(profile) for profile in profiles]
    validate_state_only_comparison(contracts)
    scenarios = scenarios_for_split(cohort, "validation", base_seed=100_000)
    if len(scenarios) != ppo.evaluation_episode_count:
        raise ValueError(
            "Frozen validation episode count differs from validation cohort size: "
            f"config={ppo.evaluation_episode_count}, cohort={len(scenarios)}."
        )
    bounds = action_bounds_from_profile(ppo.action_bounds_profile)
    inventory = [
        {
            "state_profile": profile,
            "seed": seed,
            "run_id": f"{profile}/seed_{seed}",
        }
        for profile in profiles
        for seed in source["seeds"]
    ]
    split_manifest = cohort.cohort.manifest
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": source["protocol_version"],
        "source_config_path": str(source_path.resolve()),
        "source_config_sha256": source_config_sha256(source_path),
        "implementation_commit": repository_commit(repo_dir),
        "execution_device": device,
        "runtime_at_creation": {
            "packages": package_versions(),
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "library": {
            "stable_baselines3_required": source["library"]["stable_baselines3"],
            "stable_baselines3_observed": stable_baselines3.__version__,
        },
        "profiles": list(profiles),
        "seeds": list(source["seeds"]),
        "inventory": inventory,
        "inventory_count": len(inventory),
        "confirmation_text": "RUN_12_PRIMARY_STATE_PILOT_RUNS",
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
                "train": len(split_manifest.train_patient_ids),
                "validation": len(split_manifest.validation_patient_ids),
                "test": len(split_manifest.test_patient_ids),
            },
            "split_patient_ids": {
                "train": list(split_manifest.train_patient_ids),
                "validation": list(split_manifest.validation_patient_ids),
                "test": list(split_manifest.test_patient_ids),
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
    payload["protocol_hash"] = pilot_protocol_hash(payload)
    verify_pilot_protocol(payload)
    return payload


def verify_pilot_protocol(payload: Mapping[str, Any]) -> None:
    """Reject any mutation or scientific mismatch in a frozen pilot protocol."""

    if payload.get("protocol_hash") != pilot_protocol_hash(payload):
        raise ValueError("Primary-state pilot protocol hash mismatch.")
    if tuple(payload.get("profiles", ())) != PILOT_PROFILES:
        raise ValueError("Frozen pilot profile inventory changed.")
    if tuple(payload.get("seeds", ())) != PILOT_SEEDS:
        raise ValueError("Frozen pilot seed inventory changed.")
    inventory = payload.get("inventory", ())
    if payload.get("inventory_count") != 12 or len(inventory) != 12:
        raise ValueError("Frozen pilot requires exactly four profiles by three seeds.")
    expected = {(profile, seed) for profile in PILOT_PROFILES for seed in PILOT_SEEDS}
    observed = {
        (item.get("state_profile"), int(item.get("seed"))) for item in inventory
    }
    if observed != expected:
        raise ValueError("Frozen pilot inventory identities changed.")
    if payload.get("library", {}).get("stable_baselines3_required") != "2.9.0":
        raise ValueError("Frozen pilot no longer pins stable_baselines3 2.9.0.")
    if (
        payload.get("library", {}).get("stable_baselines3_observed")
        != payload.get("library", {}).get("stable_baselines3_required")
    ):
        raise ValueError("Frozen pilot was created under an incompatible SB3 version.")
    if payload.get("execution_device") not in {"cpu", "cuda"}:
        raise ValueError("Frozen pilot execution_device is invalid.")
    ppo = PPOConfig(**cast(Mapping[str, Any], payload.get("ppo", {})))
    if ppo.total_timesteps != 102_400 or ppo.evaluation_frequency_timesteps != 51_200:
        raise ValueError("Frozen pilot training or evaluation budget changed.")
    seals = cast(Mapping[str, Any], payload.get("test_seal", {}))
    forbidden = (
        "test_trajectory_loaded",
        "test_outcomes_evaluated",
        "test_policy_rollout_performed",
        "test_checkpoint_selection",
    )
    if any(seals.get(key) is not False for key in forbidden):
        raise ValueError("Frozen pilot test-cohort seal is not intact.")
    contract_values = list(cast(Mapping[str, Any], payload["policy_contracts"]).values())
    signatures = {canonical_json(item["architecture_signature"]) for item in contract_values}
    if len(signatures) != 1:
        raise ValueError("Frozen pilot changes policy architecture between state profiles.")


def _protocol_markdown(payload: Mapping[str, Any]) -> str:
    return f"""# Frozen Primary-State PPO Pilot

- Protocol: `{payload['protocol_version']}`
- Hash: `{payload['protocol_hash']}`
- Implementation: `{payload['implementation_commit']}`
- Device: `{payload['execution_device']}`
- Profiles: `{', '.join(payload['profiles'])}`
- Seeds: `{payload['seeds']}`
- Runs: `{payload['inventory_count']}`
- Timesteps per run: `{payload['ppo']['total_timesteps']}`
- Validation interval: `{payload['ppo']['evaluation_frequency_timesteps']}`
- Cohort fingerprint: `{payload['cohort_contract']['fingerprint']}`
- Test trajectories/outcomes/policy rollouts: sealed

This is an exploratory pilot, not a final state winner analysis or a clinical result.
"""


def freeze_pilot_protocol(
    payload: dict[str, Any],
    output_dir: Path,
    *,
    run_output_root: Path | None = None,
) -> dict[str, Any]:
    """Create once, reuse exactly, and refuse incompatible pilot output roots."""

    verify_pilot_protocol(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / PILOT_PROTOCOL_FILENAME
    run_files = (
        sorted(item for item in run_output_root.rglob("*") if item.is_file())
        if run_output_root is not None and run_output_root.exists()
        else []
    )
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        verify_pilot_protocol(observed)
        if observed["protocol_hash"] != payload["protocol_hash"]:
            raise ValueError(
                "Existing primary-state pilot protocol differs from the requested cohort, "
                "commit, device, or scientific configuration. Existing outputs were preserved."
            )
    else:
        if run_files:
            raise ValueError(
                "Pilot run output exists without its frozen protocol; refusing recovery by "
                f"guessing. First files: {[str(item) for item in run_files[:5]]}."
            )
        atomic_write_json(path, payload)
        observed = payload
    atomic_write_text(output_dir / "frozen_primary_state_pilot_protocol.md", _protocol_markdown(observed))
    return observed


def load_frozen_pilot_protocol(path: Path) -> dict[str, Any]:
    """Load one frozen protocol and validate its hash before use."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Frozen pilot protocol root must be an object.")
    verify_pilot_protocol(payload)
    return payload


def select_inventory(
    protocol: Mapping[str, Any],
    *,
    profiles: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Select an ordered subset without creating identities outside the frozen inventory."""

    verify_pilot_protocol(protocol)
    selected_profiles = set(profiles or PILOT_PROFILES)
    selected_seeds = {int(seed) for seed in (seeds or PILOT_SEEDS)}
    unknown_profiles = selected_profiles - set(PILOT_PROFILES)
    unknown_seeds = selected_seeds - set(PILOT_SEEDS)
    if unknown_profiles or unknown_seeds:
        raise ValueError(
            f"Requested identities are outside the pilot: profiles={sorted(unknown_profiles)}, "
            f"seeds={sorted(unknown_seeds)}."
        )
    return [
        dict(item)
        for item in protocol["inventory"]
        if item["state_profile"] in selected_profiles and int(item["seed"]) in selected_seeds
    ]
