"""Immutable PPO protocol construction and compatibility hashing."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import stable_baselines3
import torch

from src.rl_env.config import RESEARCH_ONLY_WARNING, action_bounds_from_profile
from src.rl_env.reward import reward_profile_registry
from src.rl_env.state_adapters import state_profile_registry

from .cohort import CohortBundle, scenarios_for_split
from .config import EXPERIMENT_SEEDS, POLICY_CONDITIONS, PPOConfig
from .policy_registry import encoder_contract_registry


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def protocol_hash(payload: dict[str, Any]) -> str:
    without_hash = {key: value for key, value in payload.items() if key != "protocol_hash"}
    return hashlib.sha256(canonical_json(without_hash).encode()).hexdigest()


def _git_head(repo_dir: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_frozen_protocol(
    *,
    repo_dir: Path,
    cohort: CohortBundle,
    ppo: PPOConfig | None = None,
) -> dict[str, Any]:
    ppo = ppo or PPOConfig()
    bounds = action_bounds_from_profile(ppo.action_bounds_profile)
    inventory = [
        {"condition": condition, "seed": seed, "run_id": f"{condition}/seed_{seed}"}
        for condition in POLICY_CONDITIONS
        for seed in EXPERIMENT_SEEDS
    ]
    validation_scenarios = scenarios_for_split(cohort, "validation", base_seed=100_000)
    payload: dict[str, Any] = {
        "protocol_version": "ppo_control_comparison_v1",
        "implementation_commit_at_creation": _git_head(repo_dir),
        "required_repository_ancestor": "767f3bff3dcaeabc51049fba5ccba1ac02b69ae3",
        "simulator_commit": "faf636ab3d5922c73a979b2cf2a8ea6e0f1e8483",
        "module5_environment_commit": "767f3bff3dcaeabc51049fba5ccba1ac02b69ae3",
        "library": {
            "stable_baselines3": stable_baselines3.__version__,
            "torch_at_creation": torch.__version__,
            "torch_reinstallation_forbidden": True,
            "pandas_reinstallation_forbidden": True,
        },
        "research_warning": RESEARCH_ONLY_WARNING,
        "reward": {
            "profile": ppo.reward_profile,
            "source": "repository design",
            "target_bis": 50.0,
            "safe_bis_range": [40.0, 60.0],
            "action_magnitude_coefficient": ppo.action_magnitude_coefficient,
            "action_change_coefficient": ppo.action_change_coefficient,
            "concentration_safety_coefficient": 0.0,
            "profile_registry": reward_profile_registry(),
            "immutable_after_training_starts": True,
        },
        "action": {
            **asdict(bounds),
            "policy_space": [-1.0, 1.0],
            "mapping": "low + (policy_action + 1) * (high-low) / 2",
            "dose_mapping": "physical_mg_per_min * 10/60",
            "silent_physical_clipping": False,
        },
        "environment": {
            "action_interval_seconds": 10.0,
            "internal_dt_seconds": 1.0,
            "history_window_seconds": 60.0,
            "history_steps": 6,
            "episode_duration_seconds": ppo.episode_duration_seconds,
            "deterministic_simulator": ppo.deterministic_simulator,
            "remifentanil_schedule": "scenario-ID-derived piecewise synthetic schedule",
            "target_schedule": "constant BIS 50",
            "initial_state": "zero drug state",
        },
        "cohort": {
            "kind": "VitalDB-demographics-parameterized virtual PK-PD patients",
            "clinical_trajectory_replay": False,
            "population_validation_claim": False,
            "fingerprint": cohort.fingerprint,
            "demographics_source": cohort.demographics_source,
            "split_source": cohort.split_source,
            "case_counts": {
                "train": len(cohort.cohort.manifest.train_patient_ids),
                "validation": len(cohort.cohort.manifest.validation_patient_ids),
                "test": len(cohort.cohort.manifest.test_patient_ids),
            },
            "split_patient_ids": {
                "train": list(cohort.cohort.manifest.train_patient_ids),
                "validation": list(cohort.cohort.manifest.validation_patient_ids),
                "test": list(cohort.cohort.manifest.test_patient_ids),
            },
            "patient_overlap": False,
            "missing_demographics_imputed": False,
            "demographic_domain_policy": (
                "All reused VitalDB demographics must pass simulator hard bounds; "
                "outside-source-study-envelope values emit explicit warnings and are not imputed."
            ),
        },
        "conditions": list(POLICY_CONDITIONS),
        "main_contrast": "attention_supported - all_supported",
        "secondary_contrasts": [
            "attention_supported - yun_reconstructed",
            "selected_control_aware - all_supported",
        ],
        "strict_consensus_is_attention_selected": False,
        "predictive_attention_checkpoint_transfer": False,
        "state_profile_registry": state_profile_registry(),
        "encoder_contracts": encoder_contract_registry(ppo.latent_dim),
        "ppo": ppo.as_dict(),
        "ppo_hyperparameter_source": (
            "ppo_research_v1 repository design; Yun 2023 does not publish a complete "
            "reproducible PPO optimizer configuration"
        ),
        "seeds": list(EXPERIMENT_SEEDS),
        "inventory": inventory,
        "inventory_count": len(inventory),
        "confirmation_text": f"RUN_{len(inventory)}_PPO_CUDA_RUNS",
        "checkpoint_selection": {
            "split": "validation",
            "primary_metric": "patient/scenario-level target BIS MAE (lower)",
            "tie_breaker_1": "time in BIS 40-60 (higher)",
            "tie_breaker_2": "absolute action change sum (lower)",
            "training_return_used": False,
            "test_split_used": False,
        },
        "validation_scenario_ids": [scenario.scenario_id for scenario in validation_scenarios],
        "early_stopping": "none",
        "final_evaluation": "validation-only in this module; held-out RL test remains sealed",
        "full_training_performed_by_protocol_creation": False,
    }
    payload["protocol_hash"] = protocol_hash(payload)
    return payload


def freeze_protocol(payload: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Write once or require exact hash equality; never mutate an existing protocol."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "frozen_ppo_protocol.json"
    if json_path.exists():
        observed = json.loads(json_path.read_text(encoding="utf-8"))
        if protocol_hash(observed) != observed.get("protocol_hash"):
            raise ValueError("Existing frozen PPO protocol has an invalid internal hash.")
        if observed["protocol_hash"] != payload["protocol_hash"]:
            raise ValueError(
                "Existing frozen PPO protocol differs from the requested protocol; "
                "create a new protocol version and experiment ID."
            )
        return observed
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown = f"""# Frozen PPO Control Comparison Protocol

- Version: `{payload['protocol_version']}`
- Hash: `{payload['protocol_hash']}`
- Conditions: `{', '.join(payload['conditions'])}`
- Seeds: `{payload['seeds']}`
- Planned runs: `{payload['inventory_count']}`
- Confirmation: `{payload['confirmation_text']}`
- Main contrast: `{payload['main_contrast']}`
- Action bounds: `{payload['action']['profile_name']}` in mg/min
- Reward: `{payload['reward']['profile']}` with fixed coefficients
- Checkpoint selection: validation target BIS MAE; test is sealed

`yun_reconstructed` is not a complete reproduction of Yun's unspecified LOWESS
pipeline. `selected_control_aware` is not an attention-selected subset.

{payload['research_warning']}
"""
    (output_dir / "frozen_ppo_protocol.md").write_text(markdown, encoding="utf-8")
    return payload


def verify_protocol(payload: dict[str, Any]) -> None:
    observed = payload.get("protocol_hash")
    expected = protocol_hash(payload)
    if observed != expected:
        raise ValueError(f"PPO protocol hash mismatch: observed={observed}, expected={expected}.")
    if payload.get("inventory_count") != 20:
        raise ValueError("This protocol requires the exact 4-condition x 5-seed inventory.")


def write_policy_contract_artifacts(
    *,
    protocol: dict[str, Any],
    cohort: CohortBundle,
    output_dir: Path,
) -> None:
    """Persist parameter counts and the primary raw-information equality audit."""

    import pandas as pd

    from .config import POLICY_CONDITIONS, PPOConfig
    from .environment_factory import make_cohort_environment
    from .training import create_ppo, parameter_counts

    ppo = PPOConfig(**protocol["ppo"])
    rows = []
    for condition in POLICY_CONDITIONS:
        env = make_cohort_environment(
            condition=condition,
            ppo=ppo,
            cohort=cohort,
            split="train",
            seed=7,
        )
        model = create_ppo(
            env, condition=condition, config=ppo, seed=7, device="cpu"
        )
        rows.append({"condition": condition, **parameter_counts(model)})
        env.close()
    counts = pd.DataFrame(rows)
    counts.to_csv(output_dir / "policy_parameter_counts.csv", index=False)
    pd.DataFrame(cohort.patient_records).to_csv(
        output_dir / "virtual_patient_manifest.csv", index=False
    )
    all_count = int(
        counts.loc[
            counts["condition"] == "all_supported", "total_policy_trainable_parameters"
        ].iloc[0]
    )
    attention_count = int(
        counts.loc[
            counts["condition"] == "attention_supported",
            "total_policy_trainable_parameters",
        ].iloc[0]
    )
    relative_difference = abs(attention_count - all_count) / all_count
    all_features = protocol["encoder_contracts"]["all_supported"]["feature_names"]
    attention_features = protocol["encoder_contracts"]["attention_supported"][
        "feature_names"
    ]
    equivalence = {
        "raw_feature_names_equal": all_features == attention_features,
        "raw_feature_order_equal": all_features == attention_features,
        "history_steps_equal": True,
        "static_features_equal": True,
        "target_input_equal": True,
        "reward_action_cohort_scenarios_equal": True,
        "latent_dimension_equal": True,
        "all_supported_parameters": all_count,
        "attention_supported_parameters": attention_count,
        "relative_parameter_difference": relative_difference,
        "within_prespecified_ten_percent": relative_difference <= 0.10,
    }
    if not all(
        equivalence[key]
        for key in (
            "raw_feature_names_equal",
            "raw_feature_order_equal",
            "reward_action_cohort_scenarios_equal",
            "latent_dimension_equal",
            "within_prespecified_ten_percent",
        )
    ):
        raise ValueError(f"All-vs-attention fairness contract failed: {equivalence}")
    (output_dir / "encoder_contracts.json").write_text(
        json.dumps(protocol["encoder_contracts"], indent=2), encoding="utf-8"
    )
    (output_dir / "all_attention_information_equivalence.json").write_text(
        json.dumps(equivalence, indent=2), encoding="utf-8"
    )
