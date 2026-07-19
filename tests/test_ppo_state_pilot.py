"""Contracts for the frozen common-MLP PPO primary-state pilot."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from src.rl_training.cohort import scenarios_for_split
from src.rl_training.config import PPOConfig
from src.rl_training.environment_factory import make_primary_state_environment
from src.rl_training.pilot_analysis import (
    PAIRED_METRICS,
    paired_patient_differences,
    run_pilot_analysis,
)
from src.rl_training.pilot_experiment import (
    _atomic_model_save,
    _advance_training_sampler_for_resume,
    _assert_resume_frames,
    evaluate_primary_state_scenarios,
    next_evaluation_boundary,
    _recover_pending_rollout,
)
from src.rl_training.pilot_protocol import (
    PILOT_PROFILES,
    PILOT_SEEDS,
    build_pilot_protocol,
    freeze_pilot_protocol,
    load_pilot_source,
    pilot_protocol_hash,
    select_inventory,
    verify_pilot_protocol,
)
from src.rl_training.training import create_primary_state_ppo


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "configs/ppo_primary_state_pilot.json"


@pytest.fixture(scope="module")
def pilot_cohort(ppo_test_cohort):
    return ppo_test_cohort


@pytest.fixture(scope="module")
def pilot_protocol(pilot_cohort):
    return build_pilot_protocol(
        source_path=SOURCE,
        repo_dir=ROOT,
        cohort=pilot_cohort,
        execution_device="cpu",
    )


def test_source_config_is_exact_non_smoke_inventory() -> None:
    source = load_pilot_source(SOURCE)
    assert tuple(source["profiles"]) == PILOT_PROFILES
    assert tuple(source["seeds"]) == PILOT_SEEDS
    assert source["ppo"]["total_timesteps"] == 102_400
    assert source["ppo"]["evaluation_frequency_timesteps"] == 51_200
    assert source["ppo"]["n_steps"] == 2_048
    assert source["policy"]["activation"] == "Tanh"
    assert source["policy"]["optimizer"] == "Adam"
    assert source["cohort"]["test_trajectory_access"] is False


def test_protocol_hash_inventory_and_state_only_contract(pilot_protocol) -> None:
    verify_pilot_protocol(pilot_protocol)
    assert pilot_protocol["protocol_hash"] == pilot_protocol_hash(pilot_protocol)
    assert pilot_protocol["inventory_count"] == 12
    assert len(pilot_protocol["cohort_contract"]["validation_scenario_ids"]) == 15
    signatures = {
        json.dumps(value["architecture_signature"], sort_keys=True)
        for value in pilot_protocol["policy_contracts"].values()
    }
    assert len(signatures) == 1
    assert pilot_protocol["test_seal"]["test_policy_rollout_performed"] is False


def test_protocol_mutation_and_incompatible_refreeze_are_rejected(
    tmp_path: Path, pilot_protocol
) -> None:
    frozen = freeze_pilot_protocol(pilot_protocol, tmp_path)
    assert frozen["protocol_hash"] == pilot_protocol["protocol_hash"]
    corrupted = json.loads(json.dumps(pilot_protocol))
    corrupted["ppo"]["gamma"] = 0.9
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_pilot_protocol(corrupted)
    changed = json.loads(json.dumps(pilot_protocol))
    changed["execution_device"] = "cuda"
    changed["protocol_hash"] = pilot_protocol_hash(changed)
    with pytest.raises(ValueError, match="differs"):
        freeze_pilot_protocol(changed, tmp_path)


def test_inventory_subset_rejects_unknown_identity(pilot_protocol) -> None:
    selected = select_inventory(
        pilot_protocol,
        profiles=["prediction_minimal"],
        seeds=[7, 84],
    )
    assert [(row["state_profile"], row["seed"]) for row in selected] == [
        ("prediction_minimal", 7),
        ("prediction_minimal", 84),
    ]
    with pytest.raises(ValueError, match="outside the pilot"):
        select_inventory(pilot_protocol, profiles=["selected"])


def test_resume_boundaries_preserve_frozen_evaluation_steps() -> None:
    config = PPOConfig(**load_pilot_source(SOURCE)["ppo"])
    assert next_evaluation_boundary(0, config) == 51_200
    assert next_evaluation_boundary(10_240, config) == 51_200
    assert next_evaluation_boundary(51_200, config) == 102_400
    assert next_evaluation_boundary(102_400, config) == 102_400
    with pytest.raises(ValueError, match="rollout boundary"):
        next_evaluation_boundary(10_000, config)


def test_resume_advances_train_patient_rng_past_discarded_partial_episode(
    pilot_cohort, pilot_protocol
) -> None:
    config = PPOConfig(**pilot_protocol["ppo"])
    env = make_primary_state_environment(
        state_profile="prediction_minimal",
        ppo=config,
        seed=7,
        cohort=pilot_cohort,
        split="train",
    )
    assert _advance_training_sampler_for_resume(env, 2_048) == 12
    with pytest.raises(ValueError, match="non-negative"):
        _advance_training_sampler_for_resume(env, -1)
    env.close()


def test_resume_frames_reject_progress_ahead_of_checkpoint() -> None:
    training = pd.DataFrame({"timesteps": [51_200]})
    evaluation = pd.DataFrame({"timesteps": [51_200]})
    _assert_resume_frames(
        model_timesteps=51_200,
        training_progress=training,
        evaluation_progress=evaluation,
    )
    with pytest.raises(ValueError, match="ahead"):
        _assert_resume_frames(
            model_timesteps=49_152,
            training_progress=training,
            evaluation_progress=evaluation,
        )


def test_pending_rollout_journal_recovers_only_checkpointed_update(
    tmp_path: Path,
) -> None:
    pending = tmp_path / "pending_rollout.json"
    training_path = tmp_path / "training.csv"
    action_path = tmp_path / "action.csv"
    payload = {
        "timesteps": 2_048,
        "training_row": {"timesteps": 2_048, "policy_loss": -0.1},
        "action_rows": [{"timesteps": 2_048, "clipping_rate": 0.2}],
    }
    pending.write_text(json.dumps(payload), encoding="utf-8")
    training, actions = _recover_pending_rollout(
        pending_path=pending,
        model_timesteps=2_048,
        training_progress=pd.DataFrame(),
        action_progress=pd.DataFrame(),
        training_path=training_path,
        action_path=action_path,
    )
    assert training["timesteps"].tolist() == [2_048]
    assert actions["timesteps"].tolist() == [2_048]
    assert not pending.exists()

    payload["timesteps"] = 4_096
    pending.write_text(json.dumps(payload), encoding="utf-8")
    training, actions = _recover_pending_rollout(
        pending_path=pending,
        model_timesteps=2_048,
        training_progress=training,
        action_progress=actions,
        training_path=training_path,
        action_path=action_path,
    )
    assert training["timesteps"].tolist() == [2_048]
    assert actions["timesteps"].tolist() == [2_048]


def test_paired_evaluation_uses_same_validation_identity_and_rejects_test(
    tmp_path: Path, pilot_cohort, pilot_protocol
) -> None:
    config = PPOConfig(**pilot_protocol["ppo"])
    scenario = scenarios_for_split(pilot_cohort, "validation", base_seed=100_000)[0]
    observed = []
    for profile in ("original_reconstructed", "prediction_minimal"):
        env = make_primary_state_environment(
            state_profile=profile,
            ppo=config,
            seed=7,
            cohort=pilot_cohort,
            split="train",
        )
        model = create_primary_state_ppo(
            env,
            state_profile=profile,
            config=config,
            seed=7,
            device="cpu",
        )
        checkpoint = tmp_path / f"{profile}.zip"
        _atomic_model_save(model, checkpoint)
        assert checkpoint.is_file()
        frame = evaluate_primary_state_scenarios(
            model,
            state_profile=profile,
            config=config,
            cohort=pilot_cohort,
            scenarios=(scenario,),
            training_seed=7,
        )
        observed.append((frame.iloc[0]["scenario_id"], frame.iloc[0]["patient_id"]))
        assert {
            "integrated_absolute_bis_error",
            "fraction_time_bis_below_30",
            "evaluation_action_clipping_fraction",
            "lower_action_saturation_fraction",
            "upper_action_saturation_fraction",
            "reward_component_target_tracking",
        }.issubset(frame.columns)
        env.close()
    assert observed[0] == observed[1]
    test_scenario = replace(scenario, split="test")
    with pytest.raises(ValueError, match="test is sealed"):
        evaluate_primary_state_scenarios(
            model,
            state_profile="prediction_minimal",
            config=config,
            cohort=pilot_cohort,
            scenarios=(test_scenario,),
            training_seed=7,
        )


def test_paired_result_schema_is_one_to_one() -> None:
    rows = []
    for profile, offset in (("original_reconstructed", 0.0), ("all_supported", -1.0)):
        for patient in ("a", "b"):
            row = {
                "state_profile": profile,
                "training_seed": 7,
                "scenario_id": f"scenario-{patient}",
                "patient_id": patient,
            }
            row.update({metric: 5.0 + offset for metric in PAIRED_METRICS})
            rows.append(row)
    paired = paired_patient_differences(pd.DataFrame(rows))
    assert set(paired["state_profile"]) == {"all_supported"}
    assert set(paired["difference_candidate_minus_original"]) == {-1.0}
    assert len(paired) == 2 * len(set(paired["metric"]))


def test_pending_inventory_analysis_writes_complete_schema(
    tmp_path: Path, pilot_protocol
) -> None:
    analysis_dir = tmp_path / "analysis"
    result = run_pilot_analysis(
        protocol=pilot_protocol,
        output_root=tmp_path / "runs",
        analysis_dir=analysis_dir,
    )
    assert result["completed_runs"] == 0
    assert result["pending_runs"] == 12
    expected = {
        "run_level_summary.csv",
        "evaluation_checkpoint_summary.csv",
        "patient_level_paired_metrics.csv",
        "action_diagnostics.csv",
        "learning_curve.csv",
        "failed_runs_manifest.json",
        "reproducibility_manifest.json",
        "pilot_report.md",
    }
    assert expected.issubset(path.name for path in analysis_dir.iterdir())
    assert len(list((analysis_dir / "figures").glob("*.png"))) == 6
