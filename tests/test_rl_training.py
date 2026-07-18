"""Module 5 audit and fair PPO workflow contract tests."""

from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from stable_baselines3.common.utils import obs_as_tensor

from src.rl_env import (
    EnvironmentConfig,
    PropofolControlEnv,
    SYNTHETIC_NONCLINICAL_ACTION_BOUNDS,
    YUN_REPORTED_ACTION_BOUNDS,
)
from src.rl_env.state_adapters import (
    ALL_SUPPORTED_FEATURES,
    STATE_PROFILES,
    get_state_profile,
)
from src.rl_training.action_wrapper import (
    NormalizedPropofolActionWrapper,
    physical_to_policy,
    policy_to_physical,
)
from src.rl_training.analysis import (
    audit_run_inventory,
    hierarchical_bootstrap,
    paired_contrast,
)
from src.rl_training.attention_logging import (
    save_attention_artifact,
    verify_attention_checkpoint,
)
from src.rl_training.cohort import (
    load_vitaldb_virtual_cohort,
    remifentanil_schedule_for_scenario,
    scenarios_for_split,
)
from src.rl_training.config import (
    EXPERIMENT_SEEDS,
    POLICY_CONDITIONS,
    PRIMARY_STATE_PROFILES,
    PPOConfig,
    smoke_ppo_config,
)
from src.rl_training.callbacks import PPOProgressCallback
from src.rl_training.environment_factory import make_cohort_environment
from src.rl_training.evaluation import checkpoint_score, evaluate_scenarios
from src.rl_training.feature_extractors import (
    FactorizedAttentionControlExtractor,
    GRUControlExtractor,
)
from src.rl_training.manifests import (
    build_frozen_protocol,
    freeze_protocol,
    protocol_hash,
    verify_protocol,
    write_policy_contract_artifacts,
)
from src.rl_training.experiment_protocol import run_experiment
from src.rl_training.module5_audit import run_module5_audit
from src.rl_training.policy_registry import policy_contract
from src.rl_training.run_status import (
    begin_run_status,
    complete_run_status,
    fail_run_status,
    update_running_config,
)
from src.rl_training.smoke import (
    make_synthetic_smoke_env,
    run_condition_smoke,
    run_primary_state_smoke,
)
from src.rl_training.training import create_ppo, parameter_counts
from scripts.run_ppo_experiment import main as run_ppo_main, validate_confirmation


ROOT = Path(__file__).parents[1]


def test_primary_state_options_cannot_fall_through_to_legacy_full_training() -> None:
    with pytest.raises(ValueError, match="smoke-only"):
        run_ppo_main(["--state-profile", "selected"])


def test_primary_candidate_profiles_are_cli_selectable_without_resolving_selected() -> None:
    assert PRIMARY_STATE_PROFILES == (
        "original_reconstructed",
        "all_supported",
        "prediction_minimal",
        "selected_control_core",
        "selected",
    )
    for profile in ("prediction_minimal", "selected_control_core"):
        assert get_state_profile(profile).name == profile
    with pytest.raises(ValueError, match="requires selected_state_manifest"):
        get_state_profile("selected")


def test_run_status_running_complete_and_failed_are_unambiguous(tmp_path: Path) -> None:
    run_dir = tmp_path / "complete"
    begin_run_status(
        run_dir,
        resolved_config={
            "seed": 42,
            "state_profile": "original_reconstructed",
            "ordered_feature_names": ["bis"],
        },
        repo_dir=ROOT,
    )
    assert json.loads((run_dir / "run_status.json").read_text())["status"] == "running"
    update_running_config(run_dir, updates={"total_trainable_parameters": 123})
    running = json.loads((run_dir / "run_status.json").read_text())
    assert running["resolved_config"]["total_trainable_parameters"] == 123
    checkpoint = run_dir / "model.zip"
    evaluation = run_dir / "evaluation.csv"
    checkpoint.write_bytes(b"model")
    evaluation.write_text("metric\n1\n", encoding="utf-8")
    complete_run_status(
        run_dir, final_checkpoint=checkpoint, evaluation_artifacts=[evaluation]
    )
    assert json.loads((run_dir / "run_status.json").read_text())["status"] == "complete"

    failed_dir = tmp_path / "failed"
    begin_run_status(
        failed_dir,
        resolved_config={
            "seed": 7,
            "state_profile": "all_supported",
            "ordered_feature_names": ["bis"],
        },
        repo_dir=ROOT,
    )
    try:
        raise RuntimeError("deliberate failure")
    except RuntimeError as exc:
        fail_run_status(failed_dir, exc)
    failed = json.loads((failed_dir / "run_status.json").read_text())
    assert failed["status"] == "failed"
    assert failed["exception_type"] == "RuntimeError"
    assert "deliberate failure" in failed["traceback"]


def test_clipping_callback_separates_lower_upper_and_boundary_actions() -> None:
    callback = PPOProgressCallback(bounds=SYNTHETIC_NONCLINICAL_ACTION_BOUNDS)
    callback.locals = {
        "actions": np.asarray([[-2.0], [0.0], [3.0]]),
        "clipped_actions": np.asarray([[-1.0], [0.0], [1.0]]),
        "dones": np.asarray([False, False, True]),
    }
    assert callback._on_step() is True
    diagnostics = callback.diagnostics()
    assert diagnostics["normalized_clipping_count"] == 2
    assert diagnostics["lower_bound_clipping_count"] == 1
    assert diagnostics["upper_bound_clipping_count"] == 1
    assert diagnostics["physical_action_bounds_mg_per_min"] == [0.0, 12.0]


def test_primary_common_mlp_smoke_writes_reload_and_evaluation_artifacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "primary_smoke"
    summary = run_primary_state_smoke(
        state_profile="original_reconstructed",
        seed=42,
        total_timesteps=100,
        output_dir=output,
        repo_dir=ROOT,
        device="cpu",
    )
    assert summary["training_timesteps"] == 100
    assert summary["gymnasium_check_env_passed"] is True
    assert summary["model_reload_passed"] is True
    status = json.loads((output / "run_status.json").read_text())
    assert status["status"] == "complete"
    assert status["resolved_config"]["total_trainable_parameters"] == summary[
        "parameter_counts"
    ]["total_policy_trainable_parameters"]
    assert (output / "final_model.zip").is_file()
    assert (output / "deterministic_evaluation.csv").is_file()


@pytest.fixture(scope="module")
def cohort_bundle():
    with pytest.warns(UserWarning):
        return load_vitaldb_virtual_cohort(
            ROOT / "data/modeling/full", project_data_root=ROOT / "data"
        )


def _model(condition: str, seed: int = 1):
    env = make_synthetic_smoke_env(condition)  # type: ignore[arg-type]
    model = create_ppo(
        env,
        condition=condition,  # type: ignore[arg-type]
        config=smoke_ppo_config(),
        seed=seed,
        device="cpu",
    )
    return env, model


def _tensor_observation(env, model, *, all_valid: bool = False):
    observation, _ = env.reset(seed=1)
    if all_valid:
        observation["history_mask"][:] = 1
        observation["history"] = np.random.default_rng(1).normal(
            size=observation["history"].shape
        ).astype(np.float32)
    return obs_as_tensor({key: value[None] for key, value in observation.items()}, model.device)


def test_module5_max_action_delivers_exactly_27_7_mg_without_double_conversion() -> None:
    config = EnvironmentConfig(
        episode_duration_seconds=10,
        action_bounds=YUN_REPORTED_ACTION_BOUNDS,
    )
    env = PropofolControlEnv(config)
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.asarray([166.2], dtype=np.float32))
    assert info["applied_dose_mg_per_10s"] == pytest.approx(27.7)
    assert info["propofol_cumulative_dose_mg"] == pytest.approx(27.7)
    assert info["action_rate_unit"] == "mg/min"
    assert "10-second" in info["applied_dose_unit"]


def test_module5_reward_profiles_and_missing_alpha_contract() -> None:
    assert EnvironmentConfig().reward_profile == "transparent_tracking_v1"
    with pytest.raises(ValueError, match="explicit positive alpha"):
        EnvironmentConfig(reward_profile="paper_yun2023_parameterized")


def test_yun_reconstructed_is_official_nonexact_alias() -> None:
    with pytest.warns(DeprecationWarning):
        alias = get_state_profile("yun_reconstructed")
    original = get_state_profile("original_reconstructed")
    assert alias.dynamic_feature_names == original.dynamic_feature_names
    assert alias.name == "original_reconstructed"
    assert "unpublished code" in alias.purpose


def test_module5_independent_audit_outputs_and_alignment(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    manifest = run_module5_audit(output, ROOT)
    assert manifest["status"] == "passed"
    action = pd.read_csv(output / "action_unit_audit.csv")
    assert action.iloc[-1]["simulator_cumulative_dose_mg"] == pytest.approx(27.7)
    assert not action["double_conversion_detected"].any()
    history = pd.read_csv(output / "history_alignment_audit.csv")
    assert history.iloc[0]["history_mask"] == "0|0|0|0|0|1"
    assert history["reward_alignment_error"].dropna().max() == 0.0


@pytest.mark.parametrize(
    ("normalized", "expected"), [(-1.0, 0.0), (0.0, 83.1), (1.0, 166.2)]
)
def test_normalized_action_mapping(normalized: float, expected: float) -> None:
    transformed = policy_to_physical(normalized, YUN_REPORTED_ACTION_BOUNDS)
    assert transformed.physical_action_mg_per_min == pytest.approx(expected)
    assert physical_to_policy(expected, YUN_REPORTED_ACTION_BOUNDS) == pytest.approx(normalized)


def test_normalized_upper_action_maps_to_27_7_mg_per_step() -> None:
    transformed = policy_to_physical(1.0, YUN_REPORTED_ACTION_BOUNDS)
    assert transformed.applied_dose_mg_per_10s == pytest.approx(27.7)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.01, 1.01])
def test_normalized_action_rejects_nonfinite_and_out_of_bounds(bad: float) -> None:
    with pytest.raises(ValueError):
        policy_to_physical(bad, SYNTHETIC_NONCLINICAL_ACTION_BOUNDS)


def test_action_wrapper_reports_every_transform_without_physical_clipping() -> None:
    base = PropofolControlEnv(
        EnvironmentConfig(
            episode_duration_seconds=10,
            action_bounds=SYNTHETIC_NONCLINICAL_ACTION_BOUNDS,
        )
    )
    env = NormalizedPropofolActionWrapper(base, SYNTHETIC_NONCLINICAL_ACTION_BOUNDS)
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.asarray([0.0], dtype=np.float32))
    assert info["policy_raw_action"] == 0.0
    assert info["physical_action_mg_per_min"] == 6.0
    assert info["normalized_clipping_applied"] is False


def test_vitaldb_virtual_cohort_reuses_disjoint_splits_without_imputation(cohort_bundle) -> None:
    manifest = cohort_bundle.cohort.manifest
    assert (len(manifest.train_patient_ids), len(manifest.validation_patient_ids), len(manifest.test_patient_ids)) == (68, 15, 15)
    assert len(cohort_bundle.patient_records) == 98
    assert cohort_bundle.demographics_source.endswith("vitaldb_clean_100cases.csv")


def test_scenario_ids_and_remifentanil_are_deterministic_and_paired(cohort_bundle) -> None:
    left = scenarios_for_split(cohort_bundle, "validation", base_seed=100_000)
    right = scenarios_for_split(cohort_bundle, "validation", base_seed=100_000)
    assert left == right
    schedule_left = remifentanil_schedule_for_scenario(left[0], 120.0)
    schedule_right = remifentanil_schedule_for_scenario(right[0], 120.0)
    assert [schedule_left.rate_at(time) for time in (0, 40, 80)] == [
        schedule_right.rate_at(time) for time in (0, 40, 80)
    ]


def test_test_cohort_is_sealed_from_environment_factory(cohort_bundle) -> None:
    with pytest.raises(ValueError, match="Test cohort access is sealed"):
        make_cohort_environment(
            condition="all_supported",
            ppo=replace(smoke_ppo_config(), episode_duration_seconds=120),
            cohort=cohort_bundle,
            split="test",
            seed=1,
        )


def test_all_and_attention_raw_information_and_shapes_are_identical() -> None:
    all_contract = policy_contract("all_supported")
    attention_contract = policy_contract("attention_supported")
    assert all_contract.feature_names == attention_contract.feature_names == ALL_SUPPORTED_FEATURES
    assert all_contract.latent_dim == attention_contract.latent_dim == 64
    all_env = make_synthetic_smoke_env("all_supported")
    attention_env = make_synthetic_smoke_env("attention_supported")
    left, _ = all_env.reset(seed=1)
    right, _ = attention_env.reset(seed=1)
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])
    assert left["history"].shape == (6, len(ALL_SUPPORTED_FEATURES))
    assert left["static"].shape == (4,)
    assert left["target_bis"].shape == (1,)


def test_attention_padding_zero_normalization_finite_latent_and_gradient() -> None:
    env, model = _model("attention_supported")
    extractor = model.policy.features_extractor
    assert isinstance(extractor, FactorizedAttentionControlExtractor)
    observations = _tensor_observation(env, model, all_valid=False)
    output = extractor.forward_with_attention(observations)
    mask = observations["history_mask"].bool()
    assert torch.count_nonzero(output.feature_attention[~mask]) == 0
    assert torch.count_nonzero(output.temporal_attention[~mask]) == 0
    assert torch.allclose(
        output.feature_attention.sum(dim=2)[mask], torch.ones_like(output.feature_attention.sum(dim=2)[mask])
    )
    assert torch.allclose(output.temporal_attention.sum(dim=1), torch.ones(1))
    assert torch.isfinite(output.latent).all() and output.latent.shape == (1, 64)
    observations = _tensor_observation(env, model, all_valid=True)
    output = extractor.forward_with_attention(observations)
    output.latent.sum().backward()
    assert extractor.feature_scorer[0].weight.grad.abs().sum() > 0
    assert extractor.temporal_scorer[0].weight.grad.abs().sum() > 0
    env.close()


def test_gru_and_attention_latent_dimensions_and_parameter_counts_are_fair() -> None:
    all_env, all_model = _model("all_supported")
    attention_env, attention_model = _model("attention_supported")
    assert isinstance(all_model.policy.features_extractor, GRUControlExtractor)
    all_count = parameter_counts(all_model)["total_policy_trainable_parameters"]
    attention_count = parameter_counts(attention_model)["total_policy_trainable_parameters"]
    assert abs(attention_count - all_count) / all_count <= 0.10
    assert all_model.policy.features_extractor.features_dim == 64
    assert attention_model.policy.features_extractor.features_dim == 64
    all_env.close()
    attention_env.close()


@pytest.mark.parametrize("condition", POLICY_CONDITIONS)
def test_existing_encoder_conditions_are_labeled_legacy_secondary(condition: str) -> None:
    contract = policy_contract(condition)  # type: ignore[arg-type]
    assert contract.main_comparison_role == "legacy_secondary_architecture"


def test_actor_critic_shapes_and_deterministic_forward_reproducibility() -> None:
    env, model = _model("attention_supported")
    observations = _tensor_observation(env, model)
    model.policy.eval()
    first = model.policy(observations, deterministic=True)
    second = model.policy(observations, deterministic=True)
    assert first[0].shape == (1, 1)
    assert first[1].shape == (1, 1)
    assert first[2].shape == (1,)
    for left, right in zip(first, second):
        assert torch.equal(left, right)
    env.close()


def test_no_predictive_attention_checkpoint_transfer_api() -> None:
    assert FactorizedAttentionControlExtractor.predictive_checkpoint_transfer_supported is False
    assert "checkpoint" not in FactorizedAttentionControlExtractor.__init__.__annotations__


def test_tiny_ppo_smoke_updates_finite_saves_loads_and_resumes(tmp_path: Path) -> None:
    summary = run_condition_smoke(
        condition="attention_supported",
        seed=3,
        total_timesteps=128,
        output_dir=tmp_path / "smoke",
    )
    assert summary["finite_loss"] is True
    assert summary["parameter_update_detected"] is True
    assert summary["checkpoint_save_load_action_equal"] is True
    assert summary["resume_advanced_timesteps"] == 64
    assert summary["test_cohort_accessed"] is False
    assert (tmp_path / "smoke/smoke_attention.npz").exists()


def test_same_seed_policy_initialization_reproducible_and_different_seed_varies() -> None:
    env1, model1 = _model("all_supported", seed=7)
    env2, model2 = _model("all_supported", seed=7)
    env3, model3 = _model("all_supported", seed=8)
    observation, _ = env1.reset(seed=11)
    action1 = model1.predict(observation, deterministic=True)[0]
    action2 = model2.predict(observation, deterministic=True)[0]
    action3 = model3.predict(observation, deterministic=True)[0]
    np.testing.assert_array_equal(action1, action2)
    assert not np.array_equal(action1, action3)
    env1.close(); env2.close(); env3.close()


def test_validation_checkpoint_selection_uses_prespecified_metrics_only() -> None:
    frame = pd.DataFrame(
        {
            "cohort_split": ["validation", "validation"],
            "bis_target_mae": [5.0, 7.0],
            "fraction_time_in_bis_40_60": [0.8, 0.6],
            "absolute_action_change_sum": [2.0, 3.0],
        }
    )
    assert checkpoint_score(frame) == pytest.approx((6.0, -0.7, 2.5))
    frame["cohort_split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        checkpoint_score(frame)


def test_state_condition_does_not_change_underlying_dynamics_reward_or_scenario(cohort_bundle) -> None:
    ppo = replace(smoke_ppo_config(), episode_duration_seconds=120.0)
    values = []
    for condition in POLICY_CONDITIONS:
        env = make_cohort_environment(
            condition=condition, ppo=ppo, cohort=cohort_bundle, split="validation", seed=5, cycle=True
        )
        env.reset(seed=5)
        _, reward, _, _, info = env.step(np.asarray([0.0], dtype=np.float32))
        values.append((info["scenario_id"], info["bis"], reward, info["remifentanil_ce_micrograms_per_l"]))
        env.close()
    assert values.count(values[0]) == len(values)


def test_attention_artifact_contract_and_checkpoint_matching(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"checkpoint")
    mask = np.asarray([[False, False, False, False, False, True]])
    feature = np.zeros((1, 6, 2), dtype=np.float32)
    feature[0, 5] = [0.25, 0.75]
    temporal = np.asarray([[0, 0, 0, 0, 0, 1]], dtype=np.float32)
    artifact = tmp_path / "attention.npz"
    metadata = save_attention_artifact(
        artifact,
        feature_attention=feature,
        temporal_attention=temporal,
        history_mask=mask,
        feature_names=("a", "b"),
        scenario_ids=np.asarray(["s1"]),
        bis=np.asarray([50.0]),
        checkpoint_path=checkpoint,
    )
    assert metadata["feature_names"] == ["a", "b"]
    assert metadata["lag_seconds"] == [-50, -40, -30, -20, -10, 0]
    verify_attention_checkpoint(artifact, checkpoint)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="mismatch"):
        verify_attention_checkpoint(artifact, checkpoint)


def test_frozen_protocol_exact_inventory_hash_and_contract_outputs(tmp_path: Path, cohort_bundle) -> None:
    protocol = build_frozen_protocol(repo_dir=ROOT, cohort=cohort_bundle)
    assert protocol["inventory_count"] == 20
    assert protocol["confirmation_text"] == "RUN_20_PPO_CUDA_RUNS"
    assert protocol["protocol_hash"] == protocol_hash(protocol)
    assert protocol["checkpoint_selection"]["test_split_used"] is False
    frozen = freeze_protocol(protocol, tmp_path)
    write_policy_contract_artifacts(protocol=frozen, cohort=cohort_bundle, output_dir=tmp_path)
    equivalence = json.loads((tmp_path / "all_attention_information_equivalence.json").read_text())
    assert equivalence["raw_feature_order_equal"] is True
    assert equivalence["within_prespecified_ten_percent"] is True
    counts = pd.read_csv(tmp_path / "policy_parameter_counts.csv")
    assert len(counts) == 4


def test_protocol_hash_mismatch_and_refreeze_mismatch_are_rejected(tmp_path: Path, cohort_bundle) -> None:
    protocol = build_frozen_protocol(repo_dir=ROOT, cohort=cohort_bundle)
    freeze_protocol(protocol, tmp_path)
    corrupted = dict(protocol)
    corrupted["seeds"] = [1]
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_protocol(corrupted)
    changed = dict(protocol)
    changed["protocol_version"] = "changed"
    changed["protocol_hash"] = protocol_hash(changed)
    with pytest.raises(ValueError, match="differs"):
        freeze_protocol(changed, tmp_path)


def test_confirmation_lock_requires_exact_generated_text(cohort_bundle) -> None:
    protocol = build_frozen_protocol(repo_dir=ROOT, cohort=cohort_bundle)
    with pytest.raises(ValueError, match="RUN_20_PPO_CUDA_RUNS"):
        validate_confirmation(protocol, "wrong")
    validate_confirmation(protocol, "RUN_20_PPO_CUDA_RUNS")


def test_completed_run_skip_and_partial_config_mismatch(tmp_path: Path, cohort_bundle) -> None:
    protocol = build_frozen_protocol(
        repo_dir=ROOT, cohort=cohort_bundle, ppo=smoke_ppo_config(128)
    )
    run_dir = tmp_path / "all_supported" / "seed_7"
    run_dir.mkdir(parents=True)
    completion = {
        "status": "complete",
        "condition": "all_supported",
        "seed": 7,
        "protocol_hash": protocol["protocol_hash"],
    }
    (run_dir / "completion.json").write_text(json.dumps(completion), encoding="utf-8")
    result = run_experiment(
        protocol=protocol,
        condition="all_supported",
        seed=7,
        cohort=cohort_bundle,
        output_root=tmp_path,
        device="cpu",
    )
    assert result["skipped_complete"] is True
    (run_dir / "completion.json").unlink()
    (run_dir / "config.json").write_text(
        json.dumps({"condition": "wrong"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="incompatible"):
        run_experiment(
            protocol=protocol,
            condition="all_supported",
            seed=7,
            cohort=cohort_bundle,
            output_root=tmp_path,
            device="cpu",
        )


def test_inventory_missing_and_complete_detection(tmp_path: Path) -> None:
    assert audit_run_inventory(tmp_path)["completed_count"] == 0
    for condition in POLICY_CONDITIONS:
        for seed in EXPERIMENT_SEEDS:
            path = tmp_path / condition / f"seed_{seed}"
            path.mkdir(parents=True)
            (path / "completion.json").write_text(
                json.dumps({"condition": condition, "seed": seed}), encoding="utf-8"
            )
    assert audit_run_inventory(tmp_path)["complete"] is True


def test_paired_comparison_and_hierarchical_bootstrap_are_scenario_seed_matched() -> None:
    rows = []
    for seed in EXPERIMENT_SEEDS:
        for scenario in ("a", "b", "c"):
            rows.extend(
                [
                    {"condition": "attention_supported", "training_seed": seed, "scenario_id": scenario, "patient_id": scenario, "bis_target_mae": 4.0},
                    {"condition": "all_supported", "training_seed": seed, "scenario_id": scenario, "patient_id": scenario, "bis_target_mae": 5.0},
                ]
            )
    paired = paired_contrast(pd.DataFrame(rows), left="attention_supported", right="all_supported", metric="bis_target_mae")
    assert len(paired) == 15
    summary = hierarchical_bootstrap(paired, replicates=200, seed=1)
    assert summary["observed_mean_difference"] == -1.0
    assert summary["p_value_used_as_winner_rule"] is False


@pytest.mark.parametrize(
    "notebook_name",
    ["colab_ppo_full_training.ipynb", "colab_ppo_validation_analysis.ipynb"],
)
def test_ppo_notebooks_are_clean_json_with_compilable_code(notebook_name: str) -> None:
    path = ROOT / "notebooks" / notebook_name
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{path}:cell-{index}")
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
    assert "test cohort" in source.lower()
    if notebook_name == "colab_ppo_full_training.ipynb":
        assert "RUN_20_PPO_CUDA_RUNS" not in source
        assert "torch_after.cuda.is_available()" in source
        assert "input(" in source
        assert "estimated_remaining_seconds" not in source
        assert "RUN_FULL_TRAINING = False" in source
        assert "--smoke-timesteps', '1000" in source
        assert "stdout=subprocess.PIPE" not in source


def test_rl_dependency_profile_pins_sb3_without_torch_or_pandas() -> None:
    lines = [
        line.strip()
        for line in (ROOT / "requirements-rl.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines == ["stable-baselines3==2.9.0"]
