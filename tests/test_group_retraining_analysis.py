"""Synthetic tests for validation-only group-retraining analysis."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.group_retraining_analysis import (
    CONDITIONS,
    EXPECTED_FEATURES,
    MODELS,
    PRIMARY_METRIC,
    SEEDS,
    aggregate_conditions,
    build_run_level_summary,
    discover_run_directories,
    hierarchical_paired_bootstrap,
    paired_condition_comparisons,
    pareto_candidates,
    patient_level_metrics,
    run_group_retraining_analysis,
    validate_experiment,
)


def _dump_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _patient_mae(predictions: pd.DataFrame) -> float:
    errors = np.abs(
        predictions["predicted_future_bis"] - predictions["observed_future_bis"]
    )
    return float(pd.DataFrame({"case_id": predictions["case_id"], "error": errors}).groupby("case_id")["error"].mean().mean())


def _write_synthetic_experiment(tmp_path: Path) -> tuple[Path, Path]:
    experiment_dir = tmp_path / "group_retraining_validation_only"
    dataset_dir = tmp_path / "modeling" / "full"
    dataset_dir.mkdir(parents=True)
    _dump_json(
        {
            "dynamic_feature_names": [*EXPECTED_FEATURES["full17"], "bis_error"],
            "static_feature_names": ["age", "sex_male"],
            "history_steps": 6,
            "history_window_seconds": 60,
            "prediction_horizon_seconds": 30,
            "resampling_interval_seconds": 10,
        },
        dataset_dir / "dataset_metadata.json",
    )
    (dataset_dir / "preprocessing.pkl").write_bytes(b"train-fitted-preprocessing")
    pd.DataFrame({"feature_name": EXPECTED_FEATURES["full17"]}).to_csv(
        dataset_dir / "preprocessing_statistics.csv", index=False
    )
    pd.DataFrame({"case_id": [1, 2]}).to_csv(dataset_dir / "train_metadata.csv", index=False)
    pd.DataFrame(
        {
            "case_id": [101, 101, 102, 102, 103, 103],
            "target_timestamp": np.arange(100, 160, 10),
        }
    ).to_csv(dataset_dir / "val_metadata.csv", index=False)

    condition_error = {
        "full17": 1.0,
        "no_remifentanil": 0.8,
        "no_respiratory": 1.1,
        "no_remifentanil_or_respiratory": 0.7,
    }
    targets = np.asarray([42.0, 44.0, 50.0, 52.0, 62.0, 64.0])
    case_ids = np.asarray([101, 101, 102, 102, 103, 103])
    for condition in CONDITIONS:
        for model in MODELS:
            for seed in SEEDS:
                run_dir = experiment_dir / condition / model / f"seed_{seed}"
                run_dir.mkdir(parents=True)
                error = condition_error[condition] + (0.01 * SEEDS.index(seed))
                if model == "attention":
                    error -= 0.1
                predictions = pd.DataFrame(
                    {
                        "sample_index": np.arange(len(targets)),
                        "case_id": case_ids,
                        "target_timestamp": np.arange(100, 160, 10),
                        "observed_future_bis": targets,
                        "predicted_future_bis": targets + error,
                    }
                )
                predictions["absolute_error"] = np.abs(
                    predictions["predicted_future_bis"] - predictions["observed_future_bis"]
                )
                predictions.to_csv(run_dir / "val_predictions.csv", index=False)
                mae = _patient_mae(predictions)
                _dump_json(
                    {
                        "patient_level": {"mae": {"mean": mae}},
                        "pooled_window": {
                            "regression": {
                                "mae": mae,
                                "rmse": mae,
                                "r_squared": 0.5,
                            }
                        },
                    },
                    run_dir / "val_metrics.json",
                )
                history = pd.DataFrame(
                    {
                        "epoch": [1, 2],
                        "train_loss": [2.0, 1.0],
                        "validation_patient_level_mae": [mae + 0.5, mae],
                    }
                )
                history.to_csv(run_dir / "training_history.csv", index=False)
                config = {
                    "seed": seed,
                    "device": "cuda",
                    "resolved_device": "cuda",
                    "backend": "cuda",
                    "learning_rate": 1e-3,
                    "weight_decay": 1e-4,
                    "batch_size": 256,
                    "max_epochs": 50,
                    "patience": 8,
                    "case_balanced_sampling": True,
                    "dynamic_feature_names": list(EXPECTED_FEATURES[condition]),
                    "static_feature_names": ["age", "sex_male"],
                    "selected_training_cases": [1, 2],
                    "selected_validation_cases": [101, 102, 103],
                    "git_commit_hash": "3387a7e",
                    "dataset_dir": str(dataset_dir.resolve()),
                    "evaluate_test": False,
                }
                _dump_json(config, run_dir / "config.json")
                _dump_json(
                    {
                        "status": "complete",
                        "seed": seed,
                        "resolved_device": "cuda",
                        "test_evaluated": False,
                        "best_epoch": 2,
                    },
                    run_dir / "run_status.json",
                )
                pd.DataFrame(
                    {"split": ["val"] * 3, "case_id": [101, 102, 103], "mae": [mae] * 3}
                ).to_csv(run_dir / "case_metrics.csv", index=False)
                (run_dir / "best_model.pt").write_bytes(b"best")
                (run_dir / "last_model.pt").write_bytes(b"last")
                if model == "gru":
                    _dump_json({"completed_epochs": 2}, run_dir / "runtime.json")
                else:
                    np.savez_compressed(
                        run_dir / "val_attention.npz",
                        sample_index=np.arange(len(targets)),
                        case_id=case_ids,
                        feature_attention=np.ones((len(targets), 6, len(EXPECTED_FEATURES[condition]))),
                    )
                    _dump_json(
                        {"runtime_breakdown": {"completed_epochs": 2}},
                        run_dir / "attention_metadata.json",
                    )
    return experiment_dir, dataset_dir


def _hash_run_inputs(experiment_dir: Path) -> dict[str, str]:
    hashes = {}
    for path in experiment_dir.rglob("*"):
        if path.is_file() and "analysis" not in path.relative_to(experiment_dir).parts:
            hashes[str(path.relative_to(experiment_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def test_missing_combination_is_detected(tmp_path: Path) -> None:
    experiment_dir, _ = _write_synthetic_experiment(tmp_path)
    (experiment_dir / "full17" / "gru" / "seed_7" / "config.json").unlink()
    with pytest.raises(ValueError, match="missing combinations"):
        discover_run_directories(experiment_dir)


def test_duplicate_run_is_detected(tmp_path: Path) -> None:
    experiment_dir, _ = _write_synthetic_experiment(tmp_path)
    source = experiment_dir / "full17" / "gru" / "seed_7"
    shutil.copytree(source, experiment_dir / "full17" / "gru" / "seed_7_duplicate")
    with pytest.raises(ValueError, match="duplicate combinations"):
        discover_run_directories(experiment_dir)


def test_incomplete_run_is_detected(tmp_path: Path) -> None:
    experiment_dir, _ = _write_synthetic_experiment(tmp_path)
    (experiment_dir / "full17" / "gru" / "seed_7" / "best_model.pt").unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete run"):
        validate_experiment(experiment_dir)


def test_incompatible_training_setting_is_detected(tmp_path: Path) -> None:
    experiment_dir, _ = _write_synthetic_experiment(tmp_path)
    config_path = experiment_dir / "full17" / "gru" / "seed_7" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["batch_size"] = 128
    _dump_json(config, config_path)
    with pytest.raises(ValueError, match="incompatible batch_size"):
        validate_experiment(experiment_dir)


def test_test_evaluation_and_forbidden_artifact_are_detected(tmp_path: Path) -> None:
    experiment_dir, _ = _write_synthetic_experiment(tmp_path)
    run_dir = experiment_dir / "full17" / "gru" / "seed_7"
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    status["test_evaluated"] = True
    _dump_json(status, run_dir / "run_status.json")
    with pytest.raises(ValueError, match="Test evaluation was enabled"):
        validate_experiment(experiment_dir)

    status["test_evaluated"] = False
    _dump_json(status, run_dir / "run_status.json")
    (run_dir / "test_metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Test split is not sealed"):
        validate_experiment(experiment_dir)


@pytest.mark.parametrize(
    ("column", "message"),
    (("case_id", "Validation patient IDs disagree|Mismatched validation case_id"),
     ("observed_future_bis", "Mismatched validation targets")),
)
def test_mismatched_validation_rows_and_targets_are_detected(
    tmp_path: Path, column: str, message: str
) -> None:
    experiment_dir, _ = _write_synthetic_experiment(tmp_path)
    path = experiment_dir / "no_remifentanil" / "gru" / "seed_7" / "val_predictions.csv"
    frame = pd.read_csv(path)
    frame.loc[0, column] = frame.loc[0, column] + 1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        validate_experiment(experiment_dir)


def test_paired_seed_deltas_and_patient_aggregation(tmp_path: Path) -> None:
    experiment_dir, _ = _write_synthetic_experiment(tmp_path)
    runs = validate_experiment(experiment_dir)
    run_level = build_run_level_summary(runs)
    deltas, contrasts = paired_condition_comparisons(run_level)
    selected = deltas.loc[
        (deltas["model"] == "gru")
        & (deltas["candidate_condition"] == "no_remifentanil")
    ]
    assert len(selected) == 5
    assert np.allclose(selected["paired_delta_candidate_minus_full17"], -0.2)
    summary = contrasts.loc[
        (contrasts["model"] == "gru")
        & (contrasts["candidate_condition"] == "no_remifentanil")
    ].iloc[0]
    assert summary["candidate_better_seed_count"] == 5

    patient = patient_level_metrics(runs)
    assert len(patient) == 40 * 3
    full_seed7 = patient.loc[
        (patient["condition"] == "full17")
        & (patient["model"] == "gru")
        & (patient["seed"] == 7)
    ]
    assert np.allclose(full_seed7["patient_mae"], 1.0)
    assert (full_seed7["window_count"] == 2).all()


def test_hierarchical_bootstrap_is_reproducible() -> None:
    frame = pd.DataFrame(
        [
            {"seed": seed, "case_id": case_id, "paired_delta": -0.2 + 0.01 * case_id}
            for seed in SEEDS
            for case_id in (1, 2, 3)
        ]
    )
    first = hierarchical_paired_bootstrap(frame, replicates=200, seed=123)
    second = hierarchical_paired_bootstrap(frame, replicates=200, seed=123)
    assert first == second
    assert first["patient_count_per_seed"] == 3
    assert first["percentile_95_ci_lower"] <= first["point_estimate_mean_delta"]
    assert first["percentile_95_ci_upper"] >= first["point_estimate_mean_delta"]


def test_pareto_dominance() -> None:
    run_level = pd.DataFrame(
        [
            {"condition": condition, "model": model, PRIMARY_METRIC: value, "dynamic_feature_count": count, "best_epoch": 2}
            for model in MODELS
            for condition, value, count in (
                ("full17", 1.0, 17),
                ("no_remifentanil", 0.9, 13),
                ("no_respiratory", 1.1, 15),
                ("no_remifentanil_or_respiratory", 0.95, 11),
            )
            for _ in SEEDS
        ]
    )
    aggregate = aggregate_conditions(run_level)
    result = pareto_candidates(aggregate)
    for model in MODELS:
        model_rows = result.loc[result["model"] == model].set_index("condition")
        assert bool(model_rows.loc["full17", "dominated"])
        assert bool(model_rows.loc["no_respiratory", "dominated"])
        assert bool(model_rows.loc["no_remifentanil", "pareto_frontier"])
        assert bool(model_rows.loc["no_remifentanil_or_respiratory", "pareto_frontier"])


def test_full_analysis_writes_only_analysis_directory(tmp_path: Path) -> None:
    experiment_dir, dataset_dir = _write_synthetic_experiment(tmp_path)
    before = _hash_run_inputs(experiment_dir)
    result = run_group_retraining_analysis(
        experiment_dir,
        dataset_dir,
        bootstrap_replicates=50,
        bootstrap_seed=123,
    )
    after = _hash_run_inputs(experiment_dir)
    assert before == after
    assert result["run_count"] == 40
    assert result["test_split_sealed"] is True
    analysis_dir = experiment_dir / "analysis"
    assert (analysis_dir / "analysis_manifest.json").is_file()
    assert (analysis_dir / "validation_analysis_report.md").is_file()
    assert len(list((analysis_dir / "figures").glob("*.png"))) == 5


def test_analysis_notebook_is_valid_and_has_no_training_commands() -> None:
    notebook_path = Path("notebooks/colab_group_retraining_analysis.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    source = "\n".join(code_cells)
    script_source = Path("scripts/analyze_group_retraining.py").read_text(encoding="utf-8")
    assert notebook["nbformat"] == 4
    assert "scripts/analyze_group_retraining.py" in source
    assert "run_baselines.py" not in source
    assert "run_attention.py" not in source
    assert "colab_full_training" not in source
    assert "test.npz" not in source
    assert "test.npz" not in script_source
    for index, cell_source in enumerate(code_cells):
        compile(cell_source, f"colab_group_analysis_cell_{index}", "exec")
