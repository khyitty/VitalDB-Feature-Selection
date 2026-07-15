"""Synthetic tests for patient-grouped train-only predictive feature selection."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from src.predictive_feature_selection import (
    DEFAULT_STABILITY_THRESHOLDS,
    SelectionConfig,
    build_candidate_subsets,
    build_consensus_table,
    build_design_matrix,
    build_feature_inventory,
    elastic_net_stability_selection,
    grouped_permutation_importance,
    load_train_selection_data,
    patient_grouped_folds,
    patient_subsample,
    run_predictive_feature_selection,
    train_only_correlations,
    tree_stability_selection,
)
from src.preprocessing import FeatureStatistics, PreprocessingArtifact
from src.redundancy_audit import REDUCED_FEATURES


class SyntheticTreeRegressor:
    """Small deterministic estimator implementing the production tree protocol."""

    def __init__(self, parameters: Mapping[str, Any], seed: int) -> None:
        self.model = DecisionTreeRegressor(
            max_depth=int(parameters["max_depth"]), random_state=seed
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "SyntheticTreeRegressor":
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


def _tree_factory(
    parameters: Mapping[str, Any], seed: int, device: str
) -> SyntheticTreeRegressor:
    assert device in {"cpu", "cuda"}
    return SyntheticTreeRegressor(parameters, seed)


def _dump_json(payload: Mapping[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _statistics(name: str) -> FeatureStatistics:
    mean = 50.0 if name == "bis" else 0.0
    scale = 10.0 if name in {"bis", "bis_error"} else 1.0
    return FeatureStatistics(
        feature_name=name,
        training_median=mean,
        training_mean=mean,
        training_standard_deviation=scale,
        imputation_value=mean,
        normalization_scale=scale,
        feature_type="dynamic_continuous",
        standardized=True,
    )


@pytest.fixture
def synthetic_selection_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset_dir = tmp_path / "modeling" / "full"
    group_analysis_dir = tmp_path / "group_analysis"
    output_dir = tmp_path / "predictive_feature_selection_30s"
    dataset_dir.mkdir(parents=True)
    group_analysis_dir.mkdir()

    dynamic_names = (*REDUCED_FEATURES, "bis_error")
    static_names = ("age", "sex_male", "height", "weight", "bmi", "asa")
    patients = np.repeat(np.arange(1, 11), 4)
    windows = len(patients)
    history_steps = 3
    generator = np.random.default_rng(42)
    reduced = generator.normal(size=(windows, history_steps, len(REDUCED_FEATURES))).astype(
        np.float32
    )
    reduced[:, :, REDUCED_FEATURES.index("rftn_volume")] = (
        reduced[:, :, REDUCED_FEATURES.index("rftn_rate")] * 0.99
    )
    bis = reduced[:, :, REDUCED_FEATURES.index("bis")]
    full_dynamic = np.concatenate((reduced, bis[:, :, None]), axis=2)
    X_static = generator.normal(size=(windows, len(static_names))).astype(np.float32)
    y_bis = (
        50.0
        + 5.0 * reduced[:, -1, REDUCED_FEATURES.index("bis")]
        + 2.0 * reduced[:, -1, REDUCED_FEATURES.index("hr")]
        + generator.normal(scale=0.1, size=windows)
    ).astype(np.float32)
    np.savez_compressed(
        dataset_dir / "train.npz",
        X_dynamic=full_dynamic,
        X_static=X_static,
        y_bis=y_bis,
    )
    pd.DataFrame(
        {
            "case_id": patients,
            "target_timestamp": np.tile(np.arange(100, 140, 10), 10),
        }
    ).to_csv(dataset_dir / "train_metadata.csv", index=False)
    _dump_json(
        {
            "dynamic_feature_names": list(dynamic_names),
            "static_feature_names": list(static_names),
            "history_steps": history_steps,
            "history_window_seconds": 30,
            "prediction_horizon_seconds": 30,
            "resampling_interval_seconds": 10,
        },
        dataset_dir / "dataset_metadata.json",
    )
    manifest_rows = []
    for name in (*dynamic_names, *static_names):
        manifest_rows.append(
            {
                "original_column_name": name,
                "standardized_feature_name": name,
                "dynamic_or_static": "static" if name in static_names else "dynamic",
                "aggregation_rule": "constant" if name in static_names else "median",
                "percentage_missing_before_imputation": 0.0,
                "included": True,
                "exclusion_reason": "",
            }
        )
    pd.DataFrame(manifest_rows).to_csv(dataset_dir / "feature_manifest.csv", index=False)
    statistics = {name: _statistics(name) for name in (*dynamic_names, *static_names)}
    artifact = PreprocessingArtifact(statistics, dynamic_names, static_names)
    (dataset_dir / "preprocessing.pkl").write_bytes(pickle.dumps(artifact))
    artifact.statistics_frame().to_csv(
        dataset_dir / "preprocessing_statistics.csv", index=False
    )

    # Invalid sentinels prove that selection does not touch non-train arrays.
    (dataset_dir / "val.npz").write_bytes(b"must-not-open")
    (dataset_dir / "test.npz").write_bytes(b"must-not-open")
    _dump_json(
        {
            "test_split_sealed": True,
            "training_git_commits": ["3387a7e"],
            "run_count": 40,
        },
        group_analysis_dir / "analysis_manifest.json",
    )
    pd.DataFrame({"condition": ["full17"], "model": ["gru"]}).to_csv(
        group_analysis_dir / "validation_condition_aggregate.csv", index=False
    )
    pd.DataFrame({"contrast": ["no_respiratory - full17"]}).to_csv(
        group_analysis_dir / "hierarchical_bootstrap_contrasts.csv", index=False
    )
    return dataset_dir, group_analysis_dir, output_dir


def _hash_directory(path: Path) -> dict[str, str]:
    return {
        str(file.relative_to(path)): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in path.rglob("*")
        if file.is_file()
    }


def test_grouped_folds_never_split_patients() -> None:
    patients = np.repeat(np.arange(12), [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])
    folds = patient_grouped_folds(patients, n_splits=3, seed=7)
    all_evaluation_rows = []
    for train_indices, evaluation_indices in folds:
        assert not set(patients[train_indices]) & set(patients[evaluation_indices])
        all_evaluation_rows.extend(evaluation_indices.tolist())
    assert sorted(all_evaluation_rows) == list(range(len(patients)))


def test_patient_subsampling_keeps_complete_patient_blocks() -> None:
    patients = np.repeat(np.arange(10), 3)
    selected, heldout, selected_patients = patient_subsample(
        patients, 0.5, np.random.default_rng(42)
    )
    assert len(selected_patients) == 5
    assert not set(patients[selected]) & set(patients[heldout])
    for patient in selected_patients:
        assert np.sum(patients[selected] == patient) == 3


def test_train_loader_excludes_duplicate_and_ignores_non_train_sentinels(
    synthetic_selection_inputs: tuple[Path, Path, Path],
) -> None:
    dataset_dir, _, _ = synthetic_selection_inputs
    data = load_train_selection_data(dataset_dir)
    inventory = build_feature_inventory(data)
    assert data.dynamic_features == REDUCED_FEATURES
    assert data.X_dynamic.shape[-1] == 17
    duplicate = inventory.loc[inventory["feature"] == "bis_error"].iloc[0]
    assert duplicate["deterministic_duplicate_excluded"]
    assert not duplicate["eligible_for_predictive_selection"]
    assert set(data.static_features) == {"age", "sex_male", "height", "weight", "bmi", "asa"}


def test_elastic_net_stability_is_reproducible(
    synthetic_selection_inputs: tuple[Path, Path, Path],
) -> None:
    dataset_dir, _, _ = synthetic_selection_inputs
    data = load_train_selection_data(dataset_dir)
    design = build_design_matrix(data)
    parameters = {"alpha": 0.01, "l1_ratio": 0.5, "parameter_id": 0}
    first, first_coefficients, _ = elastic_net_stability_selection(
        data,
        design,
        parameters,
        iterations=4,
        subsample_fraction=0.5,
        seed=123,
    )
    second, second_coefficients, _ = elastic_net_stability_selection(
        data,
        design,
        parameters,
        iterations=4,
        subsample_fraction=0.5,
        seed=123,
    )
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_coefficients, second_coefficients)


def test_tree_grouped_permutation_and_stability_are_reproducible(
    synthetic_selection_inputs: tuple[Path, Path, Path],
) -> None:
    dataset_dir, _, _ = synthetic_selection_inputs
    data = load_train_selection_data(dataset_dir)
    design = build_design_matrix(data)
    model = _tree_factory({"max_depth": 3}, 42, "cpu")
    model.fit(design.X, data.y_bis)
    first_importance = grouped_permutation_importance(
        model,
        design.X,
        data.y_bis,
        data.patient_ids,
        data.target_timestamps,
        design.dynamic_column_indices,
        repeats=2,
        seed=11,
    )
    second_importance = grouped_permutation_importance(
        model,
        design.X,
        data.y_bis,
        data.patient_ids,
        data.target_timestamps,
        design.dynamic_column_indices,
        repeats=2,
        seed=11,
    )
    pd.testing.assert_frame_equal(first_importance, second_importance)
    parameters = {
        "max_depth": 3,
        "min_child_weight": 5.0,
        "learning_rate": 0.1,
        "n_estimators": 10,
        "parameter_id": 0,
    }
    first, first_subsamples = tree_stability_selection(
        data,
        design,
        parameters,
        iterations=3,
        subsample_fraction=0.5,
        device="cpu",
        seed=55,
        estimator_factory=_tree_factory,
    )
    second, second_subsamples = tree_stability_selection(
        data,
        design,
        parameters,
        iterations=3,
        subsample_fraction=0.5,
        device="cpu",
        seed=55,
        estimator_factory=_tree_factory,
    )
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_subsamples, second_subsamples)


def test_correlation_clusters_are_train_only_and_detect_redundancy(
    synthetic_selection_inputs: tuple[Path, Path, Path],
) -> None:
    dataset_dir, _, _ = synthetic_selection_inputs
    data = load_train_selection_data(dataset_dir)
    pearson, spearman, clusters = train_only_correlations(data, cluster_threshold=0.8)
    assert len(pearson) == len(spearman) == 17 * 16 // 2
    indexed = clusters.set_index("feature")
    assert (
        indexed.loc["rftn_rate", "correlation_cluster"]
        == indexed.loc["rftn_volume", "correlation_cluster"]
    )
    assert "not an automatic removal rule" in indexed.loc["rftn_rate", "removal_note"]


def _consensus_fixture() -> pd.DataFrame:
    rows = []
    for index, feature in enumerate(REDUCED_FEATURES):
        rows.append(
            {
                "feature": feature,
                "elastic_net_stable": index < 5,
                "tree_stable": 2 <= index < 8,
                "elastic_net_selection_probability": 0.9 if index < 5 else 0.2,
                "tree_selection_probability": 0.9 if 2 <= index < 8 else 0.2,
                "method_agreement_count": int(index < 5) + int(2 <= index < 8),
                "tree_stability_importance_mean": float(17 - index),
                "elastic_net_median_absolute_coefficient": float(10 - index),
                "correlation_cluster": f"cluster_{index // 2}",
            }
        )
    return pd.DataFrame(rows)


def test_consensus_merge_and_candidate_rules() -> None:
    elastic = pd.DataFrame(
        {
            "feature": REDUCED_FEATURES,
            "selection_probability": [0.8] * 17,
            "median_standardized_coefficient": [0.1] * 17,
            "median_absolute_standardized_coefficient": [0.1] * 17,
            "coefficient_sign_consistency": [1.0] * 17,
        }
    )
    tree_importance = pd.DataFrame(
        {
            "feature": REDUCED_FEATURES,
            "grouped_permutation_importance_mean": [0.1] * 17,
            "grouped_permutation_importance_standard_deviation": [0.01] * 17,
        }
    )
    tree_stability = pd.DataFrame(
        {
            "feature": REDUCED_FEATURES,
            "selection_probability": [0.8] * 17,
            "grouped_permutation_importance_mean": [0.1] * 17,
            "grouped_permutation_importance_standard_deviation": [0.01] * 17,
        }
    )
    clusters = pd.DataFrame(
        {
            "feature": REDUCED_FEATURES,
            "correlation_cluster": [f"cluster_{index}" for index in range(17)],
            "cluster_members": [json.dumps([feature]) for feature in REDUCED_FEATURES],
            "feature_group": ["current_bis"] * 17,
        }
    )
    consensus = build_consensus_table(
        elastic,
        tree_importance,
        tree_stability,
        clusters,
        stable_threshold=0.7,
    )
    assert (consensus["method_agreement_count"] == 2).all()
    assert not consensus["automatic_final_selection"].any()

    candidates = build_candidate_subsets(
        _consensus_fixture(),
        protected_control_features=("rftn_ce",),
        max_frozen_candidates=6,
    )
    required = {
        "elastic_net_stable",
        "tree_stable",
        "strict_consensus",
        "consensus_union",
        "compact_consensus",
        "no_respiratory_anchor",
        "compact11_anchor",
        "full17_reference",
    }
    assert required.issubset(candidates["all_candidate_subsets"])
    assert len(candidates["frozen_retraining_candidates"]) <= 6
    assert "rftn_ce" in candidates["all_candidate_subsets"]["control_aware_consensus"]["features"]
    assert all(
        "bis_error" not in payload["features"]
        for payload in candidates["all_candidate_subsets"].values()
    )


def test_end_to_end_selection_writes_only_output_directory(
    synthetic_selection_inputs: tuple[Path, Path, Path],
) -> None:
    dataset_dir, group_analysis_dir, output_dir = synthetic_selection_inputs
    dataset_before = _hash_directory(dataset_dir)
    group_before = _hash_directory(group_analysis_dir)
    result = run_predictive_feature_selection(
        SelectionConfig(
            dataset_dir=dataset_dir,
            group_analysis_dir=group_analysis_dir,
            output_dir=output_dir,
            internal_folds=2,
            stability_iterations=3,
            tree_permutation_repeats=1,
            tree_estimators=10,
            protected_control_features=("rftn_ce",),
        ),
        estimator_factory=_tree_factory,
    )
    assert _hash_directory(dataset_dir) == dataset_before
    assert _hash_directory(group_analysis_dir) == group_before
    assert result["train_only"] is True
    assert result["validation_loaded"] is False
    assert result["test_loaded"] is False
    expected = {
        "feature_inventory.csv",
        "train_internal_cv_manifest.csv",
        "elastic_net_coefficients.csv",
        "elastic_net_stability.csv",
        "tree_permutation_importance.csv",
        "tree_stability.csv",
        "feature_correlations_pearson.csv",
        "feature_correlations_spearman.csv",
        "correlation_clusters.csv",
        "consensus_feature_table.csv",
        "candidate_subsets.json",
        "selection_manifest.json",
        "predictive_feature_selection_report.md",
    }
    assert expected.issubset(path.name for path in output_dir.iterdir())
    assert len(list((output_dir / "figures").glob("*.png"))) == 7
    cv = pd.read_csv(output_dir / "train_internal_cv_manifest.csv")
    assert (cv["patient_overlap_count"] == 0).all()


def test_notebook_is_valid_and_contains_no_deep_training_or_test_access() -> None:
    notebook_path = Path("notebooks/colab_predictive_feature_selection.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    source = "\n".join(code_cells)
    script = Path("scripts/run_predictive_feature_selection.py").read_text(encoding="utf-8")
    assert notebook["nbformat"] == 4
    assert "scripts/run_predictive_feature_selection.py" in source
    assert "run_baselines.py" not in source
    assert "run_attention.py" not in source
    assert "colab_full_training" not in source
    assert "test.npz" not in source
    assert "test.npz" not in script
    for index, cell_source in enumerate(code_cells):
        compile(cell_source, f"colab_predictive_selection_cell_{index}", "exec")


def test_scientific_cli_rejects_fewer_than_100_iterations() -> None:
    from scripts.run_predictive_feature_selection import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--dataset-dir",
                "data",
                "--group-analysis-dir",
                "analysis",
                "--output-dir",
                "output",
                "--stability-iterations",
                "99",
            ]
        )
    parsed = parse_args(
        [
            "--dataset-dir",
            "data",
            "--group-analysis-dir",
            "analysis",
            "--output-dir",
            "output",
        ]
    )
    assert parsed.stability_iterations == 100
    assert DEFAULT_STABILITY_THRESHOLDS == (0.6, 0.7, 0.8, 0.9)
