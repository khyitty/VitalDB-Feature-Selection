"""Guarded one-time held-out evaluation of the frozen predictive decision."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.attention_training import AttentionTrainingConfig, _fresh_attention_model
from src.datasets import VitalBISDataset
from src.frozen_candidate_retraining import (
    DATASET_FINGERPRINT_FILES,
    dataset_fingerprint,
    dump_json,
    sha256_file,
)
from src.group_retraining_analysis import hierarchical_paired_bootstrap, load_json
from src.metrics import patient_level_evaluation, pooled_evaluation
from src.models.baselines import GRUBaseline
from src.training import (
    PredictionBundle,
    TrainingConfig,
    _load_checkpoint,
    make_data_loader,
    predict_model,
    prediction_frame,
    resolve_device,
    set_deterministic_seed,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

LOGGER = logging.getLogger(__name__)

CONFIRMATION_PHRASE = "RUN_ONE_TIME_FROZEN_PREDICTIVE_TEST"
PRIMARY_CANDIDATE = "strict_consensus"
REFERENCE_CANDIDATE = "full17_reference"
CANDIDATES = (PRIMARY_CANDIDATE, REFERENCE_CANDIDATE)
MODELS = ("gru", "attention")
SEEDS = (7, 21, 42, 84, 123)
PRIMARY_FEATURES = (
    "bis",
    "bis_sqi",
    "ppf_rate",
    "ppf_volume",
    "ppf_cp",
    "rftn_volume",
    "bis_slope",
)
REFERENCE_FEATURES = (
    "bis",
    "bis_sqi",
    "hr",
    "mbp",
    "sbp",
    "dbp",
    "spo2",
    "etco2",
    "ppf_rate",
    "ppf_volume",
    "ppf_cp",
    "ppf_ce",
    "rftn_rate",
    "rftn_volume",
    "rftn_cp",
    "rftn_ce",
    "bis_slope",
)
FEATURES = {
    PRIMARY_CANDIDATE: PRIMARY_FEATURES,
    REFERENCE_CANDIDATE: REFERENCE_FEATURES,
}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260718
REQUIRED_DECISION_KEYS = (
    "decision_timestamp_utc",
    "decision_code_commit",
    "source_analysis_manifest_sha256",
    "primary_candidate",
    "reference_candidate",
    "primary_dynamic_feature_names",
    "reference_dynamic_feature_names",
    "pre_test_freeze",
    "candidate_changes_after_test_prohibited",
)
RUN_ARTIFACTS = ("test_predictions.csv", "test_metrics.json", "patient_metrics.csv")
FINAL_TABLES = (
    "test_run_level_metrics.csv",
    "test_candidate_aggregate.csv",
    "paired_test_seed_deltas.csv",
    "patient_level_test_metrics.csv",
    "hierarchical_bootstrap_test_contrasts.csv",
    "paired_model_test_contrasts.csv",
)

InferenceFunction = Callable[[Mapping[str, Any], Path, str, int], pd.DataFrame]


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_once_text(path: Path, content: str) -> None:
    """Create one artifact or verify that its existing bytes are identical."""

    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Refusing to overwrite incompatible test artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_once_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    _write_once_text(path, content)


def _write_once_csv(path: Path, frame: pd.DataFrame) -> None:
    _write_once_text(path, frame.to_csv(index=False, lineterminator="\n"))


def validate_frozen_decision(payload: Mapping[str, Any]) -> None:
    """Require the exact pre-test primary/reference decision and exclusions."""

    missing = [key for key in REQUIRED_DECISION_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Frozen decision is missing required fields: {missing}")
    if payload["primary_candidate"] != PRIMARY_CANDIDATE:
        raise ValueError("Frozen primary candidate is not strict_consensus.")
    if payload["reference_candidate"] != REFERENCE_CANDIDATE:
        raise ValueError("Frozen reference candidate is not full17_reference.")
    if tuple(payload["primary_dynamic_feature_names"]) != PRIMARY_FEATURES:
        raise ValueError("Frozen strict_consensus feature order has changed.")
    if tuple(payload["reference_dynamic_feature_names"]) != REFERENCE_FEATURES:
        raise ValueError("Frozen full17_reference feature order has changed.")
    if payload.get("secondary_validation_only_candidate") != "compact_consensus":
        raise ValueError("compact_consensus must remain secondary validation-only evidence.")
    if payload.get("test_evaluation_candidates") != list(CANDIDATES):
        raise ValueError("Held-out test candidates must be exactly strict_consensus and full17_reference.")
    if payload.get("pre_test_freeze") is not True:
        raise ValueError("Decision manifest does not certify a pre-test freeze.")
    if payload.get("candidate_changes_after_test_prohibited") is not True:
        raise ValueError("Decision manifest does not prohibit post-test candidate changes.")


def freeze_decision_from_template(
    template_dir: Path,
    destination_dir: Path,
    source_analysis_manifest: Path,
) -> dict[str, Any]:
    """Persist or verify an immutable decision before any test data are opened."""

    template_json = template_dir / "frozen_predictive_decision.json"
    template_md = template_dir / "frozen_predictive_decision.md"
    if not template_json.is_file() or not template_md.is_file():
        raise FileNotFoundError(f"Frozen decision template is incomplete: {template_dir}")
    decision = load_json(template_json)
    validate_frozen_decision(decision)
    observed_analysis_sha = sha256_file(source_analysis_manifest)
    if decision["source_analysis_manifest_sha256"] != observed_analysis_sha:
        raise ValueError(
            "Source validation analysis manifest changed after the predictive decision: "
            f"expected {decision['source_analysis_manifest_sha256']}, "
            f"observed {observed_analysis_sha}."
        )
    _write_once_text(
        destination_dir / template_json.name,
        template_json.read_text(encoding="utf-8"),
    )
    _write_once_text(
        destination_dir / template_md.name,
        template_md.read_text(encoding="utf-8"),
    )
    return decision


def _source_run_dir(
    candidate: str,
    model: str,
    seed: int,
    strict_root: Path,
    full17_root: Path,
) -> Path:
    root = strict_root if candidate == PRIMARY_CANDIDATE else full17_root
    return root / model / f"seed_{seed}"


def _validate_source_config(
    config: Mapping[str, Any],
    status: Mapping[str, Any],
    *,
    candidate: str,
    model: str,
    seed: int,
    run_dir: Path,
    dataset_dir: Path,
) -> None:
    if int(config.get("seed", -1)) != seed:
        raise ValueError(f"Source seed mismatch in {run_dir}.")
    if tuple(config.get("dynamic_feature_names", ())) != FEATURES[candidate]:
        raise ValueError(f"Source feature mismatch in {run_dir}.")
    if config.get("evaluate_test") is not False:
        raise ValueError(f"Source run was not validation-only: {run_dir}")
    if status.get("status") != "complete" or status.get("test_evaluated") is not False:
        raise ValueError(f"Source run is incomplete or its test seal is broken: {run_dir}")
    expected_model_name = "FactorizedAttentionGRU" if model == "attention" else None
    if config.get("model_name") != expected_model_name:
        raise ValueError(f"Source model identity mismatch in {run_dir}.")
    configured_dataset = Path(str(config.get("dataset_dir", ""))).resolve()
    if configured_dataset != dataset_dir.resolve():
        raise ValueError(
            f"Source dataset path mismatch in {run_dir}: "
            f"{configured_dataset} != {dataset_dir.resolve()}"
        )


def build_checkpoint_inventory(
    dataset_dir: Path,
    strict_root: Path,
    full17_root: Path,
) -> pd.DataFrame:
    """Validate and fingerprint the exact 20 validation-selected checkpoints."""

    fingerprint = dataset_fingerprint(dataset_dir)["combined_sha256"]
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for model in MODELS:
            for seed in SEEDS:
                run_dir = _source_run_dir(
                    candidate, model, seed, strict_root, full17_root
                )
                config_path = run_dir / "config.json"
                status_path = run_dir / "run_status.json"
                checkpoint_path = run_dir / "best_model.pt"
                for path in (config_path, status_path, checkpoint_path):
                    if not path.is_file():
                        raise FileNotFoundError(f"Required source artifact is missing: {path}")
                config = load_json(config_path)
                status = load_json(status_path)
                _validate_source_config(
                    config,
                    status,
                    candidate=candidate,
                    model=model,
                    seed=seed,
                    run_dir=run_dir,
                    dataset_dir=dataset_dir,
                )
                rows.append(
                    {
                        "candidate": candidate,
                        "model": model,
                        "seed": seed,
                        "source_run_directory": str(run_dir),
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_name": checkpoint_path.name,
                        "checkpoint_sha256": sha256_file(checkpoint_path),
                        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
                        "config_sha256": sha256_file(config_path),
                        "training_git_commit": config.get("git_commit_hash"),
                        "dynamic_feature_names": json.dumps(
                            list(FEATURES[candidate]), separators=(",", ":")
                        ),
                        "dataset_fingerprint": fingerprint,
                        "validation_selected_checkpoint_only": True,
                        "run_complete": status.get("status") == "complete",
                        "run_test_evaluated": status.get("test_evaluated"),
                        "evaluate_test": config.get("evaluate_test"),
                        "inference_only": True,
                    }
                )
    inventory = pd.DataFrame(rows)
    if len(inventory) != 20 or inventory.duplicated(
        ["candidate", "model", "seed"]
    ).any():
        raise AssertionError("Checkpoint inventory must contain exactly 20 unique runs.")
    if set(inventory["checkpoint_name"]) != {"best_model.pt"}:
        raise AssertionError("Only best_model.pt may be evaluated.")
    return inventory


def prepare_test_preflight(
    *,
    decision_template_dir: Path,
    decision_dir: Path,
    source_analysis_manifest: Path,
    dataset_dir: Path,
    strict_root: Path,
    full17_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Freeze the decision and inventory without opening the held-out split."""

    decision = freeze_decision_from_template(
        decision_template_dir, decision_dir, source_analysis_manifest
    )
    inventory = build_checkpoint_inventory(dataset_dir, strict_root, full17_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_once_json(output_dir / "frozen_decision_snapshot.json", decision)
    _write_once_csv(output_dir / "evaluated_checkpoint_inventory.csv", inventory)
    return {
        "preflight_complete": True,
        "test_split_opened": False,
        "checkpoint_count": len(inventory),
        "confirmation_required": CONFIRMATION_PHRASE,
        "decision_snapshot": str(output_dir / "frozen_decision_snapshot.json"),
        "checkpoint_inventory": str(output_dir / "evaluated_checkpoint_inventory.csv"),
    }


def full_test_dataset_fingerprint(dataset_dir: Path) -> dict[str, Any]:
    """Fingerprint train-fitted metadata plus the held-out arrays after confirmation."""

    validation_safe = dataset_fingerprint(dataset_dir)
    test_files = ("test.npz", "test_metadata.csv")
    missing = [name for name in test_files if not (dataset_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Held-out dataset artifacts are missing: {missing}")
    files = {
        **validation_safe["files"],
        **{
            name: {
                "sha256": sha256_file(dataset_dir / name),
                "size_bytes": (dataset_dir / name).stat().st_size,
            }
            for name in test_files
        },
    }
    combined = _sha256_text(_canonical_json(files))
    return {
        "combined_sha256": combined,
        "validation_safe_combined_sha256": validation_safe["combined_sha256"],
        "files": files,
    }


def _training_config(config: Mapping[str, Any], run_dir: Path) -> TrainingConfig:
    return TrainingConfig(
        dataset_dir=Path(str(config["dataset_dir"])),
        output_dir=run_dir,
        seed=int(config["seed"]),
        device=str(config.get("device", "auto")),
        learning_rate=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
        batch_size=int(config.get("batch_size", 256)),
        max_epochs=int(config.get("max_epochs", 50)),
        patience=int(config.get("patience", 8)),
        hidden_size=int(config.get("hidden_size", 64)),
        projection_size=int(config.get("projection_size", 64)),
        static_hidden_size=int(config.get("static_hidden_size", 16)),
        prediction_hidden_size=int(config.get("prediction_hidden_size", 32)),
        dropout=float(config.get("dropout", 0.0)),
        dynamic_features=tuple(config["dynamic_feature_names"]),
        evaluate_test=False,
    )


def _default_inference(
    record: Mapping[str, Any], dataset_dir: Path, device_name: str, batch_size: int
) -> pd.DataFrame:
    """Load one best checkpoint and infer on test without training or selection."""

    run_dir = Path(str(record["source_run_directory"]))
    config_payload = load_json(run_dir / "config.json")
    seed = int(record["seed"])
    set_deterministic_seed(seed)
    device = resolve_device(device_name)
    dataset = VitalBISDataset(
        dataset_dir,
        "test",
        dynamic_features=FEATURES[str(record["candidate"])],
    )
    base_config = _training_config(config_payload, run_dir)
    if record["model"] == "attention":
        attention_config = AttentionTrainingConfig(
            **base_config.__dict__,
            feature_token_embedding_dim=int(
                config_payload.get("feature_token_embedding_dim", 16)
            ),
            static_context_dim=int(config_payload.get("static_context_dim", 16)),
        )
        model: nn.Module = _fresh_attention_model(attention_config, dataset)
    else:
        model = GRUBaseline(
            dynamic_feature_count=len(dataset.dynamic_feature_names),
            static_feature_count=len(dataset.static_feature_names),
            hidden_size=base_config.hidden_size,
            projection_size=base_config.projection_size,
            static_hidden_size=base_config.static_hidden_size,
            prediction_hidden_size=base_config.prediction_hidden_size,
            dropout=base_config.dropout,
        )
    model = model.to(device)
    _load_checkpoint(
        Path(str(record["checkpoint_path"])), model, optimizer=None, device=device
    )
    indices = np.arange(len(dataset), dtype=np.int64)
    loader = make_data_loader(
        dataset,
        indices,
        batch_size,
        seed,
        training=False,
        case_balanced=False,
        num_workers=0,
    )
    bundle = predict_model(model, loader, nn.HuberLoss(delta=1.0), device)
    return prediction_frame(bundle)


def _validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sample_index",
        "case_id",
        "target_timestamp",
        "observed_future_bis",
        "predicted_future_bis",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"Test predictions are empty or incomplete: {missing}")
    result = frame.sort_values("sample_index", kind="stable").reset_index(drop=True)
    if result["sample_index"].duplicated().any():
        raise ValueError("Test predictions contain duplicate sample indices.")
    numeric = result.loc[:, list(required)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Test predictions contain non-finite values.")
    return result


def _numeric_values_are_finite(value: Any) -> bool:
    """Return false only for non-finite numeric leaves in a nested artifact."""

    if isinstance(value, Mapping):
        return all(_numeric_values_are_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_numeric_values_are_finite(item) for item in value)
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    ):
        return bool(np.isfinite(value))
    return True


def verify_test_result_integrity(
    inventory: pd.DataFrame,
    output_dir: Path,
    dataset_dir: Path,
) -> dict[str, Any]:
    """Verify all 20 outputs, source inventory, and exact held-out row pairing."""

    expected_keys = {
        (candidate, model, seed)
        for candidate in CANDIDATES
        for model in MODELS
        for seed in SEEDS
    }
    observed_keys = {
        (str(row.candidate), str(row.model), int(row.seed))
        for row in inventory.itertuples()
    }
    if len(inventory) != 20 or len(observed_keys) != 20 or observed_keys != expected_keys:
        raise ValueError("Held-out inventory is missing, duplicated, or contains an extra run.")
    if "compact_consensus" in set(inventory["candidate"]):
        raise ValueError("compact_consensus is prohibited from held-out evaluation.")
    if set(inventory["checkpoint_name"]) != {"best_model.pt"}:
        raise ValueError("Held-out inventory contains a non-best checkpoint.")

    dataset = VitalBISDataset(
        dataset_dir, "test", dynamic_features=REFERENCE_FEATURES
    )
    expected = pd.DataFrame(
        {
            "sample_index": np.arange(len(dataset), dtype=np.int64),
            "case_id": dataset.case_ids,
            "target_timestamp": dataset.metadata["target_timestamp"].to_numpy(
                dtype=np.int64
            ),
            "observed_future_bis": dataset.arrays["y_bis"].astype(float),
        }
    )
    alignment_columns = [
        "sample_index",
        "case_id",
        "target_timestamp",
        "observed_future_bis",
    ]
    canonical: pd.DataFrame | None = None
    patient_ids = sorted(expected["case_id"].astype(int).unique().tolist())
    for record in inventory.to_dict("records"):
        key = (record["candidate"], record["model"], int(record["seed"]))
        run_output = _run_output_dir(output_dir, record)
        prediction_path = run_output / "test_predictions.csv"
        metrics_path = run_output / "test_metrics.json"
        patient_path = run_output / "patient_metrics.csv"
        if not all(path.is_file() for path in (prediction_path, metrics_path, patient_path)):
            raise ValueError(f"Missing held-out run artifact for {key}: {run_output}")
        prediction = _validate_predictions(pd.read_csv(prediction_path))
        alignment = prediction.loc[:, alignment_columns]
        if not np.array_equal(
            alignment[["sample_index", "case_id", "target_timestamp"]].to_numpy(
                dtype=np.int64
            ),
            expected[["sample_index", "case_id", "target_timestamp"]].to_numpy(
                dtype=np.int64
            ),
        ) or not np.allclose(
            alignment["observed_future_bis"].to_numpy(float),
            expected["observed_future_bis"].to_numpy(float),
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(f"Held-out patient/timestamp/target alignment mismatch for {key}.")
        if canonical is None:
            canonical = alignment
        elif not np.array_equal(
            alignment[["sample_index", "case_id", "target_timestamp"]].to_numpy(),
            canonical[["sample_index", "case_id", "target_timestamp"]].to_numpy(),
        ) or not np.allclose(
            alignment["observed_future_bis"].to_numpy(float),
            canonical["observed_future_bis"].to_numpy(float),
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(f"Candidate/reference target pairing mismatch for {key}.")
        metrics = load_json(metrics_path)
        if not _numeric_values_are_finite(metrics):
            raise ValueError(f"Non-finite held-out metric detected for {key}.")
        if metrics.get("predictions_sha256") != sha256_file(prediction_path):
            raise ValueError(f"Prediction hash mismatch for {key}.")
        patients = pd.read_csv(patient_path)
        if sorted(patients["case_id"].astype(int).tolist()) != patient_ids:
            raise ValueError(f"Patient-level aggregation mismatch for {key}.")
        required_patient_metrics = ["case_id", "number_of_windows", "mae", "rmse"]
        missing_patient_metrics = sorted(
            set(required_patient_metrics) - set(patients.columns)
        )
        if missing_patient_metrics:
            raise ValueError(
                f"Missing patient-level metrics for {key}: {missing_patient_metrics}"
            )
        required_numeric = patients[required_patient_metrics].to_numpy(float)
        if not np.isfinite(required_numeric).all():
            raise ValueError(f"Non-finite patient metric detected for {key}.")
        for metric_name in ("high_bis_auprc", "high_bis_auroc"):
            defined_name = f"{metric_name}_defined"
            if metric_name not in patients or defined_name not in patients:
                raise ValueError(
                    f"Missing patient-level classification fields for {key}: "
                    f"{metric_name}, {defined_name}"
                )
            metric = pd.to_numeric(patients[metric_name], errors="coerce")
            defined = patients[defined_name].astype(bool)
            if not np.isfinite(metric[defined].to_numpy(float)).all():
                raise ValueError(
                    f"Defined patient-level {metric_name} is non-finite for {key}."
                )
            if metric[~defined].notna().any():
                raise ValueError(
                    f"Undefined patient-level {metric_name} has a numeric value for {key}."
                )
    assert canonical is not None
    canonical_sha = _sha256_text(
        canonical.to_csv(index=False, lineterminator="\n")
    )
    return {
        "checkpoint_evaluation_count": 20,
        "duplicate_run_count": 0,
        "missing_run_count": 0,
        "test_window_count": len(expected),
        "test_patient_count": len(patient_ids),
        "row_alignment_exact": True,
        "candidate_reference_pairing_exact": True,
        "all_metrics_finite": True,
        "canonical_alignment_sha256": canonical_sha,
    }


def _summarize_prediction(
    frame: pd.DataFrame, record: Mapping[str, Any], dataset_sha: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    observed = frame["observed_future_bis"].to_numpy(float)
    predicted = frame["predicted_future_bis"].to_numpy(float)
    case_ids = frame["case_id"].to_numpy(int)
    pooled = pooled_evaluation(observed, predicted)
    patient = patient_level_evaluation(observed, predicted, case_ids)
    metrics = {
        "candidate": record["candidate"],
        "model": record["model"],
        "seed": int(record["seed"]),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "dataset_fingerprint": dataset_sha,
        "validation_selected_checkpoint": "best_model.pt",
        "test_windows": len(frame),
        "test_patients": int(patient.summary["number_of_evaluated_cases"]),
        "test_patient_level_mae": float(patient.summary["mae"]["mean"]),
        "test_patient_level_rmse": float(patient.summary["rmse"]["mean"]),
        "pooled_window": pooled,
    }
    patients = patient.case_metrics.copy()
    patients.insert(0, "seed", int(record["seed"]))
    patients.insert(0, "model", str(record["model"]))
    patients.insert(0, "candidate", str(record["candidate"]))
    return metrics, patients


def _run_output_dir(output_dir: Path, record: Mapping[str, Any]) -> Path:
    return (
        output_dir
        / "runs"
        / str(record["candidate"])
        / str(record["model"])
        / f"seed_{int(record['seed'])}"
    )


def _load_or_evaluate_run(
    record: Mapping[str, Any],
    *,
    dataset_dir: Path,
    output_dir: Path,
    dataset_sha: str,
    device: str,
    batch_size: int,
    inference_fn: InferenceFunction,
) -> tuple[dict[str, Any], pd.DataFrame]:
    run_output = _run_output_dir(output_dir, record)
    paths = [run_output / name for name in RUN_ARTIFACTS]
    existing = [path.exists() for path in paths]
    if any(existing) and not all(existing):
        raise ValueError(f"Partial run artifacts cannot be overwritten: {run_output}")
    if all(existing):
        metrics = load_json(paths[1])
        expected = {
            "candidate": record["candidate"],
            "model": record["model"],
            "seed": int(record["seed"]),
            "checkpoint_sha256": record["checkpoint_sha256"],
            "dataset_fingerprint": dataset_sha,
        }
        mismatches = {
            key: (value, metrics.get(key))
            for key, value in expected.items()
            if metrics.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Incompatible completed test run {run_output}: {mismatches}")
        predictions = _validate_predictions(pd.read_csv(paths[0]))
        if metrics.get("predictions_sha256") != sha256_file(paths[0]):
            raise ValueError(f"Completed prediction hash mismatch: {paths[0]}")
        patients = pd.read_csv(paths[2])
        return metrics, patients

    predictions = _validate_predictions(
        inference_fn(record, dataset_dir, device, batch_size)
    )
    metrics, patients = _summarize_prediction(predictions, record, dataset_sha)
    _write_once_csv(paths[0], predictions)
    metrics["predictions_sha256"] = sha256_file(paths[0])
    _write_once_json(paths[1], metrics)
    _write_once_csv(paths[2], patients)
    return metrics, patients


def _run_level_frame(metrics: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in metrics:
        regression = item["pooled_window"]["regression"]
        rows.append(
            {
                "candidate": item["candidate"],
                "model": item["model"],
                "seed": item["seed"],
                "test_patient_level_mae": item["test_patient_level_mae"],
                "test_patient_level_rmse": item["test_patient_level_rmse"],
                "test_pooled_mae": regression["mae"],
                "test_pooled_rmse": regression["rmse"],
                "test_pooled_r_squared": regression["r_squared"],
                "checkpoint_sha256": item["checkpoint_sha256"],
                "dataset_fingerprint": item["dataset_fingerprint"],
            }
        )
    return pd.DataFrame(rows).sort_values(["candidate", "model", "seed"])


def aggregate_test_candidates(run_level: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the prespecified patient-level MAE across five seeds."""

    rows = []
    for (candidate, model), group in run_level.groupby(["candidate", "model"]):
        values = group["test_patient_level_mae"].to_numpy(float)
        rows.append(
            {
                "candidate": candidate,
                "model": model,
                "seed_count": len(values),
                "mean_test_patient_level_mae": float(values.mean()),
                "standard_deviation": float(values.std(ddof=1)),
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
                "mean_test_pooled_mae": float(group["test_pooled_mae"].mean()),
                "mean_test_pooled_rmse": float(group["test_pooled_rmse"].mean()),
                "mean_test_pooled_r_squared": float(
                    group["test_pooled_r_squared"].mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["candidate", "model"])


def paired_test_statistics(
    run_level: pd.DataFrame,
    patients: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute candidate/model seed deltas and patient hierarchical bootstrap CIs."""

    delta_rows: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, Any]] = []
    comparisons: list[tuple[str, str, str, str, str]] = []
    for model in MODELS:
        comparisons.append(
            (
                "candidate",
                model,
                PRIMARY_CANDIDATE,
                REFERENCE_CANDIDATE,
                "strict_consensus_minus_full17_reference",
            )
        )
    for candidate in CANDIDATES:
        comparisons.append(
            ("model", candidate, "attention", "gru", "attention_minus_gru")
        )

    for index, (kind, fixed, left_name, right_name, label) in enumerate(comparisons):
        if kind == "candidate":
            left = run_level.query("candidate == @left_name and model == @fixed")
            right = run_level.query("candidate == @right_name and model == @fixed")
            patient_left = patients.query("candidate == @left_name and model == @fixed")
            patient_right = patients.query("candidate == @right_name and model == @fixed")
        else:
            left = run_level.query("candidate == @fixed and model == @left_name")
            right = run_level.query("candidate == @fixed and model == @right_name")
            patient_left = patients.query("candidate == @fixed and model == @left_name")
            patient_right = patients.query("candidate == @fixed and model == @right_name")
        paired = left[["seed", "test_patient_level_mae"]].merge(
            right[["seed", "test_patient_level_mae"]],
            on="seed",
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        paired["delta"] = (
            paired["test_patient_level_mae_left"]
            - paired["test_patient_level_mae_right"]
        )
        paired["relative_mae_change_percent"] = (
            100.0 * paired["delta"] / paired["test_patient_level_mae_right"]
        )
        paired.insert(0, "comparison", label)
        paired.insert(1, "fixed_group", fixed)
        delta_rows.append(paired)

        patient_pairs = patient_left[["seed", "case_id", "mae"]].merge(
            patient_right[["seed", "case_id", "mae"]],
            on=["seed", "case_id"],
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        patient_pairs["paired_delta"] = patient_pairs["mae_left"] - patient_pairs["mae_right"]
        bootstrap_rows.append(
            {
                "comparison": label,
                "fixed_group": fixed,
                "left": left_name,
                "right": right_name,
                "direction": "negative favors left",
                **hierarchical_paired_bootstrap(
                    patient_pairs[["seed", "case_id", "paired_delta"]],
                    replicates=replicates,
                    seed=seed + index,
                ),
            }
        )
    deltas = pd.concat(delta_rows, ignore_index=True)
    summaries = []
    for (comparison, fixed), group in deltas.groupby(["comparison", "fixed_group"]):
        values = group["delta"].to_numpy(float)
        summaries.append(
            {
                "comparison": comparison,
                "fixed_group": fixed,
                "mean_delta": float(values.mean()),
                "delta_standard_deviation": float(values.std(ddof=1)),
                "median_delta": float(np.median(values)),
                "min_delta": float(values.min()),
                "max_delta": float(values.max()),
                "left_better_seed_count": int((values < 0).sum()),
                "mean_relative_mae_change_percent": float(
                    group["relative_mae_change_percent"].mean()
                ),
                "direction": "negative favors left",
            }
        )
    summary = pd.DataFrame(summaries)
    seed_details = deltas.merge(summary, on=["comparison", "fixed_group"], suffixes=("", "_summary"))
    return seed_details, pd.DataFrame(bootstrap_rows)


def paired_model_test_contrasts(seed_details: pd.DataFrame) -> pd.DataFrame:
    """Return one prespecified Attention-minus-GRU summary per candidate."""

    columns = [
        "comparison",
        "fixed_group",
        "mean_delta",
        "delta_standard_deviation",
        "median_delta",
        "min_delta",
        "max_delta",
        "left_better_seed_count",
        "mean_relative_mae_change_percent",
        "direction",
    ]
    return (
        seed_details.loc[
            seed_details["comparison"] == "attention_minus_gru", columns
        ]
        .drop_duplicates()
        .sort_values("fixed_group")
        .reset_index(drop=True)
    )


def _save_figures(run_level: pd.DataFrame, bootstrap: pd.DataFrame, output_dir: Path) -> list[Path]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(9, 5))
    labels = []
    values = []
    for model in MODELS:
        for candidate in CANDIDATES:
            labels.append(f"{candidate}\n{model}")
            values.append(
                run_level.query("candidate == @candidate and model == @model")[
                    "test_patient_level_mae"
                ].to_numpy(float)
            )
    axis.boxplot(values, showfliers=False)
    for index, group in enumerate(values, start=1):
        axis.scatter(np.full(len(group), index), group, color="#16697a")
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_ylabel("Held-out patient-level MAE")
    axis.set_title("Frozen one-time held-out evaluation")
    figure.tight_layout()
    path = figures_dir / "test_seed_mae.png"
    if path.exists():
        raise ValueError(f"Refusing to overwrite existing test figure: {path}")
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(9, 5))
    y = np.arange(len(bootstrap))
    point = bootstrap["point_estimate_mean_delta"].to_numpy(float)
    lower = bootstrap["percentile_95_ci_lower"].to_numpy(float)
    upper = bootstrap["percentile_95_ci_upper"].to_numpy(float)
    axis.errorbar(point, y, xerr=np.vstack((point - lower, upper - point)), fmt="o")
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_yticks(
        y,
        [f"{row.comparison}: {row.fixed_group}" for row in bootstrap.itertuples()],
    )
    axis.set_xlabel("Paired patient MAE delta")
    axis.set_title("Hierarchical paired bootstrap")
    figure.tight_layout()
    path = figures_dir / "test_bootstrap_contrasts.png"
    if path.exists():
        raise ValueError(f"Refusing to overwrite existing test figure: {path}")
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)
    return paths


def _report(
    aggregate: pd.DataFrame,
    deltas: pd.DataFrame,
    model_contrasts: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> str:
    return f"""# Frozen Predictive Held-Out Test Evaluation

## Prespecified scope
This one-time internal held-out evaluation compares only `strict_consensus` and `full17_reference`, using the validation-selected `best_model.pt` checkpoints for GRU and explicit-attention models across seeds 7, 21, 42, 84, and 123. No model was trained, reselected, or tuned on test data.

The primary endpoint is patient-level MAE (lower is better). The predictive primary remains `strict_consensus` regardless of these test results. `compact_consensus` was validation-only secondary evidence and was not tested.

## Aggregate results
```text
{aggregate.to_string(index=False)}
```

## Paired seed contrasts
Negative deltas favor the left member of each named comparison. These descriptive results are not a p-value winner rule.
```text
{deltas.to_string(index=False)}
```

## Paired model contrasts
Attention-minus-GRU comparisons are paired within the same candidate and seed.
```text
{model_contrasts.to_string(index=False)}
```

## Hierarchical paired bootstrap
The percentile intervals resample paired seeds and paired patients, never windows as independent observations.
```text
{bootstrap.to_string(index=False)}
```

## Interpretation limits
- Test results cannot change the frozen predictive primary or introduce another candidate.
- Small differences must not be overstated as clinically meaningful.
- Predictive utility does not establish an RL-optimal control state.
- `strict_consensus` retains only `rftn_volume` from remifentanil-related features, so the professor's external control baseline must be preserved during RL handoff.
- This is an internal held-out split, not pristine external validation.
"""


def _verify_complete_output(output_dir: Path, manifest: Mapping[str, Any]) -> None:
    if manifest.get("status") != "complete" or manifest.get("run_count") != 20:
        raise ValueError("Existing test manifest is not a complete 20-run evaluation.")
    missing = [name for name in FINAL_TABLES if not (output_dir / name).is_file()]
    if missing:
        raise ValueError(f"Complete test manifest is missing required tables: {missing}")
    for item in manifest.get("generated_output_fingerprints", []):
        path = output_dir / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Completed test output changed or is missing: {path}")


def run_frozen_predictive_test_evaluation(
    *,
    decision_template_dir: Path,
    decision_dir: Path,
    source_analysis_manifest: Path,
    dataset_dir: Path,
    strict_root: Path,
    full17_root: Path,
    output_dir: Path,
    confirmation: str,
    device: str = "auto",
    batch_size: int = 256,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    inference_fn: InferenceFunction = _default_inference,
) -> dict[str, Any]:
    """Execute or resume the single prespecified 20-checkpoint test inference."""

    overall_started = perf_counter()
    if confirmation != CONFIRMATION_PHRASE:
        raise ValueError(f"Exact confirmation required: {CONFIRMATION_PHRASE}")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive.")
    prepare_test_preflight(
        decision_template_dir=decision_template_dir,
        decision_dir=decision_dir,
        source_analysis_manifest=source_analysis_manifest,
        dataset_dir=dataset_dir,
        strict_root=strict_root,
        full17_root=full17_root,
        output_dir=output_dir,
    )
    decision = load_json(output_dir / "frozen_decision_snapshot.json")
    inventory_path = output_dir / "evaluated_checkpoint_inventory.csv"
    inventory = pd.read_csv(inventory_path)
    current_inventory = build_checkpoint_inventory(dataset_dir, strict_root, full17_root)
    if inventory.to_csv(index=False, lineterminator="\n") != current_inventory.to_csv(
        index=False, lineterminator="\n"
    ):
        raise ValueError("Checkpoint inventory changed after preflight; test evaluation stopped.")
    test_fingerprint = full_test_dataset_fingerprint(dataset_dir)
    contract = {
        "decision_snapshot_sha256": sha256_file(
            output_dir / "frozen_decision_snapshot.json"
        ),
        "checkpoint_inventory_sha256": sha256_file(inventory_path),
        "dataset_fingerprint": test_fingerprint,
        "run_count": 20,
        "training_permitted": False,
        "checkpoint_selection_permitted": False,
    }
    _write_once_json(output_dir / "test_evaluation_contract.json", contract)

    manifest_path = output_dir / "test_evaluation_manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest.get("execution_contract") != contract:
            raise ValueError("Existing complete test evaluation has an incompatible contract.")
        _verify_complete_output(output_dir, manifest)
        integrity = verify_test_result_integrity(inventory, output_dir, dataset_dir)
        if manifest.get("integrity_verification") != integrity:
            raise ValueError("Existing test integrity verification no longer matches outputs.")
        LOGGER.info("Verified and skipped complete 20-checkpoint held-out evaluation.")
        return {"status": "complete", "skipped": True, "run_count": 20}

    metrics_rows: list[dict[str, Any]] = []
    patient_frames: list[pd.DataFrame] = []
    records = inventory.to_dict("records")
    for index, record in enumerate(records, start=1):
        run_started = perf_counter()
        LOGGER.info(
            "[%d/20] %s / %s / seed %s: inference verification started",
            index,
            record["candidate"],
            record["model"],
            record["seed"],
        )
        try:
            metrics, patients = _load_or_evaluate_run(
                record,
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                dataset_sha=test_fingerprint["combined_sha256"],
                device=device,
                batch_size=batch_size,
                inference_fn=inference_fn,
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed at [{index}/20] {record['candidate']} / {record['model']} / "
                f"seed {record['seed']}: {error}"
            ) from error
        metrics_rows.append(metrics)
        patient_frames.append(patients)
        LOGGER.info(
            "[%d/20] complete in %.2fs; %d complete, %d remaining",
            index,
            perf_counter() - run_started,
            index,
            20 - index,
        )

    integrity = verify_test_result_integrity(inventory, output_dir, dataset_dir)

    final_paths = [output_dir / name for name in FINAL_TABLES]
    if any(path.exists() for path in final_paths):
        raise ValueError("Final test tables exist without a complete manifest; refusing overwrite.")
    run_level = _run_level_frame(metrics_rows)
    patients = pd.concat(patient_frames, ignore_index=True)
    aggregate = aggregate_test_candidates(run_level)
    LOGGER.info(
        "Hierarchical paired bootstrap started: %d replicates, seed %d",
        bootstrap_replicates,
        bootstrap_seed,
    )
    bootstrap_started = perf_counter()
    deltas, bootstrap = paired_test_statistics(
        run_level,
        patients,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    LOGGER.info(
        "Hierarchical paired bootstrap complete in %.2fs",
        perf_counter() - bootstrap_started,
    )
    model_contrasts = paired_model_test_contrasts(deltas)
    tables = {
        "test_run_level_metrics.csv": run_level,
        "test_candidate_aggregate.csv": aggregate,
        "paired_test_seed_deltas.csv": deltas,
        "patient_level_test_metrics.csv": patients,
        "hierarchical_bootstrap_test_contrasts.csv": bootstrap,
        "paired_model_test_contrasts.csv": model_contrasts,
    }
    for name, frame in tables.items():
        _write_once_csv(output_dir / name, frame)
    figures = _save_figures(run_level, bootstrap, output_dir)
    report_path = output_dir / "frozen_predictive_test_report.md"
    _write_once_text(
        report_path, _report(aggregate, deltas, model_contrasts, bootstrap)
    )

    after_inventory = build_checkpoint_inventory(dataset_dir, strict_root, full17_root)
    if current_inventory.to_csv(index=False) != after_inventory.to_csv(index=False):
        raise ValueError("Source checkpoints or configs changed during test inference.")
    generated = [
        *(output_dir / name for name in tables),
        report_path,
        *figures,
        *(path for path in output_dir.glob("runs/**/*") if path.is_file()),
    ]
    manifest = {
        "status": "complete",
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_git_commit": _git_head(),
        "run_count": 20,
        "candidates": list(CANDIDATES),
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "primary_metric": "test_patient_level_mae",
        "primary_metric_direction": "lower is better",
        "primary_candidate_remains_frozen": PRIMARY_CANDIDATE,
        "compact_consensus_tested": False,
        "training_performed": False,
        "checkpoint_reselection_performed": False,
        "internal_heldout_not_external_validation": True,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_resampling_units": "paired seed and held-out patient",
        "execution_contract": contract,
        "integrity_verification": integrity,
        "input_fingerprints": [
            {
                "path": "frozen_decision_snapshot.json",
                "sha256": sha256_file(output_dir / "frozen_decision_snapshot.json"),
            },
            {
                "path": "evaluated_checkpoint_inventory.csv",
                "sha256": sha256_file(inventory_path),
            },
            {
                "path": str(source_analysis_manifest),
                "sha256": sha256_file(source_analysis_manifest),
            },
        ],
        "generated_output_fingerprints": [
            {
                "path": str(path.relative_to(output_dir)).replace("\\", "/"),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(generated)
        ],
    }
    _write_once_json(manifest_path, manifest)
    LOGGER.info(
        "One-time 20-checkpoint held-out evaluation complete in %.2fs",
        perf_counter() - overall_started,
    )
    return {"status": "complete", "skipped": False, "run_count": 20}


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
