"""Training and aligned attention extraction for the factorized-attention GRU."""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets import VitalBISDataset
from src.metrics import (
    patient_level_evaluation,
    regression_metrics,
    select_validation_thresholds,
)
from src.models.attention import FactorizedAttentionGRU, FactorizedAttentionOutput
from src.models.baselines import count_trainable_parameters
from src.training import (
    PredictionBundle,
    TrainingConfig,
    _checkpoint_payload,
    _git_commit_hash,
    _json_ready_config,
    _load_checkpoint,
    _save_json,
    _selected_indices,
    evaluate_bundle,
    make_data_loader,
    predict_model,
    prediction_frame,
    resolve_device,
    set_deterministic_seed,
    train_epoch,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttentionTrainingConfig(TrainingConfig):
    """Configuration for factorized feature/temporal-attention training."""

    feature_token_embedding_dim: int = 16
    static_context_dim: int = 16


@dataclass(frozen=True)
class AttentionBundle:
    """Attention arrays and row identifiers collected from an evaluation loader."""

    sample_indices: np.ndarray
    case_ids: np.ndarray
    feature_attention: np.ndarray
    temporal_attention: np.ndarray
    combined_attention: np.ndarray
    feature_normalization_max_error: float
    temporal_normalization_max_error: float
    combined_normalization_max_error: float
    maximum_missing_feature_weight: float


@dataclass(frozen=True)
class JointEvaluationTiming:
    """Timing for a single pass that collects predictions and attention."""

    total_seconds: float
    shared_model_forward_seconds: float
    prediction_collection_seconds: float
    attention_collection_seconds: float


def _fresh_attention_model(
    config: AttentionTrainingConfig, dataset: VitalBISDataset
) -> FactorizedAttentionGRU:
    return FactorizedAttentionGRU(
        dynamic_feature_count=len(dataset.dynamic_feature_names),
        static_feature_count=len(dataset.static_feature_names),
        history_steps=int(dataset.dataset_metadata["history_steps"]),
        feature_token_embedding_dim=config.feature_token_embedding_dim,
        static_context_dim=config.static_context_dim,
        hidden_size=config.hidden_size,
        prediction_hidden_size=config.prediction_hidden_size,
        dropout=config.dropout,
    )


def _move_attention_inputs(
    batch: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["X_dynamic"].to(device=device, dtype=torch.float32),
        batch["X_static"].to(device=device, dtype=torch.float32),
        batch["observation_mask"].to(device=device),
    )


@torch.no_grad()
def extract_attention(
    model: FactorizedAttentionGRU,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> AttentionBundle:
    """Extract attention in loader order and retain exact sample/case alignment."""

    model.eval()
    sample_indices: list[np.ndarray] = []
    case_ids: list[np.ndarray] = []
    feature_weights: list[np.ndarray] = []
    temporal_weights: list[np.ndarray] = []
    combined_weights: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for batch in loader:
        X_dynamic, X_static, mask = _move_attention_inputs(batch, device)
        output = model(X_dynamic, X_static, mask, return_attention=True)
        if not isinstance(output, FactorizedAttentionOutput):
            raise TypeError("Attention model did not return FactorizedAttentionOutput.")
        sample_indices.append(batch["sample_index"].numpy())
        case_ids.append(batch["case_id"].numpy())
        feature_weights.append(output.feature_attention.detach().cpu().numpy())
        temporal_weights.append(output.temporal_attention.detach().cpu().numpy())
        combined_weights.append(output.combined_attention.detach().cpu().numpy())
        masks.append(mask.detach().cpu().numpy().astype(bool, copy=False))

    feature = np.concatenate(feature_weights)
    temporal = np.concatenate(temporal_weights)
    combined = np.concatenate(combined_weights)
    observed_mask = np.concatenate(masks)
    arrays = (feature, temporal, combined)
    if not all(np.isfinite(array).all() for array in arrays):
        raise FloatingPointError("Extracted attention contains NaN or infinite values.")
    return AttentionBundle(
        sample_indices=np.concatenate(sample_indices).astype(np.int64),
        case_ids=np.concatenate(case_ids).astype(np.int64),
        feature_attention=feature,
        temporal_attention=temporal,
        combined_attention=combined,
        feature_normalization_max_error=float(
            np.max(np.abs(feature.sum(axis=2) - 1.0))
        ),
        temporal_normalization_max_error=float(
            np.max(np.abs(temporal.sum(axis=1) - 1.0))
        ),
        combined_normalization_max_error=float(
            np.max(np.abs(combined.sum(axis=(1, 2)) - 1.0))
        ),
        maximum_missing_feature_weight=float(
            np.max(np.abs(feature[~observed_mask]), initial=0.0)
        ),
    )


@torch.no_grad()
def predict_and_extract_attention(
    model: FactorizedAttentionGRU,
    loader: DataLoader[dict[str, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[PredictionBundle, AttentionBundle, JointEvaluationTiming]:
    """Collect predictions and attention together without a repeated loader pass."""

    model.eval()
    started = perf_counter()
    forward_seconds = 0.0
    prediction_collection_seconds = 0.0
    attention_collection_seconds = 0.0
    predicted: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    case_ids: list[np.ndarray] = []
    target_timestamps: list[np.ndarray] = []
    feature_weights: list[np.ndarray] = []
    temporal_weights: list[np.ndarray] = []
    combined_weights: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        X_dynamic, X_static, mask = _move_attention_inputs(batch, device)
        target = batch["y_bis"].to(device=device, dtype=torch.float32)
        forward_started = perf_counter()
        output = model(X_dynamic, X_static, mask, return_attention=True)
        forward_seconds += perf_counter() - forward_started
        if not isinstance(output, FactorizedAttentionOutput):
            raise TypeError("Attention model did not return FactorizedAttentionOutput.")

        prediction_started = perf_counter()
        loss = criterion(output.prediction, target)
        predicted.append(output.prediction.detach().cpu().numpy())
        observed.append(target.detach().cpu().numpy())
        sample_indices.append(batch["sample_index"].numpy())
        case_ids.append(batch["case_id"].numpy())
        target_timestamps.append(batch["target_timestamp"].numpy())
        total_loss += float(loss) * len(target)
        total_samples += len(target)
        prediction_collection_seconds += perf_counter() - prediction_started

        attention_started = perf_counter()
        feature_weights.append(output.feature_attention.detach().cpu().numpy())
        temporal_weights.append(output.temporal_attention.detach().cpu().numpy())
        combined_weights.append(output.combined_attention.detach().cpu().numpy())
        masks.append(mask.detach().cpu().numpy().astype(bool, copy=False))
        attention_collection_seconds += perf_counter() - attention_started

    feature = np.concatenate(feature_weights)
    temporal = np.concatenate(temporal_weights)
    combined = np.concatenate(combined_weights)
    observed_mask = np.concatenate(masks)
    prediction_bundle = PredictionBundle(
        y_true=np.concatenate(observed),
        y_pred=np.concatenate(predicted),
        case_ids=np.concatenate(case_ids).astype(np.int64),
        target_timestamps=np.concatenate(target_timestamps).astype(np.int64),
        sample_indices=np.concatenate(sample_indices).astype(np.int64),
        mean_loss=total_loss / total_samples,
    )
    attention_bundle = AttentionBundle(
        sample_indices=prediction_bundle.sample_indices,
        case_ids=prediction_bundle.case_ids,
        feature_attention=feature,
        temporal_attention=temporal,
        combined_attention=combined,
        feature_normalization_max_error=float(
            np.max(np.abs(feature.sum(axis=2) - 1.0))
        ),
        temporal_normalization_max_error=float(
            np.max(np.abs(temporal.sum(axis=1) - 1.0))
        ),
        combined_normalization_max_error=float(
            np.max(np.abs(combined.sum(axis=(1, 2)) - 1.0))
        ),
        maximum_missing_feature_weight=float(
            np.max(np.abs(feature[~observed_mask]), initial=0.0)
        ),
    )
    arrays = (
        prediction_bundle.y_pred,
        feature,
        temporal,
        combined,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise FloatingPointError("Joint prediction/attention output is not finite.")
    timing = JointEvaluationTiming(
        total_seconds=perf_counter() - started,
        shared_model_forward_seconds=forward_seconds,
        prediction_collection_seconds=prediction_collection_seconds,
        attention_collection_seconds=attention_collection_seconds,
    )
    return prediction_bundle, attention_bundle, timing


def _verify_attention_checkpoint_reload(
    model: FactorizedAttentionGRU,
    checkpoint_path: Path,
    loader: DataLoader[dict[str, torch.Tensor]],
    config: AttentionTrainingConfig,
    dataset: VitalBISDataset,
    device: torch.device,
) -> tuple[FactorizedAttentionGRU, bool, bool]:
    reloaded = _fresh_attention_model(config, dataset).to(device)
    _load_checkpoint(checkpoint_path, reloaded, optimizer=None, device=device)
    batch = next(iter(loader))
    X_dynamic, X_static, mask = _move_attention_inputs(batch, device)
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        original = model(X_dynamic, X_static, mask, return_attention=True)
        restored = reloaded(X_dynamic, X_static, mask, return_attention=True)
    if not isinstance(original, FactorizedAttentionOutput) or not isinstance(
        restored, FactorizedAttentionOutput
    ):
        raise TypeError("Checkpoint verification requires structured attention output.")
    prediction_identical = bool(torch.equal(original.prediction, restored.prediction))
    attention_identical = all(
        torch.equal(getattr(original, name), getattr(restored, name))
        for name in (
            "feature_attention",
            "temporal_attention",
            "combined_attention",
        )
    )
    return reloaded, prediction_identical, attention_identical


def _attention_metadata(
    dataset: VitalBISDataset,
    bundles: dict[str, AttentionBundle],
    best_epoch: int,
    reload_attention_identical: bool,
    runtime_breakdown: dict[str, Any],
) -> dict[str, Any]:
    history_steps = int(dataset.dataset_metadata["history_steps"])
    interval = int(dataset.dataset_metadata["resampling_interval_seconds"])
    time_lags = [-(history_steps - 1 - index) * interval for index in range(history_steps)]
    split_metadata = {
        split: {
            "feature_attention_shape": list(bundle.feature_attention.shape),
            "temporal_attention_shape": list(bundle.temporal_attention.shape),
            "combined_attention_shape": list(bundle.combined_attention.shape),
            "feature_attention_normalization_max_absolute_error": (
                bundle.feature_normalization_max_error
            ),
            "temporal_attention_normalization_max_absolute_error": (
                bundle.temporal_normalization_max_error
            ),
            "combined_attention_normalization_max_absolute_error": (
                bundle.combined_normalization_max_error
            ),
            "maximum_missing_feature_attention_weight": (
                bundle.maximum_missing_feature_weight
            ),
            "all_attention_values_finite": True,
        }
        for split, bundle in bundles.items()
    }
    validation = split_metadata["val"]
    return {
        "dynamic_feature_names": list(dataset.dynamic_feature_names),
        "time_lags_seconds": time_lags,
        "feature_attention_shape": validation["feature_attention_shape"],
        "temporal_attention_shape": validation["temporal_attention_shape"],
        "combined_attention_shape": validation["combined_attention_shape"],
        "feature_attention_normalization_dimension": "P, dynamic features (axis 2)",
        "temporal_attention_normalization_dimension": "L, historical time (axis 1)",
        "combined_attention_definition": (
            "temporal_attention[:, :, None] * feature_attention"
        ),
        "combined_attention_interpretation": (
            "factorized model-importance weight, not a causal effect"
        ),
        "model_checkpoint_identifier": f"best_model.pt:epoch_{best_epoch}",
        "checkpoint_reload_attention_identical": reload_attention_identical,
        "feature_attention_normalization_max_absolute_error": (
            validation["feature_attention_normalization_max_absolute_error"]
        ),
        "temporal_attention_normalization_max_absolute_error": (
            validation["temporal_attention_normalization_max_absolute_error"]
        ),
        "combined_attention_normalization_max_absolute_error": (
            validation["combined_attention_normalization_max_absolute_error"]
        ),
        "maximum_missing_feature_attention_weight": (
            validation["maximum_missing_feature_attention_weight"]
        ),
        "all_attention_values_finite": True,
        "splits": split_metadata,
        "runtime_breakdown": runtime_breakdown,
    }


def _save_attention_bundle(
    bundle: AttentionBundle, output_dir: Path, split: str
) -> None:
    np.savez_compressed(
        output_dir / f"{split}_attention.npz",
        sample_index=bundle.sample_indices,
        case_id=bundle.case_ids,
        feature_attention=bundle.feature_attention,
        temporal_attention=bundle.temporal_attention,
        combined_attention=bundle.combined_attention,
    )


def run_attention_training(config: AttentionTrainingConfig) -> dict[str, Any]:
    """Train, select, reload, evaluate, and extract explicit attention outputs."""

    run_started = perf_counter()
    checkpoint_save_seconds = 0.0
    serialization_seconds = 0.0
    set_deterministic_seed(config.seed)
    device = torch.device("cpu") if config.smoke else resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = VitalBISDataset(config.dataset_dir, "train")
    val_dataset = VitalBISDataset(config.dataset_dir, "val")
    train_indices = _selected_indices(train_dataset, 4 if config.smoke else None)
    val_indices = _selected_indices(val_dataset, 3 if config.smoke else None)
    train_loader = make_data_loader(
        train_dataset,
        train_indices,
        config.batch_size,
        config.seed,
        training=True,
        case_balanced=config.case_balanced_sampling,
        num_workers=config.num_workers,
    )
    val_loader = make_data_loader(
        val_dataset,
        val_indices,
        config.batch_size,
        config.seed,
        training=False,
        case_balanced=False,
        num_workers=config.num_workers,
    )

    model = _fresh_attention_model(config, train_dataset).to(device)
    parameter_count = count_trainable_parameters(model)
    if parameter_count >= 100_000:
        LOGGER.warning("Attention model exceeds preferred 100,000 parameters: %d", parameter_count)
    LOGGER.info("Factorized-attention GRU trainable parameters: %d", parameter_count)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    start_epoch = 1
    best_patient_mae = float("inf")
    history: list[dict[str, float | int]] = []
    if config.resume_checkpoint is not None:
        checkpoint = _load_checkpoint(
            config.resume_checkpoint, model, optimizer=optimizer, device=device
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_patient_mae = float(checkpoint["best_validation_patient_mae"])
        history = list(checkpoint.get("history", []))

    resolved_config = {
        **_json_ready_config(config),
        "model_name": "FactorizedAttentionGRU",
        "resolved_device": str(device),
        "model_parameter_count": parameter_count,
        "dynamic_feature_names": list(train_dataset.dynamic_feature_names),
        "static_feature_names": list(train_dataset.static_feature_names),
        "selected_training_cases": sorted(
            np.unique(train_dataset.case_ids[train_indices]).astype(int).tolist()
        ),
        "selected_validation_cases": sorted(
            np.unique(val_dataset.case_ids[val_indices]).astype(int).tolist()
        ),
        "package_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pytorch": torch.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "git_commit_hash": _git_commit_hash(),
    }
    serialization_started = perf_counter()
    _save_json(resolved_config, config.output_dir / "config.json")
    serialization_seconds += perf_counter() - serialization_started

    epochs_without_improvement = 0
    best_path = config.output_dir / "best_model.pt"
    last_path = config.output_dir / "last_model.pt"
    max_epochs = min(config.max_epochs, 2) if config.smoke else config.max_epochs
    for epoch in range(start_epoch, max_epochs + 1):
        training_started = perf_counter()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            config.gradient_clip_norm,
        )
        training_seconds = perf_counter() - training_started
        validation_started = perf_counter()
        val_bundle = predict_model(model, val_loader, criterion, device)
        patient = patient_level_evaluation(
            val_bundle.y_true, val_bundle.y_pred, val_bundle.case_ids
        )
        validation_patient_mae = float(patient.summary["mae"]["mean"])
        validation_regression = regression_metrics(
            val_bundle.y_true, val_bundle.y_pred
        )
        validation_seconds = perf_counter() - validation_started
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_bundle.mean_loss,
                "validation_pooled_mae": float(validation_regression["mae"]),
                "validation_rmse": float(validation_regression["rmse"]),
                "validation_patient_level_mae": validation_patient_mae,
                "training_time_seconds": training_seconds,
                "validation_evaluation_time_seconds": validation_seconds,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if validation_patient_mae < best_patient_mae:
            best_patient_mae = validation_patient_mae
            epochs_without_improvement = 0
            checkpoint_started = perf_counter()
            torch.save(
                _checkpoint_payload(
                    model, optimizer, epoch, best_patient_mae, history, config
                ),
                best_path,
            )
            checkpoint_save_seconds += perf_counter() - checkpoint_started
        else:
            epochs_without_improvement += 1
        checkpoint_started = perf_counter()
        torch.save(
            _checkpoint_payload(model, optimizer, epoch, best_patient_mae, history, config),
            last_path,
        )
        checkpoint_save_seconds += perf_counter() - checkpoint_started
        serialization_started = perf_counter()
        pd.DataFrame(history).to_csv(
            config.output_dir / "training_history.csv", index=False
        )
        serialization_seconds += perf_counter() - serialization_started
        LOGGER.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_patient_mae=%.4f",
            epoch,
            train_loss,
            val_bundle.mean_loss,
            validation_patient_mae,
        )
        if epochs_without_improvement >= config.patience:
            LOGGER.info("Early stopping after epoch %d", epoch)
            break

    if not best_path.exists():
        raise RuntimeError("Training completed without creating a best checkpoint.")
    checkpoint_load_started = perf_counter()
    checkpoint = _load_checkpoint(best_path, model, optimizer=None, device=device)
    best_epoch = int(checkpoint["epoch"])
    reloaded, prediction_identical, attention_identical = (
        _verify_attention_checkpoint_reload(
            model, best_path, val_loader, config, train_dataset, device
        )
    )
    checkpoint_load_seconds = perf_counter() - checkpoint_load_started
    if not prediction_identical or not attention_identical:
        raise AssertionError("Reloaded checkpoint predictions or attention differ.")

    val_bundle, val_attention, val_joint_timing = predict_and_extract_attention(
        reloaded, val_loader, criterion, device
    )
    thresholds = select_validation_thresholds(val_bundle.y_true, val_bundle.y_pred)
    val_metrics, val_case_metrics = evaluate_bundle(val_bundle, thresholds)
    val_metrics["checkpoint_reload_predictions_identical"] = prediction_identical
    val_metrics["checkpoint_reload_attention_identical"] = attention_identical
    val_metrics["model_parameter_count"] = parameter_count
    serialization_started = perf_counter()
    prediction_frame(val_bundle).to_csv(
        config.output_dir / "val_predictions.csv", index=False
    )
    val_case_metrics.insert(0, "split", "val")
    val_case_metrics.to_csv(config.output_dir / "case_metrics.csv", index=False)
    _save_json(val_metrics, config.output_dir / "val_metrics.json")
    _save_attention_bundle(val_attention, config.output_dir, "val")
    serialization_seconds += perf_counter() - serialization_started
    attention_bundles = {"val": val_attention}
    joint_timings = {"val": val_joint_timing}
    test_metrics: dict[str, Any] | None = None

    if not config.smoke:
        test_dataset = VitalBISDataset(config.dataset_dir, "test")
        test_indices = _selected_indices(test_dataset, None)
        test_loader = make_data_loader(
            test_dataset,
            test_indices,
            config.batch_size,
            config.seed,
            training=False,
            case_balanced=False,
            num_workers=config.num_workers,
        )
        test_bundle, test_attention, test_joint_timing = predict_and_extract_attention(
            reloaded, test_loader, criterion, device
        )
        test_metrics, test_case_metrics = evaluate_bundle(test_bundle, thresholds)
        test_metrics["checkpoint_reload_predictions_identical"] = prediction_identical
        test_metrics["checkpoint_reload_attention_identical"] = attention_identical
        test_metrics["model_parameter_count"] = parameter_count
        serialization_started = perf_counter()
        prediction_frame(test_bundle).to_csv(
            config.output_dir / "test_predictions.csv", index=False
        )
        _save_attention_bundle(test_attention, config.output_dir, "test")
        test_case_metrics.insert(0, "split", "test")
        pd.concat((val_case_metrics, test_case_metrics), ignore_index=True).to_csv(
            config.output_dir / "case_metrics.csv", index=False
        )
        _save_json(test_metrics, config.output_dir / "test_metrics.json")
        serialization_seconds += perf_counter() - serialization_started
        attention_bundles["test"] = test_attention
        joint_timings["test"] = test_joint_timing

    completed_epochs = len(history)
    runtime_breakdown = {
        "total_internal_runtime_seconds": perf_counter() - run_started,
        "completed_epochs": completed_epochs,
        "training_time_seconds": float(
            sum(float(row["training_time_seconds"]) for row in history)
        ),
        "mean_training_time_per_completed_epoch_seconds": float(
            np.mean([float(row["training_time_seconds"]) for row in history])
        ),
        "validation_evaluation_time_per_epoch_seconds": [
            float(row["validation_evaluation_time_seconds"]) for row in history
        ],
        "mean_validation_evaluation_time_per_epoch_seconds": float(
            np.mean(
                [float(row["validation_evaluation_time_seconds"]) for row in history]
            )
        ),
        "checkpoint_save_time_seconds": checkpoint_save_seconds,
        "checkpoint_load_and_reload_verification_time_seconds": (
            checkpoint_load_seconds
        ),
        "serialization_time_seconds": serialization_seconds,
        "final_joint_prediction_attention_passes": {
            split: {
                "total_seconds": timing.total_seconds,
                "shared_model_forward_seconds": timing.shared_model_forward_seconds,
                "prediction_collection_seconds": timing.prediction_collection_seconds,
                "attention_collection_seconds": timing.attention_collection_seconds,
            }
            for split, timing in joint_timings.items()
        },
        "repeated_final_dataset_pass_avoided": True,
        "peak_memory": None,
        "peak_memory_note": "not measured; no memory profiler was added",
    }
    attention_metadata = _attention_metadata(
        val_dataset,
        attention_bundles,
        best_epoch,
        attention_identical,
        runtime_breakdown,
    )
    runtime_breakdown["total_internal_runtime_seconds"] = perf_counter() - run_started
    _save_json(attention_metadata, config.output_dir / "attention_metadata.json")

    result: dict[str, Any] = {
        "output_dir": str(config.output_dir),
        "parameter_count": parameter_count,
        "device": str(device),
        "train_case_count": int(len(np.unique(train_dataset.case_ids[train_indices]))),
        "validation_case_count": int(len(np.unique(val_dataset.case_ids[val_indices]))),
        "train_tensor_shape": [
            len(train_indices),
            int(train_dataset.dataset_metadata["history_steps"]),
            len(train_dataset.dynamic_feature_names),
        ],
        "validation_tensor_shape": [
            len(val_indices),
            int(val_dataset.dataset_metadata["history_steps"]),
            len(val_dataset.dynamic_feature_names),
        ],
        "attention_shapes": {
            "feature_attention": list(val_attention.feature_attention.shape),
            "temporal_attention": list(val_attention.temporal_attention.shape),
            "combined_attention": list(val_attention.combined_attention.shape),
        },
        "checkpoint_reload_predictions_identical": prediction_identical,
        "checkpoint_reload_attention_identical": attention_identical,
        "attention_validation": attention_metadata,
        "runtime_breakdown": runtime_breakdown,
        "validation_metrics": val_metrics,
    }
    if test_metrics is not None:
        result["test_metrics"] = test_metrics
    return result
