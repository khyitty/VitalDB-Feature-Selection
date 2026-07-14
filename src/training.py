"""Reproducible training and evaluation for non-attention BIS baselines."""

from __future__ import annotations

import json
import logging
import platform
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.datasets import (
    VitalBISDataset,
    make_case_balanced_sampler,
    seed_worker,
)
from src.metrics import (
    patient_level_evaluation,
    pooled_evaluation,
    regression_metrics,
    select_validation_thresholds,
)
from src.models.baselines import (
    GRUBaseline,
    PersistenceBaseline,
    count_trainable_parameters,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for compact GRU regression training."""

    dataset_dir: Path
    output_dir: Path
    seed: int = 42
    device: str = "auto"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 50
    patience: int = 8
    gradient_clip_norm: float = 1.0
    hidden_size: int = 64
    projection_size: int = 64
    static_hidden_size: int = 16
    prediction_hidden_size: int = 32
    dropout: float = 0.0
    case_balanced_sampling: bool = True
    num_workers: int = 0
    smoke: bool = False
    resume_checkpoint: Path | None = None


@dataclass(frozen=True)
class PredictionBundle:
    """Predictions and aligned metadata collected from one DataLoader."""

    y_true: np.ndarray
    y_pred: np.ndarray
    case_ids: np.ndarray
    target_timestamps: np.ndarray
    sample_indices: np.ndarray
    mean_loss: float


def set_deterministic_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch CPU/CUDA deterministically."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda with a clear CPU fallback."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA was requested but is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return device


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_ready_config(config: TrainingConfig) -> dict[str, Any]:
    values = asdict(config)
    for key in ("dataset_dir", "output_dir", "resume_checkpoint"):
        if values[key] is not None:
            values[key] = str(values[key])
    return values


def _save_json(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")


def _selected_indices(dataset: VitalBISDataset, case_limit: int | None) -> np.ndarray:
    if case_limit is None:
        return np.arange(len(dataset), dtype=np.int64)
    selected_cases = sorted(np.unique(dataset.case_ids).tolist())[:case_limit]
    return dataset.indices_for_cases(selected_cases)


def make_data_loader(
    dataset: VitalBISDataset,
    indices: np.ndarray,
    batch_size: int,
    seed: int,
    training: bool,
    case_balanced: bool,
    num_workers: int = 0,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Construct deterministic train or exhaustive evaluation loaders."""

    subset = Subset(dataset, indices.tolist())
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    shuffle = False
    if training and case_balanced:
        sampler = make_case_balanced_sampler(dataset.case_ids[indices], seed=seed)
    elif training:
        shuffle = True
    return DataLoader(
        subset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def _move_inputs(
    batch: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["X_dynamic"].to(device=device, dtype=torch.float32),
        batch["X_static"].to(device=device, dtype=torch.float32),
        batch["observation_mask"].to(device=device),
        batch["y_bis"].to(device=device, dtype=torch.float32),
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    """Train for one sampled epoch and return mean window loss."""

    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        X_dynamic, X_static, mask, target = _move_inputs(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(X_dynamic, X_static, mask)
        loss = criterion(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss became NaN or infinite.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        total_loss += float(loss.detach()) * len(target)
        total_samples += len(target)
    return total_loss / total_samples


@torch.no_grad()
def predict_model(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> PredictionBundle:
    """Evaluate every loader item exactly once and retain aligned metadata."""

    model.eval()
    predicted: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    case_ids: list[np.ndarray] = []
    target_timestamps: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        X_dynamic, X_static, mask, target = _move_inputs(batch, device)
        prediction = model(X_dynamic, X_static, mask)
        loss = criterion(prediction, target)
        predicted.append(prediction.cpu().numpy())
        observed.append(target.cpu().numpy())
        case_ids.append(batch["case_id"].numpy())
        target_timestamps.append(batch["target_timestamp"].numpy())
        sample_indices.append(batch["sample_index"].numpy())
        total_loss += float(loss) * len(target)
        total_samples += len(target)
    return PredictionBundle(
        y_true=np.concatenate(observed),
        y_pred=np.concatenate(predicted),
        case_ids=np.concatenate(case_ids).astype(np.int64),
        target_timestamps=np.concatenate(target_timestamps).astype(np.int64),
        sample_indices=np.concatenate(sample_indices).astype(np.int64),
        mean_loss=total_loss / total_samples,
    )


def prediction_frame(bundle: PredictionBundle) -> pd.DataFrame:
    """Create a stable, model-independent prediction table."""

    absolute_error = np.abs(bundle.y_pred - bundle.y_true)
    region = np.select(
        [bundle.y_true < 40.0, bundle.y_true > 60.0],
        ["bis_below_40", "bis_above_60"],
        default="bis_40_to_60",
    )
    return pd.DataFrame(
        {
            "sample_index": bundle.sample_indices,
            "case_id": bundle.case_ids,
            "target_timestamp": bundle.target_timestamps,
            "observed_future_bis": bundle.y_true,
            "predicted_future_bis": bundle.y_pred,
            "absolute_error": absolute_error,
            "squared_error": np.square(bundle.y_pred - bundle.y_true),
            "bis_region": region,
            "high_bis_label": (bundle.y_true > 60.0).astype(np.int8),
            "low_bis_label": (bundle.y_true < 40.0).astype(np.int8),
        }
    )


def evaluate_bundle(
    bundle: PredictionBundle,
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate pooled windows and equally weighted patient-level metrics."""

    patient = patient_level_evaluation(
        bundle.y_true, bundle.y_pred, bundle.case_ids
    )
    metrics = {
        "pooled_window": pooled_evaluation(bundle.y_true, bundle.y_pred, thresholds),
        "patient_level": patient.summary,
        "mean_huber_loss": bundle.mean_loss,
        "thresholds_selected_on_validation": thresholds,
    }
    return metrics, patient.case_metrics


def _persistence_bundle(
    dataset: VitalBISDataset, model: PersistenceBaseline
) -> PredictionBundle:
    prediction = model.predict(dataset.arrays["X_dynamic"])
    return PredictionBundle(
        y_true=dataset.arrays["y_bis"].astype(np.float32, copy=False),
        y_pred=prediction,
        case_ids=dataset.case_ids,
        target_timestamps=dataset.metadata["target_timestamp"].to_numpy(
            dtype=np.int64, copy=False
        ),
        sample_indices=np.arange(len(dataset), dtype=np.int64),
        mean_loss=float(
            np.mean(
                np.where(
                    np.abs(prediction - dataset.arrays["y_bis"]) < 1.0,
                    0.5 * np.square(prediction - dataset.arrays["y_bis"]),
                    np.abs(prediction - dataset.arrays["y_bis"]) - 0.5,
                )
            )
        ),
    )


def run_persistence_baseline(dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Evaluate latest-observed-BIS persistence on validation and test."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = dataset_dir / "preprocessing_statistics.csv"
    if not stats_path.exists():
        raise FileNotFoundError(f"Expected preprocessing statistics are missing: {stats_path}")
    stats = pd.read_csv(stats_path).set_index("feature_name")
    if "bis" not in stats.index:
        raise ValueError("Preprocessing statistics do not contain the BIS feature.")
    val_dataset = VitalBISDataset(dataset_dir, "val")
    test_dataset = VitalBISDataset(dataset_dir, "test")
    model = PersistenceBaseline.from_feature_metadata(
        val_dataset.dynamic_feature_names,
        training_mean=float(stats.loc["bis", "training_mean"]),
        training_standard_deviation=float(
            stats.loc["bis", "training_standard_deviation"]
        ),
    )
    val_bundle = _persistence_bundle(val_dataset, model)
    test_bundle = _persistence_bundle(test_dataset, model)
    thresholds = select_validation_thresholds(val_bundle.y_true, val_bundle.y_pred)
    val_metrics, _ = evaluate_bundle(val_bundle, thresholds)
    test_metrics, _ = evaluate_bundle(test_bundle, thresholds)
    prediction_frame(val_bundle).to_csv(output_dir / "val_predictions.csv", index=False)
    prediction_frame(test_bundle).to_csv(output_dir / "test_predictions.csv", index=False)
    payload = {
        "baseline": "persistence_latest_historical_bis",
        "bis_feature_index_found_by_name": model.bis_feature_index,
        "inverse_normalization": {
            "training_mean": model.training_mean,
            "training_standard_deviation": model.training_standard_deviation,
        },
        "thresholds_selected_on_validation": thresholds,
        "validation": val_metrics,
        "test": test_metrics,
    }
    _save_json(payload, output_dir / "metrics.json")
    return payload


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_patient_mae: float,
    history: list[dict[str, float | int]],
    config: TrainingConfig,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_patient_mae": best_patient_mae,
        "history": history,
        "config": _json_ready_config(config),
    }


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def _remifentanil_diagnostics(
    dataset: VitalBISDataset,
    bundle: PredictionBundle,
    case_metrics: pd.DataFrame,
) -> dict[str, Any]:
    names = ("rftn_rate", "rftn_volume", "rftn_cp", "rftn_ce")
    feature_indices = [dataset.dynamic_feature_names.index(name) for name in names]
    diagnostics: dict[str, Any] = {}
    for case_id in (97, 154):
        positions = np.flatnonzero(bundle.case_ids == case_id)
        sample_indices = bundle.sample_indices[positions]
        if not len(positions):
            diagnostics[str(case_id)] = {"included": False}
            continue
        mask = np.take(
            dataset.arrays["observation_mask"][sample_indices], feature_indices, axis=2
        )
        case_row = case_metrics[case_metrics["case_id"] == case_id]
        diagnostics[str(case_id)] = {
            "included": True,
            "number_of_windows": len(positions),
            "remifentanil_feature_names": list(names),
            "all_remifentanil_observation_masks_zero": bool(~mask.any()),
            "all_predictions_finite": bool(np.isfinite(bundle.y_pred[positions]).all()),
            "patient_metrics_reported": len(case_row) == 1,
            "mae": float(case_row.iloc[0]["mae"]),
            "rmse": float(case_row.iloc[0]["rmse"]),
        }
    return diagnostics


def _fresh_model(config: TrainingConfig, dataset: VitalBISDataset) -> GRUBaseline:
    return GRUBaseline(
        dynamic_feature_count=len(dataset.dynamic_feature_names),
        static_feature_count=len(dataset.static_feature_names),
        hidden_size=config.hidden_size,
        projection_size=config.projection_size,
        static_hidden_size=config.static_hidden_size,
        prediction_hidden_size=config.prediction_hidden_size,
        dropout=config.dropout,
    )


def _verify_checkpoint_reload(
    model: GRUBaseline,
    checkpoint_path: Path,
    loader: DataLoader[dict[str, torch.Tensor]],
    config: TrainingConfig,
    dataset: VitalBISDataset,
    device: torch.device,
) -> tuple[GRUBaseline, bool]:
    reloaded = _fresh_model(config, dataset).to(device)
    _load_checkpoint(checkpoint_path, reloaded, optimizer=None, device=device)
    first_batch = next(iter(loader))
    X_dynamic, X_static, mask, _ = _move_inputs(first_batch, device)
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        original = model(X_dynamic, X_static, mask)
        restored = reloaded(X_dynamic, X_static, mask)
    return reloaded, bool(torch.equal(original, restored))


def run_gru_training(config: TrainingConfig) -> dict[str, Any]:
    """Train, select, reload, and evaluate the compact GRU baseline."""

    set_deterministic_seed(config.seed)
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = VitalBISDataset(config.dataset_dir, "train")
    val_dataset = VitalBISDataset(config.dataset_dir, "val")
    test_dataset = VitalBISDataset(config.dataset_dir, "test")
    train_indices = _selected_indices(train_dataset, 4 if config.smoke else None)
    val_indices = _selected_indices(val_dataset, 3 if config.smoke else None)
    test_indices = _selected_indices(test_dataset, None)

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
    test_loader = make_data_loader(
        test_dataset,
        test_indices,
        config.batch_size,
        config.seed,
        training=False,
        case_balanced=False,
        num_workers=config.num_workers,
    )

    model = _fresh_model(config, train_dataset).to(device)
    parameter_count = count_trainable_parameters(model)
    LOGGER.info("GRU trainable parameters: %d", parameter_count)
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
    _save_json(resolved_config, config.output_dir / "config.json")

    epochs_without_improvement = 0
    best_path = config.output_dir / "best_model.pt"
    last_path = config.output_dir / "last_model.pt"
    max_epochs = min(config.max_epochs, 2) if config.smoke else config.max_epochs
    for epoch in range(start_epoch, max_epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            config.gradient_clip_norm,
        )
        val_bundle = predict_model(model, val_loader, criterion, device)
        val_patient = patient_level_evaluation(
            val_bundle.y_true, val_bundle.y_pred, val_bundle.case_ids
        )
        validation_patient_mae = float(val_patient.summary["mae"]["mean"])
        pooled_mae = float(regression_metrics(val_bundle.y_true, val_bundle.y_pred)["mae"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_bundle.mean_loss,
                "validation_pooled_mae": pooled_mae,
                "validation_patient_level_mae": validation_patient_mae,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        improved = validation_patient_mae < best_patient_mae
        if improved:
            best_patient_mae = validation_patient_mae
            epochs_without_improvement = 0
            torch.save(
                _checkpoint_payload(
                    model, optimizer, epoch, best_patient_mae, history, config
                ),
                best_path,
            )
        else:
            epochs_without_improvement += 1
        torch.save(
            _checkpoint_payload(model, optimizer, epoch, best_patient_mae, history, config),
            last_path,
        )
        pd.DataFrame(history).to_csv(
            config.output_dir / "training_history.csv", index=False
        )
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
    _load_checkpoint(best_path, model, optimizer=None, device=device)
    reloaded_model, reload_identical = _verify_checkpoint_reload(
        model, best_path, val_loader, config, train_dataset, device
    )
    if not reload_identical:
        raise AssertionError("Reloaded checkpoint predictions differ from the saved model.")

    val_bundle = predict_model(reloaded_model, val_loader, criterion, device)
    test_bundle = predict_model(reloaded_model, test_loader, criterion, device)
    thresholds = select_validation_thresholds(val_bundle.y_true, val_bundle.y_pred)
    val_metrics, val_case_metrics = evaluate_bundle(val_bundle, thresholds)
    test_metrics, test_case_metrics = evaluate_bundle(test_bundle, thresholds)
    val_case_metrics.insert(0, "split", "val")
    test_case_metrics.insert(0, "split", "test")
    test_metrics["entirely_missing_remifentanil_case_diagnostics"] = (
        _remifentanil_diagnostics(test_dataset, test_bundle, test_case_metrics)
    )
    val_metrics["checkpoint_reload_predictions_identical"] = reload_identical
    test_metrics["checkpoint_reload_predictions_identical"] = reload_identical
    val_metrics["model_parameter_count"] = parameter_count
    test_metrics["model_parameter_count"] = parameter_count

    prediction_frame(val_bundle).to_csv(
        config.output_dir / "val_predictions.csv", index=False
    )
    prediction_frame(test_bundle).to_csv(
        config.output_dir / "test_predictions.csv", index=False
    )
    pd.concat((val_case_metrics, test_case_metrics), ignore_index=True).to_csv(
        config.output_dir / "case_metrics.csv", index=False
    )
    _save_json(val_metrics, config.output_dir / "val_metrics.json")
    _save_json(test_metrics, config.output_dir / "test_metrics.json")
    return {
        "output_dir": str(config.output_dir),
        "parameter_count": parameter_count,
        "device": str(device),
        "train_tensor_shape": [len(train_indices), 6, len(train_dataset.dynamic_feature_names)],
        "validation_tensor_shape": [
            len(val_indices),
            6,
            len(val_dataset.dynamic_feature_names),
        ],
        "test_tensor_shape": [
            len(test_indices),
            6,
            len(test_dataset.dynamic_feature_names),
        ],
        "checkpoint_reload_predictions_identical": reload_identical,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

