"""Validation and short CUDA benchmark helpers for the Google Colab workflow."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import pickle
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.datasets import VitalBISDataset
from src.redundancy_audit import REDUCED_FEATURES
from src.runtime_benchmark import (
    _fresh_benchmark_model,
    benchmark_one_configuration,
)
from src.training import set_deterministic_seed

REQUIRED_MODELING_FILES = (
    "train.npz",
    "val.npz",
    "test.npz",
    "train_metadata.csv",
    "val_metadata.csv",
    "test_metadata.csv",
    "dataset_metadata.json",
    "preprocessing.pkl",
    "preprocessing_statistics.csv",
    "feature_manifest.csv",
)
GRU_SMOKE_REQUIRED = (
    "run_status.json",
    "config.json",
    "best_model.pt",
    "last_model.pt",
    "training_history.csv",
    "val_predictions.csv",
    "val_metrics.json",
    "case_metrics.csv",
    "runtime.json",
)
ATTENTION_SMOKE_REQUIRED = (
    "run_status.json",
    "config.json",
    "best_model.pt",
    "last_model.pt",
    "training_history.csv",
    "val_predictions.csv",
    "val_metrics.json",
    "case_metrics.csv",
    "val_attention.npz",
    "attention_metadata.json",
)
GPU_BENCHMARK_COLUMNS = (
    "model",
    "device",
    "gpu_name",
    "batch_size",
    "measured_batches",
    "mean_batch_time_seconds",
    "windows_per_second",
    "total_benchmark_seconds",
    "peak_allocated_cuda_memory_bytes",
    "status",
)
FORBIDDEN_COLAB_DISTRIBUTIONS = frozenset({"torch", "torchvision", "torchaudio"})
FROZEN_TEST_PROTECTED_DISTRIBUTIONS = frozenset(
    {*FORBIDDEN_COLAB_DISTRIBUTIONS, "pandas"}
)
FROZEN_TEST_REQUIRED_IMPORTS = (
    "numpy",
    "scipy",
    "pandas",
    "torch",
    "matplotlib",
    "sklearn",
)


def dump_json(payload: Mapping[str, Any], path: Path) -> None:
    """Write a strict JSON object after creating its parent directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")


def _distribution_name(requirement: str) -> str | None:
    stripped = requirement.split("#", maxsplit=1)[0].strip()
    if not stripped:
        return None
    match = re.match(r"^[A-Za-z0-9_.-]+", stripped)
    if match is None:
        raise ValueError(f"Unsupported Colab requirement line: {requirement!r}")
    return match.group(0).lower().replace("_", "-")


def validate_colab_requirements(path: Path) -> list[str]:
    """Return dependency lines and reject anything that could replace PyTorch."""

    lines = path.read_text(encoding="utf-8").splitlines()
    requirements = [line.strip() for line in lines if _distribution_name(line)]
    names = {_distribution_name(line) for line in requirements}
    forbidden = sorted(FORBIDDEN_COLAB_DISTRIBUTIONS & names)
    if forbidden:
        raise ValueError(
            "Colab dependencies must retain preinstalled CUDA PyTorch; remove "
            f"{forbidden}."
        )
    return requirements


def missing_colab_requirements(path: Path) -> list[str]:
    """Return only requirement lines whose distributions are not installed."""

    missing: list[str] = []
    for requirement in validate_colab_requirements(path):
        name = _distribution_name(requirement)
        assert name is not None
        try:
            importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(requirement)
    return missing


def validate_pip_install_plan(
    payload: Mapping[str, Any],
    protected_distributions: Sequence[str] = FORBIDDEN_COLAB_DISTRIBUTIONS,
) -> None:
    """Reject a pip dry-run plan that would replace protected distributions."""

    planned = {
        str(item.get("metadata", {}).get("name", "")).lower().replace("_", "-")
        for item in payload.get("install", [])
    }
    protected = {name.lower().replace("_", "-") for name in protected_distributions}
    forbidden = sorted(protected & planned)
    if forbidden:
        raise RuntimeError(
            "Dependency resolution would replace protected Colab packages "
            f"{forbidden}; installation aborted."
        )


def frozen_test_runtime_versions() -> dict[str, Any]:
    """Validate inference imports while retaining Colab pandas and CUDA PyTorch."""

    imported: dict[str, str | None] = {}
    for module_name in FROZEN_TEST_REQUIRED_IMPORTS:
        module = __import__(module_name)
        imported[module_name] = getattr(module, "__version__", None)
    pandas_major = int(pd.__version__.split(".", maxsplit=1)[0])
    if pandas_major >= 3:
        raise RuntimeError(
            f"Frozen-test inference requires the Colab pandas 2.x line; found {pd.__version__}."
        )
    return {
        "package_versions": imported,
        "pandas_version": pd.__version__,
        "pandas_major_version": pandas_major,
        "torch_version": torch.__version__,
        "torch_cuda_runtime_version": torch.version.cuda,
        "torch_cuda_is_available": bool(torch.cuda.is_available()),
        "vitaldb_required": False,
        "wfdb_required": False,
    }


def _git_commit(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"path": None, "output": None, "return_code": None}
    result = subprocess.run(
        [executable], check=False, capture_output=True, text=True, timeout=30
    )
    return {
        "path": executable,
        "output": result.stdout.strip() or result.stderr.strip(),
        "return_code": result.returncode,
    }


def audit_colab_environment(
    output_path: Path,
    repo_dir: Path,
    require_cuda: bool = True,
) -> dict[str, Any]:
    """Record the Colab backend and stop clearly when a GPU was not assigned."""

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count())
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(device_count):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(properties.total_memory),
                    "total_memory_gib": float(properties.total_memory / 2**30),
                    "compute_capability": [properties.major, properties.minor],
                }
            )
    package_names = ("numpy", "pandas", "scikit-learn", "pytest", "vitaldb")
    packages: dict[str, str | None] = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    classification = "A. CUDA READY" if cuda_available and device_count > 0 else "B. COLAB GPU NOT ASSIGNED"
    payload = {
        "classification": classification,
        "operating_system": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_runtime_version": torch.version.cuda,
        "torch_cuda_is_available": cuda_available,
        "torch_cuda_device_count": device_count,
        "cuda_devices": devices,
        "nvidia_smi": _nvidia_smi(),
        "git_commit": _git_commit(repo_dir),
        "package_versions": packages,
        "mixed_precision_enabled": False,
        "environment_or_driver_changes_performed": False,
    }
    dump_json(payload, output_path)
    if require_cuda and classification != "A. CUDA READY":
        raise RuntimeError(
            "Colab GPU is not assigned. Select Runtime > Change runtime type > GPU, "
            "reconnect, and rerun the environment audit. CPU fallback is disabled."
        )
    return payload


def validate_modeling_artifacts(
    dataset_dir: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Validate all modeling artifacts without requiring raw one-second VitalDB data."""

    missing = [name for name in REQUIRED_MODELING_FILES if not (dataset_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Modeling directory is incomplete: {missing}")
    with (dataset_dir / "dataset_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with (dataset_dir / "preprocessing.pkl").open("rb") as handle:
        preprocessing = pickle.load(handle)
    if preprocessing is None:
        raise ValueError("preprocessing.pkl did not contain a readable object.")
    statistics = pd.read_csv(dataset_dir / "preprocessing_statistics.csv")
    manifest = pd.read_csv(dataset_dir / "feature_manifest.csv")
    if statistics.empty or manifest.empty:
        raise ValueError("Preprocessing statistics or feature manifest is empty.")

    split_rows: dict[str, Any] = {}
    split_cases: dict[str, set[int]] = {}
    for split in ("train", "val", "test"):
        dataset = VitalBISDataset(
            dataset_dir,
            split,
            dynamic_features=REDUCED_FEATURES,
        )
        dynamic = dataset.arrays["X_dynamic"]
        static = dataset.arrays["X_static"]
        masks = dataset.arrays["observation_mask"]
        expected_rows = pd.read_csv(dataset_dir / f"{split}_metadata.csv")
        if len(expected_rows) != len(dataset):
            raise ValueError(f"{split} metadata row count does not match NPZ arrays.")
        if not np.array_equal(
            expected_rows["case_id"].to_numpy(dtype=np.int64), dataset.case_ids
        ):
            raise ValueError(f"{split} case IDs are not row-aligned.")
        if not np.isfinite(dynamic).all() or not np.isfinite(static).all():
            raise ValueError(f"{split} contains NaN or infinite feature values.")
        if not np.isfinite(dataset.arrays["y_bis"]).all():
            raise ValueError(f"{split} contains NaN or infinite targets.")
        if dynamic.shape[2] != 17 or static.shape[1] != 6 or masks.shape != dynamic.shape:
            raise ValueError(f"{split} resolved tensor shapes are invalid.")
        if tuple(dataset.dynamic_feature_names) != REDUCED_FEATURES:
            raise ValueError(f"{split} does not resolve the fixed 17-feature order.")
        split_cases[split] = set(dataset.case_ids.astype(int).tolist())
        split_rows[split] = {
            "dynamic_shape": list(dynamic.shape),
            "static_shape": list(static.shape),
            "mask_shape": list(masks.shape),
            "target_shape": list(dataset.arrays["y_bis"].shape),
            "metadata_rows": len(expected_rows),
            "case_count": len(split_cases[split]),
            "all_arrays_finite": True,
            "metadata_row_alignment": True,
        }
    overlaps = {
        "train_validation": sorted(split_cases["train"] & split_cases["val"]),
        "train_test": sorted(split_cases["train"] & split_cases["test"]),
        "validation_test": sorted(split_cases["val"] & split_cases["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Case-level split leakage detected: {overlaps}")
    payload = {
        "dataset_dir": str(dataset_dir),
        "required_files": list(REQUIRED_MODELING_FILES),
        "source_dynamic_feature_count": len(metadata["dynamic_feature_names"]),
        "resolved_dynamic_features": list(REDUCED_FEATURES),
        "bis_error_excluded": "bis_error" not in REDUCED_FEATURES,
        "static_feature_count": len(metadata["static_feature_names"]),
        "splits": split_rows,
        "case_id_overlaps": overlaps,
        "case_level_split_integrity": True,
        "raw_one_second_csv_required": False,
        "large_artifacts_duplicated": False,
    }
    if output_path is not None:
        dump_json(payload, output_path)
    return payload


def inspect_run_completion(run_dir: Path, model: str) -> dict[str, Any]:
    """Distinguish complete Colab smoke runs from interrupted directories."""

    if model not in {"gru", "attention"}:
        raise ValueError("model must be 'gru' or 'attention'.")
    required = GRU_SMOKE_REQUIRED if model == "gru" else ATTENTION_SMOKE_REQUIRED
    missing = [name for name in required if not (run_dir / name).is_file()]
    status_payload: dict[str, Any] = {}
    status_path = run_dir / "run_status.json"
    if status_path.is_file():
        with status_path.open("r", encoding="utf-8") as handle:
            status_payload = json.load(handle)
    complete = not missing and status_payload.get("status") == "complete"
    return {
        "run_dir": str(run_dir),
        "model": model,
        "complete": complete,
        "missing_artifacts": missing,
        "status": status_payload.get("status", "missing"),
        "safe_action": "skip" if complete else "restart_or_resume_this_directory_only",
    }


def verify_colab_smoke_run(
    run_dir: Path,
    dataset_dir: Path,
    model: str,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Verify a CUDA validation-only smoke run without reading test artifacts."""

    completion = inspect_run_completion(run_dir, model)
    if not completion["complete"]:
        raise FileNotFoundError(f"Smoke run is incomplete: {completion}")
    with (run_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if tuple(config["dynamic_feature_names"]) != REDUCED_FEATURES:
        raise ValueError("Smoke run feature order is not the fixed 17-feature order.")
    if config["resolved_device"] != "cuda" or not config["smoke"]:
        raise ValueError("Smoke run must be an explicit CUDA smoke configuration.")
    if (run_dir / "test_predictions.csv").exists() or (run_dir / "test_metrics.json").exists():
        raise ValueError("Colab migration smoke must not evaluate the test split.")
    dataset = VitalBISDataset(dataset_dir, "val", dynamic_features=REDUCED_FEATURES)
    predictions = pd.read_csv(run_dir / "val_predictions.csv")
    sample_indices = predictions["sample_index"].to_numpy(dtype=np.int64)
    if not np.isfinite(predictions["predicted_future_bis"]).all():
        raise ValueError("Smoke predictions contain NaN or infinite values.")
    if not np.array_equal(predictions["case_id"].to_numpy(dtype=np.int64), dataset.case_ids[sample_indices]):
        raise ValueError("Smoke validation case IDs are not aligned.")
    if not np.array_equal(
        predictions["target_timestamp"].to_numpy(dtype=np.int64),
        dataset.metadata["target_timestamp"].to_numpy(dtype=np.int64)[sample_indices],
    ):
        raise ValueError("Smoke validation timestamps are not aligned.")
    with (run_dir / "val_metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    payload: dict[str, Any] = {
        **completion,
        "validation_only": True,
        "test_artifacts_absent": True,
        "dynamic_feature_names": list(REDUCED_FEATURES),
        "parameter_count": int(config["model_parameter_count"]),
        "validation_prediction_count": len(predictions),
        "predictions_finite": True,
        "metadata_alignment_exact": True,
        "checkpoint_reload_predictions_identical": bool(
            metrics["checkpoint_reload_predictions_identical"]
        ),
        "smoke_attention_is_not_scientific_evidence": model == "attention",
    }
    if model == "gru":
        with (run_dir / "runtime.json").open("r", encoding="utf-8") as handle:
            runtime = json.load(handle)
        payload["runtime_seconds"] = float(runtime["total_internal_runtime_seconds"])
    else:
        with np.load(run_dir / "val_attention.npz", allow_pickle=False) as attention:
            feature = attention["feature_attention"]
            temporal = attention["temporal_attention"]
            combined = attention["combined_attention"]
        selected_masks = dataset.arrays["observation_mask"][sample_indices]
        payload.update(
            {
                "attention_values_finite": bool(
                    np.isfinite(feature).all()
                    and np.isfinite(temporal).all()
                    and np.isfinite(combined).all()
                ),
                "feature_normalization_max_error": float(
                    np.max(np.abs(feature.sum(axis=2) - 1.0))
                ),
                "temporal_normalization_max_error": float(
                    np.max(np.abs(temporal.sum(axis=1) - 1.0))
                ),
                "combined_normalization_max_error": float(
                    np.max(np.abs(combined.sum(axis=(1, 2)) - 1.0))
                ),
                "maximum_missing_feature_weight": float(
                    feature[~selected_masks.astype(bool)].max(initial=0.0)
                ),
                "checkpoint_reload_attention_identical": bool(
                    metrics["checkpoint_reload_attention_identical"]
                ),
            }
        )
        tolerance = 1e-5
        if (
            not payload["attention_values_finite"]
            or payload["feature_normalization_max_error"] > tolerance
            or payload["temporal_normalization_max_error"] > tolerance
            or payload["combined_normalization_max_error"] > tolerance
            or payload["maximum_missing_feature_weight"] > tolerance
        ):
            raise ValueError("Smoke attention failed finite, normalization, or mask checks.")
        with (run_dir / "attention_metadata.json").open("r", encoding="utf-8") as handle:
            attention_metadata = json.load(handle)
        payload["runtime_seconds"] = float(
            attention_metadata["runtime_breakdown"]["total_internal_runtime_seconds"]
        )
    if output_path is not None:
        dump_json(payload, output_path)
    return payload


def validate_gpu_benchmark_schema(frame: pd.DataFrame) -> None:
    """Require the portable GPU benchmark output schema and finite successful rows."""

    missing = sorted(set(GPU_BENCHMARK_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"GPU benchmark output is missing columns: {missing}")
    successful = frame[frame["status"] == "ok"]
    if successful.empty:
        raise ValueError("GPU benchmark contains no successful configuration.")
    numeric = successful[
        [
            "mean_batch_time_seconds",
            "windows_per_second",
            "total_benchmark_seconds",
            "peak_allocated_cuda_memory_bytes",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric <= 0).any():
        raise ValueError("Successful GPU benchmark rows must contain positive finite metrics.")


def run_colab_gpu_benchmark(
    dataset_dir: Path,
    output_dir: Path,
    batch_sizes: Sequence[int] = (256, 512, 1024, 2048),
    measured_batches: int = 20,
    warmup_batches: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """Run short deterministic CUDA training-step benchmarks for both models."""

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            "CUDA benchmark requested without an assigned GPU. Select a Colab GPU runtime."
        )
    if measured_batches <= 0 or measured_batches > 50:
        raise ValueError("measured_batches must be between 1 and 50.")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = VitalBISDataset(dataset_dir, "train", dynamic_features=REDUCED_FEATURES)
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    initial_states: dict[str, dict[str, torch.Tensor]] = {}
    for model_name in ("gru", "attention"):
        set_deterministic_seed(seed)
        initial_states[model_name] = copy.deepcopy(
            _fresh_benchmark_model(model_name, dataset).state_dict()
        )
    rows: list[dict[str, Any]] = []
    benchmark_started = perf_counter()
    for model_name in ("gru", "attention"):
        for batch_size in batch_sizes:
            maximum_measured = len(dataset) // batch_size - warmup_batches
            actual_measured = min(measured_batches, maximum_measured)
            if actual_measured <= 0:
                rows.append(
                    {
                        "model": model_name,
                        "device": "cuda",
                        "gpu_name": gpu_name,
                        "batch_size": batch_size,
                        "measured_batches": 0,
                        "mean_batch_time_seconds": None,
                        "windows_per_second": None,
                        "total_benchmark_seconds": None,
                        "peak_allocated_cuda_memory_bytes": None,
                        "status": "insufficient_dataset_rows",
                    }
                )
                continue
            try:
                row = benchmark_one_configuration(
                    model_name,
                    dataset,
                    initial_states[model_name],
                    device,
                    batch_size,
                    actual_measured,
                    warmup_batches,
                    0,
                    seed,
                )
                row.update({"gpu_name": gpu_name, "status": "ok"})
                rows.append(row)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                rows.append(
                    {
                        "model": model_name,
                        "device": "cuda",
                        "gpu_name": gpu_name,
                        "batch_size": batch_size,
                        "measured_batches": actual_measured,
                        "mean_batch_time_seconds": None,
                        "windows_per_second": None,
                        "total_benchmark_seconds": None,
                        "peak_allocated_cuda_memory_bytes": None,
                        "status": "cuda_out_of_memory",
                    }
                )
    frame = pd.DataFrame(rows)
    validate_gpu_benchmark_schema(frame)
    successful = frame[frame["status"] == "ok"]
    fastest = {
        model: int(group.loc[group["windows_per_second"].idxmax(), "batch_size"])
        for model, group in successful.groupby("model")
    }
    conservative_batch_size = int(successful["batch_size"].min())
    payload = {
        "gpu_name": gpu_name,
        "seed": seed,
        "feature_order": list(REDUCED_FEATURES),
        "measured_batch_limit": measured_batches,
        "warmup_batches": warmup_batches,
        "identical_initial_weights_across_batch_sizes_per_model": True,
        "forward_loss_backward_optimizer_step": True,
        "mixed_precision_enabled": False,
        "determinism_settings_changed_for_speed": False,
        "fastest_safe_batch_size": fastest,
        "conservative_lower_memory_batch_size": conservative_batch_size,
        "total_benchmark_seconds": perf_counter() - benchmark_started,
        "rows": json.loads(frame.to_json(orient="records")),
    }
    frame.to_csv(output_dir / "colab_gpu_benchmark.csv", index=False)
    dump_json(payload, output_dir / "colab_gpu_benchmark.json")
    return payload
