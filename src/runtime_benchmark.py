"""Hardware audit and deterministic short training-step benchmarks."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.datasets import VitalBISDataset, seed_worker
from src.models.attention import FactorizedAttentionGRU
from src.models.baselines import GRUBaseline
from src.redundancy_audit import REDUCED_FEATURES
from src.training import configure_torch_threads, set_deterministic_seed


def _powershell_json(command: str) -> Any:
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        return json.loads(result.stdout) if result.stdout.strip() else None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def _requirements_torch_state(requirements_path: Path) -> dict[str, Any]:
    lines = requirements_path.read_text(encoding="utf-8").splitlines()
    torch_lines = [line.strip() for line in lines if line.strip().lower().startswith("torch")]
    forced_cpu = any("+cpu" in line.lower() or "cpu/torch" in line.lower() for line in torch_lines)
    return {"torch_requirement_lines": torch_lines, "explicitly_forces_cpu_only": forced_cpu}


def audit_hardware(requirements_path: Path = Path("requirements.txt")) -> dict[str, Any]:
    """Inspect local CPU, RAM, GPUs, drivers, and the active PyTorch backend."""

    processors = _powershell_json(
        "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress"
    )
    computer = _powershell_json(
        "Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory | ConvertTo-Json -Compress"
    )
    video = _powershell_json(
        "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress"
    )
    processor_rows = processors if isinstance(processors, list) else [processors] if processors else []
    video_rows = video if isinstance(video, list) else [video] if video else []
    physical_cores = sum(int(row.get("NumberOfCores") or 0) for row in processor_rows)
    logical_cores = sum(int(row.get("NumberOfLogicalProcessors") or 0) for row in processor_rows)
    total_ram = int((computer or {}).get("TotalPhysicalMemory") or 0)
    nvidia_smi = shutil.which("nvidia-smi")
    nvidia_smi_output = None
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            nvidia_smi_output = result.stdout.strip()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            nvidia_smi_output = None
    cuda_available = bool(torch.cuda.is_available())
    cuda_count = int(torch.cuda.device_count())
    cuda_names = [torch.cuda.get_device_name(index) for index in range(cuda_count)]
    nvidia_visible = any("nvidia" in str(row.get("Name", "")).lower() for row in video_rows)
    cpu_only_build = torch.version.cuda is None or "+cpu" in torch.__version__
    if cuda_available and cuda_count:
        category = "A. CUDA GPU READY"
    elif nvidia_visible:
        category = "B. NVIDIA GPU PRESENT BUT CUDA PYTORCH UNAVAILABLE"
    else:
        category = "C. NO PRACTICAL CUDA GPU"
    requirements = _requirements_torch_state(requirements_path)
    return {
        "operating_system": platform.platform(),
        "cpu_model": "; ".join(str(row.get("Name", "unknown")).strip() for row in processor_rows),
        "physical_cpu_core_count": physical_cores or None,
        "logical_cpu_core_count": logical_cores or os.cpu_count(),
        "total_system_ram_bytes": total_ram or None,
        "total_system_ram_gib": float(total_ram / 2**30) if total_ram else None,
        "windows_video_controllers": video_rows,
        "torch_version": torch.__version__,
        "torch_version_cuda": torch.version.cuda,
        "torch_cuda_is_available": cuda_available,
        "torch_cuda_device_count": cuda_count,
        "torch_cuda_device_names": cuda_names,
        "installed_pytorch_build_is_cpu_only": cpu_only_build,
        "nvidia_smi_path": nvidia_smi,
        "nvidia_smi_query_output": nvidia_smi_output,
        "nvidia_driver_appears_available": bool(nvidia_smi_output or nvidia_visible),
        "requirements": requirements,
        "hardware_classification": category,
        "automatic_environment_changes_performed": False,
    }


def candidate_cpu_configurations(
    logical_cores: int, thread_counts: Sequence[int], worker_counts: Sequence[int]
) -> list[dict[str, int]]:
    """Return supported CPU benchmark configurations in deterministic order."""

    threads = sorted({value for value in thread_counts if 0 < value <= logical_cores})
    workers = sorted({value for value in worker_counts if value >= 0})
    if not threads or not workers:
        raise ValueError("At least one supported thread and worker count is required.")
    return [
        {"torch_num_threads": thread, "torch_interop_threads": 1, "num_workers": worker}
        for thread in threads
        for worker in workers
    ]


def _fresh_benchmark_model(model_name: str, dataset: VitalBISDataset) -> nn.Module:
    if model_name == "gru":
        return GRUBaseline(
            len(dataset.dynamic_feature_names),
            len(dataset.static_feature_names),
            hidden_size=64,
            projection_size=64,
            static_hidden_size=16,
            prediction_hidden_size=32,
            dropout=0.0,
        )
    if model_name == "attention":
        return FactorizedAttentionGRU(
            len(dataset.dynamic_feature_names),
            len(dataset.static_feature_names),
            history_steps=int(dataset.dataset_metadata["history_steps"]),
            feature_token_embedding_dim=16,
            static_context_dim=16,
            hidden_size=64,
            prediction_hidden_size=32,
            dropout=0.0,
        )
    raise ValueError(f"Unsupported benchmark model: {model_name}")


def state_dict_fingerprint(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash tensor names, shapes, dtypes, and bytes for initialization checks."""

    digest = hashlib.sha256()
    for name, value in state_dict.items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _make_ordered_loader(
    dataset: VitalBISDataset,
    batch_size: int,
    total_batches: int,
    num_workers: int,
    seed: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    sample_count = min(len(dataset), batch_size * total_batches)
    subset = Subset(dataset, list(range(sample_count)))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=False,
    )


def batch_order_fingerprint(sample_indices: Iterable[torch.Tensor]) -> str:
    """Hash the exact sequence of batch sample-index vectors."""

    digest = hashlib.sha256()
    for indices in sample_indices:
        digest.update(indices.detach().cpu().numpy().astype(np.int64).tobytes())
    return digest.hexdigest()


def _process_measurements() -> tuple[float | None, int | None]:
    try:
        import psutil  # type: ignore[import-not-found]

        process = psutil.Process()
        return float(process.cpu_percent(interval=None)), int(process.memory_info().rss)
    except (ImportError, OSError):
        return None, None


def benchmark_one_configuration(
    model_name: str,
    dataset: VitalBISDataset,
    initial_state: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    measured_batches: int,
    warmup_batches: int,
    num_workers: int,
    seed: int,
) -> dict[str, Any]:
    """Time full training steps from identical initialization and ordered batches."""

    model = _fresh_benchmark_model(model_name, dataset).to(device)
    model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)
    loader = _make_ordered_loader(
        dataset, batch_size, warmup_batches + measured_batches, num_workers, seed
    )
    iterator = iter(loader)
    seen_indices: list[torch.Tensor] = []

    def step(batch: dict[str, torch.Tensor]) -> None:
        values = batch["X_dynamic"].to(device=device, dtype=torch.float32)
        static = batch["X_static"].to(device=device, dtype=torch.float32)
        masks = batch["observation_mask"].to(device=device)
        target = batch["y_bis"].to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(values, static, masks)
        if not isinstance(prediction, torch.Tensor):
            prediction = prediction.prediction
        loss = criterion(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.train()
    for _ in range(warmup_batches):
        step(next(iterator))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    cpu_before, memory_before = _process_measurements()
    elapsed_batches: list[float] = []
    total_windows = 0
    total_started = perf_counter()
    for _ in range(measured_batches):
        batch_started = perf_counter()
        batch = next(iterator)
        seen_indices.append(batch["sample_index"].clone())
        step(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_batches.append(perf_counter() - batch_started)
        total_windows += len(batch["sample_index"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_seconds = perf_counter() - total_started
    cpu_after, memory_after = _process_measurements()
    return {
        "model": model_name,
        "device": str(device),
        "batch_size": batch_size,
        "torch_num_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "num_workers": num_workers,
        "warmup_batches": warmup_batches,
        "measured_batches": measured_batches,
        "mean_batch_time_seconds": float(np.mean(elapsed_batches)),
        "windows_per_second": float(total_windows / total_seconds),
        "total_benchmark_seconds": float(total_seconds),
        "initial_state_fingerprint": state_dict_fingerprint(initial_state),
        "measured_batch_order_fingerprint": batch_order_fingerprint(seen_indices),
        "process_cpu_percent_before": cpu_before,
        "process_cpu_percent_after": cpu_after,
        "process_memory_rss_bytes_before": memory_before,
        "process_memory_rss_bytes_after": memory_after,
        "peak_allocated_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
    }


def run_training_benchmark(
    dataset_dir: Path,
    output_dir: Path,
    hardware: dict[str, Any],
    thread_counts: Sequence[int] = (1, 2, 4, 8),
    worker_counts: Sequence[int] = (0, 2),
    measured_batches: int = 20,
    warmup_batches: int = 3,
    batch_size: int = 256,
    seed: int = 42,
) -> dict[str, Any]:
    """Run CPU configurations and CUDA batch sizes only when already available."""

    output_dir.mkdir(parents=True, exist_ok=True)
    configure_torch_threads(torch_interop_threads=1)
    set_deterministic_seed(seed)
    dataset = VitalBISDataset(dataset_dir, "train", dynamic_features=REDUCED_FEATURES)
    configurations = candidate_cpu_configurations(
        int(hardware["logical_cpu_core_count"]), thread_counts, worker_counts
    )
    initial_states: dict[str, dict[str, torch.Tensor]] = {}
    for model_name in ("gru", "attention"):
        set_deterministic_seed(seed)
        model = _fresh_benchmark_model(model_name, dataset)
        initial_states[model_name] = copy.deepcopy(model.state_dict())
    rows: list[dict[str, Any]] = []
    benchmark_started = perf_counter()
    for model_name in ("gru", "attention"):
        for configuration in configurations:
            configure_torch_threads(configuration["torch_num_threads"])
            rows.append(
                benchmark_one_configuration(
                    model_name,
                    dataset,
                    initial_states[model_name],
                    torch.device("cpu"),
                    batch_size,
                    measured_batches,
                    warmup_batches,
                    configuration["num_workers"],
                    seed,
                )
            )
    if hardware["hardware_classification"] == "A. CUDA GPU READY":
        for model_name in ("gru", "attention"):
            for cuda_batch_size in (256, 512, 1024):
                try:
                    rows.append(
                        benchmark_one_configuration(
                            model_name,
                            dataset,
                            initial_states[model_name],
                            torch.device("cuda"),
                            cuda_batch_size,
                            measured_batches,
                            warmup_batches,
                            0,
                            seed,
                        )
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    rows.append(
                        {
                            "model": model_name,
                            "device": "cuda",
                            "batch_size": cuda_batch_size,
                            "status": "CUDA_OUT_OF_MEMORY",
                        }
                    )
    frame = pd.DataFrame(rows)
    valid = frame[frame["windows_per_second"].notna()].copy()
    cpu = valid[valid["device"] == "cpu"]
    configuration_summary = (
        cpu.groupby(
            ["torch_num_threads", "torch_interop_threads", "num_workers"],
            as_index=False,
        )
        .agg(
            model_count=("model", "nunique"),
            total_measured_windows=("measured_batches", lambda values: int(values.sum() * batch_size)),
            combined_timed_seconds=("total_benchmark_seconds", "sum"),
        )
    )
    configuration_summary["combined_windows_per_second"] = (
        configuration_summary["total_measured_windows"]
        / configuration_summary["combined_timed_seconds"]
    )
    complete_summary = configuration_summary[configuration_summary["model_count"] == 2]
    fastest_index = complete_summary["combined_windows_per_second"].idxmax()
    fastest = json.loads(complete_summary.loc[fastest_index].to_json())
    cooler_candidates = complete_summary[
        (complete_summary["torch_num_threads"] < fastest["torch_num_threads"])
        & (
            complete_summary["combined_windows_per_second"]
            >= 0.9 * fastest["combined_windows_per_second"]
        )
    ]
    cooler = (
        cooler_candidates.sort_values(
            ["torch_num_threads", "combined_windows_per_second"],
            ascending=[True, False],
        ).iloc[0].pipe(lambda row: json.loads(row.to_json()))
        if not cooler_candidates.empty
        else None
    )
    lower_thread_candidates = complete_summary[
        complete_summary["torch_num_threads"] < fastest["torch_num_threads"]
    ]
    closest_lower_thread = (
        json.loads(
            lower_thread_candidates.loc[
                lower_thread_candidates["combined_windows_per_second"].idxmax()
            ].to_json()
        )
        if not lower_thread_candidates.empty
        else None
    )
    initial_reused = bool(
        all(
            group["initial_state_fingerprint"].nunique() == 1
            for _, group in valid.groupby("model")
        )
    )
    batches_reused = bool(
        all(
            group["measured_batch_order_fingerprint"].nunique() == 1
            for _, group in cpu.groupby("model")
        )
    )
    payload = {
        "method": {
            "seed": seed,
            "features": list(REDUCED_FEATURES),
            "forward_loss_backward_optimizer_step": True,
            "warmup_before_timing": True,
            "identical_initial_weights_per_model": initial_reused,
            "identical_ordered_batches_across_cpu_configurations": batches_reused,
            "mixed_precision_enabled": False,
            "tf32_changed": False,
            "scientific_checkpoints_modified": False,
        },
        "total_cpu_and_optional_cuda_benchmark_seconds": perf_counter() - benchmark_started,
        "fastest_cpu_configuration": fastest,
        "cooler_cpu_configuration_within_10_percent_if_available": cooler,
        "fastest_lower_thread_configuration_when_no_10_percent_candidate": closest_lower_thread,
        "cpu_configuration_summary": json.loads(
            configuration_summary.to_json(orient="records")
        ),
        "future_device_recommendation": (
            "CUDA, using one backend for every paired model and seed"
            if hardware["hardware_classification"] == "A. CUDA GPU READY"
            else "CPU with the recommended thread configuration"
        ),
        "separate_cuda_environment_worthwhile": bool(
            hardware["hardware_classification"]
            == "B. NVIDIA GPU PRESENT BUT CUDA PYTORCH UNAVAILABLE"
        ),
        "historical_experiments_should_not_be_rerun_for_device_only": True,
        "rows": json.loads(frame.to_json(orient="records")),
    }
    frame.to_csv(output_dir / "training_benchmark.csv", index=False)
    with (output_dir / "training_benchmark.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    return payload
