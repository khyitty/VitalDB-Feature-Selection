"""PPO runtime benchmark with honest CPU/CUDA projection labels."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import time
from typing import Any

import torch

from .callbacks import PPOProgressCallback
from .config import smoke_ppo_config
from .smoke import make_synthetic_smoke_env
from .training import create_ppo


def run_runtime_benchmark(
    output_dir: Path,
    *,
    timesteps: int = 10_000,
    condition: str = "attention_supported",
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    if timesteps < 10_000:
        raise ValueError("Published runtime benchmark requires at least 10,000 timesteps.")
    output_dir.mkdir(parents=True, exist_ok=True)
    env = make_synthetic_smoke_env(condition, timesteps)  # type: ignore[arg-type]
    config = smoke_ppo_config(timesteps)
    model = create_ppo(
        env,
        condition=condition,  # type: ignore[arg-type]
        config=config,
        seed=seed,
        device=device,
    )
    callback = PPOProgressCallback()
    process = None
    start_rss = None
    if importlib.util.find_spec("psutil") is not None:
        import psutil

        process = psutil.Process()
        start_rss = int(process.memory_info().rss)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
    elapsed = time.perf_counter() - started
    end_rss = int(process.memory_info().rss) if process is not None else None
    actual_timesteps = int(model.num_timesteps)
    steps_per_second = actual_timesteps / elapsed
    one_million_seconds = 1_000_000 / steps_per_second
    summary = {
        "condition": condition,
        "seed": seed,
        "device": str(model.device),
        "cuda_available": torch.cuda.is_available(),
        "requested_benchmark_timesteps": timesteps,
        "actual_benchmark_timesteps": actual_timesteps,
        "elapsed_seconds": elapsed,
        "steps_per_second": steps_per_second,
        "process_rss_start_bytes": start_rss,
        "process_rss_end_bytes": end_rss,
        "process_rss_change_bytes": (
            end_rss - start_rss
            if end_rss is not None and start_rss is not None
            else None
        ),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else None
        ),
        "projected_100k_seconds_same_device": 100_000 / steps_per_second,
        "projected_1m_seconds_same_device": one_million_seconds,
        "projected_1_024m_seconds_same_device": 1_024_000 / steps_per_second,
        "projected_20_run_hours_same_device_serial": (
            20 * 1_024_000 / steps_per_second / 3600.0
        ),
        "gpu_projection_available": device.startswith("cuda"),
        "gpu_projection_note": (
            "Measured on CUDA."
            if device.startswith("cuda")
            else "No local CUDA measurement; same-device projections are CPU-only and must not be labeled GPU runtime."
        ),
        "smoke_or_benchmark_only": True,
        "full_training_performed": False,
        "action_diagnostics": callback.diagnostics(),
    }
    (output_dir / "runtime_benchmark.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    env.close()
    return summary
