"""Engineering-only CPU/CUDA benchmark for primary-state PPO execution."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch

from .callbacks import PPOProgressCallback
from .cohort import CohortBundle, scenarios_for_split
from .config import PPOConfig
from .environment_factory import make_primary_state_environment
from .full_protocol import FULL_PROFILES, load_full_source, source_config_sha256
from .io import atomic_write_dataframe, atomic_write_json
from .pilot_experiment import _atomic_model_save, evaluate_primary_state_scenarios
from .pilot_protocol import resolve_execution_device
from .run_status import package_versions, repository_commit
from .training import create_primary_state_ppo


BENCHMARK_SCHEMA_VERSION = 1
BENCHMARK_PROFILES = ("all_supported", "selected_control_core")
BENCHMARK_SEED = 999
BENCHMARK_TIMESTEPS = 20_480
BENCHMARK_REPEATS = 3


class _RolloutTimingCallback(BaseCallback):
    def __init__(self, action_callback: PPOProgressCallback) -> None:
        super().__init__(verbose=0)
        self.action_callback = action_callback
        self.environment_seconds = 0.0
        self._rollout_started = 0.0

    def _on_rollout_start(self) -> None:
        self._rollout_started = time.perf_counter()

    def _on_step(self) -> bool:
        self.action_callback.locals = self.locals
        return self.action_callback._on_step()

    def _on_rollout_end(self) -> None:
        self.environment_seconds += time.perf_counter() - self._rollout_started
        self.action_callback.model = self.model
        self.action_callback.num_timesteps = self.num_timesteps
        self.action_callback._on_rollout_end()


class _ResourceSampler:
    def __init__(self, device: str) -> None:
        self.device = device
        self.peak_rss_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = None
        self._cpu_start = None

    def start(self) -> None:
        try:
            import psutil
        except ImportError:
            return
        self._process = psutil.Process()
        self._cpu_start = self._process.cpu_times()

        def sample() -> None:
            while not self._stop.wait(0.1):
                rss = int(self._process.memory_info().rss)
                self.peak_rss_bytes = max(self.peak_rss_bytes or 0, rss)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def finish(self, elapsed_seconds: float) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        process_cpu_percent = None
        if self._process is not None and self._cpu_start is not None and elapsed_seconds > 0:
            end = self._process.cpu_times()
            cpu_seconds = (end.user + end.system) - (self._cpu_start.user + self._cpu_start.system)
            process_cpu_percent = 100.0 * cpu_seconds / elapsed_seconds
            self.peak_rss_bytes = max(
                self.peak_rss_bytes or 0, int(self._process.memory_info().rss)
            )
        return {
            "process_cpu_utilization_percent_one_core_equivalent": process_cpu_percent,
            "peak_process_rss_bytes": self.peak_rss_bytes,
        }


def _gpu_metadata(device: str) -> dict[str, Any]:
    if device != "cuda":
        return {
            "gpu_model": None,
            "gpu_utilization_percent_snapshot": None,
            "peak_vram_bytes": None,
        }
    utilization = None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        utilization = float(completed.stdout.splitlines()[0].strip())
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        utilization = None
    return {
        "gpu_model": torch.cuda.get_device_name(0),
        "gpu_utilization_percent_snapshot": utilization,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }


def _benchmark_config(source_path: Path) -> tuple[dict[str, Any], PPOConfig]:
    source = load_full_source(source_path)
    full = PPOConfig(**source["ppo"])
    benchmark = replace(
        full,
        profile_name="ppo_primary_state_device_benchmark_v1",
        total_timesteps=BENCHMARK_TIMESTEPS,
        evaluation_frequency_timesteps=BENCHMARK_TIMESTEPS,
    )
    return source, benchmark


def run_device_benchmark(
    *,
    source_path: Path,
    repo_dir: Path,
    cohort: CohortBundle,
    output_root: Path,
    device: str,
    profiles: tuple[str, ...] = BENCHMARK_PROFILES,
    repeats: int = BENCHMARK_REPEATS,
) -> dict[str, Any]:
    """Run the prescribed engineering benchmark; never start scientific full runs."""

    resolved = resolve_execution_device(device)
    if tuple(profiles) != BENCHMARK_PROFILES:
        raise ValueError(f"Benchmark profiles must be exactly {BENCHMARK_PROFILES}.")
    if repeats != BENCHMARK_REPEATS:
        raise ValueError(f"Benchmark repeats must be exactly {BENCHMARK_REPEATS}.")
    source, config = _benchmark_config(source_path)
    commit = repository_commit(repo_dir)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    validation_scenarios = scenarios_for_split(cohort, "validation", base_seed=100_000)
    for profile in profiles:
        if profile not in FULL_PROFILES:
            raise ValueError(f"Benchmark profile {profile!r} is not full-protocol eligible.")
        for repeat_index in range(1, repeats + 1):
            run_dir = output_root / resolved / profile / f"repeat_{repeat_index}"
            run_dir.mkdir(parents=True, exist_ok=True)
            env = None
            sampler: _ResourceSampler | None = None
            sampler_finished = False
            started = time.perf_counter()
            try:
                env = make_primary_state_environment(
                    state_profile=profile,
                    ppo=config,
                    seed=BENCHMARK_SEED,
                    cohort=cohort,
                    split="train",
                )
                if resolved == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                model = create_primary_state_ppo(
                    env,
                    state_profile=profile,
                    config=config,
                    seed=BENCHMARK_SEED,
                    device=resolved,
                    verbose=0,
                )
                sampler = _ResourceSampler(resolved)
                sampler.start()
                action_first = PPOProgressCallback(bounds=env.bounds)
                timing_first = _RolloutTimingCallback(action_first)
                train_started = time.perf_counter()
                model.learn(
                    total_timesteps=BENCHMARK_TIMESTEPS // 2,
                    callback=timing_first,
                    progress_bar=False,
                )
                checkpoint = run_dir / "resume_model.zip"
                _atomic_model_save(model, checkpoint)
                checkpoint_timestep = int(model.num_timesteps)
                model = PPO.load(checkpoint, env=env, device=resolved)
                resume_timestep = int(model.num_timesteps)
                action_second = PPOProgressCallback(bounds=env.bounds)
                timing_second = _RolloutTimingCallback(action_second)
                model.learn(
                    total_timesteps=BENCHMARK_TIMESTEPS // 2,
                    reset_num_timesteps=False,
                    callback=timing_second,
                    progress_bar=False,
                )
                training_seconds = time.perf_counter() - train_started
                environment_seconds = timing_first.environment_seconds + timing_second.environment_seconds
                update_seconds = max(0.0, training_seconds - environment_seconds)
                evaluation_started = time.perf_counter()
                validation = evaluate_primary_state_scenarios(
                    model,
                    state_profile=profile,
                    config=config,
                    cohort=cohort,
                    scenarios=validation_scenarios,
                    training_seed=BENCHMARK_SEED,
                )
                evaluation_seconds = time.perf_counter() - evaluation_started
                total_seconds = time.perf_counter() - started
                resources = sampler.finish(total_seconds)
                sampler_finished = True
                diagnostics = action_first.diagnostics()
                second = action_second.diagnostics()
                action_count = diagnostics["normalized_action_count"] + second["normalized_action_count"]
                clipping_count = diagnostics["normalized_clipping_count"] + second["normalized_clipping_count"]
                row = {
                    "schema_version": BENCHMARK_SCHEMA_VERSION,
                    "implementation_commit": commit,
                    "source_config_sha256": source_config_sha256(source_path),
                    "cohort_fingerprint": cohort.fingerprint,
                    "device": resolved,
                    "state_profile": profile,
                    "benchmark_seed": BENCHMARK_SEED,
                    "repeat_index": repeat_index,
                    "requested_timesteps": BENCHMARK_TIMESTEPS,
                    "actual_timesteps": int(model.num_timesteps),
                    "total_wall_seconds": total_seconds,
                    "training_wall_seconds": training_seconds,
                    "training_steps_per_second": BENCHMARK_TIMESTEPS / training_seconds,
                    "environment_stepping_seconds": environment_seconds,
                    "ppo_update_seconds": update_seconds,
                    "validation_evaluation_seconds": evaluation_seconds,
                    "checkpoint_timestep": checkpoint_timestep,
                    "resume_timestep": resume_timestep,
                    "resume_verified": checkpoint_timestep == resume_timestep == BENCHMARK_TIMESTEPS // 2,
                    "metric_schema_verified": len(validation) == 15 and "bis_target_mae" in validation,
                    "numerical_failure_count": int((validation["numerical_failures"] > 0).sum()),
                    "action_count": action_count,
                    "action_clipping_count": clipping_count,
                    "action_clipping_fraction": clipping_count / action_count if action_count else 0.0,
                    "diagnostic_validation_bis_mae": float(validation["bis_target_mae"].mean()),
                    "diagnostic_only_not_scientific_result": True,
                    "test_trajectory_accessed": False,
                    "test_outcomes_evaluated": False,
                    **resources,
                    **_gpu_metadata(resolved),
                }
                if row["actual_timesteps"] != BENCHMARK_TIMESTEPS:
                    raise AssertionError("Benchmark timestep count changed across resume.")
                rows.append(row)
                atomic_write_json(run_dir / "benchmark_result.json", row)
                atomic_write_dataframe(run_dir / "validation_diagnostic.csv", validation)
            except BaseException as exc:
                failures.append(
                    {
                        "device": resolved,
                        "state_profile": profile,
                        "repeat_index": repeat_index,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    }
                )
                atomic_write_json(run_dir / "benchmark_failure.json", failures[-1])
                raise
            finally:
                if sampler is not None and not sampler_finished:
                    sampler.finish(max(time.perf_counter() - started, 1e-9))
                if env is not None:
                    env.close()
    frame = pd.DataFrame(rows)
    atomic_write_dataframe(output_root / f"benchmark_results_{resolved}.csv", frame)
    summary = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "implementation_commit": commit,
        "source_config_sha256": source_config_sha256(source_path),
        "cohort_fingerprint": cohort.fingerprint,
        "device": resolved,
        "profiles": list(profiles),
        "seed": BENCHMARK_SEED,
        "timesteps_per_repeat": BENCHMARK_TIMESTEPS,
        "repeats_per_profile": repeats,
        "completed_repeats": len(rows),
        "failed_repeats": len(failures),
        "median_training_wall_seconds": float(frame["training_wall_seconds"].median()),
        "median_training_steps_per_second": float(frame["training_steps_per_second"].median()),
        "packages": package_versions(),
        "full_training_performed": False,
        "test_trajectory_accessed": False,
    }
    atomic_write_json(output_root / f"benchmark_summary_{resolved}.json", summary)
    return summary


def analyze_device_benchmarks(
    *, result_files: list[Path], output_dir: Path
) -> dict[str, Any]:
    """Merge CPU/CUDA repeats and apply the prespecified 25% engineering rule."""

    if not result_files:
        raise ValueError("At least one benchmark result CSV is required.")
    frames = [pd.read_csv(path) for path in result_files]
    combined = pd.concat(frames, ignore_index=True)
    required = {
        "schema_version",
        "implementation_commit",
        "source_config_sha256",
        "cohort_fingerprint",
        "device",
        "state_profile",
        "repeat_index",
        "training_wall_seconds",
        "resume_verified",
        "metric_schema_verified",
        "numerical_failure_count",
    }
    missing = sorted(required - set(combined.columns))
    if missing:
        raise ValueError(f"Benchmark result schema is incomplete: {missing}.")
    compatibility = ["implementation_commit", "source_config_sha256", "cohort_fingerprint"]
    for column in compatibility:
        if combined[column].nunique() != 1:
            raise ValueError(f"CPU/CUDA benchmark {column} values differ.")
    duplicates = combined.duplicated(["device", "state_profile", "repeat_index"])
    if duplicates.any():
        raise ValueError("Benchmark contains duplicate profile/device/repeat identities.")
    summary = (
        combined.groupby("device").agg(
            completed_repeats=("repeat_index", "count"),
            median_training_wall_seconds=("training_wall_seconds", "median"),
            median_training_steps_per_second=("training_steps_per_second", "median"),
            numerical_failure_count=("numerical_failure_count", "sum"),
            resume_all_verified=("resume_verified", "all"),
            metric_schema_all_verified=("metric_schema_verified", "all"),
        ).reset_index()
    )
    cpu = summary[summary["device"] == "cpu"]
    cuda = summary[summary["device"] == "cuda"]
    cuda_speedup = None
    cuda_qualifies = False
    if not cpu.empty and not cuda.empty:
        cpu_seconds = float(cpu.iloc[0]["median_training_wall_seconds"])
        cuda_seconds = float(cuda.iloc[0]["median_training_wall_seconds"])
        cuda_speedup = 1.0 - cuda_seconds / cpu_seconds
        cuda_row = cuda.iloc[0]
        cuda_qualifies = bool(
            cuda_speedup >= 0.25
            and int(cuda_row["completed_repeats"]) == 6
            and int(cuda_row["numerical_failure_count"]) == 0
            and bool(cuda_row["resume_all_verified"])
            and bool(cuda_row["metric_schema_all_verified"])
        )
    recommendation = "cuda" if cuda_qualifies else "cpu"
    reason = (
        f"CUDA median training wall time improved by {cuda_speedup:.1%} and passed all guards."
        if cuda_qualifies and cuda_speedup is not None
        else (
            "CUDA results are unavailable; CPU remains the prespecified default."
            if cuda.empty
            else f"CUDA did not satisfy the >=25% speedup and compatibility rule (observed {cuda_speedup:.1%})."
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_dataframe(output_dir / "combined_benchmark_results.csv", combined)
    atomic_write_dataframe(output_dir / "device_summary.csv", summary)
    decision = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "implementation_commit": str(combined["implementation_commit"].iloc[0]),
        "source_config_sha256": str(combined["source_config_sha256"].iloc[0]),
        "cohort_fingerprint": str(combined["cohort_fingerprint"].iloc[0]),
        "selected_backend": recommendation,
        "cuda_speedup_fraction": cuda_speedup,
        "cuda_qualifies": cuda_qualifies,
        "selection_rule": "CUDA requires >=25% lower median training wall time, six successful repeats, resume compatibility, and metric-schema compatibility; otherwise CPU.",
        "reason": reason,
        "scientific_metrics_used_for_backend_selection": False,
    }
    atomic_write_json(output_dir / "backend_decision.json", decision)
    return decision
