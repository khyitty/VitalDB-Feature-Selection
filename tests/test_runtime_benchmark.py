"""Tests for thread CLI controls and deterministic runtime benchmarking."""

from __future__ import annotations

from pathlib import Path

from scripts.benchmark_training_runtime import parse_args as parse_benchmark_args
from scripts.run_attention import parse_args as parse_attention_args
from scripts.run_baselines import parse_args as parse_baseline_args
from src.runtime_benchmark import candidate_cpu_configurations, run_training_benchmark


def test_training_clis_parse_thread_and_worker_controls() -> None:
    baseline = parse_baseline_args(
        [
            "gru",
            "--torch-num-threads",
            "4",
            "--torch-interop-threads",
            "1",
            "--num-workers",
            "2",
        ]
    )
    attention = parse_attention_args(
        [
            "--torch-num-threads",
            "2",
            "--torch-interop-threads",
            "1",
            "--num-workers",
            "0",
        ]
    )
    benchmark = parse_benchmark_args(["--thread-counts", "1,2,4", "--worker-counts", "0,2"])

    assert (baseline.torch_num_threads, baseline.torch_interop_threads, baseline.num_workers) == (4, 1, 2)
    assert (attention.torch_num_threads, attention.torch_interop_threads, attention.num_workers) == (2, 1, 0)
    assert benchmark.thread_counts == (1, 2, 4)
    assert benchmark.worker_counts == (0, 2)


def test_candidate_configurations_skip_unsupported_threads() -> None:
    configurations = candidate_cpu_configurations(4, (1, 2, 4, 8), (0, 2))
    assert len(configurations) == 6
    assert {row["torch_num_threads"] for row in configurations} == {1, 2, 4}


def test_benchmark_reuses_initial_weights_and_ordered_batches(
    synthetic_modeling_dir: Path, tmp_path: Path
) -> None:
    hardware = {
        "logical_cpu_core_count": 4,
        "hardware_classification": "C. NO PRACTICAL CUDA GPU",
    }
    result = run_training_benchmark(
        synthetic_modeling_dir,
        tmp_path / "benchmark",
        hardware,
        thread_counts=(1, 2),
        worker_counts=(0,),
        measured_batches=1,
        warmup_batches=1,
        batch_size=2,
        seed=42,
    )

    assert result["method"]["identical_initial_weights_per_model"] is True
    assert result["method"]["identical_ordered_batches_across_cpu_configurations"] is True
    assert (tmp_path / "benchmark" / "training_benchmark.csv").is_file()
