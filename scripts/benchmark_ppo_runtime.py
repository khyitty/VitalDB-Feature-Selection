"""Benchmark representative explicit-attention PPO throughput."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl_training.runtime_benchmark import run_runtime_benchmark


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PPO smoke throughput without full training.")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--condition", default="attention_supported")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/ppo_runtime_benchmark"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(json.dumps(run_runtime_benchmark(
        args.output_dir,
        timesteps=args.timesteps,
        condition=args.condition,
        seed=args.seed,
        device=args.device,
    ), indent=2))


if __name__ == "__main__":
    main()
