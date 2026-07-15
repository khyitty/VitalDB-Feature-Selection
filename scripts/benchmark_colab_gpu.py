"""Run short reduced-model CUDA benchmarks on an assigned Colab GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.colab_workflow import run_colab_gpu_benchmark  # noqa: E402


def _integers(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Provide positive comma-separated integers.")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=_integers, default=(256, 512, 1024, 2048))
    parser.add_argument("--measured-batches", type=int, default=20)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = run_colab_gpu_benchmark(
        args.dataset_dir,
        args.output_dir,
        batch_sizes=args.batch_sizes,
        measured_batches=args.measured_batches,
        warmup_batches=args.warmup_batches,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
