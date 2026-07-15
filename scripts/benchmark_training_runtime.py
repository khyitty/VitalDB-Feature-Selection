"""Audit hardware and benchmark short deterministic training-step workloads."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime_benchmark import audit_hardware, run_training_benchmark  # noqa: E402


def _integers(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("Provide comma-separated non-negative integers.")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/modeling/full"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/runtime_benchmark")
    )
    parser.add_argument("--thread-counts", type=_integers, default=(1, 2, 4, 8))
    parser.add_argument("--worker-counts", type=_integers, default=(0, 2))
    parser.add_argument("--measured-batches", type=int, default=20)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hardware-only", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hardware = audit_hardware(PROJECT_ROOT / "requirements.txt")
    with (args.output_dir / "hardware_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(hardware, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    if args.hardware_only:
        print(json.dumps({"hardware": hardware}, indent=2))
        return
    result = run_training_benchmark(
        args.dataset_dir,
        args.output_dir,
        hardware,
        thread_counts=args.thread_counts,
        worker_counts=args.worker_counts,
        measured_batches=args.measured_batches,
        warmup_batches=args.warmup_batches,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps({"hardware": hardware, "benchmark": result}, indent=2))


if __name__ == "__main__":
    main()
