"""Aggregate paired five-seed reduced GRU and factorized-attention runs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.multiseed_attention_audit import (  # noqa: E402
    DEFAULT_SEEDS,
    run_multiseed_attention_audit,
)


def _seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers.") from error
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("Seeds must be non-empty and unique.")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-dir", type=Path, default=Path("outputs/ablations/no_bis_error")
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/modeling/full"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ablations/no_bis_error/multiseed"),
    )
    parser.add_argument("--seeds", type=_seeds, default=DEFAULT_SEEDS)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    result = run_multiseed_attention_audit(
        root_dir=args.root_dir,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
    )
    print(json.dumps(result["performance"], indent=2))
    print(json.dumps(result["feature_stability"], indent=2))
    print(json.dumps(result["patient_bootstrap"], indent=2))


if __name__ == "__main__":
    main()
