"""Run validation-only contribution and attention-faithfulness diagnostics."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.faithfulness_audit import run_faithfulness_audit  # noqa: E402
from src.multiseed_attention_audit import DEFAULT_SEEDS  # noqa: E402


def _seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("Seeds must be non-empty and unique.")
    return seeds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-dir", type=Path, default=Path("outputs/ablations/no_bis_error")
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/modeling/full"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ablations/no_bis_error/faithfulness"),
    )
    parser.add_argument("--seeds", type=_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--permutation-repetitions", type=int, default=10)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = run_faithfulness_audit(
        args.root_dir,
        args.dataset_dir,
        args.output_dir,
        seeds=args.seeds,
        batch_size=args.batch_size,
        permutation_repetitions=args.permutation_repetitions,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
