"""Run the sealed-test, validation-only latest-BIS persistence baseline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.persistence_validation import (  # noqa: E402
    PersistenceValidationConfig,
    run_validation_persistence,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/modeling/simulator_compatible_v2/full"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/simulator_compatible_prediction_v2/persistence"),
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Required safety seal: load and evaluate val.npz only.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = run_validation_persistence(
        PersistenceValidationConfig(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            validation_only=args.validation_only,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
