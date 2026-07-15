"""Build the predictive-stage handoff package from frozen references."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictive_stage_handoff import build_predictive_stage_handoff  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/modeling/full"))
    parser.add_argument(
        "--decision",
        type=Path,
        default=Path(
            "outputs/frozen_predictive_decision_30s/frozen_predictive_decision.json"
        ),
    )
    parser.add_argument("--checkpoint-inventory", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/predictive_stage_handoff")
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = build_predictive_stage_handoff(
        repo_dir=args.repo_dir,
        dataset_dir=args.dataset_dir,
        decision_path=args.decision,
        output_dir=args.output_dir,
        checkpoint_inventory_path=args.checkpoint_inventory,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
