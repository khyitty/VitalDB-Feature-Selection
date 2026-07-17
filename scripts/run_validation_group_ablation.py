"""Run validation-only GRU screening for predefined dynamic feature groups."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_selection.validation_group_ablation import (  # noqa: E402
    CANDIDATE_FEATURES,
    ValidationAblationConfig,
    run_validation_group_ablation,
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
        default=Path("outputs/simulator_compatible_prediction_v2/group_ablation"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--candidate", choices=("all", *CANDIDATE_FEATURES), default="all"
    )
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = run_validation_group_ablation(
        ValidationAblationConfig(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            candidate=args.candidate,
            seed=args.seed,
            device=args.device,
            validation_only=args.validation_only,
            smoke=args.smoke,
            skip_completed=args.skip_completed,
        )
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed_candidates": result["completed_candidates"],
                "skipped_candidates": result["skipped_candidates"],
                "summary_candidate_count": len(result["summary"]),
                "test_used": result["test_used"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
