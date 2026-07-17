"""Run train-only patient-level Elastic Net stability selection."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_selection.elastic_net_stability import (  # noqa: E402
    StabilitySelectionConfig,
    run_elastic_net_stability,
)


def _l1_ratios(value: str) -> tuple[float, ...]:
    try:
        ratios = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("l1 ratios must be comma-separated numbers") from error
    if not ratios or any(ratio < 0.0 or ratio > 1.0 for ratio in ratios):
        raise argparse.ArgumentTypeError("l1 ratios must fall within [0, 1]")
    return ratios


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
        default=Path(
            "outputs/simulator_compatible_prediction_v2/elastic_net_stability"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-count", type=int, default=100)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--l1-ratios", type=_l1_ratios, default=(0.1, 0.5, 0.9, 1.0))
    parser.add_argument("--alpha-min", type=float, default=1e-4)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--alpha-count", type=int, default=9)
    parser.add_argument("--coefficient-tolerance", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=20_000)
    parser.add_argument("--optimization-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use 3 bootstraps and a compact train-only CV grid.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.alpha_min <= 0.0 or args.alpha_max < args.alpha_min or args.alpha_count <= 0:
        raise ValueError("Require 0 < alpha_min <= alpha_max and alpha_count > 0.")
    alphas = tuple(
        np.logspace(np.log10(args.alpha_min), np.log10(args.alpha_max), args.alpha_count)
    )
    l1_ratios = args.l1_ratios
    bootstrap_count = args.bootstrap_count
    cv_folds = args.cv_folds
    max_iter = args.max_iter
    optimization_tolerance = args.optimization_tolerance
    if args.smoke:
        alphas = (3e-2, 1e-1)
        l1_ratios = (1.0,)
        bootstrap_count = 3
        cv_folds = min(cv_folds, 3)
        max_iter = min(max_iter, 5_000)
        optimization_tolerance = max(optimization_tolerance, 1e-4)
    result = run_elastic_net_stability(
        StabilitySelectionConfig(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            bootstrap_count=bootstrap_count,
            cv_folds=cv_folds,
            l1_ratios=l1_ratios,
            alphas=alphas,
            coefficient_tolerance=args.coefficient_tolerance,
            max_iter=max_iter,
            optimization_tolerance=optimization_tolerance,
            smoke=args.smoke,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
