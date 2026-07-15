"""Run train-only patient-grouped predictive feature selection for future BIS."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictive_feature_selection import (  # noqa: E402
    DEFAULT_RANDOM_SEED,
    DEFAULT_STABILITY_ITERATIONS,
    SelectionConfig,
    run_predictive_feature_selection,
)


def _features(value: str) -> tuple[str, ...]:
    features = tuple(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )
    if not features:
        raise argparse.ArgumentTypeError("Provide at least one protected feature name.")
    return features


def _stability_iterations(value: str) -> int:
    iterations = int(value)
    if iterations < 100:
        raise argparse.ArgumentTypeError(
            "Scientific stability selection requires at least 100 iterations."
        )
    return iterations


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--group-analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--internal-folds", type=int, default=3)
    parser.add_argument(
        "--stability-iterations",
        type=_stability_iterations,
        default=DEFAULT_STABILITY_ITERATIONS,
    )
    parser.add_argument("--subsample-fraction", type=float, default=0.5)
    parser.add_argument("--stable-threshold", type=float, default=0.7)
    parser.add_argument("--correlation-threshold", type=float, default=0.8)
    parser.add_argument("--tree-permutation-repeats", type=int, default=2)
    parser.add_argument("--tree-estimators", type=int, default=150)
    parser.add_argument("--tree-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--compute-shap", action="store_true")
    parser.add_argument("--shap-max-windows", type=int, default=5000)
    parser.add_argument("--protected-control-features", type=_features, default=())
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = run_predictive_feature_selection(
        SelectionConfig(
            dataset_dir=args.dataset_dir,
            group_analysis_dir=args.group_analysis_dir,
            output_dir=args.output_dir,
            random_seed=args.random_seed,
            internal_folds=args.internal_folds,
            stability_iterations=args.stability_iterations,
            subsample_fraction=args.subsample_fraction,
            stable_threshold=args.stable_threshold,
            correlation_threshold=args.correlation_threshold,
            tree_permutation_repeats=args.tree_permutation_repeats,
            tree_estimators=args.tree_estimators,
            tree_device=args.tree_device,
            compute_shap=args.compute_shap,
            shap_max_windows=args.shap_max_windows,
            protected_control_features=args.protected_control_features,
        )
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
