"""Preflight or run the guarded one-time frozen predictive test evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.frozen_predictive_test_evaluation import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    prepare_test_preflight,
    run_frozen_predictive_test_evaluation,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-template-dir", type=Path, required=True)
    parser.add_argument("--decision-dir", type=Path, required=True)
    parser.add_argument("--analysis-manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--strict-root", type=Path, required=True)
    parser.add_argument("--full17-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--confirmation", default="")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Freeze the decision and list checkpoints without opening test data.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    common = {
        "decision_template_dir": args.decision_template_dir,
        "decision_dir": args.decision_dir,
        "source_analysis_manifest": args.analysis_manifest,
        "dataset_dir": args.dataset_dir,
        "strict_root": args.strict_root,
        "full17_root": args.full17_root,
        "output_dir": args.output_dir,
    }
    if args.preflight_only:
        result = prepare_test_preflight(**common)
    else:
        result = run_frozen_predictive_test_evaluation(
            **common,
            confirmation=args.confirmation,
            device=args.device,
            batch_size=args.batch_size,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
