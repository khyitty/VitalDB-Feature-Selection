"""Audit one completed full FactorizedAttentionGRU run."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attention_audit import run_attention_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/attention/factorized_gru/seed_42"),
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/modeling/full"))
    parser.add_argument(
        "--baselines-dir", type=Path, default=Path("outputs/baselines")
    )
    parser.add_argument(
        "--gru-run-dir",
        type=Path,
        help="Override the default outputs/baselines/gru/seed_42 comparison run.",
    )
    parser.add_argument("--command-wall-seconds", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    result = run_attention_audit(
        run_dir=args.run_dir,
        dataset_dir=args.dataset_dir,
        baselines_dir=args.baselines_dir,
        command_wall_seconds=args.command_wall_seconds,
        gru_run_dir=args.gru_run_dir,
    )
    print(json.dumps(result["result_classification"], indent=2))
    print(json.dumps(result["runtime"], indent=2))


if __name__ == "__main__":
    main()
