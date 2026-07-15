"""Audit the controlled seed-42 ablation that excludes ``bis_error``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.redundancy_audit import run_bis_error_ablation_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/modeling/full"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/ablations/no_bis_error")
    )
    parser.add_argument(
        "--original-gru-dir",
        type=Path,
        default=Path("outputs/baselines/gru/seed_42"),
    )
    parser.add_argument(
        "--original-attention-dir",
        type=Path,
        default=Path("outputs/attention/factorized_gru/seed_42"),
    )
    parser.add_argument(
        "--reduced-gru-dir",
        type=Path,
        default=Path("outputs/ablations/no_bis_error/gru/seed_42"),
    )
    parser.add_argument(
        "--reduced-attention-dir",
        type=Path,
        default=Path("outputs/ablations/no_bis_error/attention/seed_42"),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    result = run_bis_error_ablation_audit(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        original_gru_dir=args.original_gru_dir,
        original_attention_dir=args.original_attention_dir,
        reduced_gru_dir=args.reduced_gru_dir,
        reduced_attention_dir=args.reduced_attention_dir,
    )
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
