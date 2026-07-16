"""Run validation-only paired PPO comparison and attention stability analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl_training.analysis import run_validation_analysis


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze only frozen PPO validation scenarios.")
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs/ppo_control_comparison"
    )
    parser.add_argument(
        "--analysis-dir", type=Path, default=ROOT / "outputs/ppo_validation_analysis"
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(json.dumps(run_validation_analysis(
        args.output_root,
        args.analysis_dir,
        replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    ), indent=2))


if __name__ == "__main__":
    main()
