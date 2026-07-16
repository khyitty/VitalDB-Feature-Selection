"""Run four short synthetic PPO contract smokes on CPU by default."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl_training.smoke import run_all_smokes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic PPO smoke training; not a research comparison.")
    parser.add_argument("--timesteps", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ppo_smoke")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not 2000 <= args.timesteps <= 10_000:
        raise ValueError("Smoke timesteps must stay within 2,000--10,000.")
    summaries = run_all_smokes(
        args.output_dir, total_timesteps=args.timesteps, seed=args.seed, device=args.device
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
