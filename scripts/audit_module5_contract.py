"""Run the independent Module 5 environment contract audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl_training.module5_audit import run_module5_audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Module 5 action/reward/history/state contracts.")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/module5_independent_audit"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(json.dumps(run_module5_audit(args.output_dir, ROOT), indent=2))


if __name__ == "__main__":
    main()
