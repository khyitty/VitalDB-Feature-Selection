"""Validate reused anchors and create the frozen-candidate retraining plan."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.frozen_candidate_retraining import build_retraining_plan  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-subsets", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--group-root", type=Path, required=True)
    parser.add_argument("--group-analysis-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-dir", type=Path, default=PROJECT_ROOT)
    return parser.parse_args(argv)


def main() -> None:
    """Validate source artifacts and write the non-executing experiment plan."""

    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = build_retraining_plan(
        candidate_path=args.candidate_subsets,
        dataset_dir=args.dataset_dir,
        group_root=args.group_root,
        group_analysis_dir=args.group_analysis_dir,
        output_root=args.output_root,
        repo_dir=args.repo_dir,
        active_commit=commit,
    )
    print(json.dumps(result["plan"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
