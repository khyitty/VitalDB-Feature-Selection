"""Audit a Colab CUDA environment and stop if no GPU is assigned."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.colab_workflow import audit_colab_environment  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-no-cuda", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = audit_colab_environment(
        args.output, args.repo_dir, require_cuda=not args.allow_no_cuda
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
