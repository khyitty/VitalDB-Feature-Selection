"""Install only missing non-PyTorch dependencies while retaining Colab PyTorch."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.colab_workflow import (  # noqa: E402
    missing_colab_requirements,
    validate_colab_requirements,
    validate_pip_install_plan,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requirements", type=Path, default=PROJECT_ROOT / "requirements-colab.txt"
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    validate_colab_requirements(args.requirements)
    torch_before = torch.__version__
    missing = missing_colab_requirements(args.requirements)
    if missing:
        with tempfile.TemporaryDirectory() as temporary_dir:
            report_path = Path(temporary_dir) / "pip-report.json"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--dry-run",
                    "--report",
                    str(report_path),
                    *missing,
                ],
                check=True,
            )
            validate_pip_install_plan(
                json.loads(report_path.read_text(encoding="utf-8"))
            )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", *missing], check=True
        )
    if torch.__version__ != torch_before:
        raise RuntimeError("PyTorch changed during Colab dependency installation.")
    print(
        json.dumps(
            {
                "installed_missing_requirements": missing,
                "retained_torch_version": torch.__version__,
                "torch_version_cuda": torch.version.cuda,
                "torch_cuda_is_available": torch.cuda.is_available(),
                "torch_cuda_device_count": torch.cuda.device_count(),
                "torch_cuda_device_names": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
