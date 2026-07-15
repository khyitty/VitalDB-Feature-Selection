"""Install only missing non-PyTorch dependencies while retaining Colab PyTorch."""

from __future__ import annotations

import argparse
import importlib.metadata
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
    FROZEN_TEST_PROTECTED_DISTRIBUTIONS,
    frozen_test_runtime_versions,
    missing_colab_requirements,
    validate_colab_requirements,
    validate_pip_install_plan,
)

PROFILE_REQUIREMENTS = {
    "environment": PROJECT_ROOT / "requirements-colab.txt",
    "training": PROJECT_ROOT / "requirements-colab.txt",
    "analysis": PROJECT_ROOT / "requirements-colab.txt",
    "frozen-test": PROJECT_ROOT / "requirements-colab-test.txt",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path)
    parser.add_argument("--profile", choices=tuple(PROFILE_REQUIREMENTS))
    return parser.parse_args(argv)


def resolve_requirements(args: argparse.Namespace) -> tuple[Path, str]:
    """Resolve one explicit purpose-specific dependency set."""

    if args.requirements is not None and args.profile is not None:
        raise ValueError("Use either --requirements or --profile, not both.")
    if args.profile is not None:
        return PROFILE_REQUIREMENTS[args.profile], args.profile
    return (
        args.requirements or PROJECT_ROOT / "requirements-colab.txt",
        "custom" if args.requirements is not None else "environment",
    )


def main() -> None:
    args = parse_args()
    requirements_path, profile = resolve_requirements(args)
    validate_colab_requirements(requirements_path)
    torch_before = torch.__version__
    pandas_before = importlib.metadata.version("pandas")
    if profile == "frozen-test" and int(pandas_before.split(".", maxsplit=1)[0]) >= 3:
        raise RuntimeError(
            f"Frozen-test profile expected Colab pandas 2.x, found {pandas_before}."
        )
    missing = missing_colab_requirements(requirements_path)
    protected = (
        FROZEN_TEST_PROTECTED_DISTRIBUTIONS
        if profile == "frozen-test"
        else None
    )
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
            plan = json.loads(report_path.read_text(encoding="utf-8"))
            if protected is None:
                validate_pip_install_plan(plan)
            else:
                validate_pip_install_plan(plan, protected)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", *missing], check=True
        )
    if torch.__version__ != torch_before:
        raise RuntimeError("PyTorch changed during Colab dependency installation.")
    pandas_after = importlib.metadata.version("pandas")
    if profile == "frozen-test" and pandas_after != pandas_before:
        raise RuntimeError(
            f"pandas changed during frozen-test installation: {pandas_before} -> {pandas_after}."
        )
    runtime = frozen_test_runtime_versions() if profile == "frozen-test" else {}
    print(
        json.dumps(
            {
                "profile": profile,
                "requirements_file": str(requirements_path),
                "installed_missing_requirements": missing,
                "retained_torch_version": torch.__version__,
                "retained_pandas_version": pandas_after,
                "torch_version_cuda": torch.version.cuda,
                "torch_cuda_is_available": torch.cuda.is_available(),
                "torch_cuda_device_count": torch.cuda.device_count(),
                "torch_cuda_device_names": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
                **runtime,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
