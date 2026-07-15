"""Tests for the minimal frozen-test Colab dependency profile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.install_colab_dependencies import parse_args, resolve_requirements
from src.colab_workflow import (
    FROZEN_TEST_PROTECTED_DISTRIBUTIONS,
    frozen_test_runtime_versions,
    validate_colab_requirements,
    validate_pip_install_plan,
)


def test_frozen_test_profile_excludes_download_and_protected_packages() -> None:
    path = Path("requirements-colab-test.txt")
    requirements = validate_colab_requirements(path)
    lowered = "\n".join(requirements).lower()
    for forbidden in ("vitaldb", "wfdb", "torch", "pandas"):
        assert forbidden not in lowered
    resolved, profile = resolve_requirements(parse_args(["--profile", "frozen-test"]))
    assert resolved == Path(__file__).resolve().parents[1] / path
    assert profile == "frozen-test"


def test_frozen_test_plan_rejects_pandas_or_torch_replacement() -> None:
    for distribution in ("pandas", "torch", "torchvision"):
        with pytest.raises(RuntimeError, match="installation aborted"):
            validate_pip_install_plan(
                {"install": [{"metadata": {"name": distribution}}]},
                FROZEN_TEST_PROTECTED_DISTRIBUTIONS,
            )
    validate_pip_install_plan(
        {"install": [{"metadata": {"name": "matplotlib"}}]},
        FROZEN_TEST_PROTECTED_DISTRIBUTIONS,
    )


def test_frozen_test_runtime_imports_are_available_without_vitaldb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.colab_workflow.pd.__version__", "2.2.2")
    versions = frozen_test_runtime_versions()
    assert versions["pandas_major_version"] < 3
    assert versions["torch_version"]
    assert versions["vitaldb_required"] is False
    assert versions["wfdb_required"] is False
    assert set(versions["package_versions"]) == {
        "numpy",
        "scipy",
        "pandas",
        "torch",
        "matplotlib",
        "sklearn",
    }


def test_dependency_profile_and_requirements_are_mutually_exclusive() -> None:
    args = parse_args(
        ["--profile", "frozen-test", "--requirements", "requirements-colab-test.txt"]
    )
    with pytest.raises(ValueError, match="either"):
        resolve_requirements(args)
