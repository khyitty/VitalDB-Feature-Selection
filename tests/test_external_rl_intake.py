"""Tests for static and archive-safe external RL package intake."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.external_rl_intake import validate_external_rl_package


COMPLETE_RL_SOURCE = '''
import gymnasium as gym
from gymnasium import spaces
import torch

TRAIN_CASES = [1, 2]
VAL_CASES = [3]
TEST_CASES = [4]
SEED = 42
ACTION_UNIT = "mg/kg/h"

class PropofolEnv(gym.Env):
    observation_space = spaces.Box(low=-10.0, high=10.0, shape=(8,))
    action_space = spaces.Box(low=0.0, high=20.0, shape=(1,))
    def reset(self, seed=None, options=None):
        return [0.0] * 8, {}
    def step(self, action):
        reward = -abs(action[0])
        terminated = False
        truncated = False
        return [0.0] * 8, reward, terminated, truncated, {}

class PKPDSimulator:
    pass

class ActorNetwork:
    pass

class CriticNetwork:
    pass

class ReplayBuffer:
    pass

class PPOAgent:
    pass

def evaluate_policy():
    metrics = {"mae": 1.0, "rmse": 2.0, "episode_return": 3.0}
    return metrics

def main():
    torch.manual_seed(SEED)
'''


def _complete_package(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "rl_system.py").write_text(COMPLETE_RL_SOURCE, encoding="utf-8")
    (root / "baseline_config.json").write_text(json.dumps({"algorithm": "PPO"}), encoding="utf-8")
    (root / "patient_split.csv").write_text("case_id,split\n1,train\n4,test\n", encoding="utf-8")
    (root / "baseline_policy.pt").write_bytes(b"checkpoint")


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_complete_static_package_is_ready_and_source_is_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "external"
    _complete_package(source)
    before = _tree_hash(source)
    report = validate_external_rl_package(source, tmp_path / "report")
    after = _tree_hash(source)
    discovered = pd.read_csv(tmp_path / "report" / "discovered_interfaces.csv")
    assert report["status"] == "ready_for_adapter"
    assert report["external_code_imported"] is False
    assert report["external_code_executed"] is False
    assert report["rl_training_started"] is False
    assert report["missing_requirements"] == []
    assert before == after
    assert {"environment_class", "reward_function", "action_unit", "pk_pd_simulator"}.issubset(set(discovered["requirement"]))


def test_missing_environment_is_blocked(tmp_path: Path) -> None:
    source = tmp_path / "empty"
    source.mkdir()
    (source / "helpers.py").write_text("def reset():\n    pass\n", encoding="utf-8")
    report = validate_external_rl_package(source, tmp_path / "report")
    assert report["status"] == "blocked_missing_components"
    assert "environment_class" in report["missing_requirements"]
    assert "step_method" in report["missing_requirements"]


def test_environment_with_missing_reward_and_action_unit_is_partial(tmp_path: Path) -> None:
    source = tmp_path / "partial"
    source.mkdir()
    (source / "env.py").write_text(
        "class SimpleEnv:\n"
        "    observation_space = {'bis': 0}\n"
        "    action_space = Box(low=0, high=1)\n"
        "    def reset(self): return 0\n"
        "    def step(self, action): return 0\n",
        encoding="utf-8",
    )
    report = validate_external_rl_package(source, tmp_path / "report")
    assert report["status"] == "partially_ready"
    assert "reward_function" in report["missing_requirements"]
    assert "action_unit" in report["missing_requirements"]


def test_safe_zip_is_inspected_without_modifying_archive(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _complete_package(source)
    archive = tmp_path / "external.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in source.rglob("*"):
            if path.is_file():
                handle.write(path, path.relative_to(source).as_posix())
    before = archive.read_bytes()
    report = validate_external_rl_package(archive, tmp_path / "report")
    assert report["status"] == "ready_for_adapter"
    assert archive.read_bytes() == before


def test_zip_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.py", "raise RuntimeError('must not execute')")
    escaped = tmp_path / "escape.py"
    with pytest.raises(ValueError, match="Unsafe archive"):
        validate_external_rl_package(archive, tmp_path / "report")
    assert not escaped.exists()
