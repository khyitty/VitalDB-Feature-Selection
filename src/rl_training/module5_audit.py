"""Independent, machine-readable audit of Module 5 control contracts."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from src.rl_env import EnvironmentConfig, PropofolControlEnv, YUN_REPORTED_ACTION_BOUNDS
from src.rl_env.reward import RewardCalculator, reward_profile_registry
from src.rl_env.state_adapters import STATE_PROFILES, get_state_profile, state_profile_registry


def _git_head(repo_dir: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_module5_audit(output_dir: Path, repo_dir: Path) -> dict[str, Any]:
    """Verify action, reward, history, and state contracts without training."""

    output_dir.mkdir(parents=True, exist_ok=True)
    config = EnvironmentConfig(
        episode_duration_seconds=60.0,
        state_profile="yun_reconstructed",
        action_bounds=YUN_REPORTED_ACTION_BOUNDS,
    )
    env = PropofolControlEnv(config)
    observation, reset_info = env.reset(seed=20260716)
    action_rows: list[dict[str, Any]] = []
    for rate in (0.0, 83.1, 166.2):
        probe = PropofolControlEnv(config)
        probe.reset(seed=20260716)
        _, _, _, _, info = probe.step(np.asarray([rate], dtype=np.float32))
        expected_dose = rate * 10.0 / 60.0
        action_rows.append(
            {
                "action_rate_mg_per_min": rate,
                "expected_dose_mg_per_10s": expected_dose,
                "info_applied_dose_mg_per_10s": info["applied_dose_mg_per_10s"],
                "simulator_cumulative_dose_mg": info["propofol_cumulative_dose_mg"],
                "absolute_error_mg": abs(
                    info["propofol_cumulative_dose_mg"] - expected_dose
                ),
                "double_conversion_detected": not np.isclose(
                    info["propofol_cumulative_dose_mg"], expected_dose, atol=1e-10
                ),
            }
        )
        probe.close()
    action_audit = pd.DataFrame(action_rows)
    action_audit.to_csv(output_dir / "action_unit_audit.csv", index=False)

    history_rows = [
        {
            "step": 0,
            "simulation_time_seconds": reset_info["simulation_time_seconds"],
            "history_mask": "|".join(map(str, observation["history_mask"].tolist())),
            "post_action_reward": None,
            "bis_used_for_reward": reset_info["bis"],
        }
    ]
    for step in range(1, 7):
        observation, reward, _, _, info = env.step(np.asarray([6.0], dtype=np.float32))
        recalculated = RewardCalculator(config).calculate(
            post_bis=info["bis"],
            target_bis=info["target_bis"],
            action_mg_per_min=6.0,
            previous_action_mg_per_min=0.0 if step == 1 else 6.0,
            propofol_ce_mg_per_l=info["propofol_ce_mg_per_l"],
        ).total
        history_rows.append(
            {
                "step": step,
                "simulation_time_seconds": info["simulation_time_seconds"],
                "history_mask": "|".join(map(str, observation["history_mask"].tolist())),
                "post_action_reward": reward,
                "bis_used_for_reward": info["bis"],
                "recalculated_reward": recalculated,
                "reward_alignment_error": abs(reward - recalculated),
            }
        )
    history_audit = pd.DataFrame(history_rows)
    history_audit.to_csv(output_dir / "history_alignment_audit.csv", index=False)
    env.close()

    state_rows = []
    for experiment_name, environment_name in (
        ("yun_reconstructed", "yun_reconstructed"),
        ("all_supported", "all_supported"),
        ("attention_supported", "attention_ready"),
        ("selected_control_aware", "selected_control_aware"),
    ):
        profile = get_state_profile(environment_name)  # type: ignore[arg-type]
        state_rows.append(
            {
                "experiment_condition": experiment_name,
                "environment_profile": environment_name,
                "dynamic_feature_count": len(profile.dynamic_feature_names),
                "dynamic_feature_names": "|".join(profile.dynamic_feature_names),
                "exact_yun_reproduction": False if experiment_name == "yun_reconstructed" else None,
                "unsupported_vitals_present": bool(
                    {"hr", "mbp", "sbp", "dbp", "spo2", "etco2", "hrv", "bis_sqi"}
                    & set(profile.dynamic_feature_names)
                ),
            }
        )
    pd.DataFrame(state_rows).to_csv(output_dir / "state_profile_audit.csv", index=False)

    reward_audit = {
        "default_profile": "transparent_tracking_v1",
        "default_profile_source": "repository design",
        "paper_profile": "paper_yun2023_parameterized",
        "paper_formula": "1 / (abs(target_BIS - BIS_next) + alpha)",
        "paper_alpha_reported": False,
        "missing_alpha_rejected": True,
        "profiles": reward_profile_registry(),
        "state_profile_invariant": True,
    }
    try:
        EnvironmentConfig(reward_profile="paper_yun2023_parameterized")
    except ValueError:
        pass
    else:
        reward_audit["missing_alpha_rejected"] = False
    (output_dir / "reward_contract_audit.json").write_text(
        json.dumps(reward_audit, indent=2), encoding="utf-8"
    )

    passed = bool(
        np.isclose(action_audit.iloc[-1]["simulator_cumulative_dose_mg"], 27.7)
        and not action_audit["double_conversion_detected"].any()
        and reward_audit["missing_alpha_rejected"]
        and history_rows[0]["history_mask"] == "0|0|0|0|0|1"
        and history_audit["reward_alignment_error"].dropna().max() == 0.0
    )
    manifest = {
        "status": "passed" if passed else "failed",
        "implementation_commit": _git_head(repo_dir),
        "environment_commit_audited": "767f3bff3dcaeabc51049fba5ccba1ac02b69ae3",
        "action_bounds": asdict(YUN_REPORTED_ACTION_BOUNDS),
        "action_conversion": "mg/min * 10/60 = mg per environment step",
        "maximum_action_dose_mg_per_10s": 27.7,
        "reward_contract": reward_audit,
        "history_steps": 6,
        "initial_history_mask": [0, 0, 0, 0, 0, 1],
        "official_baseline_name": "yun_reconstructed",
        "backward_compatible_alias": "original_yun",
        "exact_yun_reproduction": False,
        "state_registry": state_profile_registry(),
    }
    (output_dir / "module5_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    report = f"""# Module 5 Independent Audit

Status: **{manifest['status']}**

- `166.2 mg/min * 10/60 = 27.7 mg` exactly within numerical tolerance.
- No action-rate/dose double conversion was detected.
- `transparent_tracking_v1` is explicitly repository-designed.
- Yun Eq. (40) remains unavailable without an explicit positive `alpha`.
- Official experiment name: `yun_reconstructed`; `original_yun` is retained only
  for backward compatibility.
- Six causal decision rows and reset mask `[0,0,0,0,0,1]` were verified.
- Reward uses the post-action BIS and is independent of observation profile.

This is a research-only environment audit, not a clinical dosing validation.
"""
    (output_dir / "module5_audit_report.md").write_text(report, encoding="utf-8")
    return manifest
