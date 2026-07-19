"""Fresh-start, resumable execution for the primary-state PPO full study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from .cohort import CohortBundle
from .config import PrimaryStateProfile
from .full_protocol import FULL_PROFILES, verify_full_protocol
from .pilot_experiment import run_primary_state_experiment


def _assert_fresh_or_full_resume(
    run_dir: Path,
    *,
    protocol_hash: str,
    state_profile: str,
    seed: int,
) -> None:
    """Reject pilot checkpoints and unidentified artifacts before full initialization."""

    if not run_dir.exists():
        return
    config_path = run_dir / "config.json"
    files = [item for item in run_dir.rglob("*") if item.is_file()]
    if not config_path.is_file():
        if files:
            raise ValueError(
                "Full run directory contains artifacts without a full config; refusing "
                "pilot checkpoint reuse or provenance guessing."
            )
        return
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "workflow": "primary_state_ppo_full",
        "protocol_hash": protocol_hash,
        "state_profile": state_profile,
        "seed": seed,
        "initialization_source": "fresh_random",
        "pilot_checkpoint_used": False,
    }
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Existing run is not a compatible fresh-start full run; pilot reuse is "
            f"forbidden. Mismatches: {mismatches}."
        )


def run_primary_state_full(
    *,
    protocol: dict[str, Any],
    state_profile: str,
    seed: int,
    cohort: CohortBundle,
    output_root: Path,
    repo_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Run or resume one full identity from a full-only optimizer checkpoint."""

    verify_full_protocol(protocol)
    run_dir = output_root / state_profile / f"seed_{seed}"
    _assert_fresh_or_full_resume(
        run_dir,
        protocol_hash=protocol["protocol_hash"],
        state_profile=state_profile,
        seed=seed,
    )
    return run_primary_state_experiment(
        protocol=protocol,
        state_profile=cast(PrimaryStateProfile, state_profile),
        seed=seed,
        cohort=cohort,
        output_root=output_root,
        repo_dir=repo_dir,
        device=device,
        protocol_verifier=verify_full_protocol,
        allowed_profiles=FULL_PROFILES,
        workflow="primary_state_ppo_full",
        experiment_label="primary-state full protocol",
        config_extras={
            "initialization_source": "fresh_random",
            "pilot_checkpoint_used": False,
            "pilot_output_imported": False,
        },
    )
