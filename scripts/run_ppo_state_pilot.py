"""Freeze, execute, resume, and analyze the common-MLP PPO state pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl_training.cohort import load_vitaldb_virtual_cohort
from src.rl_training.pilot_analysis import run_pilot_analysis
from src.rl_training.pilot_experiment import run_primary_state_pilot
from src.rl_training.pilot_protocol import (
    PILOT_PROFILES,
    PILOT_PROTOCOL_FILENAME,
    PILOT_SEEDS,
    build_pilot_protocol,
    freeze_pilot_protocol,
    load_frozen_pilot_protocol,
    resolve_execution_device,
    select_inventory,
)


DEFAULT_BASE = ROOT / "outputs/ppo_primary_state_pilot"


def validate_confirmation(protocol: dict, confirmation: str | None) -> None:
    """Require the exact generated phrase before any non-smoke training."""

    expected = protocol["confirmation_text"]
    if confirmation != expected:
        raise ValueError(f"Pilot training remains locked. Exact confirmation: {expected}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen 12-run primary-state PPO pilot; test trajectories remain sealed."
        )
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        default=ROOT / "configs/ppo_primary_state_pilot.json",
    )
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/modeling/full")
    parser.add_argument("--demographics-csv", type=Path)
    parser.add_argument("--project-data-root", type=Path)
    parser.add_argument(
        "--missing-demographics-policy", choices=("error", "train_impute"), default="error"
    )
    parser.add_argument("--allow-official-demographics-download", action="store_true")
    parser.add_argument("--official-demographics-cache", type=Path)
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_BASE / "protocol")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_BASE / "runs")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_BASE / "analysis")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--profiles", nargs="+", choices=PILOT_PROFILES)
    parser.add_argument("--seeds", nargs="+", type=int, choices=PILOT_SEEDS)
    parser.add_argument("--confirmation")
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    return parser.parse_args(argv)


def _load_or_freeze(args: argparse.Namespace, cohort) -> dict:
    frozen_path = args.protocol_dir / PILOT_PROTOCOL_FILENAME
    if frozen_path.is_file():
        frozen = load_frozen_pilot_protocol(frozen_path)
        requested_device = (
            frozen["execution_device"] if args.device == "auto" else args.device
        )
    else:
        requested_device = resolve_execution_device(args.device)
    requested = build_pilot_protocol(
        source_path=args.source_config,
        repo_dir=ROOT,
        cohort=cohort,
        execution_device=requested_device,
    )
    return freeze_pilot_protocol(
        requested,
        args.protocol_dir,
        run_output_root=args.output_root,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.initialize_only and args.analysis_only:
        raise ValueError("--initialize-only and --analysis-only are mutually exclusive.")
    cohort = load_vitaldb_virtual_cohort(
        args.dataset_dir,
        demographics_csv=args.demographics_csv,
        project_data_root=args.project_data_root,
        missing_policy=args.missing_demographics_policy,
        allow_official_demographics_download=args.allow_official_demographics_download,
        official_demographics_cache=args.official_demographics_cache,
    )
    protocol = _load_or_freeze(args, cohort)
    preflight = {
        "protocol_hash": protocol["protocol_hash"],
        "protocol_path": str((args.protocol_dir / PILOT_PROTOCOL_FILENAME).resolve()),
        "implementation_commit": protocol["implementation_commit"],
        "execution_device": protocol["execution_device"],
        "inventory_count": protocol["inventory_count"],
        "profiles": protocol["profiles"],
        "seeds": protocol["seeds"],
        "timesteps_per_run": protocol["ppo"]["total_timesteps"],
        "evaluation_frequency_timesteps": protocol["ppo"][
            "evaluation_frequency_timesteps"
        ],
        "cohort_fingerprint": cohort.fingerprint,
        "cohort_counts": protocol["cohort_contract"]["case_counts"],
        "test_trajectory_accessed": False,
        "test_outcomes_evaluated": False,
    }
    print(json.dumps(preflight, indent=2), flush=True)
    if args.initialize_only:
        return
    if args.analysis_only:
        print(
            json.dumps(
                run_pilot_analysis(
                    protocol=protocol,
                    output_root=args.output_root,
                    analysis_dir=args.analysis_dir,
                ),
                indent=2,
            ),
            flush=True,
        )
        return

    validate_confirmation(protocol, args.confirmation)
    inventory = select_inventory(
        protocol,
        profiles=args.profiles,
        seeds=args.seeds,
    )
    started = time.perf_counter()
    for index, item in enumerate(inventory, start=1):
        result = run_primary_state_pilot(
            protocol=protocol,
            state_profile=item["state_profile"],
            seed=int(item["seed"]),
            cohort=cohort,
            output_root=args.output_root,
            repo_dir=ROOT,
            device=protocol["execution_device"],
        )
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(inventory)}",
                    "run_id": item["run_id"],
                    "result": result,
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": (
                        elapsed / index * (len(inventory) - index)
                    ),
                },
                indent=2,
            ),
            flush=True,
        )
    print(
        json.dumps(
            run_pilot_analysis(
                protocol=protocol,
                output_root=args.output_root,
                analysis_dir=args.analysis_dir,
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
