"""Freeze, execute, resume, and analyze the 20-run primary-state PPO full study."""

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
from src.rl_training.full_analysis import run_full_analysis
from src.rl_training.full_experiment import run_primary_state_full
from src.rl_training.full_protocol import (
    FULL_CONFIRMATION,
    FULL_PROFILES,
    FULL_PROTOCOL_FILENAME,
    FULL_SEEDS,
    build_full_protocol,
    freeze_full_protocol,
    load_frozen_full_protocol,
    select_full_inventory,
)
from src.rl_training.pilot_protocol import resolve_execution_device


DEFAULT_BASE = ROOT / "outputs/ppo_primary_state_full"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen 20-run primary-state PPO full validation study."
    )
    parser.add_argument(
        "--source-config", type=Path, default=ROOT / "configs/ppo_primary_state_full.json"
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
    parser.add_argument(
        "--backend-decision",
        type=Path,
        help="Merged engineering benchmark decision JSON required before first freeze.",
    )
    parser.add_argument("--profiles", nargs="+", choices=FULL_PROFILES)
    parser.add_argument("--seeds", nargs="+", type=int, choices=FULL_SEEDS)
    parser.add_argument("--confirmation")
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--bootstrap-repeats", type=int, default=5_000)
    return parser.parse_args(argv)


def validate_confirmation(confirmation: str | None) -> None:
    """Keep all 20 expensive runs locked by default."""

    if confirmation != FULL_CONFIRMATION:
        raise ValueError(f"Full training remains locked. Exact confirmation: {FULL_CONFIRMATION}")


def _load_or_freeze(args: argparse.Namespace, cohort) -> dict:
    frozen_path = args.protocol_dir / FULL_PROTOCOL_FILENAME
    if frozen_path.is_file():
        frozen = load_frozen_full_protocol(frozen_path)
        requested_device = frozen["execution_device"] if args.device == "auto" else args.device
        decision = frozen["backend_decision"]
    else:
        requested_device = resolve_execution_device(args.device)
        if args.backend_decision is None or not args.backend_decision.is_file():
            raise ValueError(
                "First full protocol freeze requires --backend-decision from the "
                "engineering benchmark analysis."
            )
        decision = json.loads(args.backend_decision.read_text(encoding="utf-8"))
    requested = build_full_protocol(
        source_path=args.source_config,
        repo_dir=ROOT,
        cohort=cohort,
        execution_device=requested_device,
        backend_decision=decision,
    )
    return freeze_full_protocol(
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
        "implementation_commit": protocol["implementation_commit"],
        "execution_device": protocol["execution_device"],
        "inventory_count": protocol["inventory_count"],
        "timesteps_per_run": protocol["ppo"]["total_timesteps"],
        "evaluation_frequency_timesteps": protocol["ppo"]["evaluation_frequency_timesteps"],
        "initialization_source": protocol["initialization"]["mode"],
        "pilot_checkpoint_reuse": protocol["initialization"]["pilot_checkpoint_reuse"],
        "cohort_fingerprint": cohort.fingerprint,
        "test_trajectory_accessed": False,
        "test_outcomes_evaluated": False,
    }
    print(json.dumps(preflight, indent=2), flush=True)
    if args.initialize_only:
        return
    if args.analysis_only:
        print(
            json.dumps(
                run_full_analysis(
                    protocol=protocol,
                    output_root=args.output_root,
                    analysis_dir=args.analysis_dir,
                    bootstrap_repeats=args.bootstrap_repeats,
                ),
                indent=2,
            ),
            flush=True,
        )
        return
    validate_confirmation(args.confirmation)
    inventory = select_full_inventory(protocol, profiles=args.profiles, seeds=args.seeds)
    started = time.perf_counter()
    for index, item in enumerate(inventory, start=1):
        result = run_primary_state_full(
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
                    "estimated_remaining_seconds": elapsed / index * (len(inventory) - index),
                },
                indent=2,
            ),
            flush=True,
        )
    print(
        json.dumps(
            run_full_analysis(
                protocol=protocol,
                output_root=args.output_root,
                analysis_dir=args.analysis_dir,
                bootstrap_repeats=args.bootstrap_repeats,
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
