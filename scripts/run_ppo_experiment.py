"""Initialize or execute the frozen 20-run CUDA PPO inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl_training.cohort import load_vitaldb_virtual_cohort
from src.rl_training.config import EXPERIMENT_SEEDS, POLICY_CONDITIONS, PPOConfig
from src.rl_training.experiment_protocol import run_experiment
from src.rl_training.manifests import (
    build_frozen_protocol,
    freeze_protocol,
    verify_protocol,
    write_policy_contract_artifacts,
)


def validate_confirmation(protocol: dict, confirmation: str | None) -> None:
    expected = protocol["confirmation_text"]
    if confirmation != expected:
        raise ValueError(
            f"Full training remains locked. Exact confirmation required: {expected}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize or resume frozen CUDA PPO runs; held-out test remains sealed."
    )
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/modeling/full")
    parser.add_argument("--protocol-dir", type=Path, default=ROOT / "outputs/ppo_protocol")
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs/ppo_control_comparison"
    )
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--confirmation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--condition", choices=POLICY_CONDITIONS)
    parser.add_argument("--seed", type=int, choices=EXPERIMENT_SEEDS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cohort = load_vitaldb_virtual_cohort(args.dataset_dir, ROOT)
    requested = build_frozen_protocol(repo_dir=ROOT, cohort=cohort, ppo=PPOConfig())
    protocol = freeze_protocol(requested, args.protocol_dir)
    verify_protocol(protocol)
    write_policy_contract_artifacts(
        protocol=protocol, cohort=cohort, output_dir=args.protocol_dir
    )
    print(json.dumps({
        "protocol_hash": protocol["protocol_hash"],
        "inventory_count": protocol["inventory_count"],
        "confirmation_text": protocol["confirmation_text"],
        "cohort_fingerprint": cohort.fingerprint,
        "test_cohort_accessed": False,
    }, indent=2))
    if args.initialize_only:
        return
    validate_confirmation(protocol, args.confirmation)
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Full PPO inventory requires an available CUDA device.")
    inventory = [
        item
        for item in protocol["inventory"]
        if (args.condition is None or item["condition"] == args.condition)
        and (args.seed is None or int(item["seed"]) == args.seed)
    ]
    started = time.perf_counter()
    for index, item in enumerate(inventory, start=1):
        result = run_experiment(
            protocol=protocol,
            condition=item["condition"],
            seed=int(item["seed"]),
            cohort=cohort,
            output_root=args.output_root,
            device="cuda",
        )
        elapsed = time.perf_counter() - started
        mean_seconds = elapsed / index
        remaining_seconds = mean_seconds * (len(inventory) - index)
        print(json.dumps({
            "progress": f"{index}/{len(inventory)}",
            "run_id": item["run_id"],
            "result": result,
            "elapsed_seconds": elapsed,
            "estimated_remaining_seconds": remaining_seconds,
        }, indent=2))


if __name__ == "__main__":
    main()
