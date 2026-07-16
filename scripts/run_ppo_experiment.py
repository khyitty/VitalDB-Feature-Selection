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
    parser.add_argument(
        "--demographics-csv",
        type=Path,
        help="Explicit case-level demographics source; never inferred from the repository clone.",
    )
    parser.add_argument(
        "--project-data-root",
        type=Path,
        help="Project data root used to resolve metadata-recorded source filenames.",
    )
    parser.add_argument(
        "--missing-demographics-policy",
        choices=("error", "train_impute"),
        default="error",
    )
    parser.add_argument(
        "--allow-official-demographics-download",
        action="store_true",
        help=(
            "If local resolution fails, download only official VitalDB clinical "
            "metadata and filter it to frozen split case IDs."
        ),
    )
    parser.add_argument(
        "--official-demographics-cache",
        type=Path,
        help="Persistent CSV cache for the explicitly enabled official metadata fallback.",
    )
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
    cohort = load_vitaldb_virtual_cohort(
        args.dataset_dir,
        demographics_csv=args.demographics_csv,
        project_data_root=args.project_data_root,
        missing_policy=args.missing_demographics_policy,
        allow_official_demographics_download=args.allow_official_demographics_download,
        official_demographics_cache=args.official_demographics_cache,
    )
    requested = build_frozen_protocol(repo_dir=ROOT, cohort=cohort, ppo=PPOConfig())
    protocol = freeze_protocol(
        requested, args.protocol_dir, run_output_root=args.output_root
    )
    verify_protocol(protocol)
    write_policy_contract_artifacts(
        protocol=protocol, cohort=cohort, output_dir=args.protocol_dir
    )
    print(json.dumps({
        "stage": "initialize-only preflight" if args.initialize_only else "full-training preflight",
        "repository_head": requested["implementation_commit_at_creation"],
        "dataset_path": str(args.dataset_dir.resolve()),
        "dataset_entries": sorted(path.name for path in args.dataset_dir.iterdir()),
        "protocol_hash": protocol["protocol_hash"],
        "inventory_count": protocol["inventory_count"],
        "confirmation_text": protocol["confirmation_text"],
        "cohort_fingerprint": cohort.fingerprint,
        "demographics_source": cohort.demographics_source,
        "demographics_source_path": cohort.access_manifest["selected_demographics_path"],
        "demographics_source_kind": cohort.demographics_source_kind,
        "official_clinical_metadata": cohort.access_manifest[
            "official_clinical_metadata"
        ],
        "demographics_source_columns": list(cohort.demographics_source_columns),
        "required_demographic_columns": ["caseid", "age", "sex", "height", "weight"],
        "missing_demographic_counts": cohort.missing_demographics,
        "cohort_split_counts": cohort.access_manifest["split_counts"],
        "cohort_split_overlaps": cohort.access_manifest["split_overlaps"],
        "test_split_membership_loaded": True,
        "test_demographics_loaded": True,
        "test_trajectory_accessed": False,
        "test_outcomes_evaluated": False,
        "test_policy_rollout_performed": False,
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
