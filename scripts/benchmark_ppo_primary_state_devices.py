"""Run or merge the engineering-only primary-state PPO device benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl_training.cohort import load_vitaldb_virtual_cohort
from src.rl_training.device_benchmark import analyze_device_benchmarks, run_device_benchmark


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CPU/CUDA PPO wall-clock performance.")
    parser.add_argument("--source-config", type=Path, default=ROOT / "configs/ppo_primary_state_full.json")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/modeling/full")
    parser.add_argument("--demographics-csv", type=Path)
    parser.add_argument("--project-data-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ppo_device_benchmark")
    parser.add_argument("--analyze", nargs="+", type=Path, metavar="RESULT_CSV")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.analyze:
        print(json.dumps(analyze_device_benchmarks(result_files=args.analyze, output_dir=args.output_dir / "analysis"), indent=2))
        return
    if args.device is None:
        raise ValueError("Specify --device for a benchmark run or --analyze for result merging.")
    cohort = load_vitaldb_virtual_cohort(
        args.dataset_dir,
        demographics_csv=args.demographics_csv,
        project_data_root=args.project_data_root,
    )
    print(
        json.dumps(
            run_device_benchmark(
                source_path=args.source_config,
                repo_dir=ROOT,
                cohort=cohort,
                output_root=args.output_dir,
                device=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
