"""Run the final artifact and validation-only audit of the completed PPO pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl_training.cohort import load_vitaldb_virtual_cohort
from src.rl_training.pilot_final_audit import run_pilot_final_audit
from src.rl_training.pilot_protocol import load_frozen_pilot_protocol


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the completed 12-run PPO pilot.")
    base = ROOT / "outputs/ppo_primary_state_pilot"
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/modeling/full")
    parser.add_argument("--demographics-csv", type=Path)
    parser.add_argument("--project-data-root", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=base / "protocol/frozen_primary_state_pilot_protocol.json",
    )
    parser.add_argument("--runs-dir", type=Path, default=base / "runs")
    parser.add_argument("--analysis-dir", type=Path, default=base / "analysis")
    parser.add_argument("--audit-dir", type=Path, default=base / "final_audit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cohort = load_vitaldb_virtual_cohort(
        args.dataset_dir,
        demographics_csv=args.demographics_csv,
        project_data_root=args.project_data_root,
    )
    protocol = load_frozen_pilot_protocol(args.protocol)
    print(
        json.dumps(
            run_pilot_final_audit(
                protocol=protocol,
                output_root=args.runs_dir,
                existing_analysis_dir=args.analysis_dir,
                audit_dir=args.audit_dir,
                cohort=cohort,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
