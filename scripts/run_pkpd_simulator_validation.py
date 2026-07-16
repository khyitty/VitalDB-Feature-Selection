"""Run standalone synthetic validation of the PK-PD patient simulator."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pkpd.validation import SYNTHETIC_PATIENTS, ValidationConfig, run_validation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the research-only Schnider/Minto/Yun PK-PD simulator."
    )
    parser.add_argument("--patient", choices=sorted(SYNTHETIC_PATIENTS), default="middle_male")
    parser.add_argument("--duration-seconds", type=float, default=1800.0)
    parser.add_argument("--internal-dt-seconds", type=float, default=1.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--integrator", choices=("exact", "solve_ivp"), default="exact")
    parser.add_argument(
        "--propofol-schedule",
        choices=("induction_maintenance_recovery", "constant", "off"),
        default="induction_maintenance_recovery",
    )
    parser.add_argument(
        "--remifentanil-schedule",
        choices=("piecewise", "constant", "off"),
        default="piecewise",
    )
    parser.add_argument("--propofol-rate-mg-per-min", type=float, default=7.0)
    parser.add_argument("--remifentanil-rate-micrograms-per-min", type=float, default=6.0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/pkpd_simulator_validation",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = ValidationConfig(
        patient_name=args.patient,
        duration_seconds=args.duration_seconds,
        internal_dt_seconds=args.internal_dt_seconds,
        sample_interval_seconds=args.sample_interval_seconds,
        seed=args.seed,
        deterministic=not args.stochastic,
        integrator=args.integrator,
        propofol_schedule=args.propofol_schedule,
        remifentanil_schedule=args.remifentanil_schedule,
        propofol_constant_rate_mg_per_min=args.propofol_rate_mg_per_min,
        remifentanil_constant_rate_micrograms_per_min=(
            args.remifentanil_rate_micrograms_per_min
        ),
    )
    summary = run_validation(config, args.output_dir, ROOT)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
