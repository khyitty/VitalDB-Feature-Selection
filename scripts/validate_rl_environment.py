"""Run training-free synthetic validation of the Gymnasium RL environment."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pkpd.validation import SYNTHETIC_PATIENTS
from src.rl_env.state_adapters import STATE_PROFILES
from src.rl_env.validation import ValidationConfig, run_validation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the research-only Gymnasium propofol-control environment."
    )
    parser.add_argument("--state-profile", choices=sorted(STATE_PROFILES), default="attention_ready")
    parser.add_argument("--patient-profile", choices=sorted(SYNTHETIC_PATIENTS), default="middle_male")
    parser.add_argument("--target-bis", type=float, default=50.0)
    parser.add_argument("--episode-duration-seconds", type=float, default=600.0)
    parser.add_argument(
        "--action-schedule",
        choices=("zero", "low", "moderate", "step", "random"),
        default="step",
    )
    parser.add_argument(
        "--remifentanil-schedule",
        choices=("off", "constant", "piecewise"),
        default="piecewise",
    )
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--action-bounds-profile",
        choices=("yun2023_converted", "synthetic_nonclinical_v1"),
        default="synthetic_nonclinical_v1",
    )
    parser.add_argument(
        "--reward-profile",
        choices=("transparent_tracking_v1", "paper_yun2023_parameterized"),
        default="transparent_tracking_v1",
    )
    parser.add_argument("--paper-reward-alpha", type=float)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/rl_environment_validation",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = ValidationConfig(
        state_profile=args.state_profile,
        patient_profile=args.patient_profile,
        target_bis=args.target_bis,
        episode_duration_seconds=args.episode_duration_seconds,
        action_schedule=args.action_schedule,
        remifentanil_schedule=args.remifentanil_schedule,
        deterministic=not args.stochastic,
        action_bounds_profile=args.action_bounds_profile,
        reward_profile=args.reward_profile,
        paper_reward_alpha=args.paper_reward_alpha,
        seed=args.seed,
    )
    summary = run_validation(config, args.output_dir, ROOT)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
