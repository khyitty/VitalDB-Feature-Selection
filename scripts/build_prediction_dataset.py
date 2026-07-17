"""Command-line entry point for constructing future-BIS datasets."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PipelineConfig  # noqa: E402
from src.io import build_prediction_dataset  # noqa: E402
from src.prediction_feature_profiles import (  # noqa: E402
    FEATURE_PROFILES,
    SIMULATOR_COMPATIBLE_PROFILE,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the mutually exclusive pilot/full build mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true", help="Use the first 10 eligible cases.")
    mode.add_argument("--full", action="store_true", help="Use all eligible cases.")
    parser.add_argument(
        "--feature-profile",
        choices=tuple(FEATURE_PROFILES),
        default=SIMULATOR_COMPATIBLE_PROFILE,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--pkpd-history",
        type=Path,
        default=Path("data/raw/vitaldb_raw_100cases.csv"),
        help="Case-start raw Orchestra history used for causal Schnider/Minto reconstruction.",
    )
    parser.add_argument(
        "--split-reference-dir",
        type=Path,
        default=Path("data/modeling/full/splits"),
        help="Reuse the frozen case assignment; required for the main rerun.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Build the selected dataset mode and print a compact summary."""

    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    mode = "pilot" if args.pilot else "full"
    profile_root = (
        "simulator_compatible_v2"
        if args.feature_profile == SIMULATOR_COMPATIBLE_PROFILE
        else "legacy_physiological_exploratory"
    )
    output_dir = args.output_dir or Path("data/modeling") / profile_root / mode
    if output_dir.resolve() in {
        Path("data/modeling/pilot").resolve(),
        Path("data/modeling/full").resolve(),
        Path("data/modeling/simulator_compatible/pilot").resolve(),
        Path("data/modeling/simulator_compatible/full").resolve(),
    }:
        raise ValueError("Refusing to overwrite prior physiological-inclusive datasets.")
    split_reference = (
        args.split_reference_dir
        if args.feature_profile == SIMULATOR_COMPATIBLE_PROFILE
        else None
    )
    config = PipelineConfig(
        output_dir=output_dir,
        pkpd_history_path=args.pkpd_history,
        feature_profile=args.feature_profile,
        split_reference_dir=split_reference,
    )
    result = build_prediction_dataset(config, max_cases=10 if args.pilot else None)

    print(f"Dynamic features ({len(result.dynamic_features)}): {', '.join(result.dynamic_features)}")
    print(f"Static features ({len(result.static_features)}): {', '.join(result.static_features)}")
    print(f"Case counts: {result.case_counts}")
    print(f"Window counts: {result.window_counts}")
    for split_name in ("train", "val", "test"):
        print(f"{split_name} tensor shapes: {result.tensor_shapes[split_name]}")
        if split_name not in result.prevalence:
            print(f"{split_name} target prevalence: sealed")
            continue
        rates = result.prevalence[split_name]
        print(
            f"{split_name} prevalence: BIS<40={rates['low_bis']:.4f}, "
            f"BIS_40_to_60={rates['bis_40_to_60']:.4f}, "
            f"BIS>60={rates['high_bis']:.4f}"
        )
    print(f"Outputs: {result.output_dir.resolve()}")


if __name__ == "__main__":
    main()
