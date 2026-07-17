"""Run the validation-only GRU on the canonical simulator-compatible universe."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator_compatible_training import (  # noqa: E402
    validate_main_prediction_run,
    write_main_run_context,
)
from src.training import TrainingConfig, run_gru_training  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the locked main-rerun GRU configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/modeling/simulator_compatible_v2/full"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-num-threads", type=int)
    parser.add_argument("--torch-interop-threads", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--uniform-window-sampling", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    """Validate the scientific guard, train, and mark the new output family."""

    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    metadata = validate_main_prediction_run(
        args.dataset_dir, validation_only=args.validation_only
    )
    run_name = f"smoke_seed_{args.seed}" if args.smoke else f"seed_{args.seed}"
    output_dir = args.output_dir or Path(
        "outputs/simulator_compatible_prediction_v2/gru"
    ) / run_name
    config = TrainingConfig(
        dataset_dir=args.dataset_dir,
        output_dir=output_dir,
        seed=args.seed,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=min(args.max_epochs, 2) if args.smoke else args.max_epochs,
        patience=args.patience,
        gradient_clip_norm=args.gradient_clip,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        case_balanced_sampling=not args.uniform_window_sampling,
        num_workers=args.num_workers,
        torch_num_threads=args.torch_num_threads,
        torch_interop_threads=args.torch_interop_threads,
        smoke=args.smoke,
        evaluate_test=False,
        resume_checkpoint=args.resume,
    )
    result = run_gru_training(config)
    write_main_run_context(
        output_dir, model_name="gru", dataset_metadata=metadata
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
