"""Run persistence or compact non-attention GRU future-BIS baselines."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import (  # noqa: E402
    TrainingConfig,
    run_gru_training,
    run_persistence_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="baseline", required=True)

    persistence = subparsers.add_parser("persistence")
    persistence.add_argument(
        "--dataset-dir", type=Path, default=Path("data/modeling/full")
    )

    gru = subparsers.add_parser("gru")
    gru.add_argument("--dataset-dir", type=Path, default=Path("data/modeling/full"))
    gru.add_argument("--seed", type=int, default=42)
    gru.add_argument("--device", default="auto")
    gru.add_argument("--learning-rate", type=float, default=1e-3)
    gru.add_argument("--weight-decay", type=float, default=1e-4)
    gru.add_argument("--batch-size", type=int, default=256)
    gru.add_argument("--max-epochs", type=int, default=50)
    gru.add_argument("--patience", type=int, default=8)
    gru.add_argument("--gradient-clip", type=float, default=1.0)
    gru.add_argument("--hidden-size", type=int, default=64)
    gru.add_argument("--dropout", type=float, default=0.0)
    gru.add_argument("--num-workers", type=int, default=0)
    gru.add_argument("--resume", type=Path)
    gru.add_argument("--smoke", action="store_true")
    gru.add_argument(
        "--uniform-window-sampling",
        action="store_true",
        help="Disable default equal-case expected sampling mass.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.baseline == "persistence":
        result = run_persistence_baseline(
            args.dataset_dir, Path("outputs/baselines/persistence")
        )
        print(json.dumps(result, indent=2))
        return

    run_name = f"smoke_seed_{args.seed}" if args.smoke else f"seed_{args.seed}"
    config = TrainingConfig(
        dataset_dir=args.dataset_dir,
        output_dir=Path("outputs/baselines/gru") / run_name,
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
        smoke=args.smoke,
        resume_checkpoint=args.resume,
    )
    result = run_gru_training(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

