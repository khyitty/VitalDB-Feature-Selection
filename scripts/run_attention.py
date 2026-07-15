"""Run the explicit factorized-attention GRU future-BIS model."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attention_training import (  # noqa: E402
    AttentionTrainingConfig,
    run_attention_training,
)


def _feature_names(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise argparse.ArgumentTypeError("Provide at least one feature name.")
    return names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/modeling/full"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--feature-token-dim", type=int, default=16)
    parser.add_argument("--static-context-dim", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--prediction-hidden-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-num-threads", type=int)
    parser.add_argument("--torch-interop-threads", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Train and extract validation attention without loading the test split.",
    )
    feature_group = parser.add_mutually_exclusive_group()
    feature_group.add_argument("--dynamic-features", type=_feature_names)
    feature_group.add_argument("--exclude-dynamic-features", type=_feature_names)
    parser.add_argument(
        "--uniform-window-sampling",
        action="store_true",
        help="Disable default equal-case expected sampling mass.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_name = f"smoke_seed_{args.seed}" if args.smoke else f"seed_{args.seed}"
    output_dir = args.output_dir or Path("outputs/attention/factorized_gru") / run_name
    config = AttentionTrainingConfig(
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
        static_hidden_size=args.static_context_dim,
        prediction_hidden_size=args.prediction_hidden_size,
        dropout=args.dropout,
        case_balanced_sampling=not args.uniform_window_sampling,
        num_workers=args.num_workers,
        torch_num_threads=args.torch_num_threads,
        torch_interop_threads=args.torch_interop_threads,
        smoke=args.smoke,
        evaluate_test=not args.validation_only,
        resume_checkpoint=args.resume,
        feature_token_embedding_dim=args.feature_token_dim,
        static_context_dim=args.static_context_dim,
        dynamic_features=args.dynamic_features,
        exclude_dynamic_features=args.exclude_dynamic_features or (),
    )
    result = run_attention_training(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
