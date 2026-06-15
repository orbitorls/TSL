"""Training CLI entrypoint for the canonical src.train pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from src.train.config import get_config_from_args
from src.train.pipeline import run_training_pipeline

SUPPORTED_PRIMARY_METRICS = ("macro_f1",)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TSL-51 Thai Sign Language Training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="tsl51_user_sign",
        choices=["tsl51_user_sign", "tsl51_expert", "tsl51_expert_full", "tsl51_combined", "tsl51_full", "local"],
        help="Dataset to use (default: tsl51_user_sign)",
    )
    parser.add_argument("--data-path", type=str, default=None, help="Path to local dataset for --dataset local")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to use")
    parser.add_argument("--folds", type=int, default=5, help="Compatibility option stored in config")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--layers", type=int, default=3, help="Number of model layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument(
        "--model",
        type=str,
        default="gru",
        choices=["mlp", "gru", "mopgru", "hybrid", "ctc"],
        help="Model architecture",
    )
    parser.add_argument(
        "--feature-level",
        type=str,
        default="basic",
        choices=["basic", "finger", "full", "face"],
        help="Feature level; canonical training currently supports basic",
    )
    parser.add_argument("--target-frames", type=int, default=30, help="Target frames for sequence models")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum training epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-cache", action="store_true", help="Do not use local dataset cache")
    parser.add_argument("--force-download", action="store_true", help="Force dataset download/rebuild")
    parser.add_argument("--augment", type=int, default=0, help="Train-only augmentation factor")
    parser.add_argument(
        "--include-augmented",
        action="store_true",
        help="Include pre-augmented expert data for --dataset tsl51_expert",
    )
    parser.add_argument("--require-cuda", action="store_true", help="Abort if CUDA is unavailable")
    parser.add_argument("--smoke", action="store_true", help="Fast CPU smoke run")
    parser.add_argument("--test-split", type=float, default=0.15, help="Grouped test holdout fraction")
    parser.add_argument("--val-size", type=float, default=0.15, help="Grouped validation holdout fraction")
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="video_family_holdout",
        choices=["video_family_holdout", "video_family_grouped", "random"],
        help="Split strategy for canonical training orchestration",
    )
    parser.add_argument(
        "--primary-metric",
        type=str,
        default="macro_f1",
        help="Primary selection metric for canonical training (default: macro_f1)",
    )
    parser.add_argument(
        "--real-world-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require grouped video-family splitting safeguards (default: enabled)",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for canonical artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.require_cuda and not torch.cuda.is_available():
        parser.error("CUDA GPU not available and --require-cuda was set")
    if args.dataset == "local" and args.data_path is None:
        parser.error("--data-path is required when using --dataset local")
    if args.data_path is not None:
        data_path = Path(args.data_path)
        if args.dataset == "local" and not data_path.exists():
            parser.error(f"local dataset not found: {data_path}")
        args.data_path = data_path

    if args.smoke:
        args.epochs = 1
        args.batch = min(args.batch, 16)
        if args.samples is None or args.samples > 10:
            args.samples = 10

    args.primary_metric = str(args.primary_metric).strip().lower()
    if args.primary_metric not in SUPPORTED_PRIMARY_METRICS:
        supported = ", ".join(SUPPORTED_PRIMARY_METRICS)
        if args.real_world_mode:
            parser.error(
                "real-world mode currently supports only "
                f"--primary-metric {supported}; got {args.primary_metric!r}"
            )
        parser.error(
            f"unsupported --primary-metric {args.primary_metric!r}; supported values: {supported}"
        )

    training_config = get_config_from_args(args)
    # Keep CLI-only orchestration fields on a simple namespace so the canonical
    # pipeline accepts one config object without mutating TrainingConfig.
    config = SimpleNamespace(
        **vars(training_config),
        data_path=args.data_path,
        output_dir=args.output_dir,
        include_augmented=args.include_augmented,
        no_cache=args.no_cache,
        val_size=args.val_size,
        test_size=args.test_split,
        split_strategy=args.split_strategy,
        primary_metric=args.primary_metric,
        real_world_mode=args.real_world_mode,
    )

    result = run_training_pipeline(config)
    print(f"Primary metric ({result['primary_metric_name']}): {result['primary_metric']:.2f}")
    print(f"Checkpoint: {result['artifacts']['checkpoint']}")
    print(f"Metrics: {result['artifacts']['metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
