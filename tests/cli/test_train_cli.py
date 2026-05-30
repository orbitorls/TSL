from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from src.train.config import TrainingConfig


def test_build_parser_help_includes_canonical_pipeline_flags() -> None:
    from src.cli.train import build_parser

    help_text = build_parser().format_help()

    for flag in (
        "--dataset",
        "--model",
        "--epochs",
        "--seed",
        "--split-strategy",
        "--primary-metric",
        "--real-world-mode",
        "--output-dir",
    ):
        assert flag in help_text


def test_main_passes_cli_only_fields_to_canonical_pipeline(monkeypatch, tmp_path: Path) -> None:
    from src.cli import train as train_cli

    captured: dict[str, object] = {}

    def fake_get_config_from_args(args):
        captured["parsed_args"] = args
        return TrainingConfig(
            model=args.model,
            hidden_dim=args.hidden,
            num_layers=args.layers,
            dropout=args.dropout,
            learning_rate=args.lr,
            batch_size=args.batch,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
            device="cpu",
            use_amp=False,
            dataset=args.dataset,
            max_samples=args.samples,
            test_split=args.test_split,
            force_download=args.force_download,
            n_folds=args.folds,
            feature_level=args.feature_level,
            seq_mode=True,
            target_frames=args.target_frames,
            augmentation_factor=args.augment,
        )

    def fake_run_training_pipeline(config):
        captured["pipeline_config"] = config
        return {
            "primary_metric_name": "macro_f1",
            "primary_metric": 77.7,
            "artifacts": {"checkpoint": "checkpoint.pt", "metrics": "metrics.json"},
        }

    monkeypatch.setattr(train_cli, "get_config_from_args", fake_get_config_from_args)
    monkeypatch.setattr(train_cli, "run_training_pipeline", fake_run_training_pipeline)

    exit_code = train_cli.main(
        [
            "--dataset",
            "tsl51_combined",
            "--model",
            "mlp",
            "--epochs",
            "3",
            "--seed",
            "123",
            "--split-strategy",
            "video_family_grouped",
            "--primary-metric",
            "macro_f1",
            "--real-world-mode",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    parsed_args = cast(Any, captured["parsed_args"])
    pipeline_config = cast(Any, captured["pipeline_config"])
    assert parsed_args.split_strategy == "video_family_grouped"
    assert parsed_args.primary_metric == "macro_f1"
    assert parsed_args.real_world_mode is True
    assert pipeline_config.split_strategy == "video_family_grouped"
    assert pipeline_config.primary_metric == "macro_f1"
    assert pipeline_config.real_world_mode is True
    assert pipeline_config.output_dir == tmp_path
    assert pipeline_config.dataset == "tsl51_combined"
    assert pipeline_config.model == "mlp"
    assert pipeline_config.epochs == 3
    assert pipeline_config.seed == 123
    assert not hasattr(TrainingConfig(), "split_strategy")
    assert not hasattr(TrainingConfig(), "primary_metric")
    assert not hasattr(TrainingConfig(), "real_world_mode")


def test_main_rejects_unsupported_primary_metric() -> None:
    from src.cli.train import main

    with pytest.raises(SystemExit):
        main(["--primary-metric", "accuracy"])


def test_pyproject_console_script_targets_cli_main() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'tsl-train = "src.cli.train:main"' in pyproject
    assert 'tsl-inference = "inference:main"' in pyproject
    assert 'tsl-predict-video = "predict_video:main"' in pyproject
    assert 'tsl-camera-translate = "camera_translate:main"' in pyproject
    assert 'tsl-translate = "translate:main"' in pyproject
    assert '"train_tsl51_v3.py" = ["E402"]' in pyproject


def test_module_help_command_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.cli.train", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--split-strategy" in result.stdout
    assert "--primary-metric" in result.stdout
    assert "--real-world-mode" in result.stdout
