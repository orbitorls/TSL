from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.core.features import BASIC_FEATURE_DIM, FEATURE_SCHEMA_VERSION
from src.core.models import MLP
from src.core.normalizer import (
    PreprocessingManifest,
    load_preprocessing_manifest,
    manifest_path_for_checkpoint,
    save_preprocessing_manifest,
)
from src.inference.translate import load_model


def _checkpoint(path: Path, *, mean: list[float] | None = None, std: list[float] | None = None) -> Path:
    model = MLP(input_dim=BASIC_FEATURE_DIM, num_classes=2, hidden_dim=8, num_layers=1, dropout=0.0)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": ["hello", "thanks"],
            "input_dim": BASIC_FEATURE_DIM,
            "num_classes": 2,
            "model": "mlp",
            "config": {
                "model": "mlp",
                "hidden_dim": 8,
                "num_layers": 1,
                "dropout": 0.0,
                "feature_level": "basic",
            },
            "mean": mean if mean is not None else [0.0] * BASIC_FEATURE_DIM,
            "std": std if std is not None else [1.0] * BASIC_FEATURE_DIM,
            "seq_mode": False,
            "target_frames": 30,
        },
        path,
    )
    return path


def _manifest(*, mean: list[float] | None = None, std: list[float] | None = None, **overrides) -> PreprocessingManifest:
    values: dict[str, Any] = {
        "normalization_type": "zscore",
        "mean": mean if mean is not None else [float(i) / 10.0 for i in range(BASIC_FEATURE_DIM)],
        "std": std if std is not None else [1.0 + float(i) / 100.0 for i in range(BASIC_FEATURE_DIM)],
        "target_frames": 30,
        "seq_mode": False,
        "dataset": "unit-test-dataset",
        "split_strategy": "video_family_holdout",
        "label_metadata": {"classes": ["hello", "thanks"]},
        "source_metadata": {"mediapipe": "tasks"},
    }
    values.update(overrides)
    return PreprocessingManifest(**values)


def test_save_preprocessing_manifest_writes_json_beside_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = _checkpoint(tmp_path / "model.pt")
    manifest = _manifest()

    saved_path = save_preprocessing_manifest(checkpoint_path, manifest)

    assert saved_path == tmp_path / "preprocessing_manifest.json"
    assert manifest_path_for_checkpoint(checkpoint_path) == saved_path
    payload = json.loads(saved_path.read_text(encoding="utf-8"))
    assert payload["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert payload["feature_level"] == "basic"
    assert payload["feature_dim"] == BASIC_FEATURE_DIM
    assert payload["feature_order"] == "basic-162-v1"
    assert payload["normalization_type"] == "zscore"
    assert payload["mean"] == manifest.mean
    assert payload["std"] == manifest.std

    loaded = load_preprocessing_manifest(saved_path)
    assert loaded.mean == manifest.mean
    assert loaded.std == manifest.std


def test_inference_load_model_prefers_manifest_normalization_stats(tmp_path: Path) -> None:
    checkpoint_path = _checkpoint(
        tmp_path / "model.pt",
        mean=[999.0] * BASIC_FEATURE_DIM,
        std=[999.0] * BASIC_FEATURE_DIM,
    )
    manifest = _manifest()
    save_preprocessing_manifest(checkpoint_path, manifest)

    _model, labels, idx_to_label, mean, std, seq_mode, target_frames, feature_level = load_model(
        checkpoint_path
    )

    assert labels == ["hello", "thanks"]
    assert idx_to_label == {0: "hello", 1: "thanks"}
    np.testing.assert_allclose(mean, np.asarray(manifest.mean, dtype=np.float32))
    np.testing.assert_allclose(std, np.asarray(manifest.std, dtype=np.float32))
    assert seq_mode is False
    assert target_frames == 30
    assert feature_level == "basic"


def test_inference_load_model_rejects_manifest_dimension_mismatch(tmp_path: Path) -> None:
    checkpoint_path = _checkpoint(tmp_path / "model.pt")
    bad_manifest = _manifest(feature_dim=BASIC_FEATURE_DIM + 1)
    save_preprocessing_manifest(checkpoint_path, bad_manifest, validate=False)

    with pytest.raises(ValueError, match="feature_dim"):
        load_model(checkpoint_path)


def test_inference_load_model_rejects_enhanced_manifest(tmp_path: Path) -> None:
    checkpoint_path = _checkpoint(tmp_path / "model.pt")
    enhanced_manifest = _manifest(
        feature_level="enhanced",
        feature_schema_version="enhanced-249-v1",
        feature_dim=249,
        feature_order="enhanced-249-v1",
        mean=[0.0] * 249,
        std=[1.0] * 249,
    )
    save_preprocessing_manifest(checkpoint_path, enhanced_manifest, validate=False)

    with pytest.raises(ValueError, match="enhanced"):
        load_model(checkpoint_path)
