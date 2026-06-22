"""Canonical training orchestration for TSL isolated-sign models."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from src.core.features import BASIC_FEATURE_DIM, FEATURE_SCHEMA_VERSION, validate_feature_level
from src.core.normalizer import Normalizer
from src.data.loader import (
    load_local_dataset,
    load_tsl51_combined,
    load_tsl51_expert,
    load_tsl51_expert_full,
    load_tsl51_full,
    load_tsl51_user_sign,
)
from src.train.augment import augment_train_split_only
from src.train.config import TrainingConfig
from src.train.evaluator import save_results
from src.train.preprocessing import save_training_preprocessing_manifest
from src.train.splits import build_grouped_split_manifest
from src.train.trainer import Trainer

PRIMARY_METRIC_NAME = "macro_f1"
GROUPED_SPLIT_STRATEGIES = frozenset({"video_family_grouped", "video_family_holdout"})
DEFAULT_SPLIT_STRATEGY = "video_family_holdout"


class PipelineConfigError(ValueError):
    """Raised when canonical training pipeline configuration is invalid."""


def _cfg(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _config_to_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config) and not isinstance(config, type):
        values = asdict(config)
    elif isinstance(config, dict):
        values = dict(config)
    elif hasattr(config, "__dict__"):
        values = dict(vars(config))
    else:
        values = {}
    return {key: _json_safe(value) for key, value in values.items() if not key.startswith("_")}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {"tensor_shape": list(value.shape), "tensor_dtype": str(value.dtype)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _resolve_split_strategy(config: Any) -> str:
    strategy = str(_cfg(config, "split_strategy", DEFAULT_SPLIT_STRATEGY)).strip().lower()
    real_world_mode = bool(_cfg(config, "real_world_mode", True))
    if strategy not in GROUPED_SPLIT_STRATEGIES:
        if real_world_mode:
            accepted = ", ".join(sorted(GROUPED_SPLIT_STRATEGIES))
            raise PipelineConfigError(
                "real-world mode requires a grouped video-family split strategy; "
                f"got {strategy!r}. Accepted strategies: {accepted}."
            )
        raise PipelineConfigError(f"Unsupported split_strategy={strategy!r}; random split is not implemented here")
    return strategy


def _validate_primary_metric(config: Any) -> None:
    requested_metric = str(_cfg(config, "primary_metric", PRIMARY_METRIC_NAME)).strip().lower()
    if requested_metric == PRIMARY_METRIC_NAME:
        return

    real_world_mode = bool(_cfg(config, "real_world_mode", True))
    if real_world_mode:
        raise PipelineConfigError(
            "real-world mode currently supports only primary_metric='macro_f1'; "
            f"got {requested_metric!r}"
        )
    raise PipelineConfigError(
        f"Unsupported primary_metric={requested_metric!r}; canonical pipeline currently supports only {PRIMARY_METRIC_NAME!r}"
    )


def _training_config_from(config: Any) -> TrainingConfig:
    fields = TrainingConfig.__dataclass_fields__
    values = {name: _cfg(config, name, field.default) for name, field in fields.items()}

    # CLI aliases preserved from the legacy command surface.
    alias_map = {
        "hidden_dim": "hidden",
        "num_layers": "layers",
        "learning_rate": "lr",
        "batch_size": "batch",
        "augmentation_factor": "augment",
        "n_folds": "folds",
    }
    for canonical, alias in alias_map.items():
        alias_value = _cfg(config, alias, None)
        if alias_value is not None:
            values[canonical] = alias_value

    return TrainingConfig(**values)


def _load_dataset(config: Any) -> SimpleNamespace:
    dataset_name = str(_cfg(config, "dataset", "tsl51_user_sign"))
    max_samples = _cfg(config, "max_samples", _cfg(config, "samples", None))
    force_download = bool(_cfg(config, "force_download", False))

    features = labels = class_names = None
    if dataset_name == "local":
        data_path = _cfg(config, "data_path", None)
        if data_path is None:
            raise PipelineConfigError("data_path is required when dataset='local'")
        features, labels, class_names = load_local_dataset(data_path, use_cache=not bool(_cfg(config, "no_cache", False)))
    elif dataset_name == "tsl51_user_sign":
        features, labels, class_names = load_tsl51_user_sign(max_samples=max_samples, force_download=force_download)
    elif dataset_name == "tsl51_expert":
        features, labels, class_names = load_tsl51_expert(
            include_augmented=bool(_cfg(config, "include_augmented", False)),
            max_samples=max_samples,
            force_download=force_download,
        )
    elif dataset_name == "tsl51_expert_full":
        features, labels, class_names = load_tsl51_expert_full(max_samples=max_samples, force_download=force_download)
    elif dataset_name == "tsl51_combined":
        features, labels, class_names = load_tsl51_combined(max_samples=max_samples, force_download=force_download)
    elif dataset_name == "tsl51_full":
        features, labels, class_names = load_tsl51_full(
            include_augmented=True,
            _max_samples=max_samples,
            force_download=force_download,
        )
    else:
        raise PipelineConfigError(f"Unknown dataset: {dataset_name}")

    if features is None or labels is None or class_names is None:
        raise PipelineConfigError(f"Failed to load dataset: {dataset_name}")

    return SimpleNamespace(X=features, y=labels, classes=class_names)


def _normalise_dataset(dataset: Any, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, Any]]]:
    X = np.asarray(dataset.X, dtype=np.float32)
    y = np.asarray(dataset.y, dtype=np.int64)
    if X.ndim not in (2, 3):
        raise PipelineConfigError("X must be shaped (n_samples, 162) or (n_samples, T, 162)")
    if X.shape[-1] != BASIC_FEATURE_DIM:
        raise PipelineConfigError(f"Canonical pipeline supports only basic {BASIC_FEATURE_DIM}-dim features")
    if len(X) != len(y):
        raise PipelineConfigError("X and y length mismatch")

    sample_ids = getattr(dataset, "sample_ids", None)
    if sample_ids is None:
        sample_ids = [f"sample-{idx}" for idx in range(len(X))]
    sample_ids = [str(sample_id) for sample_id in sample_ids]

    rows = getattr(dataset, "rows", None)
    if rows is None:
        rows = _rows_from_labels(y, classes, sample_ids)
    else:
        rows = [dict(row) for row in rows]

    return X, y, sample_ids, rows


def _rows_from_labels(y: np.ndarray, classes: np.ndarray, sample_ids: list[str]) -> list[dict[str, Any]]:
    rows = []
    for idx, (label_idx, sample_id) in enumerate(zip(y, sample_ids, strict=False)):
        label = str(classes[int(label_idx)]) if 0 <= int(label_idx) < len(classes) else str(label_idx)
        rows.append(
            {
                "sample_id": sample_id,
                "video_id": sample_id,
                "landmark_path": "",
                "sign_clean": label,
                "is_augmented": False,
            }
        )
    return rows


def _fit_normalizer(train_X: np.ndarray) -> tuple[Normalizer, np.ndarray, np.ndarray]:
    normalizer = Normalizer()
    if train_X.ndim == 3:
        flattened = train_X.reshape(-1, train_X.shape[-1])
    else:
        flattened = train_X
    normalizer.fit(flattened)
    mean = np.asarray(normalizer.mean, dtype=np.float32)
    std = np.asarray(normalizer.std, dtype=np.float32)
    return normalizer, mean, std


def _transform_split(normalizer: Normalizer, X: np.ndarray) -> np.ndarray:
    transformed = normalizer.transform(np.asarray(X, dtype=np.float32))
    return np.asarray(transformed, dtype=np.float32)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_label_map(path: Path, classes: np.ndarray) -> tuple[Path, dict[str, int]]:
    label_map = {str(label): idx for idx, label in enumerate(classes.tolist())}
    _write_json(path, label_map)
    return path, label_map


def _require_metric(mapping: Mapping[str, Any], key: str, *, context: str) -> float:
    if key not in mapping:
        raise PipelineConfigError(
            f"{context} requires {key!r} before checkpoint selection or result serialization"
        )
    value = mapping[key]
    if value is None:
        raise PipelineConfigError(
            f"{context} requires non-null {key!r} before checkpoint selection or result serialization"
        )
    try:
        metric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise PipelineConfigError(
            f"{context} requires numeric {key!r} before checkpoint selection or result serialization"
        ) from exc
    if not math.isfinite(metric_value):
        raise PipelineConfigError(
            f"{context} requires finite {key!r} before checkpoint selection or result serialization"
        )
    return metric_value


def _build_confusion_matrix_contract(
    confusion_matrix_data: Any,
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "data": _json_safe(confusion_matrix_data),
        "path": str(path) if path is not None else None,
    }


def _resolve_macro_f1(train_result: Mapping[str, Any], *, context: str) -> float:
    return _require_metric(train_result, "val_macro_f1", context=context)


def _build_result_metrics_contract(train_result: dict[str, Any]) -> dict[str, Any]:
    macro_f1 = _resolve_macro_f1(train_result, context="training result contract")
    weighted_f1 = _require_metric(train_result, "val_f1_score", context="training result contract")
    accuracy = _require_metric(train_result, "val_acc", context="training result contract")
    precision = _require_metric(train_result, "val_precision", context="training result contract")
    recall = _require_metric(train_result, "val_recall", context="training result contract")
    top3_accuracy = _require_metric(train_result, "val_top3_acc", context="training result contract")
    top5_accuracy = _require_metric(train_result, "val_top5_acc", context="training result contract")

    if "per_class_metrics" not in train_result or train_result["per_class_metrics"] is None:
        raise PipelineConfigError(
            "training result contract requires 'per_class_metrics' before checkpoint selection or result serialization"
        )
    if "confusion_matrix" not in train_result or train_result["confusion_matrix"] is None:
        raise PipelineConfigError(
            "training result contract requires 'confusion_matrix' before checkpoint selection or result serialization"
        )

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "top3_accuracy": top3_accuracy,
        "top5_accuracy": top5_accuracy,
        "per_class": _json_safe(train_result["per_class_metrics"]),
        "confusion_matrix": _build_confusion_matrix_contract(train_result["confusion_matrix"], path=None),
        "most_confused": _json_safe(train_result.get("most_confused", [])),
    }


def _build_split_metadata_contract(split_manifest: dict[str, Any], *, requested_strategy: str) -> dict[str, Any]:
    return {
        "requested_strategy": requested_strategy,
        "persisted_strategy": str(split_manifest["split_strategy"]),
        "group_key": str(split_manifest["group_key"]),
        "sample_counts": _json_safe(split_manifest.get("sample_counts", {})),
        "group_counts": _json_safe(split_manifest.get("group_counts", {})),
        "class_counts": _json_safe(split_manifest.get("class_counts", {})),
    }


def _checkpoint_payload(
    *,
    trainer: Trainer,
    training_config: TrainingConfig,
    train_result: dict[str, Any],
    classes: np.ndarray,
    label_map: dict[str, int],
    mean: np.ndarray,
    std: np.ndarray,
    input_dim: int,
) -> dict[str, Any]:
    model_state = train_result.get("model_state") or {}
    state_dict = model_state.get("model_state_dict")
    if state_dict is None and getattr(trainer, "model", None) is not None:
        state_dict = trainer.model.state_dict()

    macro_f1 = _resolve_macro_f1(train_result, context="checkpoint payload")

    return {
        "model_state_dict": state_dict,
        "state_dict": state_dict,
        "optimizer_state_dict": model_state.get("optimizer_state_dict"),
        "epoch": model_state.get("epoch"),
        "classes": classes.tolist(),
        "labels": classes.tolist(),
        "label_to_idx": label_map,
        "idx_to_label": {idx: label for label, idx in label_map.items()},
        "input_dim": int(input_dim),
        "num_classes": int(len(classes)),
        "model": training_config.model,
        "config": _config_to_dict(training_config),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
        "seq_mode": bool(training_config.seq_mode),
        "target_frames": int(training_config.target_frames),
        "primary_metric_name": PRIMARY_METRIC_NAME,
        "primary_metric": macro_f1,
        "metrics": _json_safe(train_result),
    }


def run_training_pipeline(config: Any, *, dataset: Any | None = None) -> dict[str, Any]:
    """Run the canonical src/train training pipeline and persist artifacts.

    The pipeline is intentionally one-fold/holdout orchestration: grouped split
    manifest first, optional train-only augmentation, train-only normalization,
    Trainer execution, and canonical artifact writing.
    """
    training_config = _training_config_from(config)
    validate_feature_level(training_config.feature_level)
    if training_config.feature_level != "basic":
        raise PipelineConfigError("Canonical pipeline currently supports feature_level='basic' only")

    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(training_config.seed)

    _validate_primary_metric(config)
    requested_split_strategy = _resolve_split_strategy(config)

    dataset_obj = dataset if dataset is not None else _load_dataset(config)
    classes = np.asarray(dataset_obj.classes)
    X, y, sample_ids, rows = _normalise_dataset(dataset_obj, classes)

    dataset_name = str(_cfg(config, "dataset", training_config.dataset))
    val_size = float(_cfg(config, "val_size", 0.15))
    test_size = float(_cfg(config, "test_size", _cfg(config, "test_split", 0.15)))
    split_manifest = build_grouped_split_manifest(
        rows,
        dataset_name=dataset_name,
        seed=training_config.seed,
        val_size=val_size,
        test_size=test_size,
    )

    split_data = augment_train_split_only(
        X,
        y,
        sample_ids,
        split_manifest,
        classes,
        augmentation_factor=training_config.augmentation_factor,
        noise_level=training_config.noise_level,
        scale_range=training_config.scale_range,
    )

    normalizer, mean, std = _fit_normalizer(split_data["train"]["X"])
    X_train = _transform_split(normalizer, split_data["train"]["X"])
    X_val = _transform_split(normalizer, split_data["val"]["X"])

    trainer = Trainer(training_config)
    train_result = trainer.train(
        X_train,
        np.asarray(split_data["train"]["y"], dtype=np.int64),
        X_val,
        np.asarray(split_data["val"]["y"], dtype=np.int64),
        classes,
        fold_idx=0,
    )

    macro_f1 = _resolve_macro_f1(train_result, context="pipeline training result")
    train_result["val_macro_f1"] = macro_f1
    train_result["primary_metric_name"] = PRIMARY_METRIC_NAME
    train_result["primary_metric"] = macro_f1

    timestamp = str(_cfg(config, "timestamp", datetime.now().strftime("%Y%m%d_%H%M%S")))
    output_dir = Path(_cfg(config, "output_dir", Path("artifacts") / "runs" / f"train_{timestamp}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    label_map_path, label_map = _write_label_map(output_dir / "label_map.json", classes)
    split_manifest_path = _write_json(output_dir / "split_manifest.json", split_manifest)
    checkpoint_path = output_dir / f"tsl51_{training_config.model}_{timestamp}.pt"
    checkpoint = _checkpoint_payload(
        trainer=trainer,
        training_config=training_config,
        train_result=train_result,
        classes=classes,
        label_map=label_map,
        mean=mean,
        std=std,
        input_dim=int(X.shape[-1]),
    )
    torch.save(checkpoint, checkpoint_path)

    preprocessing_manifest_path = save_training_preprocessing_manifest(
        checkpoint_path,
        mean=mean,
        std=std,
        dataset=dataset_name,
        split_strategy=str(split_manifest["split_strategy"]),
        target_frames=training_config.target_frames,
        seq_mode=training_config.seq_mode,
        label_map_path=label_map_path,
        label_metadata={"classes": classes.tolist(), "label_to_idx": label_map},
        source_metadata={
            "split_manifest_path": str(split_manifest_path),
            "train_sample_count": int(len(split_data["train"]["y"])),
            "val_sample_count": int(len(split_data["val"]["y"])),
            "test_sample_count": int(len(split_data["test"]["y"])),
        },
    )

    artifacts = {
        "checkpoint": str(checkpoint_path),
        "metrics": str(output_dir / "metrics.json"),
        "split_manifest": str(split_manifest_path),
        "preprocessing_manifest": str(preprocessing_manifest_path),
        "label_map": str(label_map_path),
    }
    metrics_payload = {
        "dataset": dataset_name,
        "model": training_config.model,
        "seed": int(training_config.seed),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_level": training_config.feature_level,
        "primary_metric_name": PRIMARY_METRIC_NAME,
        "primary_metric": macro_f1,
        "checkpoint_path": str(checkpoint_path),
        "preprocessing_manifest_path": str(preprocessing_manifest_path),
        "split_manifest_path": str(split_manifest_path),
        "label_map_path": str(label_map_path),
        "metrics": _build_result_metrics_contract(train_result),
        "split_metadata": _build_split_metadata_contract(split_manifest, requested_strategy=requested_split_strategy),
        "train_result": train_result,
        "split_manifest": split_manifest,
        "artifacts": artifacts,
        "config": _config_to_dict(training_config),
        "requested_split_strategy": requested_split_strategy,
    }
    save_results(_json_safe(metrics_payload), artifacts["metrics"])

    return {
        "dataset": dataset_name,
        "primary_metric_name": PRIMARY_METRIC_NAME,
        "primary_metric": macro_f1,
        "train_result": train_result,
        "split_manifest": split_manifest,
        "artifacts": artifacts,
        "config": _config_to_dict(training_config),
        "requested_split_strategy": requested_split_strategy,
    }


__all__ = [
    "DEFAULT_SPLIT_STRATEGY",
    "GROUPED_SPLIT_STRATEGIES",
    "PipelineConfigError",
    "PRIMARY_METRIC_NAME",
    "run_training_pipeline",
]
