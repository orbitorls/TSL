"""Training configuration for TSL-51 models.

This module provides configuration classes for training TSL-51 models,
making it easier to manage hyperparameters and settings.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from src.core.features import FEATURE_SCHEMA_VERSION
from src.train.compat import HAS_AMP


@dataclass
class TrainingConfig:
    """Training configuration parameters.

    Attributes:
        model: Model architecture ('mlp', 'gru', 'mopgru', 'hybrid', 'ctc')
        hidden_dim: Hidden dimension size
        num_layers: Number of layers
        dropout: Dropout rate
        learning_rate: Learning rate
        batch_size: Batch size
        epochs: Maximum epochs per fold
        patience: Early stopping patience
        seed: Random seed
        device: Device to use ('cuda' or 'cpu')
        use_amp: Use automatic mixed precision
        gradient_clip_value: Gradient clipping value (None to disable)
        use_gradient_accumulation: Use gradient accumulation
        accumulation_steps: Number of steps to accumulate gradients
    """

    model: str = "gru"
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 30
    patience: int = 10
    seed: int = 42
    device: str = "cuda"
    use_amp: bool = True
    gradient_clip_value: float | None = None
    use_gradient_accumulation: bool = False
    accumulation_steps: int = 1

    # Data augmentation
    augmentation_factor: int = 0
    noise_level: float = 0.01
    scale_range: tuple[float, float] = (0.95, 1.05)

    # Dataset
    dataset: str = "tsl51_user_sign"
    max_samples: int | None = None
    test_split: float = 0.0
    force_download: bool = False

    # Cross-validation
    n_folds: int = 5
    stratified: bool = True

    # Feature extraction
    feature_level: str = "basic"
    seq_mode: bool = True
    target_frames: int = 30


TASK9_BASELINE_MACRO_F1 = 34.39885173927727
TRUSTED_TUNING_MIN_DELTA = 2.0
TRUSTED_TUNING_PRIMARY_METRIC = "macro_f1"
TRUSTED_TUNING_SPLIT_STRATEGY = "video_family_holdout"
TRUSTED_TUNING_FEATURE_LEVEL = "basic"
TRUSTED_TUNING_TARGET_FRAMES = 30


@dataclass(frozen=True)
class TuningCandidate:
    """Manual tuning candidate used by the controlled Task 10 workflow."""

    name: str
    description: str
    config: TrainingConfig
    complexity_rank: int


@dataclass(frozen=True)
class TuningSelection:
    """Result of applying the trusted Macro F1 tuning gate."""

    selected_name: str
    selected_macro_f1: float
    accepted_tuned_candidate: bool
    baseline_macro_f1: float
    required_macro_f1: float
    rationale: str


def _trusted_preprocessing_schema(
    *, seq_mode: bool = True, target_frames: int = TRUSTED_TUNING_TARGET_FRAMES
) -> str:
    return (
        f"{FEATURE_SCHEMA_VERSION}|{TRUSTED_TUNING_FEATURE_LEVEL}|"
        f"target_frames={int(target_frames)}|seq_mode={bool(seq_mode)}"
    )


def _trusted_base_config(**overrides: object) -> TrainingConfig:
    base = TrainingConfig(
        dataset="tsl51_user_sign",
        seed=42,
        device="cuda",
        use_amp=True,
        feature_level=TRUSTED_TUNING_FEATURE_LEVEL,
        seq_mode=True,
        target_frames=TRUSTED_TUNING_TARGET_FRAMES,
        test_split=0.15,
        augmentation_factor=0,
        epochs=1,
        patience=1,
        batch_size=64,
    )
    return cast(TrainingConfig, replace(base, **overrides))


def get_controlled_tuning_candidates() -> tuple[TuningCandidate, ...]:
    """Return the fixed, manual candidate set for controlled GRU/MLP comparison."""
    return (
        TuningCandidate(
            name="mlp_basic_baseline",
            description="Simple MLP baseline on the trusted basic-162 schema.",
            config=_trusted_base_config(model="mlp", hidden_dim=256, num_layers=3, dropout=0.3),
            complexity_rank=0,
        ),
        TuningCandidate(
            name="gru_sequence",
            description="Baseline GRU sequence model using the trusted grouped split.",
            config=_trusted_base_config(model="gru", hidden_dim=256, num_layers=3, dropout=0.3),
            complexity_rank=1,
        ),
        TuningCandidate(
            name="gru_regularized",
            description="GRU variant with stronger regularization and gradient clipping.",
            config=_trusted_base_config(
                model="gru",
                hidden_dim=256,
                num_layers=3,
                dropout=0.45,
                learning_rate=7.5e-4,
                gradient_clip_value=1.0,
            ),
            complexity_rank=2,
        ),
    )


def _metadata_path(metrics: Mapping[str, Any], key: str) -> Path:
    value = metrics.get(key)
    if not value:
        artifacts = metrics.get("artifacts")
        if isinstance(artifacts, Mapping):
            artifact_key = key.removesuffix("_path")
            value = artifacts.get(artifact_key)
    if not value:
        raise ValueError(f"baseline metrics must include {key!r}")
    return Path(str(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object at {path}")
    return data


def _coerce_float(value: object, *, context: str) -> float:
    if isinstance(value, str | int | float):
        return float(value)
    raise ValueError(f"{context} must be numeric")


def _baseline_macro_f1(metrics: Mapping[str, Any]) -> float:
    if (
        metrics.get("primary_metric_name") == TRUSTED_TUNING_PRIMARY_METRIC
        and "primary_metric" in metrics
    ):
        return _coerce_float(metrics["primary_metric"], context="baseline primary_metric")
    nested_metrics = metrics.get("metrics")
    if isinstance(nested_metrics, Mapping) and TRUSTED_TUNING_PRIMARY_METRIC in nested_metrics:
        return _coerce_float(
            nested_metrics[TRUSTED_TUNING_PRIMARY_METRIC],
            context="baseline metrics.macro_f1",
        )
    return TASK9_BASELINE_MACRO_F1


def build_trusted_tuning_run_configs(
    baseline_metrics: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Build deterministic run metadata for the three controlled tuning candidates.

    The returned dictionaries are intentionally lightweight orchestration inputs:
    they carry the candidate config plus the split/preprocessing identity that all
    candidate runs must preserve when comparing Macro F1.
    """
    split_manifest_path = _metadata_path(baseline_metrics, "split_manifest_path")
    preprocessing_manifest_path = _metadata_path(baseline_metrics, "preprocessing_manifest_path")
    split_manifest = _load_json(split_manifest_path)
    preprocessing_manifest = _load_json(preprocessing_manifest_path)

    split_strategy = str(split_manifest.get("split_strategy", ""))
    if split_strategy != TRUSTED_TUNING_SPLIT_STRATEGY:
        raise ValueError(
            f"trusted tuning requires split_strategy={TRUSTED_TUNING_SPLIT_STRATEGY!r}; got {split_strategy!r}"
        )

    feature_schema_version = str(baseline_metrics.get("feature_schema_version") or "")
    if feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"trusted tuning requires feature_schema_version={FEATURE_SCHEMA_VERSION!r}; got {feature_schema_version!r}"
        )

    preprocessing_schema = _trusted_preprocessing_schema(
        seq_mode=bool(preprocessing_manifest.get("seq_mode", True)),
        target_frames=int(
            preprocessing_manifest.get("target_frames", TRUSTED_TUNING_TARGET_FRAMES)
        ),
    )
    split_manifest_hash = _sha256_file(split_manifest_path)

    run_configs = []
    for candidate in get_controlled_tuning_candidates():
        run_configs.append(
            {
                "candidate": candidate,
                "config": candidate.config,
                "primary_metric_name": TRUSTED_TUNING_PRIMARY_METRIC,
                "split_strategy": split_strategy,
                "split_manifest_path": str(split_manifest_path),
                "split_manifest_hash": split_manifest_hash,
                "preprocessing_manifest_path": str(preprocessing_manifest_path),
                "preprocessing_schema": preprocessing_schema,
                "feature_schema_version": feature_schema_version,
                "feature_level": TRUSTED_TUNING_FEATURE_LEVEL,
                "baseline_macro_f1": _baseline_macro_f1(baseline_metrics),
                "min_required_macro_f1": _baseline_macro_f1(baseline_metrics)
                + TRUSTED_TUNING_MIN_DELTA,
            }
        )
    return tuple(run_configs)


def _result_macro_f1(result: Mapping[str, Any]) -> float:
    if (
        result.get("primary_metric_name") == TRUSTED_TUNING_PRIMARY_METRIC
        and "primary_metric" in result
    ):
        return _coerce_float(result["primary_metric"], context="candidate primary_metric")
    nested_metrics = result.get("metrics")
    if isinstance(nested_metrics, Mapping) and TRUSTED_TUNING_PRIMARY_METRIC in nested_metrics:
        return _coerce_float(
            nested_metrics[TRUSTED_TUNING_PRIMARY_METRIC],
            context="candidate metrics.macro_f1",
        )
    raise ValueError("candidate result must include Macro F1 as primary_metric or metrics.macro_f1")


def _candidate_rank_by_name() -> dict[str, int]:
    return {
        candidate.name: candidate.complexity_rank
        for candidate in get_controlled_tuning_candidates()
    }


def _ensure_single_value(results: Sequence[Mapping[str, Any]], key: str, message: str) -> None:
    values = {str(result.get(key, "")) for result in results}
    if len(values) != 1 or "" in values:
        raise ValueError(message)


def _trusted_identity_by_name(
    run_configs: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    trusted: dict[str, Mapping[str, Any]] = {}
    for run_config in run_configs:
        candidate = run_config.get("candidate")
        if not isinstance(candidate, TuningCandidate):
            raise ValueError("run config must include a TuningCandidate under 'candidate'")
        trusted[candidate.name] = run_config
    return trusted


def _validate_candidate_result_contract(
    results: Sequence[Mapping[str, Any]],
    *,
    run_configs: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    if not results:
        raise ValueError("at least one candidate result is required")

    trusted_by_name = _trusted_identity_by_name(run_configs) if run_configs is not None else None
    for result in results:
        if result.get("primary_metric_name") != TRUSTED_TUNING_PRIMARY_METRIC:
            raise ValueError("candidate results must use macro_f1 as the primary metric")
        if result.get("feature_level") != TRUSTED_TUNING_FEATURE_LEVEL:
            raise ValueError("candidate results must use feature_level='basic'")
        if trusted_by_name is None:
            continue
        candidate_name = _candidate_name(result)
        if candidate_name not in trusted_by_name:
            raise ValueError(f"candidate result {candidate_name!r} has no trusted run config")
        trusted = trusted_by_name[candidate_name]
        for key in (
            "split_manifest_hash",
            "preprocessing_schema",
            "feature_schema_version",
            "feature_level",
        ):
            result_value = str(result.get(key, ""))
            trusted_value = str(trusted.get(key, ""))
            if result_value != trusted_value:
                raise ValueError(
                    f"candidate result {candidate_name!r} does not match trusted run config {key}: "
                    f"result={result_value!r}, trusted={trusted_value!r}"
                )

    _ensure_single_value(
        results, "split_manifest_hash", "candidate results must use the same split manifest hash"
    )
    _ensure_single_value(
        results, "preprocessing_schema", "candidate results must use the same preprocessing schema"
    )
    _ensure_single_value(
        results,
        "feature_schema_version",
        "candidate results must use the same feature schema version",
    )


def select_trusted_tuning_result(
    candidate_results: Sequence[Mapping[str, Any]],
    baseline_metrics: Mapping[str, Any],
    *,
    run_configs: Sequence[Mapping[str, Any]],
    min_delta: float = TRUSTED_TUNING_MIN_DELTA,
) -> TuningSelection:
    """Select a tuned candidate only when it clears the trusted Macro F1 gate."""
    results = tuple(candidate_results)
    _validate_candidate_result_contract(results, run_configs=run_configs)

    baseline_macro_f1 = _baseline_macro_f1(baseline_metrics)
    required_macro_f1 = baseline_macro_f1 + float(min_delta)
    ranks = _candidate_rank_by_name()
    ranked_results = sorted(
        results,
        key=lambda item: (
            -_result_macro_f1(item),
            ranks.get(str(item.get("candidate_name", "")), 999),
        ),
    )
    best = ranked_results[0]
    best_name = str(best.get("candidate_name", ""))
    best_macro_f1 = _result_macro_f1(best)

    if best_macro_f1 >= required_macro_f1:
        return TuningSelection(
            selected_name=best_name,
            selected_macro_f1=best_macro_f1,
            accepted_tuned_candidate=True,
            baseline_macro_f1=baseline_macro_f1,
            required_macro_f1=required_macro_f1,
            rationale=(
                f"{best_name} reached Macro F1 {best_macro_f1:.4f}, "
                f">= {float(min_delta):.2f} percentage points over Task 9 baseline."
            ),
        )

    return TuningSelection(
        selected_name="task9_baseline",
        selected_macro_f1=baseline_macro_f1,
        accepted_tuned_candidate=False,
        baseline_macro_f1=baseline_macro_f1,
        required_macro_f1=required_macro_f1,
        rationale=(
            f"Best candidate {best_name} reached Macro F1 {best_macro_f1:.4f} but did not clear "
            f"the required {float(min_delta):.2f} point improvement over Task 9 baseline; "
            "keeping the trusted baseline to avoid regression."
        ),
    )


TRUSTED_TUNING_SECONDARY_METRICS = (
    "weighted_f1",
    "accuracy",
    "precision",
    "recall",
    "top3_accuracy",
    "top5_accuracy",
)
TRUSTED_TUNING_ARTIFACT_KEYS = (
    "checkpoint",
    "metrics",
    "split_manifest",
    "preprocessing_manifest",
    "label_map",
)


def _config_payload(config: TrainingConfig) -> dict[str, object]:
    return {
        "model": config.model,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "patience": config.patience,
        "seed": config.seed,
        "feature_level": config.feature_level,
        "seq_mode": config.seq_mode,
        "target_frames": config.target_frames,
        "augmentation_factor": config.augmentation_factor,
        "gradient_clip_value": config.gradient_clip_value,
    }


def _candidate_command(
    config: TrainingConfig, *, split_strategy: str = TRUSTED_TUNING_SPLIT_STRATEGY
) -> str:
    parts = [
        "python",
        "-m",
        "src.cli.train",
        "--dataset",
        config.dataset,
        "--model",
        config.model,
        "--split-strategy",
        split_strategy,
        "--primary-metric",
        TRUSTED_TUNING_PRIMARY_METRIC,
        "--feature-level",
        config.feature_level,
        "--epochs",
        str(config.epochs),
        "--seed",
        str(config.seed),
        "--hidden",
        str(config.hidden_dim),
        "--layers",
        str(config.num_layers),
        "--dropout",
        str(config.dropout),
        "--lr",
        str(config.learning_rate),
        "--batch",
        str(config.batch_size),
    ]
    return " ".join(parts)


def _candidate_name(value: Mapping[str, Any]) -> str:
    name = value.get("candidate_name")
    if not name:
        candidate = value.get("candidate")
        if isinstance(candidate, TuningCandidate):
            name = candidate.name
    if not name:
        raise ValueError("candidate result/config must include candidate_name")
    return str(name)


def _run_config_by_name(run_configs: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    by_name: dict[str, Mapping[str, Any]] = {}
    for run_config in run_configs:
        candidate = run_config.get("candidate")
        if not isinstance(candidate, TuningCandidate):
            raise ValueError("run config must include a TuningCandidate under 'candidate'")
        by_name[candidate.name] = run_config
    return by_name


def _result_by_name(candidate_results: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {_candidate_name(result): result for result in candidate_results}


def _secondary_metrics(result: Mapping[str, Any]) -> dict[str, float]:
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("candidate result must include a metrics mapping with secondary metrics")
    secondary: dict[str, float] = {}
    for key in TRUSTED_TUNING_SECONDARY_METRICS:
        if key not in metrics:
            raise ValueError(f"candidate result metrics must include {key!r}")
        secondary[key] = _coerce_float(metrics[key], context=f"candidate metrics.{key}")
    return secondary


def _artifact_paths(result: Mapping[str, Any]) -> dict[str, str]:
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("candidate result must include artifact paths")
    paths: dict[str, str] = {}
    for key in TRUSTED_TUNING_ARTIFACT_KEYS:
        value = artifacts.get(key)
        if not value:
            raise ValueError(f"candidate result artifacts must include non-empty {key!r}")
        paths[key] = str(value)
    return paths


def _selection_payload(selection: TuningSelection) -> dict[str, object]:
    return {
        "selected_name": selection.selected_name,
        "selected_macro_f1": selection.selected_macro_f1,
        "accepted_tuned_candidate": selection.accepted_tuned_candidate,
        "baseline_macro_f1": selection.baseline_macro_f1,
        "required_macro_f1": selection.required_macro_f1,
        "rationale": selection.rationale,
    }


def build_trusted_tuning_report(
    run_configs: Sequence[Mapping[str, Any]],
    candidate_results: Sequence[Mapping[str, Any]],
    baseline_metrics: Mapping[str, Any],
) -> dict[str, object]:
    """Build the Task 10 fixed-candidate tuning report contract.

    This function does not run training. It validates and packages candidate run
    outputs produced under the trusted grouped split so reviewers can verify the
    Macro F1 gate and artifact identity without introducing heavy HPO state.
    """
    results = tuple(candidate_results)
    _validate_candidate_result_contract(results, run_configs=run_configs)

    configs_by_name = _run_config_by_name(run_configs)
    results_by_name = _result_by_name(results)
    candidate_runs: list[dict[str, object]] = []
    for candidate in get_controlled_tuning_candidates():
        if candidate.name not in configs_by_name:
            raise ValueError(f"missing run config for candidate {candidate.name!r}")
        if candidate.name not in results_by_name:
            raise ValueError(f"missing result for candidate {candidate.name!r}")
        run_config = configs_by_name[candidate.name]
        result = results_by_name[candidate.name]
        config = candidate.config
        artifacts = _artifact_paths(result)
        candidate_runs.append(
            {
                "candidate_name": candidate.name,
                "description": candidate.description,
                "complexity_rank": candidate.complexity_rank,
                "command": _candidate_command(
                    config,
                    split_strategy=str(
                        run_config.get("split_strategy", TRUSTED_TUNING_SPLIT_STRATEGY)
                    ),
                ),
                "config": _config_payload(config),
                "seed": config.seed,
                "primary_metric_name": TRUSTED_TUNING_PRIMARY_METRIC,
                "grouped_macro_f1": _result_macro_f1(result),
                "secondary_metrics": _secondary_metrics(result),
                "artifacts": artifacts,
                "split_manifest_hash": str(result.get("split_manifest_hash", "")),
                "split_manifest_path": artifacts["split_manifest"],
                "preprocessing_schema": str(result.get("preprocessing_schema", "")),
                "preprocessing_manifest_path": artifacts["preprocessing_manifest"],
                "feature_schema_version": str(result.get("feature_schema_version", "")),
                "feature_level": str(result.get("feature_level", "")),
            }
        )

    selection = select_trusted_tuning_result(results, baseline_metrics, run_configs=run_configs)
    baseline_macro_f1 = _baseline_macro_f1(baseline_metrics)
    return {
        "report_version": "trusted-tuning-report-v1",
        "primary_metric_name": TRUSTED_TUNING_PRIMARY_METRIC,
        "baseline": {
            "name": "task9_baseline",
            "macro_f1": baseline_macro_f1,
            "metrics_path": str(baseline_metrics.get("metrics_path", "")),
            "split_manifest_path": str(baseline_metrics.get("split_manifest_path", "")),
            "preprocessing_manifest_path": str(
                baseline_metrics.get("preprocessing_manifest_path", "")
            ),
            "feature_schema_version": str(baseline_metrics.get("feature_schema_version", "")),
            "feature_level": str(baseline_metrics.get("feature_level", "")),
        },
        "min_delta_macro_f1": TRUSTED_TUNING_MIN_DELTA,
        "selection": _selection_payload(selection),
        "candidate_runs": candidate_runs,
        "shared_identity": {
            "split_manifest_hash": candidate_runs[0]["split_manifest_hash"],
            "preprocessing_schema": candidate_runs[0]["preprocessing_schema"],
            "feature_schema_version": candidate_runs[0]["feature_schema_version"],
            "feature_level": candidate_runs[0]["feature_level"],
        },
    }


@dataclass
class PresetConfig:
    """Predefined training configurations for common use cases."""

    @staticmethod
    def quick() -> TrainingConfig:
        """Quick training for testing (5 epochs, smaller model)."""
        return TrainingConfig(
            model="gru",
            hidden_dim=128,
            num_layers=2,
            epochs=5,
            batch_size=64,
            n_folds=2,
            patience=3,
            augmentation_factor=0,
        )

    @staticmethod
    def default() -> TrainingConfig:
        """Default training configuration with seq_mode and augmentation."""
        return TrainingConfig(
            model="gru",
            hidden_dim=256,
            num_layers=3,
            epochs=50,
            batch_size=128,
            n_folds=5,
            patience=10,
            feature_level="basic",
            seq_mode=True,
            target_frames=30,
            augmentation_factor=5,
        )

    @staticmethod
    def full_cv() -> TrainingConfig:
        """Full K-Fold CV with more epochs."""
        return TrainingConfig(
            model="gru",
            hidden_dim=256,
            num_layers=3,
            epochs=100,
            batch_size=128,
            n_folds=5,
            patience=15,
            augmentation_factor=10,
        )

    @staticmethod
    def mlp_fast() -> TrainingConfig:
        """Fast MLP training."""
        return TrainingConfig(
            model="mlp",
            hidden_dim=256,
            num_layers=3,
            epochs=50,
            batch_size=128,
            n_folds=5,
            patience=10,
            augmentation_factor=5,
        )

    @staticmethod
    def large_dataset() -> TrainingConfig:
        """Configuration for large datasets (45k+ samples)."""
        return TrainingConfig(
            model="gru",
            hidden_dim=512,
            num_layers=4,
            epochs=100,
            batch_size=256,
            n_folds=5,
            patience=20,
            augmentation_factor=0,  # No augmentation needed for large dataset
            use_gradient_accumulation=True,
            accumulation_steps=2,
        )


def get_config_from_args(args) -> TrainingConfig:
    """Create TrainingConfig from argparse arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        TrainingConfig instance
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        device=device,
        use_amp=HAS_AMP,
        dataset=args.dataset,
        max_samples=args.samples,
        test_split=args.test_split,
        force_download=args.force_download,
        n_folds=args.folds,
        feature_level=args.feature_level,
        target_frames=args.target_frames,
        augmentation_factor=args.augment,
    )
