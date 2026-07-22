"""Evaluation metrics and utilities for TSL-51 models.

This module is the single source of truth for all training/evaluation metrics.
All metrics are returned as percentages (0-100) for consistency.
"""

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)

# ---------------------------------------------------------------------------
# Key constants so downstream code (visualize, reports) never hard-codes strings
# ---------------------------------------------------------------------------
METRIC_ACCURACY = "accuracy"
METRIC_PRECISION = "precision"
METRIC_RECALL = "recall"
METRIC_F1 = "f1_score"
METRIC_TOP3_ACC = "top3_accuracy"
METRIC_TOP5_ACC = "top5_accuracy"
METRIC_PER_CLASS = "per_class"
METRIC_CONFUSION_MATRIX = "confusion_matrix"


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
    y_probs: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute comprehensive evaluation metrics.

    All scalar metrics are returned as percentages (0-100) for consistency.

    Args:
        y_true: True labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
        classes: Class names (n_classes,)
        y_probs: Predicted probabilities (n_samples, n_classes). Required for top-k accuracy.

    Returns:
        Dictionary with keys: accuracy, precision, recall, f1_score,
        top3_accuracy, top5_accuracy, per_class, confusion_matrix.
    """
    metrics: dict[str, Any] = {
        METRIC_ACCURACY: float(accuracy_score(y_true, y_pred)) * 100,
        METRIC_PRECISION: float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        * 100,
        METRIC_RECALL: float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        * 100,
        METRIC_F1: float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) * 100,
    }

    # Top-k accuracy (requires probability matrix)
    metrics[METRIC_TOP3_ACC] = _top_k_accuracy(y_true, y_probs, k=3)
    metrics[METRIC_TOP5_ACC] = _top_k_accuracy(y_true, y_probs, k=5)

    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Per-class accuracy from confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    per_class_accuracy = _per_class_accuracy_from_cm(cm)

    metrics[METRIC_PER_CLASS] = {
        str(class_name): {
            "accuracy": float(acc) * 100,
            "precision": float(prec) * 100,
            "recall": float(rec) * 100,
            "f1": float(f1) * 100,
        }
        for class_name, acc, prec, rec, f1 in zip(
            classes,
            per_class_accuracy,
            per_class_precision,
            per_class_recall,
            per_class_f1,
            strict=False,
        )
    }

    # Macro / Micro averages
    metrics["macro"] = {
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)) * 100,
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)) * 100,
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) * 100,
    }
    metrics["micro"] = {
        "precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)) * 100,
        "recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)) * 100,
        "f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)) * 100,
    }

    # Confusion matrix + error analysis
    metrics[METRIC_CONFUSION_MATRIX] = cm
    metrics["most_confused"] = _find_most_confused_pairs(cm, classes)

    return metrics


def _top_k_accuracy(y_true: np.ndarray, y_probs: np.ndarray | None, k: int) -> float:
    """Return top-k accuracy as a percentage, or 0.0 if probabilities unavailable."""
    if y_probs is None or len(y_probs) == 0:
        return 0.0
    try:
        score = top_k_accuracy_score(y_true, y_probs, k=k, labels=np.arange(y_probs.shape[1]))
        return float(score) * 100
    except ValueError:
        return 0.0


def _per_class_accuracy_from_cm(cm: np.ndarray) -> np.ndarray:
    """Compute per-class accuracy from a confusion matrix.

    per-class accuracy = TP / (TP + FN) for each class
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc: np.ndarray = np.diag(cm) / np.sum(cm, axis=1)
    per_class_acc = np.nan_to_num(per_class_acc, nan=0.0, posinf=0.0, neginf=0.0)
    return per_class_acc


def _find_most_confused_pairs(
    cm: np.ndarray, classes: np.ndarray, top_n: int = 5
) -> list[dict[str, Any]]:
    """Find the most commonly confused class pairs (excluding diagonal)."""
    n_classes = len(classes)
    confused: list[tuple[int, int, int]] = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confused.append((cm[i, j], i, j))
    confused.sort(reverse=True)
    return [
        {
            "true": str(classes[i]),
            "predicted": str(classes[j]),
            "count": int(count),
        }
        for count, i, j in confused[:top_n]
    ]


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def print_metrics_report(metrics: dict[str, Any], fold_idx: int = 0):
    """Print formatted metrics report."""
    print(f"\n{'=' * 70}")
    print(f"FOLD {fold_idx} RESULTS")
    print(f"{'=' * 70}")
    print(f"Accuracy:     {metrics[METRIC_ACCURACY]:.2f}%")
    print(f"Top-3 Acc:    {metrics[METRIC_TOP3_ACC]:.2f}%")
    print(f"Top-5 Acc:    {metrics[METRIC_TOP5_ACC]:.2f}%")
    print(f"Precision:    {metrics[METRIC_PRECISION]:.2f}%")
    print(f"Recall:       {metrics[METRIC_RECALL]:.2f}%")
    print(f"F1-Score:     {metrics[METRIC_F1]:.2f}%")
    print(f"{'=' * 70}\n")


def aggregate_fold_results(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate results from multiple folds.

    Expects each result dict to contain at least val_acc and val_f1_score.
    """
    n_folds = len(fold_results)
    val_accs = [r["val_acc"] for r in fold_results]
    val_f1s = [r["val_f1_score"] for r in fold_results]
    val_precisions = [r["val_precision"] for r in fold_results]
    val_recalls = [r["val_recall"] for r in fold_results]

    return {
        "n_folds": n_folds,
        "val_acc_mean": float(np.mean(val_accs)),
        "val_acc_std": float(np.std(val_accs)),
        "val_acc_min": float(np.min(val_accs)),
        "val_acc_max": float(np.max(val_accs)),
        "val_f1_mean": float(np.mean(val_f1s)),
        "val_f1_std": float(np.std(val_f1s)),
        "val_f1_min": float(np.min(val_f1s)),
        "val_f1_max": float(np.max(val_f1s)),
        "val_precision_mean": float(np.mean(val_precisions)),
        "val_precision_std": float(np.std(val_precisions)),
        "val_recall_mean": float(np.mean(val_recalls)),
        "val_recall_std": float(np.std(val_recalls)),
        "fold_results": fold_results,
    }


def print_aggregated_report(aggregated: dict[str, Any]):
    """Print aggregated cross-validation results."""
    print(f"\n{'=' * 70}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'=' * 70}")
    print(f"Folds: {aggregated['n_folds']}")
    print("\nAccuracy:")
    print(f"  Mean: {aggregated['val_acc_mean']:.2f}% ± {aggregated['val_acc_std']:.2f}%")
    print(f"  Range: [{aggregated['val_acc_min']:.2f}%, {aggregated['val_acc_max']:.2f}%]")
    print("\nF1-Score:")
    print(f"  Mean: {aggregated['val_f1_mean']:.2f}% ± {aggregated['val_f1_std']:.2f}%")
    print(f"  Range: [{aggregated['val_f1_min']:.2f}%, {aggregated['val_f1_max']:.2f}%]")
    print(
        f"\nPrecision: {aggregated['val_precision_mean']:.2f}% ± {aggregated['val_precision_std']:.2f}%"
    )
    print(f"Recall:    {aggregated['val_recall_mean']:.2f}% ± {aggregated['val_recall_std']:.2f}%")
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _convert_numpy(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy(item) for item in obj]
    return obj


def save_results(results: dict[str, Any], output_path: str | Path) -> None:
    """Save results to JSON file."""
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(_convert_numpy(results), f, indent=2)

    print(f"Results saved to: {output_path}")


def load_results(results_path: str) -> dict[str, Any]:
    """Load results from JSON file."""
    import json

    with open(results_path) as f:
        return json.load(f)  # type: ignore[no-any-return]
