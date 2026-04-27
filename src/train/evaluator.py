"""Evaluation metrics and utilities for TSL-51 models.

This module contains functions for computing and reporting evaluation metrics.
"""

import numpy as np
import torch
from typing import Dict, List, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
    }
    
    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['per_class'] = {
        class_name: {
            'precision': float(precision) * 100,
            'recall': float(recall) * 100,
            'f1': float(f1) * 100,
        }
        for class_name, precision, recall, f1 in zip(
            classes, per_class_precision, per_class_recall, per_class_f1
        )
    }
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics


def print_metrics_report(metrics: Dict[str, Any], fold_idx: int = 0):
    """Print formatted metrics report.
    
    Args:
        metrics: Metrics dictionary from compute_metrics()
        fold_idx: Current fold index
    """
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx} RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"Precision: {metrics['precision']:.2f}%")
    print(f"Recall:    {metrics['recall']:.2f}%")
    print(f"F1-Score:  {metrics['f1_score']:.2f}%")
    print(f"{'='*70}\n")


def aggregate_fold_results(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results from multiple folds.
    
    Args:
        fold_results: List of result dictionaries from each fold
        
    Returns:
        Dictionary containing aggregated statistics
    """
    n_folds = len(fold_results)
    
    # Extract metrics
    val_accs = [r['val_acc'] for r in fold_results]
    val_f1s = [r['val_f1_score'] for r in fold_results]
    val_precisions = [r['val_precision'] for r in fold_results]
    val_recalls = [r['val_recall'] for r in fold_results]
    
    aggregated = {
        'n_folds': n_folds,
        'val_acc_mean': np.mean(val_accs),
        'val_acc_std': np.std(val_accs),
        'val_acc_min': np.min(val_accs),
        'val_acc_max': np.max(val_accs),
        'val_f1_mean': np.mean(val_f1s),
        'val_f1_std': np.std(val_f1s),
        'val_f1_min': np.min(val_f1s),
        'val_f1_max': np.max(val_f1s),
        'val_precision_mean': np.mean(val_precisions),
        'val_precision_std': np.std(val_precisions),
        'val_recall_mean': np.mean(val_recalls),
        'val_recall_std': np.std(val_recalls),
        'fold_results': fold_results,
    }
    
    return aggregated


def print_aggregated_report(aggregated: Dict[str, Any]):
    """Print aggregated cross-validation results.
    
    Args:
        aggregated: Aggregated results dictionary
    """
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"Folds: {aggregated['n_folds']}")
    print(f"\nAccuracy:")
    print(f"  Mean: {aggregated['val_acc_mean']:.2f}% ± {aggregated['val_acc_std']:.2f}%")
    print(f"  Range: [{aggregated['val_acc_min']:.2f}%, {aggregated['val_acc_max']:.2f}%]")
    print(f"\nF1-Score:")
    print(f"  Mean: {aggregated['val_f1_mean']:.2f}% ± {aggregated['val_f1_std']:.2f}%")
    print(f"  Range: [{aggregated['val_f1_min']:.2f}%, {aggregated['val_f1_max']:.2f}%]")
    print(f"\nPrecision: {aggregated['val_precision_mean']:.2f}% ± {aggregated['val_precision_std']:.2f}%")
    print(f"Recall:    {aggregated['val_recall_mean']:.2f}% ± {aggregated['val_recall_std']:.2f}%")
    print(f"{'='*70}\n")


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Path to save results
    """
    import json
    from pathlib import Path
    
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj
    
    results_serializable = convert(results)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def load_results(results_path: str) -> Dict[str, Any]:
    """Load results from JSON file.
    
    Args:
        results_path: Path to results file
        
    Returns:
        Results dictionary
    """
    import json
    from pathlib import Path
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results
