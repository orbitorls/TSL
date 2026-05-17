"""Tests for the unified metrics evaluator."""

import numpy as np
import pytest

from src.train.evaluator import (
    compute_metrics,
    aggregate_fold_results,
    METRIC_ACCURACY,
    METRIC_PRECISION,
    METRIC_RECALL,
    METRIC_F1,
    METRIC_TOP3_ACC,
    METRIC_TOP5_ACC,
    METRIC_PER_CLASS,
    METRIC_CONFUSION_MATRIX,
)


@pytest.fixture
def perfect_predictions():
    """100% correct predictions."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    probs = np.eye(3)[y_pred]  # one-hot = perfect confidence
    classes = np.array(["sign_A", "sign_B", "sign_C"])
    return y_true, y_pred, classes, probs


@pytest.fixture
def imperfect_predictions():
    """Mixed predictions with known errors."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])  # 2 errors
    # Softmax-like probabilities
    probs = np.array([
        [0.8, 0.1, 0.1],  # correct
        [0.3, 0.6, 0.1],  # wrong (predicted 1, true 0)
        [0.1, 0.8, 0.1],  # correct
        [0.1, 0.7, 0.2],  # correct
        [0.1, 0.1, 0.8],  # correct
        [0.6, 0.2, 0.2],  # wrong (predicted 0, true 2)
    ])
    classes = np.array(["sign_A", "sign_B", "sign_C"])
    return y_true, y_pred, classes, probs


class TestComputeMetrics:
    def test_perfect_accuracy(self, perfect_predictions):
        y_true, y_pred, classes, probs = perfect_predictions
        m = compute_metrics(y_true, y_pred, classes, probs)
        assert m[METRIC_ACCURACY] == pytest.approx(100.0)
        assert m[METRIC_PRECISION] == pytest.approx(100.0)
        assert m[METRIC_RECALL] == pytest.approx(100.0)
        assert m[METRIC_F1] == pytest.approx(100.0)

    def test_top_k_perfect(self, perfect_predictions):
        y_true, y_pred, classes, probs = perfect_predictions
        m = compute_metrics(y_true, y_pred, classes, probs)
        assert m[METRIC_TOP3_ACC] == pytest.approx(100.0)
        assert m[METRIC_TOP5_ACC] == pytest.approx(100.0)

    def test_top_k_without_probs(self, perfect_predictions):
        y_true, y_pred, classes, _ = perfect_predictions
        m = compute_metrics(y_true, y_pred, classes, y_probs=None)
        assert m[METRIC_TOP3_ACC] == 0.0
        assert m[METRIC_TOP5_ACC] == 0.0

    def test_imperfect_metrics(self, imperfect_predictions):
        y_true, y_pred, classes, probs = imperfect_predictions
        m = compute_metrics(y_true, y_pred, classes, probs)
        # 4/6 correct -> ~66.67%
        assert 60.0 < m[METRIC_ACCURACY] < 70.0
        assert 60.0 < m[METRIC_F1] < 70.0

    def test_per_class_keys(self, imperfect_predictions):
        y_true, y_pred, classes, probs = imperfect_predictions
        m = compute_metrics(y_true, y_pred, classes, probs)
        assert METRIC_PER_CLASS in m
        for cls in classes:
            assert cls in m[METRIC_PER_CLASS]
            assert "accuracy" in m[METRIC_PER_CLASS][cls]
            assert "precision" in m[METRIC_PER_CLASS][cls]
            assert "recall" in m[METRIC_PER_CLASS][cls]
            assert "f1" in m[METRIC_PER_CLASS][cls]

    def test_macro_micro_present(self, imperfect_predictions):
        y_true, y_pred, classes, probs = imperfect_predictions
        m = compute_metrics(y_true, y_pred, classes, probs)
        assert "macro" in m
        assert "micro" in m
        for key in ["precision", "recall", "f1"]:
            assert key in m["macro"]
            assert key in m["micro"]

    def test_confusion_matrix_shape(self, imperfect_predictions):
        y_true, y_pred, classes, probs = imperfect_predictions
        m = compute_metrics(y_true, y_pred, classes, probs)
        cm = m[METRIC_CONFUSION_MATRIX]
        assert cm.shape == (3, 3)

    def test_most_confused(self, imperfect_predictions):
        y_true, y_pred, classes, probs = imperfect_predictions
        m = compute_metrics(y_true, y_pred, classes, probs)
        confused = m["most_confused"]
        assert isinstance(confused, list)
        # At least one confusion pair exists (true 0->pred 1 and true 2->pred 0)
        assert len(confused) >= 1
        assert "true" in confused[0]
        assert "predicted" in confused[0]
        assert "count" in confused[0]

    def test_all_percentages(self, imperfect_predictions):
        y_true, y_pred, classes, probs = imperfect_predictions
        m = compute_metrics(y_true, y_pred, classes, probs)
        for key in [METRIC_ACCURACY, METRIC_PRECISION, METRIC_RECALL, METRIC_F1]:
            assert 0.0 <= m[key] <= 100.0, f"{key} should be 0-100"


class TestAggregateFoldResults:
    def test_basic_aggregation(self):
        fold_results = [
            {"val_acc": 80.0, "val_f1_score": 78.0, "val_precision": 79.0, "val_recall": 77.0},
            {"val_acc": 85.0, "val_f1_score": 83.0, "val_precision": 84.0, "val_recall": 82.0},
        ]
        agg = aggregate_fold_results(fold_results)
        assert agg["n_folds"] == 2
        assert agg["val_acc_mean"] == pytest.approx(82.5)
        assert agg["val_acc_min"] == 80.0
        assert agg["val_acc_max"] == 85.0
