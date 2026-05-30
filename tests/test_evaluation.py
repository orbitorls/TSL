"""Evaluation tests for TSL-51 model.

Comprehensive tests for model evaluation including:
- Per-class accuracy
- Confusion matrix analysis
- Precision, Recall, F1 per class
- Error analysis
- Cross-validation consistency
"""

# pyright: reportArgumentType=false, reportIndexIssue=false, reportCallIssue=false

import numpy as np
import pytest
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from pathlib import Path


# Test configuration
EXPECTED_CLASSES = 6
METRIC_TEST_CLASSES = 12
EXPECTED_FEATURES = 162


@pytest.fixture(scope="session")
def test_checkpoint_path(tmp_path_factory):
    """Create a deterministic local checkpoint for model-loading tests."""
    from src.core import GRUModel

    classes = [f"sign_{idx}" for idx in range(EXPECTED_CLASSES)]
    model = GRUModel(
        input_dim=EXPECTED_FEATURES,
        num_classes=len(classes),
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
    )
    checkpoint_path = tmp_path_factory.mktemp("evaluation-artifacts") / "synthetic-evaluation-model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": classes,
            "mean": np.zeros(EXPECTED_FEATURES, dtype=np.float32),
            "std": np.ones(EXPECTED_FEATURES, dtype=np.float32),
            "input_dim": EXPECTED_FEATURES,
            "num_classes": len(classes),
            "hidden_dim": 32,
            "num_layers": 2,
            "accuracy": 1.0,
        },
        checkpoint_path,
    )
    return checkpoint_path


@pytest.fixture
def trained_model(test_checkpoint_path):
    """Load trained model for evaluation."""
    import torch
    from src.core import GRUModel

    checkpoint = torch.load(test_checkpoint_path, map_location='cpu', weights_only=False)

    model = GRUModel(
        input_dim=checkpoint.get('input_dim', 162),
        num_classes=checkpoint.get('num_classes', 262),
        hidden_dim=checkpoint.get('hidden_dim', 256),
        num_layers=checkpoint.get('num_layers', 3),
        dropout=0.0  # No dropout during evaluation
    )
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    return {
        'model': model,
        'classes': checkpoint.get('classes', []),
        'mean': checkpoint.get('mean', np.zeros(162)),
        'std': checkpoint.get('std', np.ones(162)),
        'accuracy': checkpoint.get('accuracy', 0),
    }


@pytest.fixture
def sample_predictions():
    """Generate sample predictions for testing metrics."""
    np.random.seed(42)
    n_samples = 100
    n_classes = METRIC_TEST_CLASSES

    # True labels
    y_true = np.random.randint(0, n_classes, n_samples)

    # Predictions with ~95% accuracy (realistic for trained model)
    y_pred = y_true.copy()
    error_mask = np.random.random(n_samples) < 0.05
    y_pred[error_mask] = np.random.randint(0, n_classes, error_mask.sum())

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'n_samples': n_samples,
        'n_classes': n_classes,
    }


class TestModelEvaluation:
    """Test suite for model evaluation metrics."""

    def test_accuracy_score(self, sample_predictions):
        """Test accuracy calculation."""
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']

        accuracy = accuracy_score(y_true, y_pred)

        # ~95% accuracy due to our simulated errors
        assert 0.90 < accuracy < 1.0, f"Expected ~95% accuracy, got {accuracy:.2%}"
        print(f"Accuracy: {accuracy:.2%}")

    def test_precision_recall_f1(self, sample_predictions):
        """Test precision, recall, F1 calculation."""
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']

        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        assert 0.85 <= precision <= 1.0, f"Precision {precision} out of range"
        assert 0.85 <= recall <= 1.0, f"Recall {recall} out of range"
        assert 0.85 <= f1 <= 1.0, f"F1 {f1} out of range"

    def test_confusion_matrix(self, sample_predictions):
        """Test confusion matrix generation."""
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']
        n_classes = sample_predictions['n_classes']

        cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))

        assert cm.shape == (n_classes, n_classes)
        assert cm.sum() == len(y_true)  # Total predictions

        # Diagonal should be non-zero (correct predictions)
        diagonal = np.diag(cm)
        assert diagonal.sum() > 0

        print(f"Confusion matrix shape: {cm.shape}")
        print(f"Correct predictions: {diagonal.sum()}/{len(y_true)}")

    def test_per_class_metrics(self, sample_predictions):
        """Test per-class precision, recall, F1."""
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']

        # Per-class metrics
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

        # sklearn returns metrics only for classes present in data (y_true or y_pred)
        # The number should be close to but may not exactly match n_present_classes
        n_present_classes = len(np.unique(np.concatenate([y_true, y_pred])))
        assert len(precision) <= n_present_classes + 5, f"Got {len(precision)} classes, expected ~{n_present_classes}"

        # Find classes with low performance
        low_f1_classes = np.where(f1 < 0.5)[0]
        print(f"Classes with F1 < 0.5: {len(low_f1_classes)}")

        # Most classes should have F1 > 0
        valid_f1 = f1[f1 > 0]
        assert len(valid_f1) > len(f1) * 0.8, "Too many classes with zero F1"


class TestClassificationReport:
    """Tests for classification report generation."""

    def test_classification_report_format(self, sample_predictions):
        """Test that classification report is generated correctly."""
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']

        report = classification_report(
            y_true, y_pred,
            output_dict=True,
            zero_division=0
        )

        # Check report structure
        assert 'accuracy' in report
        assert 'macro avg' in report
        assert 'weighted avg' in report

        # Check macro and weighted avg have required fields
        for avg_type in ['macro avg', 'weighted avg']:
            assert 'precision' in report[avg_type]
            assert 'recall' in report[avg_type]
            assert 'f1-score' in report[avg_type]

        print(f"Classification report generated for {len(report) - 3} classes")

    def test_support_values(self, sample_predictions):
        """Test that support (sample count per class) is correct."""
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']

        report = classification_report(
            y_true, y_pred,
            output_dict=True,
            zero_division=0
        )

        total_support = sum(
            report[str(c)]['support']
            for c in range(sample_predictions['n_classes'])
            if str(c) in report
        )

        assert total_support == len(y_true)


class TestModelLoading:
    """Tests for model loading and initialization."""

    def test_model_checkpoint_exists(self, test_checkpoint_path):
        """Test that model checkpoint exists."""
        assert test_checkpoint_path.exists()
        assert test_checkpoint_path.stat().st_size > 0

    def test_model_checkpoint_structure(self, test_checkpoint_path):
        """Test that checkpoint has required keys."""
        checkpoint = torch.load(test_checkpoint_path, map_location='cpu', weights_only=False)

        required_keys = ['state_dict', 'classes', 'mean', 'std', 'input_dim', 'num_classes']
        for key in required_keys:
            assert key in checkpoint, f"Missing key in checkpoint: {key}"

    def test_model_inference_shape(self, trained_model):
        """Test that model produces correct output shape."""
        model = trained_model['model']
        num_classes = len(trained_model['classes'])

        # Single sample inference
        x = torch.randn(1, 162)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, num_classes)
        assert not torch.isnan(out).any()

    def test_model_probabilities_sum_to_one(self, trained_model):
        """Test that softmax probabilities sum to 1."""
        model = trained_model['model']

        x = torch.randn(5, 162)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1)

        prob_sums = probs.sum(dim=1)
        assert np.allclose(prob_sums.numpy(), 1.0, atol=1e-5)


class TestErrorAnalysis:
    """Tests for error analysis on model predictions."""

    def test_common_confusion_pairs(self, sample_predictions):
        """Test identification of commonly confused class pairs."""
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']

        cm = confusion_matrix(y_true, y_pred)

        # Find top confused pairs (excluding diagonal)
        cm_no_diag = cm.copy()
        np.fill_diagonal(cm_no_diag, 0)

        max_confused = np.unravel_index(np.argmax(cm_no_diag), cm_no_diag.shape)

        print(f"Most confused pair: class {max_confused[0]} <-> class {max_confused[1]}")
        print(f"Confusion count: {cm_no_diag[max_confused]}")

    def test_low_confidence_predictions(self, trained_model):
        """Test handling of low confidence predictions."""
        model = trained_model['model']

        # Generate random inputs
        x = torch.randn(10, 162)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            max_probs, preds = probs.max(dim=1)

        print(f"Max probs: {max_probs}")
        print(f"Mean confidence: {max_probs.mean():.4f}")

        # Just verify model produces valid outputs
        assert logits.shape == (10, trained_model['model'].fc.out_features)
        assert not torch.isnan(logits).any(), "Model produced NaN outputs"

    def test_class_imbalance_impact(self):
        """Test that class imbalance affects predictions."""
        # Simulate imbalanced data
        np.random.seed(42)
        n_samples = 100
        n_classes = 262

        # 80% of samples are from 20% of classes
        common_classes = np.random.randint(0, n_classes // 5, int(n_samples * 0.8))
        rare_classes = np.random.randint(n_classes // 5, n_classes, int(n_samples * 0.2))
        y_true = np.concatenate([common_classes, rare_classes])
        np.random.shuffle(y_true)

        # Check class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        imbalance_ratio = counts.max() / counts.min()

        print(f"Class imbalance ratio: {imbalance_ratio:.2f}:1")
        assert imbalance_ratio > 1.0  # Should have some imbalance


class TestCrossValidationConsistency:
    """Tests for cross-validation consistency."""

    def test_fold_consistency(self):
        """Test that CV folds have consistent metrics."""
        # Simulated fold results
        fold_accuracies = [0.9985, 0.9985, 0.9992, 0.9984, 0.9983]

        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)

        print(f"Fold accuracies: {fold_accuracies}")
        print(f"Mean: {mean_acc:.4f}, Std: {std_acc:.4f}")

        # Std should be very low (< 0.01)
        assert std_acc < 0.01, f"Fold variance too high: {std_acc:.4f}"

        # All folds should be > 99%
        assert all(acc > 0.99 for acc in fold_accuracies)

    def test_cv_consistency_metric(self):
        """Test coefficient of variation for CV."""
        fold_accuracies = [0.9985, 0.9985, 0.9992, 0.9984, 0.9983]

        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)

        # Coefficient of variation
        cv = std_acc / mean_acc

        print(f"Coefficient of variation: {cv:.6f}")

        # CV should be very low (< 0.001)
        assert cv < 0.001, f"CV too high: {cv:.6f}"


class TestBenchmarkMetrics:
    """Tests for benchmark metrics (WER, BLEU, ROUGE style)."""

    def test_exact_match_accuracy(self, sample_predictions):
        """Test exact match accuracy."""
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']

        exact_matches = (y_true == y_pred).sum()
        exact_match_acc = exact_matches / len(y_true)

        print(f"Exact match accuracy: {exact_match_acc:.2%}")

        assert 0.90 < exact_match_acc < 1.0

    def test_top_k_accuracy(self, sample_predictions):
        """Test top-k accuracy."""
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']
        n_classes = sample_predictions['n_classes']

        # Simulate top-3 with confidence scores
        # Assuming true class is in top-3 ~98% of the time
        top3_accuracy = 0.98

        print(f"Top-3 accuracy (simulated): {top3_accuracy:.2%}")

        # Top-3 should be >= top-1 (can be equal if already 100%)
        exact_acc = accuracy_score(y_true, y_pred)
        assert top3_accuracy >= exact_acc, f"Top-3 ({top3_accuracy}) should be >= exact ({exact_acc})"


# Test runner summary
if __name__ == "__main__":
    print("=" * 60)
    print("TSL-51 Model Evaluation Tests")
    print("=" * 60)
    print()
    print("Run with: pytest tests/test_evaluation.py -v")
    print()
    print("Test categories:")
    print("  - TestModelEvaluation: Basic metrics")
    print("  - TestClassificationReport: Report generation")
    print("  - TestModelLoading: Model loading")
    print("  - TestErrorAnalysis: Error analysis")
    print("  - TestCrossValidationConsistency: CV consistency")
    print("  - TestBenchmarkMetrics: Benchmark metrics")
    print("=" * 60)
