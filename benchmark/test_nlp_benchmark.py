"""Tests for NLP Benchmark Suite."""

import pytest
import numpy as np
import torch
from pathlib import Path

# Test configuration
TEST_MODEL_PATH = Path(__file__).parent.parent / "models" / "tsl51_gru_20260503_183943.pt"


class TestWordErrorRate:
    """Tests for WER computation."""

    def test_wer_perfect_match(self):
        """Test WER with identical strings."""
        from benchmark.nlp_benchmark import NLPEvaluator

        evaluator = NLPEvaluator(TEST_MODEL_PATH)
        refs = ["hello world", "test sign"]
        preds = ["hello world", "test sign"]

        wer = evaluator.compute_wer(refs, preds)
        assert wer == 0.0, "Perfect match should have WER=0"

    def test_wer_one_error(self):
        """Test WER with one substitution."""
        from benchmark.nlp_benchmark import NLPEvaluator

        evaluator = NLPEvaluator(TEST_MODEL_PATH)
        refs = ["hello"]
        preds = ["goodbye"]

        wer = evaluator.compute_wer(refs, preds)
        assert 0.0 <= wer <= 1.0, "One word error should have valid WER"


class TestCharacterErrorRate:
    """Tests for CER computation."""

    def test_cer_perfect_match(self):
        """Test CER with identical strings."""
        from benchmark.nlp_benchmark import NLPEvaluator

        evaluator = NLPEvaluator(TEST_MODEL_PATH)
        refs = ["hello", "test"]
        preds = ["hello", "test"]

        cer = evaluator.compute_cer(refs, preds)
        assert cer == 0.0, "Perfect match should have CER=0"


class TestBLEScore:
    """Tests for BLEU score computation."""

    def test_bleu_perfect_match(self):
        """Test BLEU with identical strings."""
        from benchmark.nlp_benchmark import NLPEvaluator

        evaluator = NLPEvaluator(TEST_MODEL_PATH)
        refs = ["the quick brown fox"]
        preds = ["the quick brown fox"]

        bleu = evaluator.compute_bleu(refs, preds)
        assert bleu > 0.9, "Perfect match should have high BLEU"

    def test_bleu_no_overlap(self):
        """Test BLEU with no overlap."""
        from benchmark.nlp_benchmark import NLPEvaluator

        evaluator = NLPEvaluator(TEST_MODEL_PATH)
        refs = ["hello world"]
        preds = ["goodbye world"]

        bleu = evaluator.compute_bleu(refs, preds)
        assert 0.0 <= bleu <= 1.0, "BLEU should be between 0 and 1"


class TestROUGEScore:
    """Tests for ROUGE score computation."""

    def test_rouge_perfect_match(self):
        """Test ROUGE with identical strings."""
        from benchmark.nlp_benchmark import NLPEvaluator

        evaluator = NLPEvaluator(TEST_MODEL_PATH)
        refs = ["the quick brown fox"]
        preds = ["the quick brown fox"]

        rouge = evaluator.compute_rouge(refs, preds)
        assert rouge['fmeasure'] > 0.9, "Perfect match should have high ROUGE F1"


class TestModelEvaluator:
    """Tests for model evaluation."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator fixture."""
        from benchmark.nlp_benchmark import NLPEvaluator
        return NLPEvaluator(TEST_MODEL_PATH)

    def test_load_model(self, evaluator):
        """Test model loading."""
        assert evaluator.load_model(), "Model should load successfully"
        assert evaluator.model is not None, "Model should be set"
        assert evaluator.checkpoint is not None, "Checkpoint should be set"

    def test_sign_evaluation(self, evaluator):
        """Test sign evaluation on samples."""
        if not evaluator.load_model():
            pytest.skip("Model not found")

        results = evaluator.evaluate_sign_samples(n_samples=100)

        assert 'n_samples' in results
        assert 'mean_confidence' in results
        assert 'approximate_perplexity' in results
        assert 0.0 <= results['mean_confidence'] <= 1.0

    def test_confusion_pairs(self, evaluator):
        """Test confusion pair analysis."""
        if not evaluator.load_model():
            pytest.skip("Model not found")

        y_true = np.random.randint(0, 262, 100)
        y_pred = y_true.copy()
        # Add some confusion
        y_pred[::10] = np.random.randint(0, 262, 10)

        pairs = evaluator.analyze_confusion_pairs(y_true, y_pred, top_n=5)
        assert isinstance(pairs, list)
        assert len(pairs) <= 5


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_benchmark_result_creation(self):
        """Test creating a benchmark result."""
        from benchmark.nlp_benchmark import BenchmarkResult

        result = BenchmarkResult(
            metric_name="accuracy",
            value=0.95,
            details={"samples": 100}
        )

        assert result.metric_name == "accuracy"
        assert result.value == 0.95
        assert result.details["samples"] == 100

    def test_benchmark_result_to_dict(self):
        """Test converting result to dictionary."""
        from benchmark.nlp_benchmark import BenchmarkResult

        result = BenchmarkResult(
            metric_name="wer",
            value=0.1
        )

        result_dict = result.to_dict()
        assert "metric" in result_dict
        assert "value" in result_dict
        assert result_dict["metric"] == "wer"
        assert result_dict["value"] == 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
