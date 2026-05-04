"""Inference utilities for TSL-51."""

import torch
import numpy as np


class InferenceUtils:
    @staticmethod
    def model_warmup(model, input_dim, device, num_runs=3):
        """Warm up model with dummy data."""
        model.eval()
        dummy = torch.randn(1, input_dim).to(device)
        with torch.no_grad():
            for _ in range(num_runs):
                model(dummy)

    @staticmethod
    def get_prediction_entropy(probs: np.ndarray) -> float:
        """Compute prediction entropy for uncertainty."""
        probs = np.clip(probs, 1e-10, 1.0)
        return -np.sum(probs * np.log(probs), axis=-1)

    @staticmethod
    def get_top_k_predictions(probs: np.ndarray, k=5):
        """Get top-k predictions with probabilities."""
        indices = np.argsort(probs)[::-1][:k]
        return [(i, probs[i]) for i in indices]

    @staticmethod
    def confidence_threshold(predictions, threshold=0.5):
        """Filter predictions by confidence threshold."""
        return [(p, conf) for p, conf in predictions if conf >= threshold]
