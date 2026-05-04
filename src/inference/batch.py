"""Batch inference utilities for TSL models."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.train.compat import setup_windows_encoding
from src.train.models import MODEL_CLASSES


class BatchInference:
    """Batch inference handler for TSL models."""

    def __init__(
        self,
        model_path: str,
        model_type: str = "gru",
        device: Optional[str] = None,
        batch_size: int = 64,
    ):
        """
        Initialize batch inference handler.

        Args:
            model_path: Path to PyTorch model checkpoint
            model_type: Model architecture
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for inference
        """
        self.model_path = model_path
        self.model_type = model_type
        self.batch_size = batch_size

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")
        self.classes = checkpoint["classes"]
        self.mean = np.array(checkpoint["mean"])
        self.std = np.array(checkpoint["std"])

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load model
        model_class = MODEL_CLASSES.get(model_type, MODEL_CLASSES["gru"])
        self.model = model_class(
            input_dim=162,
            num_classes=len(self.classes),
            hidden_dim=checkpoint.get("hidden_dim", 256),
            num_layers=checkpoint.get("num_layers", 3),
            dropout=0.0,  # No dropout during inference
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded: {model_type}")
        print(f"Device: {self.device}")
        print(f"Classes: {len(self.classes)}")

    def predict(
        self,
        features: np.ndarray,
        return_probs: bool = False,
        top_k: int = 1,
    ) -> Tuple[List[str], Optional[np.ndarray]]:
        """
        Predict signs for a batch of features.

        Args:
            features: Feature array (n_samples, 162)
            return_probs: Whether to return probabilities
            top_k: Number of top predictions to return

        Returns:
            Tuple of (predictions, probabilities)
        """
        n_samples = len(features)
        predictions = []
        all_probs = None

        # Process in batches
        for i in range(0, n_samples, self.batch_size):
            batch_features = features[i : i + self.batch_size]

            # Normalize
            batch_normalized = (batch_features - self.mean) / self.std

            # Convert to tensor
            batch_tensor = torch.FloatTensor(batch_normalized).to(self.device)

            # Predict
            with torch.no_grad():
                logits = self.model(batch_tensor)
                probs = logits.softmax(dim=1)

                if top_k > 1:
                    top_probs, top_indices = torch.topk(probs, top_k, dim=1)
                    batch_preds = [
                        [self.classes[idx.item()] for idx in indices]
                        for indices in top_indices
                    ]
                    batch_probs = top_probs.cpu().numpy()
                else:
                    top_indices = probs.argmax(dim=1)
                    batch_preds = [self.classes[idx.item()] for idx in top_indices]
                    batch_probs = probs.cpu().numpy()

                predictions.extend(batch_preds)
                if return_probs:
                    if all_probs is None:
                        all_probs = batch_probs
                    else:
                        all_probs = np.concatenate([all_probs, batch_probs], axis=0)

        return predictions, all_probs if return_probs else None

    def predict_from_npz(
        self,
        npz_path: str,
        return_probs: bool = False,
        top_k: int = 1,
    ) -> Tuple[List[str], Optional[np.ndarray]]:
        """
        Predict from NumPy archive.

        Args:
            npz_path: Path to .npz file with 'X' key
            return_probs: Whether to return probabilities
            top_k: Number of top predictions

        Returns:
            Tuple of (predictions, probabilities)
        """
        data = np.load(npz_path)
        features = data["X"]
        return self.predict(features, return_probs=return_probs, top_k=top_k)

    def predict_from_csv(
        self,
        csv_path: str,
        return_probs: bool = False,
        top_k: int = 1,
    ) -> Tuple[List[str], Optional[np.ndarray]]:
        """
        Predict from CSV file (assumes first column is label, rest are features).

        Args:
            csv_path: Path to CSV file
            return_probs: Whether to return probabilities
            top_k: Number of top predictions

        Returns:
            Tuple of (predictions, probabilities)
        """
        import pandas as pd

        df = pd.read_csv(csv_path)
        # Assume all columns except first are features
        features = df.iloc[:, 1:].values.astype(np.float32)
        return self.predict(features, return_probs=return_probs, top_k=top_k)

    def save_results(
        self,
        predictions: List[str],
        output_path: str,
        probs: Optional[np.ndarray] = None,
    ):
        """
        Save predictions to file.

        Args:
            predictions: List of predictions
            output_path: Output file path (.csv, .json, or .txt)
            probs: Optional probabilities array
        """
        output_path = Path(output_path)

        if output_path.suffix == ".csv":
            import pandas as pd

            df = pd.DataFrame({"prediction": predictions})
            if probs is not None:
                for i, cls in enumerate(self.classes):
                    df[f"prob_{cls}"] = probs[:, i]
            df.to_csv(output_path, index=False)

        elif output_path.suffix == ".json":
            import json

            results = {"predictions": predictions}
            if probs is not None:
                results["probabilities"] = probs.tolist()
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

        else:  # txt
            with open(output_path, "w") as f:
                for i, pred in enumerate(predictions):
                    if probs is not None:
                        f.write(f"{i}: {pred} ({probs[i].max():.4f})\n")
                    else:
                        f.write(f"{i}: {pred}\n")

        print(f"Results saved to: {output_path}")


def main():
    """CLI for batch inference."""
    import argparse

    parser = argparse.ArgumentParser(description="Batch inference for TSL models")
    parser.add_argument("--model", type=str, required=True, help="Path to PyTorch model")
    parser.add_argument("--input", type=str, required=True, help="Input file (.npz or .csv)")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--type", type=str, default="gru", choices=MODEL_CLASSES.keys())
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, help="Device (cuda/cpu)")
    parser.add_argument("--top-k", type=int, default=1, help="Top-k predictions")
    parser.add_argument("--probs", action="store_true", help="Return probabilities")

    args = parser.parse_args()

    # Initialize batch inference
    inference = BatchInference(
        model_path=args.model,
        model_type=args.type,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Load input and predict
    input_path = Path(args.input)
    if input_path.suffix == ".npz":
        predictions, probs = inference.predict_from_npz(
            args.input, return_probs=args.probs, top_k=args.top_k
        )
    elif input_path.suffix == ".csv":
        predictions, probs = inference.predict_from_csv(
            args.input, return_probs=args.probs, top_k=args.top_k
        )
    else:
        print(f"Unsupported input format: {input_path.suffix}")
        return 1

    # Save results
    if args.output:
        inference.save_results(predictions, args.output, probs)
    else:
        # Print results
        for i, pred in enumerate(predictions):
            if isinstance(pred, list):
                print(f"{i}: {pred}")
            else:
                print(f"{i}: {pred}")

    return 0


if __name__ == "__main__":
    setup_windows_encoding()
    sys.exit(main())
