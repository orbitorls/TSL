# pyright: reportMissingImports=false, reportGeneralTypeIssues=false, reportCallIssue=false, reportArgumentType=false, reportOperatorIssue=false
"""
TSL-51 Inference Script
=======================
Use this script to run inference with trained model.

FEATURE LEVELS:
===============
- basic: 162 features (hand + pose)
- finger: 252 features (hand + pose + finger)
- full: 1596 features (hand + pose + face)
- face: 1434 features (face only)

INPUT FORMAT:
=============
- numpy array shape: (features,) or (batch_size, features)
- Values: normalized coordinates

USAGE:
======
# Load model and predict
tsl-inference --model models/tsl51_xxx.pt --input your_data.npz

# Or use as Python module
from src.inference.runner import TSLPredictor
predictor = TSLPredictor('models/tsl51_xxx.pt')
label, confidence = predictor.predict(landmarks)

# AutoGluon model (directory instead of .pt file)
tsl-inference --model models/autogluon_tsl51 --input your_data.npz
"""

import argparse
import os
import sys
from pathlib import Path

from ..utils.security import validate_file_path

if sys.platform == "win32":
    reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure_stdout):
        reconfigure_stdout(encoding="utf-8", errors="replace")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch

from src.core.models import (
    MLP,
)
from src.core.models import (
    MODEL_REGISTRY as MODEL_CLASSES,
)
from src.core.normalizer import resolve_checkpoint_preprocessing

# NEW: Use core module for shared functionality
from ..data.feature_extraction import (
    FEATURE_DIMS,
    extract_features_from_landmark_df,
    extract_sequence_from_landmark_df,
)

# Try to import AutoGluon support
try:
    from ..train.autogluon_model import AutoGluonModel

    _has_autogluon = True
except ImportError:
    _has_autogluon = False
    AutoGluonModel = None

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent


# ============================================================================
# PREDICTOR CLASS
# ============================================================================
class TSLPredictor:
    """
    Thai Sign Language Predictor

    Usage:
        predictor = TSLPredictor('models/tsl51_xxx.pt')
        label, confidence = predictor.predict(landmarks)

    Supports both PyTorch models (.pt files) and AutoGluon models (directories).
    """

    FEATURE_LEVELS = FEATURE_DIMS

    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        self.is_autogluon = False

        # Initialize instance variables
        self.model = None
        self.classes = []
        self.input_dim = 162
        self.num_classes = 51
        self.mean = None
        self.std = None
        self.model_name = "unknown"
        self.accuracy = 0.0
        self.feature_level = "basic"
        self.seq_mode = False
        self.target_frames = 30

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Detect model type: AutoGluon (directory) or PyTorch (.pt file)
        if self.model_path.is_dir():
            # AutoGluon model (saved as directory)
            if not _has_autogluon:
                raise ImportError(
                    "AutoGluon model detected but autogluon not installed. "
                    "Install with: pip install autogluon.tabular[all]"
                )
            self._load_autogluon_model()
        elif self.model_path.suffix == ".pt":
            # PyTorch model
            self._load_pytorch_model()
        else:
            raise ValueError(
                f"Unknown model format: {model_path}. Expected .pt file or AutoGluon directory."
            )

    def _load_autogluon_model(self):
        """Load AutoGluon model from directory."""
        self.is_autogluon = True

        # Load AutoGluon model
        if AutoGluonModel is not None:
            self.model = AutoGluonModel.load(str(self.model_path))

        # Get metadata from AutoGluon model
        if self.model is not None:
            if hasattr(self.model, "classes_"):
                self.classes = [str(c) for c in self.model.classes_]
            if hasattr(self.model, "input_dim"):
                self.input_dim = self.model.input_dim
            if hasattr(self.model, "num_classes"):
                self.num_classes = self.model.num_classes

            # Get normalization stats
            if hasattr(self.model, "mean_") and hasattr(self.model, "std_"):
                if self.model.mean_ is not None and self.model.std_ is not None:
                    self.mean = self.model.mean_
                    self.std = self.model.std_
                else:
                    # Try to load from metadata file
                    metadata_path = Path(str(self.model_path) + ".pkl")
                    if metadata_path.exists():
                        import pickle

                        # Safe pickle loading with validation
                        with open(metadata_path, "rb") as f:
                            try:
                                metadata = pickle.load(f)
                                # Validate metadata structure before using
                                if not isinstance(metadata, dict):
                                    raise ValueError("Invalid metadata: expected dict")
                                allowed_keys = {
                                    "normalization_mean",
                                    "normalization_std",
                                    "fold_idx",
                                    "val_acc",
                                    "val_f1",
                                    "classes",
                                    "model_path",
                                }
                                unexpected_keys = set(metadata.keys()) - allowed_keys
                                if unexpected_keys:
                                    raise ValueError(f"Invalid metadata keys: {unexpected_keys}")
                                if (
                                    "normalization_mean" in metadata
                                    and "normalization_std" in metadata
                                ):
                                    self.mean = np.array(metadata["normalization_mean"])
                                    self.std = np.array(metadata["normalization_std"])
                                else:
                                    raise KeyError("AutoGluon model missing normalization stats")
                            except (pickle.PickleError, ValueError) as e:
                                raise ValueError(f"Invalid or corrupted metadata file: {e}")
                    else:
                        raise KeyError("AutoGluon model missing normalization stats (mean/std)")

        self.model_name = "autogluon"
        self.accuracy = 0.0  # AutoGluon doesn't store this in wrapper
        self.feature_level = "basic"  # Default for AutoGluon
        self.seq_mode = False
        self.target_frames = 30

        print("Model loaded: AutoGluon (TabularPredictor)")
        print(f"Feature level: {self.feature_level}")
        print(f"Input dim: {self.input_dim}")
        print(f"Classes: {len(self.classes)}")
        print(f"Device: {self.device}")

        # Show leaderboard if available
        try:
            if self.model is not None and hasattr(self.model, "get_model_summary"):
                leaderboard = self.model.get_model_summary()
                print(f"\nAutoGluon Leaderboard:\n{leaderboard}")
        except Exception:
            pass

    def _load_pytorch_model(self):
        """Load PyTorch model from .pt file."""
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)

        # Load metadata with legacy schema support
        if "classes" in checkpoint:
            self.classes = [str(c) for c in checkpoint["classes"]]
        elif "labels" in checkpoint and "label_to_idx" in checkpoint:
            label_to_idx = checkpoint["label_to_idx"]
            self.classes = [None] * len(label_to_idx)
            for label, idx in label_to_idx.items():
                if 0 <= idx < len(self.classes):
                    self.classes[idx] = str(label)
            self.classes = [c if c is not None else str(i) for i, c in enumerate(self.classes)]
        else:
            raise KeyError("Checkpoint missing classes/labels metadata")

        preprocessing = resolve_checkpoint_preprocessing(self.model_path, checkpoint)
        self.mean = preprocessing["mean"]
        self.std = preprocessing["std"]

        self.input_dim = int(preprocessing["input_dim"])
        self.num_classes = checkpoint.get("num_classes", len(self.classes))
        self.model_name = checkpoint.get("model", checkpoint.get("config", {}).get("model", "gru"))
        self.accuracy = checkpoint.get("accuracy", 0.0)

        # Get feature level and sequence mode from manifest when present,
        # otherwise preserve legacy checkpoint metadata.
        self.feature_level = str(preprocessing["feature_level"])
        self.seq_mode = bool(preprocessing["seq_mode"])
        self.target_frames = int(preprocessing["target_frames"])

        if self.seq_mode:
            print(f"Sequence mode: ON (target_frames={self.target_frames})")

        # Validate dimensions. Sidecar manifests are already strict; legacy
        # checkpoints retain the existing warning-only behavior.
        expected_dim = FEATURE_DIMS.get(self.feature_level, 162)
        if self.input_dim != expected_dim and preprocessing["manifest"] is None:
            print(f"WARNING: Model has {self.input_dim} features but expected {expected_dim}")

        # Create model based on type
        model_class = MODEL_CLASSES.get(self.model_name, MLP)

        config = checkpoint.get("config", {})
        self.model = model_class(
            self.input_dim,
            self.num_classes,
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 3),
            dropout=config.get("dropout", 0.3),
        )

        # Load weights with legacy fallback
        state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            raise KeyError("Checkpoint missing state_dict")
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            if self.model_name == "gru":
                fallback = MLP(
                    self.input_dim,
                    self.num_classes,
                    hidden_dim=config.get("hidden_dim", 256),
                    num_layers=config.get("num_layers", 3),
                    dropout=config.get("dropout", 0.3),
                )
                fallback.load_state_dict(state_dict)
                self.model = fallback
            else:
                raise
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded: {self.model_name}")
        print(f"Feature level: {self.feature_level}")
        print(f"Input dim: {self.input_dim}")
        print(f"Classes: {len(self.classes)}")
        print(f"Training accuracy: {self.accuracy * 100:.2f}%")
        print(f"Device: {self.device}")

    def validate_input(self, x):
        """Validate and fix input dimensions.

        Accepts:
        - (feature_dim,)                 — single mean-aggregated sample
        - (batch, feature_dim)           — batch of mean-aggregated samples
        - (T, feature_dim)               — single sequence (if seq_mode)
        - (batch, T, feature_dim)        — batch of sequences (if seq_mode)
        """
        x = np.array(x, dtype=np.float32)

        if self.seq_mode:
            # Sequence mode: expected shape (batch, T, input_dim)
            if x.ndim == 2:
                # (T, input_dim) — single sample, add batch dim
                x = x[np.newaxis, :, :]  # (1, T, input_dim)
            elif x.ndim == 3:
                pass  # already (batch, T, input_dim)
            else:
                raise ValueError(f"seq_mode expects 2D or 3D input, got shape {x.shape}")
            if x.shape[2] != self.input_dim:
                raise ValueError(f"Expected {self.input_dim} features per frame, got {x.shape[2]}")
        else:
            # Flat mode: expected shape (batch, input_dim)
            if x.ndim > 2:
                x = x.reshape(-1, self.input_dim)
            if x.ndim == 1:
                if len(x) != self.input_dim:
                    raise ValueError(f"Expected {self.input_dim} features, got {len(x)}")
                x = x.reshape(1, -1)
            else:
                if x.shape[1] != self.input_dim:
                    if x.shape[0] == self.input_dim:
                        x = x.reshape(1, -1)
                    else:
                        raise ValueError(
                            f"Expected features dimension {self.input_dim}, got {x.shape[1]}"
                        )

        return x

    def predict(self, landmarks, return_top_k=1):
        """
        Predict sign from landmarks.

        Args:
            landmarks: numpy array of shape (features,) or (batch, features)
            return_top_k: number of top predictions to return

        Returns:
            If return_top_k=1: (label, confidence)
            If return_top_k>1: [(label, confidence), ...]
        """
        # Validate and fix input
        x = self.validate_input(landmarks)

        # Normalize (using training statistics)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std

        if self.is_autogluon:
            # AutoGluon prediction
            # AutoGluon expects 2D input (batch, features)
            if x.ndim == 1:
                x = x.reshape(1, -1)

            # Get probability predictions
            if self.model is not None and hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(x)

                # Get top k predictions
                if return_top_k == 1:
                    prob = float(probs[0].max())
                    pred = int(probs[0].argmax())
                    return (self.classes[pred], prob)
                else:
                    # Get top k indices and probabilities
                    top_indices = np.argsort(probs[0])[-return_top_k:][::-1]
                    return [(self.classes[idx], float(probs[0][idx])) for idx in top_indices]
        else:
            # PyTorch prediction
            # Convert to tensor
            x = torch.tensor(x, dtype=torch.float32).to(self.device)

            # Predict
            with torch.no_grad():
                if self.model is not None:
                    logits = self.model(x)
                    probs = torch.softmax(logits, dim=1)

                    # Get top k predictions
                    if return_top_k == 1:
                        prob, pred = probs[0].max(0)
                        return (self.classes[pred.item()], prob.item())
                    else:
                        top_probs, top_indices = probs[0].topk(return_top_k)
                        return [
                            (self.classes[idx.item()], prob.item())
                            for idx, prob in zip(top_indices, top_probs, strict=False)
                        ]

        return (self.classes[0], 0.0) if return_top_k == 1 else [(self.classes[0], 0.0)]

    def predict_from_csv(self, csv_path, feature_level=None, return_top_k=1):
        """Predict from CSV file containing landmarks.

        Automatically uses sequence extraction when the model was trained
        with ``--seq-mode``, otherwise falls back to mean-aggregated features.
        """
        import pandas as pd

        df = pd.read_csv(csv_path)
        level = feature_level or self.feature_level
        if self.seq_mode:
            features = extract_sequence_from_landmark_df(df, level, self.target_frames)
            # shape: (target_frames, input_dim) — validate_input will add batch dim
        else:
            features = extract_features_from_landmark_df(df, level)
        return self.predict(features, return_top_k=return_top_k)

    def predict_from_npz(self, npz_path, return_top_k=1):
        """Predict from NumPy archive file."""
        data = np.load(npz_path)
        if "X" in data:
            features = data["X"]
        elif "features" in data:
            features = data["features"]
        else:
            raise ValueError(f"Unknown npz format: {npz_path}. Expected 'X' or 'features' key.")
        return self.predict(features, return_top_k=return_top_k)


def load_model(model_path, device=None):
    """Compatibility wrapper returning a loaded TSLPredictor."""
    return TSLPredictor(model_path, device=device)


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="TSL-51 Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    tsl-inference --model models/tsl51_gru.pt --input data.npz
    tsl-inference --model models/autogluon_tsl51 --input data.csv
    """,
    )
    parser.add_argument(
        "--model", type=Path, required=True, help="Model checkpoint file or directory"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input file (.npz or .csv)")
    parser.add_argument("--top-k", type=int, default=1, help="Number of top predictions")
    args = parser.parse_args()

    # Validate file paths for security
    try:
        validate_file_path(args.model, allowed_extensions={".pt", ".pkl"})
    except ValueError as e:
        print(f"Error validating model path: {e}")
        return 1

    try:
        validate_file_path(args.input, allowed_extensions={".npz", ".csv"})
    except ValueError as e:
        print(f"Error validating input path: {e}")
        return 1

    print("=" * 60)
    print("TSL-51 INFERENCE")
    print("=" * 60)

    # Load predictor
    predictor = TSLPredictor(args.model)

    # Load input
    input_path = Path(args.input)
    print(f"\nInput: {input_path}")

    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return 1

    # Determine file type and predict
    if input_path.suffix == ".npz":
        data = np.load(input_path)
        landmarks = data["X"] if "X" in data else data[data.files[0]]
        preds = predictor.predict(landmarks, return_top_k=args.top_k)
    elif input_path.suffix == ".csv":
        preds = predictor.predict_from_csv(input_path, return_top_k=args.top_k)
    else:
        print("ERROR: Unsupported file format. Use .npz or .csv")
        return 1

    # Show results
    print("\n" + "=" * 60)
    print("PREDICTIONS")
    print("=" * 60)
    if isinstance(preds, list):
        for i, (label, conf) in enumerate(preds):
            print(f"{i + 1}. {label}: {conf * 100:.2f}%")
    else:
        label, conf = preds
        print(f"1. {label}: {conf * 100:.2f}%")

    print("\n" + "=" * 60)
    print("MODEL INFO")
    print("=" * 60)
    print(f"Model type: {predictor.model_name}")
    print(f"Classes: {len(predictor.classes)}")
    print(f"Feature level: {predictor.feature_level}")
    print(f"Input features: {predictor.input_dim}")

    if not predictor.is_autogluon:
        print(f"Model accuracy: {predictor.accuracy * 100:.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
