"""
TSL-51 Inference Script
=======================
Use this script to run inference with trained model.

FEATURE LEVELS:
===============
- basic: 162 features (hand + pose)
- finger: 258 features (hand + pose + finger)
- full: 1596 features (hand + pose + face)
- face: 1434 features (face only)

INPUT FORMAT:
=============
- numpy array shape: (features,) or (batch_size, features)
- Values: normalized coordinates

USAGE:
======
# Load model and predict
python inference.py --model models/tsl51_xxx.pt --input your_data.npz

# Or use as Python module
from inference import TSLPredictor
predictor = TSLPredictor('models/tsl51_xxx.pt')
label, confidence = predictor.predict(landmarks)
"""

import argparse
import sys
import os
from pathlib import Path

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from ..data.feature_extraction import extract_features_from_landmark_df, extract_sequence_from_landmark_df, FEATURE_DIMS
from ..train.models import MLP, GRUModel, MOPGRU, HybridGRUTransformer, MODEL_CLASSES

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent


# ============================================================================
# Shared feature extraction and model definitions
# ============================================================================
# Feature extraction and model classes are imported from shared modules to keep
# training and inference behavior synchronized.


# ============================================================================
# PREDICTOR CLASS
# ============================================================================
class TSLPredictor:
    """
    Thai Sign Language Predictor
    
    Usage:
        predictor = TSLPredictor('models/tsl51_xxx.pt')
        label, confidence = predictor.predict(landmarks)
    """
    
    FEATURE_LEVELS = FEATURE_DIMS
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Load metadata with legacy schema support
        if 'classes' in checkpoint:
            self.classes = [str(c) for c in checkpoint['classes']]
        elif 'labels' in checkpoint and 'label_to_idx' in checkpoint:
            label_to_idx = checkpoint['label_to_idx']
            self.classes = [None] * len(label_to_idx)
            for label, idx in label_to_idx.items():
                if 0 <= idx < len(self.classes):
                    self.classes[idx] = str(label)
            self.classes = [c if c is not None else str(i) for i, c in enumerate(self.classes)]
        else:
            raise KeyError('Checkpoint missing classes/labels metadata')

        self.mean = np.array(checkpoint.get('normalization_mean', checkpoint.get('mean')))
        self.std = np.array(checkpoint.get('normalization_std', checkpoint.get('std')))
        if self.mean is None or self.std is None:
            raise KeyError('Checkpoint missing normalization stats (mean/std)')

        self.input_dim = checkpoint.get('input_dim', int(self.mean.shape[0]))
        self.num_classes = checkpoint.get('num_classes', len(self.classes))
        self.model_name = checkpoint.get('model', checkpoint.get('config', {}).get('model', 'gru'))
        self.accuracy = checkpoint.get('accuracy', 0.0)
        
        # Get feature level and sequence mode from checkpoint
        self.feature_level = checkpoint.get('config', {}).get('feature_level', 'basic')
        self.seq_mode = checkpoint.get('seq_mode', False)
        self.target_frames = checkpoint.get('target_frames', 30)

        if self.seq_mode:
            print(f"Sequence mode: ON (target_frames={self.target_frames})")

        # Validate dimensions
        expected_dim = FEATURE_DIMS.get(self.feature_level, 162)
        if self.input_dim != expected_dim:
            print(f"WARNING: Model has {self.input_dim} features but expected {expected_dim}")
        
        # Create model based on type
        model_class = MODEL_CLASSES.get(self.model_name, MLP)
        
        config = checkpoint.get('config', {})
        self.model = model_class(
            self.input_dim, 
            self.num_classes,
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.3)
        )
        
        # Load weights with legacy fallback
        state_dict = checkpoint.get('state_dict')
        if state_dict is None:
            raise KeyError('Checkpoint missing state_dict')
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            if self.model_name == 'gru':
                fallback = MLP(
                    self.input_dim,
                    self.num_classes,
                    hidden_dim=config.get('hidden_dim', 256),
                    num_layers=config.get('num_layers', 3),
                    dropout=config.get('dropout', 0.3),
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
        print(f"Training accuracy: {self.accuracy*100:.2f}%")
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
                        raise ValueError(f"Expected features dimension {self.input_dim}, got {x.shape[1]}")

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
        x = (x - self.mean) / self.std
        
        # Convert to tensor
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
        
        # Get top k predictions
        if return_top_k == 1:
            prob, pred = probs[0].max(0)
            return self.classes[pred.item()], prob.item()
        else:
            top_probs, top_indices = probs[0].topk(return_top_k)
            return [(self.classes[idx.item()], prob.item()) 
                    for idx, prob in zip(top_indices, top_probs)]
    
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
        if 'X' in data:
            features = data['X']
        elif 'features' in data:
            features = data['features']
        else:
            raise ValueError(f"Unknown npz format: {npz_path}. Expected 'X' or 'features' key.")
        return self.predict(features, return_top_k=return_top_k)


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='TSL-51 Inference')
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model (.pt)")
    parser.add_argument("--input", type=str, required=True,
                       help="Input data: .npz, .csv, or feature array")
    parser.add_argument("--top-k", type=int, default=3,
                       help="Number of top predictions to show")
    parser.add_argument("--feature-level", type=str, default=None,
                       choices=["basic", "finger", "enhanced", "full", "face"],
                       help="Feature level for model compatibility (stored in model)")
    args = parser.parse_args()
    
    print("="*60)
    print("TSL-51 INFERENCE")
    print("="*60)
    
    # Load predictor
    predictor = TSLPredictor(args.model)
    
    # Load input
    input_path = Path(args.input)
    print(f"\nInput: {input_path}")
    
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return 1
    
    # Determine file type and predict
    if input_path.suffix == '.npz':
        data = np.load(input_path)
        if 'X' in data:
            landmarks = data['X']
        else:
            landmarks = data[data.files[0]]
        preds = predictor.predict(landmarks, return_top_k=args.top_k)
    elif input_path.suffix == '.csv':
        preds = predictor.predict_from_csv(input_path, return_top_k=args.top_k)
    else:
        print("ERROR: Unsupported file format. Use .npz or .csv")
        return 1
    
    # Show results
    print("\n" + "="*60)
    print("PREDICTIONS")
    print("="*60)
    for i, (label, conf) in enumerate(preds):
        print(f"{i+1}. {label}: {conf*100:.2f}%")
    
    print("\n" + "="*60)
    print("MODEL INFO")
    print("="*60)
    print(f"Classes: {len(predictor.classes)}")
    print(f"Feature level: {predictor.feature_level}")
    print(f"Input features: {predictor.input_dim}")
    print(f"Model accuracy: {predictor.accuracy*100:.2f}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
