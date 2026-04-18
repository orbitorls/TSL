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

PROJECT_DIR = Path(__file__).resolve().parent


# ============================================================================
# FEATURE EXTRACTION UTILITIES
# ============================================================================
def extract_features_from_landmark_df(lm_df, feature_level='basic'):
    """
    Extract features from landmark DataFrame based on feature level.
    This MUST match the training feature extraction exactly!
    """
    features = []
    
    from utils.dataset_utils import safe_mean
    # ===== 1. BASIC: Hand + Pose (162) =====
    # Left hand (21 * 3 = 63)
    for i in range(21):
        for c in ['x', 'y', 'z']:
            col = f'lh_{c}{i}'
            if col in lm_df.columns:
                features.append(safe_mean(lm_df[col]))
            else:
                features.append(0.0)
    
    # Right hand (21 * 3 = 63)
    for i in range(21):
        for c in ['x', 'y', 'z']:
            col = f'rh_{c}{i}'
            if col in lm_df.columns:
                features.append(safe_mean(lm_df[col]))
            else:
                features.append(0.0)
    
    # Pose (12 * 3 = 36)
    pose_cols = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
                'lbrow_outer', 'lbrow_inner', 'rbrow_inner', 'rbrow_outer',
                'mouth_right', 'mouth_left']
    for base in pose_cols:
        for c in ['x', 'y', 'z']:
            col = f'{base}_{c}'
            if col in lm_df.columns:
                features.append(safe_mean(lm_df[col]))
            else:
                features.append(0.0)
    
    # ===== 2. FINGER: Additional finger details =====
    if feature_level in ['finger', 'full', 'face']:
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for hand_prefix in ['lh_', 'rh_']:
            for finger in finger_names:
                for c in ['x', 'y', 'z']:
                    for joint in ['mcp', 'pip', 'dip']:
                        col = f'{hand_prefix}{finger}_{joint}_{c}'
                        if col in lm_df.columns:
                            features.append(safe_mean(lm_df[col]))
                        else:
                            features.append(0.0)
    
    # ===== 3. FACE: 478 Facial Landmarks (1434 features) =====
    if feature_level in ['face', 'full']:
        for i in range(478):
            for c in ['x', 'y', 'z']:
                col = f'face_{c}{i}'
                if col in lm_df.columns:
                    features.append(safe_mean(lm_df[col]))
                else:
                    features.append(0.0)
    
    return np.array(features, dtype=np.float32)


# Feature level to dimension mapping
FEATURE_DIMS = {
    'basic': 162,
    'finger': 258,
    'full': 1596,
    'face': 1434,
}


# ============================================================================
# MODEL DEFINITIONS (must match training)
# ============================================================================
class GRUModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.gru = torch.nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                         dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.norm = torch.nn.LayerNorm(hidden_dim * 2)
        self.fc = torch.nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.norm(out)
        out = self.dropout(out)
        return self.fc(out)


class MOPGRU(torch.nn.Module):
    """Modified GRU with multiplied update gate."""
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.rnn = torch.nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, 
                              bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.norm = torch.nn.LayerNorm(hidden_dim * 2)
        self.fc = torch.nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.norm(out)
        out = self.dropout(out)
        return self.fc(out)


class HybridGRUTransformer(torch.nn.Module):
    """Hybrid GRU + Transformer."""
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_layers=3, dropout=0.3, nhead=8):
        super().__init__()
        self.gru = torch.nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, 
                              bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.proj = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        gru_out, _ = self.gru(x)
        proj_out = self.proj(gru_out)
        trans_out = self.transformer(proj_out)
        out = trans_out[:, -1, :]
        out = self.norm(out)
        out = self.dropout(out)
        return self.fc(out)


class MLP(torch.nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()
        layers = [
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout)
        ]
        for _ in range(num_layers - 1):
            layers.extend([
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout)
            ])
        layers.append(torch.nn.Linear(hidden_dim, num_classes))
        self.net = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


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
        
        # Load metadata
        self.classes = checkpoint['classes']
        self.mean = np.array(checkpoint['mean'])
        self.std = np.array(checkpoint['std'])
        self.input_dim = checkpoint['input_dim']
        self.num_classes = checkpoint['num_classes']
        self.model_name = checkpoint.get('model', 'gru')
        self.accuracy = checkpoint.get('accuracy', 0.0)
        
        # Get feature level from config
        self.feature_level = checkpoint.get('config', {}).get('feature_level', 'basic')
        
        # Validate dimensions
        expected_dim = FEATURE_DIMS.get(self.feature_level, 162)
        if self.input_dim != expected_dim:
            print(f"WARNING: Model has {self.input_dim} features but expected {expected_dim}")
        
        # Create model based on type
        model_classes = {
            'mlp': MLP,
            'gru': GRUModel,
            'mopgru': MOPGRU,
            'hybrid': HybridGRUTransformer,
        }
        model_class = model_classes.get(self.model_name, MLP)
        
        config = checkpoint.get('config', {})
        self.model = model_class(
            self.input_dim, 
            self.num_classes,
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.3)
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded: {self.model_name}")
        print(f"Feature level: {self.feature_level}")
        print(f"Input dim: {self.input_dim}")
        print(f"Classes: {len(self.classes)}")
        print(f"Training accuracy: {self.accuracy*100:.2f}%")
        print(f"Device: {self.device}")
    
    def validate_input(self, x):
        """Validate and fix input dimensions."""
        x = np.array(x, dtype=np.float32)
        
        # Flatten if needed
        if x.ndim > 2:
            x = x.reshape(-1, self.input_dim)
        
        # Handle single sample
        if x.ndim == 1:
            if len(x) != self.input_dim:
                raise ValueError(f"Expected {self.input_dim} features, got {len(x)}")
            x = x.reshape(1, -1)
        else:
            # Batch processing
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
    
    def predict_from_csv(self, csv_path, feature_level=None):
        """Predict from CSV file containing landmarks."""
        import pandas as pd
        df = pd.read_csv(csv_path)
        level = feature_level or self.feature_level
        features = extract_features_from_landmark_df(df, level)
        return self.predict(features)
    
    def predict_from_npz(self, npz_path):
        """Predict from NumPy archive file."""
        data = np.load(npz_path)
        features = data['X']  # Assume X contains features
        return self.predict(features)


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
