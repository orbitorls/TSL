"""
TSL-51 Inference Script
=======================
Use this script to run inference with trained model.

FEATURE EXTRACTION:
==================
The model expects 162 features in this order:
1. Left Hand: 21 points × 3 coordinates (x, y, z) = 63 features
   - Order: lh_x0, lh_y0, lh_z0, lh_x1, lh_y1, lh_z1, ... lh_x20, lh_y20, lh_z20
   
2. Right Hand: 21 points × 3 coordinates = 63 features
   - Order: rh_x0, rh_y0, rh_z0, rh_x1, rh_y1, rh_z1, ... rh_x20, rh_y20, rh_z20

3. Pose Landmarks: 12 points × 3 coordinates = 36 features
   - Order: l_shoulder_x/y/z, r_shoulder_x/y/z, l_elbow_x/y/z, r_elbow_x/y/z,
            l_wrist_x/y/z, r_wrist_x/y/z, lbrow_outer_x/y/z, lbrow_inner_x/y/z,
            rbrow_inner_x/y/z, rbrow_outer_x/y/z, mouth_right_x/y/z, mouth_left_x/y/z

INPUT FORMAT:
============
- numpy array shape: (162,) or (batch_size, 162)
- Values: normalized coordinates (typical range -1 to 1 or 0 to 1)

USAGE:
======
# Load model and predict
python inference.py --model models/tsl51_gru_best.pt --input your_data.npz

# Or use as Python module
from inference import TSLPredictor
predictor = TSLPredictor('models/tsl51_gru_best.pt')
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
def extract_features_from_landmark_df(lm_df):
    """
    Extract 162 features from landmark DataFrame.
    This MUST match the training feature extraction exactly!
    """
    features = []
    
    # Left hand (21 points * 3 = 63)
    for i in range(21):
        for c in ['x', 'y', 'z']:
            col = f'lh_{c}{i}'
            if col in lm_df.columns:
                features.append(lm_df[col].mean())
            else:
                features.append(0.0)
    
    # Right hand (21 points * 3 = 63)
    for i in range(21):
        for c in ['x', 'y', 'z']:
            col = f'rh_{c}{i}'
            if col in lm_df.columns:
                features.append(lm_df[col].mean())
            else:
                features.append(0.0)
    
    # Pose landmarks (12 points * 3 = 36)
    pose_cols = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
                'lbrow_outer', 'lbrow_inner', 'rbrow_inner', 'rbrow_outer',
                'mouth_right', 'mouth_left']
    for base in pose_cols:
        for c in ['x', 'y', 'z']:
            col = f'{base}_{c}'
            if col in lm_df.columns:
                features.append(lm_df[col].mean())
            else:
                features.append(0.0)
    
    return np.array(features, dtype=np.float32)


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
        predictor = TSLPredictor('models/tsl51_gru_best.pt')
        label, confidence = predictor.predict(landmarks)
    """
    
    EXPECTED_FEATURES = 162  # Must match training
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Load metadata
        self.classes = checkpoint['classes']
        self.mean = np.array(checkpoint['mean'])
        self.std = np.array(checkpoint['std'])
        self.input_dim = checkpoint['input_dim']
        self.num_classes = checkpoint['num_classes']
        self.model_name = checkpoint.get('model', 'gru')
        self.accuracy = checkpoint.get('accuracy', 0.0)
        
        # Validate dimensions
        if self.input_dim != self.EXPECTED_FEATURES:
            print(f"WARNING: Model expects {self.input_dim} features but expected {self.EXPECTED_FEATURES}")
            print("This may cause prediction errors!")
        
        # Create model
        if self.model_name == 'gru':
            self.model = GRUModel(self.input_dim, self.num_classes)
        else:
            self.model = MLP(self.input_dim, self.num_classes)
        
        # Load weights
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded: {self.model_name}")
        print(f"Input dim: {self.input_dim}")
        print(f"Classes: {len(self.classes)}")
        print(f"Training accuracy: {self.accuracy*100:.2f}%")
        print(f"Device: {self.device}")
    
    def validate_input(self, x):
        """Validate and fix input dimensions."""
        x = np.array(x, dtype=np.float32)
        
        # Flatten if needed
        if x.ndim > 2:
            x = x.reshape(-1, self.EXPECTED_FEATURES)
        
        # Handle single sample
        if x.ndim == 1:
            if len(x) != self.EXPECTED_FEATURES:
                raise ValueError(f"Expected {self.EXPECTED_FEATURES} features, got {len(x)}")
            x = x.reshape(1, -1)
        else:
            # Batch processing
            if x.shape[1] != self.EXPECTED_FEATURES:
                # Try to auto-fix: maybe user passed different format
                if x.shape[0] == self.EXPECTED_FEATURES:
                    x = x.reshape(1, -1)
                else:
                    raise ValueError(f"Expected features dimension {self.EXPECTED_FEATURES}, got {x.shape[1]}")
        
        return x
    
    def predict(self, landmarks, return_top_k=1):
        """
        Predict sign from landmarks.
        
        Args:
            landmarks: numpy array of shape (162,) or (batch, 162)
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
    
    def predict_from_csv(self, csv_path):
        """Predict from CSV file containing landmarks."""
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        features = extract_features_from_landmark_df(df)
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
            # Assume first array
            landmarks = data[data.files[0]]
        preds = predictor.predict(landmarks, return_top_k=args.top_k)
    elif input_path.suffix == '.csv':
        import pandas as pd
        df = pd.read_csv(input_path)
        landmarks = extract_features_from_landmark_df(df)
        preds = predictor.predict(landmarks, return_top_k=args.top_k)
    else:
        print(f"ERROR: Unsupported file format. Use .npz or .csv")
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
    print(f"Expected features: {predictor.EXPECTED_FEATURES}")
    print(f"Model accuracy: {predictor.accuracy*100:.2f}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
