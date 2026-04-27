"""
TSL-51 Standalone Inference
===========================
Simple inference script that works standalone without dependencies on training code.

Usage:
    python inference_standalone.py --model exported_model.pt --video video.mp4
    
    # Or from Python
    from inference_standalone import predict
    result = predict("video.mp4", "model.pt")
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Try to import optional dependencies
try:
    import cv2
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

PROJECT_DIR = Path(__file__).resolve().parent


# ============================================================================
# MODEL DEFINITIONS
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


class HybridGRUTransformer(torch.nn.Module):
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


MODEL_CLASSES = {
    'mlp': MLP,
    'gru': GRUModel,
    'hybrid': HybridGRUTransformer,
}


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
def extract_features_from_video(video_path, feature_level='full', min_frames=30):
    """Extract features from video using MediaPipe."""
    if not HAS_MEDIAPIPE:
        raise ImportError("Please install mediapipe and opencv-python: pip install mediapipe opencv-python")
    
    import mediapipe as mp
    import cv2
    
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=(feature_level in ['face', 'full'])
    )
    
    cap = cv2.VideoCapture(str(video_path))
    all_features = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        
        features = _extract_frame_features(results, feature_level)
        all_features.append(features)
    
    cap.release()
    holistic.close()
    
    if len(all_features) < min_frames:
        raise ValueError(f"Too few frames with landmarks: {len(all_features)} < {min_frames}")
    
    return np.mean(all_features, axis=0)


def _extract_frame_features(results, feature_level):
    """Extract features from a single frame."""
    features = []
    
    # Hand landmarks
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 63)
    
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 63)
    
    # Pose landmarks
    if results.pose_landmarks:
        pose_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        for i in pose_indices:
            lm = results.pose_landmarks.landmark[i]
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 36)
    
    # Face landmarks
    if feature_level in ['face', 'full'] and results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    elif feature_level in ['face', 'full']:
        features.extend([0.0] * 1434)
    
    return np.array(features, dtype=np.float32)


# ============================================================================
# PREDICTOR
# ============================================================================
class TSLPredictor:
    """Simple TSL predictor."""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        self.classes = checkpoint['classes']
        self.mean = np.array(checkpoint['mean'])
        self.std = np.array(checkpoint['std'])
        self.input_dim = checkpoint['input_dim']
        self.num_classes = checkpoint['num_classes']
        
        # Get model type and config
        config = checkpoint.get('model_config', {})
        model_type = checkpoint.get('model_type', 'gru')
        
        # Create model
        model_class = MODEL_CLASSES.get(model_type, MLP)
        self.model = model_class(
            self.input_dim,
            self.num_classes,
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.3)
        )
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
        print(f"Loaded: {model_type}, {self.input_dim} features, {self.num_classes} classes")
    
    def predict(self, features, return_top_k=1):
        """Predict from features."""
        # Normalize
        x = (np.array(features) - self.mean) / self.std
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
        
        if return_top_k == 1:
            prob, pred = probs.max(0)
            return self.classes[pred.item()], prob.item()
        else:
            top_probs, top_indices = probs.topk(return_top_k)
            return [(self.classes[idx.item()], prob.item()) 
                    for idx, prob in zip(top_indices, top_probs)]


def predict(video_path, model_path, return_top_k=3):
    """Quick predict function."""
    predictor = TSLPredictor(model_path)
    
    # Get feature level from model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    feature_level = checkpoint.get('feature_info', {}).get('feature_level', 'full')
    
    print(f"Extracting features ({feature_level})...")
    features = extract_features_from_video(video_path, feature_level)
    print(f"Features shape: {features.shape}")
    
    return predictor.predict(features, return_top_k)


# ============================================================================
# MAIN
# ============================================================================
def main():
    if not HAS_MEDIAPIPE:
        print("ERROR: Please install dependencies:")
        print("  pip install mediapipe opencv-python torch numpy")
        return 1
    
    parser = argparse.ArgumentParser(description='TSL-51 Standalone Inference')
    parser.add_argument("--video", type=str, required=True, help="Input video file")
    parser.add_argument("--model", type=str, required=True, help="Model file")
    parser.add_argument("--top-k", type=int, default=3, help="Number of predictions")
    args = parser.parse_args()
    
    print("="*60)
    print("TSL-51 STANDALONE INFERENCE")
    print("="*60)
    
    results = predict(args.video, args.model, args.top_k)
    
    print("\n" + "="*60)
    print("PREDICTIONS")
    print("="*60)
    for i, (label, conf) in enumerate(results):
        print(f"{i+1}. {label}: {conf*100:.2f}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())