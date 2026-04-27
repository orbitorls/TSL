"""
TSL-51 Model Export Script
===========================
Export trained model to a portable format that can be used anywhere.

Usage:
    python tools/export_model.py --input models/tsl51_hybrid_*.pt --output tsl51_model.pt
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch

PROJECT_DIR = Path(__file__).resolve().parent


def export_model(input_path, output_path=None):
    """Export model to portable format."""
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"ERROR: Model not found: {input_path}")
        return 1
    
    # Load checkpoint
    print(f"Loading model: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    
    # Create portable package
    package = {
        # Model weights
        'state_dict': checkpoint['state_dict'],
        
        # Model architecture info
        'model_type': checkpoint.get('model', 'gru'),
        'model_config': {
            'hidden_dim': checkpoint.get('config', {}).get('hidden_dim', checkpoint.get('config', {}).get('hidden', 256)),
            'num_layers': checkpoint.get('config', {}).get('num_layers', checkpoint.get('config', {}).get('layers', 3)),
            'dropout': checkpoint.get('config', {}).get('dropout', 0.3),
            'nhead': 8,
        },
        
        # Input/Output specs
        'input_dim': checkpoint['input_dim'],
        'num_classes': checkpoint['num_classes'],
        'classes': checkpoint['classes'],
        
        # Normalization (REQUIRED for inference)
        'mean': checkpoint['mean'],
        'std': checkpoint['std'],

        # Temporal sequence mode (new in v2)
        'seq_mode': checkpoint.get('seq_mode', False),
        'target_frames': checkpoint.get('target_frames', 30),

        # Feature extraction info
        'feature_info': {
            'feature_level': checkpoint.get('config', {}).get('feature_level', 'basic'),
            'feature_count': checkpoint['input_dim'],
            'feature_breakdown': _get_feature_breakdown(checkpoint['input_dim']),
        },
        
        # Training info (optional)
        'training_info': {
            'accuracy': checkpoint.get('accuracy', 0),
            'dataset': checkpoint.get('config', {}).get('dataset', 'unknown'),
            'epochs': checkpoint.get('config', {}).get('epochs', 0),
            'test_split': checkpoint.get('config', {}).get('test_split', 0),
        },
        
        # Version
        'version': '1.0',
        'exported_at': datetime.now().isoformat(),
    }
    
    # Determine output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_export.pt"
    else:
        output_path = Path(output_path)
    
    # Save
    print(f"Exporting to: {output_path}")
    torch.save(package, output_path)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPORT SUMMARY")
    print("="*60)
    print(f"Model type: {package['model_type']}")
    print(f"Input features: {package['input_dim']}")
    print(f"Classes: {package['num_classes']}")
    print(f"Feature level: {package['feature_info']['feature_level']}")
    print(f"Feature breakdown: {package['feature_info']['feature_breakdown']}")
    print(f"Training accuracy: {package['training_info']['accuracy']*100:.2f}%")
    print("="*60)
    print(f"\nSaved to: {output_path}")
    
    # Save feature extraction code
    _save_feature_extractor(output_path.parent / f"{output_path.stem}_feature_extractor.py", package)
    
    return 0


def _get_feature_breakdown(input_dim):
    """Get feature breakdown based on dimension."""
    if input_dim == 162:
        return "hand (126) + pose (36)"
    elif input_dim == 258:
        return "hand (126) + pose (36) + finger details (96)"
    elif input_dim == 1596:
        return "hand (126) + pose (36) + finger (96) + face (1338)"
    elif input_dim == 1434:
        return "face only (1434)"
    else:
        return f"unknown ({input_dim})"


def _save_feature_extractor(output_path, package):
    """Save standalone feature extractor script."""
    feature_level = package['feature_info']['feature_level']
    input_dim = package['input_dim']
    
    extractor_code = f'''"""
TSL-51 Feature Extractor
=========================
Standalone feature extraction for TSL-51 model (feature_level={feature_level})

REQUIREMENTS:
    pip install mediapipe opencv-python numpy pandas

USAGE:
    from tsl51_feature_extractor import extract_features
    features = extract_features(video_path="video.mp4")
    # features shape: ({input_dim},)
"""

import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path


# Feature configuration
FEATURE_LEVEL = "{feature_level}"
INPUT_DIM = {input_dim}


def extract_features_from_frame(landmarks, feature_level=FEATURE_LEVEL):
    """Extract features from MediaPipe landmarks.
    
    Args:
        landmarks: MediaPipe landmark result
        feature_level: 'basic', 'finger', 'full', 'face'
    
    Returns:
        numpy array of shape (INPUT_DIM,)
    """
    features = []
    
    # ===== 1. Hand landmarks =====
    # Left hand (21 points * 3 = 63)
    if landmarks.left_hand_landmarks:
        for i in range(21):
            lm = landmarks.left_hand_landmarks.landmark[i]
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 63)
    
    # Right hand (21 points * 3 = 63)
    if landmarks.right_hand_landmarks:
        for i in range(21):
            lm = landmarks.right_hand_landmarks.landmark[i]
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 63)
    
    # ===== 2. Pose landmarks (12 points * 3 = 36) =====
    if landmarks.pose_landmarks:
        pose_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        for i in pose_indices:
            lm = landmarks.pose_landmarks.landmark[i]
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 36)
    
    # ===== 3. Finger details (if enabled) =====
    if feature_level in ['finger', 'full']:
        # Additional finger joints would need model_complexity=1 in MediaPipe
        features.extend([0.0] * 96)  # Placeholder
    
    # ===== 4. Face landmarks (if enabled) =====
    if feature_level in ['face', 'full']:
        if landmarks.face_landmarks:
            for i in range(478):
                lm = landmarks.face_landmarks.landmark[i]
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * 1434)
    
    return np.array(features[:INPUT_DIM], dtype=np.float32)


def extract_features(video_path, model_complexity=1):
    """Extract features from video file.
    
    Args:
        video_path: Path to video file
        model_complexity: MediaPipe model complexity (0, 1, 2)
    
    Returns:
        numpy array of shape (INPUT_DIM,) - averaged across frames
    """
    import pandas as pd
    
    # Initialize MediaPipe
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        refine_face_landmarks=(FEATURE_LEVEL in ['face', 'full'])
    )
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    all_features = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = holistic.process(frame_rgb)
        
        # Extract features
        features = extract_features_from_frame(results)
        all_features.append(features)
    
    cap.release()
    holistic.close()
    
    if not all_features:
        raise ValueError("No landmarks detected in video")
    
    # Average across frames: stack to ensure correct shape, fallback defensively
    try:
        arr = np.stack(all_features, axis=0)
        return np.mean(arr, axis=0)
    except Exception:
        arr = np.asarray(all_features)
        if arr.ndim == 1:
            return np.array(arr, dtype=np.float32)
        return np.mean(arr, axis=0)


def normalize(features, mean, std):
    """Normalize features using training statistics."""
    return (features - np.array(mean)) / np.array(std)


# Standalone usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <video_path> [model.pt]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    print(f"Extracting features from: {video_path}")
    features = extract_features(video_path)
    print(f"Features shape: {features.shape}")
    print(f"Feature level: {FEATURE_LEVEL}")
'''
    
    output_path.write_text(extractor_code, encoding='utf-8')
    print(f"\nAlso saved feature extractor: {output_path.parent / f'{output_path.stem}_feature_extractor.py'}")


def main():
    parser = argparse.ArgumentParser(description='Export TSL-51 model to portable format')
    parser.add_argument("--input", type=str, required=True, help="Input model file")
    parser.add_argument("--output", type=str, default=None, help="Output file (default: <input>_export.pt)")
    args = parser.parse_args()
    
    return export_model(args.input, args.output)


if __name__ == "__main__":
    sys.exit(main())
