#!/usr/bin/env python3
"""
ONNX Runtime inference for TSL-51 model.
This script runs inference using the exported ONNX model with feature vectors (NOT images).

Usage:
    python onnx_inference.py --model models/tsl51_gru_20260413_120426.onnx --features features.npz
    
The model expects:
- Input: 162-dimensional feature vector (NOT an image)
- Format: [left_hand_63 + right_hand_63 + pose_36]
"""

import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
import numpy as np

# TSL-51 class labels
CLASSES = [
    'กรุงเทพ', 'กิน', 'กลัว', 'ขอบคุณ', 'ข้าว', 'ขนมปัง',
    'คุณ', 'ฉัน', 'ชื่อ', 'ชอบ', 'ดี', 'ด้วยกัน',
    'ตลาด', 'ทำงาน', 'ทำไม', 'ที่ไหน', 'น้อง', 'น้ำ',
    'บ้าน', 'ประเทศ', 'ผู้', 'พ่อ', 'พี่', 'พรุ่งนี้',
    'ภาษามือ', 'มะม่วง', 'แม่', 'แมว', 'โรงเรียน',
    'วันนี้', 'วันหยุด', 'สวัสดี', 'สบายดี', 'หูหนวก',
    'หญิง', 'อะไร', 'อ่าน', 'อย่า', 'อยู่บ้าน',
    'เกิด', 'เรียน', 'เรียก', 'เช้า', 'เที่ยว',
    'เหงา', 'เหนื่อย', 'แต่งงาน', 'โกรธ', 'โสด', 'null_act'
]


def load_features(npz_path: str) -> np.ndarray:
    """Load features from npz file."""
    data = np.load(npz_path)
    # Handle both 'features' and 'X' key names
    if 'features' in data:
        return data['features']
    elif 'X' in data:
        return data['X']
    else:
        raise ValueError(f"Unknown key in {npz_path}. Expected 'features' or 'X'")


def run_inference(onnx_path: str, features: np.ndarray, top_k: int = 3):
    """Run ONNX inference."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not installed.")
        print("Install with: pip install onnxruntime")
        return None
    
    # Create inference session
    session = ort.InferenceSession(onnx_path)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Ensure features is 2D (batch, features)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    # Run inference
    output = session.run(None, {input_name: features.astype(np.float32)})[0]
    
    # Get top-k predictions
    top_indices = np.argsort(output[0])[-top_k:][::-1]
    
    # Apply softmax for confidence
    exp_output = np.exp(output[0] - np.max(output[0]))
    softmax_output = exp_output / exp_output.sum()
    
    results = []
    for idx in top_indices:
        results.append((CLASSES[idx], float(softmax_output[idx])))
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='ONNX inference for TSL-51',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with feature npz file
    python onnx_inference.py --model models/tsl51_gru_20260413_120426.onnx --features features.npz
    
    # Run with raw features
    python onnx_inference.py --model models/tsl51_gru_20260413_120426.onnx --features 0.1,0.2,0.3,...
    
NOTE: The ONNX model expects 162-dimensional FEATURE VECTORS, NOT images.
      - 63 features: left hand (21 points × 3)
      - 63 features: right hand (21 points × 3)  
      - 36 features: pose (12 points × 3)
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                    help='Path to ONNX model')
    parser.add_argument('--features', type=str, required=True,
                    help='Path to features npz file, or comma-separated features')
    parser.add_argument('--top-k', type=int, default=3,
                    help='Number of top predictions to return')
    
    args = parser.parse_args()
    
    # Load features
    if args.features.endswith('.npz'):
        features = load_features(args.features)
    else:
        # Parse comma-separated features
        features = np.array([float(x) for x in args.features.split(',')], dtype=np.float32)
    
    print(f"Features shape: {features.shape}")
    print("Expected shape: (162,)")
    
    if features.shape != (162,) and not (features.ndim == 2 and features.shape[1] == 162):
        print(f"ERROR: Expected 162 features, got {features.shape}")
        return 1
    
    # Run inference
    print(f"\nLoading ONNX model: {args.model}")
    results = run_inference(args.model, features, args.top_k)
    
    if results is None:
        return 1
    
    print("\n" + "=" * 40)
    print("PREDICTIONS")
    print("=" * 40)
    for i, (label, conf) in enumerate(results):
        print(f"{i+1}. {label}: {conf*100:.2f}%")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())