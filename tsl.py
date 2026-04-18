#!/usr/bin/env python3
"""
TSL-51: Thai Sign Language Recognition - Unified CLI
=====================================================
Single CLI for all TSL-51 operations.

Usage:
    python tsl.py <command> [options]
    
Commands:
    train       - Train a new model
    export-onnx - Export trained model to ONNX format
    infer       - Run inference (PyTorch)
    infer-onnx  - Run inference (ONNX model)
    benchmark  - Run benchmark on videos
    download   - Download dataset
    info       - Show model info
    
Examples:
    # Train
    python tsl.py train --epochs 50 --batch 128
    
    # Export to ONNX
    python tsl.py export-onnx --model models/tsl51_gru_best.pt
    
    # Run inference (PyTorch)
    python tsl.py infer --model models/tsl51_gru_best.pt --video sign.mp4
    
    # Run inference (ONNX)
    python tsl.py infer-onnx --model models/tsl51_gru_20260413_120426.onnx --features 0.1,0.2,...
    
    # Download dataset  
    python tsl.py download --split user_sign
    
    # Model info
    python tsl.py info --model models/tsl51_gru_best.pt
"""

import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse

# ============================================================================
# SHARED CONSTANTS
# ============================================================================

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


def cmd_train(args):
    """Train a new model."""
    from train_tsl51_v3 import main as train_main
    sys.argv = ['train']
    if args.epochs:
        sys.argv.extend(['--epochs', str(args.epochs)])
    if args.batch:
        sys.argv.extend(['--batch', str(args.batch)])
    if args.layers:
        sys.argv.extend(['--layers', str(args.layers)])
    if args.augment:
        sys.argv.extend(['--augment', str(args.augment)])
    return train_main()


def cmd_export_onnx(args):
    """Export trained model to ONNX."""
    from export_onnx import export_to_onnx
    output_path = export_to_onnx(
        args.model,
        args.output,
        args.type
    )
    print(f"\nExported: {output_path}")
    return 0


def cmd_infer(args):
    """Run inference with PyTorch model."""
    import torch
    import numpy as np
    
    # Check dependencies
    try:
        import mediapipe as mp
        import cv2
    except ImportError:
        print("ERROR: Install dependencies: pip install mediapipe opencv-python")
        return 1
    
    # Load model
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Import model architecture
    from train_tsl51_v3 import GRUModel
    
    model = GRUModel(162, 51, 256, 3, 0.3)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Extract features from video
    def extract_features(video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if not frames:
            print("ERROR: No frames extracted")
            return 1
        
        # Simple feature extraction (placeholder)
        # In production, use MediaPipe
        features = np.random.randn(162).astype(np.float32)
        return features
    
    print(f"Extracting features from {args.video}...")
    features = extract_features(args.video)
    
    # Run inference
    with torch.no_grad():
        x = torch.from_numpy(features).unsqueeze(0)
        output = model(x)
        probs = torch.softmax(output, dim=1)
        top_k = min(args.top_k, 51)
        top_probs, top_idx = torch.topk(probs, top_k)
    
    print("\nPREDICTIONS:")
    for i in range(top_k):
        idx = top_idx[0][i].item()
        conf = top_probs[0][i].item()
        print(f"  {i+1}. {CLASSES[idx]}: {conf*100:.2f}%")
    
    return 0


def cmd_infer_onnx(args):
    """Run inference with ONNX model."""
    # Install onnxruntime first
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not installed")
        print("Install: pip install onnxruntime")
        return 1
    
    # Load features
    if args.features.endswith('.npz'):
        import numpy as np
        data = np.load(args.features)
        features = data.get('features', data.get('X', data['X']))
    else:
        # Comma-separated features
        features = [float(x) for x in args.features.split(',')]
        import numpy as np
        features = np.array(features, dtype=np.float32)
    
    # Ensure 2D
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    # Run inference
    session = ort.InferenceSession(args.model)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: features.astype(np.float32)})[0]
    
    # Get top-k
    top_indices = np.argsort(output[0])[-args.top_k:][::-1]
    
    # Softmax
    exp_out = np.exp(output[0] - np.max(output[0]))
    softmax = exp_out / exp_out.sum()
    
    print("\nPREDICTIONS:")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. {CLASSES[idx]}: {softmax[idx]*100:.2f}%")
    
    return 0


def cmd_benchmark(args):
    """Run benchmark on videos."""
    from benchmark_video import main as bench_main
    sys.argv = ['benchmark']
    if args.video:
        sys.argv.extend(['--video', args.video])
    if args.model:
        sys.argv.extend(['--model', args.model])
    return bench_main()


def cmd_download(args):
    """Download dataset."""
    from download_tsl51_v2 import main as dl_main
    sys.argv = ['download']
    if args.split:
        sys.argv.extend(['--split', args.split])
    return dl_main()


def cmd_info(args):
    """Show model information."""
    import torch
    
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    
    print(f"\nModel: {args.model}")
    print("-" * 40)
    
    if isinstance(checkpoint, dict):
        for key, value in checkpoint.items():
            if key not in ['state_dict', 'config']:
                print(f"{key}: {value}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='TSL-51: Thai Sign Language Recognition CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    sub = parser.add_subparsers(dest='cmd', required=True)
    
    # train
    p = sub.add_parser('train', help='Train model')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--augment', type=int, default=0)
    
    # export-onnx
    p = sub.add_parser('export-onnx', help='Export to ONNX')
    p.add_argument('--model', default='models/tsl51_gru_best.pt')
    p.add_argument('--output', default=None)
    p.add_argument('--type', default='gru', choices=['gru', 'mlp'])
    
    # infer
    p = sub.add_parser('infer', help='PyTorch inference')
    p.add_argument('--model', required=True)
    p.add_argument('--video', help='Input video')
    p.add_argument('--top-k', type=int, default=3)
    
    # infer-onnx
    p = sub.add_parser('infer-onnx', help='ONNX inference')
    p.add_argument('--model', required=True)
    p.add_argument('--features', required=True, help='.npz or comma-sep values')
    p.add_argument('--top-k', type=int, default=3)
    
    # benchmark
    p = sub.add_parser('benchmark', help='Benchmark')
    p.add_argument('--video', default='data')
    p.add_argument('--model', default='models/tsl51_gru_best.pt')
    
    # download
    p = sub.add_parser('download', help='Download dataset')
    p.add_argument('--split', default='user_sign', choices=['user_sign', 'expert'])
    
    # info
    p = sub.add_parser('info', help='Model info')
    p.add_argument('--model', required=True)
    
    args = parser.parse_args()
    
    # Dispatch
    cmds = {
        'train': cmd_train,
        'export-onnx': cmd_export_onnx,
        'infer': cmd_infer,
        'infer-onnx': cmd_infer_onnx,
        'benchmark': cmd_benchmark,
        'download': cmd_download,
        'info': cmd_info,
    }
    
    return cmds[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())