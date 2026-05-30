#!/usr/bin/env python3
"""
Export trained TSL-51 model to ONNX format for inference in web/edge environments.
"""

import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
import argparse
import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import model architecture from shared module
from src.train.models import GRUModel, MLP, MOPGRU, HybridGRUTransformer


def export_to_onnx(model_path: str, output_path: str = None, model_type: str = 'gru'):
    """Export trained model to ONNX format."""
    
    # Load checkpoint
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Get config from checkpoint
    if isinstance(checkpoint, dict):
        # Check if model is stored directly in checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Get config - try multiple sources
        config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
        input_dim = config.get('input_dim', checkpoint.get('input_dim', 162))
        num_classes = config.get('num_classes', checkpoint.get('num_classes', 51))
        hidden_dim = config.get('hidden_dim', config.get('hidden', 256))
        num_layers = config.get('num_layers', config.get('layers', 3))
        dropout = config.get('dropout', 0.3)
    else:
        # Legacy format
        state_dict = checkpoint
        input_dim = 162
        num_classes = 51
        hidden_dim = 256
        num_layers = 3
        dropout = 0.3
    
    # Create model
    checkpoint_model_type = checkpoint.get('model', config.get('model', model_type)) if isinstance(checkpoint, dict) else model_type
    model_type = (checkpoint_model_type or model_type).lower()
    model_classes = {
        'mlp': MLP,
        'gru': GRUModel,
        'mopgru': MOPGRU,
        'hybrid': HybridGRUTransformer,
    }
    model_class = model_classes.get(model_type, GRUModel)
    model = model_class(input_dim, num_classes, hidden_dim, num_layers, dropout)
    
    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create dummy input
    # Determine input shape from checkpoint metadata
    seq_mode = checkpoint.get('seq_mode', False) if isinstance(checkpoint, dict) else False
    target_frames = checkpoint.get('target_frames', 30) if isinstance(checkpoint, dict) else 30

    if model_type.lower() == 'mlp':
        # MLP always receives flat (batch, input_dim)
        dummy_input = torch.randn(1, input_dim)
    elif seq_mode:
        # Sequence-trained GRU: pass real (batch, T, input_dim) so ONNX graph is 3D
        dummy_input = torch.randn(1, target_frames, input_dim)
    else:
        # Mean-aggregated GRU: (batch, input_dim) — model does unsqueeze(1) internally
        dummy_input = torch.randn(1, input_dim)
    
    # Output path
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'models/tsl51_{model_type}_{timestamp}.onnx'
    
    # Export to ONNX
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'seq_len'} if seq_mode else {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify
    print("Verifying export...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # Get file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Export complete: {output_path} ({size_mb:.2f} MB)")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export TSL-51 model to ONNX')
    parser.add_argument('--model', type=str, default='models/tsl51_gru_best.pt',
                     help='Path to trained model')
    parser.add_argument('--output', type=str, default=None,
                     help='Output ONNX path')
    parser.add_argument('--type', type=str, default='gru', choices=['gru', 'mlp', 'mopgru', 'hybrid'],
                     help='Model type')
    
    args = parser.parse_args()
    
    export_to_onnx(args.model, args.output, args.type)


if __name__ == '__main__':
    main()
