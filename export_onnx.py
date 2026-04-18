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

# Import model architecture
from train_tsl51_v3 import GRUModel, MLP


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
        hidden_dim = config.get('hidden_dim', 256)
        num_layers = config.get('num_layers', 3)
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
    if model_type.lower() == 'mlp':
        model = MLP(input_dim, num_classes, hidden_dim, num_layers, dropout)
    else:
        model = GRUModel(input_dim, num_classes, hidden_dim, num_layers, dropout)
    
    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create dummy input
    # Note: GRUModel.forward() in train_tsl51_v3.py calls x = x.unsqueeze(1)
    # which means the model expects input shaped (batch, input_dim) and
    # performs an internal unsqueeze to (batch, 1, input_dim). Passing a
    # (batch, 1, input_dim) here would produce a 4D tensor after unsqueeze
    # (which breaks torch.onnx.export). Use (batch, input_dim) for GRU.
    if model_type.lower() == 'mlp':
        dummy_input = torch.randn(1, input_dim)
    else:
        # GRUModel expects (batch, input_dim) and will unsqueeze internally
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
            'input': {0: 'batch_size'},
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
    parser.add_argument('--type', type=str, default='gru', choices=['gru', 'mlp'],
                     help='Model type')
    
    args = parser.parse_args()
    
    export_to_onnx(args.model, args.output, args.type)


if __name__ == '__main__':
    main()
