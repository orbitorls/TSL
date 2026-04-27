"""Model benchmarking script for TSL-51.

This script compares different model architectures on the TSL-51 dataset
to determine which performs best.
"""

import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from src.data.loader import load_tsl51_user_sign, validate_dataset, print_dataset_quality_report
from src.train.models import MODEL_CLASSES
from src.train.config import TrainingConfig


def benchmark_model(model_name: str, X: np.ndarray, y: np.ndarray, classes: np.ndarray, config: TrainingConfig) -> dict:
    """Benchmark a single model.
    
    Args:
        model_name: Name of the model to benchmark
        X: Feature array
        y: Label array
        classes: Class names
        config: Training configuration
        
    Returns:
        Dictionary containing benchmark results
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARKING: {model_name.upper()}")
    print(f"{'='*70}")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = X.shape[1]
    num_classes = len(classes)
    
    # Get model class
    model_class = MODEL_CLASSES.get(model_name, MODEL_CLASSES['gru'])
    
    # Create model
    model = model_class(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Measure inference time
    model.eval()
    X_tensor = torch.FloatTensor(X[:100]).to(device)  # Use first 100 samples
    
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(X_tensor)
        
        # Measure
        import time
        start = time.time()
        for _ in range(100):
            _ = model(X_tensor)
        end = time.time()
    
    avg_inference_time = (end - start) / 100
    inferences_per_second = 1.0 / avg_inference_time
    
    results = {
        'model_name': model_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'avg_inference_time_ms': avg_inference_time * 1000,
        'inferences_per_second': inferences_per_second,
        'device': device,
    }
    
    print(f"Parameters: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")
    print(f"Inference Time: {avg_inference_time*1000:.2f}ms")
    print(f"Throughput: {inferences_per_second:.1f} inferences/sec")
    print(f"Device: {device}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark TSL-51 model architectures')
    parser.add_argument('--dataset', type=str, default='tsl51_user_sign',
                       choices=['tsl51_user_sign', 'tsl51_expert', 'tsl51_expert_full'],
                       help='Dataset to use')
    parser.add_argument('--samples', type=int, default=None,
                       help='Number of samples (default: all)')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['gru', 'mlp', 'mopgru', 'hybrid'],
                       help='Models to benchmark')
    parser.add_argument('--hidden', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--layers', type=int, default=3,
                       help='Number of layers')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    # Load dataset
    if args.dataset == 'tsl51_user_sign':
        X, y, classes = load_tsl51_user_sign(max_samples=args.samples)
    elif args.dataset == 'tsl51_expert':
        from src.data.loader import load_tsl51_expert
        X, y, classes = load_tsl51_expert(include_augmented=False, max_samples=args.samples)
    elif args.dataset == 'tsl51_expert_full':
        from src.data.loader import load_tsl51_expert_full
        X, y, classes = load_tsl51_expert_full(max_samples=args.samples)
    
    if X is None:
        print("ERROR: Failed to load dataset")
        return 1
    
    # Validate dataset
    validation_results = validate_dataset(X, y, classes)
    print_dataset_quality_report(validation_results)
    
    # Configuration
    config = TrainingConfig(
        hidden_dim=args.hidden,
        num_layers=args.layers,
    )
    
    # Benchmark each model
    all_results = []
    
    for model_name in args.models:
        try:
            results = benchmark_model(model_name, X, y, classes, config)
            all_results.append(results)
        except Exception as e:
            print(f"ERROR benchmarking {model_name}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(classes)}")
    print(f"\n{'Model':<15} {'Params':<12} {'Time (ms)':<12} {'Inf/sec':<12}")
    print("-" * 70)
    
    for r in all_results:
        print(f"{r['model_name']:<15} {r['total_params']:<12,} {r['avg_inference_time_ms']:<12.2f} {r['inferences_per_second']:<12.1f}")
    
    print(f"{'='*70}\n")
    
    # Save results
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'dataset': args.dataset,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': len(classes),
            'config': {
                'hidden_dim': args.hidden,
                'num_layers': args.layers,
            },
            'results': all_results,
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to: {output_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
