"""
Thai Sign Language Translation - Single Model File

Usage:
    python translate.py --input sample.json
    python translate.py --input file1.json file2.json
    python translate.py --input data/tsl51_full_processed/test/*.json

Output:
    Predicted word in Thai (e.g., "กรุงเทพ", "กิน", "รัก")
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Fix Windows console encoding for Thai characters
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn

from tsl_tasks_extractor import (
    extract_features as extract_sequence_features,
    normalize_features,
    report_extractor_compatibility,
)

PROJECT_DIR = Path(__file__).resolve().parent

# =============================================================================
# MODEL
# =============================================================================
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


# =============================================================================
# PREDICTION
# =============================================================================
def load_model(model_path):
    """Load model from single checkpoint file."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    report_extractor_compatibility(checkpoint)
    
    # Extract model info
    labels = checkpoint['labels']
    label_to_idx = checkpoint['label_to_idx']
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    mean = checkpoint['normalization_mean']
    std = checkpoint['normalization_std']
    input_dim = checkpoint['input_dim']
    num_classes = checkpoint['num_classes']
    config = checkpoint.get('config', {})
    
    # Create and load model
    model = MLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 3),
        dropout=config.get('dropout', 0.3)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, labels, idx_to_label, mean, std


def predict(model, features, mean, std, idx_to_label, top_k=3):
    """Predict top-k Thai words."""
    normalized = normalize_features(features, mean, std)
    if normalized is None:
        return []
    tensor = torch.tensor([normalized], dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_indices = probs.argsort(descending=True)[:top_k]
        predictions = [(idx_to_label[idx.item()], probs[idx].item()) for idx in top_indices]
    
    return predictions


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Thai Sign Language Translation - Output Thai words",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
    # Single file
    python translate.py --input sample.json
    
    # Multiple files
    python translate.py --input file1.json file2.json
    
    # PowerShell: all test files
    python translate.py --input (Get-ChildItem data\tsl51_full_processed\test\*.json).FullName
"""
    )
    parser.add_argument("--model", type=Path, default=Path("models/tsl_model.pt"), help="Model file (default: models/tsl_model.pt)")
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="Input JSON file(s)")
    parser.add_argument("--top-k", type=int, default=3, help="Show top-k predictions")
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.model}")
    try:
        model, labels, idx_to_label, mean, std = load_model(args.model)
        print(f"Loaded {len(labels)} Thai word classes")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train_cv.py first to train the model.")
        return 1
    
    # Process inputs
    correct_count = 0
    total_count = 0
    predicted_words = []
    
    print(f"\n{'='*60}")
    print("THAI SIGN LANGUAGE TRANSLATION")
    print(f"{'='*60}")
    
    for input_path in args.input:
        if not input_path.exists():
            print(f"Skipping: {input_path} (not found)")
            continue
        
        # Load JSON
        try:
            data = json.loads(input_path.read_text(encoding='utf-8'))
            frames = data["frames"]
        except Exception as e:
            print(f"Error loading {input_path}: {e}")
            continue
        
        # Extract features and predict
        features = extract_sequence_features(frames)
        if features is None:
            print(f"Error: Could not extract features from {input_path.name}")
            continue
        if len(features) != len(mean):
            print(
                f"Error: Feature dimension mismatch for {input_path.name}: "
                f"expected {len(mean)}, got {len(features)}"
            )
            continue

        predictions = predict(model, features, mean, std, idx_to_label, args.top_k)
        if not predictions:
            print(f"Error: Could not normalize or predict for {input_path.name}")
            continue
        
        # Get top prediction (Thai word)
        top_word, confidence = predictions[0]
        
        # Display result
        true_label = data.get("label", None)
        
        print(f"\n{input_path.name}")
        print(f"  Result: {top_word} ({confidence:.1%})")
        
        if true_label:
            is_correct = top_word == true_label
            status = "OK" if is_correct else "X"
            print(f"  Actual:  {true_label} [{status}]")
            if is_correct:
                correct_count += 1
            total_count += 1
        
        if args.top_k > 1:
            print(f"  Top {args.top_k}:")
            for word, prob in predictions:
                print(f"    {word}: {prob:.1%}")
        
        predicted_words.append(top_word)
    
    # Summary
    print(f"\n{'='*60}")
    if len(predicted_words) > 1:
        print(f"Phrase: {' '.join(predicted_words)}")
    
    if total_count > 0:
        print(f"Accuracy: {correct_count}/{total_count} ({correct_count/total_count:.1%})")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
