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
import sys
from pathlib import Path

from ..train.compat import setup_mkl_threads, setup_windows_encoding

import torch
import numpy as np

from src.core.models import MLP, GRUModel, MOPGRU, HybridGRUTransformer

from ..data.extractor import (
    FEATURE_DIMS,
    adapt_features_to_model,
    build_enhanced_sequence,
    extract_features,
    extract_sequence_features,
    normalize_features,
    report_extractor_compatibility,
    resolve_feature_level_for_inference,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

# =============================================================================
# PREDICTION
# =============================================================================
def _detect_feature_level_from_dim(input_dim: int) -> str:
    """Auto-detect feature level from input dimension."""
    for level, dim in FEATURE_DIMS.items():
        if input_dim == dim:
            return level
    # If no exact match, return 'basic' as default
    return "basic"


def load_model(model_path):
    """Load model from single checkpoint file with feature dimension validation."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    report_extractor_compatibility(checkpoint)

    # Support both legacy and current checkpoint schemas.
    if 'labels' in checkpoint and 'label_to_idx' in checkpoint:
        labels = checkpoint['labels']
        idx_to_label = {idx: label for label, idx in checkpoint['label_to_idx'].items()}
    elif 'classes' in checkpoint:
        labels = [str(x) for x in checkpoint['classes']]
        idx_to_label = {idx: label for idx, label in enumerate(labels)}
    else:
        raise KeyError('Checkpoint missing labels/classes metadata')

    mean = checkpoint.get('normalization_mean', checkpoint.get('mean'))
    std = checkpoint.get('normalization_std', checkpoint.get('std'))
    if mean is None or std is None:
        raise KeyError('Checkpoint missing normalization stats (mean/std)')
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    input_dim = checkpoint.get('input_dim', int(mean.shape[0]))
    num_classes = checkpoint.get('num_classes', len(labels))
    config = checkpoint.get('config', {})

    # Auto-detect or use explicit feature_level
    feature_level = str(config.get('feature_level', _detect_feature_level_from_dim(input_dim)))
    expected_dim = FEATURE_DIMS.get(feature_level, input_dim)

    # Validate feature dimension consistency
    if input_dim != len(mean):
        print(f"WARNING: input_dim={input_dim} but mean/std have {len(mean)} dimensions")
        print(f"Using mean/std dimension: {len(mean)}")
        input_dim = len(mean)

    if input_dim != expected_dim:
        print(f"WARNING: Feature level '{feature_level}' expects {expected_dim} features but model has {input_dim}")
        print(f"Available feature levels: {FEATURE_DIMS}")
        # Auto-correct feature_level based on actual input_dim
        feature_level = _detect_feature_level_from_dim(input_dim)
        print(f"Auto-detected feature level: '{feature_level}'")

    model_name = config.get('model', checkpoint.get('model', 'mlp'))
    model_classes = {
        'mlp': MLP,
        'gru': GRUModel,
        'mopgru': MOPGRU,
        'hybrid': HybridGRUTransformer,
    }
    model_class = model_classes.get(model_name, MLP)

    model = model_class(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=config.get('hidden_dim', config.get('hidden', 256)),
        num_layers=config.get('num_layers', config.get('layers', 3)),
        dropout=config.get('dropout', 0.3)
    )
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict'))
    if state_dict is None:
        raise KeyError('Checkpoint missing model weights (model_state_dict/state_dict)')
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        if model_name == 'gru':
            fallback = MLP(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dim=config.get('hidden_dim', config.get('hidden', 256)),
                num_layers=config.get('num_layers', config.get('layers', 3)),
                dropout=config.get('dropout', 0.3)
            )
            fallback.load_state_dict(state_dict)
            model = fallback
        else:
            raise
    model.eval()

    seq_mode = bool(checkpoint.get('seq_mode', False))
    target_frames = int(checkpoint.get('target_frames', 30))

    print(f"Model configuration: feature_level='{feature_level}', input_dim={input_dim}")

    return model, labels, idx_to_label, mean, std, seq_mode, target_frames, feature_level


def predict(model, features, mean, std, idx_to_label, top_k=3):
    """Predict top-k Thai words."""
    normalized = normalize_features(features, mean, std)
    if normalized is None:
        return []
    if np.asarray(normalized).ndim == 2:
        tensor = torch.tensor(normalized[None, ...], dtype=torch.float32)
    else:
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
    setup_windows_encoding()
    setup_mkl_threads()

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
    parser.add_argument("--feature-level", type=str, default=None,
                        choices=['basic', 'finger', 'enhanced', 'full'],
                        help="Override feature level for inference (default: auto-detect from model)")
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.model}")
    try:
        model, labels, idx_to_label, mean, std, seq_mode, target_frames, feature_level = load_model(args.model)
        resolved_level, warn_msg = resolve_feature_level_for_inference(
            args.feature_level,
            model_input_dim=len(mean),
            checkpoint_feature_level=feature_level,
        )
        if warn_msg:
            print(f"[WARN] {warn_msg}")
        if args.feature_level is not None and args.feature_level != resolved_level:
            print(f"[INFO] Auto-adjusted feature level to '{resolved_level}' for model compatibility")
        feature_level = resolved_level
        print(f"Using feature level: {feature_level}")
        print(f"Loaded {len(labels)} Thai word classes")
        print(f"Sequence mode: {seq_mode} (target_frames={target_frames})")
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
            # Log and continue; print concise message
            try:
                print(f"Error loading {input_path}: {e}")
            except Exception:
                pass
            continue
        
        # Extract features and predict
        if seq_mode:
            if feature_level == 'enhanced':
                features = build_enhanced_sequence(
                    frames,
                    feature_level='enhanced',
                    target_frames=target_frames,
                )
            else:
                features = extract_sequence_features(frames, feature_level=feature_level, target_frames=target_frames)
        else:
            features = extract_features(frames, feature_level=feature_level)
        if features is None:
            print(f"Error: Could not extract features from {input_path.name}")
            continue

        # Adapt features to model's expected dimension (handles mismatches gracefully)
        features = adapt_features_to_model(features, len(mean), feature_level)
        if features is None:
            print(f"Error: Could not adapt features for {input_path.name}")
            continue

        feature_dim = features.shape[-1] if isinstance(features, np.ndarray) and features.ndim >= 2 else len(features)

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
