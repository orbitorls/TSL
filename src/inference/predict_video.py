"""
Thai Sign Language Video Prediction Script
Predict Thai words from video files (pre-recorded clips)

Usage:
    python predict_video.py --input video.mp4
    python predict_video.py --input video.mp4 --model models/tsl_model.pt
    python predict_video.py --input video.mp4 --top-k 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import numpy as np

from ..train.models import MLP, GRUModel, MOPGRU, HybridGRUTransformer

from ..data.extractor import (
    FEATURE_DIMS,
    MediaPipeTasksLandmarkExtractor,
    adapt_features_to_model,
    build_enhanced_sequence,
    extract_features,
    extract_sequence_features,
    extract_video_landmarks,
    normalize_features,
    report_extractor_compatibility,
    resolve_feature_level_for_inference,
)


# =============================================================================
# MODEL LOADING
# =============================================================================
def _detect_feature_level_from_dim(input_dim: int) -> str:
    """Auto-detect feature level from input dimension."""
    for level, dim in FEATURE_DIMS.items():
        if input_dim == dim:
            return level
    # If no exact match, return 'basic' as default
    return "basic"


def load_model(model_path):
    """Load model from checkpoint with feature dimension validation."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    report_extractor_compatibility(checkpoint)

    # Support both legacy and current checkpoint schemas.
    if "labels" in checkpoint and "label_to_idx" in checkpoint:
        labels = checkpoint["labels"]
        idx_to_label = {idx: label for label, idx in checkpoint["label_to_idx"].items()}
    elif "classes" in checkpoint:
        labels = [str(x) for x in checkpoint["classes"]]
        idx_to_label = {idx: label for idx, label in enumerate(labels)}
    else:
        raise KeyError("Checkpoint missing labels/classes metadata")

    mean = checkpoint.get("normalization_mean", checkpoint.get("mean"))
    std = checkpoint.get("normalization_std", checkpoint.get("std"))
    if mean is None or std is None:
        raise KeyError("Checkpoint missing normalization stats (mean/std)")
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    input_dim = checkpoint.get("input_dim", int(mean.shape[0]))
    num_classes = checkpoint.get("num_classes", len(labels))
    config = checkpoint.get("config", {})

    # Auto-detect or use explicit feature_level
    feature_level = str(config.get("feature_level", _detect_feature_level_from_dim(input_dim)))
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

    model_name = config.get("model", checkpoint.get("model", "mlp"))
    model_classes = {
        "mlp": MLP,
        "gru": GRUModel,
        "mopgru": MOPGRU,
        "hybrid": HybridGRUTransformer,
    }
    model_class = model_classes.get(model_name, MLP)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None:
        raise KeyError("Checkpoint missing model weights (model_state_dict/state_dict)")

    model = model_class(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=config.get("hidden_dim", config.get("hidden", 256)),
        num_layers=config.get("num_layers", config.get("layers", 3)),
        dropout=config.get("dropout", 0.3),
    )
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Some older checkpoints tagged as GRU actually store MLP weights.
        if model_name == "gru":
            fallback = MLP(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dim=config.get("hidden_dim", config.get("hidden", 256)),
                num_layers=config.get("num_layers", config.get("layers", 3)),
                dropout=config.get("dropout", 0.3),
            )
            fallback.load_state_dict(state_dict)
            model = fallback
        else:
            raise
    model.eval()

    seq_mode = bool(checkpoint.get("seq_mode", False))
    target_frames = int(checkpoint.get("target_frames", 30))

    print(f"Model configuration: feature_level='{feature_level}', input_dim={input_dim}")

    return model, labels, idx_to_label, mean, std, seq_mode, target_frames, feature_level


LandmarkExtractor = MediaPipeTasksLandmarkExtractor


# =============================================================================
# VIDEO PROCESSING
# =============================================================================
def process_video(
    video_path,
    model,
    idx_to_label,
    mean,
    std,
    extractor,
    top_k=3,
    verbose=True,
    seq_mode=False,
    target_frames=30,
    feature_level="basic",
):
    """Process video and predict Thai word.

    Args:
        video_path: Path to video file
        model: PyTorch model
        idx_to_label: Index to label mapping
        mean: Normalization mean
        std: Normalization std
        extractor: LandmarkExtractor instance
        top_k: Number of top predictions to return
        verbose: Print progress

    Returns:
        (predicted_word, confidence, all_predictions)
    """
    if verbose:
        print(f"Processing video: {video_path}")

    frames, stats = extract_video_landmarks(video_path, extractor, verbose=verbose)

    if verbose:
        print(f"  Total: {stats['total_frames']} frames, {len(frames)} with landmarks")

    min_frames = target_frames if seq_mode else 1
    if len(frames) < min_frames:
        if verbose:
            warning = f"  Warning: Too few frames with landmarks detected ({len(frames)} < {min_frames})"
            print(warning)
        return None, 0.0, []

    if seq_mode:
        if feature_level == 'enhanced':
            features = build_enhanced_sequence(
                frames,
                feature_level='enhanced',
                target_frames=target_frames,
            )
        else:
            features = extract_sequence_features(frames, feature_level=feature_level, target_frames=target_frames)
        if features is None:
            return None, 0.0, []
    else:
        features = extract_features(frames, feature_level=feature_level)
        if features is None:
            return None, 0.0, []

    # Adapt features to model's expected dimension (handles mismatches gracefully)
    features = adapt_features_to_model(features, len(mean), feature_level)
    if features is None:
        return None, 0.0, []

    normalized = normalize_features(features, mean, std)
    if normalized is None:
        return None, 0.0, []

    if seq_mode:
        tensor = torch.tensor([normalized], dtype=torch.float32)
    else:
        tensor = torch.tensor([normalized], dtype=torch.float32)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_indices = probs.argsort(descending=True)[:top_k].tolist()
        predictions = [(idx_to_label[idx], probs[idx].item()) for idx in top_indices]

    top_word, confidence = predictions[0]

    if confidence < 0.70:
        if verbose:
            print(f"  Low confidence ({confidence:.1%}), no prediction returned")
        return None, confidence, predictions

    return top_word, confidence, predictions


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Thai Sign Language Video Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python predict_video.py --input video.mp4
    python predict_video.py --input video.mp4 --top-k 5
    python predict_video.py --input video.mp4 --model models/tsl_model.pt
""",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input video file")
    parser.add_argument("--model", type=Path, default=Path("models/tsl_model.pt"), help="Model checkpoint")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions")
    parser.add_argument("--feature-level", type=str, default=None,
                        choices=['basic', 'finger', 'enhanced', 'full'],
                        help="Override feature level for inference (default: auto-detect from model)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Video file not found: {args.input}")
        return 1

    if not args.model.exists():
        print(f"Error: Model file not found: {args.model}")
        print("Please run train_cv.py first to train the model.")
        return 1

    print(f"Loading model from: {args.model}")
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
    print(f"Input dimension: {len(mean)}")
    print(f"Sequence mode: {seq_mode} (target_frames={target_frames})")

    print("\nInitializing MediaPipe landmarkers...")
    with LandmarkExtractor() as extractor:
        print(f"\n{'=' * 60}")
        print("THAI SIGN LANGUAGE VIDEO PREDICTION")
        print(f"{'=' * 60}")
        print(f"Input: {args.input}")
        print(f"{'=' * 60}\n")

        predicted_word, confidence, all_predictions = process_video(
            args.input,
            model,
            idx_to_label,
            mean,
            std,
            extractor,
            top_k=args.top_k,
            seq_mode=seq_mode,
            target_frames=target_frames,
            feature_level=feature_level,
        )

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")

    if predicted_word:
        print(f"\nPredicted: {predicted_word} ({confidence:.1%})")

        if args.top_k > 1:
            print(f"\nTop {args.top_k} predictions:")
            for word, prob in all_predictions:
                print(f"  {word}: {prob:.1%}")
    else:
        print("\nNo prediction could be made (insufficient landmarks detected)")

    print(f"\n{'=' * 60}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
