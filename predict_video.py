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
import math
from pathlib import Path

import torch
import torch.nn as nn

from tsl_tasks_extractor import (
    MediaPipeTasksLandmarkExtractor,
    extract_features,
    extract_video_landmarks,
    normalize_features,
    report_extractor_compatibility,
)


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
# MODEL LOADING
# =============================================================================
def load_model(model_path):
    """Load model from checkpoint."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    report_extractor_compatibility(checkpoint)

    labels = checkpoint["labels"]
    idx_to_label = {idx: label for label, idx in checkpoint["label_to_idx"].items()}
    mean = checkpoint["normalization_mean"]
    std = checkpoint["normalization_std"]
    input_dim = checkpoint["input_dim"]
    num_classes = checkpoint["num_classes"]
    config = checkpoint.get("config", {})

    model = MLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=config.get("hidden_dim", 256),
        num_layers=config.get("num_layers", 3),
        dropout=config.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, labels, idx_to_label, mean, std


LandmarkExtractor = MediaPipeTasksLandmarkExtractor


# =============================================================================
# VIDEO PROCESSING
# =============================================================================
def process_video(video_path, model, idx_to_label, mean, std, extractor, top_k=3, verbose=True):
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

    if len(frames) < 60:
        if verbose:
            print(f"  Warning: Too few frames with landmarks detected ({len(frames)} < 60)")
        return None, 0.0, []

    features = extract_features(frames)

    if features is None:
        return None, 0.0, []

    if len(features) != len(mean):
        if verbose:
            print(f"  Warning: feature dimension mismatch: expected {len(mean)}, got {len(features)}")
        return None, 0.0, []

    normalized = normalize_features(features, mean, std)
    if normalized is None:
        return None, 0.0, []
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
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Video file not found: {args.input}")
        return 1

    if not args.model.exists():
        print(f"Error: Model file not found: {args.model}")
        print("Please run train_cv.py first to train the model.")
        return 1

    print(f"Loading model from: {args.model}")
    model, labels, idx_to_label, mean, std = load_model(args.model)
    print(f"Loaded {len(labels)} Thai word classes")
    print(f"Input dimension: {len(mean)}")

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
