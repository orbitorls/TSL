"""
Thai Sign Language Real-Time Camera Translation
Real-time Thai sign language translation from camera

Requirements:
    pip install torch numpy opencv-python mediapipe pillow
"""

from __future__ import annotations

import os
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from ..data.extractor import (
    FEATURE_DIMS,
    LANDMARK_VECTOR_DIM,
    MediaPipeTasksLandmarkExtractor,
    adapt_features_to_model,
    build_enhanced_sequence,
    draw_debug_overlay,
    extract_features,
    extract_sequence_features,
    normalize_features,
    report_extractor_compatibility,
    resolve_feature_level_for_inference,
)
from ..train.models import MLP, MOPGRU, GRUModel, HybridGRUTransformer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


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

    mean_raw = checkpoint.get("normalization_mean", checkpoint.get("mean"))
    std_raw = checkpoint.get("normalization_std", checkpoint.get("std"))
    if mean_raw is None or std_raw is None:
        raise KeyError("Checkpoint missing normalization stats (mean/std)")
    mean = np.asarray(mean_raw, dtype=np.float32)
    std = np.asarray(std_raw, dtype=np.float32)

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

    model = model_class(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=config.get("hidden_dim", config.get("hidden", 256)),
        num_layers=config.get("num_layers", config.get("layers", 3)),
        dropout=config.get("dropout", 0.3),
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None:
        raise KeyError("Checkpoint missing model weights (model_state_dict/state_dict)")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Some legacy checkpoints are tagged as GRU but contain MLP weights.
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


# =============================================================================
# REAL-TIME TRANSLATION
# =============================================================================
class ThaiSignTranslator:
    def __init__(self, model_path="models/tsl_model.pt", feature_level=None):
        print(f"Loading model from: {model_path}")
        (
            self.model,
            self.labels,
            self.idx_to_label,
            self.mean,
            self.std,
            self.seq_mode,
            self.target_frames,
            self.feature_level,
        ) = load_model(model_path)
        print(f"Loaded {len(self.labels)} Thai word classes")
        print(f"Sequence mode: {self.seq_mode} (target_frames={self.target_frames})")

        # Resolve feature level safely against model constraints
        resolved_level, warn_msg = resolve_feature_level_for_inference(
            feature_level,
            model_input_dim=len(self.mean),
            checkpoint_feature_level=self.feature_level,
        )
        if warn_msg:
            print(f"[WARN] {warn_msg}")
        self.feature_level = resolved_level
        if feature_level is not None and feature_level != resolved_level:
            print(f"[INFO] Auto-adjusted feature level to '{resolved_level}' for model compatibility")
        print(f"Using feature level: {self.feature_level}")

        # Enhanced features support
        self.use_enhanced = (self.feature_level == 'enhanced')
        self.enhanced_frame_buffer = deque(maxlen=100)  # stores landmark dicts

        self.extractor: MediaPipeTasksLandmarkExtractor | None = None
        self.mediapipe_available = False

        self.sequence_buffer = deque(maxlen=100)
        self.prediction_history = deque(maxlen=5)
        self.current_prediction = ""
        self.confidence = 0.0
        self.consecutive_missing = 0
        self._font_cache = {}

        try:
            print("Initializing MediaPipe Tasks landmarkers...")
            self.extractor = MediaPipeTasksLandmarkExtractor()
            self.mediapipe_available = True
            print("✓ MediaPipe initialized successfully")
            print(f"  - Expected input dim: {len(self.mean)}")
        except Exception as exc:
            print(f"✗ MediaPipe initialization failed: {exc}")
            print("\nTo fix this, run:")
            print("  pip install --upgrade mediapipe")
            print("  pip install opencv-python")
            self.mediapipe_available = False

    def _get_cached_font(self, font_size, font_paths, default_size=None):
        """Retrieve font from cache or load and cache it."""
        if default_size is None:
            default_size = font_size

        # Use immutable key for caching
        cache_key = (font_size, tuple(font_paths), default_size)
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except Exception:
                continue

        if font is None:
            try:
                font = ImageFont.load_default(size=default_size)
            except Exception:
                # Older Pillow versions don't support size in load_default
                font = ImageFont.load_default()

        self._font_cache[cache_key] = font
        return font

    def process_frame(self, frame):
        """Process a single frame and return dense landmarks dict (162 values)."""
        if not self.mediapipe_available or self.extractor is None:
            return frame, None

        try:
            result = self.extractor.extract_frame(frame)
            frame = draw_debug_overlay(frame, result)
            return frame, result.landmarks
        except Exception as exc:
            print(f"Error processing frame: {exc}")
            return frame, None

    def _store_enhanced_frame(self, landmarks):
        """Store raw landmark dict for enhanced feature computation."""
        if landmarks is not None and len(landmarks) == LANDMARK_VECTOR_DIM:
            self.enhanced_frame_buffer.append(landmarks)

    def predict(self):
        """Make prediction from buffered sequence."""
        if len(self.sequence_buffer) < self.target_frames:
            return None, 0.0

        frames = list(self.sequence_buffer)
        if not frames:
            return None, 0.0

        for index, frame in enumerate(frames):
            if len(frame) != LANDMARK_VECTOR_DIM:
                print(f"ERROR: Frame {index} has {len(frame)} values (expected {LANDMARK_VECTOR_DIM})")
                return None, 0.0

        min_frames = self.target_frames if self.seq_mode else 1
        if len(self.sequence_buffer) < min_frames:
            return None, 0.0

        target_len = len(self.mean)
        if target_len == 0:
            return None, 0.0

        if self.use_enhanced and self.seq_mode:
            features = build_enhanced_sequence(
                frames,
                feature_level='enhanced',
                target_frames=self.target_frames,
            )
        elif self.seq_mode:
            features = extract_sequence_features(
                frames,
                feature_level=self.feature_level,
                target_frames=self.target_frames,
            )
        else:
            features = extract_features(frames, feature_level=self.feature_level)

        # Adapt features to model's expected dimension (handles mismatches gracefully)
        features = adapt_features_to_model(features, target_len, self.feature_level)
        if features is None:
            return None, 0.0

        feature_dim = features.shape[-1] if isinstance(features, np.ndarray) and features.ndim >= 2 else len(features)
        tensor_input = normalize_features(features, self.mean, self.std)

        if self.seq_mode:
            tensor = torch.tensor(tensor_input[None, ...], dtype=torch.float32)
        else:
            tensor = torch.tensor([tensor_input], dtype=torch.float32)

        normalized = tensor_input
        if normalized is None:
            return None, 0.0

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            top_idx = probs.argmax().item()
            confidence = probs[top_idx].item()

        if confidence < 0.70:
            return None, confidence

        return self.idx_to_label[top_idx], confidence

    def _draw_thai_text(self, frame, text, pos, font_size=60, color=(255, 255, 255), bg_color=None):
        """Draw Thai text using PIL (supports Thai characters)."""
        x, y = pos
        pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        font_paths = [
            "C:/Windows/Fonts/phagspa.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "C:/Windows/Fonts/seguisym.ttf",
            "C:/Windows/Fonts/FONTA.TTF",
            "C:/Windows/Fonts/FONTB.TTF",
        ]
        font = self._get_cached_font(font_size, font_paths)

        bbox = draw.textbbox((x, y), text, font=font)

        if bg_color:
            padding = 15
            draw.rectangle(
                [(bbox[0] - padding, bbox[1] - padding), (bbox[2] + padding, bbox[3] + padding)],
                fill=bg_color,
            )

        draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=color)

        frame[:, :] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return frame

    def run(self):
        """Run real-time translation from camera."""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Cannot open camera")
            print("Make sure you have a webcam connected")
            return

        print("\n" + "=" * 60)
        print("Thai Sign Language Real-Time Translation")
        print("=" * 60)
        print("Controls:")
        print("  'q' - Quit")
        print("  'c' - Clear prediction")
        print("=" * 60 + "\n")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame, landmarks = self.process_frame(frame)

            if landmarks and len(landmarks) == LANDMARK_VECTOR_DIM:
                self.sequence_buffer.append(landmarks)
                self._store_enhanced_frame(landmarks)
                self.consecutive_missing = 0
            else:
                self.consecutive_missing += 1
                if self.consecutive_missing > 10 and len(self.sequence_buffer) > 20:
                    self.sequence_buffer.clear()
                    self.enhanced_frame_buffer.clear()
                    self.consecutive_missing = 0

            min_frames = self.target_frames if self.seq_mode else 1
            if frame_count % 5 == 0 and len(self.sequence_buffer) >= min_frames:
                pred, conf = self.predict()
                if pred:
                    self.prediction_history.append(pred)
                    most_common = Counter(self.prediction_history).most_common(1)
                    if most_common:
                        self.current_prediction = most_common[0][0]
                        self.confidence = conf

            h, w = frame.shape[:2]

            if self.current_prediction:
                text = self.current_prediction
                conf_text = f"Confidence: {self.confidence:.1%}"

                pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                draw = ImageDraw.Draw(pil_img)

                font_paths = [
                    "C:/Windows/Fonts/tahoma.ttf",
                    "C:/Windows/Fonts/phagspa.ttf"
                ]
                font = self._get_cached_font(80, font_paths, default_size=60)
                conf_font = self._get_cached_font(30, font_paths, default_size=25)

                bbox = draw.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
                text_bbox = draw.textbbox((0, 0), conf_text, font=conf_font)
                conf_w = text_bbox[2] - text_bbox[0]

                box_w = max(text_w, conf_w) + 60
                box_h = 160
                box_x = (w - box_w) // 2
                box_y = h - box_h - 40

                draw.rounded_rectangle(
                    [(box_x, box_y), (box_x + box_w, box_y + box_h)],
                    radius=20,
                    fill=(0, 100, 0),
                )

                text_x = (w - text_w) // 2
                text_y = box_y + 20
                draw.text((text_x + 3, text_y + 3), text, font=font, fill=(0, 0, 0))
                draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))

                conf_x = (w - conf_w) // 2
                conf_y = box_y + 110
                draw.text((conf_x + 2, conf_y + 2), conf_text, font=conf_font, fill=(0, 0, 0))
                draw.text((conf_x, conf_y), conf_text, font=conf_font, fill=(200, 255, 200))

                frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            min_frames = self.target_frames if self.seq_mode else 1
            display_target = self.target_frames if self.seq_mode else 1
            buffer_count = len(self.sequence_buffer)
            buffer_text = f"Buffer: {buffer_count}/{display_target}"
            cv2.rectangle(frame, (8, 8), (220, 55), (0, 0, 0), -1)
            cv2.rectangle(frame, (8, 8), (220, 55), (0, 200, 0), 2)

            bar_width = 150
            bar_x, bar_y = 15, 40
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 10), (50, 50, 50), -1)
            fill_width = int(bar_width * min(buffer_count, display_target) / display_target)
            if fill_width > 0:
                color = (0, 255, 0) if buffer_count >= min_frames else (0, 200, 0)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + 10), color, -1)
            cv2.putText(frame, buffer_text, (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if not landmarks:
                warning_text = "Show your hand/face to camera"
                cv2.rectangle(frame, (w - 350, 8), (w - 8, 55), (50, 0, 0), -1)
                cv2.rectangle(frame, (w - 350, 8), (w - 8, 55), (0, 0, 255), 2)
                cv2.putText(frame, warning_text, (w - 340, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                if len(self.sequence_buffer) < min_frames:
                    status_text = f"Collecting: {len(self.sequence_buffer)}/{display_target}"
                    cv2.rectangle(frame, (w - 220, 8), (w - 8, 55), (50, 50, 0), -1)
                    cv2.rectangle(frame, (w - 220, 8), (w - 8, 55), (0, 200, 255), 2)
                    cv2.putText(frame, status_text, (w - 210, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
                else:
                    cv2.rectangle(frame, (w - 200, 8), (w - 8, 55), (0, 50, 0), -1)
                    cv2.rectangle(frame, (w - 200, 8), (w - 8, 55), (0, 255, 0), 2)
                    cv2.putText(frame, "Ready", (w - 190, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.rectangle(frame, (w - 200, h - 50), (w - 8, h - 8), (30, 30, 30), -1)
            cv2.putText(frame, "q: Quit | c: Clear", (w - 195, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            if len(self.sequence_buffer) >= min_frames:
                if self.use_enhanced and self.seq_mode:
                    features = build_enhanced_sequence(
                        list(self.sequence_buffer),
                        feature_level='enhanced',
                        target_frames=self.target_frames,
                    )
                elif self.seq_mode:
                    features = extract_sequence_features(
                        list(self.sequence_buffer),
                        feature_level=self.feature_level,
                        target_frames=self.target_frames,
                    )
                else:
                    features = extract_features(list(self.sequence_buffer), feature_level=self.feature_level)

                # Adapt features to model's expected dimension (handles mismatches gracefully)
                features = adapt_features_to_model(features, len(self.mean), self.feature_level)
                feature_dim = features.shape[-1] if isinstance(features, np.ndarray) else len(features) if features is not None else -1
                normalized = normalize_features(features, self.mean, self.std) if features is not None else None
                tensor = (
                    torch.tensor(normalized[None, ...], dtype=torch.float32)
                    if normalized is not None and self.seq_mode
                    else torch.tensor([normalized], dtype=torch.float32) if normalized is not None else None
                )

                if features is not None and normalized is not None and tensor is not None:
                    with torch.no_grad():
                        outputs = self.model(tensor)
                        probs = torch.softmax(outputs, dim=1)[0]
                        top3 = torch.topk(probs, 3)
                        top3_idx = top3.indices.tolist()
                        top3_conf = top3.values.tolist()

                    sidebar_x = w - 220
                    sidebar_y = 70
                    sidebar_h = 120
                    cv2.rectangle(frame, (sidebar_x, sidebar_y), (w - 8, sidebar_y + sidebar_h), (20, 20, 40), -1)
                    cv2.rectangle(frame, (sidebar_x, sidebar_y), (w - 8, sidebar_y + sidebar_h), (100, 100, 150), 1)

                    for index, (label_index, conf) in enumerate(zip(top3_idx, top3_conf)):
                        word = self.idx_to_label[label_index]
                        bar_w = int(150 * conf)
                        y_pos = sidebar_y + 15 + index * 38
                        cv2.rectangle(frame, (sidebar_x + 10, y_pos + 18), (sidebar_x + 10 + bar_w, y_pos + 28), (40, 40, 60), -1)
                        cv2.rectangle(frame, (sidebar_x + 10, y_pos + 18), (sidebar_x + 10 + bar_w, y_pos + 28), (0, 200, 100), -1)

                        pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pil_img)

                        font_paths = [
                            "C:/Windows/Fonts/tahoma.ttf",
                            "C:/Windows/Fonts/phagspa.ttf"
                        ]
                        font = self._get_cached_font(22, font_paths, default_size=18)

                        draw.text((sidebar_x + 12, y_pos), f"{index + 1}. {word}", font=font, fill=(255, 255, 255))
                        draw.text((sidebar_x + 170, y_pos), f"{conf:.0%}", font=font, fill=(200, 255, 200))
                        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            cv2.imshow("Thai Sign Language Translation", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                self.sequence_buffer.clear()
                self.enhanced_frame_buffer.clear()
                self.prediction_history.clear()
                self.current_prediction = ""
                self.confidence = 0.0
                print("Cleared predictions")

        cap.release()
        cv2.destroyAllWindows()

        if self.extractor is not None:
            self.extractor.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Thai Sign Language Real-Time Camera Translation")
    parser.add_argument("--model", type=str, default="models/tsl_model.pt")
    parser.add_argument("--feature-level", type=str, default=None,
                        choices=['basic', 'finger', 'enhanced', 'full'],
                        help="Override feature level for inference (default: auto-detect from model)")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("Please run train_cv.py first to train the model.")
        return 1

    translator = ThaiSignTranslator(model_path=args.model, feature_level=args.feature_level)
    translator.run()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
