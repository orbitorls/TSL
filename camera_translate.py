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
import torch.nn as nn
from PIL import Image as PILImage, ImageDraw, ImageFont

from tsl_tasks_extractor import (
    LANDMARK_VECTOR_DIM,
    MediaPipeTasksLandmarkExtractor,
    draw_debug_overlay,
    extract_features,
    normalize_features,
    report_extractor_compatibility,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


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


# =============================================================================
# REAL-TIME TRANSLATION
# =============================================================================
class ThaiSignTranslator:
    def __init__(self, model_path="models/tsl_model.pt"):
        print(f"Loading model from: {model_path}")
        self.model, self.labels, self.idx_to_label, self.mean, self.std = load_model(model_path)
        print(f"Loaded {len(self.labels)} Thai word classes")

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

    def _get_cached_font(self, font_size):
        """Retrieve a cached PIL font of the requested size, loading it if necessary."""
        if font_size in self._font_cache:
            return self._font_cache[font_size]

        font_paths = [
            "C:/Windows/Fonts/phagspa.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "C:/Windows/Fonts/seguisym.ttf",
            "C:/Windows/Fonts/FONTA.TTF",
            "C:/Windows/Fonts/FONTB.TTF",
        ]
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default(size=font_size)

        self._font_cache[font_size] = font
        return font

    def process_frame(self, frame):
        """Process a single frame and return landmarks vector (174 values)."""
        if not self.mediapipe_available or self.extractor is None:
            return frame, None

        try:
            result = self.extractor.extract_frame(frame)
            frame = draw_debug_overlay(frame, result)
            return frame, result.landmarks
        except Exception as exc:
            print(f"Error processing frame: {exc}")
            return frame, None

    def predict(self):
        """Make prediction from buffered sequence."""
        if len(self.sequence_buffer) < 60:
            return None, 0.0

        frames = list(self.sequence_buffer)
        if not frames:
            return None, 0.0

        for index, frame in enumerate(frames):
            if len(frame) != LANDMARK_VECTOR_DIM:
                print(f"ERROR: Frame {index} has {len(frame)} values (expected {LANDMARK_VECTOR_DIM})")
                return None, 0.0

        features = extract_features(frames)
        if features is None:
            return None, 0.0

        target_len = len(self.mean)
        if target_len == 0:
            return None, 0.0

        if len(features) != target_len:
            print(f"ERROR: Feature dimension mismatch: expected {target_len}, got {len(features)}")
            return None, 0.0

        normalized = normalize_features(features, self.mean, self.std)
        if normalized is None:
            return None, 0.0
        tensor = torch.tensor([normalized], dtype=torch.float32)

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

        font = self._get_cached_font(font_size)

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
                self.consecutive_missing = 0
            else:
                self.consecutive_missing += 1
                if self.consecutive_missing > 10 and len(self.sequence_buffer) > 20:
                    self.sequence_buffer.clear()
                    self.consecutive_missing = 0

            if frame_count % 5 == 0 and len(self.sequence_buffer) >= 60:
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
                font = self._get_cached_font(80)
                conf_font = self._get_cached_font(30)

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

            buffer_count = len(self.sequence_buffer)
            buffer_text = f"Buffer: {buffer_count}/100"
            cv2.rectangle(frame, (8, 8), (220, 55), (0, 0, 0), -1)
            cv2.rectangle(frame, (8, 8), (220, 55), (0, 200, 0), 2)

            bar_width = 150
            bar_x, bar_y = 15, 40
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 10), (50, 50, 50), -1)
            fill_width = int(bar_width * buffer_count / 100)
            if fill_width > 0:
                color = (0, 255, 0) if buffer_count >= 60 else (0, 200, 0)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + 10), color, -1)
            cv2.putText(frame, buffer_text, (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if not landmarks:
                warning_text = "Show your hand/face to camera"
                cv2.rectangle(frame, (w - 350, 8), (w - 8, 55), (50, 0, 0), -1)
                cv2.rectangle(frame, (w - 350, 8), (w - 8, 55), (0, 0, 255), 2)
                cv2.putText(frame, warning_text, (w - 340, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                if len(self.sequence_buffer) < 60:
                    status_text = f"Collecting: {len(self.sequence_buffer)}/60"
                    cv2.rectangle(frame, (w - 220, 8), (w - 8, 55), (50, 50, 0), -1)
                    cv2.rectangle(frame, (w - 220, 8), (w - 8, 55), (0, 200, 255), 2)
                    cv2.putText(frame, status_text, (w - 210, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
                else:
                    cv2.rectangle(frame, (w - 200, 8), (w - 8, 55), (0, 50, 0), -1)
                    cv2.rectangle(frame, (w - 200, 8), (w - 8, 55), (0, 255, 0), 2)
                    cv2.putText(frame, "Ready", (w - 190, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.rectangle(frame, (w - 200, h - 50), (w - 8, h - 8), (30, 30, 30), -1)
            cv2.putText(frame, "q: Quit | c: Clear", (w - 195, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            if len(self.sequence_buffer) >= 60:
                features = extract_features(list(self.sequence_buffer))
                if features and len(features) == len(self.mean):
                    normalized = normalize_features(features, self.mean, self.std)
                    if normalized is None:
                        print(f"ERROR: normalize_features failed (mean={len(self.mean)}, std={len(self.std)})")
                    if normalized:
                        tensor = torch.tensor([normalized], dtype=torch.float32)

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
                            font = self._get_cached_font(22)

                            draw.text((sidebar_x + 12, y_pos), f"{index + 1}. {word}", font=font, fill=(255, 255, 255))
                            draw.text((sidebar_x + 170, y_pos), f"{conf:.0%}", font=font, fill=(200, 255, 200))
                            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            cv2.imshow("Thai Sign Language Translation", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                self.sequence_buffer.clear()
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
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("Please run train_cv.py first to train the model.")
        return 1

    translator = ThaiSignTranslator(model_path=args.model)
    translator.run()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
