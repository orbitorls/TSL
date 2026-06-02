"""
webcam_demo.py — live Thai fingerspelling recognition.

Requires (all in requirements.txt):
  mediapipe==0.10.14  opencv-python  tensorflow  scikit-learn  joblib  pillow

Optionally (inference-only, lighter than tensorflow):
  pip install tflite-runtime
  → auto-detected when --use-tflite is active or model.tflite is present.

Also requires:
  assets/NotoSansThai-Regular.ttf   (see README for download link)
  model.keras  OR  model.tflite     (trained in Colab, downloaded here)
  labels.json                        (class-index → Thai letter)
  scaler.pkl                         (StandardScaler fitted during training)

Run:
  python webcam_demo.py
  python webcam_demo.py --use-tflite --threshold 0.7 --top-k 3
  python webcam_demo.py --save-log session.csv
  Press  q  to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from src.keypoints import FEATURE_SIZE, extract_and_normalize
from src.webcam_runtime import (
    EMABuffer,
    draw_thai_text,
    format_topk,
    iso_timestamp_ms,
    load_demo_artifacts,
    load_thai_font,
    open_csv_writer,
    read_frame_with_retry,
)


DEFAULT_MODEL = "model.keras"
DEFAULT_LABELS = "labels.json"
DEFAULT_SCALER = "scaler.pkl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Thai fingerspelling live demo")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--labels", default=DEFAULT_LABELS)
    p.add_argument("--scaler", default=DEFAULT_SCALER)
    p.add_argument("--cam", type=int, default=0, help="Webcam index")
    p.add_argument("--threshold", type=float, default=0.7,
                   help="Minimum smoothed top-1 probability to show a letter")
    p.add_argument("--smoothing-alpha", type=float, default=0.4,
                   help="EMA weight for new frames; 1.0 = no smoothing")
    p.add_argument("--frame-skip", type=int, default=1,
                   help="Run model inference every N frames (>=1)")
    p.add_argument("--top-k", type=int, default=2,
                   help="How many top candidates to display")
    p.add_argument("--save-log", type=str, default=None,
                   help="Path to CSV: ts_iso,label,confidence")
    p.add_argument("--use-tflite", action="store_true", default=False,
                   help="Force TFLite backend; otherwise auto-detect")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.frame_skip = max(1, int(args.frame_skip))
    if args.smoothing_alpha < 0.05:
        args.smoothing_alpha = 0.05
    if args.smoothing_alpha > 1.0:
        args.smoothing_alpha = 1.0
    args.top_k = max(1, int(args.top_k))

    print("Loading model artifacts …")
    try:
        predictor, labels, scaler = load_demo_artifacts(
            args.model,
            args.labels,
            args.scaler,
            args.use_tflite,
            expected_feature_dim=FEATURE_SIZE,
            track="thai_fingerspelling",
            manifest_path=Path(__file__).parent / "model_manifest.json",
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        sys.exit(f"[ERROR] {exc}")

    print(f"  Classes: {sorted(labels.values())}")
    print(f"  Threshold={args.threshold}  alpha={args.smoothing_alpha}  "
          f"frame_skip={args.frame_skip}  top_k={args.top_k}")

    font_root = Path(__file__).parent
    font_big = load_thai_font(font_root, 72)
    font_med = load_thai_font(font_root, 32)
    smoother = EMABuffer(args.smoothing_alpha)

    csv_file = None
    csv_writer = None
    if args.save_log:
        csv_file, csv_writer = open_csv_writer(
            args.save_log,
            ["timestamp_iso", "label", "confidence", "top2_label", "top2_confidence"],
        )

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hand_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
    hand_conn_spec = mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        if csv_file is not None:
            csv_file.close()
        sys.exit(f"[ERROR] Cannot open webcam index {args.cam}")

    print("Webcam open. Press  q  to quit.")

    display_letter = ""
    overlay_text = "ไม่พบมือ / no hand"
    top_k_text = ""
    confidence = 0.0
    top_pairs: list[tuple[str, float]] = []
    frame_idx = 0
    last_good_frame_time = time.monotonic()

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        ) as hands_cfg:
            while True:
                ok, frame = read_frame_with_retry(cap)
                if not ok or frame is None:
                    now = time.monotonic()
                    if now - last_good_frame_time > 2.0:
                        print("[WARN] No frames for >2s — attempting webcam reconnect …")
                        cap.release()
                        cap = cv2.VideoCapture(args.cam)
                        last_good_frame_time = time.monotonic()
                        if not cap.isOpened():
                            print("[ERROR] Reconnect failed; exiting.")
                            break
                    continue
                last_good_frame_time = time.monotonic()

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hands_cfg.process(rgb)
                rgb.flags.writeable = True

                if results.multi_hand_landmarks:
                    for hand_lm in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                            hand_spec,
                            hand_conn_spec,
                        )

                should_predict = (frame_idx % args.frame_skip) == 0
                frame_idx += 1

                if not results.multi_hand_landmarks:
                    smoother.reset()
                    overlay_text = "ไม่พบมือ / no hand"
                    display_letter = ""
                    top_k_text = ""
                    confidence = 0.0
                    top_pairs = []
                elif should_predict:
                    feat = extract_and_normalize(results)
                    if feat is None:
                        smoother.reset()
                        overlay_text = "ไม่พบมือ / no hand"
                        display_letter = ""
                        top_k_text = ""
                        confidence = 0.0
                        top_pairs = []
                    else:
                        feat_scaled = scaler.transform(feat.reshape(1, -1))
                        raw_probs = predictor.predict(feat_scaled)
                        smoothed = smoother.update(raw_probs)

                        pred_idx = int(np.argmax(smoothed))
                        confidence = float(smoothed[pred_idx])

                        if confidence >= args.threshold:
                            display_letter = labels.get(str(pred_idx), "?")
                            overlay_text = f"{display_letter}  ({confidence * 100:.0f}%)"
                            top_k_text, top_pairs = format_topk(smoothed, labels, args.top_k)
                        else:
                            display_letter = "?"
                            overlay_text = f"?  ({confidence * 100:.0f}%)"
                            top_k_text = ""
                            top_pairs = []

                        if csv_writer is not None:
                            top2_label = top_pairs[1][0] if len(top_pairs) > 1 else ""
                            top2_conf = f"{top_pairs[1][1]:.4f}" if len(top_pairs) > 1 else ""
                            csv_writer.writerow([
                                iso_timestamp_ms(),
                                display_letter,
                                f"{confidence:.4f}",
                                top2_label,
                                top2_conf,
                            ])
                            csv_file.flush()

                h, w = frame.shape[:2]
                banner_h = 140
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

                if display_letter and display_letter != "?":
                    frame = draw_thai_text(
                        frame,
                        display_letter,
                        (20, h - banner_h + 5),
                        font_big,
                        color=(0, 255, 120),
                    )
                if top_k_text:
                    frame = draw_thai_text(
                        frame,
                        top_k_text,
                        (20, h - 78),
                        font_med,
                        color=(180, 220, 255),
                    )
                frame = draw_thai_text(
                    frame,
                    overlay_text,
                    (20, h - 40),
                    font_med,
                    color=(220, 220, 220),
                )
                if confidence > 0:
                    bar_w = int((w - 40) * min(max(confidence, 0.0), 1.0))
                    bar_color = (0, 200, 80) if confidence >= args.threshold else (0, 120, 255)
                    cv2.rectangle(frame, (20, h - 10), (20 + bar_w, h - 4), bar_color, -1)

                cv2.imshow("Thai Fingerspelling — press q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if csv_file is not None:
            try:
                csv_file.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
