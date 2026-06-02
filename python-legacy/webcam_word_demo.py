"""
webcam_word_demo.py — live Thai word sign recognition (TSL-51 track).

License: trained on TSL-51 dataset (CC BY-NC-SA 4.0 — NON-COMMERCIAL use only).
Do not use for commercial applications without permission from dataset maintainers.

Requires (all in requirements.txt):
  mediapipe==0.10.14  opencv-python  tensorflow  scikit-learn  joblib  pillow

Optionally (inference-only, lighter than tensorflow):
  pip install tflite-runtime
  → auto-detected when --use-tflite is active or tsl51_model.tflite is present.

Also requires:
  assets/NotoSansThai-Regular.ttf   (see README for download link)
  tsl51_model.keras  OR  tsl51_model.tflite
  tsl51_labels.json
  tsl51_scaler.pkl

Run:
  python webcam_word_demo.py
  python webcam_word_demo.py --use-tflite --threshold 0.55 --motion-min 0.008
  python webcam_word_demo.py --save-log session.csv
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
from src.sequence_keypoints import (
    FEATURE_DIM,
    SEQ_LEN_DEFAULT,
    SequenceBuffer,
    extract_holistic_frame,
)
from src.webcam_runtime import (
    EMABuffer,
    draw_thai_text,
    iso_timestamp_ms,
    load_demo_artifacts,
    load_thai_font,
    open_csv_writer,
    read_frame_with_retry,
)


DEFAULT_MODEL = "tsl51_model.keras"
DEFAULT_LABELS = "tsl51_labels.json"
DEFAULT_SCALER = "tsl51_scaler.pkl"
NO_HAND_RESET_S = 0.5
CAPTION_MAX_WORDS = 3
PREDICT_COOLDOWN_S = 0.35


def _extract_hand_coords(results) -> np.ndarray | None:
    """Return fixed-size (126,) array of left+right hand xyz; zero-fill missing hands.

    Using a fixed size ensures ``prev.shape == curr.shape`` even when the number
    of detected hands changes between frames (1↔2), which would otherwise cause
    ``_mean_hand_displacement`` to return 0.0 and block every prediction.
    """
    left = getattr(results, "left_hand_landmarks", None)
    right = getattr(results, "right_hand_landmarks", None)
    if left is None and right is None:
        return None
    out = np.zeros(126, dtype=np.float32)
    for slot, hand in enumerate((left, right)):
        if hand is not None:
            base = slot * 63
            for j, lm in enumerate(hand.landmark):
                out[base + j * 3] = lm.x
                out[base + j * 3 + 1] = lm.y
                out[base + j * 3 + 2] = lm.z
    return out


def _mean_hand_displacement(prev: np.ndarray | None, curr: np.ndarray | None) -> float:
    if prev is None or curr is None or prev.shape != curr.shape:
        return 0.0
    n_pts = prev.shape[0] // 3
    p = prev.reshape(n_pts, 3)
    c = curr.reshape(n_pts, 3)
    return float(np.mean(np.linalg.norm(c - p, axis=1)))


def scale_sequence(seq: np.ndarray, scaler) -> np.ndarray:
    """Apply StandardScaler per 162-D frame."""
    return scaler.transform(seq.astype(np.float32)).astype(np.float32)


def _format_caption(recent_words: list[tuple[str, float]]) -> str:
    if not recent_words:
        return ""
    return " · ".join(f"{word} ({confidence * 100:.0f}%)" for word, confidence in recent_words)


def parse_args() -> argparse.Namespace:
    desc = (
        "Thai word sign live demo (TSL-51, CC BY-NC-SA 4.0 — non-commercial only)"
    )
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--labels", default=DEFAULT_LABELS)
    p.add_argument("--scaler", default=DEFAULT_SCALER)
    p.add_argument("--cam", type=int, default=0, help="Webcam index")
    p.add_argument("--seq-len", type=int, default=SEQ_LEN_DEFAULT)
    p.add_argument("--threshold", type=float, default=0.55,
                   help="Minimum smoothed top-1 probability to show a word")
    p.add_argument("--smoothing-alpha", type=float, default=0.4,
                   help="EMA weight for new predictions; 1.0 = no smoothing")
    p.add_argument("--motion-min", type=float, default=0.008,
                   help="Min mean hand L2 displacement to trigger prediction")
    p.add_argument("--save-log", type=str, default=None,
                   help="Path to CSV: ts_iso,label,confidence")
    p.add_argument("--use-tflite", action="store_true", default=False,
                   help="Force TFLite backend; otherwise auto-detect")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.seq_len = max(1, int(args.seq_len))
    if args.smoothing_alpha < 0.05:
        args.smoothing_alpha = 0.05
    if args.smoothing_alpha > 1.0:
        args.smoothing_alpha = 1.0

    print("Loading model artifacts …")
    try:
        predictor, labels, scaler = load_demo_artifacts(
            args.model,
            args.labels,
            args.scaler,
            args.use_tflite,
            expected_feature_dim=FEATURE_DIM,
            expected_sequence_len=args.seq_len,
            track="tsl51_word_signs",
            manifest_path=Path(__file__).parent / "tsl51_model_manifest.json",
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        sys.exit(f"[ERROR] {exc}")

    print(f"  Classes: {len(labels)}")
    print(
        f"  seq_len={args.seq_len}  threshold={args.threshold}  "
        f"alpha={args.smoothing_alpha}  motion_min={args.motion_min}"
    )

    font_root = Path(__file__).parent
    font_big = load_thai_font(font_root, 64)
    font_med = load_thai_font(font_root, 28)
    font_sml = load_thai_font(font_root, 22)
    smoother = EMABuffer(args.smoothing_alpha)
    seq_buf = SequenceBuffer(seq_len=args.seq_len, feature_dim=FEATURE_DIM)

    csv_file = None
    csv_writer = None
    if args.save_log:
        csv_file, csv_writer = open_csv_writer(
            args.save_log,
            ["timestamp_iso", "label", "confidence"],
        )

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    pose_spec = mp_drawing.DrawingSpec(color=(80, 80, 255), thickness=1, circle_radius=2)
    hand_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
    hand_conn_spec = mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        if csv_file is not None:
            csv_file.close()
        sys.exit(f"[ERROR] Cannot open webcam index {args.cam}")

    print("Webcam open. Press  q  to quit.")

    display_word = ""
    overlay_text = "ไม่พบมือ / no hands"
    caption_text = ""
    confidence = 0.0
    recent_words: list[tuple[str, float]] = []
    prev_hand_coords: np.ndarray | None = None
    last_hand_time = time.monotonic()
    last_good_frame_time = time.monotonic()
    last_prediction_time = 0.0

    try:
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
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

                # Run holistic on the ORIGINAL (unflipped) frame so that landmark
                # coordinates and left/right hand assignment match the training data
                # convention.  Flip only the *display* frame so the user sees a
                # natural mirror view.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = holistic.process(rgb)
                rgb.flags.writeable = True

                # Draw landmarks on the unflipped frame, then flip for display.
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        pose_spec,
                        pose_spec,
                    )
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        hand_spec,
                        hand_conn_spec,
                    )
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        hand_spec,
                        hand_conn_spec,
                    )
                # Mirror for display (natural selfie view) — done AFTER inference.
                frame = cv2.flip(frame, 1)

                feat = extract_holistic_frame(results)
                hand_coords = _extract_hand_coords(results)
                now = time.monotonic()

                if feat is None:
                    if now - last_hand_time > NO_HAND_RESET_S:
                        seq_buf.reset()
                        smoother.reset()
                        prev_hand_coords = None
                    display_word = ""
                    overlay_text = "ไม่พบมือ / no hands"
                    confidence = 0.0
                    caption_text = _format_caption(recent_words)
                else:
                    last_hand_time = now
                    seq_buf.push(feat)
                    motion = _mean_hand_displacement(prev_hand_coords, hand_coords)
                    prev_hand_coords = hand_coords

                    if seq_buf.is_full() and motion >= args.motion_min and (
                        now - last_prediction_time >= PREDICT_COOLDOWN_S
                    ):
                        last_prediction_time = now
                        seq = seq_buf.get_padded()
                        seq_scaled = scale_sequence(seq, scaler)
                        raw_probs = predictor.predict(seq_scaled.reshape(1, args.seq_len, FEATURE_DIM))
                        smoothed = smoother.update(raw_probs)

                        pred_idx = int(np.argmax(smoothed))
                        confidence = float(smoothed[pred_idx])
                        if confidence >= args.threshold:
                            display_word = labels.get(str(pred_idx), "?")
                            overlay_text = f"{display_word}  ({confidence * 100:.0f}%)"
                            recent_words.append((display_word, confidence))
                            if len(recent_words) > CAPTION_MAX_WORDS:
                                recent_words.pop(0)
                        else:
                            display_word = "?"
                            overlay_text = f"?  ({confidence * 100:.0f}%)"

                        caption_text = _format_caption(recent_words)
                        if csv_writer is not None and display_word and display_word != "?":
                            csv_writer.writerow([iso_timestamp_ms(), display_word, f"{confidence:.4f}"])
                            csv_file.flush()

                h, w = frame.shape[:2]
                banner_h = 160
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

                if display_word and display_word != "?":
                    frame = draw_thai_text(
                        frame,
                        display_word,
                        (20, h - banner_h + 5),
                        font_big,
                        color=(0, 255, 120),
                    )
                if caption_text:
                    frame = draw_thai_text(
                        frame,
                        caption_text,
                        (20, h - 78),
                        font_sml,
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

                cv2.putText(
                    frame,
                    f"buf {len(seq_buf)}/{args.seq_len}",
                    (w - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )

                cv2.imshow("Thai Word Signs (TSL-51) — press q to quit", frame)
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
