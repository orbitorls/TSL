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
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover - import-only tests may not install OpenCV
    cv2 = None  # type: ignore[assignment]
try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - import-only tests may not install MediaPipe
    mp = None  # type: ignore[assignment]
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))
from sequence_keypoints import FEATURE_DIM, SEQ_LEN_DEFAULT
from tsl_translate.inference import InferenceSettings, process_rgb_frame
from tsl_translate.registry import ModelRegistry
from tsl_translate.session import LoadedModel, MediaPipeRuntime, PredictService
from tsl_translate.tracks import TRACKS
from webcam_runtime import (
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
TSL51_TRACK = TRACKS["tsl51"]
CAPTION_MAX_WORDS = 3
SIGN_END_FRAMES_DEFAULT = 8
MIN_SIGN_FRAMES_DEFAULT = 15


def _format_caption(recent_words: list[tuple[str, float]]) -> str:
    if not recent_words:
        return ""
    return " · ".join(f"{word} ({confidence * 100:.0f}%)" for word, confidence in recent_words)


def resolve_tsl51_artifacts(args: argparse.Namespace) -> tuple[Path, Path, Path, Path | None]:
    track = TSL51_TRACK
    repo_root = Path(__file__).resolve().parents[1]
    explicit = args.model is not None or args.labels is not None or args.scaler is not None
    if explicit:
        if args.model is None or args.labels is None or args.scaler is None:
            raise ValueError("When overriding artifacts, pass --model, --labels, and --scaler together")
        model = Path(args.model)
        labels = Path(args.labels)
        scaler = Path(args.scaler)
        manifest = model.parent / track.manifest
        return model, labels, scaler, manifest if manifest.exists() else None

    if args.artifact_dir is not None:
        artifact_dir = Path(args.artifact_dir)
        model = artifact_dir / track.default_model
        if not model.exists():
            model = model.with_suffix(".tflite")
        labels = artifact_dir / track.default_labels
        scaler = artifact_dir / track.default_scaler
        manifest = artifact_dir / track.manifest
        return model, labels, scaler, manifest if manifest.exists() else None

    candidates = ModelRegistry(repo_root).discover(track)
    if candidates:
        chosen = candidates[0]
        print(f"Selected artifact set: {chosen.name}")
        return chosen.model, chosen.labels, chosen.scaler, chosen.manifest

    artifact_dir = repo_root / "artifacts" / track.key
    raise FileNotFoundError(
        "No valid TSL51 artifacts found. Expected "
        f"{artifact_dir / track.default_model}, {artifact_dir / track.default_labels}, "
        f"and {artifact_dir / track.default_scaler}."
    )


def parse_args() -> argparse.Namespace:
    desc = (
        "Thai word sign live demo (TSL-51, CC BY-NC-SA 4.0 — non-commercial only)"
    )
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--model", default=None)
    p.add_argument("--labels", default=None)
    p.add_argument("--scaler", default=None)
    p.add_argument("--artifact-dir", default=None, help="Directory containing TSL51 model/labels/scaler")
    p.add_argument("--cam", type=int, default=0, help="Webcam index")
    p.add_argument("--seq-len", type=int, default=SEQ_LEN_DEFAULT)
    p.add_argument("--threshold", type=float, default=0.55,
                   help="Minimum smoothed top-1 probability to show a word")
    p.add_argument("--smoothing-alpha", type=float, default=0.4,
                   help="EMA weight for new predictions; 1.0 = no smoothing")
    p.add_argument("--motion-min", type=float, default=0.008,
                   help="Min mean hand L2 displacement to trigger prediction")
    p.add_argument("--top-k", type=int, default=3, help="Number of debug top-k predictions")
    p.add_argument("--min-stable-preds", type=int, default=2,
                   help="Repeated high-confidence predictions needed before showing a label")
    p.add_argument("--sign-end-frames", type=int, default=SIGN_END_FRAMES_DEFAULT,
                   help="Consecutive low-motion frames to mark end-of-sign (default: 8)")
    p.add_argument("--min-sign-frames", type=int, default=MIN_SIGN_FRAMES_DEFAULT,
                   help="Min detected frames in a sign segment to attempt prediction (default: 15)")
    p.add_argument("--debug-overlay", action="store_true", help="Show top-k and motion diagnostics")
    p.add_argument("--save-log", type=str, default=None,
                   help="Path to CSV: ts_iso,label,confidence")
    p.add_argument("--use-tflite", action="store_true", default=False,
                   help="Force TFLite backend; otherwise auto-detect")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if cv2 is None:
        raise ImportError("opencv-python is required for webcam_word_demo.py")
    if mp is None:
        raise ImportError("mediapipe is required for webcam_word_demo.py")
    args.seq_len = max(1, int(args.seq_len))
    if args.smoothing_alpha < 0.05:
        args.smoothing_alpha = 0.05
    if args.smoothing_alpha > 1.0:
        args.smoothing_alpha = 1.0
    args.top_k = max(1, int(args.top_k))
    args.min_stable_preds = max(1, int(args.min_stable_preds))

    print("Loading model artifacts …")
    try:
        model_path, labels_path, scaler_path, manifest_path = resolve_tsl51_artifacts(args)
        predictor, labels, scaler = load_demo_artifacts(
            model_path,
            labels_path,
            scaler_path,
            args.use_tflite,
            expected_feature_dim=FEATURE_DIM,
            expected_sequence_len=args.seq_len,
            expected_num_classes=TSL51_TRACK.expected_num_classes,
            track="tsl51_word_signs",
            manifest_path=manifest_path,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        sys.exit(f"[ERROR] {exc}")

    print(f"  Model: {model_path}")
    print(f"  Labels: {labels_path}")
    print(f"  Scaler: {scaler_path}")
    print(f"  Backend: {predictor.kind}")
    print(f"  Classes: {len(labels)}")
    print(
        f"  seq_len={args.seq_len}  threshold={args.threshold}  "
        f"alpha={args.smoothing_alpha}  motion_min={args.motion_min}  top_k={args.top_k}"
    )

    font_root = Path(__file__).parent
    font_big = load_thai_font(font_root, 64)
    font_med = load_thai_font(font_root, 28)
    font_sml = load_thai_font(font_root, 22)
    loaded = LoadedModel(
        predictor=predictor,
        labels=labels,
        scaler=scaler,
        backend=predictor.kind,
        model_path=model_path,
        labels_path=labels_path,
        scaler_path=scaler_path,
        load_time_ms=0.0,
    )
    service = PredictService(TSL51_TRACK, alpha=args.smoothing_alpha)
    settings = InferenceSettings(
        threshold=args.threshold,
        alpha=args.smoothing_alpha,
        top_k=args.top_k,
        motion_min=args.motion_min,
        sign_end_frames=args.sign_end_frames,
        min_sign_frames=args.min_sign_frames,
    )

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
    committed_word = ""
    last_good_frame_time = 0.0
    last_topk_text = ""
    state_text = "buffering"
    active_result = None

    mp_runtime = MediaPipeRuntime()
    try:
        while True:
                ok, frame = read_frame_with_retry(cap)
                if not ok or frame is None:
                    now = service.last_hand_time or 0.0
                    if now - last_good_frame_time > 2.0:
                        print("[WARN] No frames for >2s — attempting webcam reconnect …")
                        cap.release()
                        cap = cv2.VideoCapture(args.cam)
                        last_good_frame_time = now
                        if not cap.isOpened():
                            print("[ERROR] Reconnect failed; exiting.")
                            break
                    continue
                last_good_frame_time = service.last_hand_time or 0.0

                # Run holistic on the ORIGINAL (unflipped) frame so that landmark
                # coordinates and left/right hand assignment match the training data
                # convention.  Flip only the *display* frame so the user sees a
                # natural mirror view.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = mp_runtime.holistic.process(rgb)
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

                active_result = process_rgb_frame(
                    TSL51_TRACK,
                    loaded,
                    service,
                    mp_runtime,
                    rgb,
                    settings,
                )
                confidence = active_result.confidence
                display_word = active_result.label if active_result.label not in ("?", "ไม่พบมือ", "ไม่พบมือ / no hands") else ""
                overlay_text = active_result.label or "ไม่พบมือ / no hands"
                state_text = active_result.status
                last_topk_text = active_result.topk_text

                if active_result.committed_label:
                    committed_word = active_result.committed_label
                    recent_words.append((committed_word, confidence))
                    if len(recent_words) > CAPTION_MAX_WORDS:
                        recent_words.pop(0)
                    if csv_writer is not None:
                        csv_writer.writerow([iso_timestamp_ms(), committed_word, f"{confidence:.4f}"])
                        csv_file.flush()
                caption_text = _format_caption(recent_words)

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
                    f"buf {len(service.seq_buf)}/{args.seq_len}",
                    (w - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )
                if args.debug_overlay:
                    _state_color = (0, 255, 80) if state_text in {"signing", "previewing", "predicted"} else (220, 220, 220)
                    cv2.putText(
                        frame,
                        f"state {state_text}",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        _state_color,
                        1,
                    )
                    cv2.putText(
                        frame,
                        f"sign {len(service.sign_frames)}/{args.min_sign_frames}",
                        (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 80) if service.sign_frames else (200, 200, 200),
                        1,
                    )
                    cv2.putText(
                        frame,
                        f"top {last_topk_text}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (220, 220, 220),
                        1,
                    )

                cv2.imshow("Thai Word Signs (TSL-51) — press q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        mp_runtime.close()
        cap.release()
        cv2.destroyAllWindows()
        if csv_file is not None:
            try:
                csv_file.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
