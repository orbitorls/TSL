from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only in minimal test envs
    cv2 = None  # type: ignore[assignment]
try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - exercised only in minimal test envs
    mp = None  # type: ignore[assignment]
import numpy as np

from sequence_keypoints import FEATURE_DIM, SequenceBuffer
from tsl_translate.tracks import TrackSpec
from webcam_runtime import EMABuffer, read_frame_with_retry


@dataclass
class LoadedModel:
    predictor: Any
    labels: dict[str, str]
    scaler: Any
    backend: str
    model_path: Path
    labels_path: Path
    scaler_path: Path
    load_time_ms: float


class PredictService:
    def __init__(self, track: TrackSpec, alpha: float = 0.4) -> None:
        self.track = track
        self.smoother = EMABuffer(alpha)
        self.seq_buf = SequenceBuffer(seq_len=track.expected_seq_len or 1, feature_dim=FEATURE_DIM)
        self.prev_hand_coords: np.ndarray | None = None
        self.last_prediction_time = 0.0
        self.last_hand_time = 0.0
        # Sign-boundary detection state (matches webcam_word_demo.py)
        self.sign_frames: list[np.ndarray] = []
        self.low_motion_count: int = 0
        # Latch: True once a sign has committed a label; reset on next sign start.
        # Prevents the sign-end path and the hands-disappear path from both
        # committing for the same sign when hand detection flickers.
        self.committed_this_sign: bool = False
        self.candidate_label: str | None = None
        self.candidate_count: int = 0
        self.last_preview_time: float = 0.0

    def reset(self) -> None:
        self.smoother.reset()
        self.seq_buf.reset()
        self.prev_hand_coords = None
        self.last_prediction_time = 0.0
        self.last_hand_time = 0.0
        self.sign_frames.clear()
        self.low_motion_count = 0
        self.committed_this_sign = False
        self.candidate_label = None
        self.candidate_count = 0
        self.last_preview_time = 0.0

    def set_alpha(self, alpha: float) -> None:
        alpha = max(0.05, min(1.0, float(alpha)))
        if abs(self.smoother.alpha - alpha) < 1e-9:
            return
        self.smoother = EMABuffer(alpha)


class MediaPipeRuntime:
    """MediaPipe hands/holistic without a local OpenCV capture device."""

    def __init__(self) -> None:
        if mp is None:
            raise ImportError("mediapipe is required for camera inference")
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self) -> None:
        for resource in (self.hands, self.holistic):
            try:
                resource.close()
            except Exception:
                pass


class CameraRuntime(MediaPipeRuntime):
    """Server-side webcam used by the legacy Streamlit app."""

    def __init__(self, cam_index: int) -> None:
        if cv2 is None:
            raise ImportError("opencv-python is required for camera capture")
        super().__init__()
        self.cam_index = cam_index
        self.cap = cv2.VideoCapture(cam_index)
        self.last_ok = time.monotonic()

    def read_rgb_unflipped(self) -> np.ndarray:
        ok, frame = read_frame_with_retry(self.cap)
        if not ok or frame is None:
            raise RuntimeError("อ่านภาพจากกล้องไม่ได้ กรุณาตรวจสอบดัชนีกล้องและสิทธิ์การใช้งาน")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def read_rgb_flipped(self) -> np.ndarray:
        ok, frame = read_frame_with_retry(self.cap)
        if not ok or frame is None:
            raise RuntimeError("อ่านภาพจากกล้องไม่ได้ กรุณาตรวจสอบดัชนีกล้องและสิทธิ์การใช้งาน")
        frame = cv2.flip(frame, 1)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass
        super().close()
