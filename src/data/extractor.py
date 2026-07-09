"""
tsl_tasks_extractor.py — Compatibility shim for video/camera inference scripts.

Provides MediaPipe landmark extraction utilities used by predict_video.py,
camera_translate.py, and translate.py.  Built on top of the canonical
src.data.feature_extraction module so that feature computation stays in sync
with training.

Requirements:
    pip install mediapipe opencv-python
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .feature_extraction import (
    FEATURE_DIMS,
    pad_or_truncate,
    sample_frames_uniform,
)

LANDMARK_VECTOR_DIM = FEATURE_DIMS["basic"]

# Normalization std floor — must match training pipeline
NORMALIZATION_STD_FLOOR = 1e-8

# ─────────────────────────────────────────────────────────────
# Normalization helpers
# ─────────────────────────────────────────────────────────────


def normalize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score normalize features using pre-computed statistics.

    Args:
        features: Raw feature array, shape (feature_dim,) or (batch, feature_dim).
        mean: Per-feature mean, shape (feature_dim,).
        std:  Per-feature std,  shape (feature_dim,).

    Returns:
        Normalized features with the same shape as input.
    """
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    # Same floor used in training normalization
    std = np.where(std == 0, 1.0, std) + NORMALIZATION_STD_FLOOR
    return (np.asarray(features, dtype=np.float32) - mean) / std  # type: ignore[no-any-return]


def report_extractor_compatibility(checkpoint: dict) -> None:
    """Print a brief compatibility note about a model checkpoint.

    Args:
        checkpoint: Raw dict loaded from a .pt file.
    """
    # Detect legacy vs current schema
    is_legacy = "labels" in checkpoint and "label_to_idx" in checkpoint
    schema = "legacy (labels/label_to_idx)" if is_legacy else "current (classes)"
    model_type = checkpoint.get("model", checkpoint.get("config", {}).get("model", "unknown"))
    accuracy = checkpoint.get("accuracy")
    acc_str = f"{accuracy * 100:.2f}%" if accuracy is not None else "unknown"
    print(
        f"[tsl_tasks_extractor] checkpoint schema={schema}, model={model_type}, accuracy={acc_str}"
    )


# ─────────────────────────────────────────────────────────────
# Feature extraction from frame buffers
# ─────────────────────────────────────────────────────────────

_POSE_BASES = [
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "lbrow_outer",
    "lbrow_inner",
    "rbrow_inner",
    "rbrow_outer",
    "mouth_right",
    "mouth_left",
]


_BASIC_KEYS = (
    [f"lh_{c}{i}" for i in range(21) for c in ("x", "y", "z")]
    + [f"rh_{c}{i}" for i in range(21) for c in ("x", "y", "z")]
    + [f"{base}_{c}" for base in _POSE_BASES for c in ("x", "y", "z")]
)


def extract_sequence_features(
    frames: list,
    feature_level: str = "basic",
    target_frames: int = 30,
) -> np.ndarray | None:
    """Build a fixed-length temporal sequence from per-frame landmarks.

    Returns an array shaped (target_frames, feature_dim).
    """
    if not frames:
        return None

    feature_dim = FEATURE_DIMS.get(feature_level, 162)
    seq = []
    for frame in frames:
        vec = _frame_dict_to_vector(frame, feature_level, feature_dim)
        if vec is not None:
            seq.append(vec)

    if not seq:
        return None

    sampled = sample_frames_uniform(seq, target_frames=target_frames)
    fixed = pad_or_truncate(sampled, target_length=target_frames, pad_value=0.0)
    return np.asarray(fixed, dtype=np.float32)


class FrameExtractionResult:
    """Compatibility result object for older camera_translate expectations."""

    def __init__(self, landmarks: dict | None, detected: bool):
        self.landmarks = landmarks
        self.detected = detected


def draw_debug_overlay(frame: np.ndarray, result: FrameExtractionResult) -> np.ndarray:
    """Draw a minimal detection status overlay and return frame."""
    try:
        import cv2
    except Exception:
        return frame

    text = "Landmarks: ON" if result and result.detected else "Landmarks: OFF"
    color = (0, 220, 0) if result and result.detected else (0, 0, 220)
    cv2.rectangle(frame, (8, 60), (180, 88), (20, 20, 20), -1)
    cv2.rectangle(frame, (8, 60), (180, 88), color, 1)
    cv2.putText(frame, text, (14, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return frame


def extract_features(frames: list, feature_level: str = "basic") -> np.ndarray | None:
    """Compute mean-aggregated features from a list of landmark dicts.

    Each element of *frames* must be a dict with keys like ``lh_x0``, ``rh_y3``,
    ``l_shoulder_x``, etc.  Missing keys default to 0.0.

    Args:
        frames: List of per-frame landmark dicts (from MediaPipe extraction).
        feature_level: One of 'basic' (162), 'finger' (252), 'full' (1596).

    Returns:
        Float32 array of shape (feature_dim,), or *None* if *frames* is empty.
    """
    if not frames:
        return None

    feature_dim = FEATURE_DIMS.get(feature_level, 162)
    accumulator = np.zeros(feature_dim, dtype=np.float64)
    count = 0

    for frame in frames:
        vec = _frame_dict_to_vector(frame, feature_level, feature_dim)
        if vec is not None:
            accumulator += vec
            count += 1

    if count == 0:
        return None
    return (accumulator / count).astype(np.float32)


def _frame_dict_to_vector(frame: dict, _feature_level: str, feature_dim: int) -> np.ndarray | None:
    """Convert a single landmark dict to a feature vector."""
    if _feature_level == "basic":
        feats = [float(frame.get(k, 0.0)) for k in _BASIC_KEYS]
    else:
        feats: list[float] = []

        # Left hand (63)
        for i in range(21):
            for c in ("x", "y", "z"):
                feats.append(float(frame.get(f"lh_{c}{i}", 0.0)))

        # Right hand (63)
        for i in range(21):
            for c in ("x", "y", "z"):
                feats.append(float(frame.get(f"rh_{c}{i}", 0.0)))

        # Pose (36)
        for base in _POSE_BASES:
            for c in ("x", "y", "z"):
                feats.append(float(frame.get(f"{base}_{c}", 0.0)))

    if len(feats) < feature_dim:
        feats.extend([0.0] * (feature_dim - len(feats)))

    return np.array(feats[:feature_dim], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Video landmark extraction
# ─────────────────────────────────────────────────────────────


def extract_video_landmarks(
    video_path: str,
    extractor: MediaPipeTasksLandmarkExtractor,
    verbose: bool = True,
) -> tuple[list, dict]:
    """Extract per-frame landmarks from a video file.

    Args:
        video_path: Path to video file.
        extractor: Initialized MediaPipeTasksLandmarkExtractor instance.
        verbose: Print progress info.

    Returns:
        Tuple of (frames, stats) where *frames* is a list of landmark dicts
        and *stats* is a dict with frame counts.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required. Install with: pip install opencv-python")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames: list[dict] = []
    total_frames = 0
    detected_frames = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        total_frames += 1

        landmarks = extractor.extract(frame_bgr)
        if landmarks is not None:
            frames.append(landmarks)
            detected_frames += 1

    cap.release()

    stats = {
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "detection_rate": detected_frames / max(total_frames, 1),
    }

    if verbose:
        print(
            f"  {detected_frames}/{total_frames} frames with landmarks ({stats['detection_rate'] * 100:.1f}%)"
        )

    return frames, stats


# ─────────────────────────────────────────────────────────────
# MediaPipe Tasks landmark extractor
# ─────────────────────────────────────────────────────────────


class MediaPipeTasksLandmarkExtractor:
    """Extracts 162-dim hand + pose landmarks from BGR frames using MediaPipe.

    Uses the Holistic or legacy solution API depending on the installed version.

    Args:
        model_complexity: MediaPipe model complexity (0-2).
        min_detection_confidence: Minimum hand/pose detection confidence.
        min_tracking_confidence: Minimum landmark tracking confidence.
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError("mediapipe is required. Install with: pip install mediapipe")

        import cv2  # noqa: F401 (ensure cv2 available before use)

        self._mp = mp
        self._solutions = mp.solutions
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, frame_bgr: Any) -> dict | None:
        """Extract landmarks from a single BGR frame.

        Args:
            frame_bgr: OpenCV BGR frame (numpy array).

        Returns:
            Dict of landmark values, or *None* if no hands/pose detected.
        """
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._holistic.process(rgb)

        # Build dense basic feature dict so downstream code sees consistent length.
        landmarks: dict[str, float] = dict.fromkeys(_BASIC_KEYS, 0.0)
        detected = False

        # Left hand
        if result.left_hand_landmarks:
            detected = True
            for i, lm in enumerate(result.left_hand_landmarks.landmark):
                landmarks[f"lh_x{i}"] = lm.x
                landmarks[f"lh_y{i}"] = lm.y
                landmarks[f"lh_z{i}"] = lm.z

        # Right hand
        if result.right_hand_landmarks:
            detected = True
            for i, lm in enumerate(result.right_hand_landmarks.landmark):
                landmarks[f"rh_x{i}"] = lm.x
                landmarks[f"rh_y{i}"] = lm.y
                landmarks[f"rh_z{i}"] = lm.z

        # Pose (subset of 12 key points)
        _POSE_IDX = {
            "l_shoulder": 11,
            "r_shoulder": 12,
            "l_elbow": 13,
            "r_elbow": 14,
            "l_wrist": 15,
            "r_wrist": 16,
        }
        if result.pose_landmarks:
            detected = True
            lms = result.pose_landmarks.landmark
            for name, idx in _POSE_IDX.items():
                landmarks[f"{name}_x"] = lms[idx].x
                landmarks[f"{name}_y"] = lms[idx].y
                landmarks[f"{name}_z"] = lms[idx].z

        # Face landmarks for brow/mouth (approximate indices)
        _FACE_IDX = {
            "lbrow_outer": 70,
            "lbrow_inner": 107,
            "rbrow_inner": 336,
            "rbrow_outer": 300,
            "mouth_right": 61,
            "mouth_left": 291,
        }
        if result.face_landmarks:
            flms = result.face_landmarks.landmark
            for name, idx in _FACE_IDX.items():
                if idx < len(flms):
                    landmarks[f"{name}_x"] = flms[idx].x
                    landmarks[f"{name}_y"] = flms[idx].y
                    landmarks[f"{name}_z"] = flms[idx].z

        return landmarks if detected else None

    def extract_frame(self, frame_bgr: Any) -> FrameExtractionResult:
        """Compatibility wrapper expected by camera_translate.py."""
        landmarks = self.extract(frame_bgr)
        return FrameExtractionResult(landmarks=landmarks, detected=landmarks is not None)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._holistic.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
