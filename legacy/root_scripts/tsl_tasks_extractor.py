"""Legacy shim for src.data.extractor."""

from src.data.extractor import *  # noqa: F401,F403
from src.data.extractor import (  # noqa: F401
    LANDMARK_VECTOR_DIM,
    NORMALIZATION_STD_FLOOR,
    FrameExtractionResult,
    MediaPipeTasksLandmarkExtractor,
    draw_debug_overlay,
    extract_features,
    extract_sequence_features,
    extract_video_landmarks,
    normalize_features,
    report_extractor_compatibility,
    sample_frames_uniform,
    pad_or_truncate,
)
