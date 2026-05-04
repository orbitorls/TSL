"""Quality assessment for TSL-51 inference."""

import numpy as np


class LandmarkQuality:
    @staticmethod
    def score_landmark_quality(landmarks: np.ndarray) -> float:
        """Score landmark quality (0-1)."""
        missing = np.sum(landmarks == 0) / len(landmarks)
        has_nan = np.any(np.isnan(landmarks))
        return max(0.0, 1.0 - missing - (0.5 if has_nan else 0))

    @staticmethod
    def detect_frame_drops(landmarks_sequence) -> list:
        """Detect missing/dropped frames."""
        drops = []
        for i, lm in enumerate(landmarks_sequence):
            if lm is None or len(lm) == 0:
                drops.append(i)
        return drops
