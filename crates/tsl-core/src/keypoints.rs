//! Hand keypoint extraction and normalization (port of `src/keypoints.py`).

pub const FEATURE_SIZE: usize = 63;
pub const NUM_LANDMARKS: usize = 21;
pub const WRIST_IDX: usize = 0;

/// One hand detection: 21 landmarks × (x, y, z).
#[derive(Debug, Clone)]
pub struct HandLandmarks {
    pub coords: [f32; FEATURE_SIZE],
}

/// MediaPipe Hands–style results (duck-typed).
#[derive(Debug, Default)]
pub struct HandResults {
    pub hands: Vec<HandLandmarks>,
    /// Parallel handedness scores; empty → use index 0.
    pub handedness_scores: Vec<f32>,
}

/// Pull raw (x, y, z) for the highest-confidence hand.
pub fn extract_hand_landmarks(results: &HandResults) -> Option<[f32; FEATURE_SIZE]> {
    if results.hands.is_empty() {
        return None;
    }
    let best_idx = if results.handedness_scores.len() > 1 {
        results
            .handedness_scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    } else {
        0
    };
    Some(results.hands[best_idx].coords)
}

/// Wrist-relative + scale by distance wrist → landmark 9.
pub fn normalize_landmarks(arr: &[f32; FEATURE_SIZE]) -> Option<[f32; FEATURE_SIZE]> {
    let mut pts = [[0.0f32; 3]; NUM_LANDMARKS];
    for (i, chunk) in arr.chunks_exact(3).enumerate().take(NUM_LANDMARKS) {
        pts[i] = [chunk[0], chunk[1], chunk[2]];
    }

    let wrist = pts[WRIST_IDX];
    for p in &mut pts {
        p[0] -= wrist[0];
        p[1] -= wrist[1];
        p[2] -= wrist[2];
    }

    let mid = pts[9];
    let hand_span = (mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2]).sqrt();
    if hand_span < 1e-6 {
        return None;
    }

    let mut out = [0.0f32; FEATURE_SIZE];
    for (i, p) in pts.iter().enumerate() {
        let base = i * 3;
        out[base] = p[0] / hand_span;
        out[base + 1] = p[1] / hand_span;
        out[base + 2] = p[2] / hand_span;
    }
    Some(out)
}

/// Extract then normalize.
pub fn extract_and_normalize(results: &HandResults) -> Option<[f32; FEATURE_SIZE]> {
    let raw = extract_hand_landmarks(results)?;
    normalize_landmarks(&raw)
}
