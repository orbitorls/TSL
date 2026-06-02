//! Per-frame landmark outputs (TSL feature contract).

pub const HAND_FRAME_DIM: usize = 63;
pub const HOLISTIC_FRAME_DIM: usize = 162;

/// Normalized hand landmarks (21×3), matching `tsl_core::normalize_landmarks`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HandFrame {
    pub landmarks: [f32; HAND_FRAME_DIM],
}

impl HandFrame {
    pub fn new(landmarks: [f32; HAND_FRAME_DIM]) -> Self {
        Self { landmarks }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.landmarks
    }
}

/// Shoulder-anchored holistic frame, matching `tsl_core::extract_holistic_frame`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HolisticFrame {
    pub landmarks: [f32; HOLISTIC_FRAME_DIM],
}

impl HolisticFrame {
    pub fn new(landmarks: [f32; HOLISTIC_FRAME_DIM]) -> Self {
        Self { landmarks }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.landmarks
    }
}

impl From<[f32; HAND_FRAME_DIM]> for HandFrame {
    fn from(landmarks: [f32; HAND_FRAME_DIM]) -> Self {
        Self::new(landmarks)
    }
}

impl From<[f32; HOLISTIC_FRAME_DIM]> for HolisticFrame {
    fn from(landmarks: [f32; HOLISTIC_FRAME_DIM]) -> Self {
        Self::new(landmarks)
    }
}
