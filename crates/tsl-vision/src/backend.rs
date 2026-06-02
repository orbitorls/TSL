//! `LandmarkBackend` trait and shared frame input.

use thiserror::Error;

use crate::frame::{HandFrame, HolisticFrame};

#[derive(Debug, Error)]
pub enum VisionError {
    #[error("invalid frame dimensions")]
    InvalidFrame,
    #[error("no hand detected")]
    NoHand,
    #[error("no holistic landmarks")]
    NoHolistic,
    #[error("model not found: {0}")]
    ModelNotFound(std::path::PathBuf),
    #[error("ONNX feature disabled; rebuild with --features onnx")]
    OnnxFeatureDisabled,
    #[error("backend not ready: {0}")]
    NotReady(String),
}

/// RGB frame reference (row-major, 3 bytes per pixel).
pub struct FrameInput<'a> {
    pub width: u32,
    pub height: u32,
    pub rgb: &'a [u8],
}

impl<'a> FrameInput<'a> {
    pub fn new(width: u32, height: u32, rgb: &'a [u8]) -> Self {
        Self { width, height, rgb }
    }

    pub fn validate(&self) -> Result<(), VisionError> {
        let expected = (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|n| n.checked_mul(3));
        match expected {
            Some(n) if n == self.rgb.len() => Ok(()),
            _ => Err(VisionError::InvalidFrame),
        }
    }
}

pub trait LandmarkBackend: Send {
    fn detect_hand(&mut self, frame: &FrameInput<'_>) -> Result<Option<HandFrame>, VisionError>;
    fn detect_holistic(
        &mut self,
        frame: &FrameInput<'_>,
    ) -> Result<Option<HolisticFrame>, VisionError>;
}
