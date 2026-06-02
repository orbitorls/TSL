//! Stub backends for CI, golden tests, and offline pipelines.

use tsl_core::keypoints::{extract_and_normalize, HandResults};
use tsl_core::sequence::{extract_holistic_frame, HolisticResults};

use crate::backend::{FrameInput, LandmarkBackend, VisionError};
use crate::frame::{HandFrame, HolisticFrame};

/// Ignores pixels; always returns `None`.
#[derive(Debug, Default, Clone, Copy)]
pub struct StubBackend;

impl LandmarkBackend for StubBackend {
    fn detect_hand(&mut self, frame: &FrameInput<'_>) -> Result<Option<HandFrame>, VisionError> {
        frame.validate()?;
        Ok(None)
    }

    fn detect_holistic(
        &mut self,
        frame: &FrameInput<'_>,
    ) -> Result<Option<HolisticFrame>, VisionError> {
        frame.validate()?;
        Ok(None)
    }
}

/// Injects pre-built `tsl_core` results (golden / unit tests).
#[derive(Debug, Default)]
pub struct InjectedStubBackend {
    pub hand: Option<HandResults>,
    pub holistic: Option<HolisticResults>,
}

impl InjectedStubBackend {
    pub fn from_hand_coords(coords: [f32; 63]) -> Self {
        use tsl_core::keypoints::{HandLandmarks, HandResults};
        Self {
            hand: Some(HandResults {
                hands: vec![HandLandmarks { coords }],
                handedness_scores: vec![1.0],
            }),
            holistic: None,
        }
    }
}

impl LandmarkBackend for InjectedStubBackend {
    fn detect_hand(&mut self, frame: &FrameInput<'_>) -> Result<Option<HandFrame>, VisionError> {
        frame.validate()?;
        let results = self.hand.as_ref().ok_or(VisionError::NoHand)?;
        Ok(extract_and_normalize(results).map(HandFrame::new))
    }

    fn detect_holistic(
        &mut self,
        frame: &FrameInput<'_>,
    ) -> Result<Option<HolisticFrame>, VisionError> {
        frame.validate()?;
        let results = self.holistic.as_ref().ok_or(VisionError::NoHolistic)?;
        Ok(extract_holistic_frame(results).map(HolisticFrame::new))
    }
}
