//! Camera + landmark extraction abstraction.
//!
//! Model paths (gitignored): see `README.md` under `artifacts/vision/`.

mod backend;
mod frame;
mod onnx;
mod stub;

pub use backend::{FrameInput, LandmarkBackend, VisionError};
pub use frame::{HandFrame, HolisticFrame, HAND_FRAME_DIM, HOLISTIC_FRAME_DIM};
pub use onnx::{OnnxLandmarkBackend, OnnxModelPaths};
pub use stub::{InjectedStubBackend, StubBackend};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_returns_none() {
        let mut backend = StubBackend;
        let buf = [0u8; 12];
        let frame = FrameInput::new(2, 2, &buf);
        assert!(backend.detect_hand(&frame).unwrap().is_none());
        assert!(backend.detect_holistic(&frame).unwrap().is_none());
    }

    #[test]
    fn frame_dims_match_core() {
        assert_eq!(HAND_FRAME_DIM, tsl_core::FEATURE_SIZE);
        assert_eq!(HOLISTIC_FRAME_DIM, tsl_core::FEATURE_DIM);
    }
}
