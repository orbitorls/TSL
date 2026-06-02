//! ONNX landmark backend skeleton (`--features onnx` for future `ort` sessions).

use std::path::{Path, PathBuf};

use crate::backend::{FrameInput, LandmarkBackend, VisionError};
use crate::frame::{HandFrame, HolisticFrame};

/// ONNX model files on disk (see crate `README.md`).
#[derive(Debug, Clone)]
pub struct OnnxModelPaths {
    pub hand_landmarker: PathBuf,
    pub pose_landmarker: PathBuf,
    pub face_landmarker: PathBuf,
}

impl OnnxModelPaths {
    pub fn from_repo_root(root: impl AsRef<Path>) -> Self {
        let vision = root.as_ref().join("artifacts").join("vision");
        Self {
            hand_landmarker: vision.join("hand_landmarker.onnx"),
            pose_landmarker: vision.join("pose_landmarker.onnx"),
            face_landmarker: vision.join("face_landmarker.onnx"),
        }
    }

    pub fn validate_exists(&self) -> Result<(), VisionError> {
        for path in [
            &self.hand_landmarker,
            &self.pose_landmarker,
            &self.face_landmarker,
        ] {
            if !path.is_file() {
                return Err(VisionError::ModelNotFound(path.clone()));
            }
        }
        Ok(())
    }
}

/// Loads model paths; ONNX Runtime sessions are not wired yet.
pub struct OnnxLandmarkBackend {
    paths: OnnxModelPaths,
}

impl OnnxLandmarkBackend {
    pub fn open(paths: OnnxModelPaths) -> Result<Self, VisionError> {
        paths.validate_exists()?;
        Ok(Self { paths })
    }

    pub fn paths(&self) -> &OnnxModelPaths {
        &self.paths
    }

    #[cfg(feature = "onnx")]
    fn run_hand_session(
        &mut self,
        _frame: &FrameInput<'_>,
    ) -> Result<Option<HandFrame>, VisionError> {
        // TODO: preprocess RGB → tensor, run hand_landmarker, map to tsl_core::HandResults,
        // then extract_and_normalize.
        let _ = &self.paths.hand_landmarker;
        Err(VisionError::NotReady("hand ONNX session not wired".into()))
    }

    #[cfg(not(feature = "onnx"))]
    fn run_hand_session(
        &mut self,
        _frame: &FrameInput<'_>,
    ) -> Result<Option<HandFrame>, VisionError> {
        let _ = self;
        Err(VisionError::OnnxFeatureDisabled)
    }

    #[cfg(feature = "onnx")]
    fn run_holistic_session(
        &mut self,
        _frame: &FrameInput<'_>,
    ) -> Result<Option<HolisticFrame>, VisionError> {
        // TODO: pose + face + LH/RH models → tsl_core::HolisticResults → extract_holistic_frame.
        Err(VisionError::NotReady(
            "holistic ONNX session not wired".into(),
        ))
    }

    #[cfg(not(feature = "onnx"))]
    fn run_holistic_session(
        &mut self,
        _frame: &FrameInput<'_>,
    ) -> Result<Option<HolisticFrame>, VisionError> {
        let _ = self;
        Err(VisionError::OnnxFeatureDisabled)
    }
}

impl LandmarkBackend for OnnxLandmarkBackend {
    fn detect_hand(&mut self, frame: &FrameInput<'_>) -> Result<Option<HandFrame>, VisionError> {
        frame.validate()?;
        self.run_hand_session(frame)
    }

    fn detect_holistic(
        &mut self,
        frame: &FrameInput<'_>,
    ) -> Result<Option<HolisticFrame>, VisionError> {
        frame.validate()?;
        self.run_holistic_session(frame)
    }
}
