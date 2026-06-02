//! ONNX inference, EMA smoothing, and artifact loading for TSL webcam demos.

pub mod artifacts;
pub mod ema;
pub mod predictor;
pub mod topk;

pub use artifacts::{load_demo_artifacts, ArtifactError, DemoArtifacts, DemoTrack};
pub use ema::{EmaBuffer, EmaError};
pub use predictor::{resolve_onnx_path, Predictor, PredictorError};
pub use topk::format_topk;
