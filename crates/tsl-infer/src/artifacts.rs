//! Load and validate demo artifacts (labels, scaler JSON, ONNX model).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::Value;
use thiserror::Error;
use tsl_core::{load_labels, StandardScaler, FEATURE_DIM, FEATURE_SIZE, SEQ_LEN_DEFAULT};

use crate::predictor::{resolve_onnx_path, Predictor};

#[derive(Debug, Error)]
pub enum ArtifactError {
    #[error("IO: {0}")]
    Io(#[from] std::io::Error),

    #[error("labels: {0}")]
    Labels(String),

    #[error("scaler: {0}")]
    Scaler(#[from] tsl_core::scaler::ScalerError),

    #[error("predictor: {0}")]
    Predictor(#[from] crate::predictor::PredictorError),

    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),

    #[error("{0}")]
    Message(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemoTrack {
    Fingerspelling,
    Tsl51,
}

impl DemoTrack {
    pub fn as_str(self) -> &'static str {
        match self {
            DemoTrack::Fingerspelling => "thai_fingerspelling",
            DemoTrack::Tsl51 => "tsl51",
        }
    }

    pub fn expected_feature_dim(self) -> usize {
        match self {
            DemoTrack::Fingerspelling => FEATURE_SIZE,
            DemoTrack::Tsl51 => FEATURE_DIM,
        }
    }

    pub fn expected_sequence_len(self) -> Option<usize> {
        match self {
            DemoTrack::Fingerspelling => None,
            DemoTrack::Tsl51 => Some(SEQ_LEN_DEFAULT),
        }
    }
}

#[derive(Debug)]
pub struct DemoArtifacts {
    pub predictor: Predictor,
    pub labels: HashMap<String, String>,
    pub scaler: StandardScaler,
    pub model_path: PathBuf,
}

/// Ensure required paths exist; error lists all missing files.
pub fn ensure_required_files(paths: &[(&str, &Path)]) -> Result<(), ArtifactError> {
    let missing: Vec<_> = paths
        .iter()
        .filter(|(_, p)| !p.exists())
        .map(|(name, p)| format!("  - {name}: {}", p.display()))
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    let mut msg = String::from("Required artifact file(s) not found:\n");
    msg.push_str(&missing.join("\n"));
    Err(ArtifactError::Message(msg))
}

/// Load ONNX model, labels, and scaler with defensive validation.
pub fn load_demo_artifacts(
    model_path: impl AsRef<Path>,
    labels_path: impl AsRef<Path>,
    scaler_path: impl AsRef<Path>,
    track: DemoTrack,
    manifest_path: Option<&Path>,
    hint: Option<&str>,
) -> Result<DemoArtifacts, ArtifactError> {
    let model_p = model_path.as_ref();
    let labels_p = labels_path.as_ref();
    let scaler_p = scaler_path.as_ref();

    let onnx_p = resolve_onnx_path(model_p)?;
    ensure_required_files(&[
        ("labels", labels_p),
        ("scaler", scaler_p),
        ("model", &onnx_p),
    ])
    .map_err(|e| {
        if let Some(h) = hint {
            ArtifactError::Message(format!("{e}\n{h}"))
        } else {
            e
        }
    })?;

    let predictor = Predictor::load_onnx(&onnx_p)?;
    let labels = load_labels(labels_p).map_err(|e| ArtifactError::Labels(e.to_string()))?;
    predictor.validate_output(labels.len())?;
    predictor.validate_input(track.expected_feature_dim(), track.expected_sequence_len())?;

    let scaler = StandardScaler::load(scaler_p)?;
    if scaler.n_features != track.expected_feature_dim() {
        return Err(ArtifactError::Message(format!(
            "Scaler expects {} features, but {} track expects {}",
            scaler.n_features,
            track.as_str(),
            track.expected_feature_dim()
        )));
    }

    if let Some(manifest_file) = manifest_path {
        if manifest_file.exists() {
            let raw: Value = serde_json::from_str(&std::fs::read_to_string(manifest_file)?)?;
            if let Some(manifest_track) = raw.get("track").and_then(|v| v.as_str()) {
                if manifest_track != track.as_str() {
                    return Err(ArtifactError::Message(format!(
                        "Manifest track '{manifest_track}' does not match loaded track '{}'",
                        track.as_str()
                    )));
                }
            }
        }
    }

    Ok(DemoArtifacts {
        predictor,
        labels,
        scaler,
        model_path: onnx_p,
    })
}
