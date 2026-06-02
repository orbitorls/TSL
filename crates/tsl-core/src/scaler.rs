//! StandardScaler as JSON (`mean`, `scale`, `n_features`).

use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ScalerError {
    #[error("IO: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("expected {expected} features, got {got}")]
    FeatureMismatch { expected: usize, got: usize },
    #[error("mean/scale length mismatch")]
    LengthMismatch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardScaler {
    pub mean: Vec<f32>,
    pub scale: Vec<f32>,
    pub n_features: usize,
}

impl StandardScaler {
    pub fn from_json_str(s: &str) -> Result<Self, ScalerError> {
        let scaler: Self = serde_json::from_str(s)?;
        scaler.validate()?;
        Ok(scaler)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, ScalerError> {
        let text = std::fs::read_to_string(path)?;
        Self::from_json_str(&text)
    }

    fn validate(&self) -> Result<(), ScalerError> {
        if self.mean.len() != self.scale.len() {
            return Err(ScalerError::LengthMismatch);
        }
        if self.n_features != self.mean.len() {
            return Err(ScalerError::FeatureMismatch {
                expected: self.n_features,
                got: self.mean.len(),
            });
        }
        Ok(())
    }

    /// sklearn `transform`: (x - mean) / scale
    pub fn transform(&self, x: &[f32]) -> Result<Vec<f32>, ScalerError> {
        if x.len() != self.n_features {
            return Err(ScalerError::FeatureMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }
        let mut out = Vec::with_capacity(self.n_features);
        for i in 0..self.n_features {
            let scale = if self.scale[i].abs() < 1e-12 {
                1.0
            } else {
                self.scale[i]
            };
            out.push((x[i] - self.mean[i]) / scale);
        }
        Ok(out)
    }

    pub fn transform_fixed<const N: usize>(&self, x: &[f32; N]) -> Result<[f32; N], ScalerError> {
        let v = self.transform(x)?;
        let mut out = [0.0; N];
        out.copy_from_slice(&v);
        Ok(out)
    }
}
