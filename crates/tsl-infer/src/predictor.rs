//! ONNX classifier via `tract-onnx`.

use std::path::{Path, PathBuf};

use thiserror::Error;
use tract_core::prelude::*;
use tract_onnx::prelude::*;

fn shape_to_isize(shape: &[TDim]) -> Vec<isize> {
    shape
        .iter()
        .map(|d| {
            if let Ok(v) = d.to_i64() {
                v as isize
            } else {
                -1
            }
        })
        .collect()
}

#[derive(Debug, Error)]
pub enum PredictorError {
    #[error("tract: {0}")]
    Tract(#[from] TractError),

    #[error("model file not found: {0}")]
    ModelNotFound(PathBuf),

    #[error("unsupported input rank {rank} (expected 2 or 3)")]
    UnsupportedInputRank { rank: usize },

    #[error("model outputs {got} classes but labels have {expected}")]
    ClassCountMismatch { got: usize, expected: usize },

    #[error("model input feature dim {got} does not match expected {expected}")]
    FeatureDimMismatch { got: usize, expected: usize },

    #[error("model sequence length {got} does not match expected {expected}")]
    SequenceLenMismatch { got: usize, expected: usize },

    #[error("expected_classes must be > 0")]
    InvalidClassCount,

    #[error("expected_feature_dim must be > 0")]
    InvalidFeatureDim,

    #[error("model output is empty")]
    EmptyOutput,
}

/// ONNX runtime wrapper with shape validation (port of Python `Predictor`).
pub struct Predictor {
    model: RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    input_shape: Vec<isize>,
    output_classes: usize,
}

impl std::fmt::Debug for Predictor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Predictor")
            .field("input_shape", &self.input_shape)
            .field("output_classes", &self.output_classes)
            .finish_non_exhaustive()
    }
}

impl Predictor {
    /// Load and optimize an ONNX graph from disk.
    pub fn load_onnx(path: impl AsRef<Path>) -> Result<Self, PredictorError> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(PredictorError::ModelNotFound(path.to_path_buf()));
        }
        let model = tract_onnx::onnx()
            .model_for_path(path)?
            .into_optimized()?
            .into_runnable()?;

        let input_fact = model.model().input_fact(0)?.clone();
        let input_shape = shape_to_isize(&input_fact.shape);

        let output_fact = model.model().output_fact(0)?.clone();
        let out_shape = shape_to_isize(&output_fact.shape);
        let output_classes = out_shape
            .last()
            .copied()
            .filter(|&d| d > 0)
            .map(|d| d as usize)
            .unwrap_or(0);

        Ok(Self {
            model,
            input_shape,
            output_classes,
        })
    }

    pub fn input_shape(&self) -> &[isize] {
        &self.input_shape
    }

    pub fn num_classes(&self) -> usize {
        self.output_classes
    }

    pub fn validate_output(&self, expected_classes: usize) -> Result<(), PredictorError> {
        if expected_classes == 0 {
            return Err(PredictorError::InvalidClassCount);
        }
        if self.output_classes != expected_classes {
            return Err(PredictorError::ClassCountMismatch {
                got: self.output_classes,
                expected: expected_classes,
            });
        }
        Ok(())
    }

    pub fn validate_input(
        &self,
        expected_feature_dim: usize,
        expected_sequence_len: Option<usize>,
    ) -> Result<(), PredictorError> {
        if expected_feature_dim == 0 {
            return Err(PredictorError::InvalidFeatureDim);
        }
        let rank = self.input_shape.len();
        if rank != 2 && rank != 3 {
            return Err(PredictorError::UnsupportedInputRank { rank });
        }
        let input_dim = self
            .input_shape
            .last()
            .copied()
            .filter(|&d| d > 0)
            .map(|d| d as usize)
            .ok_or(PredictorError::FeatureDimMismatch {
                got: 0,
                expected: expected_feature_dim,
            })?;
        if input_dim != expected_feature_dim {
            return Err(PredictorError::FeatureDimMismatch {
                got: input_dim,
                expected: expected_feature_dim,
            });
        }
        if let Some(expected_seq) = expected_sequence_len {
            if rank == 3 {
                let seq_len = self.input_shape[1];
                if seq_len > 0 && seq_len as usize != expected_seq {
                    return Err(PredictorError::SequenceLenMismatch {
                        got: seq_len as usize,
                        expected: expected_seq,
                    });
                }
            }
        }
        Ok(())
    }

    /// Run inference; `features` is flat row-major matching the ONNX input (batch 1).
    pub fn predict(&self, features: &[f32]) -> Result<Vec<f32>, PredictorError> {
        let rank = self.input_shape.len();
        let tensor = match rank {
            2 => {
                let feat_dim = self.input_shape[1];
                if feat_dim > 0 && features.len() != feat_dim as usize {
                    return Err(PredictorError::FeatureDimMismatch {
                        got: features.len(),
                        expected: feat_dim as usize,
                    });
                }
                Tensor::from_shape(&[1, features.len()], features)?
            }
            3 => {
                let seq = self.input_shape[1];
                let feat = self.input_shape[2];
                let expected = if seq > 0 && feat > 0 {
                    seq as usize * feat as usize
                } else {
                    features.len()
                };
                if features.len() != expected {
                    return Err(PredictorError::FeatureDimMismatch {
                        got: features.len(),
                        expected,
                    });
                }
                let feat_usize = if feat > 0 {
                    feat as usize
                } else {
                    features.len() / seq.max(1) as usize
                };
                let seq_usize = if seq > 0 {
                    seq as usize
                } else {
                    features.len() / feat_usize.max(1)
                };
                Tensor::from_shape(&[1, seq_usize, feat_usize], features)?
            }
            r => return Err(PredictorError::UnsupportedInputRank { rank: r }),
        };

        let outputs = self.model.run(tvec!(tensor.into()))?;
        let view = outputs[0].to_array_view::<f32>()?;
        let probs: Vec<f32> = view.iter().copied().collect();
        if probs.is_empty() {
            return Err(PredictorError::EmptyOutput);
        }
        // Models may return [1, C] or [C].
        if probs.len() > self.output_classes && self.output_classes > 0 {
            let start = probs.len() - self.output_classes;
            return Ok(probs[start..].to_vec());
        }
        Ok(probs)
    }
}

/// Resolve `model.keras` / `model.tflite` arguments to a sibling `.onnx` file.
pub fn resolve_onnx_path(model_arg: &Path) -> Result<PathBuf, PredictorError> {
    if model_arg.extension().and_then(|e| e.to_str()) == Some("onnx") && model_arg.is_file() {
        return Ok(model_arg.to_path_buf());
    }
    let onnx = model_arg.with_extension("onnx");
    if onnx.is_file() {
        return Ok(onnx);
    }
    Err(PredictorError::ModelNotFound(onnx))
}
