//! Dense DNN for One-Stage-TFS fingerspelling (matches Keras layout in `train_local_all.py`).

use candle_core::{Device, Result, Tensor};
use candle_nn::{linear, Dropout, Linear, Module, VarBuilder};

use crate::FS_INPUT_DIM;

#[derive(Debug, Clone)]
pub struct FingerspellingConfig {
    pub input_dim: usize,
    pub num_classes: usize,
    pub dropout1: f32,
    pub dropout2: f32,
}

impl Default for FingerspellingConfig {
    fn default() -> Self {
        Self {
            input_dim: FS_INPUT_DIM,
            num_classes: 15,
            dropout1: 0.30,
            dropout2: 0.25,
        }
    }
}

pub struct FingerspellingModel {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
    fc_out: Linear,
    dropout1: Dropout,
    dropout2: Dropout,
    num_classes: usize,
}

impl FingerspellingModel {
    pub fn new(cfg: &FingerspellingConfig, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(cfg.input_dim, 256, vb.pp("fc1"))?;
        let fc2 = linear(256, 128, vb.pp("fc2"))?;
        let fc3 = linear(128, 64, vb.pp("fc3"))?;
        let fc_out = linear(64, cfg.num_classes, vb.pp("fc_out"))?;
        Ok(Self {
            fc1,
            fc2,
            fc3,
            fc_out,
            dropout1: Dropout::new(cfg.dropout1),
            dropout2: Dropout::new(cfg.dropout2),
            num_classes: cfg.num_classes,
        })
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Forward pass: `(batch, FEATURE_SIZE)` → logits `(batch, num_classes)`.
    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?.relu()?;
        let xs = if train {
            self.dropout1.forward(&xs, true)?
        } else {
            xs
        };
        let xs = self.fc2.forward(&xs)?.relu()?;
        let xs = if train {
            self.dropout2.forward(&xs, true)?
        } else {
            xs
        };
        let xs = self.fc3.forward(&xs)?.relu()?;
        self.fc_out.forward(&xs)
    }
}

/// Build model on the given device (stub-friendly entry point).
pub fn build_fingerspelling(
    cfg: &FingerspellingConfig,
    device: &Device,
) -> Result<FingerspellingModel> {
    let vb = VarBuilder::zeros(candle_core::DType::F32, device);
    FingerspellingModel::new(cfg, vb)
}
