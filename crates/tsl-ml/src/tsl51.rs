//! BiLSTM skeleton for TSL-51 word signs (Keras: 96+64 units, 128-D head).

use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::{linear, Dropout, LSTMConfig, Linear, Module, VarBuilder, LSTM, RNN};

use crate::{TSL51_FEATURE_DIM, TSL51_SEQ_LEN};

#[derive(Debug, Clone)]
pub struct Tsl51Config {
    pub seq_len: usize,
    pub feature_dim: usize,
    pub lstm_units: usize,
    pub lstm_units2: usize,
    pub dense_hidden: usize,
    pub num_classes: usize,
    pub dropout: f32,
    pub dropout_head: f32,
}

impl Default for Tsl51Config {
    fn default() -> Self {
        Self {
            seq_len: TSL51_SEQ_LEN,
            feature_dim: TSL51_FEATURE_DIM,
            lstm_units: 96,
            lstm_units2: 64,
            dense_hidden: 128,
            num_classes: 51,
            dropout: 0.25,
            dropout_head: 0.20,
        }
    }
}

/// Bidirectional stack + classification head (training loop not wired yet).
pub struct Tsl51Model {
    lstm_fwd: LSTM,
    lstm_bwd: LSTM,
    lstm_fwd2: LSTM,
    lstm_bwd2: LSTM,
    head: Linear,
    out: Linear,
    dropout: Dropout,
    dropout_head: Dropout,
    num_classes: usize,
}

impl Tsl51Model {
    pub fn new(cfg: &Tsl51Config, vb: VarBuilder) -> Result<Self> {
        let lstm_cfg = LSTMConfig::default();
        let lstm_fwd = LSTM::new(cfg.feature_dim, cfg.lstm_units, lstm_cfg, vb.pp("lstm_fwd"))?;
        let lstm_bwd = LSTM::new(cfg.feature_dim, cfg.lstm_units, lstm_cfg, vb.pp("lstm_bwd"))?;
        let concat = cfg.lstm_units * 2;
        let lstm_fwd2 = LSTM::new(concat, cfg.lstm_units2, lstm_cfg, vb.pp("lstm_fwd2"))?;
        let lstm_bwd2 = LSTM::new(concat, cfg.lstm_units2, lstm_cfg, vb.pp("lstm_bwd2"))?;
        let head_in = cfg.lstm_units2 * 2;
        let head = linear(head_in, cfg.dense_hidden, vb.pp("head"))?;
        let out = linear(cfg.dense_hidden, cfg.num_classes, vb.pp("out"))?;
        Ok(Self {
            lstm_fwd,
            lstm_bwd,
            lstm_fwd2,
            lstm_bwd2,
            head,
            out,
            dropout: Dropout::new(cfg.dropout),
            dropout_head: Dropout::new(cfg.dropout_head),
            num_classes: cfg.num_classes,
        })
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Run one BiLSTM block. Input/output layout: `(batch, seq_len, features)`.
    fn bilstm_step(lstm_fwd: &LSTM, lstm_bwd: &LSTM, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.transpose(0, 1)?; // (seq, batch, in_dim)
        let states_fwd = lstm_fwd.seq(&xs)?;
        let fwd = lstm_fwd.states_to_tensor(&states_fwd)?;
        let rev = reverse_along_dim(&xs, 0)?;
        let states_bwd = lstm_bwd.seq(&rev)?;
        let mut bwd = lstm_bwd.states_to_tensor(&states_bwd)?;
        bwd = reverse_along_dim(&bwd, 0)?;
        let merged = Tensor::cat(&[&fwd, &bwd], 2)?;
        merged.transpose(0, 1)
    }

    /// Forward: `(batch, seq_len, feature_dim)` → logits `(batch, num_classes)`.
    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let h1 = Self::bilstm_step(&self.lstm_fwd, &self.lstm_bwd, xs)?;
        let h1 = if train {
            self.dropout.forward(&h1, true)?
        } else {
            h1
        };
        let h2 = Self::bilstm_step(&self.lstm_fwd2, &self.lstm_bwd2, &h1)?;
        let h2 = if train {
            self.dropout.forward(&h2, true)?
        } else {
            h2
        };
        let seq_len = h2.dim(1)?;
        let last = h2.i((.., seq_len - 1, ..))?;
        let h = self.head.forward(&last)?.relu()?;
        let h = if train {
            self.dropout_head.forward(&h, true)?
        } else {
            h
        };
        self.out.forward(&h)
    }
}

fn reverse_along_dim(t: &Tensor, dim: usize) -> Result<Tensor> {
    let len = t.dim(dim)?;
    let mut parts = Vec::with_capacity(len);
    for i in (0..len).rev() {
        parts.push(t.narrow(dim, i, 1)?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, dim)
}

pub fn build_tsl51(cfg: &Tsl51Config, device: &Device) -> Result<Tsl51Model> {
    let vb = VarBuilder::zeros(candle_core::DType::F32, device);
    Tsl51Model::new(cfg, vb)
}
