//! Training configuration, Candle model skeletons, and loop stubs.

use serde::{Deserialize, Serialize};

pub const FS_INPUT_DIM: usize = 63;
pub const FS_HIDDEN: [usize; 3] = [256, 128, 64];
pub const TSL51_FEATURE_DIM: usize = 162;
pub const TSL51_SEQ_LEN: usize = 60;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainTrack {
    Fingerspelling,
    Tsl51,
    Both,
}

impl TrainTrack {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fingerspelling => "fingerspelling",
            Self::Tsl51 => "tsl51",
            Self::Both => "both",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub track: TrainTrack,
    pub epochs: u32,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub artifact_dir: String,
    pub seed: u64,
    /// NPZ feature cache from `train_local_all.py` (enables real CPU training).
    pub feature_cache: Option<String>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            track: TrainTrack::Both,
            epochs: 50,
            batch_size: 64,
            learning_rate: 1e-3,
            artifact_dir: "artifacts".into(),
            seed: 42,
            feature_cache: None,
        }
    }
}

#[cfg(feature = "candle")]
pub mod fingerspelling;
#[cfg(feature = "candle")]
pub mod train;
#[cfg(feature = "candle")]
pub mod tsl51;

#[cfg(feature = "candle")]
pub use fingerspelling::{build_fingerspelling, FingerspellingConfig, FingerspellingModel};
#[cfg(feature = "candle")]
pub use train::{run_training, run_training_from_npz, run_training_stub, TrainingError};
#[cfg(feature = "candle")]
pub use tsl51::{build_tsl51, Tsl51Config, Tsl51Model};

/// Entry used by `tsl-train`: builds Candle skeletons when enabled, otherwise returns a note.
pub fn train_placeholder(config: &TrainConfig) -> anyhow::Result<String> {
    #[cfg(feature = "candle")]
    {
        use std::path::PathBuf;

        use candle_core::Device;

        let train_cfg = train::TrainConfig {
            track: config.track,
            artifact_dir: PathBuf::from(&config.artifact_dir),
            feature_cache: config.feature_cache.as_ref().map(PathBuf::from),
            epochs: config.epochs,
            batch_size: config.batch_size,
            learning_rate: config.learning_rate,
            device: Device::Cpu,
        };
        run_training(&train_cfg)?;
        let mode = if config.feature_cache.is_some() {
            "npz"
        } else {
            "stub"
        };
        return Ok(format!(
            "tsl-ml candle {mode}: track={} epochs={} artifact_dir={}",
            config.track.as_str(),
            config.epochs,
            config.artifact_dir
        ));
    }

    #[cfg(not(feature = "candle"))]
    {
        let _ = config;
        anyhow::bail!("rebuild tsl-ml with `--features candle` for training stubs")
    }
}
