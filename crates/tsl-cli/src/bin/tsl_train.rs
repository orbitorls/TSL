//! `tsl-train` — Rust/Candle training from NPZ feature caches or stub mode.

use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use tsl_ml::{train_placeholder, TrainConfig, TrainTrack};

#[derive(Clone, Copy, ValueEnum, Debug)]
enum TrackArg {
    Fingerspelling,
    Tsl51,
    Both,
}

impl From<TrackArg> for TrainTrack {
    fn from(v: TrackArg) -> Self {
        match v {
            TrackArg::Fingerspelling => TrainTrack::Fingerspelling,
            TrackArg::Tsl51 => TrainTrack::Tsl51,
            TrackArg::Both => TrainTrack::Both,
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "tsl-train",
    about = "Train TSL fingerspelling and/or TSL-51 models (Candle CPU; NPZ cache or stub)"
)]
struct Args {
    /// Which dataset track to train.
    #[arg(long, value_enum, default_value_t = TrackArg::Both)]
    track: TrackArg,

    #[arg(long, default_value = "artifacts")]
    artifact_dir: String,

    #[arg(long, default_value_t = 50)]
    epochs: u32,

    #[arg(long, default_value_t = 64)]
    batch_size: usize,

    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f64,

    /// NPZ feature cache from `train_local_all.py` (`keypoints_cache.npz` or `tsl51_features.npz`).
    #[arg(long, value_name = "PATH")]
    feature_cache: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let cfg = TrainConfig {
        track: args.track.into(),
        artifact_dir: args.artifact_dir,
        epochs: args.epochs,
        batch_size: args.batch_size,
        learning_rate: args.learning_rate,
        feature_cache: args.feature_cache.map(|p| p.display().to_string()),
        ..TrainConfig::default()
    };
    println!("{}", train_placeholder(&cfg)?);
    Ok(())
}
