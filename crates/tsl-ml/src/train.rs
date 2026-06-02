//! Training from Python NPZ feature caches (CPU / Candle).

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::{loss::cross_entropy, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use thiserror::Error;
use tsl_core::npz::{detect_npz_kind, FingerspellingFeatureCache, Tsl51FeatureCache};
use tsl_core::npz::{load_fingerspelling_npz, load_tsl51_npz, NpzError, NpzKind};

use crate::fingerspelling::{FingerspellingConfig, FingerspellingModel};
use crate::tsl51::{Tsl51Config, Tsl51Model};
use crate::{TrainConfig as PublicTrainConfig, TrainTrack, TSL51_FEATURE_DIM, TSL51_SEQ_LEN};

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub track: TrainTrack,
    pub artifact_dir: PathBuf,
    pub epochs: u32,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub device: Device,
    pub feature_cache: Option<PathBuf>,
}

#[derive(Debug, Error)]
pub enum TrainingError {
    #[error("candle: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("npz: {0}")]
    Npz(#[from] NpzError),
    #[error("zip: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("{0}")]
    Other(String),
}

pub fn run_training(cfg: &TrainConfig) -> Result<(), TrainingError> {
    if let Some(path) = &cfg.feature_cache {
        return run_training_from_npz(cfg, path);
    }
    run_training_stub(cfg)
}

pub fn run_training_from_npz(cfg: &TrainConfig, cache_path: &Path) -> Result<(), TrainingError> {
    std::fs::create_dir_all(&cfg.artifact_dir)?;
    let file = std::fs::File::open(cache_path)?;
    let mut archive = zip::read::ZipArchive::new(file)?;
    let names: Vec<String> = (0..archive.len())
        .map(|i| archive.by_index(i).map(|e| e.name().to_string()))
        .collect::<Result<_, _>>()?;
    let kind = detect_npz_kind(&names).ok_or_else(|| {
        TrainingError::Other(format!(
            "unrecognized NPZ layout in {}; expected fingerspelling or tsl51 keys",
            cache_path.display()
        ))
    })?;

    let manifest = match (cfg.track, kind) {
        (TrainTrack::Fingerspelling, NpzKind::Fingerspelling) => {
            let cache = load_fingerspelling_npz(cache_path)?;
            train_fingerspelling(cfg, &cache)?;
            serde_json::json!({
                "status": "trained",
                "track": "fingerspelling",
                "samples": cache.num_samples(),
                "classes": cache.num_classes(),
            })
        }
        (TrainTrack::Tsl51, NpzKind::Tsl51) => {
            let cache = load_tsl51_npz(cache_path)?;
            train_tsl51(cfg, &cache)?;
            serde_json::json!({
                "status": "trained",
                "track": "tsl51",
                "samples": cache.num_samples(),
                "classes": cache.num_classes(),
                "signature": cache.signature,
            })
        }
        (TrainTrack::Both, kind) => {
            return Err(TrainingError::Other(format!(
                "track both cannot use a single --feature-cache ({kind:?}); run \
                 tsl-train twice with --track fingerspelling and --track tsl51"
            )));
        }
        (track, kind) => {
            return Err(TrainingError::Other(format!(
                "track {:?} does not match NPZ kind {:?} in {}",
                track,
                kind,
                cache_path.display()
            )));
        }
    };

    let path = cfg.artifact_dir.join("train_manifest.json");
    std::fs::write(path, serde_json::to_string_pretty(&manifest)?)?;
    Ok(())
}

fn train_fingerspelling(
    cfg: &TrainConfig,
    cache: &FingerspellingFeatureCache,
) -> Result<(), TrainingError> {
    if cache.num_samples() == 0 {
        return Err(TrainingError::Other("empty fingerspelling cache".into()));
    }
    let num_classes = cache.num_classes();
    if num_classes == 0 {
        return Err(TrainingError::Other(
            "no classes in fingerspelling cache".into(),
        ));
    }

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &cfg.device);
    let model_cfg = FingerspellingConfig {
        input_dim: cache.feature_size,
        num_classes,
        ..FingerspellingConfig::default()
    };
    let model = FingerspellingModel::new(&model_cfg, vb)?;
    let opt_params = ParamsAdamW {
        lr: cfg.learning_rate,
        ..ParamsAdamW::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), opt_params)?;

    let n = cache.num_samples();
    let input_dim = cache.feature_size;
    let mut indices: Vec<usize> = (0..n).collect();

    for epoch in 0..cfg.epochs {
        shuffle(&mut indices, epoch as u64 + 1);
        let mut loss_sum = 0f32;
        let mut correct = 0usize;
        let mut steps = 0usize;

        for batch_start in (0..n).step_by(cfg.batch_size.max(1)) {
            let batch_end = (batch_start + cfg.batch_size).min(n);
            let batch_len = batch_end - batch_start;
            let mut x_batch = Vec::with_capacity(batch_len * input_dim);
            let mut y_batch = Vec::with_capacity(batch_len);
            for &idx in &indices[batch_start..batch_end] {
                let row_start = idx * input_dim;
                x_batch.extend_from_slice(&cache.x_train[row_start..row_start + input_dim]);
                y_batch.push(cache.y_train[idx] as u32);
            }
            let x = Tensor::from_vec(x_batch, (batch_len, input_dim), &cfg.device)?;
            let y = Tensor::from_vec(y_batch, batch_len, &cfg.device)?;
            let logits = model.forward(&x, true)?;
            let loss = cross_entropy(&logits, &y)?;
            opt.backward_step(&loss)?;
            loss_sum += loss.to_scalar::<f32>()?;
            steps += 1;
            let pred = logits.argmax(1)?;
            correct += count_matches(&pred, &y)?;
        }
        let avg_loss = loss_sum / steps.max(1) as f32;
        let acc = 100.0 * correct as f32 / n as f32;
        eprintln!(
            "[fs] epoch {}/{} loss={avg_loss:.4} acc={acc:.1}% (n={n} classes={num_classes})",
            epoch + 1,
            cfg.epochs,
        );
    }
    Ok(())
}

fn train_tsl51(cfg: &TrainConfig, cache: &Tsl51FeatureCache) -> Result<(), TrainingError> {
    if cache.num_samples() == 0 {
        return Err(TrainingError::Other("empty tsl51 cache".into()));
    }
    let num_classes = cache.num_classes();
    if num_classes == 0 {
        return Err(TrainingError::Other("no classes in tsl51 cache".into()));
    }

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &cfg.device);
    let model_cfg = Tsl51Config {
        seq_len: cache.seq_len,
        feature_dim: cache.feature_dim,
        num_classes,
        ..Tsl51Config::default()
    };
    let model = Tsl51Model::new(&model_cfg, vb)?;
    let opt_params = ParamsAdamW {
        lr: cfg.learning_rate,
        ..ParamsAdamW::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), opt_params)?;

    let n = cache.num_samples();
    let seq_len = cache.seq_len;
    let feature_dim = cache.feature_dim;
    let frame_size = seq_len * feature_dim;
    let mut indices: Vec<usize> = (0..n).collect();

    for epoch in 0..cfg.epochs {
        shuffle(&mut indices, epoch as u64 + 101);
        let mut loss_sum = 0f32;
        let mut correct = 0usize;
        let mut steps = 0usize;

        for batch_start in (0..n).step_by(cfg.batch_size.max(1)) {
            let batch_end = (batch_start + cfg.batch_size).min(n);
            let batch_len = batch_end - batch_start;
            let mut x_batch = Vec::with_capacity(batch_len * frame_size);
            let mut y_batch = Vec::with_capacity(batch_len);
            for &idx in &indices[batch_start..batch_end] {
                let row_start = idx * frame_size;
                x_batch.extend_from_slice(&cache.x[row_start..row_start + frame_size]);
                y_batch.push(cache.y[idx] as u32);
            }
            let x = Tensor::from_vec(x_batch, (batch_len, seq_len, feature_dim), &cfg.device)?;
            let y = Tensor::from_vec(y_batch, batch_len, &cfg.device)?;
            let logits = model.forward(&x, true)?;
            let loss = cross_entropy(&logits, &y)?;
            opt.backward_step(&loss)?;
            loss_sum += loss.to_scalar::<f32>()?;
            steps += 1;
            let pred = logits.argmax(1)?;
            correct += count_matches(&pred, &y)?;
        }
        let avg_loss = loss_sum / steps.max(1) as f32;
        let acc = 100.0 * correct as f32 / n as f32;
        eprintln!(
            "[tsl51] epoch {}/{} loss={avg_loss:.4} acc={acc:.1}% (n={n} classes={num_classes})",
            epoch + 1,
            cfg.epochs,
        );
    }
    Ok(())
}

fn count_matches(pred: &Tensor, y: &Tensor) -> Result<usize, TrainingError> {
    let pred_v = pred.to_vec1::<u32>()?;
    let y_v = y.to_vec1::<u32>()?;
    Ok(pred_v
        .iter()
        .zip(y_v.iter())
        .filter(|(p, t)| p == t)
        .count())
}

fn shuffle(indices: &mut [usize], seed: u64) {
    let mut state = seed;
    for i in (1..indices.len()).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (state as usize) % (i + 1);
        indices.swap(i, j);
    }
}

/// Legacy stub when no `--feature-cache` is provided.
pub fn run_training_stub(cfg: &TrainConfig) -> Result<(), TrainingError> {
    use crate::fingerspelling::build_fingerspelling;
    use crate::tsl51::build_tsl51;

    std::fs::create_dir_all(&cfg.artifact_dir)?;

    match cfg.track {
        TrainTrack::Fingerspelling | TrainTrack::Both => {
            let fs_cfg = FingerspellingConfig::default();
            let model = build_fingerspelling(&fs_cfg, &cfg.device)?;
            eprintln!(
                "[stub] fingerspelling Dense: classes={} epochs={} batch={} out={}",
                model.num_classes(),
                cfg.epochs,
                cfg.batch_size,
                cfg.artifact_dir.display()
            );
        }
        _ => {}
    }

    match cfg.track {
        TrainTrack::Tsl51 | TrainTrack::Both => {
            let tsl_cfg = Tsl51Config::default();
            let model = build_tsl51(&tsl_cfg, &cfg.device)?;
            eprintln!(
                "[stub] tsl51 BiLSTM: classes={} seq={} dim={} epochs={}",
                model.num_classes(),
                TSL51_SEQ_LEN,
                TSL51_FEATURE_DIM,
                cfg.epochs
            );
        }
        _ => {}
    }

    let manifest = serde_json::json!({
        "status": "stub",
        "track": cfg.track.as_str(),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
    });
    let path = cfg.artifact_dir.join("train_stub_manifest.json");
    std::fs::write(path, serde_json::to_string_pretty(&manifest)?)?;
    let _ = PublicTrainConfig::default();
    Ok(())
}

// Back-compat alias used by older call sites.
pub type TrainingStubError = TrainingError;
