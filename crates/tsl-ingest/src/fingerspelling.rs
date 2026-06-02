//! One-Stage-TFS fingerspelling ingest (zip or existing tree).

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::zip_walk::{count_images, extract_zip_tolerant};
use crate::FEATURE_SIZE;

#[derive(Debug, Clone)]
pub struct FingerspellingIngestOptions {
    pub zip_path: Option<PathBuf>,
    pub dataset_root: Option<PathBuf>,
    pub output: PathBuf,
    pub force: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerspellingManifest {
    pub track: String,
    pub source: String,
    pub dataset_root: String,
    pub training_root: String,
    pub test_root: String,
    pub num_classes: usize,
    pub num_training_images: usize,
    pub num_test_images: usize,
    pub feature_size: usize,
    pub extracted_members_ok: usize,
    pub extracted_members_failed: usize,
}

#[derive(Debug, Error)]
pub enum IngestError {
    #[error("provide either zip_path or dataset_root")]
    MissingSource,
    #[error("zip not found: {0}")]
    ZipNotFound(PathBuf),
    #[error("dataset root not found: {0}")]
    DatasetRootNotFound(PathBuf),
    #[error("output exists: {0} (use force=true)")]
    OutputExists(PathBuf),
    #[error("could not find \"Training set\" under {0}")]
    TrainingSetNotFound(PathBuf),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("zip: {0}")]
    Zip(#[from] crate::zip_walk::ZipWalkError),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

fn ensure_dir(path: &Path, force: bool) -> Result<(), IngestError> {
    if path.exists() {
        if force {
            fs::remove_dir_all(path)?;
        } else {
            return Err(IngestError::OutputExists(path.to_path_buf()));
        }
    }
    fs::create_dir_all(path)?;
    Ok(())
}

fn find_training_root(dataset_root: &Path) -> Result<PathBuf, IngestError> {
    for entry in walkdir::WalkDir::new(dataset_root)
        .into_iter()
        .filter_map(Result::ok)
    {
        if entry.file_type().is_dir() && entry.file_name() == "Training set" {
            return Ok(entry.path().to_path_buf());
        }
    }
    Err(IngestError::TrainingSetNotFound(dataset_root.to_path_buf()))
}

pub fn ingest_fingerspelling(
    opts: FingerspellingIngestOptions,
) -> Result<FingerspellingManifest, IngestError> {
    let from_dataset_root = opts.dataset_root.is_some();
    let (dataset_root, extracted_ok, extracted_failed) = if let Some(root) = opts.dataset_root {
        if !root.exists() {
            return Err(IngestError::DatasetRootNotFound(root));
        }
        (root, 0usize, 0usize)
    } else if let Some(zip) = opts.zip_path {
        if !zip.exists() {
            return Err(IngestError::ZipNotFound(zip));
        }
        ensure_dir(&opts.output, opts.force)?;
        let (ok, failed) = extract_zip_tolerant(&zip, &opts.output)?;
        (opts.output.clone(), ok, failed)
    } else {
        return Err(IngestError::MissingSource);
    };

    let out_root = if from_dataset_root && opts.output != dataset_root {
        dataset_root.clone()
    } else {
        opts.output.clone()
    };

    let training_root = find_training_root(&dataset_root)?;
    let test_root = training_root
        .parent()
        .unwrap_or(&dataset_root)
        .join("Test set");
    let train_classes = fs::read_dir(&training_root)?
        .filter_map(Result::ok)
        .filter(|e| e.path().is_dir())
        .count();
    let train_images = count_images(&training_root);
    let test_images = if test_root.is_dir() {
        count_images(&test_root)
    } else {
        0
    };

    let manifest = FingerspellingManifest {
        track: "thai_fingerspelling".into(),
        source: dataset_root.display().to_string(),
        dataset_root: out_root.display().to_string(),
        training_root: training_root.display().to_string(),
        test_root: if test_root.is_dir() {
            test_root.display().to_string()
        } else {
            String::new()
        },
        num_classes: train_classes,
        num_training_images: train_images,
        num_test_images: test_images,
        feature_size: FEATURE_SIZE,
        extracted_members_ok: extracted_ok,
        extracted_members_failed: extracted_failed,
    };

    let manifest_path = out_root.join("ingest_manifest.json");
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;
    Ok(manifest)
}
