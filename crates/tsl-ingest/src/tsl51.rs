//! TSL-51 metadata ingest via local copy or Hugging Face Hub.

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use hf_hub::api::sync::Api;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{FEATURE_DIM, SEQ_LEN_DEFAULT, TSL51_REPO_ID};

const METADATA_FILES: &[&str] = &[
    "metadata/expert_metadata.csv",
    "metadata/user_sign_metadata.csv",
];

#[derive(Debug, Clone)]
pub struct Tsl51IngestOptions {
    pub source: Option<PathBuf>,
    pub output: PathBuf,
    pub force: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tsl51Manifest {
    pub track: String,
    pub dataset: String,
    pub metadata_dir: String,
    pub meta_rows: usize,
    pub num_classes: usize,
    pub num_training_rows: usize,
    pub num_null_rows: usize,
    pub seq_len: usize,
    pub feature_dim: usize,
}

#[derive(Debug, Error)]
pub enum Tsl51IngestError {
    #[error("source not found: {0}")]
    SourceNotFound(PathBuf),
    #[error("output exists: {0} (use force=true)")]
    OutputExists(PathBuf),
    #[error("missing metadata file: {0}")]
    MissingMetadata(PathBuf),
    #[error("hf hub: {0}")]
    Hub(String),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

fn ensure_dir(path: &Path, force: bool) -> Result<(), Tsl51IngestError> {
    if path.exists() {
        if force {
            fs::remove_dir_all(path)?;
        } else {
            return Err(Tsl51IngestError::OutputExists(path.to_path_buf()));
        }
    }
    fs::create_dir_all(path)?;
    Ok(())
}

fn collect_source_metadata_files(source_root: &Path) -> Result<Vec<PathBuf>, Tsl51IngestError> {
    let direct = [
        source_root.join("expert_metadata.csv"),
        source_root.join("user_sign_metadata.csv"),
    ];
    if direct.iter().all(|p| p.is_file()) {
        return Ok(direct.to_vec());
    }
    let nested = source_root.join("metadata");
    let nested_files = [
        nested.join("expert_metadata.csv"),
        nested.join("user_sign_metadata.csv"),
    ];
    if nested_files.iter().all(|p| p.is_file()) {
        return Ok(nested_files.to_vec());
    }
    Err(Tsl51IngestError::MissingMetadata(source_root.to_path_buf()))
}

fn download_metadata_hf(metadata_out: &Path) -> Result<(), Tsl51IngestError> {
    let api = Api::new().map_err(|e| Tsl51IngestError::Hub(e.to_string()))?;
    let repo = api.dataset(TSL51_REPO_ID.to_string());
    for rel in METADATA_FILES {
        let local = repo
            .get(rel)
            .map_err(|e| Tsl51IngestError::Hub(e.to_string()))?;
        let name = Path::new(rel)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("metadata.csv");
        let target = metadata_out.join(name);
        fs::copy(&local, &target)?;
        println!("downloaded {rel}");
    }
    Ok(())
}

fn concat_metadata(
    files: &[PathBuf],
    out_file: &Path,
) -> Result<(usize, usize, usize, usize), Tsl51IngestError> {
    let mut total_rows = 0usize;
    let mut train_rows = 0usize;
    let mut null_rows = 0usize;
    let mut class_ids = std::collections::HashSet::new();
    let mut header: Option<Vec<String>> = None;

    let mut out = fs::File::create(out_file)?;
    for path in files {
        let f = fs::File::open(path)?;
        let mut lines = BufReader::new(f).lines();
        let Some(first) = lines.next() else { continue };
        let cols: Vec<String> = first?.split(',').map(|s| s.trim().to_string()).collect();
        if header.is_none() {
            header = Some(cols.clone());
            writeln!(out, "{}", cols.join(","))?;
        }
        for line in lines {
            let row = line?;
            let parts: Vec<&str> = row.split(',').collect();
            if let Some(hdr) = &header {
                if let Some(idx) = hdr.iter().position(|c| c == "sign_id") {
                    let sid = parts.get(idx).copied().unwrap_or("").trim();
                    if sid == "null_act" {
                        null_rows += 1;
                    } else if !sid.is_empty() {
                        train_rows += 1;
                        class_ids.insert(sid.to_string());
                    }
                }
            }
            total_rows += 1;
            writeln!(out, "{row}")?;
        }
    }
    Ok((total_rows, train_rows, null_rows, class_ids.len()))
}

/// Prepare TSL-51 metadata snapshot (local copy or HF Hub download).
pub fn ingest_tsl51_metadata(opts: Tsl51IngestOptions) -> Result<Tsl51Manifest, Tsl51IngestError> {
    ensure_dir(&opts.output, opts.force)?;
    let metadata_out = opts.output.join("metadata");
    fs::create_dir_all(&metadata_out)?;

    if let Some(source) = opts.source {
        if !source.exists() {
            return Err(Tsl51IngestError::SourceNotFound(source));
        }
        for src in collect_source_metadata_files(&source)? {
            let name = src
                .file_name()
                .ok_or_else(|| Tsl51IngestError::MissingMetadata(src.clone()))?;
            fs::copy(&src, metadata_out.join(name))?;
        }
    } else {
        download_metadata_hf(&metadata_out)?;
    }

    let expert = metadata_out.join("expert_metadata.csv");
    let user = metadata_out.join("user_sign_metadata.csv");
    if !expert.is_file() || !user.is_file() {
        return Err(Tsl51IngestError::MissingMetadata(metadata_out.clone()));
    }

    let combined = metadata_out.join("combined_metadata.csv");
    let (meta_rows, num_training_rows, num_null_rows, num_classes) =
        concat_metadata(&[expert, user], &combined)?;

    let manifest = Tsl51Manifest {
        track: "tsl51_word_signs".into(),
        dataset: TSL51_REPO_ID.into(),
        metadata_dir: metadata_out.display().to_string(),
        meta_rows,
        num_classes,
        num_training_rows,
        num_null_rows,
        seq_len: SEQ_LEN_DEFAULT,
        feature_dim: FEATURE_DIM,
    };

    let manifest_path = opts.output.join("ingest_manifest.json");
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;
    Ok(manifest)
}

/// Skeleton for on-demand landmark file fetch (full training ingest later).
pub fn hf_download_landmark_skeleton(
    relative_path: &str,
    cache_dir: &Path,
) -> Result<PathBuf, Tsl51IngestError> {
    let rel = relative_path.replace('\\', "/");
    let cached = cache_dir.join(&rel);
    if cached.is_file() {
        return Ok(cached);
    }
    if let Some(parent) = cached.parent() {
        fs::create_dir_all(parent)?;
    }
    let api = Api::new().map_err(|e| Tsl51IngestError::Hub(e.to_string()))?;
    let repo = api.dataset(TSL51_REPO_ID.to_string());
    let local = repo
        .get(&rel)
        .map_err(|e| Tsl51IngestError::Hub(e.to_string()))?;
    fs::copy(&local, &cached)?;
    Ok(cached)
}
