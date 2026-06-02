//! ZIP iteration and tolerant extraction (mirrors `scripts/ingest_local.py`).

use std::fs::File;
use std::io;
use std::path::Path;

use thiserror::Error;
use zip::read::ZipArchive;
use zip::result::ZipError;

#[derive(Debug, Clone)]
pub struct ZipEntryInfo {
    pub name: String,
    pub is_dir: bool,
    pub compressed_size: u64,
    pub uncompressed_size: u64,
}

#[derive(Debug, Error)]
pub enum ZipWalkError {
    #[error("failed to open zip: {0}")]
    Open(#[from] io::Error),
    #[error("invalid zip archive: {0}")]
    Archive(#[from] ZipError),
}

/// List every member in a ZIP without extracting.
pub fn walk_zip_entries(zip_path: &Path) -> Result<Vec<ZipEntryInfo>, ZipWalkError> {
    let file = File::open(zip_path)?;
    let mut archive = ZipArchive::new(file)?;
    let mut out = Vec::with_capacity(archive.len());
    for i in 0..archive.len() {
        let entry = archive.by_index(i)?;
        out.push(ZipEntryInfo {
            name: entry.name().to_string(),
            is_dir: entry.is_dir(),
            compressed_size: entry.compressed_size(),
            uncompressed_size: entry.size(),
        });
    }
    Ok(out)
}

/// Extract members one-by-one; skip corrupted entries and continue.
pub fn extract_zip_tolerant(
    zip_path: &Path,
    out_root: &Path,
) -> Result<(usize, usize), ZipWalkError> {
    std::fs::create_dir_all(out_root)?;
    let file = File::open(zip_path)?;
    let mut archive = ZipArchive::new(file)?;
    let mut ok = 0usize;
    let mut failed = 0usize;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let name = entry.name().to_string();
        let out_path = out_root.join(&name);
        if entry.is_dir() {
            std::fs::create_dir_all(&out_path)?;
            ok += 1;
            continue;
        }
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        match (|| -> Result<(), ZipWalkError> {
            let mut outfile = File::create(&out_path)?;
            io::copy(&mut entry, &mut outfile)?;
            Ok(())
        })() {
            Ok(()) => ok += 1,
            Err(_) => {
                failed += 1;
                eprintln!("[warn] skipped corrupted member: {name}");
            }
        }
    }
    Ok((ok, failed))
}

/// Collect image paths under `root` (recursive).
pub fn count_images(root: &Path) -> usize {
    const EXTS: &[&str] = &["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"];
    walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|ext| EXTS.contains(&ext))
        })
        .count()
}
