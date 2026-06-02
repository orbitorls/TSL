//! Local dataset ingestion for fingerspelling (One-Stage-TFS) and TSL-51 metadata.

pub mod fingerspelling;
pub mod tsl51;
pub mod zip_walk;

pub use fingerspelling::{
    ingest_fingerspelling, FingerspellingIngestOptions, FingerspellingManifest,
};
pub use tsl51::{
    hf_download_landmark_skeleton, ingest_tsl51_metadata, Tsl51IngestOptions, Tsl51Manifest,
};
pub use zip_walk::{extract_zip_tolerant, walk_zip_entries, ZipEntryInfo};

/// MediaPipe hand keypoints: 21 landmarks × (x, y, z).
pub const FEATURE_SIZE: usize = 63;

/// Holistic sequence: 162-D per frame, 60 frames (TSL-51 convention).
pub const FEATURE_DIM: usize = 162;
pub const SEQ_LEN_DEFAULT: usize = 60;

pub const TSL51_REPO_ID: &str = "Namonpas/thai-sign-language-tsl51";
