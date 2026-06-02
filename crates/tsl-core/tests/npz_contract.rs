//! NPZ contract tests against Python-generated fixtures.

use std::path::PathBuf;

use tsl_core::npz::{detect_npz_kind, load_fingerspelling_npz, load_tsl51_npz, NpzKind};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

#[test]
fn load_fingerspelling_smoke_npz() {
    let path = fixture("fs_smoke.npz");
    let cache = load_fingerspelling_npz(&path).expect("load fs npz");
    assert_eq!(cache.feature_size, 63);
    assert_eq!(cache.num_samples(), 3);
    assert_eq!(cache.y_train, vec![0, 1, 0]);
    assert!(!cache.class_names.is_empty() || cache.num_classes() >= 2);
}

#[test]
fn load_tsl51_smoke_npz() {
    let path = fixture("tsl51_smoke.npz");
    let cache = load_tsl51_npz(&path).expect("load tsl51 npz");
    assert_eq!(cache.feature_dim, 162);
    assert_eq!(cache.seq_len, 60);
    assert_eq!(cache.num_samples(), 4);
    assert_eq!(cache.signature, "smoke");
}

#[test]
fn detect_npz_kinds() {
    let file = std::fs::File::open(fixture("fs_smoke.npz")).unwrap();
    let mut archive = zip::read::ZipArchive::new(file).unwrap();
    let names: Vec<String> = (0..archive.len())
        .map(|i| archive.by_index(i).unwrap().name().to_string())
        .collect();
    assert_eq!(detect_npz_kind(&names), Some(NpzKind::Fingerspelling));
}
