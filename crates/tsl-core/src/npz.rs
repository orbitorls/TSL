//! NumPy `.npz` feature caches produced by `train_local_all.py`.
//!
//! Fingerspelling (`keypoints_cache.npz`): `feature_size`, `X_train`, `y_train`,
//! optional `X_test` / `y_test`, optional `class_names`.
//!
//! TSL-51 (`tsl51_features.npz`): `signature`, `feature_dim`, `seq_len`, `X`, `y`,
//! optional `class_names`.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use thiserror::Error;
use zip::read::ZipArchive;

use crate::keypoints::FEATURE_SIZE;
use crate::sequence::{FEATURE_DIM, SEQ_LEN_DEFAULT};

#[derive(Debug, Error)]
pub enum NpzError {
    #[error("zip: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid npy in {name}: {message}")]
    InvalidNpy { name: String, message: String },
    #[error("missing array {0} in npz")]
    MissingArray(String),
    #[error("contract mismatch: {0}")]
    Contract(String),
}

/// Fingerspelling NPZ (`keypoints_cache.npz`).
#[derive(Debug, Clone)]
pub struct FingerspellingFeatureCache {
    pub feature_size: usize,
    pub x_train: Vec<f32>,
    pub y_train: Vec<i32>,
    pub x_test: Vec<f32>,
    pub y_test: Vec<i32>,
    pub class_names: Vec<String>,
}

impl FingerspellingFeatureCache {
    pub fn num_samples(&self) -> usize {
        self.y_train.len()
    }

    pub fn num_classes(&self) -> usize {
        if !self.class_names.is_empty() {
            return self.class_names.len();
        }
        self.y_train
            .iter()
            .map(|&y| y as usize)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0)
    }
}

/// TSL-51 NPZ (`tsl51_features.npz`).
#[derive(Debug, Clone)]
pub struct Tsl51FeatureCache {
    pub signature: String,
    pub feature_dim: usize,
    pub seq_len: usize,
    /// Row-major `(n, seq_len, feature_dim)`.
    pub x: Vec<f32>,
    pub y: Vec<i32>,
    pub class_names: Vec<String>,
}

impl Tsl51FeatureCache {
    pub fn num_samples(&self) -> usize {
        self.y.len()
    }

    pub fn num_classes(&self) -> usize {
        if !self.class_names.is_empty() {
            return self.class_names.len();
        }
        self.y
            .iter()
            .map(|&y| y as usize)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0)
    }
}

/// Detect cache kind from NPZ member names.
pub fn detect_npz_kind(names: &[String]) -> Option<NpzKind> {
    let has = |stem: &str| names.iter().any(|n| n.strip_suffix(".npy") == Some(stem));
    if has("X_train") && has("feature_size") {
        Some(NpzKind::Fingerspelling)
    } else if has("X") && has("feature_dim") {
        Some(NpzKind::Tsl51)
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpzKind {
    Fingerspelling,
    Tsl51,
}

pub fn load_fingerspelling_npz(
    path: impl AsRef<Path>,
) -> Result<FingerspellingFeatureCache, NpzError> {
    let arrays = read_npz(path)?;
    let feature_size = read_scalar_i64(&arrays, "feature_size")? as usize;
    if feature_size != FEATURE_SIZE {
        return Err(NpzError::Contract(format!(
            "feature_size expected {FEATURE_SIZE}, got {feature_size}"
        )));
    }
    let (x_train, x_shape) = read_f32_array(&arrays, "X_train")?;
    let (n_train, train_cols) = matrix_rows_cols(&x_shape);
    if train_cols != feature_size {
        return Err(NpzError::Contract(format!(
            "X_train columns expected {feature_size}, got {train_cols}"
        )));
    }
    let y_train = read_i32_vec(&arrays, "y_train")?;
    if y_train.len() != n_train {
        return Err(NpzError::Contract(format!(
            "y_train length {} != X_train rows {}",
            y_train.len(),
            n_train
        )));
    }
    let (x_test, x_test_shape) =
        read_f32_array(&arrays, "X_test").unwrap_or((Vec::new(), vec![0, feature_size]));
    let (n_test, _) = matrix_rows_cols(&x_test_shape);
    let y_test = read_i32_vec(&arrays, "y_test").unwrap_or_default();
    if n_test != y_test.len() {
        return Err(NpzError::Contract(
            "X_test row count does not match y_test".into(),
        ));
    }
    let class_names = read_class_names(&arrays).unwrap_or_default();
    Ok(FingerspellingFeatureCache {
        feature_size,
        x_train,
        y_train,
        x_test,
        y_test,
        class_names,
    })
}

pub fn load_tsl51_npz(path: impl AsRef<Path>) -> Result<Tsl51FeatureCache, NpzError> {
    let arrays = read_npz(path)?;
    let feature_dim = read_scalar_i64(&arrays, "feature_dim")? as usize;
    let seq_len = read_scalar_i64(&arrays, "seq_len")? as usize;
    if feature_dim != FEATURE_DIM {
        return Err(NpzError::Contract(format!(
            "feature_dim expected {FEATURE_DIM}, got {feature_dim}"
        )));
    }
    if seq_len != SEQ_LEN_DEFAULT {
        return Err(NpzError::Contract(format!(
            "seq_len expected {SEQ_LEN_DEFAULT}, got {seq_len}"
        )));
    }
    let signature = read_string_scalar(&arrays, "signature").unwrap_or_default();
    let (x, shape) = read_f32_array(&arrays, "X")?;
    if shape.len() != 3 || shape[1] != seq_len || shape[2] != feature_dim {
        return Err(NpzError::Contract(format!(
            "X shape expected (n, {seq_len}, {feature_dim}), got {:?}",
            shape
        )));
    }
    let y = read_i32_vec(&arrays, "y")?;
    if y.len() != shape[0] {
        return Err(NpzError::Contract(format!(
            "y length {} != X batch {}",
            y.len(),
            shape[0]
        )));
    }
    let class_names = read_class_names(&arrays).unwrap_or_default();
    Ok(Tsl51FeatureCache {
        signature,
        feature_dim,
        seq_len,
        x,
        y,
        class_names,
    })
}

fn read_npz(path: impl AsRef<Path>) -> Result<HashMap<String, Vec<u8>>, NpzError> {
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)?;
    let mut out = HashMap::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let name = entry.name().to_string();
        let mut buf = Vec::new();
        entry.read_to_end(&mut buf)?;
        out.insert(name, buf);
    }
    Ok(out)
}

fn read_class_names(arrays: &HashMap<String, Vec<u8>>) -> Result<Vec<String>, NpzError> {
    let key = arrays
        .keys()
        .find(|k| k.strip_suffix(".npy") == Some("class_names"))
        .cloned();
    let Some(key) = key else {
        return Ok(Vec::new());
    };
    let bytes = arrays
        .get(&key)
        .ok_or_else(|| NpzError::MissingArray("class_names".into()))?;
    let parsed = parse_npy(bytes, "class_names")?;
    if parsed.descr.starts_with("<U") || parsed.descr.starts_with(">U") {
        return decode_unicode_array(&parsed);
    }
    if parsed.descr == "|O" {
        return decode_object_class_names(&parsed);
    }
    Err(NpzError::InvalidNpy {
        name: "class_names".into(),
        message: format!("unsupported descr {}", parsed.descr),
    })
}

/// Python `dtype=object` string arrays are pickled; best-effort UTF-8 scan.
fn decode_object_class_names(parsed: &ParsedNpy) -> Result<Vec<String>, NpzError> {
    let n = parsed.shape.first().copied().unwrap_or(0);
    let mut names = Vec::with_capacity(n);
    let data = &parsed.data;
    let mut i = 0usize;
    while i + 1 < data.len() && names.len() < n {
        if data[i] == 0x8c {
            let len = data[i + 1] as usize;
            let start = i + 2;
            let end = start + len;
            if end <= data.len() {
                if let Ok(s) = std::str::from_utf8(&data[start..end]) {
                    if !s.is_empty()
                        && !s.contains("numpy")
                        && !s.contains("ndarray")
                        && !s.contains("multiarray")
                    {
                        names.push(s.to_string());
                    }
                }
                i = end;
                continue;
            }
        }
        i += 1;
    }
    if names.is_empty() {
        return Ok(Vec::new());
    }
    Ok(names)
}

fn decode_unicode_scalar(parsed: &ParsedNpy) -> Result<String, NpzError> {
    let chars = parsed
        .descr
        .trim_start_matches(['<', '>'])
        .trim_start_matches('U')
        .parse::<usize>()
        .map_err(|_| NpzError::InvalidNpy {
            name: "scalar".into(),
            message: format!("bad unicode descr {}", parsed.descr),
        })?;
    let bytes_per = chars * 4;
    let mut codepoints = Vec::new();
    for chunk in parsed.data[..bytes_per.min(parsed.data.len())].chunks_exact(4) {
        let cp = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if cp == 0 {
            break;
        }
        codepoints.push(cp);
    }
    Ok(codepoints_to_string(&codepoints))
}

fn codepoints_to_string(codepoints: &[u32]) -> String {
    codepoints.iter().filter_map(|&c| char::from_u32(c)).collect()
}

fn decode_unicode_array(parsed: &ParsedNpy) -> Result<Vec<String>, NpzError> {
    let count = parsed.shape.first().copied().unwrap_or(0);
    let chars = parsed
        .descr
        .trim_start_matches(['<', '>'])
        .trim_start_matches('U')
        .parse::<usize>()
        .map_err(|_| NpzError::InvalidNpy {
            name: "class_names".into(),
            message: format!("bad unicode descr {}", parsed.descr),
        })?;
    let bytes_per = chars * 4;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let start = i * bytes_per;
        let end = start + bytes_per;
        if end > parsed.data.len() {
            break;
        }
        let mut codepoints = Vec::new();
        for chunk in parsed.data[start..end].chunks_exact(4) {
            let cp = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if cp == 0 {
                break;
            }
            codepoints.push(cp);
        }
        out.push(codepoints_to_string(&codepoints));
    }
    Ok(out)
}

fn read_scalar_i64(arrays: &HashMap<String, Vec<u8>>, stem: &str) -> Result<i64, NpzError> {
    let parsed = read_npy_entry(arrays, stem)?;
    if !parsed.shape.is_empty() {
        return Err(NpzError::InvalidNpy {
            name: stem.into(),
            message: "expected scalar".into(),
        });
    }
    match parsed.descr.as_str() {
        "<i8" | ">i8" => {
            let v = i64::from_le_bytes(parsed.data[..8].try_into().unwrap());
            Ok(v)
        }
        "<i4" | ">i4" => {
            let v = i32::from_le_bytes(parsed.data[..4].try_into().unwrap());
            Ok(i64::from(v))
        }
        "<i2" | ">i2" => {
            let v = i16::from_le_bytes(parsed.data[..2].try_into().unwrap());
            Ok(i64::from(v))
        }
        "<i1" | "|i1" | "i1" => Ok(i64::from(parsed.data[0] as i8)),
        "<u1" | "|u1" | "u1" => Ok(i64::from(parsed.data[0])),
        other => Err(NpzError::InvalidNpy {
            name: stem.into(),
            message: format!("unsupported scalar dtype {other}"),
        }),
    }
}

fn read_string_scalar(arrays: &HashMap<String, Vec<u8>>, stem: &str) -> Result<String, NpzError> {
    let parsed = read_npy_entry(arrays, stem)?;
    if parsed.descr.starts_with("<U") || parsed.descr.starts_with(">U") {
        if parsed.shape.is_empty() {
            return decode_unicode_scalar(&parsed);
        }
        let mut names = decode_unicode_array(&parsed)?;
        return names.pop().ok_or_else(|| NpzError::InvalidNpy {
            name: stem.into(),
            message: "empty unicode array".into(),
        });
    }
    if parsed.descr == "|O" {
        let names = decode_object_class_names(&parsed)?;
        return names
            .into_iter()
            .next()
            .ok_or_else(|| NpzError::InvalidNpy {
                name: stem.into(),
                message: "could not decode object string scalar".into(),
            });
    }
    if parsed.descr == "|S0" || parsed.descr.starts_with("|S") {
        let s = std::str::from_utf8(&parsed.data)
            .map_err(|e| NpzError::InvalidNpy {
                name: stem.into(),
                message: e.to_string(),
            })?
            .trim_end_matches('\0');
        return Ok(s.to_string());
    }
    Err(NpzError::InvalidNpy {
        name: stem.into(),
        message: format!("unsupported string dtype {}", parsed.descr),
    })
}

fn read_i32_vec(arrays: &HashMap<String, Vec<u8>>, stem: &str) -> Result<Vec<i32>, NpzError> {
    let parsed = read_npy_entry(arrays, stem)?;
    if parsed.descr != "<i4" && parsed.descr != ">i4" {
        return Err(NpzError::InvalidNpy {
            name: stem.into(),
            message: format!("expected <i4>, got {}", parsed.descr),
        });
    }
    let count: usize = parsed.shape.iter().product();
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let start = i * 4;
        let bytes: [u8; 4] =
            parsed.data[start..start + 4]
                .try_into()
                .map_err(|_| NpzError::InvalidNpy {
                    name: stem.into(),
                    message: "truncated i32 data".into(),
                })?;
        let v = if parsed.descr == "<i4" {
            i32::from_le_bytes(bytes)
        } else {
            i32::from_be_bytes(bytes)
        };
        out.push(v);
    }
    Ok(out)
}

fn matrix_rows_cols(shape: &[usize]) -> (usize, usize) {
    match shape {
        [] => (0, 0),
        [n] => (*n, 1),
        [n, d] => (*n, *d),
        [n, t, d] => (*n, *t * *d),
        dims => {
            let rows = dims[0];
            let cols: usize = dims[1..].iter().product();
            (rows, cols)
        }
    }
}

fn read_f32_array(
    arrays: &HashMap<String, Vec<u8>>,
    stem: &str,
) -> Result<(Vec<f32>, Vec<usize>), NpzError> {
    let parsed = read_npy_entry(arrays, stem)?;
    if parsed.descr != "<f4" && parsed.descr != ">f4" {
        return Err(NpzError::InvalidNpy {
            name: stem.into(),
            message: format!("expected <f4>, got {}", parsed.descr),
        });
    }
    let count: usize = parsed.shape.iter().product();
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let start = i * 4;
        let bytes: [u8; 4] =
            parsed.data[start..start + 4]
                .try_into()
                .map_err(|_| NpzError::InvalidNpy {
                    name: stem.into(),
                    message: "truncated f32 data".into(),
                })?;
        let v = if parsed.descr == "<f4" {
            f32::from_le_bytes(bytes)
        } else {
            f32::from_be_bytes(bytes)
        };
        out.push(v);
    }
    Ok((out, parsed.shape))
}

fn read_npy_entry(arrays: &HashMap<String, Vec<u8>>, stem: &str) -> Result<ParsedNpy, NpzError> {
    let key = format!("{stem}.npy");
    let bytes = arrays
        .get(&key)
        .ok_or_else(|| NpzError::MissingArray(stem.to_string()))?;
    parse_npy(bytes, stem)
}

struct ParsedNpy {
    descr: String,
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn parse_npy(bytes: &[u8], name: &str) -> Result<ParsedNpy, NpzError> {
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return Err(NpzError::InvalidNpy {
            name: name.into(),
            message: "missing NUMPY magic".into(),
        });
    }
    let version = bytes[6];
    let (header_len, header_start) = if version == 1 {
        if bytes.len() < 10 {
            return Err(NpzError::InvalidNpy {
                name: name.into(),
                message: "truncated v1 header".into(),
            });
        }
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else if version == 2 {
        if bytes.len() < 12 {
            return Err(NpzError::InvalidNpy {
                name: name.into(),
                message: "truncated v2 header".into(),
            });
        }
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        )
    } else {
        return Err(NpzError::InvalidNpy {
            name: name.into(),
            message: format!("unsupported npy version {version}"),
        });
    };
    let header_end = header_start + header_len;
    if bytes.len() < header_end {
        return Err(NpzError::InvalidNpy {
            name: name.into(),
            message: "truncated header".into(),
        });
    }
    let header = std::str::from_utf8(&bytes[header_start..header_end]).map_err(|e| {
        NpzError::InvalidNpy {
            name: name.into(),
            message: e.to_string(),
        }
    })?;
    let descr = parse_header_field(header, "descr").ok_or_else(|| NpzError::InvalidNpy {
        name: name.into(),
        message: "missing descr".into(),
    })?;
    let shape = parse_shape(header).map_err(|e| NpzError::InvalidNpy {
        name: name.into(),
        message: e,
    })?;
    let data_offset = ((header_end + 63) / 64) * 64;
    if bytes.len() < data_offset {
        return Err(NpzError::InvalidNpy {
            name: name.into(),
            message: "missing data segment".into(),
        });
    }
    Ok(ParsedNpy {
        descr,
        shape,
        data: bytes[data_offset..].to_vec(),
    })
}

fn parse_header_field(header: &str, key: &str) -> Option<String> {
    let needle = format!("'{key}'");
    let start = header.find(&needle)? + needle.len();
    let rest = header[start..].trim_start_matches([':', ' ']);
    if !rest.starts_with('\'') {
        return None;
    }
    let rest = &rest[1..];
    let end = rest.find('\'')?;
    Some(rest[..end].to_string())
}

fn parse_shape(header: &str) -> Result<Vec<usize>, String> {
    let start = header
        .find("'shape'")
        .ok_or_else(|| "missing shape".to_string())?;
    let after = &header[start..];
    let open = after
        .find('(')
        .ok_or_else(|| "missing shape paren".to_string())?;
    let close = after[open..]
        .find(')')
        .ok_or_else(|| "unclosed shape".to_string())?
        + open;
    let inner = after[open + 1..close].trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|part| {
            let part = part.trim();
            if part.is_empty() {
                Ok(1)
            } else {
                part.parse::<usize>()
                    .map_err(|e| format!("bad shape component {part}: {e}"))
            }
        })
        .collect()
}
