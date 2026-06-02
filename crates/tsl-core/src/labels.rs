//! Load `labels.json` (list or dict with contiguous keys 0..N-1).

use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LabelsError {
    #[error("IO: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("labels must be a JSON list or object")]
    InvalidFormat,
    #[error("labels keys must be contiguous 0..{max}")]
    NonContiguous { max: usize },
}

pub fn load_labels(path: impl AsRef<Path>) -> Result<HashMap<String, String>, LabelsError> {
    let text = std::fs::read_to_string(path)?;
    load_labels_from_str(&text)
}

pub fn load_labels_from_str(text: &str) -> Result<HashMap<String, String>, LabelsError> {
    let raw: Value = serde_json::from_str(text)?;
    let labels: HashMap<String, String> = match raw {
        Value::Array(arr) => arr
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i.to_string(), value_to_string(v)))
            .collect(),
        Value::Object(map) => map
            .into_iter()
            .map(|(k, v)| (k, value_to_string(v)))
            .collect(),
        _ => return Err(LabelsError::InvalidFormat),
    };

    let n = labels.len();
    let expected: std::collections::HashSet<String> = (0..n).map(|i| i.to_string()).collect();
    if labels
        .keys()
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        != expected
    {
        return Err(LabelsError::NonContiguous {
            max: n.saturating_sub(1),
        });
    }
    Ok(labels)
}

fn value_to_string(v: Value) -> String {
    match v {
        Value::String(s) => s,
        other => other.to_string(),
    }
}
