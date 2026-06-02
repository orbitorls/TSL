//! Top-k label formatting (port of `format_topk`).

use std::collections::HashMap;

/// Return display text and `(label, prob)` pairs for the top `k` classes.
pub fn format_topk(
    smoothed: &[f32],
    labels: &HashMap<String, String>,
    k: usize,
) -> (String, Vec<(String, f32)>) {
    if k == 0 || smoothed.is_empty() {
        return (String::new(), Vec::new());
    }
    let k = k.min(smoothed.len());
    let mut idxs: Vec<usize> = (0..smoothed.len()).collect();
    idxs.sort_by(|a, b| {
        smoothed[*b]
            .partial_cmp(&smoothed[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idxs.truncate(k);

    let mut pairs = Vec::with_capacity(k);
    let mut parts = Vec::with_capacity(k);
    for idx in idxs {
        let label = labels
            .get(&idx.to_string())
            .cloned()
            .unwrap_or_else(|| "?".to_string());
        let prob = smoothed[idx];
        pairs.push((label.clone(), prob));
        parts.push(format!("{label} {:.0}%", prob * 100.0));
    }
    (parts.join(" · "), pairs)
}
