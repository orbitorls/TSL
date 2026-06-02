//! TSL-51 holistic features and sequence buffering (port of `sequence_keypoints.py`).

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Cursor};
pub const FEATURE_DIM: usize = 162;
pub const SEQ_LEN_DEFAULT: usize = 60;
pub const NUM_POSE_LANDMARKS: usize = 6;
pub const NUM_FACE_LANDMARKS: usize = 6;
pub const NUM_HAND_LANDMARKS: usize = 21;

pub const POSE_INDICES: [usize; 6] = [11, 12, 13, 14, 15, 16];
pub const FACE_INDICES: [usize; 6] = [105, 70, 300, 334, 61, 291];
pub const LEFT_SHOULDER_IDX: usize = 11;
pub const RIGHT_SHOULDER_IDX: usize = 12;

/// Normalized landmark list indexed by MediaPipe index.
#[derive(Debug, Clone, Default)]
pub struct LandmarkList {
    /// Sparse map: index → (x, y, z). Missing indices → zeros.
    pub points: Vec<(usize, [f32; 3])>,
    pub max_index: usize,
}

impl LandmarkList {
    pub fn xyz(&self, index: usize) -> [f32; 3] {
        for (i, xyz) in &self.points {
            if *i == index {
                return *xyz;
            }
        }
        [0.0, 0.0, 0.0]
    }

    pub fn from_dense(coords: &[[f32; 3]]) -> Self {
        let max_index = coords.len().saturating_sub(1);
        let points = coords.iter().enumerate().map(|(i, c)| (i, *c)).collect();
        Self { points, max_index }
    }
}

#[derive(Debug, Default)]
pub struct HolisticResults {
    pub pose: Option<LandmarkList>,
    pub face: Option<LandmarkList>,
    pub left_hand: Option<HandBlock>,
    pub right_hand: Option<HandBlock>,
}

#[derive(Debug, Clone)]
pub struct HandBlock {
    pub coords: [f32; NUM_HAND_LANDMARKS * 3],
}

fn extract_pose_block(pose: Option<&LandmarkList>) -> [f32; 18] {
    let mut out = [0.0f32; 18];
    let pose = match pose {
        Some(p) => p,
        None => return out,
    };
    for (i, &idx) in POSE_INDICES.iter().enumerate() {
        let xyz = pose.xyz(idx);
        let base = i * 3;
        out[base..base + 3].copy_from_slice(&xyz);
    }
    out
}

fn extract_face_block(face: Option<&LandmarkList>) -> [f32; 18] {
    let mut out = [0.0f32; 18];
    let face = match face {
        Some(f) => f,
        None => return out,
    };
    for (i, &idx) in FACE_INDICES.iter().enumerate() {
        let xyz = face.xyz(idx);
        let base = i * 3;
        out[base..base + 3].copy_from_slice(&xyz);
    }
    out
}

fn extract_hand_block(hand: Option<&HandBlock>) -> [f32; NUM_HAND_LANDMARKS * 3] {
    hand.map(|h| h.coords)
        .unwrap_or([0.0; NUM_HAND_LANDMARKS * 3])
}

fn shoulder_anchor(pose: Option<&LandmarkList>) -> [f32; 3] {
    let pose = match pose {
        Some(p) => p,
        None => return [0.0, 0.0, 0.0],
    };
    let left = pose.xyz(LEFT_SHOULDER_IDX);
    let right = pose.xyz(RIGHT_SHOULDER_IDX);
    [
        (left[0] + right[0]) / 2.0,
        (left[1] + right[1]) / 2.0,
        (left[2] + right[2]) / 2.0,
    ]
}

/// Build 162-D shoulder-anchored holistic vector.
///
/// Layout (canonical interleaved order matching `tsl51_csv_column_names`):
///   dims  0–17 : 6 pose landmarks (x,y,z each)
///   dims 18–35 : 6 face landmarks (x,y,z each)
///   dims 36–161: 21 × (lh_i x,y,z, rh_i x,y,z) — interleaved per landmark
///                i.e. lh0, rh0, lh1, rh1, …, lh20, rh20
///
/// This matches the training CSV order; scaler and model weights depend on it.
pub fn extract_holistic_frame(results: &HolisticResults) -> Option<[f32; FEATURE_DIM]> {
    if results.left_hand.is_none() && results.right_hand.is_none() {
        return None;
    }

    let mut vec = [0.0f32; FEATURE_DIM];
    let mut offset = 0usize;

    let pose_block = extract_pose_block(results.pose.as_ref());
    vec[offset..offset + 18].copy_from_slice(&pose_block);
    offset += 18;

    let face_block = extract_face_block(results.face.as_ref());
    vec[offset..offset + 18].copy_from_slice(&face_block);
    offset += 18;

    // Interleave left and right hand per landmark: lh0, rh0, lh1, rh1, …
    // Must match tsl51_csv_column_names() and Python extract_holistic_frame.
    let lh = extract_hand_block(results.left_hand.as_ref());
    let rh = extract_hand_block(results.right_hand.as_ref());
    for i in 0..NUM_HAND_LANDMARKS {
        let base = offset + i * 6;
        vec[base..base + 3].copy_from_slice(&lh[i * 3..i * 3 + 3]);
        vec[base + 3..base + 6].copy_from_slice(&rh[i * 3..i * 3 + 3]);
    }

    let anchor = shoulder_anchor(results.pose.as_ref());
    for i in 0..FEATURE_DIM / 3 {
        let base = i * 3;
        vec[base] -= anchor[0];
        vec[base + 1] -= anchor[1];
        vec[base + 2] -= anchor[2];
    }

    Some(vec)
}

/// Fixed-length ring buffer with leading zero-padding.
pub struct SequenceBuffer {
    pub seq_len: usize,
    pub feature_dim: usize,
    frames: VecDeque<[f32; FEATURE_DIM]>,
}

impl SequenceBuffer {
    pub fn new(seq_len: usize) -> Self {
        assert!(seq_len >= 1, "seq_len must be >= 1");
        Self {
            seq_len,
            feature_dim: FEATURE_DIM,
            frames: VecDeque::with_capacity(seq_len),
        }
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn push(&mut self, frame: [f32; FEATURE_DIM]) {
        if self.frames.len() >= self.seq_len {
            self.frames.pop_front();
        }
        self.frames.push_back(frame);
    }

    pub fn get_padded(&self) -> Vec<Vec<f32>> {
        let mut out = vec![vec![0.0; FEATURE_DIM]; self.seq_len];
        let n = self.frames.len();
        if n > 0 {
            for (i, frame) in self.frames.iter().enumerate() {
                let row = &mut out[self.seq_len - n + i];
                row.copy_from_slice(frame);
            }
        }
        out
    }

    pub fn get_padded_flat(&self) -> Vec<f32> {
        let rows = self.get_padded();
        rows.into_iter().flatten().collect()
    }

    pub fn is_full(&self) -> bool {
        self.frames.len() >= self.seq_len
    }

    pub fn reset(&mut self) {
        self.frames.clear();
    }
}

pub fn tsl51_csv_column_names() -> Vec<String> {
    let pose = [
        "l_shoulder",
        "r_shoulder",
        "l_elbow",
        "r_elbow",
        "l_wrist",
        "r_wrist",
    ];
    let face = [
        "lbrow_outer",
        "lbrow_inner",
        "rbrow_inner",
        "rbrow_outer",
        "mouth_right",
        "mouth_left",
    ];
    let mut cols = Vec::with_capacity(FEATURE_DIM);
    for prefix in pose.iter().chain(face.iter()) {
        cols.push(format!("{prefix}_x"));
        cols.push(format!("{prefix}_y"));
        cols.push(format!("{prefix}_z"));
    }
    for i in 0..NUM_HAND_LANDMARKS {
        for hand in ["lh", "rh"] {
            cols.push(format!("{hand}_x{i}"));
            cols.push(format!("{hand}_y{i}"));
            cols.push(format!("{hand}_z{i}"));
        }
    }
    debug_assert_eq!(cols.len(), FEATURE_DIM);
    cols
}

fn parse_csv_float(value: Option<&str>) -> f32 {
    let Some(text) = value else {
        return 0.0;
    };
    let text = text.trim();
    if text.is_empty() || text.eq_ignore_ascii_case("nan") {
        return 0.0;
    }
    text.parse().unwrap_or(0.0)
}

fn anchor_feature_frames(flat: &mut [[f32; FEATURE_DIM]]) {
    for frame in flat.iter_mut() {
        let anchor = [
            (frame[0] + frame[3]) / 2.0,
            (frame[1] + frame[4]) / 2.0,
            (frame[2] + frame[5]) / 2.0,
        ];
        for i in 0..FEATURE_DIM / 3 {
            let base = i * 3;
            frame[base] -= anchor[0];
            frame[base + 1] -= anchor[1];
            frame[base + 2] -= anchor[2];
        }
    }
}

/// Parse TSL-51 landmark CSV text → `(T, FEATURE_DIM)`.
pub fn read_landmark_csv(source: &str) -> Vec<[f32; FEATURE_DIM]> {
    let col_names = tsl51_csv_column_names();
    let reader = BufReader::new(Cursor::new(source.as_bytes()));
    let mut lines = reader.lines();
    let header = match lines.next() {
        Some(Ok(h)) => h,
        _ => return Vec::new(),
    };
    let fields: Vec<&str> = header.split(',').collect();
    for col in &col_names {
        if !fields.contains(&col.as_str()) {
            panic!("CSV missing column: {col}");
        }
    }

    let mut rows: Vec<[f32; FEATURE_DIM]> = Vec::new();
    for line in lines.flatten() {
        let row_map: std::collections::HashMap<&str, &str> = {
            let parts: Vec<&str> = line.split(',').collect();
            fields
                .iter()
                .zip(parts.iter())
                .map(|(k, v)| (*k, *v))
                .collect()
        };
        let mut frame = [0.0f32; FEATURE_DIM];
        for (i, col) in col_names.iter().enumerate() {
            frame[i] = parse_csv_float(row_map.get(col.as_str()).copied());
        }
        rows.push(frame);
    }

    let mut out = rows;
    anchor_feature_frames(&mut out);
    out
}

/// Pad/truncate `(T, FEATURE_DIM)` → `(seq_len, FEATURE_DIM)` with leading pad.
pub fn pad_truncate_sequence(
    features: &[[f32; FEATURE_DIM]],
    seq_len: usize,
) -> Vec<[f32; FEATURE_DIM]> {
    let t = features.len();
    if t >= seq_len {
        return features[..seq_len].to_vec();
    }
    let pad_rows = seq_len - t;
    let mut out = vec![[0.0; FEATURE_DIM]; pad_rows];
    out.extend_from_slice(features);
    out
}

pub fn csv_to_sequence(source: &str, seq_len: usize) -> Vec<[f32; FEATURE_DIM]> {
    pad_truncate_sequence(&read_landmark_csv(source), seq_len)
}
