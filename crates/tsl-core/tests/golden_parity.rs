//! Parity tests against `tests/golden/*.json` (tolerance from manifest, default 1e-4).

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use serde::Deserialize;
use serde_json::Value;
use tsl_core::{
    csv_to_sequence, extract_and_normalize, extract_hand_landmarks, extract_holistic_frame,
    load_labels_from_str, normalize_landmarks, pad_truncate_sequence, read_landmark_csv, HandBlock,
    HandLandmarks, HandResults, HolisticResults, LandmarkList, SequenceBuffer, StandardScaler,
    FEATURE_DIM, FEATURE_SIZE,
};

const GOLDEN_REL: &str = "../../tests/golden";

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(GOLDEN_REL)
}

#[derive(Deserialize)]
struct Manifest {
    tolerance_f32: f32,
}

fn load_manifest() -> Manifest {
    let text = fs::read_to_string(golden_dir().join("manifest.json")).expect("manifest.json");
    serde_json::from_str(&text).expect("parse manifest")
}

fn load_json(name: &str) -> Value {
    let text = fs::read_to_string(golden_dir().join(name)).expect(name);
    serde_json::from_str(&text).expect("parse json")
}

fn assert_allclose(actual: &[f32], expected: &[f32], tol: f32, ctx: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{ctx}: length mismatch {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{ctx}: index {i}: {a} vs {e} (tol {tol})"
        );
    }
}

fn vec63(v: &Value) -> [f32; FEATURE_SIZE] {
    let arr = v.as_array().expect("array");
    let mut out = [0.0f32; FEATURE_SIZE];
    for (i, x) in arr.iter().enumerate() {
        out[i] = x.as_f64().expect("f64") as f32;
    }
    out
}

fn vec162(v: &Value) -> [f32; FEATURE_DIM] {
    let arr = v.as_array().expect("array");
    let mut out = [0.0f32; FEATURE_DIM];
    for (i, x) in arr.iter().enumerate() {
        out[i] = x.as_f64().expect("f64") as f32;
    }
    out
}

fn matrix(rows: &Value) -> Vec<[f32; FEATURE_DIM]> {
    rows.as_array()
        .expect("matrix")
        .iter()
        .map(|row| {
            let mut frame = [0.0f32; FEATURE_DIM];
            for (i, x) in row.as_array().expect("row").iter().enumerate() {
                frame[i] = x.as_f64().expect("f64") as f32;
            }
            frame
        })
        .collect()
}

fn landmark_list_sparse(indices: &[(usize, [f32; 3])]) -> LandmarkList {
    let max_index = indices.iter().map(|(i, _)| *i).max().unwrap_or(0);
    LandmarkList {
        points: indices.to_vec(),
        max_index,
    }
}

#[test]
fn golden_keypoints_match_python() {
    let tol = load_manifest().tolerance_f32;
    let g = load_json("keypoints.json");

    let valid = vec63(&g["valid_landmarks_flat"]);
    let norm = normalize_landmarks(&valid).expect("normalize");
    assert_allclose(
        &norm,
        &vec63(&g["normalize_output"]),
        tol,
        "normalize_output",
    );

    let collapsed = vec63(&g["collapsed_input"]);
    assert!(normalize_landmarks(&collapsed).is_none());

    let lo = vec63(&g["multi_hand_coords_lo"]);
    let hi = vec63(&g["multi_hand_coords_hi"]);
    let results = HandResults {
        hands: vec![HandLandmarks { coords: lo }, HandLandmarks { coords: hi }],
        handedness_scores: vec![0.4, 0.9],
    };
    let extracted = extract_hand_landmarks(&results).expect("extract");
    assert_allclose(
        &extracted,
        &vec63(&g["extract_highest_confidence"]),
        tol,
        "extract",
    );

    let e2e = HandResults {
        hands: vec![HandLandmarks {
            coords: vec63(&g["e2e_coords"]),
        }],
        handedness_scores: vec![0.99],
    };
    let e2e_out = extract_and_normalize(&e2e).expect("e2e");
    assert_allclose(
        &e2e_out,
        &vec63(&g["extract_and_normalize"]),
        tol,
        "extract_and_normalize",
    );
}

#[test]
fn golden_sequence_match_python() {
    let tol = load_manifest().tolerance_f32;
    let g = load_json("sequence_keypoints.json");

    let mut buf = SequenceBuffer::new(5);
    let mut frame = [0.0f32; FEATURE_DIM];
    for (i, v) in frame.iter_mut().enumerate() {
        *v = i as f32;
    }
    buf.push(frame);
    let padded = buf.get_padded();
    let expected = matrix(&g["sequence_buffer_padded_5x162"]);
    assert_eq!(padded.len(), expected.len());
    for (i, row) in padded.iter().enumerate() {
        assert_allclose(row, &expected[i], tol, "sequence_buffer_5");
    }

    let mut buf60 = SequenceBuffer::new(60);
    for val in [1.0f32, 2.0, 3.0] {
        buf60.push([val; FEATURE_DIM]);
    }
    let padded60 = buf60.get_padded();
    let expected60 = matrix(&g["sequence_buffer_padded_60x162_3frames"]);
    for (i, row) in padded60.iter().enumerate() {
        assert_allclose(row, &expected60[i], tol, "sequence_buffer_60");
    }

    let hand_flat = g["holistic_one_hand_coords"].as_array().expect("coords");
    let mut hand_coords = [0.0f32; 63];
    for (i, x) in hand_flat.iter().enumerate() {
        hand_coords[i] = x.as_f64().expect("f64") as f32;
    }
    let holistic = HolisticResults {
        pose: Some(landmark_list_sparse(&[
            (11, [0.3, 0.4, 0.0]),
            (12, [0.7, 0.4, 0.0]),
        ])),
        face: None,
        left_hand: Some(HandBlock {
            coords: hand_coords,
        }),
        right_hand: None,
    };
    let frame_out = extract_holistic_frame(&holistic).expect("holistic");
    assert_allclose(
        &frame_out,
        &vec162(&g["holistic_one_hand"]),
        tol,
        "holistic_one_hand",
    );

    let shoulder = HolisticResults {
        pose: Some(landmark_list_sparse(&[
            (11, [0.3, 0.4, 0.0]),
            (12, [0.7, 0.4, 0.0]),
            (13, [0.5, 0.4, 0.0]),
        ])),
        face: None,
        left_hand: Some({
            let mut coords = [0.0f32; 63];
            for i in 0..21 {
                coords[i * 3] = 0.5;
                coords[i * 3 + 1] = 0.4;
                coords[i * 3 + 2] = 0.0;
            }
            HandBlock { coords }
        }),
        right_hand: None,
    };
    let shoulder_out = extract_holistic_frame(&shoulder).expect("shoulder");
    assert_allclose(
        &shoulder_out,
        &vec162(&g["shoulder_anchor_frame"]),
        tol,
        "shoulder_anchor",
    );

    let csv_text = _make_tsl51_csv_rows(2);
    let read = read_landmark_csv(&csv_text);
    let expected_read = matrix(&g["csv_read_2_frames"]);
    assert_eq!(read.len(), expected_read.len());
    for (i, row) in read.iter().enumerate() {
        assert_allclose(row, &expected_read[i], tol, "csv_read");
    }

    let seq4 = csv_to_sequence(&csv_text, 4);
    let expected_seq = matrix(&g["csv_to_sequence_4"]);
    for (i, row) in seq4.iter().enumerate() {
        assert_allclose(row, &expected_seq[i], tol, "csv_to_sequence");
    }

    let pad = pad_truncate_sequence(&read, 4);
    let expected_pad = matrix(&g["pad_truncate_short"]);
    for (i, row) in pad.iter().enumerate() {
        assert_allclose(row, &expected_pad[i], tol, "pad_truncate");
    }
}

#[test]
fn golden_labels_and_scaler_json() {
    let tol = load_manifest().tolerance_f32;
    let g = load_json("artifacts.json");

    let labels_list: Vec<String> = serde_json::from_value(g["labels_list"].clone()).unwrap();
    let labels_json = serde_json::to_string(&labels_list).unwrap();
    let loaded = load_labels_from_str(&labels_json).unwrap();
    let expected_dict: HashMap<String, String> =
        serde_json::from_value(g["labels_dict"].clone()).unwrap();
    assert_eq!(loaded, expected_dict);

    let scaler: StandardScaler =
        StandardScaler::from_json_str(&serde_json::to_string(&g["scaler_json"]).unwrap()).unwrap();
    let input = vec63(&g["scaler_transform_input"]);
    let out = scaler.transform(&input).unwrap();
    assert_allclose(
        &out,
        &g["scaler_transform_output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect::<Vec<_>>(),
        tol,
        "scaler_transform",
    );
}

fn _make_tsl51_csv_rows(num_frames: usize) -> String {
    use tsl_core::tsl51_csv_column_names;
    let cols = ["frame", "t_ms"]
        .into_iter()
        .chain(tsl51_csv_column_names().iter().map(String::as_str))
        .collect::<Vec<_>>()
        .join(",");
    let mut lines = vec![cols];
    for frame_idx in 0..num_frames {
        let mut values = vec![frame_idx.to_string(), (frame_idx * 33).to_string()];
        for prefix in [
            "l_shoulder",
            "r_shoulder",
            "l_elbow",
            "r_elbow",
            "l_wrist",
            "r_wrist",
        ] {
            match prefix {
                "l_shoulder" => values.extend(["0.3", "0.4", "0.0"].map(str::to_string)),
                "r_shoulder" => values.extend(["0.7", "0.4", "0.0"].map(str::to_string)),
                _ => values.extend(["0.5", "0.4", "0.0"].map(str::to_string)),
            }
        }
        for _ in 0..6 {
            values.extend(["0.5", "0.4", "0.0"].map(str::to_string));
        }
        for _ in 0..42 {
            values.extend(["0.5", "0.4", "0.0"].map(str::to_string));
        }
        lines.push(values.join(","));
    }
    lines.join("\n")
}
