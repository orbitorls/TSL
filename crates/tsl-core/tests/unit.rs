//! Unit tests mirroring Python pytest contracts.

use tsl_core::keypoints::{
    extract_hand_landmarks, normalize_landmarks, HandLandmarks, HandResults, FEATURE_SIZE,
    NUM_LANDMARKS,
};
use tsl_core::sequence::{
    extract_holistic_frame, HolisticResults, LandmarkList, SequenceBuffer, FEATURE_DIM,
};

#[test]
fn normalize_wrist_at_origin_and_unit_span() {
    let mut coords = [0.0f32; FEATURE_SIZE];
    for i in 0..NUM_LANDMARKS {
        coords[i * 3] = 0.1 * i as f32;
        coords[i * 3 + 1] = 0.2 * i as f32;
        coords[i * 3 + 2] = 0.05 * i as f32;
    }
    coords[9 * 3] = coords[0] + 0.3;
    coords[9 * 3 + 1] = coords[1] + 0.2;
    coords[9 * 3 + 2] = coords[2] + 0.1;

    let out = normalize_landmarks(&coords).unwrap();
    assert!((out[0].abs() + out[1].abs() + out[2].abs()) < 1e-5);
    let span = (out[27].powi(2) + out[28].powi(2) + out[29].powi(2)).sqrt();
    assert!((span - 1.0).abs() < 1e-5);
}

#[test]
fn extract_picks_highest_score() {
    let mut results = HandResults::default();
    let lo = [1.0f32; FEATURE_SIZE];
    let hi = [2.0f32; FEATURE_SIZE];
    results.hands.push(HandLandmarks { coords: lo });
    results.hands.push(HandLandmarks { coords: hi });
    results.handedness_scores = vec![0.4, 0.9];
    assert_eq!(extract_hand_landmarks(&results).unwrap(), hi);
}

#[test]
fn sequence_buffer_leading_pad() {
    let mut buf = SequenceBuffer::new(4);
    assert!(!buf.is_full());
    buf.push([1.0; FEATURE_DIM]);
    let padded = buf.get_padded();
    assert_eq!(padded.len(), 4);
    assert!(padded[0].iter().all(|&x| x == 0.0));
    assert!(padded[1].iter().all(|&x| x == 0.0));
    assert!(padded[2].iter().all(|&x| x == 0.0));
    assert!(padded[3][0] == 1.0);
}

#[test]
fn holistic_none_without_hands() {
    let results = HolisticResults {
        pose: Some(LandmarkList {
            points: vec![(11, [0.3, 0.4, 0.0]), (12, [0.7, 0.4, 0.0])],
            max_index: 12,
        }),
        ..Default::default()
    };
    assert!(extract_holistic_frame(&results).is_none());
}
