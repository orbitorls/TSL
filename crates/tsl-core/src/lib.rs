//! Core feature contract for TSL fingerspelling (63-D) and TSL-51 (162×60).

pub mod keypoints;
pub mod labels;
pub mod npz;
pub mod scaler;
pub mod sequence;

pub use keypoints::{
    extract_and_normalize, extract_hand_landmarks, normalize_landmarks, HandLandmarks, HandResults,
    FEATURE_SIZE, NUM_LANDMARKS, WRIST_IDX,
};
pub use labels::{load_labels, load_labels_from_str};
pub use npz::{
    detect_npz_kind, load_fingerspelling_npz, load_tsl51_npz, FingerspellingFeatureCache, NpzError,
    NpzKind, Tsl51FeatureCache,
};
pub use scaler::StandardScaler;
pub use sequence::{
    csv_to_sequence, extract_holistic_frame, pad_truncate_sequence, read_landmark_csv,
    tsl51_csv_column_names, HandBlock, HolisticResults, LandmarkList, SequenceBuffer, FEATURE_DIM,
    SEQ_LEN_DEFAULT,
};
