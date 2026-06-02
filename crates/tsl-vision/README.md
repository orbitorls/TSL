# tsl-vision

Landmark backends for live webcam inference. Outputs match the Python contract in
`src/keypoints.py` (63-D normalized hand) and `src/sequence_keypoints.py` (162-D
shoulder-anchored holistic per frame).

## Model paths

Place ONNX exports under the repo `artifacts/vision/` directory (gitignored). Suggested layout:

| File | Track | Notes |
|------|-------|--------|
| `artifacts/vision/hand_landmarker.onnx` | Fingerspelling | MediaPipe Hand Landmarker–compatible 21×3 output |
| `artifacts/vision/pose_landmarker.onnx` | TSL-51 | Pose subset for indices 11–16 |
| `artifacts/vision/face_landmarker.onnx` | TSL-51 | Face indices 105, 70, 300, 334, 61, 291 |

Holistic mode uses pose + face + left/right hand models (or a single fused graph once mapped).

### Default paths in code

```rust
use std::path::PathBuf;
use tsl_vision::onnx::OnnxModelPaths;

let paths = OnnxModelPaths::from_repo_root(PathBuf::from("."));
// → artifacts/vision/hand_landmarker.onnx, pose_landmarker.onnx, face_landmarker.onnx
```

Override with `OnnxModelPaths { hand_landmarker, pose_landmarker, face_landmarker }`.

### Exporting models (one-off, Python)

Use MediaPipe Tasks or community ONNX ports; verify against `tests/golden/` once
`export_golden.py` is available. Until ONNX parity is proven, `StubBackend` returns
`None` and demos can feed precomputed landmarks.

## Backends

| Type | Feature | Status |
|------|---------|--------|
| `StubBackend` | (default) | Compiles; always returns `None` |
| `OnnxLandmarkBackend` | `onnx` | Path validation + inference skeleton (`ort` not wired yet) |

```bash
cargo check -p tsl-vision
cargo check -p tsl-vision --features onnx
```

## `LandmarkBackend`

- `detect_hand` → `HandFrame` (63 floats, normalized)
- `detect_holistic` → `HolisticFrame` (162 floats, shoulder anchor)

Downstream crates (`tsl-core` `SequenceBuffer`, `tsl-infer`) consume these vectors unchanged.
