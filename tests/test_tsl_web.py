import contextlib
import importlib
import tempfile
from pathlib import Path

import torch


def test_tsl_web_index_serves_redesigned_shell():
    webapp = importlib.import_module("tsl_web.app")

    client = webapp.app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "แปลคำศัพท์ TSL-51" in html
    assert "/static/styles.css" in html
    assert "/static/app.js" in html
    assert "@tensorflow/tfjs@4.22.0" not in html
    assert "@tensorflow/tfjs-core" not in html
    assert "@tensorflow-models/hand-pose-detection" not in html
    assert "btnStart" in html


def test_tsl_web_client_tracks_hands_separately_from_pose():
    app_js = Path("tsl_web/static/app.js").read_text(encoding="utf-8")

    assert "function getDetectionState(result, handState = null)" in app_js
    assert "hasHands: handCount > 0" in app_js
    assert '"เห็นร่างกาย แต่ TFJS ยังไม่พบมือ"' in app_js
    assert '"โหลดตัวจับมือ TFJS ไม่สำเร็จ"' in app_js
    assert 'const source = (leftHandLandmarks || rightHandLandmarks) ? "Holistic"' in app_js
    assert "if (detection.hasHands)" in app_js


def test_tsl_web_client_uses_tfjs_hand_detector():
    app_js = Path("tsl_web/static/app.js").read_text(encoding="utf-8")

    assert "let tfLib = null;" in app_js
    assert "let handPoseDetectionLib = null;" in app_js
    assert "tf.setWasmPaths" in app_js
    assert "let handDetector = null;" in app_js
    assert "const VISION_BUNDLE_URLS" in app_js
    assert "const TFJS_SCRIPT_URLS" in app_js
    assert "const TFJS_WASM_SCRIPT_URLS" in app_js
    assert "const HAND_POSE_SCRIPT_URLS" in app_js
    assert "async function ensureLibrary(" in app_js
    assert "async function loadFirstModule(" in app_js
    assert "const HAND_SCORE_THRESHOLD = 0.15;" in app_js
    assert "async function initHandDetector()" in app_js
    assert "async function detectHands(videoFrame)" in app_js
    assert "createDetector(" in app_js
    assert 'runtime: "tfjs"' in app_js
    assert 'modelType: "lite"' in app_js
    assert "detectHands(video)" in app_js
    assert "const detection = getDetectionState(result, handState);" in app_js
    assert "handSource: source" in app_js
    assert "ตรวจพบมือ" in app_js
    assert "async function detectHolisticFrame" in app_js
    assert "if (result && typeof result.then === \"function\")" in app_js
    assert "const isNormalized =" in app_js


def test_tsl_web_client_allows_holistic_fallback_without_tfjs_detector():
    app_js = Path("tsl_web/static/app.js").read_text(encoding="utf-8")

    start_camera = app_js.split("async function startCamera()", 1)[1]
    start_camera = start_camera.split("function stopCamera()", 1)[0]

    assert "if (!holisticLandmarker)" in start_camera
    assert "!handDetector" not in start_camera


def test_tsl_web_client_keeps_pose_overlay_when_hands_are_missing():
    app_js = Path("tsl_web/static/app.js").read_text(encoding="utf-8")

    pose_branch = app_js.split("} else if (detection.hasPose || detection.hasFace) {", 1)[1]
    pose_branch = pose_branch.split("} else {", 1)[0]

    assert "drawLandmarks(result, detectionForRender);" in pose_branch
    assert "detectionForRender" in pose_branch
    assert "clearLandmarks();" not in pose_branch


def test_tsl_web_client_clamps_landmark_points_to_canvas():
    app_js = Path("tsl_web/static/app.js").read_text(encoding="utf-8")

    assert "Math.min(width, Math.max(0, (1 - point.x) * width))" in app_js
    assert "Math.min(height, Math.max(0, point.y * height))" in app_js


def test_tsl_web_client_resets_stale_landmark_buffer_when_hands_disappear():
    app_js = Path("tsl_web/static/app.js").read_text(encoding="utf-8")

    assert "let missingHandFrames = 0;" in app_js
    assert "const MISSING_HAND_RESET_FRAMES" in app_js
    assert "function resetLiveBuffer()" in app_js
    assert "missingHandFrames >= MISSING_HAND_RESET_FRAMES" in app_js


def test_tsl_web_predict_rejects_missing_landmarks(monkeypatch):
    webapp = importlib.import_module("tsl_web.app")

    monkeypatch.setattr(webapp, "model", object())
    client = webapp.app.test_client()
    response = client.post("/predict", json={})

    assert response.status_code == 400
    assert response.get_json() == {"error": "No landmark data"}


def test_tsl_web_predict_rejects_wrong_feature_length(monkeypatch):
    webapp = importlib.import_module("tsl_web.app")

    monkeypatch.setattr(webapp, "model", object())
    monkeypatch.setattr(webapp, "expected_input_dim", 162)
    client = webapp.app.test_client()
    response = client.post("/predict", json={"landmarks": [[0.0] * 161]})

    assert response.status_code == 400
    assert response.get_json() == {"error": "Invalid landmark data (expected 162-dim)"}


def test_tsl_web_predict_accepts_single_frame(monkeypatch):
    webapp = importlib.import_module("tsl_web.app")

    def fake_predict(_landmarks):
        return {
            "top1": {"label": "ดี", "confidence": 0.8},
            "top3": [{"label": "ดี", "confidence": 0.8}],
            "buffer_filled": 1.0,
        }

    monkeypatch.setattr(webapp, "predict", fake_predict)
    monkeypatch.setattr(webapp, "model", object())
    monkeypatch.setattr(webapp, "sequence_mode", False)
    monkeypatch.setattr(webapp, "INFERENCE_EVERY_N", 1)
    monkeypatch.setattr(webapp, "expected_input_dim", 162)
    webapp.LANDMARK_BUFFER.clear()
    webapp._session_buffers.clear()
    webapp._session_cached_predictions.clear()

    client = webapp.app.test_client()
    response = client.post("/predict", json={"landmarks": [0.0] * 162})

    assert response.status_code == 200
    assert response.get_json() == {
        "top1": {"label": "ดี", "confidence": 0.8},
        "top3": [{"label": "ดี", "confidence": 0.8}],
        "buffer_filled": 1.0,
    }


def test_tsl_web_predict_rejects_malformed_landmarks(monkeypatch):
    webapp = importlib.import_module("tsl_web.app")

    monkeypatch.setattr(webapp, "model", object())
    monkeypatch.setattr(webapp, "expected_input_dim", 162)
    client = webapp.app.test_client()

    response = client.post("/predict", json={"landmarks": [1.0]})
    assert response.status_code == 400
    assert response.get_json() == {"error": "Invalid landmark data (expected 162-dim)"}

    response = client.post("/predict", json={"landmarks": [["x"] * 162]})
    assert response.status_code == 400
    assert response.get_json() == {"error": "Invalid landmark data (must be numeric)"}

    response = client.post("/predict", json={"landmarks": [[float("nan")] * 162]})
    assert response.status_code == 400
    assert response.get_json() == {"error": "Invalid landmark data (must be finite numeric values)"}


def test_tsl_web_health_endpoint():
    webapp = importlib.import_module("tsl_web.app")

    response = webapp.app.test_client().get("/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["buffer_limit"] == webapp.TARGET_FRAMES
    assert payload["input_dim"] == webapp.expected_input_dim
    assert "model_loaded" in payload
    assert "sessions" in payload


def test_tsl_web_resolves_default_model_path():
    webapp = importlib.import_module("tsl_web.app")

    resolved = webapp._resolve_default_model_path()
    assert resolved.exists()


def test_tsl_web_predict_returns_503_when_model_not_loaded(monkeypatch):
    webapp = importlib.import_module("tsl_web.app")

    monkeypatch.setattr(webapp, "ensure_model_loaded", lambda: None)
    monkeypatch.setattr(webapp, "model", None)
    client = webapp.app.test_client()
    response = client.post("/predict", json={"landmarks": [[0.0] * 162]})

    assert response.status_code == 503
    assert response.get_json() == {"error": "Model not loaded"}


def test_tsl_web_predict_accepts_landmark_batch(monkeypatch):
    webapp = importlib.import_module("tsl_web.app")

    def fake_predict(_landmarks):
        return {
            "top1": {"label": "สวัสดี", "confidence": 0.91},
            "top3": [
                {"label": "สวัสดี", "confidence": 0.91},
                {"label": "ขอบคุณ", "confidence": 0.07},
                {"label": "บ้าน", "confidence": 0.02},
            ],
            "buffer_filled": 1.0,
        }

    monkeypatch.setattr(webapp, "predict", fake_predict)
    monkeypatch.setattr(webapp, "model", object())
    monkeypatch.setattr(webapp, "expected_input_dim", 162)
    webapp.LANDMARK_BUFFER.clear()
    webapp._session_buffers.clear()
    webapp._session_cached_predictions.clear()
    webapp.cached_prediction = {
        "top1": {"label": "", "confidence": 0.0},
        "top3": [],
        "buffer_filled": 0.0,
    }

    client = webapp.app.test_client()
    response = client.post("/predict", json={"landmarks": [[0.0] * 162 for _ in range(30)]})

    assert response.status_code == 200
    assert response.get_json() == {
        "top1": {"label": "สวัสดี", "confidence": 0.91},
        "top3": [
            {"label": "สวัสดี", "confidence": 0.91},
            {"label": "ขอบคุณ", "confidence": 0.07},
            {"label": "บ้าน", "confidence": 0.02},
        ],
        "buffer_filled": 1.0,
    }


def test_tsl_web_predict_rejects_too_many_frames(monkeypatch):
    webapp = importlib.import_module("tsl_web.app")

    monkeypatch.setattr(webapp, "model", object())
    monkeypatch.setattr(webapp, "expected_input_dim", 162)
    client = webapp.app.test_client()
    response = client.post("/predict", json={"landmarks": [[0.0] * 162 for _ in range(121)]})

    assert response.status_code == 400
    assert response.get_json() == {"error": "Too many landmark frames"}


def test_tsl_web_session_eviction_maintains_limit(monkeypatch):
    webapp = importlib.import_module("tsl_web.app")

    monkeypatch.setattr(webapp, "MAX_ACTIVE_SESSIONS", 5)
    webapp._session_buffers.clear()
    webapp._session_cached_predictions.clear()
    webapp._session_last_active.clear()
    webapp._model_load_error = None
    webapp.model = object()
    webapp.MAX_ACTIVE_SESSIONS = 5

    client = webapp.app.test_client()
    payload = {"landmarks": [[0.0] * webapp.expected_input_dim for _ in range(1)]}
    for idx in range(10):
        headers = {"X-TSL-Session": f"sess-{idx}"}
        response = client.post("/predict", json=payload, headers=headers)
        assert response.status_code in (200, 201, 204, 400)

    assert len(webapp._session_buffers) <= webapp.MAX_ACTIVE_SESSIONS


def test_tsl_web_defaults_sequence_mode_for_legacy_gru_checkpoint(monkeypatch):
    webapp = importlib.import_module("tsl_web.app")

    checkpoint = {
        "model": "gru",
        "classes": ["a", "b"],
        "input_dim": 162,
        "num_classes": 2,
        "mean": [0.0] * 162,
        "std": [1.0] * 162,
        "config": {},
        "state_dict": webapp.GRUModel(
            input_dim=162,
            num_classes=2,
        ).state_dict(),
        "accuracy": 100,
    }

    ckpt_file = Path(tempfile.gettempdir()) / "tsl_web_legacy_gru_checkpoint.pt"
    torch.save(checkpoint, ckpt_file)

    original_path = webapp.MODEL_PATH
    try:
        webapp.MODEL_PATH = str(ckpt_file)
        webapp._model_load_error = None
        webapp.model = None
        webapp.idx_to_label = {}
        webapp.mean = []
        webapp.std = []
        webapp.norm_std = [1.0]
        webapp._model_warmup_done = False
        webapp._session_buffers.clear()
        webapp._session_cached_predictions.clear()
        webapp._session_last_active.clear()
        webapp.ensure_model_loaded()

        assert webapp.sequence_mode is True
    finally:
        webapp.MODEL_PATH = original_path
        webapp.model = None
        webapp._model_load_error = None
        if ckpt_file.exists():
            with contextlib.suppress(OSError):
                ckpt_file.unlink()
