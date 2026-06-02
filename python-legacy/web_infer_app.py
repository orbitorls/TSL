from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tsl_translate.inference import InferenceSettings, process_rgb_frame
from tsl_translate.loader import load_artifacts, save_upload_bytes
from tsl_translate.registry import ArtifactCandidate, ModelRegistry, discover_candidates
from tsl_translate.session import CameraRuntime, LoadedModel, PredictService
from tsl_translate.tracks import TRACKS, TrackSpec

UPLOAD_ROOT = ROOT / "runtime" / "web_uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


def app_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --tsl-color-bg-page:#f6f4ef;
          --tsl-color-bg-panel:#ffffff;
          --tsl-color-border:#d8d0bf;
          --tsl-color-brand:#0f4f4a;
          --tsl-color-brand-2:#13a08f;
          --tsl-color-brand-ghost:#e8f4f1;
          --tsl-color-text:#23312f;
          --tsl-color-text-subtle:#5f6660;
          --tsl-color-success:#0f8f64;
          --tsl-color-warning:#b97a00;
          --tsl-color-danger:#b91c1c;
          --tsl-font-ui:"Noto Sans Thai","Sarabun","Prompt",system-ui,sans-serif;
          --tsl-space-2:0.65rem;
          --tsl-space-3:1rem;
          --tsl-space-4:1.35rem;
          --tsl-radius-md:16px;
          --tsl-shadow-sm:0 1px 2px rgba(20,32,28,.08);
          --tsl-btn-height:44px;
          --tsl-control-radius:12px;
        }

        .stApp {
          color: var(--tsl-color-text);
          font-family: var(--tsl-font-ui);
          background:
            radial-gradient(1000px 500px at -8% -10%, #d9efe8 0%, transparent 58%),
            radial-gradient(900px 430px at 108% -10%, #fbe9d3 0%, transparent 60%),
            var(--tsl-color-bg-page);
        }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { padding-top: 1.1rem; padding-bottom: 2.2rem; max-width: 1200px; }
        .main-title {
          font-family: var(--tsl-font-ui);
          font-size: clamp(1.8rem, 2.2vw, 2.3rem);
          font-weight: 700;
          color: var(--tsl-color-brand);
          margin-bottom: 0.2rem;
        }
        .sub-title { color: var(--tsl-color-text-subtle); margin-bottom: .6rem; }
        .hero {
          border: 1px solid var(--tsl-color-border);
          border-radius: 18px;
          padding: 1rem 1.1rem;
          background: linear-gradient(115deg, rgba(15,79,74,.08), rgba(19,160,143,.03) 46%, rgba(255,255,255,.72));
          box-shadow: var(--tsl-shadow-sm);
          margin-bottom: .9rem;
        }
        .section-title {
          margin: 0 0 .45rem 0;
          color: var(--tsl-color-brand);
          font-size: 1.15rem;
          font-weight: 700;
        }
        .status-card {
          border: 1px solid var(--tsl-color-border);
          border-radius: var(--tsl-radius-md);
          padding: var(--tsl-space-4);
          background: var(--tsl-color-bg-panel);
          box-shadow: var(--tsl-shadow-sm);
        }
        .step-grid {
          display:grid;
          grid-template-columns:repeat(4,minmax(0,1fr));
          gap: .55rem;
          margin: .5rem 0 1rem;
        }
        .step {
          border: 1px solid var(--tsl-color-border);
          border-radius: 12px;
          background: #fff;
          padding: .55rem .65rem;
          font-size: .84rem;
          color: var(--tsl-color-text-subtle);
        }
        .panel {
          border: 1px solid var(--tsl-color-border);
          border-radius: var(--tsl-radius-md);
          background: var(--tsl-color-bg-panel);
          box-shadow: var(--tsl-shadow-sm);
          padding: .75rem .85rem;
          margin-bottom: .75rem;
        }
        .panel h4 {
          margin: 0 0 .4rem 0;
          color: var(--tsl-color-brand);
        }
        .tiny { font-size: .84rem; color: var(--tsl-color-text-subtle); line-height: 1.45; }
        .chip {
          display:inline-flex;align-items:center;padding:0.22rem 0.7rem;border-radius:999px;
          background:var(--tsl-color-brand-ghost);color:var(--tsl-color-brand);font-size:0.86rem;font-weight:600;
        }
        .live {
          background:#e6fff5;color:var(--tsl-color-success);border:1px solid #a7f3d0;
        }
        .warn {
          background:#fff8e8;color:var(--tsl-color-warning);border:1px solid #fcd34d;
        }
        .muted {
          background:#f3f4f6;color:#4b5563;border:1px solid #d1d5db;
        }
        .stButton > button{
          height: var(--tsl-btn-height) !important;
          border-radius: var(--tsl-control-radius) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    defaults = {
        "loaded": None,
        "loaded_track": None,
        "service": None,
        "camera": None,
        "running": False,
        "latest_rgb": None,
        "latest_overlay": "",
        "latest_topk": "",
        "latest_conf": 0.0,
        "fps": 0.0,
        "hist": [],
        "last_error": "",
        "manual_cache_dir": None,
        "busy_load": False,
        "busy_start": False,
        "was_running_once": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def cleanup_manual_cache() -> None:
    cache_dir = st.session_state.get("manual_cache_dir")
    if cache_dir:
        shutil.rmtree(cache_dir, ignore_errors=True)
        st.session_state["manual_cache_dir"] = None


def release_camera() -> None:
    cam = st.session_state.get("camera")
    if cam is not None:
        cam.close()
    st.session_state["camera"] = None
    st.session_state["running"] = False


def clear_preview_state() -> None:
    st.session_state["latest_rgb"] = None
    st.session_state["latest_overlay"] = ""
    st.session_state["latest_topk"] = ""
    st.session_state["latest_conf"] = 0.0
    st.session_state["fps"] = 0.0


def runtime_state_label(loaded: LoadedModel | None, running: bool, busy_start: bool) -> str:
    if busy_start:
        return "กำลังเปิดกล้อง"
    if running:
        return "กำลังรัน"
    if loaded is None:
        return "ยังไม่โหลดโมเดล"
    if st.session_state.get("was_running_once"):
        return "หยุดแล้ว"
    return "พร้อมเปิดกล้อง"


def load_model(
    track: TrackSpec,
    model_path: Path,
    labels_path: Path,
    scaler_path: Path,
    manifest_path: Path | None,
) -> None:
    loaded = load_artifacts(track, model_path, labels_path, scaler_path, manifest_path)
    st.session_state["loaded"] = loaded
    st.session_state["loaded_track"] = track.key
    st.session_state["service"] = PredictService(track)
    st.session_state["hist"] = []
    st.session_state["last_error"] = ""


def save_uploaded(track: TrackSpec, model_file, labels_file, scaler_file, manifest_file):
    cleanup_manual_cache()
    manifest_bytes = manifest_file.getvalue() if manifest_file else None
    manifest_name = manifest_file.name if manifest_file else None
    paths = save_upload_bytes(
        UPLOAD_ROOT,
        track,
        model_file.getvalue(),
        model_file.name,
        labels_file.getvalue(),
        labels_file.name,
        scaler_file.getvalue(),
        scaler_file.name,
        manifest_bytes,
        manifest_name,
    )
    st.session_state["manual_cache_dir"] = paths[0].parent
    return paths


def process_one_frame(track: TrackSpec, threshold: float, alpha: float, top_k: int, motion_min: float) -> None:
    loaded: LoadedModel | None = st.session_state.get("loaded")
    service: PredictService | None = st.session_state.get("service")
    camera: CameraRuntime | None = st.session_state.get("camera")
    if loaded is None or service is None or camera is None:
        return

    rgb = camera.read_rgb_flipped()
    result = process_rgb_frame(
        track,
        loaded,
        service,
        camera,
        rgb,
        InferenceSettings(threshold=threshold, alpha=alpha, top_k=top_k, motion_min=motion_min),
    )

    st.session_state["fps"] = result.fps
    st.session_state["latest_rgb"] = rgb
    st.session_state["latest_overlay"] = result.label
    st.session_state["latest_topk"] = result.topk_text
    st.session_state["latest_conf"] = result.confidence
    hist: list[float] = st.session_state["hist"]
    hist.append(result.confidence)
    if len(hist) > 100:
        del hist[:-100]


def main() -> None:
    st.set_page_config(page_title="TSL Web Inference", layout="wide")
    app_css()
    init_state()

    st.markdown(
        "<div class='hero'><div class='main-title'>TSL Web Inference Console</div>"
        "<div class='sub-title'>ระบบตรวจจับและทำนายสัญญาณผ่านเว็บแคมแบบเรียลไทม์ "
        "(แอปแปลภาษามือใหม่: <code>web/translate</code>)</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='step-grid'><div class='step'><strong>1) ตั้งค่าระบบ</strong><br/>เลือกแทร็กและแหล่งโมเดล</div>"
        "<div class='step'><strong>2) โหลดโมเดล</strong><br/>ตรวจสถานะโหลดให้ผ่าน</div>"
        "<div class='step'><strong>3) เปิดกล้อง</strong><br/>เริ่มทำนายแบบเรียลไทม์</div>"
        "<div class='step'><strong>4) ตรวจผล</strong><br/>ดู Top-K และแนวโน้ม</div></div>",
        unsafe_allow_html=True,
    )

    registry = ModelRegistry(ROOT)
    loaded: LoadedModel | None = st.session_state.get("loaded")
    running = bool(st.session_state.get("running"))
    busy_load = bool(st.session_state.get("busy_load"))
    busy_start = bool(st.session_state.get("busy_start"))

    with st.sidebar:
        st.header("ควบคุมระบบ")
        track_key = st.selectbox("แทร็กโมเดล", options=list(TRACKS), format_func=lambda k: TRACKS[k].title)
        track = TRACKS[track_key]
        source_mode = st.radio(
            "แหล่งที่มาของโมเดล", options=["ค้นหาอัตโนมัติ", "อัปโหลดด้วยตนเอง"], horizontal=False, index=0
        )
        cam_index = st.number_input("ดัชนีกล้อง", min_value=0, max_value=10, value=0, step=1)
        threshold = st.slider("เกณฑ์ความมั่นใจ (Threshold)", 0.1, 0.99, 0.7 if track_key == "fingerspelling" else 0.55, 0.01)
        alpha = st.slider("ค่าหน่วงสัญญาณ (Smoothing alpha)", 0.05, 1.0, 0.4, 0.05)
        top_k = st.slider("จำนวนผลลัพธ์สูงสุด (Top-K)", 1, 5, 3)
        motion_min = st.slider("ความเคลื่อนไหวขั้นต่ำ (TSL-51)", 0.001, 0.05, 0.008, 0.001)

        selected_candidate: ArtifactCandidate | None = None
        auto_has_candidate = False
        manual_ready = False
        if source_mode == "ค้นหาอัตโนมัติ":
            candidates = discover_candidates(registry, track)
            if not candidates:
                st.warning("ยังไม่พบชุดไฟล์ครบสำหรับแทร็กนี้ ให้เทรนหรืออัปโหลดเอง")
            else:
                select_key = f"candidate_idx_{track.key}"
                if select_key not in st.session_state:
                    st.session_state[select_key] = 0
                idx = st.selectbox(
                    "ชุดไฟล์ที่ค้นพบ",
                    options=list(range(len(candidates))),
                    format_func=lambda i: candidates[i].name,
                    key=select_key,
                )
                selected_candidate = candidates[idx]
                auto_has_candidate = True
                st.info("เลือกรุ่นล่าสุดให้อัตโนมัติแล้ว (ยังไม่โหลดโมเดล กดปุ่มโหลดโมเดลก่อนใช้งาน)")
        else:
            st.caption("อัปโหลด Model, Labels และ Scaler สำหรับแทร็กนี้")
            model_file = st.file_uploader("ไฟล์โมเดล (.keras หรือ .tflite)", type=["keras", "tflite"], key=f"model_{track.key}")
            labels_file = st.file_uploader("ไฟล์ป้ายชื่อ (.json)", type=["json"], key=f"labels_{track.key}")
            scaler_file = st.file_uploader("ไฟล์ scaler (.pkl)", type=["pkl"], key=f"scaler_{track.key}")
            manifest_file = st.file_uploader("ไฟล์ manifest (.json, ถ้ามี)", type=["json"], key=f"manifest_{track.key}")
            manual_ready = bool(model_file and labels_file and scaler_file)

        load_enabled = (auto_has_candidate if source_mode == "ค้นหาอัตโนมัติ" else manual_ready) and (not busy_load) and (not busy_start)
        start_enabled = (not running) and (not busy_load) and (not busy_start)
        stop_enabled = (running or st.session_state.get("camera") is not None) and (not busy_load) and (not busy_start)
        load_clicked = st.button("โหลดโมเดล", type="primary", use_container_width=True, disabled=not load_enabled)
        start_clicked = st.button("เริ่มกล้อง", use_container_width=True, disabled=not start_enabled)
        stop_clicked = st.button("หยุดกล้อง", use_container_width=True, disabled=not stop_enabled)

    if load_clicked:
        st.session_state["last_error"] = ""
        st.session_state["busy_load"] = True
        try:
            if source_mode == "ค้นหาอัตโนมัติ":
                if selected_candidate is None:
                    raise RuntimeError("ยังไม่ได้เลือกไฟล์ที่ค้นพบอัตโนมัติ")
                with st.spinner("กำลังโหลดโมเดล..."):
                    load_model(
                        track,
                        selected_candidate.model,
                        selected_candidate.labels,
                        selected_candidate.scaler,
                        selected_candidate.manifest,
                    )
            else:
                if not model_file or not labels_file or not scaler_file:
                    raise RuntimeError("เมื่ออัปโหลดเอง กรุณาใส่ครบ: โมเดล, labels และ scaler")
                model_path, labels_path, scaler_path, manifest_path = save_uploaded(
                    track, model_file, labels_file, scaler_file, manifest_file
                )
                with st.spinner("กำลังโหลดโมเดล..."):
                    load_model(track, model_path, labels_path, scaler_path, manifest_path)
            st.success("โหลดโมเดลสำเร็จ")
        except Exception as exc:
            st.session_state["last_error"] = str(exc)
            st.error(f"โหลดไม่สำเร็จ: {exc}")
        finally:
            st.session_state["busy_load"] = False

    if start_clicked:
        st.session_state["last_error"] = ""
        st.session_state["busy_start"] = True
        if st.session_state.get("loaded") is None:
            st.warning("โปรดโหลดโมเดลก่อนเริ่มกล้อง")
            st.session_state["busy_start"] = False
        else:
            try:
                release_camera()
                with st.spinner("กำลังเริ่มกล้อง..."):
                    st.session_state["camera"] = CameraRuntime(int(cam_index))
                    if not st.session_state["camera"].cap.isOpened():
                        raise RuntimeError(f"เปิดกล้องดัชนี {cam_index} ไม่ได้")
                    st.session_state["running"] = True
                    st.session_state["was_running_once"] = True
            except Exception as exc:
                release_camera()
                st.session_state["last_error"] = str(exc)
                st.error(f"เปิดกล้องไม่สำเร็จ: {exc} | แนะนำให้ลองเปลี่ยนดัชนีกล้อง เช่น 0 หรือ 1")
            finally:
                st.session_state["busy_start"] = False

    if stop_clicked:
        st.session_state["last_error"] = ""
        release_camera()
        clear_preview_state()

    loaded = st.session_state.get("loaded")
    running = bool(st.session_state.get("running"))
    busy_start = bool(st.session_state.get("busy_start"))
    state_text = runtime_state_label(loaded, running, busy_start)
    live_badge = (
        "<span class='chip live'>LIVE</span>"
        if running
        else (
            "<span class='chip warn'>กำลังเริ่มกล้อง...</span>"
            if busy_start
            else ("<span class='chip'>" + state_text + "</span>" if loaded else "<span class='chip muted'>" + state_text + "</span>")
        )
    )
    st.markdown(live_badge, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Runtime Health</div>", unsafe_allow_html=True)
    status_cols = st.columns(4)
    status_cols[0].metric("สถานะโมเดล", "พร้อมใช้งาน" if loaded else "ยังไม่พร้อม")
    status_cols[1].metric("สถานะกล้อง", "พร้อมใช้งาน" if running else "ยังไม่พร้อม")
    status_cols[2].metric("FPS", f"{st.session_state.get('fps', 0.0):.1f}")
    status_cols[3].metric("สถานะระบบ", state_text)

    st.markdown("<div class='section-title'>Live Inference</div>", unsafe_allow_html=True)
    left, right = st.columns([1.6, 1.0])
    with left:
        st.markdown("<div class='panel'><h4>ดูตัวอย่างสด</h4>", unsafe_allow_html=True)
        img = st.session_state.get("latest_rgb")
        if img is not None:
            st.image(img, channels="RGB", use_container_width=True)
        else:
            st.info("ภาพจากกล้องจะแสดงที่นี่")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.subheader("ผลการทำนาย")
        st.markdown("<div class='status-card'>", unsafe_allow_html=True)
        st.write(f"**อันดับ 1:** {st.session_state.get('latest_overlay', '-')}")
        st.write(f"**อันดับสูงสุด K:** {st.session_state.get('latest_topk', '-')}")
        if loaded is not None:
            st.write(f"**Backend:** {loaded.backend}")
            st.write(f"**จำนวนคลาส:** {len(loaded.labels)}")
            st.write(f"**เวลาโหลด:** {loaded.load_time_ms:.2f} ms")
            with st.expander("รายละเอียดไฟล์โมเดล", expanded=False):
                st.markdown(f"<div class='tiny'>ไฟล์โมเดล: {loaded.model_path}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='tiny'>ไฟล์ labels: {loaded.labels_path}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='tiny'>ไฟล์ scaler: {loaded.scaler_path}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>แนวโน้มความมั่นใจ</div>", unsafe_allow_html=True)
    hist = st.session_state.get("hist", [])
    if hist:
        st.line_chart(np.asarray(hist, dtype=np.float32))
    else:
        st.caption("ยังไม่มีกระแสข้อมูลทำนาย")

    if st.session_state.get("last_error"):
        st.error(f"ข้อผิดพลาดล่าสุด: {st.session_state['last_error']}")

    if st.session_state.get("running"):
        try:
            process_one_frame(track, threshold, alpha, top_k, motion_min)
            time.sleep(0.03)
            st.rerun()
        except Exception as exc:
            st.session_state["last_error"] = str(exc)
            release_camera()
            st.error(f"หยุดทำงานเนื่องจากข้อผิดพลาด: {exc}")


if __name__ == "__main__":
    main()
