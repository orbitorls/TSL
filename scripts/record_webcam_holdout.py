"""
record_webcam_holdout.py — อัดคลิปเว็บแคมต่อคำสัญลักษณ์ เพื่อสร้างชุดทดสอบ (holdout) ตรง domain

คลิปที่ได้จะเข้ากันได้กับ evaluate_tsl51_video.py โดยตรง:
  python scripts/evaluate_tsl51_video.py \\
      --samples <out-dir>/webcam_holdout.csv \\
      --artifact-dir artifacts/tsl51 \\
      --strategy first          # หรือ uniform

ขั้นตอนการใช้งาน:
  1. python scripts/record_webcam_holdout.py --out-dir data/webcam_holdout
  2. สคริปต์จะแสดงชื่อคำให้เซ็น ตามด้วยนับถอยหลัง 3-2-1
  3. เซ็นท่าปกติให้ครบเวลา (--record-s วินาที)
  4. กด Space = เริ่มอัดคลิปถัดไป  |  r = ทำซ้ำคลิปล่าสุด  |  s = ข้ามคำนี้  |  q = ออก

Output:
  <out-dir>/videos/<sign_label>_clip<n>.mp4   — คลิปดิบจากเว็บแคม
  <out-dir>/webcam_holdout.csv                — manifest สำหรับ evaluate_tsl51_video.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

try:
    import cv2
except ImportError:
    sys.exit("[ERROR] opencv-python is required: pip install opencv-python")
try:
    import numpy as np
except ImportError:
    sys.exit("[ERROR] numpy is required")

# ---------------------------------------------------------------------------
# Thai font rendering (optional — falls back to ASCII if pillow not available)
# ---------------------------------------------------------------------------
try:
    from PIL import Image, ImageDraw, ImageFont
    _PILLOW = True
except ImportError:
    _PILLOW = False

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_LEGACY = REPO_ROOT / "python-legacy"
sys.path.insert(0, str(PY_LEGACY))
sys.path.insert(0, str(PY_LEGACY / "src"))

FONT_PATH = PY_LEGACY / "assets" / "NotoSansThai-Regular.ttf"
MANIFEST_COLS = ["id", "path", "expected", "start_s", "end_s", "split", "source"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_labels(labels_path: Path) -> list[str]:
    raw = json.loads(labels_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return [raw[str(i)] for i in sorted(int(k) for k in raw)]
    raise ValueError(f"Unexpected labels format in {labels_path}")


def _safe_filename(label: str) -> str:
    """Convert Thai label to a safe filename component."""
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in label)


def _draw_text(frame: np.ndarray, text: str, pos: tuple[int, int],
               size: int = 36, color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Draw Thai text with Pillow fallback to cv2.putText if unavailable."""
    if _PILLOW and FONT_PATH.exists():
        try:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            font = ImageFont.truetype(str(FONT_PATH), size)
            draw.text(pos, text, font=font, fill=color)
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size / 40.0,
                (color[2], color[1], color[0]), 2, cv2.LINE_AA)
    return frame


def _record_clip(
    cap: "cv2.VideoCapture",
    out_path: Path,
    record_s: float,
    label: str,
    clip_num: int,
    total_clips: int,
    countdown_s: float = 3.0,
) -> bool:
    """
    Run a single recording session:
    - Countdown phase: show label + countdown on live feed
    - Recording phase: capture frames to MP4, show REC indicator

    Returns True if the clip was recorded, False if user aborted (q).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps_cam = cap.get(cv2.CAP_PROP_FPS)
    if fps_cam <= 0 or fps_cam > 120:
        fps_cam = 30.0

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_cam, (w, h))

    phase = "countdown"
    record_start: float | None = None
    done = False
    aborted = False

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        # Mirror display only (inference should run on unflipped, but we're
        # recording raw video for later offline evaluation with the same pipeline)
        display = cv2.flip(frame, 1)
        now = time.monotonic()

        # Status banner
        banner_h = 120
        overlay = display.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay, 0.6, display, 0.4, 0)

        if phase == "countdown":
            remaining = countdown_s - (now - (record_start or now))
            if record_start is None:
                record_start = now
            remaining = countdown_s - (now - record_start)
            if remaining <= 0:
                phase = "recording"
                record_start = now
                continue
            countdown_int = max(1, int(remaining) + 1)
            display = _draw_text(display, label, (20, h - banner_h + 8), 42, (50, 220, 50))
            display = _draw_text(
                display,
                f"คลิป {clip_num}/{total_clips}  เริ่มอัด: {countdown_int}",
                (20, h - 36), 28, (220, 220, 220))

        elif phase == "recording":
            elapsed = now - record_start
            remaining = record_s - elapsed
            if remaining <= 0 or done:
                phase = "done"
                break
            # Write the UNFLIPPED frame to file (matches eval pipeline convention)
            writer.write(frame)
            bar_w = int((w - 40) * min(elapsed / record_s, 1.0))
            cv2.rectangle(display, (20, h - 8), (20 + bar_w, h - 2), (0, 50, 220), -1)
            display = _draw_text(display, f"● REC  {label}", (20, h - banner_h + 8), 42, (50, 50, 255))
            display = _draw_text(
                display,
                f"คลิป {clip_num}/{total_clips}  เหลือ {remaining:.1f}s  q=ยกเลิก",
                (20, h - 36), 28, (180, 180, 180))

        cv2.imshow("TSL-51 Webcam Recorder", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            aborted = True
            break

    writer.release()

    if aborted:
        out_path.unlink(missing_ok=True)
        return False

    return True


def _show_standby(cap: "cv2.VideoCapture", label: str, clip_num: int,
                  total_clips: int, all_done: int, total_signs: int) -> str:
    """
    Show live feed with current sign info. Returns the key pressed:
      ' ' (space) = record next clip
      'r'         = redo previous clip
      's'         = skip this sign
      'q'         = quit entirely
    """
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue
        display = cv2.flip(frame, 1)

        banner_h = 140
        overlay = display.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay, 0.6, display, 0.4, 0)

        display = _draw_text(display, label, (20, h - banner_h + 8), 48, (50, 255, 100))
        display = _draw_text(
            display,
            f"คลิปที่ {clip_num}  |  คำที่ {all_done + 1}/{total_signs}",
            (20, h - 70), 26, (200, 200, 200))
        display = _draw_text(
            display,
            "Space=อัด  s=ข้าม  r=ทำซ้ำ  q=ออก",
            (20, h - 32), 24, (160, 160, 160))

        cv2.imshow("TSL-51 Webcam Recorder", display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord(" "):
            return "record"
        if key == ord("s"):
            return "skip"
        if key == ord("r"):
            return "redo"
        if key == ord("q"):
            return "quit"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="อัดคลิปเว็บแคมต่อคำสัญลักษณ์สำหรับชุดทดสอบ TSL-51"
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=REPO_ROOT / "artifacts" / "tsl51" / "tsl51_labels.json",
        help="ไฟล์ tsl51_labels.json (default: artifacts/tsl51/tsl51_labels.json)",
    )
    p.add_argument(
        "--signs",
        type=str,
        default=None,
        help=(
            "เซ็นเฉพาะบางคำ ระบุเป็น label ตรง ๆ คั่นด้วยจุลภาค เช่น "
            "'สวัสดี_อายุเท่ากันหรือน้อยกว่า,ขอโทษ_อายุเท่ากันหรือน้อยกว่า' "
            "หรือ index range เช่น '0:10' (default: ทุกคำ)"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "webcam_holdout",
        help="โฟลเดอร์บันทึกคลิปและ manifest (default: data/webcam_holdout)",
    )
    p.add_argument("--num-clips", type=int, default=3,
                   help="จำนวนคลิปต่อคำ (default: 3)")
    p.add_argument("--record-s", type=float, default=5.0,
                   help="ความยาวต่อคลิป วินาที (default: 5)")
    p.add_argument("--countdown-s", type=float, default=3.0,
                   help="เวลานับถอยหลังก่อนอัด วินาที (default: 3)")
    p.add_argument("--cam", type=int, default=0, help="กล้องหมายเลข (default: 0)")
    p.add_argument(
        "--split",
        choices=("holdout", "train", "val"),
        default="holdout",
        help="ค่า split ในไฟล์ manifest (default: holdout — ใช้เป็นชุดทดสอบ)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load labels
    if not args.labels.exists():
        sys.exit(f"[ERROR] Labels file not found: {args.labels}\n"
                 "       ระบุ --labels <path> หรือตรวจสอบ artifacts/tsl51/")
    all_labels = _load_labels(args.labels)
    print(f"[INFO] Loaded {len(all_labels)} labels from {args.labels}")

    # Filter signs
    if args.signs is not None:
        spec = args.signs.strip()
        if ":" in spec and not any(c.isalpha() for c in spec):
            # index range e.g. "0:10"
            parts = spec.split(":")
            start_i = int(parts[0]) if parts[0] else 0
            end_i = int(parts[1]) if len(parts) > 1 and parts[1] else len(all_labels)
            signs = all_labels[start_i:end_i]
        else:
            requested = [s.strip() for s in spec.split(",") if s.strip()]
            label_set = set(all_labels)
            missing = [s for s in requested if s not in label_set]
            if missing:
                print(f"[WARN] Signs not found in labels and will be skipped: {missing}")
            signs = [s for s in requested if s in label_set]
    else:
        signs = list(all_labels)

    if not signs:
        sys.exit("[ERROR] ไม่มีคำที่จะอัด ตรวจสอบ --signs")

    print(f"[INFO] จะอัด {len(signs)} คำ × {args.num_clips} คลิป = "
          f"{len(signs) * args.num_clips} คลิปทั้งหมด")
    print(f"[INFO] บันทึกไปที่: {args.out_dir}")
    print()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "webcam_holdout.csv"

    # Load existing manifest rows to allow resuming
    existing_rows: list[dict] = []
    recorded_ids: set[str] = set()
    if manifest_path.exists():
        with manifest_path.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if Path(row["path"]).exists():
                    existing_rows.append(row)
                    recorded_ids.add(row["id"])
        print(f"[INFO] พบ manifest เดิม — มี {len(existing_rows)} คลิปที่อัดแล้ว (ข้ามคลิปที่มีอยู่)")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        sys.exit(f"[ERROR] ไม่สามารถเปิดกล้อง {args.cam}")

    cv2.namedWindow("TSL-51 Webcam Recorder", cv2.WINDOW_NORMAL)

    new_rows: list[dict] = []
    done_signs = 0
    aborted = False

    try:
        for sign_label in signs:
            safe_name = _safe_filename(sign_label)
            clips_done = sum(
                1 for r in existing_rows
                if r["expected"] == sign_label
            )
            clips_needed = max(0, args.num_clips - clips_done)
            if clips_needed == 0:
                print(f"[SKIP] {sign_label} — ครบ {args.num_clips} คลิปแล้ว")
                done_signs += 1
                continue

            clip_n = clips_done + 1
            while clip_n <= args.num_clips:
                clip_id = f"{safe_name}_clip{clip_n:02d}"
                if clip_id in recorded_ids:
                    clip_n += 1
                    continue

                out_path = videos_dir / f"{clip_id}.mp4"

                action = _show_standby(
                    cap, sign_label, clip_n, args.num_clips, done_signs, len(signs)
                )

                if action == "quit":
                    aborted = True
                    break
                if action == "skip":
                    print(f"[SKIP] ข้ามคำ: {sign_label}")
                    break
                if action == "redo":
                    if clip_n > clips_done + 1:
                        clip_n -= 1
                        # Remove previous clip and row
                        prev_id = f"{safe_name}_clip{clip_n:02d}"
                        prev_path = videos_dir / f"{prev_id}.mp4"
                        prev_path.unlink(missing_ok=True)
                        new_rows = [r for r in new_rows if r["id"] != prev_id]
                        print(f"[REDO] ลบคลิปก่อนหน้า: {prev_id}")
                    continue

                # action == "record"
                print(f"[REC ] {sign_label}  คลิป {clip_n}/{args.num_clips} …")
                ok = _record_clip(
                    cap, out_path, args.record_s, sign_label,
                    clip_n, args.num_clips, args.countdown_s,
                )
                if not ok:
                    print("[ABORT] ผู้ใช้กด q ระหว่างอัด")
                    aborted = True
                    break

                row = {
                    "id": clip_id,
                    "path": str(out_path.resolve()),
                    "expected": sign_label,
                    "start_s": "0.0",
                    "end_s": str(round(args.record_s, 3)),
                    "split": args.split,
                    "source": "webcam_holdout",
                }
                new_rows.append(row)
                recorded_ids.add(clip_id)
                print(f"  ✓ บันทึก: {out_path.name}")
                clip_n += 1

            if aborted:
                break

            done_signs += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Write combined manifest
    all_rows = existing_rows + new_rows
    if all_rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLS)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n[DONE] บันทึก manifest: {manifest_path}")
        print(f"       {len(all_rows)} คลิปทั้งหมด"
              f"  ({len(new_rows)} คลิปใหม่  {len(existing_rows)} คลิปเดิม)")
        print()
        print("ใช้งานต่อ:")
        print(f"  python scripts/evaluate_tsl51_video.py \\")
        print(f"      --samples {manifest_path} \\")
        print(f"      --artifact-dir artifacts/tsl51 \\")
        print(f"      --strategy first")
    else:
        print("\n[INFO] ไม่มีคลิปถูกอัด")

    if aborted:
        print("\n[INFO] หยุดกลางคัน — รัน script ใหม่เพื่ออัดต่อ (คลิปที่มีอยู่จะถูกข้าม)")


if __name__ == "__main__":
    main()
