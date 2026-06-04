"""
review_queue_interactive.py — ดูและตัดสินคลิปใน review queue แบบ interactive

เปิดคลิปแต่ละคลิปในหน้าต่าง OpenCV แล้วกดปุ่มเพื่อตัดสิน:
  a = approved   (คลิปถูกต้อง ใช้ได้)
  r = rejected   (คลิปผิดหรือคุณภาพต่ำ)
  s = skip       (ข้ามไปก่อน ตัดสินทีหลัง)
  q = quit       (บันทึกและออก)
  Space = เล่น/หยุดวิดีโอ

Output: อัปเดต review_decision ใน queue CSV ตรงๆ

การใช้งาน:
  python scripts/review_queue_interactive.py
  python scripts/review_queue_interactive.py --queue work/tsl51_review_queue.csv
  python scripts/review_queue_interactive.py --show-approved  # รวมแสดงที่ approved แล้ว
"""

from __future__ import annotations

import argparse
import csv
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

try:
    from PIL import Image, ImageDraw, ImageFont
    _PILLOW = True
except ImportError:
    _PILLOW = False

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_LEGACY = REPO_ROOT / "python-legacy"
FONT_PATH = PY_LEGACY / "assets" / "NotoSansThai-Regular.ttf"

VALID_DECISIONS = {"approved", "rejected", ""}


def _draw_text(frame: np.ndarray, text: str, pos: tuple[int, int],
               size: int = 30, color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ดูและตัดสินคลิปใน review queue แบบ interactive")
    p.add_argument(
        "--queue",
        type=Path,
        default=REPO_ROOT / "work" / "tsl51_review_queue.csv",
        help="ไฟล์ CSV ของ review queue (default: work/tsl51_review_queue.csv)",
    )
    p.add_argument(
        "--show-approved",
        action="store_true",
        help="แสดงคลิปที่ตัดสินแล้ว (approved/rejected) ด้วย แทนที่จะข้ามไป",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Root ของ repository (default: ค้นหาอัตโนมัติ)",
    )
    return p.parse_args()


def read_queue(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        return [dict(row) for row in reader], list(reader.fieldnames)


def write_queue(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_video_path(row: dict[str, str], repo_root: Path) -> Path | None:
    p = Path(row.get("path", ""))
    if p.is_absolute():
        return p if p.exists() else None
    resolved = repo_root / p
    return resolved if resolved.exists() else None


def review_clip(
    video_path: Path,
    row: dict[str, str],
    clip_index: int,
    total: int,
    repo_root: Path,
) -> str | None:
    """
    Play one video in a window. Returns the decision:
      'approved', 'rejected', 'skip', 'quit'
    Returns None if the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] ไม่สามารถเปิดวิดีโอ: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 25.0
    frame_delay_ms = max(1, int(1000.0 / fps))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = total_frames / fps if fps > 0 else 0.0

    start_s = float(row.get("start_s") or 0.0)
    end_s_raw = row.get("end_s") or ""
    end_s = float(end_s_raw) if end_s_raw.strip() else duration_s

    label = row.get("label") or "?"
    current_decision = (row.get("review_decision") or "").strip()
    source_url = row.get("source_url") or ""

    # Seek to start
    if start_s > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)

    paused = False
    decision: str | None = None
    loop = True

    print(f"\n[{clip_index}/{total}] {label}")
    print(f"  วิดีโอ: {video_path.name}")
    print(f"  เวลา: {start_s:.1f}s – {end_s:.1f}s")
    if source_url:
        print(f"  URL: {source_url}")
    print(f"  ปัจจุบัน: {current_decision or 'ยังไม่ตัดสิน'}")
    print("  a=approved  r=rejected  s=skip  q=quit  Space=pause")

    cv2.namedWindow("TSL-51 Review", cv2.WINDOW_NORMAL)

    while loop:
        if not paused:
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if pos_ms >= end_s:
                # Loop back to start
                cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)

            ok, frame = cap.read()
            if not ok or frame is None:
                # End of video — loop
                cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

            h, w = frame.shape[:2]
            banner_h = 130

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
            cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

            # Top bar: index + label + current decision
            dec_color = (50, 200, 50) if current_decision == "approved" else \
                        (50, 50, 220) if current_decision == "rejected" else (200, 200, 200)
            frame = _draw_text(frame, f"[{clip_index}/{total}]  {label}", (10, 10), 34, (50, 255, 120))
            frame = _draw_text(frame, f"ปัจจุบัน: {current_decision or 'ยังไม่ตัดสิน'}",
                               (10, 60), 26, dec_color)
            frame = _draw_text(frame, video_path.name[:60], (10, 95), 20, (160, 160, 160))

            # Bottom bar: controls
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame = _draw_text(frame, f"a=approved  r=rejected  s=skip  q=quit  Space=pause",
                               (10, h - 50), 22, (200, 200, 200))
            frame = _draw_text(frame, f"{pos:.1f}s / {end_s:.1f}s",
                               (10, h - 22), 20, (150, 150, 150))

            cv2.imshow("TSL-51 Review", frame)

        key = cv2.waitKey(frame_delay_ms if not paused else 100) & 0xFF

        if key == ord("a"):
            decision = "approved"
            loop = False
        elif key == ord("r"):
            decision = "rejected"
            loop = False
        elif key == ord("s"):
            decision = "skip"
            loop = False
        elif key == ord("q"):
            decision = "quit"
            loop = False
        elif key == ord(" "):
            paused = not paused

    cap.release()
    return decision


def main() -> None:
    args = parse_args()

    if not args.queue.exists():
        sys.exit(f"[ERROR] ไม่พบไฟล์ queue: {args.queue}\n"
                 "       ระบุ --queue <path> หรือตรวจสอบ work/tsl51_review_queue.csv")

    rows, fieldnames = read_queue(args.queue)

    # Add review_decision column if missing
    if "review_decision" not in fieldnames:
        fieldnames.append("review_decision")
        for row in rows:
            row.setdefault("review_decision", "")

    pending = [
        r for r in rows
        if not (r.get("review_decision") or "").strip() or args.show_approved
    ]

    print(f"[INFO] Queue: {args.queue}")
    print(f"[INFO] รวม {len(rows)} คลิป — รอตัดสิน {len(pending)} คลิป")
    print()

    if not pending:
        print("[INFO] ไม่มีคลิปที่รอตัดสิน ใช้ --show-approved เพื่อดูทั้งหมด")
        return

    changed = 0
    quit_early = False

    for i, row in enumerate(pending, start=1):
        vid_path = resolve_video_path(row, args.repo_root)
        if vid_path is None:
            print(f"[WARN] ไม่พบไฟล์: {row.get('path')} — ข้ามไป")
            continue

        result = review_clip(vid_path, row, i, len(pending), args.repo_root)

        if result is None:
            continue
        if result == "skip":
            print(f"  → ข้ามไป (ยังไม่ตัดสิน)")
        elif result == "quit":
            print(f"  → ออก (บันทึกผลที่ตัดสินแล้ว)")
            quit_early = True
            break
        else:
            # Find the corresponding row in original `rows` and update
            for orig_row in rows:
                if (orig_row.get("video_id") == row.get("video_id")
                        and orig_row.get("label") == row.get("label")):
                    orig_row["review_decision"] = result
                    changed += 1
                    print(f"  → {result}")
                    break

    cv2.destroyAllWindows()

    # Save
    write_queue(args.queue, rows, fieldnames)
    print(f"\n[DONE] บันทึก: {args.queue}")
    print(f"       เปลี่ยน {changed} คลิป")

    # Summary
    approved = sum(1 for r in rows if (r.get("review_decision") or "").strip() == "approved")
    rejected = sum(1 for r in rows if (r.get("review_decision") or "").strip() == "rejected")
    pending_remaining = sum(1 for r in rows if not (r.get("review_decision") or "").strip())
    print(f"       approved={approved}  rejected={rejected}  รอ={pending_remaining}")

    if approved > 0 and not quit_early:
        print()
        print("ขั้นตอนถัดไป — รัน pipeline เพื่อ unlock ชุดทดสอบ:")
        print("  python scripts/run_reviewed_external_pipeline.py")
    elif pending_remaining > 0:
        print()
        print(f"ยังมี {pending_remaining} คลิปที่รอตัดสิน รัน script ใหม่เพื่อดูต่อ:")
        print(f"  python scripts/review_queue_interactive.py --queue {args.queue}")


if __name__ == "__main__":
    main()
