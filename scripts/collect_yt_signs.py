"""
collect_yt_signs.py — Auto-collect Thai Sign Language clips from YouTube for each model class.

For each label in tsl51_labels.json:
  1. Search YouTube: "ภาษามือไทย {base_word}"
  2. Download top N results (default 3), max 480p, max 50MB each
  3. Get video duration, set start_s=0 end_s=min(duration, 20)
  4. Add to manifest with quality_status="prelabel"

Usage (from repo root):
    python scripts/collect_yt_signs.py \\
        --labels artifacts/tsl51/tsl51_labels.json \\
        --output-dir data/external_tsl51/videos \\
        --manifest-out data/external_tsl51/manifest.csv \\
        --results-per-sign 3

Security note: yt-dlp downloads audio/video streams only. No pickle or executable
files are loaded from YouTube. Downloaded .mp4 files are only read by cv2.VideoCapture.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_FIELDS = (
    "video_id", "path", "track", "label",
    "start_s", "end_s", "source_url",
    "split", "license_note", "quality_status",
)

# yt-dlp download format: prefer 480p MP4, small files
YT_FORMAT = (
    "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]"
    "/best[height<=480][ext=mp4]/best[height<=480]/best"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-collect YouTube clips for each TSL-51 sign label."
    )
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--manifest-out", required=True, type=Path)
    p.add_argument("--results-per-sign", type=int, default=3)
    p.add_argument("--max-duration", type=float, default=20.0,
                   help="Max seconds to use from each clip (default 20)")
    p.add_argument("--delay", type=float, default=3.0,
                   help="Seconds between downloads")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip video_ids already in output-dir (default: on)")
    return p.parse_args()


def load_labels(path: Path) -> dict[str, str]:
    """Load tsl51_labels.json → {index_str: label_name}."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return raw
    return {str(i): v for i, v in enumerate(raw)}


def sign_base_word(label: str) -> str:
    """Extract Thai base word from label like 'สวัสดี_อายุเท่ากันหรือน้อยกว่า' → 'สวัสดี'."""
    return label.split("_")[0]


def search_youtube(query: str, max_results: int, yt_dlp_exe: str,
                   max_duration_filter: int = 35) -> list[dict]:
    """Search YouTube and return list of {id, title, duration, url}.
    max_duration_filter: only return videos shorter than this many seconds.
    Short videos (< 35s) are likely single-sign demonstrations.
    """
    # Search more results than we need since duration filter reduces results
    search_url = f"ytsearch{max_results * 5}:{query}"
    cmd = [
        yt_dlp_exe,
        "--flat-playlist",
        "--print", "id,title,duration",
        "--no-warnings",
        "--quiet",
        "--match-filter", f"duration<{max_duration_filter}",
        search_url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                                encoding="utf-8", errors="replace")
        if result.returncode != 0:
            return []
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        # --print "id,title,duration" prints three separate lines per video
        videos = []
        i = 0
        while i + 2 < len(lines):
            vid_id = lines[i].strip()
            title = lines[i + 1].strip()
            dur_raw = lines[i + 2].strip()
            try:
                duration = float(dur_raw)
            except ValueError:
                duration = 30.0  # fallback
            if re.match(r'^[A-Za-z0-9_-]{11}$', vid_id):
                videos.append({
                    "id": vid_id,
                    "title": title,
                    "duration": duration,
                    "url": f"https://www.youtube.com/watch?v={vid_id}",
                })
                i += 3
            else:
                i += 1
        return videos
    except Exception as exc:  # noqa: BLE001
        print(f"    [search error] {exc}")
        return []


def _find_downloaded(video_id: str, out_dir: Path) -> Path | None:
    """Find a file that yt-dlp saved for this video_id and rename to {id}.mp4.

    Without ffmpeg, yt-dlp saves DASH video-only streams as {id}.f135.mp4 etc.
    Audio is not needed — MediaPipe uses video frames only for landmark extraction.
    """
    dest = out_dir / f"{video_id}.mp4"
    if dest.exists():
        return dest
    # yt-dlp DASH pattern: {id}.fXXX.mp4 — starts with video_id + "."
    for f in sorted(out_dir.iterdir()):
        if f.name.startswith(video_id + ".") and f.suffix in (".mp4", ".mkv", ".webm"):
            f.rename(dest)
            return dest
    return None


def download_video(video_id: str, url: str, out_dir: Path,
                   yt_dlp_exe: str, dry_run: bool) -> Path | None:
    """Download one video. Returns path to {video_id}.mp4 or None on failure.

    We request video-only MP4 (no ffmpeg required) since MediaPipe only uses
    the video stream for landmark extraction. Audio is irrelevant for training.
    """
    dest = out_dir / f"{video_id}.mp4"
    if dest.exists():
        print(f"    [skip] {video_id}.mp4 already exists")
        return dest
    existing = _find_downloaded(video_id, out_dir)
    if existing:
        return existing
    if dry_run:
        print(f"    [dry-run] would download {video_id}")
        return dest

    # Video-only format avoids needing ffmpeg for muxing. MediaPipe needs video only.
    video_only_format = (
        "bestvideo[height<=480][ext=mp4]"
        "/bestvideo[height<=360][ext=mp4]"
        "/bestvideo[ext=mp4]"
        "/best[height<=480][ext=mp4]"
        "/best[ext=mp4]"
    )
    cmd = [
        yt_dlp_exe,
        "-f", video_only_format,
        "-o", str(out_dir / "%(id)s.%(ext)s"),
        "--max-filesize", "50M",
        "--no-warnings",
        "--quiet",
        url,
    ]
    try:
        subprocess.run(cmd, timeout=120, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        print(f"    [WARN] download timed out for {video_id}")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"    [ERROR] {exc}")
        return None

    return _find_downloaded(video_id, out_dir)


def make_manifest_row(video_id: str, video_path: Path | None, label: str,
                      end_s: float, url: str, split: str) -> dict[str, str]:
    path_str = ""
    if video_path is not None:
        try:
            path_str = str(video_path.relative_to(REPO_ROOT)).replace("\\", "/")
        except ValueError:
            path_str = str(video_path).replace("\\", "/")
    return {
        "video_id": video_id,
        "path": path_str,
        "track": "tsl51",
        "label": label,
        "start_s": "0.0",
        "end_s": f"{min(end_s, 30.0):.1f}",
        "source_url": url,
        "split": split,
        "license_note": "unknown-yt",
        # "prelabel" = downloaded but not manually reviewed.
        # Change to "reviewed" after watching the clip to confirm sign is correct.
        "quality_status": "prelabel",
    }


def find_yt_dlp(out_dir: Path) -> str:
    """Find yt-dlp executable: venv Scripts first, then PATH."""
    venv_exe = REPO_ROOT / ".venv-train" / "Scripts" / "yt-dlp.exe"
    if venv_exe.exists():
        return str(venv_exe)
    # Try python -m yt_dlp
    return sys.executable.replace("python.exe", "yt-dlp.exe")


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as fh:
            existing = list(csv.DictReader(fh))

    # Merge: keep existing rows, append new rows (deduplicate by video_id+label+start_s)
    existing_keys = {(r["video_id"], r["label"], r["start_s"]) for r in existing}
    new_rows = [r for r in rows if (r["video_id"], r["label"], r["start_s"]) not in existing_keys]
    all_rows = existing + new_rows

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        writer.writerows(all_rows)
    return len(new_rows), len(all_rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.labels)
    yt_dlp_exe = find_yt_dlp(args.output_dir)
    print(f"yt-dlp: {yt_dlp_exe}")
    print(f"Labels: {len(labels)} classes")
    print(f"Results per sign: {args.results_per_sign}")
    print(f"Output dir: {args.output_dir}")
    print()

    manifest_rows: list[dict[str, str]] = []
    total_downloaded = 0
    total_failed = 0

    for idx_str, label in sorted(labels.items(), key=lambda x: int(x[0])):
        base_word = sign_base_word(label)
        query = f"ภาษามือไทย {base_word}"
        print(f"[{idx_str:>2}] {label}  →  search: {query!r}")

        videos = search_youtube(query, args.results_per_sign * 2, yt_dlp_exe)
        if not videos:
            print(f"     [no results]")
            continue

        downloaded_for_sign = 0
        for vid in videos:
            if downloaded_for_sign >= args.results_per_sign:
                break

            video_id = vid["id"]
            duration = vid["duration"]
            url = vid["url"]

            # Assign 80% train / 20% val
            split = "val" if (hash(video_id) % 5 == 0) else "train"

            print(f"     {video_id}  dur={duration:.0f}s  {vid['title'][:50]}")

            video_path = download_video(video_id, url, args.output_dir, yt_dlp_exe, args.dry_run)

            if video_path is not None or args.dry_run:
                end_s = min(duration, args.max_duration)
                row = make_manifest_row(video_id, video_path, label, end_s, url, split)
                manifest_rows.append(row)
                downloaded_for_sign += 1
                total_downloaded += 1
            else:
                total_failed += 1

            if not args.dry_run:
                time.sleep(args.delay)

        print()

    added, total = write_manifest(args.manifest_out, manifest_rows)
    print(f"Done! Downloaded: {total_downloaded}  Failed: {total_failed}")
    print(f"Manifest: {args.manifest_out}  (+{added} new rows, {total} total)")
    print()
    print("NEXT STEPS:")
    print(f"  1. Review clips and change quality_status='prelabel' -> 'reviewed' in:")
    print(f"     {args.manifest_out}")
    print(f"  2. Build NPZ cache:")
    print(f"     python scripts/build_external_dataset.py \\")
    print(f"       --manifest {args.manifest_out} --track tsl51 \\")
    print(f"       --labels {args.labels} \\")
    print(f"       --out data/external_tsl51/yt_clips.npz --target-fps 15")
    print(f"  3. Fine-tune:")
    print(f"     python scripts/train_external_augmented.py \\")
    print(f"       --track tsl51 \\")
    print(f"       --base-cache <base_npz> --external-cache data/external_tsl51/yt_clips.npz \\")
    print(f"       --out-dir .tools/yt_augmented --base-artifact-dir artifacts/tsl51 --epochs 20")


if __name__ == "__main__":
    main()
