"""
download_yt_clips.py — download YouTube clips for TSL-51 dataset expansion.

Reads a sources CSV (--input) and downloads each unique URL to --output-dir as
{video_id}.mp4 (480p max, MP4 container).  Already-downloaded files are skipped.
After downloading, the script writes a companion manifest CSV (--manifest-out)
in the format required by build_external_dataset.py / src/external_dataset.py.

Usage (from repo root):
    python scripts/download_yt_clips.py \\
        --input data/external_tsl51/youtube_sources.csv \\
        --output-dir data/external_tsl51/videos \\
        --manifest-out data/external_tsl51/manifest.csv

Sources CSV format (required columns):
    video_id, source_url, label, start_s, end_s
    [optional: note, split, license_note, quality_status]

Manifest CSV produced (all 10 fields required by load_manifest):
    video_id, path, track, label, start_s, end_s,
    source_url, split, license_note, quality_status

Defaults applied when source column is absent/empty:
    track          = tsl51
    split          = train
    license_note   = unknown
    quality_status = raw   (set to "reviewed" manually after inspecting each clip)

Security note: yt-dlp downloads video data only (audio+video streams); no pickle
or executable artifacts are loaded from YouTube.  The downloaded .mp4 files are
read later by cv2.VideoCapture for MediaPipe feature extraction only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_FIELDS = (
    "video_id",
    "path",
    "track",
    "label",
    "start_s",
    "end_s",
    "source_url",
    "split",
    "license_note",
    "quality_status",
)

# yt-dlp format string: prefer 480p MP4 to keep file sizes small
YT_FORMAT = (
    "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]"
    "/best[height<=480][ext=mp4]"
    "/best[height<=480]"
    "/best"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download YouTube clips and produce a training manifest."
    )
    p.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Sources CSV with video_id, source_url, label, start_s, end_s.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to save downloaded .mp4 files.",
    )
    p.add_argument(
        "--manifest-out",
        required=True,
        type=Path,
        help="Path to write the training manifest CSV.",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between downloads (default 2.0, be polite).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading.",
    )
    return p.parse_args()


def load_sources(path: Path) -> list[dict[str, str]]:
    """Read sources CSV; require video_id, source_url, label, start_s, end_s."""
    required = {"video_id", "source_url", "label", "start_s", "end_s"}
    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or [])
        missing = required - fieldnames
        if missing:
            sys.exit(f"[ERROR] Sources CSV missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


def download_clip(url: str, video_id: str, out_dir: Path, dry_run: bool) -> Path | None:
    """Download one YouTube URL → out_dir/{video_id}.mp4.  Returns path or None on failure."""
    dest = out_dir / f"{video_id}.mp4"
    if dest.exists():
        print(f"  [skip] {video_id}.mp4 already exists ({dest.stat().st_size // 1024} KB)")
        return dest
    if dry_run:
        print(f"  [dry-run] would download {url} → {dest}")
        return dest

    try:
        import yt_dlp  # type: ignore[import]
    except ImportError:
        sys.exit(
            "[ERROR] yt-dlp not installed. Run: pip install yt-dlp>=2024.1.0\n"
            "        (or activate the project venv: .venv-train\\Scripts\\activate)"
        )

    ydl_opts = {
        "format": YT_FORMAT,
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "quiet": False,
        "no_warnings": False,
        "retries": 3,
    }
    print(f"  [download] {video_id}  {url}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as exc:  # noqa: BLE001
        print(f"  [ERROR] failed to download {video_id}: {exc}")
        return None

    # yt-dlp may choose a different extension; find what it saved
    for ext in ("mp4", "mkv", "webm"):
        candidate = out_dir / f"{video_id}.{ext}"
        if candidate.exists():
            if ext != "mp4":
                candidate.rename(dest)
            return dest
    # Fallback: search for any file starting with video_id
    for f in out_dir.iterdir():
        if f.stem == video_id:
            f.rename(dest)
            return dest
    print(f"  [WARN] downloaded but cannot locate output for {video_id}")
    return None


def build_manifest_row(row: dict[str, str], video_path: Path | None) -> dict[str, str]:
    """Convert a sources-CSV row to a manifest row."""
    path_str = str(video_path) if video_path is not None else ""
    # Make path relative to repo root if possible
    try:
        path_str = str(video_path.relative_to(REPO_ROOT)) if video_path else ""
    except ValueError:
        pass

    return {
        "video_id": row["video_id"],
        "path": path_str,
        "track": row.get("track") or "tsl51",
        "label": row["label"],
        "start_s": row["start_s"] or "0.0",
        "end_s": row["end_s"] or "0.0",
        "source_url": row["source_url"],
        "split": row.get("split") or "train",
        "license_note": row.get("license_note") or "unknown",
        # Default to "raw" — reviewer must change to "reviewed" manually after
        # inspecting each clip to confirm the sign is correct and clearly visible.
        "quality_status": row.get("quality_status") or "raw",
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sources = load_sources(args.input)
    print(f"Loaded {len(sources)} rows from {args.input}")

    # Deduplicate video_ids for downloading (one file per video_id)
    seen_ids: set[str] = set()
    manifest_rows: list[dict[str, str]] = []

    for i, row in enumerate(sources):
        video_id = row["video_id"]
        url = row["source_url"]

        if not video_id or not url:
            print(f"  [skip] row {i+2}: missing video_id or source_url")
            manifest_rows.append(build_manifest_row(row, None))
            continue

        if video_id not in seen_ids:
            seen_ids.add(video_id)
            video_path = download_clip(url, video_id, args.output_dir, args.dry_run)
            if i < len(sources) - 1 and not args.dry_run:
                time.sleep(args.delay)
        else:
            video_path = args.output_dir / f"{video_id}.mp4"
            if not video_path.exists():
                video_path = None

        manifest_rows.append(build_manifest_row(row, video_path))

    write_manifest(args.manifest_out, manifest_rows)
    print(f"\nWrote manifest: {args.manifest_out}  ({len(manifest_rows)} rows)")

    downloaded = sum(1 for r in manifest_rows if r["path"])
    raw_count = sum(1 for r in manifest_rows if r["quality_status"] == "raw")
    print(f"Downloaded: {downloaded}/{len(manifest_rows)}")
    print(
        f"\nNEXT STEP: open {args.manifest_out} and change quality_status from 'raw' to\n"
        f"  'reviewed' for each clip you have inspected and confirmed is correct.\n"
        f"  {raw_count} rows currently marked 'raw' (will be skipped by --require-reviewed).\n"
        f"Then run:\n"
        f"  python scripts/build_external_dataset.py \\\n"
        f"    --manifest {args.manifest_out} \\\n"
        f"    --track tsl51 \\\n"
        f"    --labels artifacts/tsl51/tsl51_labels.json \\\n"
        f"    --out data/external_tsl51/yt_clips.npz \\\n"
        f"    --require-reviewed"
    )


if __name__ == "__main__":
    main()
