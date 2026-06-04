from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a TSL51 review queue as an HTML review sheet.")
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser.parse_args()


def read_queue(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        return [dict(row) for row in reader]


def video_src(row: dict[str, str], repo_root: Path) -> str:
    path = Path(row["path"])
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve().as_uri()


def queue_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    pending = [row for row in rows if not (row.get("review_decision") or "").strip()]
    approved = [row for row in rows if (row.get("review_decision") or "").strip().lower() == "approved"]
    rejected = [row for row in rows if (row.get("review_decision") or "").strip().lower() == "rejected"]
    return {
        "rows": len(rows),
        "pending": len(pending),
        "approved": len(approved),
        "rejected": len(rejected),
        "recommended_splits": {
            split: sum(1 for row in rows if row.get("recommended_split") == split)
            for split in ("external_test", "val", "train")
        },
    }


def render(rows: list[dict[str, str]], repo_root: Path) -> str:
    summary = queue_summary(rows)
    cards = []
    for row in rows:
        priority = html.escape(row.get("review_priority") or "")
        label = html.escape(row.get("label") or "")
        split = html.escape(row.get("recommended_split") or "")
        decision = html.escape(row.get("review_decision") or "pending")
        reviewed_label = html.escape(row.get("reviewed_label") or "")
        source_url = html.escape(row.get("source_url") or "")
        path = html.escape(row.get("path") or "")
        start = html.escape(row.get("start_s") or "")
        end = html.escape(row.get("end_s") or "")
        src = html.escape(video_src(row, repo_root))
        cards.append(
            f"""
            <section class="card">
              <div class="meta">
                <span class="priority">#{priority}</span>
                <span>{split}</span>
                <span>{decision}</span>
              </div>
              <video controls preload="metadata" src="{src}#t={start},{end}"></video>
              <dl>
                <dt>label</dt><dd>{label}</dd>
                <dt>reviewed_label</dt><dd>{reviewed_label or "-"}</dd>
                <dt>time</dt><dd>{start} - {end}</dd>
                <dt>path</dt><dd>{path}</dd>
                <dt>source</dt><dd><a href="{source_url}">{source_url}</a></dd>
              </dl>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TSL51 Review Queue</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #20242a; background: #f7f7f5; }}
    header {{ margin-bottom: 18px; }}
    .summary {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; }}
    .pill {{ border: 1px solid #c9d0d8; border-radius: 999px; padding: 6px 10px; background: #fff; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; }}
    .card {{ border: 1px solid #d4d8de; border-radius: 8px; background: #fff; padding: 12px; }}
    .meta {{ display: flex; gap: 8px; margin-bottom: 10px; font-size: 13px; color: #4b5563; }}
    .meta span {{ border: 1px solid #d6dbe1; border-radius: 999px; padding: 3px 8px; }}
    .priority {{ font-weight: 700; color: #0f4f4a; }}
    video {{ width: 100%; max-height: 280px; background: #111; border-radius: 6px; }}
    dl {{ display: grid; grid-template-columns: 110px 1fr; gap: 6px 10px; font-size: 13px; }}
    dt {{ color: #667085; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <header>
    <h1>TSL51 Review Queue</h1>
    <p>Fill review_decision as approved or rejected in the CSV. Use reviewed_label only when the proposed label is wrong.</p>
    <div class="summary">
      <span class="pill">rows: {summary["rows"]}</span>
      <span class="pill">pending: {summary["pending"]}</span>
      <span class="pill">approved: {summary["approved"]}</span>
      <span class="pill">rejected: {summary["rejected"]}</span>
      <span class="pill">external_test: {summary["recommended_splits"]["external_test"]}</span>
      <span class="pill">val: {summary["recommended_splits"]["val"]}</span>
      <span class="pill">train: {summary["recommended_splits"]["train"]}</span>
    </div>
  </header>
  <main class="grid">
    {''.join(cards)}
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    rows = read_queue(args.queue)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(rows, args.repo_root.resolve()), encoding="utf-8")
    summary = queue_summary(rows)
    summary_path = args.summary or args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "summary": str(summary_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
