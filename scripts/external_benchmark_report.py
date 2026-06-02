from __future__ import annotations

import argparse
import csv
import json
import sys
from html import escape
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PY_LEGACY = REPO_ROOT / "python-legacy"
sys.path.insert(0, str(PY_LEGACY))

from src.external_benchmark import summarize_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate external clip benchmark CSVs into JSON and HTML reports."
    )
    parser.add_argument("--fingerspelling", type=Path, default=None)
    parser.add_argument("--tsl51", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports" / "external_benchmark")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def summarize_track(path: Path, detected_key: str) -> dict[str, object]:
    rows = read_csv(path)
    summary = summarize_predictions(rows, detected_key=detected_key)
    summary["predictions_csv"] = str(path)
    return summary


def render_html(summary: dict[str, object]) -> str:
    cards = []
    for track, data_obj in summary["tracks"].items():
        data = data_obj if isinstance(data_obj, dict) else {}
        rows = "".join(
            f"<tr><td>{escape(str(item.get('expected')))}</td>"
            f"<td>{escape(str(item.get('predicted')))}</td>"
            f"<td>{escape(str(item.get('count')))}</td></tr>"
            for item in data.get("confusion_pairs", [])
        )
        cards.append(
            f"""
            <section class="card">
              <h2>{escape(str(track))}</h2>
              <p>Top-1: {float(data.get('top1_accuracy', 0.0)):.3f}</p>
              <p>Top-3: {float(data.get('top3_accuracy', 0.0)):.3f}</p>
              <p>Detection: {float(data.get('detection_rate', 0.0)):.3f}</p>
              <p>Rejected: {escape(str(data.get('rejected_samples', 0)))}</p>
              <table><thead><tr><th>Expected</th><th>Predicted</th><th>Count</th></tr></thead><tbody>{rows}</tbody></table>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>External Clip Benchmark</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #17202a; }}
    .card {{ border: 1px solid #d5dde5; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e5eaf0; padding: .4rem; text-align: left; }}
  </style>
</head>
<body>
  <h1>External Clip Benchmark</h1>
  {''.join(cards)}
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tracks: dict[str, object] = {}
    if args.fingerspelling:
        tracks["fingerspelling"] = summarize_track(args.fingerspelling, "detected_hand")
    if args.tsl51:
        tracks["tsl51"] = summarize_track(args.tsl51, "detected_hand")
    summary = {"tracks": tracks}
    summary_path = args.out_dir / "external_benchmark_summary.json"
    html_path = args.out_dir / "external_benchmark_report.html"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_html(summary), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "html": str(html_path)}, indent=2))


if __name__ == "__main__":
    main()
