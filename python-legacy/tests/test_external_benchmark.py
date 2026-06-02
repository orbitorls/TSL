from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.external_benchmark import summarize_predictions  # noqa: E402


def test_summarize_predictions_reports_top3_and_confusions() -> None:
    rows = [
        {
            "expected": "A",
            "predicted": "A",
            "top_k": "A:0.9 | B:0.1 | C:0.0",
            "detected_hand": True,
        },
        {
            "expected": "B",
            "predicted": "C",
            "top_k": "C:0.5 | B:0.4 | A:0.1",
            "detected_hand": True,
        },
        {
            "expected": "C",
            "predicted": "",
            "top_k": "",
            "detected_hand": False,
        },
    ]

    summary = summarize_predictions(rows, detected_key="detected_hand")

    assert summary["total_samples"] == 3
    assert summary["top1_correct"] == 1
    assert summary["top3_correct"] == 2
    assert summary["top1_accuracy"] == 1 / 3
    assert summary["top3_accuracy"] == 2 / 3
    assert summary["detected_samples"] == 2
    assert summary["rejected_samples"] == 1
    assert summary["confusion_pairs"] == [{"expected": "B", "predicted": "C", "count": 1}]
