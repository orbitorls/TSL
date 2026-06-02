"""Small reporting helpers for external clip benchmarks."""

from __future__ import annotations

from collections import Counter
from typing import Mapping


def parse_topk_labels(value: object) -> list[str]:
    """Parse strings like ``A:0.9 | B:0.1`` into ordered labels."""
    text = str(value or "").strip()
    if not text:
        return []
    labels: list[str] = []
    for part in text.split("|"):
        label = part.strip().split(":", 1)[0].strip()
        if label:
            labels.append(label)
    return labels


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def summarize_predictions(
    rows: list[Mapping[str, object]], *, detected_key: str | None = None
) -> dict[str, object]:
    """Summarize top-1/top-3 accuracy and confusion pairs for CSV rows."""
    total = len(rows)
    top1 = 0
    top3 = 0
    detected = 0
    confusions: Counter[tuple[str, str]] = Counter()

    for row in rows:
        expected = str(row.get("expected") or "")
        predicted = str(row.get("predicted") or "")
        topk = parse_topk_labels(row.get("top_k"))
        if predicted and predicted == expected:
            top1 += 1
        elif expected and predicted:
            confusions[(expected, predicted)] += 1
        if expected and expected in topk[:3]:
            top3 += 1
        if detected_key is not None:
            detected += int(_as_bool(row.get(detected_key)))

    summary: dict[str, object] = {
        "total_samples": total,
        "top1_correct": top1,
        "top3_correct": top3,
        "top1_accuracy": top1 / total if total else 0.0,
        "top3_accuracy": top3 / total if total else 0.0,
        "confusion_pairs": [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in sorted(
                confusions.items(), key=lambda item: (-item[1], item[0])
            )
        ],
    }
    if detected_key is not None:
        summary["detected_samples"] = detected
        summary["rejected_samples"] = total - detected
        summary["detection_rate"] = detected / total if total else 0.0
    return summary
