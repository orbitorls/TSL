from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "render_review_queue_html.py"


def load_module():
    spec = importlib.util.spec_from_file_location("render_review_queue_html_for_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row(**overrides: str) -> dict[str, str]:
    item = {
        "review_priority": "1",
        "path": "videos/demo.mp4",
        "label": "label-a",
        "reviewed_label": "",
        "review_decision": "",
        "recommended_split": "external_test",
        "source_url": "https://example.test/video",
        "start_s": "1.0",
        "end_s": "2.0",
    }
    item.update(overrides)
    return item


def test_queue_summary_counts_decisions_and_splits() -> None:
    module = load_module()
    summary = module.queue_summary(
        [
            row(review_decision="approved"),
            row(review_decision="rejected", recommended_split="val"),
            row(recommended_split="train"),
        ]
    )

    assert summary["rows"] == 3
    assert summary["pending"] == 1
    assert summary["approved"] == 1
    assert summary["rejected"] == 1
    assert summary["recommended_splits"]["external_test"] == 1


def test_render_includes_video_and_review_fields() -> None:
    module = load_module()
    html = module.render([row(label="<unsafe>")], Path("D:/repo"))

    assert "<video" in html
    assert "&lt;unsafe&gt;" in html
    assert "review_decision" in html
