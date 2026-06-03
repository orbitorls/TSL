from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "summarize_tsl51_eval_strategies.py"


def load_module():
    spec = importlib.util.spec_from_file_location("summarize_tsl51_eval_strategies_for_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_summary(path: Path, strategy: str, top1: float, top3: float) -> None:
    path.write_text(
        (
            '{"strategy":"%s","samples_file":"samples.csv","artifact_dir":"artifact",'
            '"total_samples":5,"detected_samples":5,"top1_accuracy":%s,'
            '"top3_accuracy":%s,"predictions_csv":"predictions.csv"}'
        )
        % (strategy, top1, top3),
        encoding="utf-8",
    )


def test_summarize_selects_best_strategy(tmp_path: Path) -> None:
    module = load_module()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_summary(first, "first", 0.2, 0.4)
    write_summary(second, "sliding", 0.8, 0.9)

    report = module.summarize([first, second])

    assert report["best_top1"]["strategy"] == "sliding"
    assert report["best_top3"]["strategy"] == "sliding"
    assert report["all_strategies_failed"] is False


def test_summarize_marks_all_failed_when_top1_zero(tmp_path: Path) -> None:
    module = load_module()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_summary(first, "first", 0.0, 0.0)
    write_summary(second, "sliding", 0.0, 0.0)

    report = module.summarize([first, second])

    assert report["all_strategies_failed"] is True
