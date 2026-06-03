from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_reviewed_external_pipeline.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_reviewed_external_pipeline_for_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pipeline_dry_run_reports_planned_artifacts(tmp_path: Path) -> None:
    module = load_module()
    args = Namespace(
        queue=Path("queue.csv"),
        base_manifest=Path("manifest.csv"),
        base_cache=Path("base.npz"),
        base_artifact_dir=Path("artifacts"),
        labels=Path("labels.json"),
        work_dir=tmp_path / "work",
        reports_dir=tmp_path / "reports",
        python=sys.executable,
        epochs=1,
        batch_size=1,
        allow_missing_decisions=True,
        dry_run=True,
    )

    summary = module.pipeline(args)

    assert summary["status"] == "dry_run"
    assert summary["trained"] is False
    assert "validate_summary" in summary
    assert "assets_summary" in summary


def test_pipeline_blocks_before_training_when_assets_not_ready(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    args = Namespace(
        queue=Path("queue.csv"),
        base_manifest=Path("manifest.csv"),
        base_cache=Path("base.npz"),
        base_artifact_dir=Path("artifacts"),
        labels=Path("labels.json"),
        work_dir=tmp_path / "work",
        reports_dir=tmp_path / "reports",
        python=sys.executable,
        epochs=1,
        batch_size=1,
        allow_missing_decisions=True,
        dry_run=False,
    )

    def fake_run(cmd, *, dry_run=False, check=True):
        if any(str(part).endswith("validate_review_queue.py") for part in cmd):
            validate_summary = args.reports_dir / "review_validate_summary.json"
            validate_summary.write_text('{"ready": true}', encoding="utf-8")
        if any(str(part).endswith("export_reviewed_external_assets.py") for part in cmd):
            assets_summary = args.reports_dir / "reviewed_assets_summary.json"
            assets_summary.write_text(
                '{"ready_for_training": false, "ready_for_eval": true, "splits": {}}',
                encoding="utf-8",
            )

    monkeypatch.setattr(module, "run", fake_run)

    summary = module.pipeline(args)

    assert summary["status"] == "blocked_missing_reviewed_train_val"
    assert summary["trained"] is False
    assert summary["evaluated"] is False


def test_pipeline_runs_validation_before_apply(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    args = Namespace(
        queue=Path("queue.csv"),
        base_manifest=Path("manifest.csv"),
        base_cache=Path("base.npz"),
        base_artifact_dir=Path("artifacts"),
        labels=Path("labels.json"),
        work_dir=tmp_path / "work",
        reports_dir=tmp_path / "reports",
        python=sys.executable,
        epochs=1,
        batch_size=1,
        allow_missing_decisions=False,
        dry_run=False,
    )
    calls: list[str] = []

    def fake_run(cmd, *, dry_run=False, check=True):
        script = next((str(part) for part in cmd if str(part).endswith(".py")), "")
        calls.append(Path(script).name)
        if script.endswith("validate_review_queue.py"):
            validate_summary = args.reports_dir / "review_validate_summary.json"
            validate_summary.write_text('{"ready": true}', encoding="utf-8")
        if script.endswith("export_reviewed_external_assets.py"):
            assets_summary = args.reports_dir / "reviewed_assets_summary.json"
            assets_summary.write_text(
                '{"ready_for_training": false, "ready_for_eval": true, "splits": {}}',
                encoding="utf-8",
            )

    monkeypatch.setattr(module, "run", fake_run)

    module.pipeline(args)

    assert calls[:2] == ["validate_review_queue.py", "apply_review_queue.py"]


def test_pipeline_blocks_immediately_when_validation_not_ready(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    args = Namespace(
        queue=Path("queue.csv"),
        base_manifest=Path("manifest.csv"),
        base_cache=Path("base.npz"),
        base_artifact_dir=Path("artifacts"),
        labels=Path("labels.json"),
        work_dir=tmp_path / "work",
        reports_dir=tmp_path / "reports",
        python=sys.executable,
        epochs=1,
        batch_size=1,
        allow_missing_decisions=True,
        dry_run=False,
    )
    calls: list[str] = []

    def fake_run(cmd, *, dry_run=False, check=True):
        script = next((str(part) for part in cmd if str(part).endswith(".py")), "")
        calls.append(Path(script).name)
        validate_summary = args.reports_dir / "review_validate_summary.json"
        validate_summary.write_text('{"ready": false, "errors": ["train has 0 approved rows"]}', encoding="utf-8")

    monkeypatch.setattr(module, "run", fake_run)

    summary = module.pipeline(args)

    assert summary["status"] == "blocked_review_queue_not_ready"
    assert summary["trained"] is False
    assert calls == ["validate_review_queue.py"]
