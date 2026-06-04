#!/usr/bin/env python3
"""TSL-51 webcam preflight: checks environment and prints the single next step."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = (
    REPO_ROOT
    / ".tools"
    / "tsl51_experiments"
    / "full51_v3_external_weighted"
    / "artifacts"
    / "tsl51"
)
DEFAULT_HOLDOUT = REPO_ROOT / "data" / "webcam_holdout" / "webcam_holdout.csv"
REVIEWED_SAMPLES = REPO_ROOT / "work" / "reviewed_external_assets" / "external_test_samples.csv"
DIAGNOSE = REPO_ROOT / "scripts" / "diagnose_live_vs_clip.py"
HOLDOUT_EVAL = REPO_ROOT / "reports" / "webcam_holdout_eval" / "summary.json"
ROOT_CAUSE = REPO_ROOT / "reports" / "live_vs_clip_diagnosis" / "root_cause_summary.json"
VENV_PYTHON = REPO_ROOT / ".venv-train" / "Scripts" / "python.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--holdout-csv", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/health")
    parser.add_argument("--run-smoke", action="store_true", help="Run live-vs-clip smoke on reviewed clips")
    parser.add_argument("--python", default=None, help="Python executable (default: .venv-train or sys.executable)")
    return parser.parse_args()


def _python(args: argparse.Namespace) -> str:
    if args.python:
        return args.python
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _artifact_ok(artifact_dir: Path) -> tuple[bool, list[str]]:
    issues: list[str] = []
    for name in ("tsl51_model.keras", "tsl51_labels.json", "tsl51_scaler.pkl"):
        if not (artifact_dir / name).exists():
            issues.append(f"missing {name}")
    if issues:
        return False, issues
    labels = json.loads((artifact_dir / "tsl51_labels.json").read_text(encoding="utf-8"))
    n = len(labels) if isinstance(labels, dict) else len(labels)
    if n != 51:
        issues.append(f"expected 51 labels, got {n}")
    return len(issues) == 0, issues


def _api_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _holdout_rows(path: Path) -> int:
    if not path.exists():
        return 0
    import csv

    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _read_holdout_top1() -> float | None:
    if not HOLDOUT_EVAL.exists():
        return None
    data = json.loads(HOLDOUT_EVAL.read_text(encoding="utf-8"))
    return float(data.get("top1_accuracy") or 0.0)


def _run_smoke(python: str, artifact_dir: Path) -> dict[str, object] | None:
    if not REVIEWED_SAMPLES.exists():
        return {"error": f"missing {REVIEWED_SAMPLES}"}
    out_dir = REPO_ROOT / "reports" / "live_vs_clip_diagnosis" / "doctor_smoke"
    cmd = [
        python,
        str(DIAGNOSE),
        "--samples",
        str(REVIEWED_SAMPLES),
        "--artifact-dir",
        str(artifact_dir),
        "--out-dir",
        str(out_dir),
        "--ab",
        "baseline",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
    except subprocess.CalledProcessError as exc:
        return {"error": exc.stderr or exc.stdout or str(exc)}
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return {"error": "smoke summary missing"}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _next_action(report: dict[str, object]) -> str:
    checks = report.get("checks") or {}
    if not checks.get("artifact_ok"):
        return (
            "Fix artifact: ensure full51_v3_external_weighted exists under .tools/tsl51_experiments/"
        )
    if not checks.get("deps_ok"):
        return "Install deps: D:\\TSL\\.venv-train\\Scripts\\pip install -r python-legacy/requirements-translate.txt"
    smoke = report.get("smoke") or {}
    if smoke and float(smoke.get("baseline_live_top1") or 0.0) < 1.0:
        return "Restart API with .venv-train Python after inference.py changes, then re-run tsl51_doctor --run-smoke"
    if not checks.get("api_ok"):
        return (
            "Start stack: powershell -File scripts\\tsl_translate_dev.ps1 "
            "→ load TSL-51 + full51_v3 + preset แม่นยำ at http://localhost:3000"
        )
    holdout_n = int(checks.get("holdout_clips") or 0)
    top1 = report.get("holdout_top1")
    if holdout_n == 0:
        py = report.get("python") or sys.executable
        return (
            "Record webcam holdout (15 words): "
            f"{py} scripts/record_webcam_holdout.py "
            "--out-dir data/webcam_holdout --signs 0:15 --num-clips 3"
        )
    if top1 is None:
        return "Evaluate holdout: python scripts/evaluate_webcam_holdout.py"
    if float(top1) < 0.7:
        return (
            "Fine-tune for your camera: python scripts/run_webcam_finetune.py "
            "(after recording train clips to data/webcam_train/)"
        )
    return (
        "Ready: open http://localhost:3000, sign with a brief pause after each sign. "
        "See docs/tsl51-webcam-playbook.md"
    )


def main() -> int:
    args = parse_args()
    python = _python(args)

    artifact_ok, artifact_issues = _artifact_ok(args.artifact_dir)
    deps_ok = all(_has_module(m) for m in ("cv2", "mediapipe", "tensorflow", "joblib", "numpy"))
    api_ok = _api_ok(args.api_url)
    holdout_n = _holdout_rows(args.holdout_csv)
    holdout_top1 = _read_holdout_top1() if holdout_n else None

    smoke: dict[str, object] | None = None
    if args.run_smoke and artifact_ok and deps_ok:
        smoke = _run_smoke(python, args.artifact_dir)

    report: dict[str, object] = {
        "python": python,
        "artifact_dir": str(args.artifact_dir),
        "checks": {
            "artifact_ok": artifact_ok,
            "artifact_issues": artifact_issues,
            "deps_ok": deps_ok,
            "api_ok": api_ok,
            "holdout_clips": holdout_n,
        },
        "holdout_top1": holdout_top1,
        "smoke": smoke,
        "root_cause_doc": str(ROOT_CAUSE) if ROOT_CAUSE.exists() else None,
    }
    report["next_action"] = _next_action({**report, "python": python})

    out_path = REPO_ROOT / "reports" / "tsl51_doctor.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[NEXT] {report['next_action']}")
    print(f"[OK] wrote {out_path}")

    if not artifact_ok or not deps_ok:
        return 1
    if smoke and float((smoke or {}).get("baseline_live_top1") or 0.0) < 1.0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
