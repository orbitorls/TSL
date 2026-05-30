from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_NOTEBOOK = REPO_ROOT / "colab" / "02_train.ipynb"
EVAL_NOTEBOOK = REPO_ROOT / "colab" / "03_evaluate.ipynb"


def _load_notebook_payload(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"{path.name} is not valid notebook JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"
        ) from exc

    if not isinstance(payload, dict):
        raise AssertionError(f"{path.name} must decode to a notebook object, got {type(payload).__name__}")

    return payload


def _cell_sources(payload: object, *, source_name: str) -> list[str]:
    if not isinstance(payload, dict):
        raise AssertionError(f"{source_name} must decode to a notebook object, got {type(payload).__name__}")

    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise AssertionError(f"{source_name} must contain a notebook 'cells' list")

    sources: list[str] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise AssertionError(f"{source_name} cell {index} must be a JSON object")

        source = cell.get("source", [])
        if isinstance(source, str):
            sources.append(source)
        elif isinstance(source, list):
            sources.append("".join(source))
        else:
            raise AssertionError(
                f"{source_name} cell {index} has unsupported source type {type(source).__name__}"
            )

    return sources


def _notebook_text(path: Path) -> str:
    payload = _load_notebook_payload(path)
    return "\n".join(_cell_sources(payload, source_name=path.name))


def test_train_notebook_uses_canonical_pipeline_contract() -> None:
    text = _notebook_text(TRAIN_NOTEBOOK)

    assert "src.train.pipeline" in text
    assert "run_training_pipeline" in text
    assert "StratifiedKFold" not in text
    assert "augment_data(" not in text


def test_evaluate_notebook_documents_task8_result_contract() -> None:
    text = _notebook_text(EVAL_NOTEBOOK)

    for expected in (
        "Macro F1",
        "macro_f1",
        "primary_metric_name",
        "primary_metric",
        "metrics",
        "split_metadata",
        "artifacts",
    ):
        assert expected in text, f"Expected {expected!r} in {EVAL_NOTEBOOK.name}"


def test_invalid_notebook_payload_helper_raises_clear_error() -> None:
    with pytest.raises(AssertionError, match=r"broken\.ipynb must contain a notebook 'cells' list"):
        _cell_sources({"nbformat": 4, "metadata": {}}, source_name="broken.ipynb")
