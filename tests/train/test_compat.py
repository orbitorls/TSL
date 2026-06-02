"""Regression tests for platform compatibility helpers."""

import io

from src.train.compat import setup_windows_encoding


def test_setup_windows_encoding_reconfigures_streams(monkeypatch):
    from unittest.mock import MagicMock
    mock_stdout = MagicMock()
    mock_stdout.buffer = MagicMock(spec=io.BytesIO)

    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr("sys.stdout", mock_stdout)

    setup_windows_encoding()

    import sys
    assert isinstance(sys.stdout, io.TextIOWrapper)
    assert sys.stdout.encoding == "utf-8"
    assert sys.stdout.errors == "replace"

def test_setup_windows_encoding_noops_outside_windows(monkeypatch):
    from unittest.mock import MagicMock
    mock_stdout = MagicMock()
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("sys.stdout", mock_stdout)

    setup_windows_encoding()

    import sys
    assert sys.stdout is mock_stdout
