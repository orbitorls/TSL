"""Regression tests for platform compatibility helpers."""

from src.train.compat import setup_windows_encoding


class _DummyStream:
    def __init__(self):
        self.calls = []

    def reconfigure(self, **kwargs):
        self.calls.append(kwargs)


def test_setup_windows_encoding_reconfigures_streams(monkeypatch):
    stdout = _DummyStream()
    stderr = _DummyStream()

    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)

    setup_windows_encoding()

    expected = {"encoding": "utf-8", "errors": "replace"}
    assert stdout.calls == [expected]
    assert stderr.calls == [expected]


def test_setup_windows_encoding_noops_outside_windows(monkeypatch):
    stdout = _DummyStream()
    stderr = _DummyStream()

    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)

    setup_windows_encoding()

    assert stdout.calls == []
    assert stderr.calls == []
