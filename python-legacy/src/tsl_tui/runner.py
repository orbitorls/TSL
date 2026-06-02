"""Subprocess runner for train / ingest scripts (no TensorFlow in TUI process)."""

from __future__ import annotations

import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from tsl_tui import paths

if TYPE_CHECKING:
    from textual.app import App

OnLine = Callable[[str], None]
OnDone = Callable[[int], None]


class SubprocessRunner:
    def __init__(self, app: App | None = None) -> None:
        self._app = app
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        with self._lock:
            proc = self._proc
        return proc is not None and proc.poll() is None

    def cancel(self) -> bool:
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return False
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
        return True

    def _emit(self, callback: Callable[..., None], *args: object) -> None:
        if self._app is not None:
            self._app.call_from_thread(callback, *args)
        else:
            callback(*args)

    def spawn(
        self,
        argv: list[str],
        *,
        cwd: Path | None = None,
        on_line: OnLine | None = None,
        on_done: OnDone | None = None,
    ) -> bool:
        """Start a subprocess. Returns False if a job is already running."""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return False
            work = cwd or paths.LEGACY_ROOT
            self._proc = subprocess.Popen(
                argv,
                cwd=str(work),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            proc = self._proc

        def _reader() -> None:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\n\r")
                if on_line is not None:
                    self._emit(on_line, line)
            code = proc.wait()
            with self._lock:
                if self._proc is proc:
                    self._proc = None
            if on_done is not None:
                self._emit(on_done, code)

        self._thread = threading.Thread(target=_reader, daemon=True)
        self._thread.start()
        return True

    def _train_argv(self, extra: list[str] | None = None) -> list[str]:
        return [
            sys.executable,
            str(paths.TRAIN_SCRIPT),
            *paths.config_to_argv(extra),
        ]

    def run_train(
        self,
        extra: list[str] | None = None,
        *,
        on_line: OnLine | None = None,
        on_done: OnDone | None = None,
    ) -> bool:
        return self.spawn(self._train_argv(extra), on_line=on_line, on_done=on_done)

    def preflight(
        self,
        *,
        on_line: OnLine | None = None,
        on_done: OnDone | None = None,
    ) -> bool:
        return self.run_train(["--preflight"], on_line=on_line, on_done=on_done)

    def smoke_train(
        self,
        *,
        on_line: OnLine | None = None,
        on_done: OnDone | None = None,
    ) -> bool:
        return self.run_train(
            [
                "--tracks",
                "both",
                "--max-tsl51-samples",
                "512",
                "--tsl51-epochs",
                "2",
            ],
            on_line=on_line,
            on_done=on_done,
        )

    def full_train(
        self,
        *,
        on_line: OnLine | None = None,
        on_done: OnDone | None = None,
    ) -> bool:
        return self.run_train(["--tracks", "both"], on_line=on_line, on_done=on_done)

    def fs_only_cache(
        self,
        *,
        on_line: OnLine | None = None,
        on_done: OnDone | None = None,
    ) -> bool:
        return self.run_train(
            ["--tracks", "fingerspelling", "--fs-only-cache"],
            on_line=on_line,
            on_done=on_done,
        )

    def tsl51_only_cache(
        self,
        *,
        on_line: OnLine | None = None,
        on_done: OnDone | None = None,
    ) -> bool:
        return self.run_train(
            ["--tracks", "tsl51", "--tsl51-only-cache"],
            on_line=on_line,
            on_done=on_done,
        )

    def run_ingest(
        self,
        ingest_argv: list[str],
        *,
        on_line: OnLine | None = None,
        on_done: OnDone | None = None,
    ) -> bool:
        argv = [sys.executable, str(paths.INGEST_SCRIPT), *ingest_argv]
        return self.spawn(argv, on_line=on_line, on_done=on_done)


def preflight_args() -> list[str]:
    return ["--preflight"]


def smoke_args() -> list[str]:
    return [
        "--tracks",
        "both",
        "--max-tsl51-samples",
        "512",
        "--tsl51-epochs",
        "2",
    ]


def full_train_args() -> list[str]:
    return ["--tracks", "both"]


def fs_cache_only_args() -> list[str]:
    return ["--tracks", "fingerspelling", "--fs-only-cache"]


def tsl51_cache_only_args() -> list[str]:
    return ["--tracks", "tsl51", "--tsl51-only-cache"]
