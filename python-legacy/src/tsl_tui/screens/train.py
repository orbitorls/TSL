"""Train / ingest job log screen with cancel."""

from __future__ import annotations

from collections.abc import Callable

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, LoadingIndicator, Log, Static

from tsl_tui.runner import OnDone, OnLine, SubprocessRunner


class TrainScreen(Screen):
    """Stream a subprocess started by the caller (train or ingest)."""

    BINDINGS = [
        ("c", "cancel_job", "Cancel"),
        ("escape", "go_back", "Back"),
    ]

    def __init__(
        self,
        title: str,
        start: Callable[[SubprocessRunner, OnLine, OnDone], bool],
    ) -> None:
        super().__init__()
        self._title = title
        self._start = start
        self._runner: SubprocessRunner | None = None
        self._started = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="train-title")
        yield LoadingIndicator(id="train-spinner")
        yield Log(id="train-log", highlight=True, markup=True)
        with Horizontal(id="train-actions"):
            yield Button("Cancel", id="btn-cancel", variant="warning")
            yield Button("Back", id="btn-back")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#train-title", Static).update(f"[bold]{self._title}[/]")
        self._runner = self.app.runner  # type: ignore[attr-defined]
        self._begin()

    def _begin(self) -> None:
        if self._started or self._runner is None:
            return
        self._started = True
        log = self.query_one("#train-log", Log)
        log.clear()
        spinner = self.query_one("#train-spinner", LoadingIndicator)
        spinner.display = True

        def on_line(text: str) -> None:
            log.write_line(text)

        def on_done(code: int) -> None:
            spinner.display = False
            if code == 0:
                log.write_line("[green]Done (exit 0).[/]")
            else:
                log.write_line(f"[red]Exited with code {code}.[/]")

        ok = self._start(self._runner, on_line, on_done)
        if not ok:
            spinner.display = False
            log.write_line("[red]Another job is already running.[/]")

    def action_cancel_job(self) -> None:
        if self._runner is not None:
            self._runner.cancel()
            self.query_one("#train-log", Log).write_line(
                "[yellow]Cancel requested…[/]"
            )

    def action_go_back(self) -> None:
        if self._runner is not None and self._runner.running:
            self.notify("Cancel the job first (c), or wait for it to finish.")
            return
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.action_cancel_job()
        elif event.button.id == "btn-back":
            self.action_go_back()
