"""Pick fingerspelling vs TSL-51 cache-only build."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Static

from tsl_tui.screens.train import TrainScreen


class CachePickScreen(Screen):
    """Choose cache-only track."""

    DEFAULT_CSS = """
    CachePickScreen {
        align: center middle;
    }
    #cache-box {
        width: 60;
        border: thick $primary;
        background: $panel;
        padding: 1 2;
    }
    """

    BINDINGS = [("escape", "pop", "Back")]

    def compose(self) -> ComposeResult:
        with Vertical(id="cache-box"):
            yield Static("[bold]Cache only[/] — build NPZ features without fitting")
            with Horizontal():
                yield Button("Fingerspelling", id="fs", variant="primary")
                yield Button("TSL-51", id="tsl51", variant="primary")
                yield Button("Back", id="back")

    def action_pop(self) -> None:
        self.app.pop_screen()

    def _start(self, title: str, method: str) -> None:
        self.app.pop_screen()

        def start(runner, on_line, on_done):  # noqa: ANN001
            fn = getattr(runner, method)
            return fn(on_line=on_line, on_done=on_done)

        self.app.push_screen(TrainScreen(title, start))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.action_pop()
        elif event.button.id == "fs":
            self._start("Fingerspelling cache only", "fs_only_cache")
        elif event.button.id == "tsl51":
            self._start("TSL-51 cache only", "tsl51_only_cache")
