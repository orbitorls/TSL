"""Textual application entry."""

from __future__ import annotations

from textual.app import App
from textual.binding import Binding

from tsl_tui.runner import SubprocessRunner
from tsl_tui.screens.cache_pick import CachePickScreen
from tsl_tui.screens.home import HomeScreen
from tsl_tui.screens.ingest import IngestScreen
from tsl_tui.screens.settings import SettingsScreen


class TslTuiApp(App):
    TITLE = "TSL Training"
    SUB_TITLE = "python-legacy"
    CSS = """
    Screen {
        background: $surface;
    }
    .section-title {
        padding: 0 1 1 1;
    }
    #status-panel {
        padding: 0 1;
        height: auto;
    }
    #home-help {
        padding: 1;
        color: $text-muted;
    }
    #train-log {
        height: 1fr;
        border: solid $primary;
        margin: 0 1;
    }
    #train-actions {
        height: auto;
        padding: 0 1 1 1;
    }
    #ingest-actions {
        height: auto;
        padding: 1;
    }
    #settings-scroll {
        height: 1fr;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=False),
    ]

    SCREENS = {
        "home": HomeScreen,
        "cache_pick": CachePickScreen,
        "ingest": IngestScreen,
        "settings": SettingsScreen,
    }

    def __init__(self) -> None:
        super().__init__()
        self.runner = SubprocessRunner(self)

    def on_mount(self) -> None:
        self.theme = "textual-dark"
        self.push_screen(HomeScreen())
