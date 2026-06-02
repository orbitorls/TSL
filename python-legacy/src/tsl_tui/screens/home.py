"""Home dashboard — dataset/artifact status and navigation."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from tsl_tui.paths import gather_status
from tsl_tui.screens.confirm import ConfirmFullTrainModal
from tsl_tui.screens.train import TrainScreen


class HomeScreen(Screen):
    """Read-only status and keyboard shortcuts."""

    BINDINGS = [
        ("p", "run_preflight", "Preflight"),
        ("s", "run_smoke", "Smoke"),
        ("f", "run_full", "Full train"),
        ("c", "cache_menu", "Cache only"),
        ("i", "go_ingest", "Ingest"),
        ("comma", "go_settings", "Settings"),
        ("q", "app.quit", "Quit"),
        ("escape", "app.quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="home-body"):
            yield Static(id="status-panel")
            yield Static(
                "[bold]Keys[/]  [dim]p[/] Preflight  [dim]s[/] Smoke  [dim]f[/] Full  "
                "[dim]c[/] Cache  [dim]i[/] Ingest  comma Settings  [dim]q[/] Quit",
                id="home-help",
            )
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_status()

    def refresh_status(self) -> None:
        status = gather_status()
        self.query_one("#status-panel", Static).update(
            "[bold]TSL Training[/]\n\n" + "\n".join(status.lines())
        )

    def on_screen_resume(self) -> None:
        self.refresh_status()

    def action_run_preflight(self) -> None:
        self.app.push_screen(
            TrainScreen(
                "Preflight",
                lambda r, ol, od: r.preflight(on_line=ol, on_done=od),
            )
        )

    def action_run_smoke(self) -> None:
        self.app.push_screen(
            TrainScreen(
                "Smoke train",
                lambda r, ol, od: r.smoke_train(on_line=ol, on_done=od),
            )
        )

    def action_run_full(self) -> None:
        def on_confirm(ok: bool | None) -> None:
            if ok:
                self.app.push_screen(
                    TrainScreen(
                        "Full train",
                        lambda r, ol, od: r.full_train(on_line=ol, on_done=od),
                    )
                )

        self.app.push_screen(ConfirmFullTrainModal(), on_confirm)

    def action_cache_menu(self) -> None:
        self.app.push_screen("cache_pick")

    def action_go_ingest(self) -> None:
        self.app.push_screen("ingest")

    def action_go_settings(self) -> None:
        self.app.push_screen("settings")
