"""Ingest form — spawns ingest_local.py."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Footer, Header, Input, Label, Select, Static

from tsl_tui import paths
from tsl_tui.screens.train import TrainScreen


class IngestScreen(Screen):
    BINDINGS = [("escape", "pop", "Back")]

    DEFAULT_OUTPUT = {
        "fingerspelling": "data/one_stage_tfs",
        "tsl51": "data/tsl51_raw",
    }

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static("[bold]Dataset ingest[/]", classes="section-title")
        yield Label("Track")
        yield Select(
            (("Fingerspelling (One-Stage-TFS)", "fingerspelling"), ("TSL-51 metadata", "tsl51")),
            id="track",
            value="fingerspelling",
        )
        yield Label("ZIP path (fingerspelling)")
        yield Input(placeholder="D:/datasets/One-Stage-TFS.zip", id="zip")
        yield Label("Dataset root (fingerspelling, alternative to ZIP)")
        yield Input(placeholder="D:/datasets/One-Stage-TFS", id="dataset-root")
        yield Label("TSL-51 source folder (optional)")
        yield Input(placeholder="Leave empty to download from Hugging Face", id="source")
        yield Label("Output under repo data/")
        yield Input(id="output", value=self.DEFAULT_OUTPUT["fingerspelling"])
        yield Checkbox("Force overwrite output", id="force")
        with Horizontal(id="ingest-actions"):
            yield Button("Run ingest", id="btn-run", variant="primary")
            yield Button("Back", id="btn-back")
        yield Static("", id="ingest-hint")
        yield Footer()

    def on_mount(self) -> None:
        self._sync_fields()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "track":
            self._sync_fields()

    def _sync_fields(self) -> None:
        track = self.query_one("#track", Select).value
        is_fs = track == "fingerspelling"
        self.query_one("#zip").disabled = not is_fs
        self.query_one("#dataset-root").disabled = not is_fs
        self.query_one("#source").disabled = is_fs
        out = self.query_one("#output", Input)
        if out.value in ("", *self.DEFAULT_OUTPUT.values()):
            out.value = self.DEFAULT_OUTPUT[str(track)]

    def _build_argv(self) -> list[str]:
        track = str(self.query_one("#track", Select).value)
        output = self.query_one("#output", Input).value.strip()
        if not output:
            output = self.DEFAULT_OUTPUT[track]
        out_path = Path(output)
        if not out_path.is_absolute():
            out_path = (paths.REPO_ROOT / output).resolve()

        argv = [track, "--output", str(out_path)]
        if self.query_one("#force", Checkbox).value:
            argv.append("--force")

        if track == "fingerspelling":
            zip_val = self.query_one("#zip", Input).value.strip()
            root_val = self.query_one("#dataset-root", Input).value.strip()
            if zip_val and root_val:
                raise ValueError("Use ZIP or dataset root, not both.")
            if not zip_val and not root_val:
                raise ValueError("Provide a ZIP path or dataset root for fingerspelling.")
            if zip_val:
                argv.extend(["--zip", zip_val])
            else:
                argv.extend(["--dataset-root", root_val])
        else:
            source = self.query_one("#source", Input).value.strip()
            if source:
                argv.extend(["--source", source])
        return argv

    def action_pop(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.action_pop()
            return
        if event.button.id != "btn-run":
            return

        hint = self.query_one("#ingest-hint", Static)
        try:
            ingest_argv = self._build_argv()
        except ValueError as exc:
            hint.update(f"[red]{exc}[/]")
            return

        argv = ingest_argv

        def start(runner, on_line, on_done):  # noqa: ANN001
            return runner.run_ingest(argv, on_line=on_line, on_done=on_done)

        self.app.push_screen(TrainScreen("Dataset ingest", start))
        hint.update("[green]Started ingest job.[/]")
