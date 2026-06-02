"""Edit config.local.json fields from the example template."""

from __future__ import annotations

import json
from typing import Any

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Static

from tsl_tui import paths

# Keys mirrored from config.local.json.example (order preserved for UX).
SETTING_FIELDS: tuple[tuple[str, str], ...] = (
    ("tracks", "Tracks (both | fingerspelling | tsl51)"),
    ("work_root", "Work root (NPZ caches)"),
    ("artifact_dir", "Artifact output directory"),
    ("fs_zip", "Fingerspelling ZIP path"),
    ("fs_dataset_root", "Fingerspelling dataset root"),
    ("tsl51_metadata_dir", "TSL-51 metadata directory"),
    ("tsl51_file_cache", "TSL-51 file cache directory"),
    ("fs_epochs", "Fingerspelling epochs"),
    ("tsl51_epochs", "TSL-51 epochs"),
    ("fs_batch_size", "Fingerspelling batch size"),
    ("tsl51_batch_size", "TSL-51 batch size"),
    ("fs_workers", "Fingerspelling MediaPipe workers"),
    ("tsl51_download_workers", "TSL-51 download workers"),
    ("tsl51_parse_workers", "TSL-51 parse workers"),
)


def _load_template() -> dict[str, Any]:
    if paths.CONFIG_EXAMPLE_PATH.is_file():
        raw = json.loads(paths.CONFIG_EXAMPLE_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {k: v for k, v in raw.items() if not str(k).startswith("_")}
    return {key: "" for key, _ in SETTING_FIELDS}


def _load_merged() -> dict[str, Any]:
    merged = _load_template()
    if paths.CONFIG_PATH.is_file():
        try:
            current = json.loads(paths.CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = {}
        if isinstance(current, dict):
            for key, value in current.items():
                if not str(key).startswith("_"):
                    merged[key] = value
    return merged


class SettingsScreen(Screen):
    BINDINGS = [("escape", "pop", "Back")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(
            f"[bold]Settings[/] — writes [cyan]{paths.CONFIG_PATH.name}[/]",
            classes="section-title",
        )
        with VerticalScroll(id="settings-scroll"):
            for key, label in SETTING_FIELDS:
                yield Label(label)
                yield Input(value="", id=f"field-{key}")
        yield Static("", id="settings-status")
        yield Button("Load example", id="btn-example", variant="default")
        yield Button("Save", id="btn-save", variant="primary")
        yield Button("Back", id="btn-back")
        yield Footer()

    def on_mount(self) -> None:
        data = _load_merged()
        for key, _ in SETTING_FIELDS:
            widget = self.query_one(f"#field-{key}", Input)
            val = data.get(key, "")
            widget.value = "" if val is None else str(val)

    def _collect(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, _ in SETTING_FIELDS:
            raw = self.query_one(f"#field-{key}", Input).value.strip()
            if not raw:
                continue
            if key.endswith(
                ("_epochs", "_batch_size", "_workers")
            ):
                out[key] = int(raw)
            else:
                out[key] = raw
        return out

    def action_pop(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        status = self.query_one("#settings-status", Static)
        if event.button.id == "btn-back":
            self.action_pop()
            return
        if event.button.id == "btn-example":
            for key, _ in SETTING_FIELDS:
                val = _load_template().get(key, "")
                self.query_one(f"#field-{key}", Input).value = (
                    "" if val is None else str(val)
                )
            status.update("[green]Loaded template defaults[/]")
            return
        if event.button.id != "btn-save":
            return
        try:
            payload = self._collect()
        except (ValueError, TypeError) as exc:
            status.update(f"[red]{exc}[/]")
            return
        paths.CONFIG_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        status.update(f"[green]Saved[/] {paths.CONFIG_PATH}")
