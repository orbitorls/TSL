"""Confirmation modals."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class ConfirmFullTrainModal(ModalScreen[bool]):
    """Confirm before starting a long full GPU training run."""

    DEFAULT_CSS = """
    ConfirmFullTrainModal {
        align: center middle;
    }
    #confirm-box {
        width: 72;
        height: auto;
        border: thick $warning;
        background: $panel;
        padding: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Static(
                "[bold yellow]Full train[/]\n\n"
                "Runs [cyan]--tracks both[/] with values from config.local.json. "
                "This can take a long time. Continue?",
            )
            with Horizontal():
                yield Button("Yes, start", id="yes", variant="success")
                yield Button("No", id="no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")
