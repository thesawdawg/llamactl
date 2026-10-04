"""HFView: stop-gap wrapper that pushes the full HFScreen (phase 5 converts it)."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import Button, Label

from ..hf_screen import HFScreen
from .base import BaseView


class HFView(BaseView):
    """Button that opens the existing Hugging Face screen."""

    BINDINGS = [Binding("enter", "open", "Open")]

    def compose(self) -> ComposeResult:
        yield Label("Search and download GGUF models from Hugging Face.")
        yield Button("Open Hugging Face (Enter)", id="hf_open", variant="primary")

    def action_open(self) -> None:
        self.app.push_screen(HFScreen(self.app.cfg, self.app.host),
                             lambda _: self.app.views["models"].rescan())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "hf_open":
            self.action_open()
