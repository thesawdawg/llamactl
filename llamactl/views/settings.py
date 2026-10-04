"""SettingsView: stop-gap wrapper around the FormScreen settings form."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import Button, Label

from ..forms import FormScreen
from .base import BaseView


class SettingsView(BaseView):
    """Button that opens the bin dir / model dirs form (phase 5 replaces this)."""

    BINDINGS = [Binding("enter", "open", "Open")]

    def compose(self) -> ComposeResult:
        yield Label("llama.cpp bin dir and model directories.")
        yield Button("Open settings (Enter)", id="set_open", variant="primary")

    def focus_primary(self) -> None:
        self.query_one("#set_open", Button).focus()

    def action_open(self) -> None:
        cfg = self.app.cfg
        fields = [
            ("bin_dir", "llama.cpp bin dir", cfg.bin_dir,
             "Folder holding llama-server and llama-cli (your build/bin)."),
            ("model_dirs", "Model dirs", ",".join(cfg.model_dirs),
             "Comma-separated folders scanned (recursively) for .gguf models. Press r to rescan."),
        ]

        def done(res: dict | None) -> None:
            if res:
                cfg.bin_dir = res["bin_dir"]
                cfg.model_dirs = [d.strip() for d in res["model_dirs"].split(",") if d.strip()]
                cfg.save()
                self.app.views["models"].rescan()

        self.app.push_screen(FormScreen("Settings", fields), done)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "set_open":
            self.action_open()
