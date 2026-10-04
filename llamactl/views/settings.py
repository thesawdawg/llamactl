"""SettingsView: editable app settings inside the shell."""
from __future__ import annotations

from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.validation import Function, Integer, Number
from textual.widgets import Button, Input, Label, ListItem, ListView

from .. import hf
from ..screens.prompt import PromptScreen
from .base import BaseView
from .hf import TOKEN_HINT


class SettingsView(BaseView):
    """Form for bin dir, model dirs, HF token and polling tunables."""

    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("delete", "remove_dir", "Remove dir"),
    ]
    CSS = """
    #set_scroll { padding: 0 1; }
    #set_scroll .hint { color: $text-muted; text-style: italic; }
    #set_scroll Input { margin-bottom: 0; }
    #dir_list { height: auto; max-height: 8; border: round $primary-darken-2; }
    #set_buttons { height: auto; margin-top: 1; }
    #set_buttons Button { margin-right: 1; }
    #token_state { color: $text-muted; }
    """

    def compose(self) -> ComposeResult:
        cfg = self.app.cfg
        with VerticalScroll(id="set_scroll"):
            yield Label("[b]llama.cpp bin dir[/b]")
            yield Input(cfg.bin_dir, id="bin_dir",
                        validators=[Function(self._bin_ok, "needs llama-server inside")])
            yield Label("Folder holding llama-server and llama-cli (your build/bin).",
                        classes="hint")
            yield Label("[b]Model dirs[/b]  (Delete removes the highlighted entry)")
            yield ListView(*[ListItem(Label(d)) for d in cfg.model_dirs], id="dir_list")
            yield Input(placeholder="Add directory - press Enter", id="add_dir",
                        validators=[Function(self._dir_ok, "not a directory")])
            yield Label("Folders scanned (recursively) for .gguf models.", classes="hint")
            yield Label("[b]Hugging Face token[/b]")
            with Horizontal():
                yield Button("Set token", id="set_token")
                yield Label("", id="token_state")
            yield Label("[b]Gauge refresh interval (seconds)[/b]")
            yield Input(str(cfg.gauge_interval), id="gauge_interval",
                        validators=[Number(minimum=1)])
            yield Label("How often VRAM/RAM gauges and Fits verdicts refresh.", classes="hint")
            yield Label("[b]VRAM headroom (%)[/b]")
            yield Input(str(cfg.headroom_pct), id="headroom_pct",
                        validators=[Integer(minimum=0, maximum=50)])
            yield Label("Share of total VRAM recommendations keep free (min 0.5 GB).",
                        classes="hint")
            with Horizontal(id="set_buttons"):
                yield Button("Save", id="save", variant="primary")

    def on_mount(self) -> None:
        self._token_state()

    def focus_primary(self) -> None:
        self.query_one("#bin_dir", Input).focus()

    @staticmethod
    def _bin_ok(value: str) -> bool:
        """True when the dir contains a llama-server executable."""
        return (Path(value).expanduser() / "llama-server").exists()

    @staticmethod
    def _dir_ok(value: str) -> bool:
        """True for an existing directory (empty input is ignored, not saved)."""
        return not value.strip() or Path(value).expanduser().is_dir()

    def _token_state(self) -> None:
        self.query_one("#token_state", Label).update(
            "token set" if hf.load_token() else "no token")

    @on(Input.Submitted, "#add_dir")
    def _add_dir(self, event: Input.Submitted) -> None:
        inp = event.input
        if not inp.is_valid or not inp.value.strip():
            return self.say("Not a directory.")
        self.query_one("#dir_list", ListView).append(ListItem(Label(inp.value.strip())))
        inp.value = ""

    def action_remove_dir(self) -> None:
        lv = self.query_one("#dir_list", ListView)
        if lv.highlighted_child is not None:
            lv.remove_children(lv.highlighted_child)

    @on(Button.Pressed, "#set_token")
    def _set_token(self) -> None:
        def done(res: str | None) -> None:
            if res is not None:
                hf.save_token(res)
                self._token_state()

        self.app.push_screen(PromptScreen("Hugging Face access token", TOKEN_HINT,
                                          hf.load_token() or "", password=True), done)

    def _fields_valid(self) -> bool:
        return all(self.query_one(f"#{i}", Input).is_valid
                   for i in ("bin_dir", "gauge_interval", "headroom_pct"))

    def action_save(self) -> None:
        """Validate, write config and rescan models."""
        if not self._fields_valid():
            return self.say("Fix the invalid fields first.")
        cfg = self.app.cfg
        cfg.bin_dir = self.query_one("#bin_dir", Input).value.strip()
        cfg.model_dirs = [
            item.children[0].content  # type: ignore[attr-defined]
            for item in self.query_one("#dir_list", ListView).children
        ]
        cfg.gauge_interval = float(self.query_one("#gauge_interval", Input).value)
        cfg.headroom_pct = int(self.query_one("#headroom_pct", Input).value)
        cfg.save()
        self.app.views["models"].rescan()
        self.say("Settings saved")

    @on(Button.Pressed, "#save")
    def _save_button(self) -> None:
        self.action_save()
