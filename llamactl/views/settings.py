"""SettingsView: llama.cpp folder, model folders, gateway, headroom, default intent, HF token."""
from __future__ import annotations

from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widget import Widget
from textual.validation import Function, Integer, Number
from textual.widgets import Button, Input, Label, ListItem, ListView, Select, Static, Switch

from .. import hf
from ..intents import INTENTS
from ..screens.prompt import PromptScreen
from .base import BaseView
from .hf import TOKEN_HINT

TEXT_INPUTS = ("bin_dir", "gauge_interval", "headroom_pct", "gw_host", "gw_port", "gw_models_max",
               "gw_sleep_idle")
RESTART_FIELDS = (("host", "host"), ("port", "port"), ("api_key", "API key"),
                  ("models_max", "models-max"))     # gateway settings that are command line options


class SettingsView(BaseView):
    """Form for the app's settings; saving applies them and reloads a running gateway."""

    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("delete", "remove_dir", "Remove dir"),
    ]
    DEFAULT_CSS = """
    SettingsView { height: 1fr; }
    #set_scroll { padding: 0 1; }
    #set_scroll .section { text-style: bold; margin-top: 1; color: $accent; }
    #set_scroll .row { height: auto; }
    #set_scroll .lbl { width: 28; padding: 0 1 0 0; }
    #set_scroll .row Input { width: 50; }
    #set_scroll .row Select { width: 50; }
    #set_scroll .hint { color: $text-muted; text-style: italic; padding: 0 1; }
    #dir_list { height: auto; max-height: 6; border: round $primary-darken-2; }
    #set_buttons { height: auto; margin-top: 1; }
    #set_buttons Button { margin-right: 1; }
    #set_msg { height: auto; margin-top: 1; }
    """

    def compose(self) -> ComposeResult:
        """The settings form: label, control and hint per row."""
        cfg = self.app.cfg
        g = cfg.gateway
        intent_keys = [i.key for i in INTENTS]

        def row(label: str, widget: Widget, hint: str = "") -> Horizontal:
            """A label on the left, the control on the right, an optional hint after it."""
            kids: list[Widget] = [Label(label, classes="lbl"), widget]
            if hint:
                kids.append(Label(hint, classes="hint"))
            return Horizontal(*kids, classes="row")

        with VerticalScroll(id="set_scroll"):
            yield Label("llama.cpp", classes="section")
            yield row("Bin dir", Input(cfg.bin_dir, id="bin_dir", compact=True, validators=[
                Function(self._bin_ok, "needs llama-server inside")]), "folder holding llama-server")
            yield Label("Model folders (scanned recursively for .gguf; Delete removes the "
                        "highlighted one)", classes="section")
            yield ListView(*[ListItem(Label(d)) for d in cfg.model_dirs], id="dir_list")
            yield row("Add folder", Input(placeholder="path, then Enter", id="add_dir",
                                          compact=True, validators=[
                Function(self._dir_ok, "not a directory")]))
            yield Label("Serving", classes="section")
            yield row("Default intent", Select(
                [(i.label, i.key) for i in INTENTS], allow_blank=False, id="default_intent",
                compact=True, value=cfg.default_intent if cfg.default_intent in intent_keys
                else "chat"), "used when you do not pick one")
            yield Label("Gateway", classes="section")
            yield row("Host", Input(g.host, id="gw_host", compact=True, validators=[Function(
                lambda v: bool(v.strip()) and " " not in v.strip(), "a host name or address")]))
            yield row("Port", Input(str(g.port), id="gw_port", compact=True,
                                    validators=[Integer(minimum=1, maximum=65535)]))
            yield row("API key", Input(g.api_key, id="gw_api_key", password=True, compact=True),
                      "Bearer token; empty = none")
            yield row("Models loaded at once", Input(str(g.models_max), id="gw_models_max",
                                                     compact=True,
                                                     validators=[Integer(minimum=1)]),
                      "models-max")
            yield row("Unload after idle seconds", Input(str(g.sleep_idle), id="gw_sleep_idle",
                                                         compact=True,
                                                         validators=[Integer(minimum=0)]),
                      "0 = never")
            yield row("Start gateway on launch", Switch(g.autostart, id="gw_autostart"))
            yield Label("Planning and display", classes="section")
            yield row("VRAM headroom (%)", Input(str(cfg.headroom_pct), id="headroom_pct",
                                                 compact=True, validators=[
                Integer(minimum=0, maximum=50)]), "kept free per device (min 0.5 GB)")
            yield row("Gauge interval (s)", Input(str(cfg.gauge_interval), id="gauge_interval",
                                                  compact=True, validators=[Number(minimum=1)]),
                      "applies on the next start")
            yield Label("Hugging Face", classes="section")
            yield row("Access token", Button("Set token", id="set_token", compact=True))
            with Horizontal(id="set_buttons"):
                yield Button("Save (Ctrl+S)", id="save", variant="primary")
                yield Button("Edit default profile", id="edit_default")
            yield Static("", id="set_msg")

    def on_mount(self) -> None:
        """Show whether a token is set."""
        self._token_state()

    def focus_primary(self) -> None:
        """Focus the first field."""
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
        """Label the token button with whether a token exists."""
        self.query_one("#set_token", Button).label = (
            "Change token (set)" if hf.load_token() else "Set token (none yet)")

    def _msg(self, text: str) -> None:
        """Show a message under the form and in the status bar."""
        self.query_one("#set_msg", Static).update(text)
        self.say(text)

    @on(Input.Submitted, "#add_dir")
    def _add_dir(self, event: Input.Submitted) -> None:
        """Enter in the add-folder box: append a valid folder to the list."""
        inp = event.input
        if not inp.is_valid or not inp.value.strip():
            return self.say("Not a directory.")
        self.query_one("#dir_list", ListView).append(ListItem(Label(inp.value.strip())))
        inp.value = ""

    def action_remove_dir(self) -> None:
        """Remove the highlighted model folder from the list (applied on save)."""
        lv = self.query_one("#dir_list", ListView)
        if lv.highlighted_child is not None:
            lv.highlighted_child.remove()

    @on(Button.Pressed, "#set_token")
    def _set_token(self) -> None:
        """Ask for the Hugging Face token."""
        def done(res: str | None) -> None:
            if res is not None:
                hf.save_token(res)
                self._token_state()

        self.app.push_screen(PromptScreen("Hugging Face access token", TOKEN_HINT,
                                          hf.load_token() or "", password=True), done)

    @on(Button.Pressed, "#edit_default")
    def _edit_default(self) -> None:
        """Open the default profile editor (Models view)."""
        self.app.views["models"].action_edit_default()          # type: ignore[attr-defined]

    def _fields_valid(self) -> bool:
        """True when every text field validates."""
        return all(self.query_one(f"#{i}", Input).is_valid for i in TEXT_INPUTS)

    def _dirs(self) -> list[str]:
        """The model folders currently listed."""
        return [item.children[0].content  # type: ignore[attr-defined]
                for item in self.query_one("#dir_list", ListView).children]

    def action_save(self) -> None:
        """Validate, apply, write the config, then rescan and reload the gateway presets."""
        if not self._fields_valid():
            return self._msg("Fix the invalid fields first.")
        a = self.app
        cfg, g = a.cfg, a.cfg.gateway
        before = (cfg.bin_dir, list(cfg.model_dirs), cfg.headroom_pct, g.host, g.port, g.api_key,
                  g.models_max)
        cfg.bin_dir = self.query_one("#bin_dir", Input).value.strip()
        cfg.model_dirs = self._dirs()
        cfg.gauge_interval = float(self.query_one("#gauge_interval", Input).value)
        cfg.headroom_pct = int(self.query_one("#headroom_pct", Input).value)
        cfg.default_intent = str(self.query_one("#default_intent", Select).value)
        g.host = self.query_one("#gw_host", Input).value.strip()
        g.port = int(self.query_one("#gw_port", Input).value)
        g.api_key = self.query_one("#gw_api_key", Input).value
        g.models_max = int(self.query_one("#gw_models_max", Input).value)
        g.sleep_idle = int(self.query_one("#gw_sleep_idle", Input).value)
        g.autostart = self.query_one("#gw_autostart", Switch).value
        try:
            cfg.save()
        except OSError as e:
            return a.fail(f"cannot save the config: {e}")
        if cfg.bin_dir != before[0]:
            a.detect_build()
            a.poll_host()
        if cfg.model_dirs != before[1]:
            a.rescan()
        a.config_changed()
        a.presets_changed(save=False)               # rewrite the preset file, reload a running gateway
        after = (g.host, g.port, g.api_key, g.models_max)
        changed = [label for (_, label), old, new in zip(RESTART_FIELDS, before[3:], after)
                   if old != new]
        if changed and a.gateway.running:
            self._msg(f"Settings saved. The gateway keeps its old {', '.join(changed)} until it "
                      "is restarted: stop it (d) and start it (u) in Home or Serving.")
        else:
            self._msg("Settings saved.")

    @on(Button.Pressed, "#save")
    def _save_button(self) -> None:
        """The Save button."""
        self.action_save()
