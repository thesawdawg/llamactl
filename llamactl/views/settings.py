"""SettingsView: llama.cpp folder, model folders, gateway, headroom, default intent, HF token."""
from __future__ import annotations

from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widget import Widget
from textual.validation import Function, Integer, Number
from textual.widgets import Button, Input, Label, Select, Static, Switch
from rich.text import Text

from .. import hf, network
from ..hardware import GiB
from ..intents import INTENTS
from ..screens.prompt import PromptScreen
from ..widgets.buttonbar import ButtonBar
from ..widgets.folders import FolderList
from .base import BaseView
from .hf import TOKEN_HINT

LISTEN_OPTIONS = [("This machine (127.0.0.1)", network.LOCAL),
                  ("Local network (0.0.0.0)", network.LAN), ("Custom address", network.CUSTOM)]
TEXT_INPUTS = ("bin_dir", "gauge_interval", "headroom_pct", "gw_host", "gw_port", "gw_models_max",
               "gw_sleep_idle", "wsl_ram")
RESTART_FIELDS = (("host", "host"), ("port", "port"), ("api_key", "API key"),
                  ("models_max", "models-max"))     # gateway settings that are command line options


class SettingsView(BaseView):
    """Form for the app's settings; saving applies them and reloads a running gateway."""

    BINDINGS = [Binding("ctrl+s", "save", "Save")]
    # The -narrow rules key off a class of the screen (an ancestor), so this CSS has to
    # stay unscoped.
    SCOPED_CSS = False
    DEFAULT_CSS = """
    SettingsView { height: 1fr; }
    #set_scroll { padding: 0 1; }
    #set_scroll .section { text-style: bold; margin-top: 1; color: $accent; width: 1fr; }
    #set_scroll .row { height: auto; }
    #set_scroll .lbl { width: 28; padding: 0 1 0 0; }
    #set_scroll .row Input { width: 1fr; max-width: 50; }
    #set_scroll .row Select { width: 1fr; max-width: 50; }
    #set_scroll .hint { color: $text-muted; text-style: italic; padding: 0 1; width: auto; }
    #set_scroll .row Button { margin-left: 1; }
    #wsl_row { display: none; }
    #gw_urls { height: auto; margin-left: 28; padding: 0 1; }
    #set_buttons { margin: 0 1; }
    #set_msg { height: auto; padding: 0 1; }
    .-narrow #set_scroll .row { layout: vertical; }
    .-narrow #set_scroll .lbl { width: 1fr; padding: 0; }
    .-narrow #set_scroll .row Button { margin-left: 0; }
    .-narrow #gw_urls { margin-left: 0; }
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
            yield FolderList(cfg.model_dirs)
            yield Label("Serving", classes="section")
            yield row("Default intent", Select(
                [(i.label, i.key) for i in INTENTS], allow_blank=False, id="default_intent",
                compact=True, value=cfg.default_intent if cfg.default_intent in intent_keys
                else "chat"), "used when you do not pick one")
            yield Label("Gateway", classes="section")
            mode = network.bind_mode(g.host)
            yield row("Listen on", Select(
                LISTEN_OPTIONS, allow_blank=False, id="gw_listen", compact=True, value=mode),
                "who can connect")
            yield row("Custom address", Input(
                g.host, id="gw_host", compact=True, disabled=mode != network.CUSTOM,
                validators=[Function(lambda v: bool(v.strip()) and " " not in v.strip(),
                                     "a host name or address")]), "only for Custom")
            yield row("Port", Input(str(g.port), id="gw_port", compact=True,
                                    validators=[Integer(minimum=1, maximum=65535)]))
            yield Horizontal(
                Label("API key", classes="lbl"),
                Input(g.api_key, id="gw_api_key", password=True, compact=True),
                Button("Generate", id="gw_gen_key", compact=True),
                Label("Bearer token; empty = none", classes="hint"), classes="row")
            yield Static("", id="gw_urls")
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
            yield Horizontal(
                Label("Windows RAM (GiB)", classes="lbl"),
                Input(str(cfg.wsl_windows_ram // GiB) if cfg.wsl_windows_ram else "",
                      id="wsl_ram", compact=True, validators=[Function(
                          lambda v: v.strip() == "" or v.strip().isdigit(),
                          "a whole number of GiB")]),
                Label("GiB Windows has; 0/empty = unknown", classes="hint"),
                classes="row", id="wsl_row")
            yield Label("Hugging Face", classes="section")
            yield row("Access token", Button("Set token", id="set_token", compact=True))
        with ButtonBar(id="set_buttons", classes="-docked"):
            yield Button("Save (Ctrl+S)", id="save", variant="primary")
            yield Button("Edit default profile", id="edit_default")
            yield Button("Run setup again", id="run_setup")
        yield Static("", id="set_msg")

    def on_mount(self) -> None:
        """Show whether a token is set and where clients would connect; detect WSL for the
        Windows-RAM row off the UI thread."""
        self._token_state()
        self._refresh_urls()
        self.app.run_bg(network.wsl_mode, label="Checking WSL", group="wsl", exclusive=True,
                        done=self._wsl_known, quiet=True)

    def _wsl_known(self, mode: str | None) -> None:
        """WSL detection finished: the Windows-RAM row only makes sense under WSL."""
        self.query_one("#wsl_row").display = mode is not None

    def focus_primary(self) -> None:
        """Focus the first field."""
        self.query_one("#bin_dir", Input).focus()

    def _form_host(self) -> str:
        """The listen address the form currently describes."""
        mode = str(self.query_one("#gw_listen", Select).value)
        return network.host_for(mode, self.query_one("#gw_host", Input).value)

    def _refresh_urls(self) -> None:
        """Show where clients would connect, whether that needs a key, and what WSL needs."""
        host = self._form_host()
        port_text = self.query_one("#gw_port", Input).value.strip()
        port = int(port_text) if port_text.isdigit() and 0 < int(port_text) < 65536 \
            else self.app.cfg.gateway.port
        key = self.query_one("#gw_api_key", Input).value
        t = Text("Clients connect to  " + "  ".join(
            f"{u}/v1" for u in network.client_urls(host, port)))
        if warning := network.key_warning(host, key):
            t.append(f"\n! {warning}", style="yellow")
        for note in network.exposure_notes(host):
            t.append(f"\n{note}", style="dim")
        self.query_one("#gw_urls", Static).update(t)

    @on(Select.Changed, "#gw_listen")
    def _listen_changed(self) -> None:
        """Another bind mode: the Custom address is editable only for Custom."""
        mode = str(self.query_one("#gw_listen", Select).value)
        box = self.query_one("#gw_host", Input)
        if mode == network.CUSTOM:
            box.disabled = False
        else:
            box.disabled = True
            box.value = network.host_for(mode)
        self._refresh_urls()

    @on(Input.Changed, "#gw_host, #gw_port, #gw_api_key")
    def _network_input_changed(self) -> None:
        """The address, port or key was edited: redraw the URL line."""
        self._refresh_urls()

    @on(Button.Pressed, "#gw_gen_key")
    def _generate_key(self) -> None:
        """Fill the API key with a new random one, shown (not hidden) and copied once."""
        key = network.new_api_key()
        box = self.query_one("#gw_api_key", Input)
        box.password = False
        box.value = key
        self.app.copy_to_clipboard(key)
        self._msg("Generated an API key and copied it. Save to apply it; clients send it as "
                  "a Bearer token.")

    @staticmethod
    def _bin_ok(value: str) -> bool:
        """True when the dir contains a llama-server executable."""
        return (Path(value).expanduser() / "llama-server").exists()

    def _token_state(self) -> None:
        """Label the token button with whether a token exists."""
        self.query_one("#set_token", Button).label = (
            "Change token (set)" if hf.load_token() else "Set token (none yet)")

    def _msg(self, text: str) -> None:
        """Show a message under the form and in the status bar."""
        self.query_one("#set_msg", Static).update(text)
        self.say(text)

    @on(Button.Pressed, "#set_token")
    def _set_token(self) -> None:
        """Ask for the Hugging Face token."""
        def done(res: str | None) -> None:
            if res is not None:
                hf.save_token(res)
                self._token_state()

        self.app.push_screen(PromptScreen("Hugging Face access token", TOKEN_HINT,
                                          hf.load_token() or "", password=True), done)

    @on(Button.Pressed, "#run_setup")
    def _run_setup(self) -> None:
        """Open the guided first-run setup again."""
        self.app.show_onboarding()

    @on(Button.Pressed, "#edit_default")
    def _edit_default(self) -> None:
        """Open the default profile editor (Models view)."""
        self.app.views["models"].action_edit_default()          # type: ignore[attr-defined]

    def _fields_valid(self) -> bool:
        """True when every text field validates."""
        return all(self.query_one(f"#{i}", Input).is_valid for i in TEXT_INPUTS)

    def action_save(self) -> None:
        """Validate, apply, write the config, then rescan and reload the gateway presets."""
        if not self._fields_valid():
            return self._msg("Fix the invalid fields first.")
        a = self.app
        cfg, g = a.cfg, a.cfg.gateway
        before = (cfg.bin_dir, list(cfg.model_dirs), cfg.headroom_pct, g.host, g.port, g.api_key,
                  g.models_max)
        cfg.bin_dir = self.query_one("#bin_dir", Input).value.strip()
        cfg.model_dirs = self.query_one(FolderList).dirs()
        cfg.gauge_interval = float(self.query_one("#gauge_interval", Input).value)
        cfg.headroom_pct = int(self.query_one("#headroom_pct", Input).value)
        cfg.default_intent = str(self.query_one("#default_intent", Select).value)
        g.host = self._form_host()
        g.port = int(self.query_one("#gw_port", Input).value)
        g.api_key = self.query_one("#gw_api_key", Input).value
        g.models_max = int(self.query_one("#gw_models_max", Input).value)
        g.sleep_idle = int(self.query_one("#gw_sleep_idle", Input).value)
        g.autostart = self.query_one("#gw_autostart", Switch).value
        if self.query_one("#wsl_row").display:
            wsl_raw = self.query_one("#wsl_ram", Input).value.strip()
            gib = int(wsl_raw) if wsl_raw.isdigit() else 0
            cfg.wsl_windows_ram = gib * GiB if gib else 0   # 0/empty = unknown, do not re-ask
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
