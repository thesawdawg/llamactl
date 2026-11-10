"""ServingView: gateway models and standalone/attached servers in one table, with a log pane."""
from __future__ import annotations

import json
import time
from pathlib import Path

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import DataTable, RichLog, Static

from .. import actions
from ..cli import tail_lines
from ..gateway import FAILED, LOADED, LOADING, SLEEPING, UNLOADED, GatewayModel
from ..screens.confirm import ConfirmScreen
from ..screens.log import LogScreen
from ..screens.prompt import PromptScreen
from ..serving import GW_STOPPED, ServingSnapshot, uptime
from ..sessions import Session, probe
from .base import BaseView
from .home import MODEL_STYLE

COLUMNS = (("Name", "name"), ("State", "state"), ("Kind", "kind"), ("URL", "url"),
           ("Uptime", "uptime"))
KIND_GATEWAY, KIND_STANDALONE, KIND_ATTACHED = "gateway", "standalone", "attached"
TAIL_LINES = 80
DETAIL_INTERVAL = 1.0           # seconds between log tail checks while the view is shown
PROPS_INTERVAL = 5.0            # seconds between /props fetches of an attached server
ATTACH_HINT = ("Address of a llama-server that is already running (URL or host:port). It is "
               "only tracked, never stopped by this tool.")


def model_state_text(m: GatewayModel) -> Text:
    """State cell of a gateway model: the state, and the exit code of a failed one."""
    label = m.state if m.state != FAILED or m.exit_code is None else f"failed (exit {m.exit_code})"
    return Text(label, style=MODEL_STYLE.get(m.state, ""))


class ServingView(BaseView):
    """One table for everything that serves: gateway models, standalone and attached servers."""

    BINDINGS = [
        Binding("l", "load", "Load"),
        Binding("x", "unload", "Unload"),
        Binding("s", "stop", "Stop"),
        Binding("u", "start_gateway", "GW up"),
        Binding("d", "stop_gateway", "GW down"),
        Binding("a", "attach", "Attach"),
        Binding("o", "open_log", "Log"),
        Binding("y", "copy_url", "Copy URL"),
        Binding("c", "copy_snippet", "Copy curl"),
    ]
    DEFAULT_CSS = """
    ServingView { height: 1fr; }
    #sv_gateway { height: 1; padding: 0 1; }
    #sv_table { height: 40%; }
    #sv_empty { color: $text-muted; padding: 1 2; height: auto; }
    #sv_detail_box { height: 1fr; border: round $primary; }
    #sv_detail { height: 1fr; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._detail_text: str | None = None
        self._detail_key = ""
        self._props: dict[str, tuple[float, str]] = {}
        self._want = ""                     # row key to select once it exists

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("", id="sv_gateway")
            yield DataTable(id="sv_table", cursor_type="row")
            yield Static("", id="sv_empty")
            with Vertical(id="sv_detail_box"):
                yield RichLog(id="sv_detail", wrap=True, markup=False, highlight=False,
                              max_lines=2000)

    def on_mount(self) -> None:
        t = self.table
        for label, key in COLUMNS:
            t.add_column(label, key=key)
        self.set_interval(DETAIL_INTERVAL, self._tick)
        self.query_one("#sv_detail_box").border_title = "Log"
        if self.app.snapshot is not None:
            self.on_serving(self.app.snapshot)
        else:
            self._empty_state()

    @property
    def table(self) -> DataTable:
        """The one table."""
        return self.query_one("#sv_table", DataTable)

    def focus_primary(self) -> None:
        self.table.focus()
        self.refresh_detail(force=True)

    # -- data -------------------------------------------------------------------------

    def snapshot(self) -> ServingSnapshot | None:
        """The app's latest serving snapshot."""
        return self.app.snapshot

    def rows(self, snap: ServingSnapshot) -> dict[str, tuple]:
        """Table rows by key for a snapshot.

        Args:
            snap: Gateway and sessions now.

        Returns:
            Key -> cells (name, state, kind, url, uptime).
        """
        out: dict[str, tuple] = {}
        for m in snap.models or ():
            out[f"gw:{m.name}"] = (m.name, model_state_text(m), KIND_GATEWAY, snap.status.url, "-")
        for s in snap.sessions:
            state = snap.health.get(s.id, "down")
            out[f"s:{s.id}"] = (s.name, Text(state, style=MODEL_STYLE.get(state, "")),
                                KIND_ATTACHED if s.external else KIND_STANDALONE, s.url,
                                uptime(s.started, snap.at))
        return out

    def on_serving(self, snapshot: ServingSnapshot) -> None:
        t = self.table
        keep = self.selected_key()
        rows = self.rows(snapshot)
        for key in [k for k in t.rows if k.value not in rows]:
            t.remove_row(key)
        for key, cells in rows.items():
            if key in t.rows:
                for col, v in zip([c for _, c in COLUMNS], cells):
                    t.update_cell(key, col, v)
            else:
                t.add_row(*cells, key=key)
        want = self._want if self._want in rows else keep
        if want in rows:
            self._want = "" if want == self._want else self._want
            idx = t.get_row_index(want)
            if t.cursor_row != idx:
                t.move_cursor(row=idx)
        self._gateway_line(snapshot)
        self._empty_state()
        self.refresh_detail()

    def on_presets_changed(self, paths) -> None:
        if self.app.snapshot is not None:
            self._gateway_line(self.app.snapshot)

    def on_config_changed(self) -> None:
        if self.app.snapshot is not None:
            self._gateway_line(self.app.snapshot)

    def _gateway_line(self, snap: ServingSnapshot) -> None:
        a = self.app
        state = a.gateway_state()
        g = a.cfg.gateway
        t = Text("Gateway ", style="bold")
        t.append(state, style={"running": "green", "stopped": "dim"}.get(state, "yellow"))
        t.append(f"  {snap.status.url}/v1  max {g.models_max} loaded at once")
        if snap.error:
            t.append(f"  ! {snap.error}", style="red")
        self.query_one("#sv_gateway", Static).update(t)

    def _empty_state(self) -> None:
        snap = self.app.snapshot
        empty = self.table.row_count == 0
        msg = ""
        if empty:
            if snap is None:
                msg = "Checking the gateway..."
            elif snap.status.running and snap.models is not None:
                msg = ("The gateway runs but lists no models: add a preset in Models (2). "
                       "Press a to attach a running server.")
            elif snap.status.running:
                msg = "The gateway does not answer its model list; see its log below."
            else:
                msg = ("The gateway is stopped. Press u to start it, or Enter on a model in "
                       "Models (2) to serve it. Press a to attach a running server.")
        lbl = self.query_one("#sv_empty", Static)
        lbl.update(msg)
        lbl.display = bool(msg)

    def focus_model(self, name: str) -> None:
        """Select a gateway model's row as soon as it exists.

        Args:
            name: Model id.
        """
        self._want = f"gw:{name}"

    # -- selection --------------------------------------------------------------------

    def selected_key(self) -> str:
        """Key of the highlighted row, "" when the table is empty."""
        t = self.table
        if not t.row_count:
            return ""
        return str(list(t.rows)[min(t.cursor_row, t.row_count - 1)].value)

    def selected_model(self) -> GatewayModel | None:
        """The highlighted gateway model, or None."""
        key, snap = self.selected_key(), self.app.snapshot
        if not key.startswith("gw:") or snap is None:
            return None
        return next((m for m in snap.models or () if m.name == key[3:]), None)

    def selected_session(self) -> Session | None:
        """The highlighted standalone or attached server, or None."""
        key, snap = self.selected_key(), self.app.snapshot
        if not key.startswith("s:") or snap is None:
            return None
        return next((s for s in snap.sessions if s.id == key[2:]), None)

    @on(DataTable.RowHighlighted, "#sv_table")
    def _highlighted(self) -> None:
        self.refresh_detail(force=True)

    # -- detail pane ------------------------------------------------------------------

    def _tick(self) -> None:
        if self.display and self.is_attached:
            self.refresh_detail()

    def detail_text(self) -> str:
        """What the log pane should show for the highlighted row."""
        snap, key = self.app.snapshot, self.selected_key()
        if snap is None:
            return ""
        if key.startswith("s:"):
            s = self.selected_session()
            if s is None:
                return ""
            if s.external:
                return self._props_text(s)
            return self._tail(s.log, f"{s.name} has no log yet")
        if snap.log:
            return self._tail(snap.log, "the gateway has not logged anything yet")
        return "The gateway is stopped." if not snap.status.running else ""

    @staticmethod
    def _tail(log: str, empty: str) -> str:
        try:
            return "\n".join(tail_lines(Path(log), TAIL_LINES)) or empty
        except OSError:
            return empty

    def _props_text(self, s: Session) -> str:
        """``/props`` of an attached server, fetched in a worker at most every few seconds."""
        at, text = self._props.get(s.id, (0.0, ""))
        if time.monotonic() - at >= PROPS_INTERVAL:
            self._props[s.id] = (time.monotonic(), text)

            def done(props: dict | None) -> None:
                body = (json.dumps(props, indent=2)[:3000] if props is not None
                        else "not reachable")
                self._props[s.id] = (time.monotonic(), f"Attached server {s.url}\n{body}")
                self.refresh_detail()

            self.app.run_bg(lambda: probe(s.url), label=f"Reading {s.url}", group="props",
                            done=done, quiet=True)
        return text or f"Attached server {s.url}\nreading /props..."

    def refresh_detail(self, force: bool = False) -> None:
        """Redraw the log pane, but only when the text or the selected row changed."""
        key = self.selected_key()
        text = self.detail_text()
        if not force and text == self._detail_text and key == self._detail_key:
            return
        self._detail_text, self._detail_key = text, key
        log = self.query_one("#sv_detail", RichLog)
        log.clear()
        log.write(text)

    # -- actions ----------------------------------------------------------------------

    def action_load(self) -> None:
        """Load the highlighted gateway model."""
        m = self.selected_model()
        if m is None:
            return self.say("Select a gateway model to load.")
        if m.state in (LOADED, LOADING, SLEEPING):
            return self.say(f"{m.name} is already {m.state}.")
        self.app.gateway_load(m.name)

    def action_unload(self) -> None:
        """Unload the highlighted gateway model."""
        m = self.selected_model()
        if m is None:
            return self.say("Select a gateway model to unload.")
        if m.state in (UNLOADED, FAILED):
            return self.say(f"{m.name} is not loaded.")
        self.app.gateway_unload(m.name)

    def action_stop(self) -> None:
        """Stop a standalone server (after a confirmation) or forget an attached one."""
        s = self.selected_session()
        if s is None:
            return self.say("Select a standalone or attached server (gateway models: x unloads).")
        if s.external:
            return self._stop(s)

        def done(ok: bool | None) -> None:
            if ok:
                self._stop(s)

        self.app.push_screen(ConfirmScreen(f"Stop {s.name} (pid {s.pid})?"), done)

    def _stop(self, s: Session) -> None:
        self.app.run_bg(lambda: self.app.store.stop(s), label=f"Stopping {s.name}", group="stop",
                        done=lambda _: self.say(f"{'Detached from' if s.external else 'Stopped'} "
                                                f"{s.name}"),
                        always=self.app.refresh_serving)

    def action_start_gateway(self) -> None:
        """Start the gateway."""
        state = self.app.gateway_state()
        if state != GW_STOPPED:
            return self.say(f"The gateway is {state}.")
        self.app.start_gateway()

    def action_stop_gateway(self) -> None:
        """Stop the gateway after a confirmation."""
        if self.app.gateway_state() == GW_STOPPED:
            return self.say("The gateway is not running.")

        def done(ok: bool | None) -> None:
            if ok:
                self.app.stop_gateway()

        self.app.push_screen(ConfirmScreen("Stop the gateway? Loaded models will be unloaded."),
                             done)

    def action_attach(self) -> None:
        """Track an already-running llama-server by URL."""
        def done(res: str | None) -> None:
            if res:
                self.app.run_bg(lambda: self.app.store.attach(res), label="Attaching",
                                group="attach",
                                done=lambda s: self.say(f"Attached to {s.url}"),
                                always=self.app.refresh_serving)

        self.app.push_screen(PromptScreen("Attach to running server", ATTACH_HINT,
                                          "localhost:8080"), done)

    def action_open_log(self) -> None:
        """Show the whole log of the gateway or of the highlighted standalone server."""
        s, snap = self.selected_session(), self.app.snapshot
        if s is not None:
            if s.external:
                return self.say("An attached server has no log here; its /props show below.")
            return self.app.push_screen(LogScreen(s.log, s.name))
        if snap is not None and snap.log:
            return self.app.push_screen(LogScreen(snap.log, "gateway"))
        self.say("The gateway is not running; it has no log to show.")

    def _target(self) -> tuple[str, str | None, bool]:
        """(base URL, model id or None, embedding) of the highlighted row, else the gateway."""
        s = self.selected_session()
        if s is not None:
            return s.url, None, False
        m = self.selected_model()
        url = self.app.snapshot.status.url if self.app.snapshot else self.app.gateway.listen_url
        if m is None:
            return url, None, False
        alias = m.name.split("@", 1)[0]
        embedding = any(ent.alias == alias and p in self.app.infos
                        and self.app.infos[p].is_embedding
                        for p, ent in self.app.cfg.models.items())
        return url, m.name, embedding

    def action_copy_url(self) -> None:
        """Copy the OpenAI-style base URL of the highlighted server (the gateway's for models)."""
        url = f"{self._target()[0]}/v1"
        self.app.copy_to_clipboard(url)
        self.say(f"Copied {url}")

    def action_copy_snippet(self) -> None:
        """Copy a curl command for the highlighted model or server."""
        url, model, embedding = self._target()
        if model is None and self.selected_session() is None:
            return self.say("Select a model or server first.")
        self.app.copy_to_clipboard(actions.curl_snippet(
            url, model, api_key=bool(self.app.cfg.gateway.api_key) and model is not None,
            embedding=embedding))
        self.say("Copied the curl command")
