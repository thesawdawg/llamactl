"""SessionsView: managed/external server table plus a log/props detail pane."""
from __future__ import annotations

import json
import time
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import DataTable, RichLog

from ..forms import FormScreen
from ..screens.chat import ChatScreen
from ..sessions import Session, probe
from .base import BaseView

SPINNER = "|/-\\"


class SessionsView(BaseView):
    """Sessions table, health polling, log tail / props detail pane."""

    BINDINGS = [
        Binding("t", "chat", "Chat"),
        Binding("s", "stop", "Stop/forget"),
        Binding("a", "attach", "Attach URL"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.health: dict[str, str] = {}
        self.props: dict[str, dict] = {}
        self.pending: dict[str, float] = {}
        self.detail_text = ""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield DataTable(id="sess_table", cursor_type="row")
            yield RichLog(id="detail", wrap=True, max_lines=2000)

    def on_mount(self) -> None:
        t = self.query_one("#sess_table", DataTable)
        t.add_columns("Name", "URL", "PID", "Kind", "State")
        self.set_interval(1.0, self.poll_health)
        self.set_interval(1.0, self.update_detail)
        self.render_sessions()

    @property
    def sess_table(self) -> DataTable:
        return self.query_one("#sess_table", DataTable)

    def selected_session(self) -> Session | None:
        """Highlighted session row."""
        if self.sess_table.row_count == 0:
            return None
        return self.app.store.sessions[self.sess_table.cursor_row]

    def launch(self, model: str) -> Session:
        """Start a managed server for a model and track it as pending.

        Args:
            model: Path to the GGUF file.

        Returns:
            The new session.
        """
        s = self.app.store.launch_server(self.app.cfg, model)
        self.pending[s.id] = time.time()
        self.health[s.id] = "starting 0s"
        self.render_sessions()
        self.sess_table.move_cursor(row=self.sess_table.row_count - 1)
        self.notify(f"Loading {s.name} on port {s.url.rsplit(':', 1)[1]}...", title="Launching")
        return s

    def render_sessions(self) -> None:
        """Redraw the sessions table, keeping the cursor row."""
        t, row = self.sess_table, self.sess_table.cursor_row
        t.clear()
        for s in self.app.store.sessions:
            t.add_row(s.name, s.url, s.pid or "-", "external" if s.external else "managed",
                      self.health.get(s.id, "?"), key=s.id)
        if t.row_count:
            t.move_cursor(row=min(row, t.row_count - 1))

    @work(thread=True, exclusive=True, group="health")
    def poll_health(self) -> None:
        """Probe all sessions off the UI thread and report launch progress/outcome."""
        for s in list(self.app.store.sessions):
            props = probe(s.url)
            self.props[s.id] = props or {}
            waited = int(time.time() - self.pending[s.id]) if s.id in self.pending else 0
            if props:
                self.health[s.id] = "ready"
                if s.id in self.pending:
                    del self.pending[s.id]
                    self.app.call_from_thread(self.notify, f"{s.name} is ready at {s.url}", title="Model loaded")
            elif not s.external and not s.alive:
                self.health[s.id] = "crashed"
                if s.id in self.pending:
                    del self.pending[s.id]
                    tail = " | ".join(self.log_tail(s, 3))
                    self.app.call_from_thread(self.notify, f"{s.name} exited during load. {tail}"[:400],
                                          title="Launch failed", severity="error", timeout=15)
            elif s.id in self.pending:
                spin = SPINNER[waited % len(SPINNER)]
                self.health[s.id] = f"{spin} {'loading model' if props == {} else 'starting'} {waited}s"
            else:
                self.health[s.id] = "down" if props is None else "loading"
        self.app.call_from_thread(self.render_sessions)

    @staticmethod
    def log_tail(s: Session, n: int) -> list[str]:
        """Last n log lines, reading only the end of the file."""
        try:
            with open(s.log, "rb") as fh:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - 16384))
                return fh.read().decode(errors="replace").splitlines()[-n:]
        except OSError:
            return []

    def update_detail(self) -> None:
        """Show the tail of the selected session's log, or props for external ones. Redraws only on change."""
        s = self.selected_session()
        if s is None:
            text = "No sessions. Select a model on the Models view and press Enter to launch, or 'a' to attach."
        elif s.external:
            text = f"External server {s.url}\n{json.dumps(self.props.get(s.id, {}), indent=2)[:1500]}"
        else:
            text = "\n".join(self.log_tail(s, 60))
        if text != self.detail_text:
            self.detail_text = text
            log = self.query_one("#detail", RichLog)
            log.clear()
            log.write(text)

    def action_chat(self) -> None:
        s = self.selected_session()
        if s is None:
            return self.say("No session selected.")
        self.app.push_screen(ChatScreen(s))

    def action_stop(self) -> None:
        s = self.selected_session()
        if s is None:
            return
        self.app.store.stop(s)
        self.render_sessions()
        self.say(f"Stopped/forgot {s.name}")

    def action_attach(self) -> None:
        def done(res: dict | None) -> None:
            if res and res["url"]:
                s = self.app.store.attach(res["url"])
                self.render_sessions()
                self.say(f"Attached to {s.url}")

        self.app.push_screen(FormScreen("Attach to running server",
                                        [("url", "URL or host:port", "localhost:8080",
                                          "Address of a llama-server that is already running. It is only tracked, never stopped by this tool.")]),
                             done)
