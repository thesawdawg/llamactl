"""SessionsView: live session states, slot counts, log modal, confirm on stop."""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, ProgressBar, RichLog, Static

from ..screens.prompt import PromptScreen
from ..screens.chat import ChatScreen
from ..screens.confirm import ConfirmScreen
from ..screens.log import LogScreen
from ..sessions import Session, probe
from .base import BaseView

SPINNER = "|/-\\"
STATE_STYLE = {"starting": "dim", "loading": "yellow", "ready": "green",
               "down": "grey50", "crashed": "red"}
PROGRESS_RES = (re.compile(r"(\d+)\s*/\s*(\d+)\s*(?:tensors|layers)"),
                re.compile(r"loading model.*?(\d+)\s*%"))


def _load_frac(lines: list[str]) -> float | None:
    """Parse a load progress fraction from log lines, or None."""
    for line in reversed(lines):
        if m := PROGRESS_RES[0].search(line):
            total = int(m.group(2))
            return int(m.group(1)) / total if total else None
        if m := PROGRESS_RES[1].search(line):
            return int(m.group(1)) / 100
    return None


COLUMNS = (("Name", "name"), ("State", "state"), ("URL", "url"), ("Port", "port"),
           ("Slots", "slots"), ("Uptime", "uptime"), ("Kind", "kind"))


def _uptime(started: float) -> str:
    """Seconds since `started` as '45s', '12m', '1h02m' or '2d'."""
    if not started:
        return "-"
    s = int(time.time() - started)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 86400}d"


def _probe_one(s: Session) -> tuple[str, dict | None, tuple[int, int] | None, bool]:
    """Probe one session off the UI thread.

    Returns:
        (id, /props or None, (used, total) slots or None, process alive).
    """
    props = probe(s.url)
    slots = None
    if props:
        try:
            r = httpx.get(f"{s.url}/slots", timeout=1.0)
            if r.status_code == 200:
                data = r.json()
                slots = (sum(1 for sl in data if sl.get("is_processing")), len(data))
        except (httpx.HTTPError, ValueError):
            pass
    return s.id, props, slots, s.alive


class SessionsView(BaseView):
    """Sessions table, health polling, log/props detail pane."""

    BINDINGS = [
        Binding("t", "chat", "Chat"),
        Binding("s", "stop", "Stop/forget"),
        Binding("a", "attach", "Attach URL"),
        Binding("o", "open_log", "Log"),
        Binding("y", "copy_url", "Copy URL"),
    ]
    CSS = """
    #sess_table { height: 55%; }
    #load_box { height: auto; display: none; }
    #loading_banner { width: 1fr; color: $warning; }
    #load_progress { width: 30; display: none; }
    #detail { height: 1fr; border: round $primary; }
    #empty_detail { height: 1fr; border: round $primary; padding: 1 2; display: none; }
    #s_buttons { height: auto; }
    #s_buttons Button { margin-right: 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.state: dict[str, str] = {}
        self.slots: dict[str, tuple[int, int] | None] = {}
        self.props: dict[str, dict] = {}
        self.pending: dict[str, float] = {}
        self._crash_flash: list[str] = []
        self.detail_text = ""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield DataTable(id="sess_table", cursor_type="row")
            with Horizontal(id="load_box"):
                yield Static(id="loading_banner")
                yield ProgressBar(id="load_progress", total=100, show_eta=False)
            yield RichLog(id="detail", wrap=True, max_lines=2000)
            yield Static("No sessions. Go to Models (1) and press Enter to launch, "
                         "or press a to attach a running server.", id="empty_detail")
            with Horizontal(id="s_buttons"):
                yield Button("Chat", id="s_chat")
                yield Button("Stop", id="s_stop", variant="error")
                yield Button("Attach", id="s_attach")
                yield Button("Log", id="s_log")
                yield Button("Copy URL", id="s_copy")

    def on_mount(self) -> None:
        t = self.sess_table
        for label, key in COLUMNS:
            t.add_column(label, key=key)
        self.set_interval(1.0, self.poll_health)
        self.set_interval(1.0, self.update_detail)
        self.sync_table()

    def focus_primary(self) -> None:
        self.sess_table.focus()

    @property
    def sess_table(self) -> DataTable:
        return self.query_one("#sess_table", DataTable)

    def selected_session(self) -> Session | None:
        """Highlighted session row."""
        rows = list(self.sess_table.rows)
        if not rows:
            return None
        key = rows[min(self.sess_table.cursor_row, len(rows) - 1)]
        return next((s for s in self.app.store.sessions if s.id == str(key.value)), None)

    def launch(self, model: str) -> None:
        """Kick off a managed server launch without blocking the UI thread.

        Args:
            model: Path to the GGUF file.
        """
        self.say(f"Launching {Path(model).stem}...")
        self.app.switch_view("sessions")
        self._launch_worker(model)

    @work(thread=True, group="launch")
    def _launch_worker(self, model: str) -> None:
        """Bind a port and spawn llama-server off the UI thread."""
        try:
            s = self.app.store.launch_server(self.app.cfg, model)
        except (RuntimeError, OSError) as e:
            self.app.call_from_thread(
                self.notify, f"Could not launch {Path(model).stem}: {e}",
                title="Launch failed", severity="error", timeout=15)
            return
        self.app.call_from_thread(self._launch_done, s)

    def _launch_done(self, s: Session) -> None:
        """Register the new session as pending and select its row."""
        self.pending[s.id] = time.time()
        self.state[s.id] = "starting"
        self.sync_table()
        for i, key in enumerate(self.sess_table.rows):
            if str(key.value) == s.id:
                self.sess_table.move_cursor(row=i)
                break
        self.notify(f"Loading {s.name} on port {s.url.rsplit(':', 1)[1]}...",
                    title="Launching")

    def _state_text(self, s: Session) -> Text:
        """Styled state cell for a session."""
        st = self.state.get(s.id, "down")
        label = st
        if st in ("starting", "loading") and s.id in self.pending:
            waited = int(time.time() - self.pending[s.id])
            label = f"{SPINNER[waited % len(SPINNER)]} {st} {waited}s"
        return Text(label, style=STATE_STYLE.get(st, ""))

    def _cells(self, s: Session) -> tuple:
        """All cell values for one session row."""
        used, total = self.slots.get(s.id) or (None, None)
        return (s.name, self._state_text(s), s.url, s.url.rsplit(":", 1)[1],
                f"{used}/{total}" if total is not None else "-",
                _uptime(s.started), "external" if s.external else "managed")

    def sync_table(self) -> None:
        """Add/update/remove rows in place to match the session list."""
        t = self.sess_table
        want = {s.id: s for s in self.app.store.sessions}
        for key in [k for k in t.rows if str(k.value) not in want]:
            t.remove_row(key)
        for s in want.values():
            if s.id in t.rows:
                for col, v in zip([k for _, k in COLUMNS], self._cells(s)):
                    t.update_cell(s.id, col, v)
            else:
                t.add_row(*self._cells(s), key=s.id)

    @work(thread=True, exclusive=True, group="health")
    def poll_health(self) -> None:
        """Probe all sessions concurrently and apply the result once."""
        sessions = list(self.app.store.sessions)
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(_probe_one, sessions))
        self.app.call_from_thread(self._apply_health, results)

    def _apply_health(self, results: list) -> None:
        """Fold probe results into state, notify transitions, redraw cells."""
        sessions = {s.id: s for s in self.app.store.sessions}
        for sid, props, slots, alive in results:
            s = sessions.get(sid)
            if not s:
                continue
            self.props[sid] = props or {}
            self.slots[sid] = slots
            if props:
                self.state[sid] = "ready"
                if sid in self.pending:
                    del self.pending[sid]
                    self.notify(f"{s.name} is ready at {s.url}", title="Model loaded")
            elif not s.external and not alive:
                self.state[sid] = "crashed"
                if sid in self.pending:
                    del self.pending[sid]
                    tail = self.log_tail(s, 3)
                    self._crash_flash.append(
                        f"[red]{s.name} crashed - {tail[-1] if tail else ''}[/red]")
                    self.notify(f"{s.name} exited during load. {' | '.join(tail)}"[:400],
                                title="Launch failed", severity="error", timeout=15)
            else:
                self.state[sid] = "loading" if sid in self.pending else "down"
        self.sync_table()
        self._update_banner()

    def _update_banner(self) -> None:
        """Refresh the loading banner: pending sessions plus one-shot crash tails."""
        box = self.query_one("#load_box")
        bar = self.query_one("#load_progress", ProgressBar)
        banner = self.query_one("#loading_banner", Static)
        if not self.pending and not self._crash_flash:
            box.display = False
            bar.display = False
            return
        box.display = True
        lines: list[str] = []
        frac: float | None = None
        sessions = {s.id: s for s in self.app.store.sessions}
        for sid, t0 in self.pending.items():
            s = sessions.get(sid)
            if not s:
                continue
            waited = int(time.time() - t0)
            tail = self.log_tail(s, 6)
            last = (tail[-1].strip() if tail else "")
            prefix = (f"{SPINNER[waited % len(SPINNER)]} Loading {s.name} "
                      f"on :{s.url.rsplit(':', 1)[1]} - {waited}s")
            width = max(20, (self.size.width or 120) - len(prefix) - 6)
            lines.append(f"{prefix} - {last[:width]}")
            frac = frac if frac is not None else _load_frac(tail)
        lines.extend(self._crash_flash)
        self._crash_flash.clear()
        banner.update("\n".join(lines))
        bar.display = frac is not None
        if frac is not None:
            bar.update(progress=frac * 100)

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
        """Show the tail of the selected session's log, or props for external ones."""
        s = self.selected_session()
        self.query_one("#empty_detail", Static).display = s is None
        log = self.query_one("#detail", RichLog)
        log.display = s is not None
        if s is None:
            return
        if s.external:
            text = f"External server {s.url}\n{json.dumps(self.props.get(s.id, {}), indent=2)[:1500]}"
        else:
            text = "\n".join(self.log_tail(s, 60))
        if text != self.detail_text:
            self.detail_text = text
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
        if s.external:
            self._stop(s)
            return

        def done(ok: bool | None) -> None:
            if ok:
                self._stop(s)

        self.app.push_screen(ConfirmScreen(f"Stop {s.name} (pid {s.pid})?"), done)

    def _stop(self, s: Session) -> None:
        """Terminate or forget the session and refresh."""
        self.app.store.stop(s)
        self.state.pop(s.id, None)
        self.pending.pop(s.id, None)
        self.sync_table()
        self.say(f"Stopped/forgot {s.name}")

    def action_attach(self) -> None:
        def done(res: str | None) -> None:
            if res:
                s = self.app.store.attach(res)
                self.sync_table()
                self.say(f"Attached to {s.url}")

        self.app.push_screen(PromptScreen(
            "Attach to running server",
            "Address of a llama-server that is already running (URL or host:port). "
            "It is only tracked, never stopped by this tool.", "localhost:8080"), done)

    def action_open_log(self) -> None:
        s = self.selected_session()
        if s is None or s.external:
            return self.say("No managed session selected.")
        self.app.push_screen(LogScreen(s.log, s.name))

    def action_copy_url(self) -> None:
        s = self.selected_session()
        if s is None:
            return self.say("No session selected.")
        self.app.copy_to_clipboard(s.url)
        self.say(f"Copied {s.url}")

    @on(Button.Pressed, "#s_chat")
    def _b_chat(self) -> None:
        self.action_chat()

    @on(Button.Pressed, "#s_stop")
    def _b_stop(self) -> None:
        self.action_stop()

    @on(Button.Pressed, "#s_attach")
    def _b_attach(self) -> None:
        self.action_attach()

    @on(Button.Pressed, "#s_log")
    def _b_log(self) -> None:
        self.action_open_log()

    @on(Button.Pressed, "#s_copy")
    def _b_copy(self) -> None:
        self.action_copy_url()
