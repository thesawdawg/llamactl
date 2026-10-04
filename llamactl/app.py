"""Textual TUI for launching, monitoring and chatting with llama.cpp servers."""
from __future__ import annotations

import json
import time
import subprocess
from pathlib import Path

import httpx
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import DataTable, Footer, Header, Input, Label, RichLog, TabbedContent, TabPane

from .forms import FormScreen
from .hf_screen import HFScreen
from . import hostinfo
from .command import build_command
from .config import Config, Profile, discover_models
from .gguf import GGUFError, ModelInfoCache, read_model_info
from .screens.profile import ProfileEditor
from .sessions import Session, SessionStore, probe

SPINNER = "|/-\\"


class ChatScreen(Screen):
    """Streaming chat against a server's OpenAI-compatible endpoint."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def __init__(self, session: Session) -> None:
        super().__init__()
        self.session_ = session
        self.history: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="chat", wrap=True, markup=True)
        yield Input(placeholder=f"Message {self.session_.name} (Esc to go back)", id="msg")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"Chat: {self.session_.name} @ {self.session_.url}"
        self.query_one("#msg", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        self.history.append({"role": "user", "content": text})
        self.query_one("#chat", RichLog).write(f"[b cyan]you>[/b cyan] {text}")
        self.stream_reply()

    @work(thread=True)
    def stream_reply(self) -> None:
        """Stream the assistant reply into the log, one line at a time."""
        log = self.query_one("#chat", RichLog)
        reply, line = "", ""
        body = {"messages": self.history, "stream": True}
        try:
            with httpx.stream("POST", f"{self.session_.url}/v1/chat/completions", json=body, timeout=None) as r:
                for raw in r.iter_lines():
                    if not raw.startswith("data: ") or raw.endswith("[DONE]"):
                        continue
                    delta = json.loads(raw[6:])["choices"][0]["delta"].get("content") or ""
                    reply, line = reply + delta, line + delta
                    while "\n" in line:
                        out, line = line.split("\n", 1)
                        self.app.call_from_thread(log.write, f"[b green]ai>[/b green] {out}")
            if line:
                self.app.call_from_thread(log.write, f"[b green]ai>[/b green] {line}")
        except (httpx.HTTPError, KeyError, ValueError) as e:
            self.app.call_from_thread(log.write, f"[red]error: {e}[/red]")
        self.history.append({"role": "assistant", "content": reply})


class LlamaCtl(App):
    """Main application."""

    TITLE = "llamactl"
    CSS = """
    #sess_table { height: 40%; }
    #detail { height: 1fr; border: round $primary; }
    #status { height: auto; padding: 0 1; }
    """
    BINDINGS = [
        Binding("l", "launch_server", "Server"),
        Binding("c", "launch_cli", "CLI chat"),
        Binding("e", "edit_profile", "Edit profile"),
        Binding("d", "edit_default", "Defaults"),
        Binding("g", "edit_settings", "Settings"),
        Binding("a", "attach", "Attach URL"),
        Binding("h", "hf", "HuggingFace"),
        Binding("t", "chat", "Chat (web API)"),
        Binding("s", "stop", "Stop/forget"),
        Binding("r", "refresh", "Rescan"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.cfg = Config.load()
        self.store = SessionStore()
        self.models: list[Path] = []
        self.health: dict[str, str] = {}
        self.props: dict[str, dict] = {}
        self.pending: dict[str, float] = {}
        self.detail_text = ""
        self.host = hostinfo.Host()
        self.info_cache = ModelInfoCache()

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="models"):
            with TabPane("Models", id="models"):
                yield DataTable(id="model_table", cursor_type="row")
            with TabPane("Sessions", id="sessions"):
                with Vertical():
                    yield DataTable(id="sess_table", cursor_type="row")
                    yield RichLog(id="detail", wrap=True, max_lines=2000)
        yield Label("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.model_table.add_columns("Model", "Size", "Fits?", "ctx", "ngl", "port", "custom")
        self.sess_table.add_columns("Name", "URL", "PID", "Kind", "State")
        self.action_refresh()
        self.set_interval(1.0, self.poll_health)
        self.set_interval(1.0, self.update_detail)

    @property
    def model_table(self) -> DataTable:
        return self.query_one("#model_table", DataTable)

    @property
    def sess_table(self) -> DataTable:
        return self.query_one("#sess_table", DataTable)

    def say(self, msg: str) -> None:
        """Show a one-line status message."""
        self.query_one("#status", Label).update(msg)

    def selected_model(self) -> str | None:
        """Path of the highlighted model row."""
        if not self.models or self.model_table.row_count == 0:
            return None
        return str(self.models[self.model_table.cursor_row])

    def selected_session(self) -> Session | None:
        """Highlighted session row."""
        if self.sess_table.row_count == 0:
            return None
        return self.store.sessions[self.sess_table.cursor_row]

    def action_refresh(self) -> None:
        """Rescan model dirs and rebuild both tables."""
        self.models = discover_models(self.cfg.model_dirs)
        self.host = hostinfo.detect()
        t = self.model_table
        t.clear()
        for m in self.models:
            p = self.cfg.profile_for(str(m))
            t.add_row(m.name, f"{m.stat().st_size / 1e9:.2f} GB",
                      hostinfo.estimate(self.host, m.stat().st_size)[0], p.ctx_size or "auto", p.gpu_layers,
                      p.port or "auto", "yes" if str(m) in self.cfg.profiles else "")
        self.render_sessions()
        self.say(f"{len(self.models)} model(s) in {', '.join(self.cfg.model_dirs)}\n{self.host.summary()}")

    def render_sessions(self) -> None:
        """Redraw the sessions table, keeping the cursor row."""
        t, row = self.sess_table, self.sess_table.cursor_row
        t.clear()
        for s in self.store.sessions:
            t.add_row(s.name, s.url, s.pid or "-", "external" if s.external else "managed", self.health.get(s.id, "?"))
        if t.row_count:
            t.move_cursor(row=min(row, t.row_count - 1))

    @work(thread=True, exclusive=True, group="health")
    def poll_health(self) -> None:
        """Probe all sessions off the UI thread and report launch progress/outcome."""
        for s in list(self.store.sessions):
            props = probe(s.url)
            self.props[s.id] = props or {}
            waited = int(time.time() - self.pending[s.id]) if s.id in self.pending else 0
            if props:
                self.health[s.id] = "ready"
                if s.id in self.pending:
                    del self.pending[s.id]
                    self.call_from_thread(self.notify, f"{s.name} is ready at {s.url}", title="Model loaded")
                    self.call_from_thread(self.say, f"{s.name} ready at {s.url} (loaded in {waited}s). Press t to chat.")
            elif not s.external and not s.alive:
                self.health[s.id] = "crashed"
                if s.id in self.pending:
                    del self.pending[s.id]
                    tail = " | ".join(self.log_tail(s, 3))
                    self.call_from_thread(self.notify, f"{s.name} exited during load. {tail}"[:400], title="Launch failed", severity="error", timeout=15)
            elif s.id in self.pending:
                self.health[s.id] = f"{SPINNER[waited % len(SPINNER)]} {'loading model' if props == {} else 'starting'} {waited}s"
                last = (self.log_tail(s, 1) or [""])[0][:100]
                self.call_from_thread(self.say, f"{self.health[s.id]} - {s.name}: {last}")
            else:
                self.health[s.id] = "down" if props is None else "loading"
        self.call_from_thread(self.render_sessions)

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
            text = "No sessions. Select a model and press 'l' to launch, or 'a' to attach."
        elif s.external:
            text = f"External server {s.url}\n{json.dumps(self.props.get(s.id, {}), indent=2)[:1500]}"
        else:
            text = "\n".join(self.log_tail(s, 60))
        if text != self.detail_text:
            self.detail_text = text
            log = self.query_one("#detail", RichLog)
            log.clear()
            log.write(text)

    def action_launch_server(self) -> None:
        model = self.selected_model()
        if model is None:
            return self.say("Select a model on the Models tab first.")
        self.say(f"Launching {Path(model).name}...")
        self.launch_worker(model)

    @work(thread=True, group="launch")
    def launch_worker(self, model: str) -> None:
        """Spawn the server off the UI thread; poll_health then tracks loading and readiness."""
        s = self.store.launch_server(self.cfg, model)
        self.pending[s.id] = time.time()
        self.health[s.id] = "starting 0s"

        def show() -> None:
            self.render_sessions()
            self.query_one(TabbedContent).active = "sessions"
            self.sess_table.move_cursor(row=self.sess_table.row_count - 1)

        self.call_from_thread(show)
        self.call_from_thread(self.notify, f"Loading {s.name} on port {s.url.rsplit(':', 1)[1]}...", title="Launching")

    def action_launch_cli(self) -> None:
        model = self.selected_model()
        if model is None:
            return self.say("Select a model on the Models tab first.")
        cmd = build_command(self.cfg, model, self.cfg.profile_for(model), "cli", 0)
        with self.suspend():
            subprocess.run(cmd)

    def _open_editor(self, model: str | None, info) -> None:
        """Push the profile editor; on save persist the profile and refresh."""
        prof = self.cfg.profile_for(model) if model else self.cfg.default_profile
        title = Path(model).name if model else "Default profile"

        def done(new: Profile | None) -> None:
            if new is None:
                return
            if model:
                self.cfg.profiles[model] = new
            else:
                self.cfg.default_profile = new
            self.cfg.save()
            self.action_refresh()

        self.push_screen(ProfileEditor(self.cfg, prof, self.host, info, title))

    def action_edit_profile(self) -> None:
        model = self.selected_model()
        if model is None:
            return self.say("Select a model first.")
        self.say("Reading model header...")
        self.load_info_worker(model)

    @work(thread=True, group="info")
    def load_info_worker(self, model: str) -> None:
        """Parse (or read the cache for) the model's GGUF header off the UI thread."""
        info = self.info_cache.get(Path(model))
        if info is None:
            try:
                info = read_model_info(Path(model))
                self.info_cache.put(info)
            except (GGUFError, OSError) as e:
                return self.call_from_thread(self.say, f"Could not read GGUF header: {e}")
        self.call_from_thread(self._open_editor, model, info)

    def action_edit_default(self) -> None:
        self._open_editor(None, None)

    def action_edit_settings(self) -> None:
        fields = [
            ("bin_dir", "llama.cpp bin dir", self.cfg.bin_dir,
             "Folder holding llama-server and llama-cli (your build/bin)."),
            ("model_dirs", "Model dirs", ",".join(self.cfg.model_dirs),
             "Comma-separated folders scanned (recursively) for .gguf models. Press r to rescan."),
        ]

        def done(res: dict | None) -> None:
            if res:
                self.cfg.bin_dir = res["bin_dir"]
                self.cfg.model_dirs = [d.strip() for d in res["model_dirs"].split(",") if d.strip()]
                self.cfg.save()
                self.action_refresh()

        self.push_screen(FormScreen("Settings", fields), done)

    def action_attach(self) -> None:
        def done(res: dict | None) -> None:
            if res and res["url"]:
                s = self.store.attach(res["url"])
                self.query_one(TabbedContent).active = "sessions"
                self.render_sessions()
                self.say(f"Attached to {s.url}")

        self.push_screen(FormScreen("Attach to running server", [("url", "URL or host:port", "localhost:8080", "Address of a llama-server that is already running. It is only tracked, never stopped by this tool.")]), done)

    def action_hf(self) -> None:
        self.push_screen(HFScreen(self.cfg), lambda _: self.action_refresh())

    def action_chat(self) -> None:
        s = self.selected_session()
        if s is None:
            return self.say("No session selected (Sessions tab).")
        self.push_screen(ChatScreen(s))

    def action_stop(self) -> None:
        s = self.selected_session()
        if s is None:
            return
        self.store.stop(s)
        self.render_sessions()
        self.say(f"Stopped/forgot {s.name}")


def main() -> None:
    """Entry point."""
    LlamaCtl().run()


if __name__ == "__main__":
    main()
