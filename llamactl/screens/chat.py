"""Streaming chat against a server's OpenAI-compatible endpoint."""
from __future__ import annotations

import json

import httpx
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, RichLog

from ..sessions import Session


class ChatScreen(Screen):
    """Streaming chat against a server's OpenAI-compatible endpoint.

    Args:
        session: The session to chat with.
    """

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
