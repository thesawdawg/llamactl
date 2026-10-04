"""LogScreen: full session log in a scrollable modal."""
from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, TextArea


class LogScreen(ModalScreen[None]):
    """Modal showing the entire log file of a session.

    Args:
        path: Log file to display.
        title: Heading line (session name).
    """

    BINDINGS = [Binding("escape", "close", "Close")]
    CSS = """
    LogScreen { align: center middle; }
    LogScreen #logbox { width: 95%; height: 90%; border: round $accent;
        background: $surface; }
    LogScreen TextArea { height: 1fr; }
    """

    def __init__(self, path: str, title: str) -> None:
        super().__init__()
        self.path, self.title = path, title

    def compose(self) -> ComposeResult:
        try:
            text = Path(self.path).read_text(errors="replace")
        except OSError as e:
            text = f"could not read {self.path}: {e}"
        with Vertical(id="logbox"):
            yield Label(f"[b]{self.title}[/b]  (Esc to close)")
            yield TextArea(text, read_only=True, show_cursor=False)

    def action_close(self) -> None:
        self.dismiss(None)
