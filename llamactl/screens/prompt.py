"""PromptScreen: single-line text input modal."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label


class PromptScreen(ModalScreen[str | None]):
    """Modal asking for one line of text. Dismisses the text or None on cancel.

    Args:
        title: Heading shown above the input.
        hint: Explanation shown under the input.
        default: Initial input value.
        password: Mask the input (tokens).
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CSS = """
    PromptScreen { align: center middle; }
    PromptScreen #pbox { width: 80%; max-width: 100; height: auto; padding: 1 2;
        border: round $accent; background: $surface; }
    PromptScreen .hint { color: $text-muted; text-style: italic; }
    """

    def __init__(self, title: str, hint: str = "", default: str = "",
                 password: bool = False) -> None:
        super().__init__()
        self.title_text, self.hint, self.default, self.password = title, hint, default, password

    def compose(self) -> ComposeResult:
        with Vertical(id="pbox"):
            yield Label(f"[b]{self.title_text}[/b]  (Enter = save, Esc = cancel)")
            yield Input(self.default, id="value", password=self.password)
            yield Label(self.hint, classes="hint")

    def on_mount(self) -> None:
        self.query_one("#value", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())

    def action_cancel(self) -> None:
        self.dismiss(None)
