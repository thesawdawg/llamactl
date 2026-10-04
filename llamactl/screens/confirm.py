"""ConfirmScreen: yes/no confirmation modal."""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label


class ConfirmScreen(ModalScreen[bool]):
    """Modal asking a yes/no question. Dismisses True/False.

    Args:
        prompt: The question shown to the user.
    """

    BINDINGS = [
        Binding("y", "yes", "Yes", show=False),
        Binding("n", "no", "No", show=False),
        Binding("escape", "no", "Cancel"),
    ]
    CSS = """
    ConfirmScreen { align: center middle; }
    ConfirmScreen #box { width: auto; max-width: 80; height: auto; padding: 1 2;
        border: round $accent; background: $surface; }
    ConfirmScreen #cbuttons { height: auto; margin-top: 1; }
    ConfirmScreen Button { margin-right: 1; }
    """

    def __init__(self, prompt: str) -> None:
        super().__init__()
        self.prompt = prompt

    def compose(self) -> ComposeResult:
        with Vertical(id="box"):
            yield Label(self.prompt)
            with Horizontal(id="cbuttons"):
                yield Button("Yes (y)", id="yes", variant="error")
                yield Button("No (n)", id="no")

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#yes")
    def _yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#no")
    def _no(self) -> None:
        self.dismiss(False)
