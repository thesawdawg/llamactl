"""HelpPanel: modal showing a setting's long help text."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Label

from ..schema import Setting


class HelpPanel(ModalScreen[None]):
    """Modal with label, flag, help text, default and allowed range/choices.

    Args:
        setting: The setting to describe.
    """

    BINDINGS = [Binding("escape", "close", "Close")]
    CSS = """
    HelpPanel { align: center middle; }
    HelpPanel #help { width: 80%; max-width: 100; height: auto; max-height: 90%;
        border: round $accent; padding: 1 2; background: $surface; }
    HelpPanel .meta { color: $text-muted; }
    """

    def __init__(self, setting: Setting) -> None:
        super().__init__()
        self.setting = setting

    def compose(self) -> ComposeResult:
        s = self.setting
        bits = [f"default: {s.default}"]
        if s.choices:
            bits.append(f"choices: {', '.join(s.choices)}")
        if s.minimum is not None or s.maximum is not None:
            bits.append(f"range: {s.minimum:g}..{s.maximum:g}" if s.minimum is not None
                        and s.maximum is not None else f"min: {s.minimum or s.maximum:g}")
        with VerticalScroll(id="help"):
            yield Label(f"[b]{s.label}[/b]  {s.flag or '(no flag)'}")
            yield Label("  ".join(bits), classes="meta")
            yield Label(s.help)
            yield Label("Esc to close", classes="meta")

    def action_close(self) -> None:
        self.dismiss(None)
