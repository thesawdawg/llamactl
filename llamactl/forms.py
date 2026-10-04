"""Reusable modal form."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Label


class FormScreen(ModalScreen[dict | None]):
    """Generic modal form: a list of (key, label, value, hint) text fields."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CSS = """
    FormScreen { align: center middle; }
    #form { width: 95%; max-width: 160; height: auto; max-height: 95%; border: round $accent; padding: 1 2; background: $surface; }
    #fields { layout: grid; grid-size: 2; grid-gutter: 0 3; height: auto; }
    .field { height: auto; }
    .field.wide { column-span: 2; }
    .field Label { margin-top: 1; }
    .field .hint { width: 100%; height: auto; margin-top: 0; color: $text-muted; text-style: italic; }
    """

    def __init__(self, title: str, fields: list[tuple[str, str, str, str]]) -> None:
        super().__init__()
        self.title_text, self.fields = title, fields

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="form"):
            yield Label(f"[b]{self.title_text}[/b]  (Enter on last field = save, Esc = cancel)")
            with Container(id="fields"):
                for i, (key, label, value, hint) in enumerate(self.fields):
                    wide = " wide" if i == len(self.fields) - 1 and i % 2 == 0 else ""
                    with Vertical(classes=f"field{wide}"):
                        yield Label(label)
                        yield Input(value, id=f"f_{key}", password=key.endswith("token"))
                        yield Label(hint, classes="hint")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        inputs = list(self.query(Input))
        if event.input is inputs[-1]:
            self.dismiss({k: self.query_one(f"#f_{k}", Input).value.strip() for k, *_ in self.fields})
        else:
            inputs[inputs.index(event.input) + 1].focus()

    def action_cancel(self) -> None:
        self.dismiss(None)
