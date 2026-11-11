"""AfterDownload: what to do with a model that was just prepared (serve it, tune it, or nothing yet)."""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label

SERVE, TUNE, DONE = "serve", "tune", "done"


class AfterDownload(ModalScreen[str]):
    """Offer Serve, Tune or Done for a model that now has a preset; dismisses with the choice.

    Escape counts as Done.

    Args:
        title: What happened, e.g. ``qwen3-8b is ready with an agent preset (estimated).``
    """

    BINDINGS = [
        Binding("s", "serve", "Serve", show=False),
        Binding("t", "tune", "Tune", show=False),
        Binding("escape", "done", "Done"),
    ]
    CSS = """
    AfterDownload { align: center middle; }
    AfterDownload #abox { width: auto; max-width: 90; height: auto; padding: 1 2;
        border: round $accent; background: $surface; }
    AfterDownload .hint { color: $text-muted; }
    AfterDownload #abuttons { height: auto; margin-top: 1; }
    AfterDownload Button { margin-right: 1; }
    """

    def __init__(self, title: str) -> None:
        super().__init__()
        self.title_text = title

    def compose(self) -> ComposeResult:
        """The message and three buttons."""
        with Vertical(id="abox"):
            yield Label(self.title_text)
            yield Label("Serve loads it in the gateway; Tune measures the best settings.",
                        classes="hint")
            with Horizontal(id="abuttons"):
                yield Button("Serve (s)", id="serve", variant="primary")
                yield Button("Tune (t)", id="tune")
                yield Button("Done (Esc)", id="done")

    def action_serve(self) -> None:
        """Serve the model."""
        self.dismiss(SERVE)

    def action_tune(self) -> None:
        """Tune the model."""
        self.dismiss(TUNE)

    def action_done(self) -> None:
        """Do nothing more."""
        self.dismiss(DONE)

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        """Route a button to its action."""
        self.dismiss(str(event.button.id))
