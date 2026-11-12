"""WslRamScreen: asks once how much RAM the Windows host has."""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label

from ..hardware import GiB, Host
from ..widgets.buttonbar import ButtonBar


class WslRamScreen(ModalScreen[int | None]):
    """Modal asking for the Windows machine's RAM so WSL advice can be computed.

    Args:
        host: Host snapshot (what the WSL VM currently sees).

    Returns (via dismiss): the Windows RAM in bytes, 0 when skipped (Esc = Skip).
    """

    BINDINGS = [Binding("escape", "skip", "Skip")]
    CSS = """
    WslRamScreen { align: center middle; }
    WslRamScreen #wslbox { width: 90%; max-width: 80; height: auto; padding: 1 2;
        border: round $accent; background: $surface; }
    WslRamScreen .hint { color: $text-muted; }
    WslRamScreen #wslbuttons { margin-top: 1; }
    """

    def __init__(self, host: Host) -> None:
        super().__init__()
        self.host = host

    def compose(self) -> ComposeResult:
        with Vertical(id="wslbox"):
            yield Label(
                "llamactl runs inside WSL, which can use "
                f"{self.host.ram_total / GiB:.1f} GiB of RAM and "
                f"{self.host.cpu_threads} threads. How much RAM does Windows have?")
            yield Input(placeholder="whole GiB, for example 32", id="value")
            yield Label("Windows reports GiB: 32 means 32 GiB "
                        "(Task Manager > Performance > Memory)", classes="hint")
            yield Label("", id="wsl_err")
            with ButtonBar(id="wslbuttons"):
                yield Button("Save", id="save", variant="primary")
                yield Button("Skip", id="skip")

    def on_mount(self) -> None:
        """Focus the input."""
        self.query_one("#value", Input).focus()

    def _save(self) -> None:
        """Dismiss with the Windows RAM in bytes, or flag the bad input."""
        raw = self.query_one("#value", Input).value.strip()
        try:
            gib = int(raw)
        except ValueError:
            gib = 0
        if gib < 1:
            self.query_one("#wsl_err", Label).update("enter a whole number of GiB, at least 1")
            return
        self.dismiss(gib * GiB)

    def action_skip(self) -> None:
        """Esc or Skip: do not ask again (stored as 0)."""
        self.dismiss(0)

    @on(Button.Pressed, "#save")
    def _save_btn(self) -> None:
        self._save()

    @on(Button.Pressed, "#skip")
    def _skip_btn(self) -> None:
        self.dismiss(0)

    @on(Input.Submitted)
    def _submitted(self) -> None:
        self._save()
