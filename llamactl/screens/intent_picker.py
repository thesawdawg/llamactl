"""IntentPicker: choose what a model is for, one card per applicable intent."""
from __future__ import annotations

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option

from ..fmt import FITS_STYLE
from ..modelrows import IntentRow

GB = 1e9


def preset_note(row: IntentRow) -> str:
    """Words for a row's preset: ``tuned, active, 41 tok/s`` or ``none yet``."""
    if not row.has_preset:
        return "none yet: an estimate is made when you pick this"
    bits = [row.source + (" (stale)" if row.stale else "")]
    if row.active:
        bits.append("active")
    if row.tps:
        bits.append(f"{row.tps:.0f} tok/s")
    return ", ".join(bits)


def card(row: IntentRow) -> Text:
    """One intent as a card: label and verdict, numbers, why, warnings, preset status.

    Args:
        row: The intent's plan summary.

    Returns:
        Multi-line styled text.
    """
    t = Text()
    t.append(row.label, style="bold")
    t.append(f"  {row.description}  ", style="dim")
    t.append(row.verdict, style=FITS_STYLE.get(row.verdict, ""))
    t.append(f"\n  {row.ctx_per_slot} ctx x {row.slots} slot{'s' * (row.slots != 1)}, "
             f"{row.kv} KV, needs {row.needs / GB:.1f} of {row.usable / GB:.1f} GB usable VRAM")
    if row.reason:
        t.append(f"\n  why: {row.reason.split('; needs')[0]}", style="dim")
    for w in row.warnings:
        t.append(f"\n  ! {w}", style="yellow")
    t.append(f"\n  preset: {preset_note(row)}\n")
    return t


class IntentPicker(ModalScreen[str | None]):
    """Modal listing the applicable intents; dismisses with the chosen intent key or None.

    Args:
        model: Model name shown in the heading.
        rows: One row per applicable intent (`modelrows.intent_rows`).
        current: Intent key to highlight first (usually the active preset's).
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CSS = """
    IntentPicker { align: center middle; }
    IntentPicker #ibox { width: 90%; max-width: 110; height: 90%; padding: 1 2;
        border: round $accent; background: $surface; }
    IntentPicker OptionList { height: 1fr; border: none; padding: 0; }
    IntentPicker .hint { color: $text-muted; }
    """

    def __init__(self, model: str, rows: list[IntentRow], current: str = "") -> None:
        super().__init__()
        self.model, self.rows, self.current = model, rows, current

    def compose(self) -> ComposeResult:
        with Vertical(id="ibox"):
            yield Label(f"[b]What is {self.model} for?[/b]", markup=True)
            yield Label("Enter picks it, Esc cancels. A preset is planned for the pick.",
                        classes="hint")
            yield OptionList(*[Option(card(r), id=r.key) for r in self.rows], id="intents")

    def on_mount(self) -> None:
        lst = self.query_one("#intents", OptionList)
        keys = [r.key for r in self.rows]
        lst.highlighted = keys.index(self.current) if self.current in keys else 0
        lst.focus()

    @on(OptionList.OptionSelected)
    def _picked(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        """Close without choosing."""
        self.dismiss(None)
