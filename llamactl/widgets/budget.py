"""BudgetPanel: stacked memory bar + VRAM/RAM usage bars + verdict line."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Label, ProgressBar

from ..hostinfo import Host
from ..recommend import Budget, verdict

_PARTS = (("weights", "red"), ("KV", "blue"), ("compute", "green"), ("overhead", "yellow"))
_BAR_W = 40


def _gb(n: float) -> str:
    return f"{n / 1e9:.2f}G"


class BudgetPanel(Vertical):
    """Shows a Budget against the host's VRAM and RAM.

    A single stacked bar (weights / KV / compute / overhead, proportional
    widths) with the GB numbers, two ProgressBars for VRAM and RAM fill,
    and a GPU / GPU+CPU / CPU / NO verdict line.
    """

    DEFAULT_CSS = """
    BudgetPanel { height: auto; border: round $primary; padding: 0 1; }
    BudgetPanel .bar { height: 1; }
    BudgetPanel .nums { color: $text-muted; }
    BudgetPanel .verdict { text-style: bold; }
    BudgetPanel ProgressBar { height: 1; margin-bottom: 0; }
    BudgetPanel ProgressBar.over Bar > .bar--bar { color: $error; }
    """

    def compose(self) -> ComposeResult:
        yield Label("", id="stacked", classes="bar")
        yield Label("", classes="nums")
        yield Label("VRAM")
        yield ProgressBar(id="vram_bar", total=100, show_eta=False)
        yield Label("RAM")
        yield ProgressBar(id="ram_bar", total=100, show_eta=False)
        yield Label("", classes="verdict")

    def update_budget(self, budget: Budget, host: Host, headroom_pct: float = 10) -> None:
        """Redraw all parts for a budget on a host.

        Args:
            budget: Memory budget to display.
            host: Host resource snapshot.
            headroom_pct: Fraction of total VRAM kept free for the verdict.
        """
        total = budget.total() or 1
        parts = [budget.weights, budget.kv, budget.compute, budget.overhead]
        cells = [max(1, round(_BAR_W * p / total)) for p in parts]
        markup = "".join(f"[on {c}]{' ' * n}[/]" for (_, c), n in zip(_PARTS, cells))
        self.query_one("#stacked", Label).update(markup)
        self.query_one(".nums", Label).update(
            "  ".join(f"{name} {_gb(p)}" for (name, _), p in zip(_PARTS, parts))
            + f"  = {_gb(total)}")

        vram_bar = self.query_one("#vram_bar", ProgressBar)
        vram_bar.display = bool(host.vram_total)
        if host.vram_total:
            vram_bar.update(total=host.vram_total, progress=min(total, host.vram_total))
            vram_bar.set_class(total > host.vram_total, "over")
        self.query_one("#ram_bar", ProgressBar).update(
            total=host.ram_total or 1, progress=min(total, host.ram_total or 1))
        self.query_one(".verdict", Label).update(
            f"Verdict: {verdict(budget, host, headroom_pct)}")
