"""BudgetPanel: stacked memory bar + VRAM/RAM usage bars + verdict line."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Label, ProgressBar

from ..hardware import Host
from ..planner import Budget, verdict

_PARTS = (("weights", "red"), ("KV", "blue"), ("compute", "green"), ("overhead", "yellow"))
# "experts (RAM)" replaces part of the weights segment on a MoE expert split
_PARTS_MOE = (("weights", "red"), ("experts (RAM)", "magenta"), ("KV", "blue"),
              ("compute", "green"), ("overhead", "yellow"))
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
    BudgetPanel { height: auto; border: round $primary; padding: 0 1;
        border-title-color: $accent; border-title-style: bold; }
    BudgetPanel:focus-within { border: round $accent; }
    BudgetPanel .bar { height: 1; }
    BudgetPanel .nums { color: $text-muted; width: 100%; height: auto; }
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

    def update_budget(self, budget: Budget, host: Host, headroom_pct: float = 10,
                      expert_ram: int = 0, verdict_text: str = "") -> None:
        """Redraw all parts for a budget on a host.

        Args:
            budget: Memory budget to display.
            host: Host resource snapshot.
            headroom_pct: Fraction of total VRAM kept free for the verdict.
            expert_ram: Bytes of expert tensors kept in RAM by a MoE plan; shown as their
                own segment, off the VRAM bar and on the RAM bar (0 = unchanged behaviour).
            verdict_text: Text after "Verdict:" (`verdict_label`, e.g. "GPU+experts");
                computed from the budget when empty.
        """
        total = budget.total() or 1
        if expert_ram:
            parts = [budget.weights - expert_ram, expert_ram,
                     budget.kv, budget.compute, budget.overhead]
            parts_def = _PARTS_MOE
        else:
            parts = [budget.weights, budget.kv, budget.compute, budget.overhead]
            parts_def = _PARTS
        cells = [max(1, round(_BAR_W * p / total)) for p in parts]
        markup = "".join(f"[on {c}]{' ' * n}[/]" for (_, c), n in zip(parts_def, cells))
        self.query_one("#stacked", Label).update(markup)
        self.query_one(".nums", Label).update(
            "  ".join(f"{name} {_gb(p)}" for (name, _), p in zip(parts_def, parts))
            + f"  = {_gb(total)}")

        gpu_part = total - expert_ram
        vram_bar = self.query_one("#vram_bar", ProgressBar)
        vram_bar.display = bool(host.vram_total)
        if host.vram_total:
            vram_bar.update(total=host.vram_total, progress=min(gpu_part, host.vram_total))
            vram_bar.set_class(gpu_part > host.vram_total, "over")
        ram_part = expert_ram or total
        self.query_one("#ram_bar", ProgressBar).update(
            total=host.ram_total or 1, progress=min(ram_part, host.ram_total or 1))
        self.query_one(".verdict", Label).update(
            f"Verdict: {verdict_text or verdict(budget, host, headroom_pct)}")
