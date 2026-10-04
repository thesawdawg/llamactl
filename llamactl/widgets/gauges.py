"""ResourceGauges: one-line VRAM/RAM meters for the header area."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Label, ProgressBar

from ..hostinfo import Host


class ResourceGauges(Widget):
    """VRAM and RAM usage bars; yellow over 80%, red over 95%."""

    DEFAULT_CSS = """
    ResourceGauges { height: 1; layout: horizontal; }
    ResourceGauges Label { width: auto; padding: 0 1; }
    ResourceGauges ProgressBar { width: 1fr; height: 1; padding-top: 0; }
    ResourceGauges ProgressBar.warn Bar > .bar--bar { color: $warning; }
    ResourceGauges ProgressBar.danger Bar > .bar--bar { color: $error; }
    """

    def compose(self) -> ComposeResult:
        yield Label("VRAM", id="g_vram_label")
        yield ProgressBar(id="g_vram", total=100, show_eta=False, show_percentage=False)
        yield Label("RAM", id="g_ram_label")
        yield ProgressBar(id="g_ram", total=100, show_eta=False, show_percentage=False)

    def update_host(self, host: Host) -> None:
        """Refresh both bars from a host snapshot.

        Args:
            host: Host resource snapshot.
        """
        for name, total, used in (("vram", host.vram_total, host.vram_total - host.vram_free),
                                  ("ram", host.ram_total, host.ram_total - host.ram_free)):
            bar = self.query_one(f"#g_{name}", ProgressBar)
            lbl = self.query_one(f"#g_{name}_label", Label)
            if not total:
                bar.display = lbl.display = False
                continue
            bar.display = lbl.display = True
            frac = used / total
            lbl.update(f"{name.upper()} {used / 1e9:.1f}/{total / 1e9:.1f} G")
            bar.update(total=total, progress=used)
            bar.set_class(frac > 0.95, "danger")
            bar.set_class(0.8 < frac <= 0.95, "warn")
