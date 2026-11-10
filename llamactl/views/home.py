"""HomeView: hardware, the gateway, what is loaded now, and the next steps."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import DataTable, OptionList, Static
from textual.widgets.option_list import Option

from .. import actions
from ..calibration import Calibration
from ..config import Config
from ..hardware import Host
from ..screens.confirm import ConfirmScreen
from ..serving import (GW_RUNNING, GW_STARTING, GW_STOPPED, GW_STOPPING, GW_UNHEALTHY,
                       ServingSnapshot, gateway_state)
from ..suggest import Suggestion, binary_found, next_steps
from .base import BaseView

BAR_WIDTH = 20
STATE_STYLE = {GW_RUNNING: "green", GW_STARTING: "yellow", GW_STOPPING: "yellow",
               GW_UNHEALTHY: "red", GW_STOPPED: "dim"}
MODEL_STYLE = {"loaded": "green", "loading": "yellow", "sleeping": "cyan", "failed": "red",
               "unloaded": "dim", "ready": "green", "down": "red", "crashed": "red"}


def usage_bar(used: float, total: float, width: int = BAR_WIDTH) -> Text:
    """A text bar for `used` of `total`: green, yellow over 80 %, red over 95 %.

    Args:
        used: Amount in use.
        total: Capacity (0 gives an empty bar).
        width: Bar width in cells.

    Returns:
        The bar as styled text.
    """
    frac = min(1.0, used / total) if total else 0.0
    full = round(frac * width)
    style = "red" if frac > 0.95 else "yellow" if frac > 0.8 else "green"
    return Text.assemble(("█" * full, style), ("░" * (width - full), "dim"))


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f}"




def _usage_line(label: str, used: float, total: float, note: str = "") -> Text:
    """``CUDA0   ████░░░░░░░░ 5.2 / 16.0 GB`` plus an optional dim note."""
    t = Text(f"{label:<8}")
    t.append_text(usage_bar(used, total))
    t.append(f"  {_gb(used)} / {_gb(total)} GB")
    if note:
        t.append(f"  {note}", style="dim")
    return t


def hardware_text(host: Host, calibration: Calibration, bin_dir: str, found: bool,
                  build: str) -> Text:
    """The hardware card: devices with usage bars, RAM, CPU, calibration and llama.cpp.

    Args:
        host: Host snapshot.
        calibration: Measured trials (the estimate factor per backend).
        bin_dir: Configured llama.cpp folder.
        found: ``llama-server`` exists there.
        build: llama-server version string, "" when unknown.

    Returns:
        The card body.
    """
    if not host.ram_total and not host.gpus:
        return Text("detecting hardware...", style="dim")
    lines: list[Text] = []
    for d in host.gpus:
        lines.append(_usage_line(d.id, d.total - d.free, d.total))
        lines.append(Text(f"{'':<8}{d.name}", style="dim"))
    if not host.gpus:
        lines.append(Text("no GPU found: models run on the CPU", style="yellow"))
    lines.append(_usage_line("RAM", host.ram_total - host.ram_free, host.ram_total,
                             "shared with the GPU" if host.unified else ""))
    lines.append(Text(f"{'CPU':<8}{host.cpu_name or '?'}"))
    lines.append(Text(f"{'':<8}{host.cpu_threads} threads, {host.cpu_cores} cores", style="dim"))
    factors = []
    for backend in dict.fromkeys(g.backend for g in host.gpus):
        n = sum(1 for t in calibration.trials if t.backend == backend and t.ratio() is not None)
        factors.append(f"{backend} x{calibration.factor(backend):.2f} ({n} trial{'s' * (n != 1)})"
                       if n else f"{backend} not calibrated yet")
    if factors:
        lines.append(Text(f"{'Fit est':<8}" + ", ".join(factors)))
    lines.append(Text(f"{'Build':<8}" + (build or ("unknown" if found else "llama-server not found")),
                      style="" if found else "red"))
    lines.append(Text(f"{'':<8}{bin_dir}", style="dim"))
    return Text("\n").join(lines)


def gateway_lines(cfg: Config, state: str, url: str, problems: list[str]) -> Text:
    """The gateway card body (without the curl snippet).

    Args:
        cfg: Config holding the gateway settings.
        state: "stopped", "starting", "running", "unhealthy" or "stopping".
        url: Base URL (``http://host:port``).
        problems: `Gateway.problems`: presets that could not be written.

    Returns:
        The card body.
    """
    g = cfg.gateway
    idle = f"unload a model after {g.sleep_idle} s unused" if g.sleep_idle > 0 else "models stay loaded"
    t = Text()
    t.append(f"{'State':<10}")
    t.append(state, style=STATE_STYLE.get(state, ""))
    t.append(f"\n{'URL':<10}{url}/v1")
    t.append(f"\n{'Models':<10}{g.models_max} loaded at a time")
    t.append(f"\n{'Idle':<10}{idle}")
    t.append(f"\n{'API key':<10}{'required' if g.api_key else 'none'}")
    if state == GW_STOPPED:
        t.append("\n\nStopped. Press u to start it; models then load on first use.", style="dim")
    elif state == GW_UNHEALTHY:
        t.append("\n\nThe process runs but does not answer /health; see its log in Serving (3).",
                 style="red")
    for p in problems:
        t.append(f"\n! {p}", style="yellow")
    return t


class HomeView(BaseView):
    """Status at a glance, and what to do next."""

    BINDINGS = [
        Binding("u", "start_gateway", "Start gateway"),
        Binding("d", "stop_gateway", "Stop gateway"),
        Binding("y", "copy_url", "Copy URL"),
        Binding("c", "copy_snippet", "Copy curl"),
    ]
    DEFAULT_CSS = """
    HomeView { height: 1fr; }
    #home_row { height: 1fr; }
    #home_left { width: 1fr; height: 1fr; }
    #home_right { width: 1fr; height: 1fr; }
    #hw_card { height: auto; border: round $primary; padding: 0 1; border-title-color: $text; }
    #gw_card { height: 1fr; border: round $primary; padding: 0 1; border-title-color: $text; }
    #gw_snippet { margin-top: 1; color: $text-muted; }
    #loaded_box { height: auto; max-height: 12; border: round $primary; padding: 0 1; }
    #loaded { height: auto; max-height: 9; }
    #loaded_empty, #steps_empty { color: $text-muted; padding: 0 1; }
    #steps_box { height: 1fr; border: round $primary; padding: 0 1; }
    #steps { height: 1fr; border: none; padding: 0; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._steps: list[Suggestion] = []

    def compose(self) -> ComposeResult:
        """Hardware and gateway cards on the left, loaded models and next steps on the right."""
        with Horizontal(id="home_row"):
            with Vertical(id="home_left"):
                yield Static("", id="hw_card")
                with VerticalScroll(id="gw_card"):
                    yield Static("", id="gw_info")
                    yield Static("", id="gw_snippet")
            with Vertical(id="home_right"):
                with Vertical(id="loaded_box"):
                    yield DataTable(id="loaded", cursor_type="row")
                    yield Static("Nothing is loaded. Press Enter on a model in Models (2); the "
                                 "gateway loads it on first use.", id="loaded_empty")
                with Vertical(id="steps_box"):
                    yield OptionList(id="steps")
                    yield Static("Nothing to do: everything is set up.", id="steps_empty")

    def on_mount(self) -> None:
        """Set the card titles, build the loaded table and draw everything."""
        self.query_one("#hw_card").border_title = "Hardware"
        self.query_one("#gw_card").border_title = "Gateway"
        self.query_one("#loaded_box").border_title = "Loaded now"
        self.query_one("#steps_box").border_title = "Next steps (Enter)"
        t = self.query_one("#loaded", DataTable)
        t.add_column("Model", key="name")
        t.add_column("State", key="state")
        t.add_column("Kind", key="kind")
        self.render_all()

    # -- hooks ------------------------------------------------------------------------

    def focus_primary(self) -> None:
        """Focus the next steps when there are any, else the loaded list."""
        steps = self.query_one("#steps", OptionList)
        (steps if self._steps else self.query_one("#loaded", DataTable)).focus()

    def on_host(self, host: Host) -> None:
        """Redraw the hardware card."""
        self.render_hardware()

    def on_scan_done(self) -> None:
        """Models are known: recompute the next steps."""
        self.render_steps()
        self.render_gateway()

    def on_models(self, models: list[Path]) -> None:
        """The model list changed: recompute the next steps."""
        self.render_steps()

    def on_serving(self, snapshot: ServingSnapshot) -> None:
        """Redraw the gateway card, the loaded list and the next steps."""
        self.render_gateway()
        self.render_loaded()
        self.render_steps()

    def on_presets_changed(self, paths: Iterable[str]) -> None:
        """Presets or the build changed: redraw cards and next steps."""
        self.render_hardware()
        self.render_steps()
        self.render_gateway()

    def on_config_changed(self) -> None:
        """Settings changed: redraw everything."""
        self.render_all()

    def render_all(self) -> None:
        """Redraw every card."""
        self.render_hardware()
        self.render_gateway()
        self.render_loaded()
        self.render_steps()

    # -- cards ------------------------------------------------------------------------

    def render_hardware(self) -> None:
        """Hardware card."""
        a = self.app
        self.query_one("#hw_card", Static).update(
            hardware_text(a.host, a.calibration, a.cfg.bin_dir, binary_found(a.cfg), a.build))

    def gateway_url(self) -> str:
        """Base URL of the gateway (running or planned)."""
        snap = self.app.snapshot
        return snap.status.url if snap else self.app.gateway.listen_url

    def render_gateway(self) -> None:
        """Gateway card and the curl snippet."""
        a = self.app
        state = a.gateway_state()
        self.query_one("#gw_info", Static).update(
            gateway_lines(a.cfg, state, self.gateway_url(), list(a.gateway.problems)))
        snippet = self.snippet()
        self.query_one("#gw_snippet", Static).update(
            Text.assemble(("Try it", "bold"), (f"  ({self.snippet_name()}; c copies)\n", "dim"),
                          snippet) if snippet else "")

    def served_names(self) -> list[str]:
        """Model ids to build a snippet for: loaded ones first, then configured ones."""
        snap = self.app.snapshot
        names = [m.name for m in snap.running_models] if snap else []
        if not names:
            for ent in self.app.cfg.models.values():
                if ent.presets and ent.alias:
                    names.append(ent.alias)
        return names

    def snippet_name(self) -> str:
        """The model the snippet is for: the highlighted loaded one, else the first known."""
        t = self.query_one("#loaded", DataTable)
        if t.row_count:
            key = list(t.rows)[min(t.cursor_row, t.row_count - 1)].value
            if key and str(key).startswith("gw:"):
                return str(key)[3:]
        names = self.served_names()
        return names[0] if names else ""

    def snippet(self) -> str:
        """The curl line for `snippet_name`, "" when there is no model to talk to."""
        name = self.snippet_name()
        if not name:
            return ""
        alias = name.split("@", 1)[0]
        embedding = any(ent.alias == alias and p in self.app.infos
                        and self.app.infos[p].is_embedding
                        for p, ent in self.app.cfg.models.items())
        return actions.curl_snippet(self.gateway_url(), name, api_key=bool(self.app.cfg.gateway.api_key),
                                    embedding=embedding)

    def render_loaded(self) -> None:
        """"Loaded now": loaded/loading/sleeping gateway models plus standalone servers."""
        snap = self.app.snapshot
        rows: dict[str, tuple[str, str, str]] = {}
        if snap is not None:
            for m in snap.running_models:
                rows[f"gw:{m.name}"] = (m.name, m.state, "gateway")
            for s in snap.sessions:
                rows[f"s:{s.id}"] = (s.name, snap.health.get(s.id, "down"),
                                     "attached" if s.external else "standalone")
        t = self.query_one("#loaded", DataTable)
        for key in [k for k in t.rows if k.value not in rows]:
            t.remove_row(key)
        for key, (name, state, kind) in rows.items():
            cells = (name, Text(state, style=MODEL_STYLE.get(state, "")), kind)
            if key in t.rows:
                for col, v in zip(("name", "state", "kind"), cells):
                    t.update_cell(key, col, v)
            else:
                t.add_row(*cells, key=key)
        t.display = bool(rows)
        self.query_one("#loaded_empty").display = not rows

    def render_steps(self) -> None:
        """"Next steps", recomputed from the current state (kept as is when unchanged)."""
        a = self.app
        lst = self.query_one("#steps", OptionList)
        if a.scanning and not a.models:
            steps: list[Suggestion] = []
            self.query_one("#steps_empty", Static).update("Scanning models...")
        else:
            steps = next_steps(a.cfg, a.models, a.infos, a.calibration,
                               a.snapshot.status if a.snapshot else None, a.build)
            self.query_one("#steps_empty", Static).update("Nothing to do: everything is set up.")
        if [(s.key, s.text) for s in steps] != [(s.key, s.text) for s in self._steps]:
            self._steps = steps
            keep = lst.highlighted
            lst.clear_options()
            lst.add_options([Option(s.text, id=s.key) for s in steps])
            if steps:
                lst.highlighted = min(keep or 0, len(steps) - 1)
        lst.display = bool(steps)
        self.query_one("#steps_empty").display = not steps
        if self.display and self.app.screen.focused is None:
            self.focus_primary()            # the first render may come after the view was shown

    # -- actions ----------------------------------------------------------------------

    @on(OptionList.OptionSelected, "#steps")
    def _step_chosen(self, event: OptionList.OptionSelected) -> None:
        """Enter on a next step: do what it says."""
        step = next((s for s in self._steps if s.key == event.option.id), None)
        if step is not None:
            self.app.run_suggestion(step)

    @on(DataTable.RowHighlighted, "#loaded")
    def _loaded_highlighted(self) -> None:
        """The highlighted loaded model changed: rebuild the snippet."""
        self.render_gateway()

    def action_start_gateway(self) -> None:
        """Start the gateway."""
        state = self.app.gateway_state()
        if state in (GW_RUNNING, GW_STARTING):
            return self.say(f"The gateway is already {state}.")
        self.app.start_gateway()

    def action_stop_gateway(self) -> None:
        """Stop the gateway after a confirmation."""
        if self.app.gateway_state() == GW_STOPPED:
            return self.say("The gateway is not running.")
        n = len(self.app.snapshot.running_models) if self.app.snapshot else 0
        extra = f" {n} loaded model{'s' * (n != 1)} will be unloaded." if n else ""

        def done(ok: bool | None) -> None:
            if ok:
                self.app.stop_gateway()

        self.app.push_screen(ConfirmScreen(f"Stop the gateway?{extra}"), done)

    def action_copy_url(self) -> None:
        """Copy the OpenAI-style base URL of the gateway."""
        url = f"{self.gateway_url()}/v1"
        self.app.copy_to_clipboard(url)
        self.say(f"Copied {url}")

    def action_copy_snippet(self) -> None:
        """Copy the curl line shown in the gateway card."""
        snippet = self.snippet()
        if not snippet:
            return self.say("No model to build a snippet for yet.")
        self.app.copy_to_clipboard(snippet)
        self.say("Copied the curl command")
