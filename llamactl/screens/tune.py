"""TuneScreen: a measured tune with live trials, cancel and a summary against the estimate."""
from __future__ import annotations

import re
import threading
import time
from pathlib import Path

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Label, Static

from .. import actions, hardware, tuner
from ..actions import ActionError, TuneOutcome
from ..gateway import LOADED, LOADING, SLEEPING, GatewayError
from ..gguf import ModelInfo
from ..hardware import Host
from ..intents import Intent
from ..measure import Measurement
from ..planner import GB, Plan

COLUMNS = (("#", "n"), ("change", "how"), ("ctx", "ctx"), ("load s", "load"), ("VRAM", "vram"),
           ("prompt tok/s", "pp"), ("gen tok/s", "tg"), ("outcome", "out"))
SUMMARY_RE = re.compile(r"^(?P<how>[^:]+): (?P<ctx>\d+) ctx, (?P<kv>\S+) KV, (?P<verdict>[^,]+)"
                        r"(?:, (?P<threads>\d+) threads)?$")
DEFAULT_TRIALS = 4
ERRORS = (GatewayError, ActionError, LookupError, ValueError, OSError)
ELAPSED_EVERY = 1.0          # seconds between progress line refreshes while running
OUTCOME_WIDTH = 60           # characters of an error shown in the outcome column

CHECKING, REFUSED, RUNNING, DONE = "checking", "refused", "running", "done"


def split_summary(summary: str) -> tuple[str, str]:
    """Split a trial summary (``ctx down: 8192 ctx, f16 KV, GPU``) into "change" and "ctx" cells.

    Args:
        summary: `Candidate.summary` as carried by a `TrialEvent`.

    Returns:
        (change text such as ``ctx down, f16 KV, GPU``, context per slot); the whole text and
        "-" when it does not parse.
    """
    m = SUMMARY_RE.match(summary)
    if m is None:
        return summary, "-"
    extra = f", {m['threads']} threads" if m["threads"] else ""
    return f"{m['how']}, {m['kv']} KV, {m['verdict']}{extra}", m["ctx"]


def trial_cells(index: int, summary: str, m: Measurement | None) -> tuple:
    """Cells of one trial row; a running trial (no measurement) shows dots.

    Args:
        index: 1-based trial number.
        summary: The trial's `Candidate.summary`.
        m: Its measurement, None while it runs.

    Returns:
        The eight cells.
    """
    change, ctx = split_summary(summary)
    if m is None:
        return (str(index), change, ctx, "...", "", "", "", Text("running", style="yellow"))
    if m.cancelled:
        return (str(index), change, ctx, "-", "-", "-", "-", Text("cancelled", style="yellow"))
    if not m.ok:
        return (str(index), change, ctx, f"{m.load_s:.0f}" if m.load_s else "-", "-", "-", "-",
                Text(m.error.split(" (log:")[0][:OUTCOME_WIDTH], style="red"))
    return (str(index), change, ctx, f"{m.load_s:.0f}", f"{m.vram_used / GB:.1f} GB",
            f"{m.prompt_tps:.0f}", f"{m.gen_tps:.1f}" if m.gen_tps else "-",
            Text("ok", style="green"))


def summary_text(plan: Plan, outcome: TuneOutcome, active: bool) -> Text:
    """The result of a tune against the estimate it started from.

    Args:
        plan: The estimate (calibrated plan) shown before the tune.
        outcome: What `actions.run_tune` returned.
        active: The tuned preset is the model's active one.

    Returns:
        Multi-line styled text.
    """
    r = outcome.result
    t = Text()
    if not r.ok or r.best is None or r.measurement is None:
        t.append(r.message or "nothing loaded", style="red")
        return t
    best, m = r.best, r.measurement
    t.append("Winner  ", style="bold")
    t.append(f"{best.ctx_per_slot} ctx per slot, {best.profile.cache_type_k} KV, {best.verdict}, "
             + (f"{m.gen_tps:.1f} tok/s" if m.gen_tps else f"{m.prompt_tps:.0f} prompt tok/s")
             + f", {m.vram_used / GB:.1f} GB VRAM, loads in {m.load_s:.0f} s\n")
    est = plan.budget.total()
    ratio = f" (measured is x{m.vram_used / est:.2f} of the estimate)" if est and m.vram_used else ""
    t.append("Estimate  ", style="bold")
    t.append(f"{plan.ctx_per_slot} ctx per slot, {plan.profile.cache_type_k} KV, {plan.verdict}, "
             f"needs {est / GB:.1f} GB{ratio}\n")
    t.append(r.message + "\n", style="dim")
    t.append("This preset is active: it is served as plain alias." if active else
             "Not active yet: Use makes it the preset served as plain alias.",
             style="green" if active else "yellow")
    if outcome.reload_error:
        t.append(f"\nThe gateway was not reloaded: {outcome.reload_error}", style="yellow")
    elif outcome.reloaded:
        t.append("\nThe running gateway re-read the presets.", style="dim")
    return t


class TuneScreen(ModalScreen[str | None]):
    """Tune a model for an intent: pre-flight, live trial table, summary.

    The search runs in a worker that outlives the screen: closing while it runs cancels it
    (the trial in flight is stopped too) and the trials so far are still recorded and the app refreshed.
    Dismisses with "used" (made active), "kept" or None.

    Args:
        model: Model path.
        info: Model metadata.
        intent: What to tune for.
        trials: Most launches the search may make.
    """

    BINDINGS = [Binding("escape", "close", "Close (cancels a running tune)")]
    CSS = """
    TuneScreen { align: center middle; }
    TuneScreen #tbox { width: 96%; height: 96%; padding: 0 2; border: round $accent;
        background: $surface; }
    TuneScreen #t_title { text-style: bold; margin-top: 1; }
    TuneScreen #t_pre, TuneScreen #t_summary { height: auto; margin: 1 0; }
    TuneScreen #t_trials { height: 1fr; }
    TuneScreen #t_progress { height: auto; color: $text-muted; }
    TuneScreen #t_buttons { height: auto; margin: 1 0; }
    TuneScreen #t_buttons Button { margin-right: 1; }
    """

    def __init__(self, model: str, info: ModelInfo, intent: Intent,
                 trials: int = DEFAULT_TRIALS) -> None:
        super().__init__()
        self.model, self.info, self.intent, self.trials = model, info, intent, trials
        self.state = CHECKING
        self.plan: Plan | None = None
        self.host: Host | None = None
        self.loaded: list[str] = []
        self.outcome: TuneOutcome | None = None
        self.error = ""
        self._cancel = threading.Event()
        self._t0 = 0.0
        self._last = ""

    def compose(self) -> ComposeResult:
        """Title, pre-flight text, trial table, progress, summary and buttons."""
        with Vertical(id="tbox"):
            yield Label("", id="t_title")
            yield Static("Checking free memory...", id="t_pre")
            yield DataTable(id="t_trials", cursor_type="row")
            yield Label("", id="t_progress")
            yield Static("", id="t_summary")
            with Horizontal(id="t_buttons"):
                yield Button("Cancel tune", id="cancel", variant="warning")
                yield Button("Unload gateway models and retry", id="unload", variant="primary")
                yield Button("Tune anyway", id="anyway", variant="error")
                yield Button("Use (make active)", id="use", variant="success")
                yield Button("Keep as is", id="keep")
                yield Button("Close", id="close")

    def on_mount(self) -> None:
        """Fill the title and columns, then run the pre-flight check."""
        a = self.app
        self.query_one("#t_title", Label).update(
            f"Tune {a.alias_of(self.model)} for {self.intent.label}  "
            f"({Path(self.model).name}, up to {self.trials} trials)")
        t = self.query_one("#t_trials", DataTable)
        for label, key in COLUMNS:
            t.add_column(label, key=key)
        self.set_interval(ELAPSED_EVERY, self._tick)
        self._buttons()
        self.preflight()

    # -- state -------------------------------------------------------------------------

    def _buttons(self) -> None:
        """Show only the buttons that make sense in the current state."""
        show = {
            "cancel": self.state == RUNNING and not self._cancel.is_set(),
            "unload": self.state == REFUSED and bool(self.loaded),
            "anyway": self.state == REFUSED,
            "use": self.state == DONE and self._can_use(),
            "keep": self.state == DONE and self.outcome is not None and self.outcome.result.ok,
            "close": self.state != RUNNING or self._cancel.is_set(),
        }
        for bid, visible in show.items():
            self.query_one(f"#{bid}", Button).display = visible
        focus = next((b for b in ("use", "unload", "close") if show[b]), None)
        if focus:
            self.query_one(f"#{focus}", Button).focus()

    def _can_use(self) -> bool:
        """True when the tuned preset exists and is not yet the active one."""
        ent = self.app.cfg.models.get(self.model)
        return bool(self.outcome and self.outcome.result.ok and
                    (ent is None or ent.active != self.intent.key))

    def _say_pre(self, text: Text | str) -> None:
        """Set the pre-flight text."""
        self.query_one("#t_pre", Static).update(text)

    # -- preflight ---------------------------------------------------------------------

    def preflight(self) -> None:
        """Read free memory and check it against the plan (worker), then start or refuse."""
        a = self.app
        self.state = CHECKING
        self._buttons()
        self._say_pre("Checking free memory...")

        def work() -> tuple[Host, Plan, str | None, list[str]]:
            host = (hardware.refresh(a.host, a.cfg.bin_dir) if a.host.ram_total
                    else hardware.detect(a.cfg.bin_dir))
            plan = actions.plan_for(a.cfg, self.info, host, self.intent, a.calibration)
            refusal = tuner.preflight(plan, host)
            loaded: list[str] = []
            if refusal and a.gateway.running:
                try:
                    loaded = [m.name for m in a.gateway.models()
                              if m.state in (LOADED, LOADING, SLEEPING)]
                except GatewayError:
                    loaded = []
            return host, plan, refusal, loaded

        a.run_bg(work, label="Checking free memory", group="tune-pre", done=self._preflight_done)

    def _preflight_done(self, result: tuple[Host, Plan, str | None, list[str]]) -> None:
        """Pre-flight finished: start the tune, or show the refusal and the options."""
        self.host, self.plan, refusal, self.loaded = result
        self.app.host = self.host
        h, plan = self.host, self.plan
        facts = (f"free VRAM {h.vram_free / GB:.1f} GB, free RAM {h.ram_free / GB:.1f} GB; the "
                 f"plan ({plan.verdict}, {plan.ctx_per_slot} ctx) needs {plan.budget.total() / GB:.1f} GB")
        if refusal is None:
            self._say_pre(Text(f"Pre-flight ok: {facts}.", style="green"))
            return self.start()
        self.state = REFUSED
        t = Text(f"Pre-flight refused: {refusal}\n", style="red")
        t.append(facts + ".\n", style="dim")
        if self.loaded:
            t.append(f"The gateway has loaded: {', '.join(self.loaded)}. Unloading them frees "
                     "memory.", style="yellow")
        self._say_pre(t)
        self._buttons()

    def unload_and_retry(self) -> None:
        """Unload every loaded gateway model (worker), then check again."""
        a = self.app
        names = list(self.loaded)
        self.state = CHECKING
        self._buttons()
        self._say_pre(f"Unloading {', '.join(names)}...")

        def work() -> None:
            for n in names:
                a.gateway.unload(n)
                actions.wait_unloaded(a.gateway, n, 30.0)

        a.run_bg(work, label="Unloading gateway models", group="tune-pre",
                 done=lambda _: self.preflight(), always=a.refresh_serving)

    # -- the tune ----------------------------------------------------------------------

    def start(self, force: bool = False) -> None:
        """Run the search in a worker; trial events flow back into the table.

        Args:
            force: Skip the free-memory check ("Tune anyway").
        """
        a = self.app
        assert self.host is not None
        host, cancel, path = self.host, self._cancel, self.model
        self.state = RUNNING
        self._t0 = time.monotonic()
        a.cancel_flags.add(cancel)
        self._buttons()

        def work() -> tuple[TuneOutcome | None, str]:
            try:
                return actions.run_tune(
                    a.cfg, self.info, host, self.intent, a.runner_factory(), a.calibration,
                    a.gateway, trials=self.trials, force=force,
                    progress=lambda e: a.ui(self._event, e), cancel=cancel.is_set), ""
            except ERRORS as e:
                return None, str(e)

        def done(result: tuple[TuneOutcome | None, str]) -> None:
            outcome, error = result
            if outcome is not None:        # trials and the winner were saved: refresh every view
                a.presets_changed(path, save=False, reload=False)
            if error:
                a.fail(f"Tune {a.alias_of(path)}: {error}")
            if self.is_attached:
                self._finished(outcome, error)

        def always() -> None:
            a.cancel_flags.discard(cancel)
            if self.is_attached and self.state == RUNNING:       # the worker died unexpectedly
                self._finished(None, "the tune did not finish; see the message in the status bar")

        a.run_bg(work, label=f"Tuning {a.alias_of(path)}", group="tune", done=done, always=always)

    def _event(self, e: tuner.TrialEvent) -> None:
        """A trial started or finished: update its row and the progress line."""
        t = self.query_one("#t_trials", DataTable)
        cells = trial_cells(e.index, e.summary, e.measurement)
        key = str(e.index)
        if key in t.rows:
            for col, v in zip([c for _, c in COLUMNS], cells):
                t.update_cell(key, col, v, update_width=True)
        else:
            t.add_row(*cells, key=key)
            t.move_cursor(row=t.row_count - 1)
        self._last = f"trial {e.index} of up to {e.total}: {e.summary}"
        self._tick()

    def _tick(self) -> None:
        """Refresh the progress line (elapsed time, cancelling note)."""
        if self.state != RUNNING:
            return
        what = "cancelling, stopping the server; " if self._cancel.is_set() else ""
        self.query_one("#t_progress", Label).update(
            f"{what}{self._last or 'starting'}  ({time.monotonic() - self._t0:.0f} s)")

    def _finished(self, outcome: TuneOutcome | None, error: str) -> None:
        """The tune ended: show the summary and the buttons for this state."""
        self.state = DONE
        self.outcome = outcome
        self.error = error
        self.query_one("#t_progress", Label).update("")
        summary = self.query_one("#t_summary", Static)
        if outcome is None:
            summary.update(Text(error or "the tune failed", style="red"))
        else:
            ent = self.app.cfg.models.get(self.model)
            active = bool(ent and ent.active == self.intent.key)
            assert self.plan is not None
            summary.update(summary_text(self.plan, outcome, active))
        self._buttons()

    # -- actions -----------------------------------------------------------------------

    def action_close(self) -> None:
        """Close; a running tune is cancelled first (its trials are still recorded)."""
        self._cancel.set()
        self.dismiss(None)

    @on(Button.Pressed, "#cancel")
    def _cancel_pressed(self) -> None:
        """Cancel tune: stop the trial in flight and the search."""
        self._cancel.set()
        self._tick()
        self._buttons()

    @on(Button.Pressed, "#unload")
    def _unload_pressed(self) -> None:
        """Unload gateway models and retry."""
        self.unload_and_retry()

    @on(Button.Pressed, "#anyway")
    def _anyway_pressed(self) -> None:
        """Tune anyway: skip the free-memory check."""
        self._say_pre(Text("Tuning without the free-memory check.", style="yellow"))
        self.start(force=True)

    @on(Button.Pressed, "#use")
    def _use_pressed(self) -> None:
        """Use: make the tuned preset the active one and close."""
        if self.app.activate_preset(self.model, self.intent.key):
            self.dismiss("used")

    @on(Button.Pressed, "#keep")
    def _keep_pressed(self) -> None:
        """Keep as is: close, leaving the active preset alone."""
        self.dismiss("kept")

    @on(Button.Pressed, "#close")
    def _close_pressed(self) -> None:
        """Close (cancels a running tune first)."""
        self.action_close()
