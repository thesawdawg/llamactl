"""OnboardingScreen: the guided first run, a thin shell over `llamactl.onboarding`.

Five steps (llama.cpp, hardware, models, purpose, finish) with Back / Next / Skip setup. All
detection, scanning and planning runs in workers through `LlamaCtl.run_bg`; the UI thread only
draws. Nothing is written to the config until the last step (or "Find a model"), which applies
the draft on the UI thread.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (Button, Checkbox, ContentSwitcher, Footer, Header, Input, Label,
                             OptionList, Static)
from textual.widgets.option_list import Option

from .. import hardware, library, network, onboarding as ob
from ..fmt import FITS_STYLE, human
from ..gguf import ModelInfo
from ..hardware import Host
from ..intents import INTENTS, by_key
from ..onboarding import Dismissal
from ..widgets import layout
from ..widgets.buttonbar import ButtonBar
from ..widgets.folders import FolderList

ERRORS = (LookupError, OSError, ValueError)
NO_MODEL = ""                 # option id of "no model for now"


class OnboardingScreen(Screen[Dismissal]):
    """The first-run flow; dismisses with an `onboarding.Dismissal`.

    Args:
        which: ``shutil.which`` compatible lookup used to find llama-server on ``PATH``.
    """

    BINDINGS = [
        Binding("ctrl+n", "next", "Next"),
        Binding("ctrl+b", "back", "Back"),
        Binding("escape", "skip", "Skip setup"),
    ]
    HORIZONTAL_BREAKPOINTS = layout.HORIZONTAL_BREAKPOINTS
    VERTICAL_BREAKPOINTS = layout.VERTICAL_BREAKPOINTS
    DEFAULT_CSS = """
    OnboardingScreen #ob_box { padding: 0 2; height: 1fr; }
    OnboardingScreen #ob_progress { text-style: bold; color: $accent; margin-top: 1; }
    OnboardingScreen #ob_steps { height: 1fr; margin-top: 1; }
    OnboardingScreen .step { height: 1fr; }
    OnboardingScreen .lead { height: auto; margin-bottom: 1; }
    OnboardingScreen .hint { color: $text-muted; height: auto; }
    OnboardingScreen .row { height: auto; }
    OnboardingScreen .row Label { width: 22; padding: 0 1 0 0; }
    OnboardingScreen.-narrow .row { layout: vertical; }
    OnboardingScreen.-narrow .row Label { width: 1fr; padding: 0; }
    OnboardingScreen #ob_bin { width: 1fr; max-width: 60; }
    OnboardingScreen #ob_bin_msg { height: auto; margin-bottom: 1; }
    OnboardingScreen #ob_hw { height: auto; border: round $primary; padding: 0 1; margin-bottom: 1; }
    OnboardingScreen #ob_hw:focus-within { border: round $accent; }
    OnboardingScreen #ob_sentence { height: auto; }
    OnboardingScreen #ob_scan { height: auto; margin-top: 1; text-style: bold; }
    OnboardingScreen #ob_find { margin-top: 1; width: auto; }
    OnboardingScreen #purpose { height: 1fr; }
    OnboardingScreen.-narrow #purpose { layout: vertical; overflow-y: auto; }
    OnboardingScreen.-narrow #purpose .pane { height: auto; min-height: 8; }
    OnboardingScreen .pane { width: 1fr; height: 1fr; padding: 0 1; }
    OnboardingScreen .pane Label { text-style: bold; }
    OnboardingScreen .pane OptionList { height: 1fr; }
    OnboardingScreen #ob_summary { height: auto; margin-bottom: 1; }
    OnboardingScreen #ob_result { height: auto; }
    OnboardingScreen #finish Checkbox { width: 1fr; text-wrap: wrap; }
    OnboardingScreen #ob_msg { height: auto; color: $warning; }
    OnboardingScreen #ob_buttons { margin-top: 1; }
    """

    def __init__(self, which: Callable[[str], str | None] = shutil.which) -> None:
        super().__init__()
        self.which = which
        self.flow: ob.Flow | None = None
        self.was_onboarded = False
        self.loc = ob.LlamaLocation()
        self.build = ""
        self.host = Host()
        self.models: list[Path] = []
        self.infos: dict[str, ModelInfo] = {}
        self.scanned: tuple | None = None            # folders the shown scan is for
        self.scanning = False
        self.choices: list[ob.ModelChoice] = []
        self.result: ob.FinishResult | None = None
        self.gateway_note = ""
        self.bind_note = ""
        self._host_for = ""
        self._choices_for: tuple = ()

    # -- layout --------------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        """Progress line, one pane per step, message line and the buttons."""
        yield Header()
        with Vertical(id="ob_box"):
            yield Label("", id="ob_progress")
            with ContentSwitcher(id="ob_steps", initial=ob.STEP_LLAMA):
                with VerticalScroll(id=ob.STEP_LLAMA, classes="step"):
                    yield Static("", id="ob_llama", classes="lead")
                    with Horizontal(classes="row"):
                        yield Label("llama.cpp folder")
                        yield Input(id="ob_bin", compact=True)
                    yield Static("", id="ob_bin_msg")
                with VerticalScroll(id=ob.STEP_HARDWARE, classes="step"):
                    yield Static("", id="ob_hw")
                    yield Static("", id="ob_sentence")
                with VerticalScroll(id=ob.STEP_MODELS, classes="step"):
                    yield Static("Folders llamactl scans for .gguf files (subfolders included).",
                                 classes="lead")
                    yield FolderList(self.app.cfg.model_dirs)
                    yield Static("", id="ob_scan")
                    yield Button("Find a model", id="ob_find", variant="primary")
                with Horizontal(id=ob.STEP_PURPOSE, classes="step"):
                    with Vertical(classes="pane"):
                        yield Label("What will you mostly use it for?")
                        yield OptionList(*[Option(self._card(i), id=i.key) for i in INTENTS
                                           if i.key != "embed"], id="ob_intents")
                    with Vertical(classes="pane"):
                        yield Label("First model (optional)")
                        yield OptionList(id="ob_models")
                with VerticalScroll(id=ob.STEP_FINISH, classes="step"):
                    yield Static("", id="ob_summary")
                    yield Checkbox("Tune now: measure the best settings (opens the tune screen "
                                   "after this)", id="ob_tune")
                    yield Checkbox("Start the gateway now (nothing else is made permanent)",
                                   True, id="ob_gateway")
                    yield Static("", id="ob_why", classes="hint")
                    yield Checkbox("Reachable from other machines (listens on the whole network "
                                   "and makes an API key)", self.app.cfg.gateway.host
                                   == network.WILDCARD, id="ob_reachable")
                    yield Static("Off keeps the gateway on this machine only.",
                                 id="ob_reach_hint", classes="hint")
                    yield Static("", id="ob_result")
            yield Label("", id="ob_msg")
            with ButtonBar(id="ob_buttons", classes="-docked"):
                yield Button("Back", id="ob_back")
                yield Button("Next", id="ob_next", variant="primary")
                yield Button("Skip setup", id="ob_skip")
                yield Button("Close", id="ob_close", variant="primary")
        yield Footer()

    @staticmethod
    def _card(intent) -> Text:
        """An intent card without per-model numbers."""
        t = Text()
        t.append(intent.label, style="bold")
        t.append(f"  {intent.description}\n", style="dim")
        t.append("  " + ob.intent_blurb(intent).split("(", 1)[1].rstrip(")") + "\n", style="dim")
        return t

    def on_mount(self) -> None:
        """Start from the config, look for llama.cpp and read the hardware."""
        a = self.app
        self.sub_title = "Setup"
        self.was_onboarded = a.cfg.onboarded
        self.flow = ob.Flow(a.cfg)
        d = self.flow.draft
        self.query_one("#ob_bin", Input).value = d.bin_dir
        self.query_one("#ob_skip", Button).label = "Cancel" if self.was_onboarded else "Skip setup"
        a.run_bg(lambda: ob.locate_llama(d.bin_dir, self.which), label="Looking for llama.cpp",
                 group="ob-locate", done=self._located, quiet=True)
        self._ensure_host()
        self._show()

    # -- helpers -------------------------------------------------------------------------

    @property
    def draft(self) -> ob.Draft:
        """The choices made so far."""
        assert self.flow is not None
        return self.flow.draft

    def _msg(self, text: str) -> None:
        """Show a line above the buttons."""
        self.query_one("#ob_msg", Label).update(text)

    # -- navigation ----------------------------------------------------------------------

    def _show(self) -> None:
        """Show the current step: progress, pane, buttons, focus; start the step's work."""
        assert self.flow is not None
        f = self.flow
        self._msg("")
        self.query_one("#ob_progress", Label).update(
            f.progress() + ("" if not self.result else "  -  done"))
        self.query_one("#ob_steps", ContentSwitcher).current = f.step
        finished = self.result is not None
        self.query_one("#ob_back", Button).display = not finished
        self.query_one("#ob_back", Button).disabled = f.first
        nxt = self.query_one("#ob_next", Button)
        nxt.display = not finished
        nxt.label = "Finish" if f.last else "Next"
        self.query_one("#ob_skip", Button).display = not finished
        close = self.query_one("#ob_close", Button)
        close.display = finished
        close.label = "Close and tune" if finished and self.result and self.result.tune else "Close"
        self.query_one("#ob_buttons", ButtonBar).reflow()
        {ob.STEP_LLAMA: self._enter_llama, ob.STEP_HARDWARE: self._enter_hardware,
         ob.STEP_MODELS: self._enter_models, ob.STEP_PURPOSE: self._enter_purpose,
         ob.STEP_FINISH: self._enter_finish}[f.step]()

    def action_next(self) -> None:
        """Next step, or finish on the last one."""
        assert self.flow is not None
        if self.result is not None:
            return
        if self.flow.last:
            return self._finish()
        self.flow.next()
        self._show()

    def action_back(self) -> None:
        """Previous step."""
        assert self.flow is not None
        if self.result is None and self.flow.back():
            self._show()

    def action_skip(self) -> None:
        """Leave without applying anything; the user is not asked again."""
        if self.result is not None:
            return self._close(Dismissal())
        try:
            ob.skip(self.app.cfg)
        except OSError as e:
            return self._msg(f"cannot save the config: {e}")
        self._close(Dismissal(ob.ACT_SKIPPED))

    def _close(self, how: Dismissal) -> None:
        """Dismiss the screen."""
        self.dismiss(how)

    @on(Button.Pressed, "#ob_next")
    def _next_pressed(self) -> None:
        """The Next / Finish button."""
        self.action_next()

    @on(Button.Pressed, "#ob_back")
    def _back_pressed(self) -> None:
        """The Back button."""
        self.action_back()

    @on(Button.Pressed, "#ob_skip")
    def _skip_pressed(self) -> None:
        """The Skip setup button."""
        self.action_skip()

    @on(Button.Pressed, "#ob_close")
    def _close_pressed(self) -> None:
        """Close after finishing: open the tune screen when asked to."""
        r = self.result
        if r is not None and r.tune:
            return self._close(Dismissal(ob.ACT_TUNE, r.model, r.intent))
        self._close(Dismissal(ob.ACT_DONE, r.model if r else "", r.intent if r else ""))

    # -- step 1: llama.cpp ---------------------------------------------------------------

    def _enter_llama(self) -> None:
        """Draw the llama.cpp step."""
        self._draw_llama()
        self.query_one("#ob_bin", Input).focus()

    def _located(self, loc: ob.LlamaLocation) -> None:
        """Detection finished: when the drafted folder is no good, offer what was found."""
        self.loc = loc
        if loc.found and not self.flow_llama_ok():
            self.query_one("#ob_bin", Input).value = loc.bin_dir      # triggers Input.Changed
        self._draw_llama()

    def flow_llama_ok(self) -> bool:
        """True when the drafted folder holds llama-server."""
        return self.flow is not None and self.flow.llama_ok()

    def _draw_llama(self) -> None:
        """Text of step 1 from the typed folder, where it came from and its version."""
        d = self.draft
        loc = ob.LlamaLocation()
        if self.flow_llama_ok():
            folder = str(Path(d.bin_dir.strip()).expanduser())
            loc = ob.LlamaLocation(folder, self.loc.source if folder == self.loc.bin_dir
                                   else ob.SRC_TYPED)
        self.query_one("#ob_llama", Static).update(ob.llama_text(loc, self.build))
        problem = ob.check_bin_dir(d.bin_dir)
        self.query_one("#ob_bin_msg", Static).update(
            Text("folder ok", style="green") if not problem else Text(problem, style="yellow"))

    @on(Input.Changed, "#ob_bin")
    def _bin_typed(self, event: Input.Changed) -> None:
        """The folder box changed: validate, and read the version of a good folder."""
        self._bin_changed(event.value)

    def _bin_changed(self, value: str) -> None:
        """Store the drafted folder and read its version in a worker when it is valid."""
        if self.flow is None:
            return
        self.draft.bin_dir = value
        self.build = ""
        self._draw_llama()
        if not ob.check_bin_dir(value):
            folder = str(Path(value.strip()).expanduser())
            self.app.run_bg(lambda: (folder, self.app.build_for(folder)),
                            label="Reading the llama.cpp version", group="ob-build",
                            exclusive=True, done=self._build_read, quiet=True)

    def _build_read(self, result: tuple[str, str]) -> None:
        """The version string arrived (ignored when the folder changed meanwhile)."""
        folder, build = result
        if self.flow_llama_ok() and str(Path(self.draft.bin_dir.strip()).expanduser()) == folder:
            self.build = build or "unknown"
            self._draw_llama()

    # -- step 2: hardware ----------------------------------------------------------------

    def _ensure_host(self) -> None:
        """Detect the hardware for the drafted folder (reusing the app's snapshot when it is for the same)."""
        a = self.app
        key = self.draft.bin_dir.strip()
        if key == self._host_for and self.host.ram_total:
            return
        self._host_for = key
        if key == a.cfg.bin_dir and a.host.ram_total:
            self.host = a.host
            return self._draw_hardware()
        a.run_bg(lambda: (key, hardware.detect(key or None)), label="Reading hardware",
                 group="ob-host", exclusive=True, done=self._host_read, quiet=True)

    def _host_read(self, result: tuple[str, Host]) -> None:
        """Detection finished for a folder (ignored when the folder changed meanwhile)."""
        key, host = result
        if key != self._host_for:
            return
        self.host = host
        self._draw_hardware()
        if self.flow is not None and self.flow.step == ob.STEP_PURPOSE:
            self._refresh_choices()

    def _enter_hardware(self) -> None:
        """Draw the hardware step."""
        self._ensure_host()
        self._draw_hardware()
        self.query_one("#ob_next", Button).focus()

    def _draw_hardware(self) -> None:
        """Devices, RAM, CPU and the one-sentence meaning."""
        self.query_one("#ob_hw", Static).update("\n".join(ob.hardware_lines(self.host)))
        intent = by_key.get(self.draft.default_intent, by_key["chat"])
        self.query_one("#ob_sentence", Static).update(
            ob.hardware_sentence(self.host, intent, self.app.cfg.headroom_pct))

    # -- step 3: models ------------------------------------------------------------------

    def _enter_models(self) -> None:
        """Draw the models step and scan when the folders changed since the last scan."""
        self._rescan_if_needed()
        self._draw_scan()
        self.query_one("#add_dir", Input).focus()

    @on(FolderList.Changed)
    def _folders_changed(self) -> None:
        """A folder was added or removed: store the list and scan again."""
        self.draft.model_dirs = self.query_one(FolderList).dirs()
        self._rescan_if_needed()
        self._draw_scan()

    def _rescan_if_needed(self) -> None:
        """Scan the drafted folders in a worker unless that scan is already shown or running."""
        dirs = tuple(self.draft.model_dirs)
        if self.scanned == dirs:
            return
        self.scanned = dirs
        self.scanning = True
        a = self.app

        def work() -> tuple[tuple, list[Path], dict[str, ModelInfo]]:
            models = library.discover_models(list(dirs))
            infos = {}
            for p in models:
                info = a.read_header(p)
                if info is not None:
                    infos[str(p)] = info
            return dirs, models, infos

        a.run_bg(work, label="Scanning models", group="ob-scan", exclusive=True,
                 done=self._scanned_done, always=self._scan_ended)

    def _scanned_done(self, result: tuple[tuple, list[Path], dict[str, ModelInfo]]) -> None:
        """A scan finished (ignored when the folders changed meanwhile)."""
        dirs, models, infos = result
        if dirs != self.scanned:
            return
        self.models, self.infos, self.scanning = models, infos, False
        self._choices_for = ()
        self._draw_scan()
        if self.flow is not None and self.flow.step == ob.STEP_PURPOSE:
            self._refresh_choices()

    def _scan_ended(self) -> None:
        """The scan worker ended: do not stay "scanning" if it died."""
        if self.scanning and self.scanned is not None:
            self.scanning = False
            self._draw_scan()

    def _draw_scan(self) -> None:
        """The count and size line, and the Find a model button when there is nothing."""
        dirs = self.draft.model_dirs
        text = ("Scanning..." if self.scanning else
                ob.scan_summary(self.models, self.infos).text(dirs))
        self.query_one("#ob_scan", Static).update(text)
        self.query_one("#ob_find", Button).display = (
            not self.scanning and self.scanned is not None and not self.models)

    @on(Button.Pressed, "#ob_find")
    def _find_pressed(self) -> None:
        """No models: apply what was chosen so far and land on Discover."""
        self._apply(model="", then=Dismissal(ob.ACT_DISCOVER))

    # -- step 4: purpose -----------------------------------------------------------------

    def _enter_purpose(self) -> None:
        """Draw the purpose step: highlight the default intent and list the models."""
        lst = self.query_one("#ob_intents", OptionList)
        keys = [str(lst.get_option_at_index(i).id) for i in range(lst.option_count)]
        want = self.draft.default_intent if self.draft.default_intent in keys else keys[0]
        if lst.highlighted != keys.index(want):
            lst.highlighted = keys.index(want)
        self._refresh_choices()
        lst.focus()

    @on(OptionList.OptionHighlighted, "#ob_intents")
    def _intent_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """A different intent: it becomes the default and the model verdicts follow."""
        if self.flow is None or event.option.id is None:
            return
        if str(event.option.id) != self.draft.default_intent:
            self.draft.default_intent = str(event.option.id)
            self._refresh_choices()

    def _refresh_choices(self) -> None:
        """Recompute each model's verdict for the drafted intent in a worker."""
        key = (self.draft.default_intent, self.host.ram_total, len(self.host.gpus),
               tuple(sorted(self.infos)))
        if key == self._choices_for:
            return
        self._choices_for = key
        a, host, infos, models = self.app, self.host, dict(self.infos), list(self.models)
        intent = self.draft.default_intent
        if not infos or (not host.gpus and not host.ram_total):
            return self._fill_models([])
        a.run_bg(lambda: (key, ob.model_choices(models, infos, host, a.cfg, a.calibration, intent)),
                 label="Checking models", group="ob-choices", exclusive=True,
                 done=self._choices_ready, quiet=True)

    def _choices_ready(self, result: tuple[tuple, list[ob.ModelChoice]]) -> None:
        """Verdicts arrived (ignored when they are for another intent or scan)."""
        key, choices = result
        if key == self._choices_for:
            self._fill_models(choices)

    def _fill_models(self, choices: list[ob.ModelChoice]) -> None:
        """Fill the model list: "no model for now" first, then one line per model."""
        self.choices = choices
        lst = self.query_one("#ob_models", OptionList)
        lst.clear_options()
        lst.add_option(Option(Text("No model for now (choose one later in Models)", style="dim"),
                              id=NO_MODEL))
        for c in choices:
            t = Text(f"{c.alias}  ", style="bold")
            t.append(f"{human(c.size)}  ")
            t.append(c.verdict, style=FITS_STYLE.get(c.verdict, ""))
            t.append(f"  {c.ctx} ctx", style="dim")
            if c.intent != self.draft.default_intent:
                t.append(f"  ({by_key[c.intent].label})", style="dim")
            lst.add_option(Option(t, id=c.path))
        paths = [c.path for c in choices]
        lst.highlighted = paths.index(self.draft.model) + 1 if self.draft.model in paths else 0
        if not choices:
            self.draft.model = NO_MODEL

    @on(OptionList.OptionHighlighted, "#ob_models")
    def _model_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """The highlighted model is the first model."""
        if self.flow is not None and event.option.id is not None:
            self.draft.model = str(event.option.id)

    # -- step 5: finish ------------------------------------------------------------------

    def _alias(self, path: str) -> str:
        """Alias of a scanned model."""
        return next((c.alias for c in self.choices if c.path == path), Path(path).stem if path else "")

    def _enter_finish(self) -> None:
        """Draw the summary and enable the checkboxes that make sense."""
        d = self.draft
        self.query_one("#ob_summary", Static).update(
            "\n".join(ob.summary_lines(d, self.flow_llama_ok(), self._alias(d.model))))
        tune, gw = self.query_one("#ob_tune", Checkbox), self.query_one("#ob_gateway", Checkbox)
        usable = bool(d.model) and self.flow_llama_ok()
        tune.disabled = gw.disabled = not usable
        self.query_one("#ob_why", Static).update(
            "" if usable else "Choose a first model (and a working llama.cpp folder) to enable these two.")
        if not usable:
            tune.value = False
            gw.value = False
        elif not gw.value and not gw.has_class("touched"):
            gw.value = True
        self.query_one("#ob_next", Button).focus()

    @on(Checkbox.Changed)
    def _box_changed(self, event: Checkbox.Changed) -> None:
        """Remember that the user chose, so a later redraw does not undo it."""
        event.checkbox.add_class("touched")

    def _finish(self) -> None:
        """Apply the draft and show the endpoint."""
        d = self.draft
        d.tune = self.query_one("#ob_tune", Checkbox).value
        d.start_gateway = self.query_one("#ob_gateway", Checkbox).value
        d.reachable = self.query_one("#ob_reachable", Checkbox).value
        self._apply(model=d.model, then=None)

    def _apply(self, *, model: str, then: Dismissal | None) -> None:
        """Run `onboarding.finish` on the UI thread and refresh the app.

        Args:
            model: The model to prepare ("" for none).
            then: How to close right away, or None to show the result pane.
        """
        a, d = self.app, self.draft
        d.model = model
        try:
            res = ob.finish(a.cfg, d, self.infos, self.host, a.calibration, a.gateway.listen_url)
        except ERRORS as e:
            return self._msg(f"Could not finish: {e}")
        a.setup_applied(res)
        if then is not None:
            return self._close(then)
        self.result = res
        running = a.gateway.state
        self.bind_note = ""
        if running is not None and running.host and network.is_exposed(running.host) != res.exposed:
            self.bind_note = (f"The gateway already runs on {running.host}: stop it (d) and start "
                              "it (u) in Home to apply the new setting.")
        self._draw_result(res)
        if res.start_gateway:
            self.gateway_note = "Starting the gateway..."
            a.start_gateway(finished=self._gateway_finished)
        else:
            self.gateway_note = ""
        self._draw_result(res)
        self._show()
        self.query_one("#ob_close", Button).focus()

    def _gateway_finished(self, error: str) -> None:
        """The gateway start ended: say how."""
        urls = ", ".join(self.result.urls) if self.result else ""
        self.gateway_note = (f"Gateway running at {urls}" if not error
                             else f"The gateway did not start: {error}")
        if self.result is not None:
            self._draw_result(self.result)

    def _draw_result(self, res: ob.FinishResult) -> None:
        """The last pane after finishing: endpoint, model name, curl and the gateway note."""
        for wid in ("ob_tune", "ob_gateway", "ob_why", "ob_reachable", "ob_reach_hint"):
            self.query_one(f"#{wid}").display = False
        t = Text()
        t.append("Setup saved.\n\n", style="bold green")
        t.append(f"Endpoint  {res.url}\n")
        for other in res.urls:
            if f"{other}/v1" != res.url:
                t.append(f"          {other}/v1\n")
        if res.new_key:
            t.append("API key   ")
            t.append(res.new_key, style="bold")
            t.append("\n          shown only now; it is kept in Settings (clients send it as "
                     "a Bearer token)\n")
        for note in network.exposure_notes(self.app.cfg.gateway.host) if res.exposed else []:
            t.append(f"          {note}\n", style="dim")
        if res.model:
            t.append(f"Model     {res.served} ({res.preset} preset"
                     f"{', planned just now' if res.created else ''})\n\n")
            t.append("Try it\n", style="bold")
            t.append(res.snippet + "\n", style="dim")
        else:
            t.append("\nNo model yet: pick one in Models (2), or find one in Discover (4).\n")
        if self.gateway_note:
            t.append("\n" + self.gateway_note, style="yellow" if "not" in self.gateway_note
                     else "")
        elif res.model:
            t.append("\nThe gateway is not started: press u in Home to start it.", style="dim")
        if self.bind_note:
            t.append("\n" + self.bind_note, style="yellow")
        self.query_one("#ob_summary", Static).update("")
        self.query_one("#ob_result", Static).update(t)
