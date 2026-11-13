"""Textual TUI: the shell, the services every view shares and the flows more than one view uses.

One `LlamaCtl` owns the config, the host snapshot, the model list with its GGUF headers, the
calibration, the gateway and the session store. Views read them from the app and never poll on
their own: the host is polled once, the serving state (gateway models plus standalone servers)
is gathered once and posted as one `ServingChanged` message, models are scanned once.

Threading rule: nothing slower than ~50 ms runs on the UI thread. Every gateway, session, tuner
and HTTP call goes through `run_bg`, whose results come back through `ui`, which drops them when
the app or the target widget is already gone.
"""
from __future__ import annotations

import shutil
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.content import Content
from textual.message import Message
from textual.widget import Widget
from textual.widgets import ContentSwitcher, Footer, Header, Label, ListItem, ListView
from textual.worker import Worker, get_current_worker

from . import actions, hardware, library, network, onboarding, planner, tuner, usage as usage_mod
from .actions import ActionError
from .calibration import Calibration
from .config import Config
from .gateway import Gateway, GatewayError
from .gguf import GGUFError, ModelInfo, ModelInfoCache, read_model_info
from .goals import TuningGoals
from .onboarding import Dismissal, FinishResult
from .serving import (GW_STARTING, GW_STOPPING, Health, ServingSnapshot, gateway_state, gather,
                      http_health)
from .sessions import Session, SessionStore
from .suggest import (ACT_DISCOVER, ACT_MODELS, ACT_SETTINGS, ACT_START_GATEWAY, ACT_TUNE,
                      Suggestion, binary_found)
from .support import SupportChecker
from .tuner import Runner, ServerRunner
from .views.base import BaseView
from .views.hf import HFView
from .views.home import HomeView
from .views.models import ModelsView
from .views.serving import ServingView
from .views.settings import SettingsView
from .widgets import layout
from .widgets.gauges import ResourceGauges

VIEWS = (("home", "Home"), ("models", "Models"), ("serving", "Serving"),
         ("discover", "Discover"), ("settings", "Settings"))
VIEW_CLASSES: dict[str, type[BaseView]] = {
    "home": HomeView, "models": ModelsView, "serving": ServingView, "discover": HFView,
    "settings": SettingsView}
SERVING_POLL = 2.0          # seconds between serving polls while something runs
IDLE_EVERY = 3              # ... and every third poll while nothing runs
UNLOAD_WAIT = 30.0          # seconds to wait for a model to leave memory
ERRORS = (GatewayError, ActionError, LookupError, ValueError, OSError, GGUFError)


class UiCall(Message):
    """A callable a worker thread asks the UI thread to run (see `LlamaCtl.ui`).

    Args:
        call: The callable with its arguments already bound.
        owner: The object `call` is a method of, or None for a plain function.
    """

    def __init__(self, call: Callable[[], Any], owner: object | None) -> None:
        super().__init__()
        self.call = call
        self.owner = owner


class ServingChanged(Message):
    """The one message the serving poll posts; Home and Serving both consume the snapshot.

    Args:
        snapshot: The gateway and session state just gathered.
    """

    def __init__(self, snapshot: ServingSnapshot) -> None:
        super().__init__()
        self.snapshot = snapshot


class LlamaCtl(App):
    """Main application: sidebar navigation, gauges, status bar, five views.

    Every argument is optional; leaving one out selects the real implementation, and tests
    pass fakes.

    Args:
        cfg: Loaded config (default: read from disk).
        calibration: Measured trials (default: ``tuning.json``).
        store: Standalone and attached sessions.
        gateway: The router-mode gateway.
        info_cache: GGUF header cache.
        runner_factory: Returns the `Runner` a tune measures with.
        build: Returns the llama-server version string (called off the UI thread).
        health: Returns "ready", "loading" or "down" for a server URL.
        probe: Process usage probe; consecutive serving polls give the CPU deltas.
        which: ``shutil.which`` compatible lookup the first-run setup uses to find llama-server.
        support: Architecture support checker (one shared instance; or None to make one).
    """

    TITLE = "llamactl"
    HORIZONTAL_BREAKPOINTS = layout.HORIZONTAL_BREAKPOINTS
    VERTICAL_BREAKPOINTS = layout.VERTICAL_BREAKPOINTS
    CSS = """
    * { scrollbar-size-vertical: 1; }
    #shell { height: 1fr; }
    #nav { width: 14; background: $surface; border-right: solid $panel; padding: 0 1; }
    .-narrow #nav { width: 12; }
    #main { width: 1fr; height: 1fr; }
    #main > * { height: 1fr; }
    #gauges { height: 1; }
    #statusbar { height: 1; padding: 0 1; background: $panel; }
    #status { width: 1fr; height: 1; }
    #gwstate { width: auto; height: 1; color: $text-muted; }
    """
    BINDINGS = [
        Binding("1", "view('home')", "Home", show=False),
        Binding("2", "view('models')", "Models", show=False),
        Binding("3", "view('serving')", "Serving", show=False),
        Binding("4", "view('discover')", "Discover", show=False),
        Binding("5", "view('settings')", "Settings", show=False),
        Binding("r", "rescan", "Rescan"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, *, cfg: Config | None = None, calibration: Calibration | None = None,
                 store: SessionStore | None = None, gateway: Gateway | None = None,
                 info_cache: ModelInfoCache | None = None,
                 runner_factory: Callable[[], Runner] | None = None,
                 build: Callable[[], str] | None = None, health: Health = http_health,
                 probe: usage_mod.UsageProbe | None = None,
                 which: Callable[[str], str | None] = shutil.which,
                 support: SupportChecker | None = None) -> None:
        super().__init__()
        self.cfg = cfg or Config.load()
        self.calibration = calibration or Calibration.load()
        self.probe = probe or usage_mod.UsageProbe()
        self.store = store or SessionStore()
        self.support = support or SupportChecker(self.cfg.bin_dir)
        self.gateway = gateway or Gateway(self.cfg, learn=self._learn_log,
                                          blocked=self.blocked_map)
        self.info_cache = info_cache or ModelInfoCache()
        self.runner_factory: Callable[[], Runner] = runner_factory or (
            lambda: ServerRunner(self.cfg, support=self.support))
        self._build_fn = build or (lambda: ServerRunner(self.cfg).build())
        self.health = health
        self.which = which
        self.host = hardware.Host()
        self.models: list[Path] = []
        self.infos: dict[str, ModelInfo] = {}
        self.scanning = False
        self.build = ""
        self.snapshot: ServingSnapshot | None = None
        self.gw_busy = ""                      # "starting" / "stopping" while the app does that
        self.views: dict[str, BaseView] = {}
        self._wsl_asked = False                # the Windows-RAM question was pushed once
        self._wsl_checking = False             # a WSL-detection worker is in flight
        self.cancel_flags: set[threading.Event] = set()    # running tunes, set on quit
        self._gw_lock = threading.Lock()       # one gateway writer/reloader at a time
        self._scan_gen = 0
        self._poll_n = 0
        self._last_failure = ""
        self._status: Label
        self._gwstate: Label

    # ------------------------------------------------------------------ shell

    def compose(self) -> ComposeResult:
        """Header, gauges, sidebar with the views, status bar and footer."""
        yield Header()
        yield ResourceGauges(id="gauges")
        with Horizontal(id="shell"):
            yield ListView(*[ListItem(Label(label), id=f"nav-{name}") for name, label in VIEWS],
                           id="nav")
            with ContentSwitcher(id="main", initial="home"):
                for name, _ in VIEWS:
                    v = VIEW_CLASSES[name]()
                    v.id = name
                    self.views[name] = v
                    yield v
        with Horizontal(id="statusbar"):
            yield Label("", id="status")
            yield Label("", id="gwstate")
        yield Footer()

    def on_mount(self) -> None:
        """Start the pollers and the first scan; autostart the gateway when configured."""
        base = self.screen_stack[0]
        self._status = base.query_one("#status", Label)
        self._gwstate = base.query_one("#gwstate", Label)
        base.query_one("#nav", ListView).index = 0
        self.poll_host()
        self.set_interval(self.cfg.gauge_interval, self.poll_host)
        self.set_interval(SERVING_POLL, self._serving_tick)
        self.rescan()
        self.detect_build()
        self.refresh_upstream()
        self.refresh_serving()
        self.views["home"].on_activate()
        if self.cfg.gateway.autostart:
            if binary_found(self.cfg):
                self.start_gateway()
            else:
                self.fail(f"autostart: llama-server not found in {self.cfg.bin_dir}")
        if not self.cfg.onboarded:
            self.show_onboarding()
        else:
            self.maybe_ask_wsl_ram()

    def _alive(self) -> bool:
        """True while the app and its base screen can still be touched."""
        base = self.screen_stack[0] if self.screen_stack else None
        return not (self._closing or self._closed) and base is not None and base.is_running

    def alive(self) -> bool:
        """True while the app and its base screen can still be touched (not shutting down)."""
        return self._alive()

    @property
    def base(self) -> Widget:
        """The base screen holding the shell (``self.screen`` may be a modal on top of it)."""
        return self.screen_stack[0]

    @property
    def current_view(self) -> str:
        """Id of the visible view."""
        return self.base.query_one("#main", ContentSwitcher).current or "home"

    def switch_view(self, name: str) -> None:
        """Activate a view by id and sync the sidebar cursor.

        Args:
            name: One of the ids in `VIEWS`.
        """
        self.base.query_one("#main", ContentSwitcher).current = name
        nav = self.base.query_one("#nav", ListView)
        idx = [n for n, _ in VIEWS].index(name)
        if nav.index != idx:
            nav.index = idx
        self.sub_title = dict(VIEWS)[name]
        self.views[name].on_activate()

    def action_view(self, name: str) -> None:
        """Binding target: show a view."""
        self.switch_view(name)

    @on(ListView.Highlighted, "#nav")
    def _nav(self, event: ListView.Highlighted) -> None:
        """Sidebar cursor moved: show that view."""
        name = event.item.id.removeprefix("nav-") if event.item and event.item.id else ""
        if name in self.views and self.current_view != name:
            self.base.query_one("#main", ContentSwitcher).current = name
            self.sub_title = dict(VIEWS)[name]
            self.views[name].on_activate()

    @on(ListView.Selected, "#nav")
    def _nav_selected(self, event: ListView.Selected) -> None:
        """Enter on a sidebar item: show the view and move focus into it."""
        if event.item and event.item.id:     # Enter on a nav item also moves focus into the view
            self.switch_view(event.item.id.removeprefix("nav-"))

    def action_quit(self) -> None:
        """Quit; a running tune is told to stop after its current trial."""
        for flag in self.cancel_flags:
            flag.set()
        self.exit()

    # ------------------------------------------------------------------ threading helpers

    def ui(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
        """Run `fn(*args, **kwargs)` on the UI thread. Safe to call from any thread.

        Never blocks the caller and never raises, whatever state the app is in. The call is
        dropped when the app is closing, or when `fn` is a method of a widget or screen that
        is no longer in the DOM: a worker that finishes after its view went away must not
        touch it. (`call_from_thread` blocks until the callback ran and re-raises its error in
        the worker, so a poll finishing during shutdown crashed the app with `WorkerFailed`.)

        Args:
            fn: Callable to run on the UI thread.
            *args: Positional arguments for `fn`.
            **kwargs: Keyword arguments for `fn`.

        Returns:
            True when the call was queued, False when it was dropped.
        """
        try:
            return self.post_message(UiCall(partial(fn, *args, **kwargs),
                                            getattr(fn, "__self__", None)))
        except RuntimeError:        # the event loop is already closed
            return False

    def on_ui_call(self, message: UiCall) -> None:
        """Run a call queued by `ui` unless the app, its base screen or the call's owner is gone."""
        if not self._alive():
            return                  # shutting down: the base screen closes before the app does
        owner = message.owner
        if isinstance(owner, Widget) and not owner.is_attached:
            return
        message.call()

    def run_bg(self, fn: Callable[[], Any], *, label: str, group: str = "bg",
               exclusive: bool = False, done: Callable[[Any], None] | None = None,
               always: Callable[[], None] | None = None, quiet: bool = False,
               failed: Callable[[Exception], None] | None = None) -> Worker:
        """Run `fn` in a thread worker and bring its outcome back to the UI thread.

        A failure never escapes as a traceback: it becomes a status-bar message and, unless
        `quiet`, a notification. Results of a worker that was cancelled meanwhile (an
        exclusive group started again) are dropped.

        Args:
            fn: The blocking work; returns the value passed to `done`.
            label: What the work does, for error messages ("Loading qwen3").
            group: Worker group.
            exclusive: Cancel earlier workers of the group.
            done: Called on the UI thread with the return value when `fn` succeeded.
            always: Called on the UI thread when `fn` finished, however it ended.
            quiet: Report failures in the status bar only (for polls).
            failed: Called on the UI thread with the exception when `fn` raised (in addition to
                the message).

        Returns:
            The worker.
        """
        def guarded() -> None:
            worker = get_current_worker()
            broke = False
            result: Any = None
            try:
                result = fn()
            except Exception as e:  # noqa: BLE001 - the single place failures become messages
                broke = True
                self.ui(self._bg_failed, label, e, quiet)
                if failed is not None:
                    self.ui(failed, e)
            if not broke and done is not None and not worker.is_cancelled:
                self.ui(done, result)
            if always is not None:
                self.ui(always)

        return self.run_worker(guarded, name=label, group=group, thread=True, exclusive=exclusive,
                               exit_on_error=False)

    def _bg_failed(self, label: str, error: Exception, quiet: bool) -> None:
        """Show a worker failure (UI thread)."""
        what = str(error) if isinstance(error, ERRORS) else f"{type(error).__name__}: {error}"
        self.fail(f"{label}: {what}", notify=not quiet)

    # ------------------------------------------------------------------ messages

    def say(self, msg: str) -> None:
        """Show a one-line status message (plain text, no markup)."""
        if self._alive():
            self._status.update(Content(msg))

    def fail(self, msg: str, *, notify: bool = True) -> None:
        """Show an error in the status bar and, unless `notify` is off, as a notification.

        The same failure is not notified twice in a row.

        Args:
            msg: What went wrong, in words the user can act on.
            notify: Also raise a notification (polls pass False).
        """
        self.say(msg)
        if notify and msg != self._last_failure and self._alive():
            self.notify(msg, severity="error", timeout=10, markup=False)
        self._last_failure = msg

    # ------------------------------------------------------------------ host

    def poll_host(self) -> None:
        """Detect host resources off the UI thread and push to the gauges and every view."""
        def work() -> hardware.Host:
            bin_dir = self.cfg.bin_dir
            if self.host.ram_total:
                return hardware.refresh(self.host, bin_dir)
            return hardware.detect(bin_dir)

        self.run_bg(work, label="Reading hardware", group="host", exclusive=True,
                    done=self._host_update, quiet=True)

    def _host_update(self, host: hardware.Host) -> None:
        """A fresh host snapshot: update the gauges and tell every view."""
        self.host = host
        self.base.query_one("#gauges", ResourceGauges).update_host(host)
        for v in self.views.values():
            v.on_host(host)
        self.maybe_ask_wsl_ram()

    def maybe_ask_wsl_ram(self) -> None:
        """Ask once how much RAM Windows has, when running under WSL.

        Guarded by `wsl_windows_ram is None` (0 counts as answered), a known host RAM and
        `_wsl_asked` so the modal opens at most once. Called from on_mount and
        `_onboarding_done` per SPEC 12.5, and again from `_host_update` because the host
        snapshot is still empty while the first poll is running.
        """
        if (self._wsl_asked or self._wsl_checking or not self.cfg.onboarded
                or self.cfg.wsl_windows_ram is not None or not self.host.ram_total):
            return
        self._wsl_checking = True
        self.run_bg(network.wsl_mode, label="Checking WSL", group="wsl", exclusive=True,
                    done=self._wsl_checked, always=self._wsl_check_done, quiet=True)

    def _wsl_check_done(self) -> None:
        """The WSL check ended (however it ended): another one may run."""
        self._wsl_checking = False

    def _wsl_checked(self, mode: str | None) -> None:
        """WSL detection finished: push the question when it applies."""
        if (mode is None or self._wsl_asked or self.cfg.wsl_windows_ram is not None
                or not self.host.ram_total):
            return
        from .screens.wsl_ram import WslRamScreen
        self._wsl_asked = True
        self.push_screen(WslRamScreen(self.host), self._wsl_answered)

    def _wsl_answered(self, windows_ram: int | None) -> None:
        """Store the answer (bytes; 0 = skipped), save and refresh the Home hardware card."""
        self.cfg.wsl_windows_ram = windows_ram or 0
        try:
            self.cfg.save()
        except OSError as e:
            self.fail(f"cannot save the config: {e}")
        if "home" in self.views:
            self.views["home"].render_hardware()

    # ------------------------------------------------------------------ models

    def alias_of(self, path: str | Path) -> str:
        """The name clients use for a model file."""
        ent = self.cfg.models.get(str(path))
        if ent is not None and ent.alias:
            return ent.alias
        return library.aliases_for(self.models or [Path(path)], self.cfg).get(
            Path(path), Path(path).stem)

    def action_rescan(self) -> None:
        """Binding target: scan the model folders again."""
        self.rescan()

    def rescan(self) -> None:
        """Scan the model folders and read the GGUF headers, progressively, in a worker."""
        self._scan_gen += 1
        gen = self._scan_gen
        self.scanning = True
        dirs = list(self.cfg.model_dirs)

        def work() -> None:
            models = library.discover_models(dirs)
            self.ui(self._models_found, gen, models)
            for p in models:
                self.ui(self._info_read, gen, p, self.read_header(p))
            self.ui(self._scan_done, gen)

        self.run_bg(work, label="Scanning models", group="scan", exclusive=True,
                    always=partial(self._scan_ended, gen))

    def read_header(self, path: Path) -> ModelInfo | None:
        """The GGUF header of a model, from the cache or the file (blocking: call in a worker).

        Args:
            path: Model file.

        Returns:
            The metadata, or None when the file cannot be read.
        """
        info = self.info_cache.get(path)
        if info is None:
            try:
                info = read_model_info(path)
                self.info_cache.put(info)
            except (GGUFError, OSError):
                return None
        return info

    def _models_found(self, gen: int, models: list[Path]) -> None:
        """The folders were scanned: tell the views (ignored when a newer scan started)."""
        if gen != self._scan_gen:
            return
        self.models = models
        self.infos = {k: v for k, v in self.infos.items() if Path(k) in set(models)}
        for v in self.views.values():
            v.on_models(models)

    def _info_read(self, gen: int, path: Path, info: ModelInfo | None) -> None:
        """One header was read: tell the views (ignored when a newer scan started)."""
        if gen != self._scan_gen:
            return
        if info is not None:
            self.infos[str(path)] = info
        for v in self.views.values():
            v.on_info(path)

    def _scan_done(self, gen: int) -> None:
        """Every header was read: report the count and tell the views."""
        if gen != self._scan_gen:
            return
        self.scanning = False
        dirs = ", ".join(self.cfg.model_dirs) or "(no model folders)"
        self.say(f"{len(self.models)} model(s) in {dirs}")
        for v in self.views.values():
            v.on_scan_done()

    def _scan_ended(self, gen: int) -> None:
        """The scan worker ended: clear the scanning flag if it died early."""
        if gen == self._scan_gen and self.scanning:    # the worker died before finishing
            self.scanning = False

    def detect_build(self) -> None:
        """Ask llama-server for its version (once per call, in a worker)."""
        def work() -> str:
            return self._build_fn() if binary_found(self.cfg) else ""

        self.run_bg(work, label="Reading the llama.cpp version", group="build", exclusive=True,
                    done=self._build_known, quiet=True)

    def build_for(self, bin_dir: str) -> str:
        """The llama-server version of a llama.cpp folder (blocking: call in a worker).

        Args:
            bin_dir: Folder holding llama-server; the configured one uses the injected reader.

        Returns:
            The version string, "" when it cannot be determined.
        """
        if bin_dir == self.cfg.bin_dir:
            return self._build_fn() if binary_found(self.cfg) else ""
        return ServerRunner(Config(bin_dir=bin_dir)).build()

    def _build_known(self, build: str) -> None:
        """The llama-server version is known: views re-judge stale presets."""
        self.build = build
        for v in self.views.values():
            v.on_presets_changed(())

    # ------------------------------------------------------------------ architecture support

    def refresh_upstream(self) -> None:
        """Refresh the llama.cpp master architecture list (worker, then re-plan)."""
        self.run_bg(self.support.refresh_upstream, label="Checking llama.cpp architectures",
                    group="upstream", exclusive=True, done=self._upstream_ready, quiet=True)

    def _upstream_ready(self, _ok: bool) -> None:
        """The upstream list is current (or unreachable): re-plan and refresh views."""
        for v in self.views.values():
            v.on_config_changed()

    def blocked_map(self) -> dict[str, str]:
        """Model path -> reason the build cannot load it, for the gateway's preset INI.

        Returns:
            A map with an entry per scanned model whose architecture support check blocks it.
        """
        out: dict[str, str] = {}
        for path, info in self.infos.items():
            s = self.support.check(info.architecture, self.build)
            if s.blocks:
                out[path] = s.reason()
        return out

    def _learn_log(self, log: Path) -> None:
        """Record an ``unknown model architecture`` line of a failed load's log.

        Args:
            log: The failed server's log file.
        """
        try:
            text = log.read_text(errors="replace")
        except OSError:
            return
        self.support.learn(text, self.build)

    # ------------------------------------------------------------------ serving state

    def _serving_tick(self) -> None:
        """Timer: poll the serving state; only every few ticks while nothing runs."""
        self._poll_n += 1
        idle = self.snapshot is not None and self.snapshot.idle and not self.gw_busy
        if idle and self._poll_n % IDLE_EVERY:
            return
        self.refresh_serving()

    def refresh_serving(self) -> None:
        """Gather the gateway and session state now (worker); posts one `ServingChanged`."""
        def work() -> None:
            estimate = usage_mod.estimator(self.cfg, self.host, self.calibration,
                                           self.infos.get)
            snap = gather(self.gateway, self.store, self.health,
                          usage=lambda st, models, sessions: usage_mod.served_usage(
                              self.probe, st, models, sessions, estimate))
            self.post_message(ServingChanged(snap))

        self.run_bg(work, label="Checking the gateway", group="serving", exclusive=True,
                    quiet=True)

    def on_serving_changed(self, message: ServingChanged) -> None:
        """Store the new snapshot and hand it to every view."""
        if not self._alive():
            return
        snap, old = message.snapshot, self.snapshot
        if snap.idle and old is not None and old.idle and snap.same_as(old) and not self.gw_busy:
            return                  # nothing running and nothing changed: nothing to redraw
        self.snapshot = snap
        self._show_gateway_state()
        for v in self.views.values():
            v.on_serving(snap)

    def gateway_state(self) -> str:
        """The gateway's state in the words the Home card uses."""
        return gateway_state(self.snapshot, self.gw_busy)

    def gateway_host(self) -> str:
        """The address the gateway listens on (running), else the one it would start with."""
        snap = self.snapshot
        return (snap.status.host if snap else "") or self.cfg.gateway.host

    def gateway_urls(self) -> list[str]:
        """Base URLs a client can use for the gateway, running or planned.

        Returns:
            Loopback first, then one URL per LAN address when it listens on every interface.
        """
        snap = self.snapshot
        return network.client_urls_for(snap.status.url if snap else self.gateway.listen_url,
                                       self.gateway_host())

    def gateway_url(self) -> str:
        """The gateway URL to hand out: the first LAN URL when exposed, else loopback."""
        return network.primary_url(self.gateway_urls())

    def warn_exposed(self, host: str, api_key: str) -> None:
        """Notify when a server others can reach has no API key.

        Args:
            host: Listen address.
            api_key: The key clients must send, "" for none.
        """
        if warning := network.key_warning(host, api_key):
            self.notify(warning, title="No API key", severity="warning", markup=False)

    def _show_gateway_state(self) -> None:
        """Right end of the status bar: gateway state and what is loaded."""
        snap = self.snapshot
        text = f"gateway {self.gateway_state()}"
        if snap is not None and snap.running_models:
            text += f" - {len(snap.running_models)} loaded"
        self._gwstate.update(Content(text))

    def _set_busy(self, state: str) -> None:
        """Mark the gateway as starting/stopping (or "" when done) and redraw."""
        self.gw_busy = state
        self._show_gateway_state()
        if self.snapshot is not None:
            for v in self.views.values():
                v.on_serving(self.snapshot)

    # ------------------------------------------------------------------ gateway flows

    def start_gateway(self, finished: Callable[[str], None] | None = None) -> None:
        """Start the gateway in a worker and wait until it answers ``/health``.

        Args:
            finished: Called on the UI thread when the attempt ended, with "" when the gateway
                runs and the reason when it could not be started.
        """
        if self.gw_busy:
            return
        if not binary_found(self.cfg):
            if finished is not None:
                finished(f"llama-server not found in {self.cfg.bin_dir}")
            return self.fail(f"llama-server not found in {self.cfg.bin_dir}; set the llama.cpp "
                             "folder in Settings")
        self._set_busy(GW_STARTING)
        self.say("Starting the gateway...")

        def work() -> tuple[bool, str, str]:
            with self._gw_lock:
                st, started = actions.ensure_gateway(self.gateway, learn=self._learn_log)
            host = st.host or self.cfg.gateway.host
            return started, host, ", ".join(network.client_urls_for(st.url, host))

        def done(result: tuple[bool, str, str]) -> None:
            started, host, urls = result
            self.say(f"Gateway running at {urls}" if started else f"Gateway already running at {urls}")
            if self.gateway.problems:
                self.say("Gateway running; " + "; ".join(self.gateway.problems))
            self.warn_exposed(host, self.cfg.gateway.api_key)
            if finished is not None:
                finished("")

        self.run_bg(work, label="Starting the gateway", group="gateway", done=done,
                    always=self._gateway_op_ended,
                    failed=(lambda e: finished(str(e))) if finished is not None else None)

    def stop_gateway(self) -> None:
        """Stop the gateway (and the models it runs) in a worker."""
        if self.gw_busy:
            return
        self._set_busy(GW_STOPPING)
        self.say("Stopping the gateway...")

        def work() -> bool:
            with self._gw_lock:
                return self.gateway.stop()

        self.run_bg(work, label="Stopping the gateway", group="gateway",
                    done=lambda was: self.say("Gateway stopped" if was
                                              else "The gateway was not running"),
                    always=self._gateway_op_ended)

    def _gateway_op_ended(self) -> None:
        """A gateway start or stop finished: clear the busy flag and poll."""
        self._set_busy("")
        self.refresh_serving()

    def gateway_load(self, name: str) -> None:
        """Ask the gateway to load a model and watch it come up.

        Args:
            name: Model id (``alias`` or ``alias@preset``).
        """
        size = self._size_of_served(name)

        def work() -> str:
            self.ui(self.say, f"Loading {name}...")
            self.gateway.load(name)
            self.ui(self.refresh_serving)
            t0 = time.monotonic()
            actions.wait_loaded(
                self.gateway, name, tuner.health_timeout(size) if size else 300.0,
                tick=lambda: self.ui(self.say, f"Loading {name}... {time.monotonic() - t0:.0f} s"),
                learn=self._learn_log)
            return name

        self.run_bg(work, label=f"Loading {name}", group="load", done=self._loaded,
                    always=self.refresh_serving)

    def gateway_unload(self, name: str) -> None:
        """Ask the gateway to unload a model and wait until it is gone.

        Args:
            name: Model id.
        """
        def work() -> bool:
            self.gateway.unload(name)
            return actions.wait_unloaded(self.gateway, name, UNLOAD_WAIT)

        self.run_bg(work, label=f"Unloading {name}", group="load",
                    done=lambda gone: self.say(f"{name} unloaded" if gone else
                                               f"{name}: unload requested; still listed as running"),
                    always=self.refresh_serving)

    def _loaded(self, name: str) -> None:
        """A model came up: say so in the status bar and as a notification."""
        self.say(f"{name} is loaded at {self.gateway_url()}/v1")
        self.notify(f"{name} is loaded", title="Model ready", markup=False)

    def _size_of_served(self, name: str) -> int:
        """File size of the model behind a served name, 0 when unknown."""
        alias = name.split("@", 1)[0]
        for path, ent in self.cfg.models.items():
            if ent.alias == alias:
                try:
                    return Path(path).stat().st_size
                except OSError:
                    return 0
        return 0

    # ------------------------------------------------------------------ serving a model

    def serve_model(self, path: str, intent_key: str | None = None, *,
                    standalone: bool = False) -> None:
        """Make sure a model has a preset, then load it in the gateway or start it standalone.

        Without `intent_key` the model's active preset is served as it is (a model with none
        gets one for the default intent). With one, that intent's preset is served, planned
        and stored as ``estimated`` when missing; it only becomes the active preset when the
        model had none, so it is then served as ``alias@intent``.

        Args:
            path: Model path.
            intent_key: Intent to serve, or None for the active preset.
            standalone: Run a llama-server of its own instead of using the gateway.
        """
        info = self.infos.get(path)
        if info is None:
            return self.say("Still reading the model header - try again in a moment.")
        if not binary_found(self.cfg):
            return self.fail(f"llama-server not found in {self.cfg.bin_dir}; set the llama.cpp "
                             "folder in Settings")
        if (s := self.support.check(info.architecture, self.build)).blocks:
            return self.fail(f"cannot serve {Path(path).stem}: {s.reason()}")
        try:
            chosen = actions.pick_preset(
                self.cfg, info, self.host, self.calibration, intent_key,
                activate=self.cfg.preset(path) is None, save=False)
        except LookupError as e:
            return self.fail(str(e))
        if chosen.created:
            self.presets_changed(path, reload=standalone)
        name = actions.served_name(self.cfg, path, chosen.name)
        timeout = tuner.health_timeout(info.size)
        if standalone:
            self.say(f"Starting {Path(path).stem} on its own port...")
            self._serve_standalone(path, chosen.name, timeout)
            return
        self.say(f"Serving {name}...")
        self.views["serving"].focus_model(name)         # type: ignore[attr-defined]
        self.switch_view("serving")
        self._serve_gateway(path, chosen.name, timeout)

    def _serve_gateway(self, path: str, preset: str, timeout: float) -> None:
        """Load a preset in the gateway in a worker and wait until it is loaded."""
        fresh = not self.gateway.running

        def loaded(name: str) -> None:
            self._loaded(name)
            if fresh:                       # this serve started the gateway: say if it is open
                self.warn_exposed(self.gateway_host(), self.cfg.gateway.api_key)

        def work() -> str:
            with self._gw_lock:
                name = actions.serve_gateway(
                    self.cfg, self.gateway, path, preset, learn=self._learn_log,
                    tick=lambda: self.ui(self.say, "Starting the gateway..."))
            self.ui(self.refresh_serving)
            t0 = time.monotonic()
            actions.wait_loaded(self.gateway, name, timeout, learn=self._learn_log,
                                tick=lambda: self.ui(
                                    self.say, f"Loading {name}... {time.monotonic() - t0:.0f} s"))
            return name

        self.run_bg(work, label=f"Serving {self.alias_of(path)}", group="serve",
                    done=loaded, always=self.refresh_serving)

    def _serve_standalone(self, path: str, preset: str, timeout: float) -> None:
        """Start a standalone server in a worker and wait until it answers."""
        def work() -> tuple[str, Session]:
            try:
                session = self.store.launch_server(self.cfg, path, preset)
            except RuntimeError as e:
                raise ActionError(str(e)) from None
            self.ui(self.refresh_serving)
            t0 = time.monotonic()
            actions.wait_server_ready(session, self.health, timeout, learn=self._learn_log,
                                      tick=lambda: self.ui(
                                          self.say, f"Loading {session.name}... "
                                          f"{time.monotonic() - t0:.0f} s"))
            return f"{session.name} is ready at {network.primary_url(session.urls)}", session

        def done(result: tuple[str, Session]) -> None:
            msg, session = result
            self.say(msg)
            self.notify(msg, markup=False)
            preset_obj = self.cfg.preset(path, preset)
            prof = preset_obj.profile if preset_obj else self.cfg.profile_for(path)
            self.warn_exposed(session.host or prof.host, prof.api_key)

        self.run_bg(work, label=f"Starting {Path(path).stem}", group="serve", done=done,
                    always=self.refresh_serving)

    # ------------------------------------------------------------------ presets

    def presets_changed(self, *paths: str, save: bool = True, reload: bool = True) -> None:
        """The one thing to call after any preset change (save, tune, delete, set active).

        Saves the config, rewrites the gateway's preset file and has a running gateway re-read
        it (in a worker), and refreshes the rows of the affected models in every view.

        Args:
            *paths: Model paths whose presets changed.
            save: Write the config (off when the caller just did).
            reload: Rewrite the preset file and reload the gateway (off when the caller just did).
        """
        if save:
            try:
                self.cfg.save()
            except OSError as e:
                self.fail(f"cannot save the config: {e}")
        for v in self.views.values():
            v.on_presets_changed(paths)
        if reload:
            def work() -> int:
                with self._gw_lock:
                    return len(self.gateway.reload())

            self.run_bg(work, label="Updating the gateway presets", group="reload",
                        exclusive=True, done=self._presets_written, always=self.refresh_serving)

    def _presets_written(self, _count: int) -> None:
        """The gateway re-read the presets: surface presets it could not take."""
        if self.gateway.problems:
            self.fail("some presets cannot go in the gateway: " + "; ".join(self.gateway.problems))
        if self.snapshot is not None:        # the Home card lists the gateway's problems
            for v in self.views.values():
                v.on_serving(self.snapshot)

    def config_changed(self) -> None:
        """Tell every view that the settings were saved."""
        for v in self.views.values():
            v.on_config_changed()

    def activate_preset(self, path: str, name: str) -> bool:
        """Make a preset the one served as plain ``alias``.

        Args:
            path: Model path.
            name: Preset name.

        Returns:
            False (with a message) when the model has no such preset.
        """
        ent = self.cfg.models.get(path)
        if ent is None or name not in ent.presets:
            self.fail(f"{self.alias_of(path)} has no '{name}' preset")
            return False
        ent.active = name
        self.presets_changed(path)
        return True

    # ------------------------------------------------------------------ screens and suggestions

    def open_tune(self, path: str, intent_key: str, *, goals: TuningGoals | None = None) -> None:
        """Open the tune screen for a model and an intent.

        Args:
            path: Model path.
            intent_key: Intent to tune for.
            goals: Unsaved preset goals to use for this candidate search.
        """
        from .screens.tune import TuneScreen

        info = self.infos.get(path)
        if info is None:
            return self.say("Still reading the model header - try again in a moment.")
        if not binary_found(self.cfg):
            return self.fail(f"llama-server not found in {self.cfg.bin_dir}; set the llama.cpp "
                             "folder in Settings")
        if (s := self.support.check(info.architecture, self.build)).blocks:
            return self.fail(f"cannot tune {Path(path).stem}: {s.reason()}")
        try:
            intent = actions.choose_intent(intent_key, self.cfg.default_intent, info)
        except LookupError as e:
            return self.fail(str(e))
        self.push_screen(TuneScreen(path, info, intent, goals=goals))

    def offer_new_model(self, path: Path) -> None:
        """A model file just arrived (a download): ask what it is for, store the preset, then offer next steps.

        Opens the `IntentPicker` for the model. Choosing an intent stores the estimated preset
        (active only when the model had none) and offers Serve / Tune / Done; cancelling the
        picker leaves the model without a preset (it shows as ``New``). Header reading and
        planning run in a worker.

        Args:
            path: The model file (first shard when split).
        """
        key = str(path)
        host, cfg, cal = self.host, self.cfg, self.calibration

        def work() -> tuple[ModelInfo, dict]:
            info = self.infos.get(key) or self.read_header(path)
            if info is None:
                raise ActionError(f"cannot read the GGUF header of {path.name}")
            return info, planner.plan_all(info, host, cfg.default_profile, cfg.headroom_pct, cal,
                                          support=self.support.check(info.architecture,
                                                                     self.build))

        self.run_bg(work, label=f"Planning {path.stem}", group="new-model",
                    done=lambda result: self._pick_for_new_model(path, *result))

    def _pick_for_new_model(self, path: Path, info: ModelInfo, plans: dict) -> None:
        """Plans are ready: show the intent picker for the new model (UI thread)."""
        from .modelrows import default_intent_key, intent_rows
        from .screens.after_download import DONE, SERVE, TUNE, AfterDownload
        from .screens.intent_picker import IntentPicker

        key = str(path)
        self.infos[key] = info
        ent = self.cfg.models.get(key)
        rows = intent_rows(plans, info, ent, path, self.host, self.cfg.headroom_pct,
                           self.calibration, self.build)
        alias = self.alias_of(path)

        def after(choice: str | None) -> None:
            if choice not in (SERVE, TUNE, DONE):
                return
            if choice == SERVE:
                self.serve_model(key, intent)
            elif choice == TUNE:
                self.open_tune(key, intent)

        def picked(intent_key: str | None) -> None:
            nonlocal intent
            if not intent_key:
                return self.say(f"{alias} is in your library without a preset (it shows as New).")
            intent = intent_key
            try:
                chosen = actions.pick_preset(self.cfg, info, self.host, self.calibration,
                                             intent_key, activate=self.cfg.preset(key) is None,
                                             save=False)
            except LookupError as e:
                return self.fail(str(e))
            self.presets_changed(key)
            how = "planned just now" if chosen.created else "already there"
            self.push_screen(AfterDownload(
                f"{alias} is ready: {chosen.name} preset ({how}, estimated)."), after)

        intent = ""
        self.push_screen(IntentPicker(
            alias, rows, default_intent_key(ent, plans, self.cfg.default_intent)), picked)

    def show_onboarding(self) -> None:
        """Open the guided setup (first run, or "Run setup again" in Settings)."""
        from .screens.onboarding import OnboardingScreen

        self.push_screen(OnboardingScreen(self.which), self._onboarding_done)

    def _onboarding_done(self, how: Dismissal | None) -> None:
        """The setup closed: land where it said (Discover, the tune screen) or stay put."""
        self.maybe_ask_wsl_ram()
        if how is None:
            return
        if how.action == onboarding.ACT_DISCOVER:
            self.switch_view("discover")
        elif how.action == onboarding.ACT_TUNE and how.model:
            self.open_tune(how.model, how.intent)

    def setup_applied(self, res: FinishResult) -> None:
        """The setup applied its draft to the config (already saved): refresh what depends on it.

        Args:
            res: What `onboarding.finish` did.
        """
        if res.bin_changed:
            self.detect_build()
            self.poll_host()
        if res.dirs_changed:
            self.rescan()
        self.config_changed()
        if res.model:
            self.presets_changed(res.model, save=False)

    def run_suggestion(self, s: Suggestion) -> None:
        """Do what a "next step" on the Home view asks for.

        Args:
            s: The chosen suggestion.
        """
        if s.action == ACT_SETTINGS:
            self.switch_view("settings")
        elif s.action == ACT_DISCOVER:
            self.switch_view("discover")
        elif s.action == ACT_MODELS:
            self.switch_view("models")
            if s.model:
                self.views["models"].select(s.model)           # type: ignore[attr-defined]
        elif s.action == ACT_TUNE:
            self.open_tune(s.model, s.preset)
        elif s.action == ACT_START_GATEWAY:
            self.start_gateway()


def main() -> None:
    """Entry point."""
    LlamaCtl().run()


if __name__ == "__main__":
    main()
