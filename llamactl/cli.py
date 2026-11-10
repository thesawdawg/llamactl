"""Command line interface: ``llamactl`` with no arguments opens the TUI, subcommands run headless.

The module is split three ways. `report` turns core objects into plain dicts, `render` turns
those into aligned text, and this file parses arguments and runs the commands. Every command is
a short ``cmd_x(args, services) -> int`` that gets everything it touches (config, host, models,
gateway, session store, tune runner, output streams, clock) from one `Services` object, built
once in `main` and replaceable wholesale in tests. Textual is imported only when the TUI is
launched, so headless commands never load it.

Exit codes: 0 success, 1 runtime failure, 2 usage error (including unknown or ambiguous models).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

import httpx

from . import actions, gateway as gateway_mod, hardware, intents, library, render, report, tuner
from .actions import ActionError
from .calibration import TUNED, Calibration
from .config import CONFIG_FILE, PRESET_SEP, Config
from .gateway import Gateway, GatewayError
from .gguf import GGUFError, ModelInfo, ModelInfoCache, read_model_info
from .hardware import Host
from .render import Style
from .sessions import Session, SessionStore
from .tuner import Runner, ServerRunner

EXIT_OK, EXIT_FAILURE, EXIT_USAGE = 0, 1, 2
DEFAULT_TAIL = 40
FOLLOW_INTERVAL = 0.5
HEALTH_PROBE_TIMEOUT = 1.0
UNLOAD_TIMEOUT = 30.0            # seconds to wait for a model to leave memory
DEFAULT_LOAD_TIMEOUT = 300.0     # seconds to wait for a model whose file size is unknown
STATE_READY, STATE_LOADING, STATE_DOWN = "ready", "loading", "down"
GATEWAY_LOG = "gateway"          # the NAME `logs` uses for the gateway's own log


class UsageError(Exception):
    """The command line asks for something that cannot be done (exit code 2)."""


class CliError(Exception):
    """A command failed at run time (exit code 1); the message is for the user."""


# --------------------------------------------------------------------------- services

def http_health(url: str, timeout: float = HEALTH_PROBE_TIMEOUT) -> str:
    """State of a llama-server from its ``/health`` endpoint.

    Args:
        url: Base URL of the server.
        timeout: Seconds before giving up.

    Returns:
        "ready" for HTTP 200, "loading" for 503 (model still loading), "down" otherwise.
    """
    try:
        code = httpx.get(f"{url.rstrip('/')}/health", timeout=timeout).status_code
    except httpx.HTTPError:
        return STATE_DOWN
    return STATE_READY if code == 200 else STATE_LOADING if code == 503 else STATE_DOWN


class Services:
    """Everything a command touches, created once in `main` and swappable in tests.

    Expensive or side-effecting parts (host detection, the session store, the calibration file,
    the GGUF cache) are created on first use. Every argument after `cfg` is optional; leaving
    it out selects the real implementation.

    Args:
        cfg: Loaded config.
        out: Stream for results (default ``sys.stdout``).
        err: Stream for progress and messages (default ``sys.stderr``).
        colour: Emit ANSI colour on `out`; None decides from `out` (a TTY, no ``NO_COLOR``).
        read_only: Never write state: the GGUF cache, ``sessions.json`` and ``gateway.json``
            are only read.
        detect: Returns the host snapshot.
        refresh: Returns a host snapshot with current free memory, given the earlier one.
        cache: GGUF header cache.
        calibration: Measured trials.
        gateway: The router-mode gateway.
        store: Standalone and attached sessions.
        runner_factory: Returns the `Runner` a tune measures with.
        build: Returns the llama-server version string.
        health: Returns "ready", "loading" or "down" for a server URL.
        sleep: Sleep function.
        clock: Monotonic clock in seconds.
        now: Wall clock in Unix seconds.
    """

    def __init__(self, cfg: Config, *, out: TextIO | None = None, err: TextIO | None = None,
                 colour: bool | None = None, read_only: bool = False,
                 detect: Callable[[], Host] | None = None,
                 refresh: Callable[[Host], Host] | None = None,
                 cache: ModelInfoCache | None = None, calibration: Calibration | None = None,
                 gateway: Gateway | None = None, store: SessionStore | None = None,
                 runner_factory: Callable[[], Runner] | None = None,
                 build: Callable[[], str] | None = None,
                 health: Callable[[str], str] = http_health,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 now: Callable[[], float] = time.time) -> None:
        self.cfg = cfg
        self.out = out or sys.stdout
        self.err = err or sys.stderr
        self.style = Style(render.use_colour(self.out) if colour is None else colour)
        self.read_only = read_only
        self.health = health
        self.sleep = sleep
        self.clock = clock
        self.now = now
        self._detect = detect or (lambda: hardware.detect(cfg.bin_dir))
        self._refresh = refresh or (lambda h: hardware.refresh(h, cfg.bin_dir))
        self._build_fn = build or (lambda: ServerRunner(cfg).build())
        self._runner_factory = runner_factory or (lambda: ServerRunner(cfg))
        self._cache, self._calibration = cache, calibration
        self._gateway, self._store = gateway, store
        self._host: Host | None = None
        self._models: list[Path] | None = None
        self._build: str | None = None

    # -- lazily created parts ----------------------------------------------------------

    @property
    def cache(self) -> ModelInfoCache:
        """The GGUF header cache (never written to in read-only mode)."""
        if self._cache is None:
            self._cache = ModelInfoCache(writable=not self.read_only)
        return self._cache

    @property
    def calibration(self) -> Calibration:
        """Measured trials from ``tuning.json``."""
        if self._calibration is None:
            self._calibration = Calibration.load()
        return self._calibration

    @property
    def gateway(self) -> Gateway:
        """The gateway."""
        if self._gateway is None:
            self._gateway = Gateway(self.cfg, read_only=self.read_only)
        return self._gateway

    @property
    def store(self) -> SessionStore:
        """The standalone and attached sessions (not rewritten in read-only mode)."""
        if self._store is None:
            self._store = SessionStore(persist=not self.read_only)
        return self._store

    def runner(self) -> Runner:
        """A new tune runner."""
        return self._runner_factory()

    # -- hardware and models -----------------------------------------------------------

    def host(self) -> Host:
        """The host snapshot, detected once."""
        if self._host is None:
            self._host = self._detect()
        return self._host

    def fresh_host(self) -> Host:
        """The host with its free memory read again (before a tune or a serve)."""
        self._host = self._refresh(self.host())
        return self._host

    def models(self) -> list[Path]:
        """Model files in the configured directories, scanned once."""
        if self._models is None:
            self._models = library.discover_models(self.cfg.model_dirs)
        return self._models

    def aliases(self) -> dict[Path, str]:
        """Alias of every model."""
        return library.aliases_for(self.models(), self.cfg)

    def resolve(self, query: str) -> Path:
        """The model a user named.

        Args:
            query: Alias, file stem or unique substring.

        Returns:
            The model path.

        Raises:
            UsageError: When nothing matches or several models do.
        """
        try:
            return library.resolve_model(query, self.models(), self.cfg)
        except LookupError as e:
            raise UsageError(str(e)) from None

    def info(self, path: Path) -> ModelInfo:
        """GGUF metadata of a model, from the cache when current.

        Args:
            path: Model file.

        Returns:
            The metadata.

        Raises:
            GGUFError: When the header cannot be read.
        """
        info = self.cache.get(path)
        if info is None:
            info = read_model_info(path)
            self.cache.put(info)
        return info

    # -- llama.cpp ---------------------------------------------------------------------

    def binary_found(self) -> bool:
        """True when ``llama-server`` exists in the configured folder."""
        return (Path(self.cfg.bin_dir) / "llama-server").is_file()

    def require_binary(self) -> None:
        """Fail with one actionable line when ``llama-server`` is missing.

        Raises:
            CliError: When ``bin_dir`` does not hold ``llama-server``.
        """
        if not self.binary_found():
            raise CliError(f"llama-server not found in {self.cfg.bin_dir}; set \"bin_dir\" to "
                           f"your llama.cpp build folder in {CONFIG_FILE}")

    def build(self) -> str:
        """The llama-server version string, asked once ("" when unknown)."""
        if self._build is None:
            self._build = self._build_fn() if self.binary_found() else ""
        return self._build

    # -- output ------------------------------------------------------------------------

    def say(self, text: str = "") -> None:
        """Print a line to the result stream."""
        self.out.write(text + "\n")

    def note(self, text: str = "") -> None:
        """Print a line to the message stream (stderr)."""
        self.err.write(text + "\n")
        self.err.flush()

    def emit(self, as_json: bool, data: dict, renderer: Callable[[dict, Style], str]) -> None:
        """Print one result as a JSON document or as text.

        Args:
            as_json: Print the dict as JSON, and nothing else, on the result stream.
            data: The result.
            renderer: Turns `data` into text.
        """
        self.say(json.dumps(data, indent=2) if as_json else renderer(data, self.style))


class Waiting:
    """Progress dots on stderr while something is awaited; ends the line on exit.

    Args:
        sv: Services (the message stream).
        label: Text printed before the dots.
    """

    def __init__(self, sv: Services, label: str) -> None:
        self.sv, self.label, self.dots = sv, label, 0

    def tick(self) -> None:
        """Print one dot."""
        if self.dots == 0:
            self.sv.err.write(self.label + " ")
        self.sv.err.write(".")
        self.sv.err.flush()
        self.dots += 1

    def __enter__(self) -> "Waiting":
        return self

    def __exit__(self, *exc: object) -> None:
        if self.dots:
            self.sv.err.write("\n")
            self.sv.err.flush()


# --------------------------------------------------------------------------- helpers

def _model_and_info(args: argparse.Namespace, sv: Services) -> tuple[Path, str, ModelInfo]:
    """Resolve the MODEL argument and read its header.

    Raises:
        UsageError: Unknown or ambiguous model.
        CliError: Unreadable GGUF file.
    """
    path = sv.resolve(args.model)
    try:
        info = sv.info(path)
    except GGUFError as e:
        raise CliError(f"cannot read {path.name}: {e}") from None
    return path, sv.aliases()[path], info


def _intent(args: argparse.Namespace, sv: Services, info: ModelInfo) -> intents.Intent:
    """The intent of ``--for``, else the configured default that fits the model.

    Raises:
        UsageError: When ``--for`` names an intent that does not apply to the model.
    """
    try:
        return actions.choose_intent(args.intent, sv.cfg.default_intent, info)
    except LookupError as e:
        raise UsageError(str(e)) from None


def _match(query: str, names: Sequence[str], what: str) -> str:
    """The one name `query` means: exact, else a unique case-insensitive substring.

    Raises:
        UsageError: When nothing or several names match.
    """
    if query in names:
        return query
    hits = [n for n in names if query.lower() in n.lower()]
    if len(hits) == 1:
        return hits[0]
    shown = ", ".join(sorted(hits or names)) or "none"
    raise UsageError(f"{'ambiguous' if hits else 'unknown'} {what} '{query}' "
                     f"({'matches' if hits else 'known'}: {shown})")


def _find_session(query: str, sessions: Sequence[Session]) -> Session:
    """The session with this id or name (exact first, then a unique substring).

    Raises:
        UsageError: When nothing or several sessions match.
    """
    for key in (lambda s: s.id, lambda s: s.name):
        hits = [s for s in sessions if key(s) == query]
        if len(hits) == 1:
            return hits[0]
        if hits:
            raise UsageError(f"'{query}' names several servers: "
                             f"{', '.join(s.id for s in hits)}; use the id")
    pick = _match(query, [s.id for s in sessions], "server")
    return next(s for s in sessions if s.id == pick)


def _model_size(sv: Services, served: str) -> int:
    """File size of the model behind a gateway model id (``alias`` or ``alias@preset``), 0 if unknown."""
    alias = served.split(PRESET_SEP, 1)[0]
    for path, ent in sv.cfg.models.items():
        if ent.alias == alias:
            try:
                return Path(path).stat().st_size
            except OSError:
                return 0
    return 0


def _load_timeout(size: int) -> float:
    """Seconds to wait for a model of `size` bytes to load (the tuner's health timeout)."""
    return tuner.health_timeout(size) if size else DEFAULT_LOAD_TIMEOUT


def _serve_hint(sv: Services, base_url: str, model: str | None, info: ModelInfo,
                api_key: bool) -> None:
    """Print the endpoint, the model name to request and a curl example."""
    sv.say(render.kv([
        ("base URL", f"{base_url.rstrip('/')}/v1  (OpenAI compatible)"),
        ("model", model or "(any; one model per server)"),
        ("try", actions.curl_snippet(base_url, model, api_key=api_key,
                                     embedding=info.is_embedding)),
    ]))


# --------------------------------------------------------------------------- commands

def cmd_hw(args: argparse.Namespace, sv: Services) -> int:
    """``llamactl hw``: devices, RAM, CPU, calibration factors and the llama.cpp build."""
    found = sv.binary_found()
    data = report.hw_data(sv.host(), sv.calibration, sv.cfg.bin_dir, found,
                          sv.build() if found else "")
    sv.emit(args.json, data, render.render_hw)
    return EXIT_OK


def cmd_ls(args: argparse.Namespace, sv: Services) -> int:
    """``llamactl ls``: every model with size, quant, plan per intent and presets."""
    cfg, host = sv.cfg, sv.host()
    tuned = any(p.source == TUNED for e in cfg.models.values() for p in e.presets.values())
    build = sv.build() if tuned else ""
    aliases, rows = sv.aliases(), []
    for path in sorted(sv.models(), key=lambda p: aliases[p]):
        try:
            info, error = sv.info(path), ""
        except GGUFError as e:
            info, error = None, str(e)
        rows.append(report.model_row(path, aliases[path], info, error, cfg, host,
                                     sv.calibration, build))
    if args.json:
        sv.emit(True, {"models": rows}, lambda d, s: "")
    else:
        sv.say(render.render_ls(rows, sv.style))
    return EXIT_OK


def cmd_plan(args: argparse.Namespace, sv: Services) -> int:
    """``llamactl plan MODEL``: the plan, its reasons and the exact command; ``--save`` stores it."""
    if args.all and args.save:
        raise UsageError("--save stores one intent's plan; combine it with --for, not --all")
    path, alias, info = _model_and_info(args, sv)
    cfg, host = sv.cfg, sv.host()
    if args.all:
        data = report.plan_all_data(cfg, path, alias, info, host, sv.calibration)
        sv.emit(args.json, data, render.render_plan_all)
        return EXIT_OK
    intent = _intent(args, sv, info)
    plan = actions.plan_for(cfg, info, host, intent, sv.calibration)
    sv.emit(args.json, report.plan_data(cfg, path, alias, info, host, plan, intent),
            render.render_plan)
    if args.save:
        actions.store_plan(cfg, str(path), plan, overwrite=args.force)
        sv.note(f"saved as the '{intent.key}' preset of {cfg.models[str(path)].alias} (estimated)")
    return EXIT_OK


def cmd_tune(args: argparse.Namespace, sv: Services) -> int:
    """``llamactl tune MODEL``: measured search, saves the tuned preset and reloads the gateway."""
    path, alias, info = _model_and_info(args, sv)
    intent = _intent(args, sv, info)
    sv.require_binary()
    host = sv.fresh_host()
    sv.note(f"tuning {alias} for {intent.key}: up to {args.trials} trials, each launches "
            "llama-server (Ctrl-C cancels)")
    try:
        outcome = actions.run_tune(sv.cfg, info, host, intent, sv.runner(), sv.calibration,
                                   sv.gateway, trials=args.trials, force=args.force,
                                   progress=lambda e: sv.note(report.event_line(e)))
    except ActionError as e:
        raise CliError(f"cannot tune {alias}: {e}; --force tries anyway") from None
    if outcome.reload_error:
        sv.note(f"warning: the gateway was not reloaded: {outcome.reload_error}")
    backend = host.gpus[0].backend if host.gpus else "cpu"
    result = outcome.result
    sv.emit(args.json, report.tune_data(alias, result, outcome.reloaded,
                                        sv.calibration.factor(backend)), render.render_tune)
    return EXIT_OK if result.ok and not result.cancelled else EXIT_FAILURE


def cmd_serve(args: argparse.Namespace, sv: Services) -> int:
    """``llamactl serve MODEL``: ensure a preset, then load it in the gateway or run standalone."""
    path, alias, info = _model_and_info(args, sv)
    if args.intent is not None:
        _intent(args, sv, info)
    if args.port is not None and not args.standalone:
        raise UsageError("--port applies to --standalone; the gateway port is a gateway setting")
    sv.require_binary()
    cfg = sv.cfg
    try:
        chosen = actions.pick_preset(cfg, info, sv.fresh_host(), sv.calibration, args.intent,
                                     activate=not args.no_activate)
    except LookupError as e:
        raise UsageError(str(e)) from None
    how = "planned just now" if chosen.created else "existing"
    sv.say(f"preset '{chosen.name}' ({chosen.preset.source}, {how})")
    if chosen.created:
        sv.say("  an estimate: run `llamactl tune` to measure it on this machine")
    timeout = _load_timeout(info.size)
    if args.standalone:
        return _serve_standalone(args, sv, path, chosen.name, info, timeout)
    with Waiting(sv, "starting the gateway") as w:
        name = actions.serve_gateway(cfg, sv.gateway, str(path), chosen.name, clock=sv.clock,
                                     sleep=sv.sleep, tick=w.tick)
    if not args.no_wait:
        with Waiting(sv, f"loading {name}") as w:
            actions.wait_loaded(sv.gateway, name, timeout, clock=sv.clock, sleep=sv.sleep,
                                tick=w.tick)
    sv.say(f"{'serving' if not args.no_wait else 'loading'} {name}")
    _serve_hint(sv, sv.gateway.listen_url, name, info, bool(cfg.gateway.api_key))
    return EXIT_OK


def _serve_standalone(args: argparse.Namespace, sv: Services, path: Path, preset: str,
                      info: ModelInfo, timeout: float) -> int:
    """Launch a standalone server for a preset and wait for it to answer."""
    try:
        sess = sv.store.launch_server(sv.cfg, str(path), preset, port=args.port)
    except (LookupError, RuntimeError, OSError) as e:
        raise CliError(f"cannot start {path.name}: {e}") from None
    if not args.no_wait:
        with Waiting(sv, f"loading {sess.name}") as w:
            actions.wait_server_ready(sess, sv.health, timeout, clock=sv.clock, sleep=sv.sleep,
                                      tick=w.tick)
    sv.say(f"{'serving' if not args.no_wait else 'started'} {sess.name} on its own port "
           f"(pid {sess.pid}, log {sess.log})")
    _serve_hint(sv, sess.url, None, info, bool(sv.cfg.profile_for(str(path)).api_key))
    return EXIT_OK


def cmd_up(args: argparse.Namespace, sv: Services) -> int:
    """``llamactl up``: start the gateway (nothing happens when it already runs)."""
    sv.require_binary()
    gw = sv.gateway
    with Waiting(sv, "starting the gateway") as w:
        st, started = actions.ensure_gateway(gw, clock=sv.clock, sleep=sv.sleep, tick=w.tick)
    if not started:
        sv.say(f"gateway already running at {st.url} (pid {st.pid})")
        return EXIT_OK
    count = actions.count_presets(gateway_mod.PRESETS_FILE)
    sv.say(f"gateway running at {st.url} (pid {st.pid}); {count} preset{'s' * (count != 1)} "
           "written; models load on first use or with `llamactl load`")
    for problem in gw.problems:
        sv.note(f"warning: {problem}")
    return EXIT_OK


def cmd_down(args: argparse.Namespace, sv: Services) -> int:
    """``llamactl down``: stop the gateway and the models it runs (idempotent)."""
    if sv.gateway.stop():
        sv.say("gateway stopped (its loaded models went with it; standalone servers keep running)")
    else:
        sv.say("gateway was not running")
    return EXIT_OK


def cmd_ps(args: argparse.Namespace, sv: Services) -> int:
    """``llamactl ps``: the gateway with its models, and standalone or attached servers."""
    gw = sv.gateway
    status = gw.status()
    models, error = None, ""
    if status.running:
        try:
            models = gw.models()
        except GatewayError as e:
            error = str(e)
    state = gw.state
    sessions = list(sv.store.sessions)
    states = {s.id: sv.health(s.url) for s in sessions}
    data = report.ps_data(status, models, error, sessions, states, sv.now(),
                          state.started if state else 0.0)
    sv.emit(args.json, data, render.render_ps)
    return EXIT_OK


def _require_gateway(sv: Services) -> list[str]:
    """Model ids of the running gateway.

    Raises:
        CliError: When the gateway is not running.
    """
    if not sv.gateway.running:
        raise CliError("the gateway is not running; start it with `llamactl up` "
                       "(or `llamactl serve MODEL`)")
    return [m.name for m in sv.gateway.models()]


def cmd_load(args: argparse.Namespace, sv: Services) -> int:
    """``llamactl load NAME``: ask the gateway to load a model and wait until it has."""
    name = _match(args.name, _require_gateway(sv), "gateway model")
    current = next(m for m in sv.gateway.models() if m.name == name)
    if current.state in (gateway_mod.LOADED, gateway_mod.SLEEPING):
        sv.say(f"{name} is already {current.state}")
        return EXIT_OK
    if current.state != gateway_mod.LOADING:
        sv.gateway.load(name)
    if not args.no_wait:
        with Waiting(sv, f"loading {name}") as w:
            actions.wait_loaded(sv.gateway, name, _load_timeout(_model_size(sv, name)),
                                clock=sv.clock, sleep=sv.sleep, tick=w.tick)
    sv.say(f"{name} is {'loading' if args.no_wait else 'loaded'}")
    return EXIT_OK


def cmd_unload(args: argparse.Namespace, sv: Services) -> int:
    """``llamactl unload NAME``: ask the gateway to unload a model."""
    name = _match(args.name, _require_gateway(sv), "gateway model")
    sv.gateway.unload(name)
    if actions.wait_unloaded(sv.gateway, name, UNLOAD_TIMEOUT, clock=sv.clock, sleep=sv.sleep):
        sv.say(f"{name} unloaded")
    else:
        sv.say(f"{name}: unload requested, but the gateway still lists it as running")
    return EXIT_OK


def cmd_stop(args: argparse.Namespace, sv: Services) -> int:
    """``llamactl stop NAME | --all``: stop standalone servers; the gateway is never touched."""
    if args.all == bool(args.name):
        raise UsageError("give a server NAME or --all")
    sessions = list(sv.store.sessions)
    if args.name == GATEWAY_LOG:
        raise UsageError("that is the gateway; stop it with `llamactl down`")
    targets = sessions if args.all else [_find_session(args.name, sessions)]
    for s in targets:
        sv.store.stop(s)
        sv.say(f"{'detached' if s.external else 'stopped'} {s.id}")
    if not targets:
        sv.say("no standalone or attached servers")
    if sv.gateway.running:
        sv.say("the gateway was not touched; stop it with `llamactl down`")
    return EXIT_OK


def tail_lines(path: Path, count: int, block: int = 65536) -> list[str]:
    """The last `count` lines of a file, reading from its end.

    Args:
        path: File to read.
        count: Number of lines wanted (0 gives none).
        block: Bytes read per step.

    Returns:
        The lines without line breaks.

    Raises:
        OSError: When the file cannot be read.
    """
    if count <= 0:
        return []
    with path.open("rb") as fh:
        end = fh.seek(0, 2)
        data, pos = b"", end
        while pos > 0 and data.count(b"\n") <= count:
            pos = max(0, pos - block)
            fh.seek(pos)
            data = fh.read(min(block, end - pos)) + data
    return data.decode(errors="replace").splitlines()[-count:]


def _log_path(name: str, sv: Services) -> Path:
    """File behind a log NAME: ``gateway``, or a standalone session's id or name.

    Raises:
        UsageError: Unknown server name.
        CliError: No log file for it.
    """
    if name == GATEWAY_LOG:
        st = sv.gateway.state
        path = Path(st.log) if st and st.log else gateway_mod.LOG_DIR / gateway_mod.LOG_NAME
    else:
        sess = _find_session(name, sv.store.sessions)
        if sess.external or not sess.log:
            raise CliError(f"'{sess.name}' is an attached server; llamactl has no log for it")
        path = Path(sess.log)
    if not path.is_file():
        raise CliError(f"no log at {path}")
    return path


def cmd_logs(args: argparse.Namespace, sv: Services) -> int:
    """``llamactl logs NAME``: print the end of a log; ``-f`` follows it until Ctrl-C."""
    path = _log_path(args.name, sv)
    for line in tail_lines(path, args.lines):
        sv.say(line)
    if not args.follow:
        return EXIT_OK
    try:
        sv.out.flush()
        pos = path.stat().st_size
        while True:
            sv.sleep(FOLLOW_INTERVAL)
            size = path.stat().st_size
            if size < pos:                  # truncated or replaced: start over
                pos = 0
            if size > pos:
                with path.open("rb") as fh:
                    fh.seek(pos)
                    chunk = fh.read(size - pos)
                pos += len(chunk)
                sv.out.write(chunk.decode(errors="replace"))
                sv.out.flush()
    except KeyboardInterrupt:
        return EXIT_OK
    except OSError as e:
        raise CliError(f"cannot follow {path}: {e}") from None


# --------------------------------------------------------------------------- parser

def _positive(text: str) -> int:
    """argparse type: an integer of at least 1."""
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return n


def _version() -> str:
    """Installed package version, or "unknown"."""
    try:
        return metadata.version("llama-tui")
    except metadata.PackageNotFoundError:
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    """The argument parser with all subcommands.

    Returns:
        The parser; ``parse_args`` yields ``command=None`` when no subcommand is given.
    """
    p = argparse.ArgumentParser(
        prog="llamactl",
        description="Plan, tune and serve llama.cpp models. Without a command, opens the TUI.")
    p.add_argument("--version", action="version", version=f"llamactl {_version()}")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    def add(name: str, func: Callable[..., int], help: str, *, json_out: bool = False,
            model: bool = False, intent: bool = False) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help, description=help)
        sp.set_defaults(func=func)
        if model:
            sp.add_argument("model", metavar="MODEL", help="alias, file stem or unique substring")
        if intent:
            sp.add_argument("--for", dest="intent", choices=list(intents.by_key), metavar="INTENT",
                            help=f"one of {', '.join(intents.by_key)} "
                                 "(default: the configured default intent)")
        if json_out:
            sp.add_argument("--json", action="store_true",
                            help="print one JSON document and nothing else on stdout")
        return sp

    add("hw", cmd_hw, "show devices, RAM, CPU, calibration and the llama.cpp build", json_out=True)
    add("ls", cmd_ls, "list models with plans per intent and presets", json_out=True)
    sp = add("plan", cmd_plan, "show the plan for a model; changes nothing unless --save",
             json_out=True, model=True, intent=True)
    sp.add_argument("--all", action="store_true", help="compare every applicable intent")
    sp.add_argument("--save", action="store_true", help="store the plan as an estimated preset")
    sp.add_argument("--force", action="store_true",
                    help="with --save: replace a tuned or edited preset")
    sp = add("tune", cmd_tune, "measure candidate settings by launching the model and save the best",
             json_out=True, model=True, intent=True)
    sp.add_argument("--trials", type=_positive, default=4, metavar="N", help="most launches (default 4)")
    sp.add_argument("--force", action="store_true", help="skip the free-memory check")
    sp = add("serve", cmd_serve, "make sure a preset exists, then serve it",
             model=True, intent=True)
    sp.add_argument("--standalone", action="store_true",
                    help="run its own llama-server on its own port instead of the gateway")
    sp.add_argument("--port", type=_positive, help="with --standalone: listen port")
    sp.add_argument("--no-wait", action="store_true", help="do not wait until the model is loaded")
    sp.add_argument("--no-activate", action="store_true",
                    help="with --for: serve as NAME@INTENT and keep the active preset")
    add("up", cmd_up, "start the gateway")
    add("down", cmd_down, "stop the gateway")
    add("ps", cmd_ps, "show gateway models and standalone servers", json_out=True)
    sp = add("load", cmd_load, "load a model in the gateway")
    sp.add_argument("name", metavar="NAME", help="model id as `llamactl ps` lists it")
    sp.add_argument("--no-wait", action="store_true", help="do not wait until it is loaded")
    sp = add("unload", cmd_unload, "unload a model from the gateway")
    sp.add_argument("name", metavar="NAME", help="model id as `llamactl ps` lists it")
    sp = add("stop", cmd_stop, "stop standalone servers (never the gateway)")
    sp.add_argument("name", nargs="?", metavar="NAME", help="server name or id")
    sp.add_argument("--all", action="store_true", help="stop every standalone server")
    sp = add("logs", cmd_logs, "print or follow a log")
    sp.add_argument("name", metavar="NAME", help=f"'{GATEWAY_LOG}', or a standalone server's name or id")
    sp.add_argument("-f", "--follow", action="store_true", help="keep printing new lines (Ctrl-C ends)")
    sp.add_argument("-n", "--lines", type=int, default=DEFAULT_TAIL, metavar="LINES",
                    help=f"lines to show first (default {DEFAULT_TAIL})")
    return p


def is_read_only(args: argparse.Namespace) -> bool:
    """True for invocations that must not write any state file."""
    return args.command in ("hw", "ls", "ps", "logs") or (
        args.command == "plan" and not args.save)


def launch_tui() -> int:
    """Open the Textual TUI (imported here so headless commands never load Textual)."""
    from . import app
    app.main()
    return EXIT_OK


def main(argv: list[str] | None = None, services: Services | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Arguments without the program name; None reads ``sys.argv``.
        services: Prebuilt services (tests); None builds the real ones from the config file.

    Returns:
        The exit code: 0 success, 1 runtime failure, 2 usage error.
    """
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:                 # argparse exits 2 on bad usage and 0 on --version/--help
        return e.code if isinstance(e.code, int) else EXIT_USAGE if e.code else EXIT_OK
    if args.command is None:
        return launch_tui()
    err = services.err if services else sys.stderr
    try:
        sv = services or Services(Config.load(), read_only=is_read_only(args))
    except (OSError, ValueError) as e:
        err.write(f"llamactl: error: cannot read the config {CONFIG_FILE}: {e}\n")
        return EXIT_FAILURE
    try:
        return args.func(args, sv)
    except UsageError as e:
        err.write(f"llamactl: error: {e}\n")
        return EXIT_USAGE
    except (CliError, ActionError, GatewayError, GGUFError, OSError) as e:
        err.write(f"llamactl: error: {e}\n")
        return EXIT_FAILURE
    except KeyboardInterrupt:
        err.write("\nllamactl: interrupted\n")
        return EXIT_FAILURE
