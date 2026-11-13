"""The gateway: one llama-server in router mode serving every model's presets.

`render_presets` turns the config into the router's ``--models-preset`` INI file. `Gateway`
starts and stops the router process (detached, like standalone sessions, so it outlives the
TUI) and talks to its HTTP API. Both the process launcher and the HTTP client factory are
injectable, so tests never start llama.cpp or touch the network.
"""
from __future__ import annotations

import json
import fcntl
import os
import signal
import subprocess
import secrets
import sys
import time
from dataclasses import asdict, dataclass, field
from contextlib import contextmanager
from functools import wraps
from threading import RLock, local
from pathlib import Path
from typing import Any, Callable, Iterator, ParamSpec, TypeVar

import httpx

from . import network
from .command import check_ini_value, preset_items
from .config import CONFIG_DIR, LOG_DIR, PRESET_SEP, Config, derive_alias
from .sessions import pid_alive, port_free

STATE_FILE = CONFIG_DIR / "gateway.json"
PRESETS_FILE = CONFIG_DIR / "gateway.ini"
LOG_NAME = "gateway.log"
HTTP_TIMEOUT = 10.0
STOP_TIMEOUT = 15.0
SETTLE_SECONDS = 1.0

# Model states reported by `GatewayModel.state`.
UNLOADED, LOADING, LOADED, SLEEPING, FAILED = "unloaded", "loading", "loaded", "sleeping", "failed"
_LOCK = RLock()
_LOCK_CONTEXT = local()
_P = ParamSpec("_P")
_R = TypeVar("_R")


@contextmanager
def _runtime_lock() -> Iterator[None]:
    """Serialize runtime changes across threads and cooperating CLI processes.

    Yields:
        Control while the scratch/state-directory runtime lock is held.
    """
    with _LOCK:
        if getattr(_LOCK_CONTEXT, "held", False):
            yield
            return
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with STATE_FILE.with_suffix(".lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            _LOCK_CONTEXT.held = True
            try:
                yield
            finally:
                _LOCK_CONTEXT.held = False
                fcntl.flock(handle, fcntl.LOCK_UN)


def _serialized(method: Callable[_P, _R]) -> Callable[_P, _R]:
    """Run a mutation under the reentrant process/runtime lock.

    Args:
        method: Gateway mutation.

    Returns:
        The serialized method with its original signature preserved.
    """
    @wraps(method)
    def locked(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Return the wrapped mutation's result while holding the runtime lock."""
        instance = args[0] if args else None
        if instance is not None and not getattr(instance, "_backend_mode", False):
            state = getattr(instance, "state", None)
            if state is not None and state.backend_url:
                return method(*args, **kwargs)
        with _runtime_lock():
            return method(*args, **kwargs)
    return locked


class GatewayError(RuntimeError):
    """The gateway could not be started, stopped or reached, or it refused a request."""


def _write_private_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically save runtime credentials with owner-only permissions.

    Args:
        path: Runtime JSON destination.
        data: JSON-serializable runtime state.

    Raises:
        OSError: If the state cannot be saved.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(json.dumps(data, indent=2))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


# --------------------------------------------------------------------------- INI rendering

def _name_problem(name: str) -> str:
    """Why `name` cannot be a section name of the preset INI, or "" when it can.

    Args:
        name: Candidate section name.

    Returns:
        A short reason, or an empty string.
    """
    if not name or name != name.strip():
        return "empty or padded with spaces"
    if "]" in name or "\n" in name or "\r" in name:
        return "contains ']' or a line break"
    return ""


def _section(name: str, model: Path, items: list[tuple[str, str]]) -> str:
    """One INI section: model path first, then the preset's items.

    Args:
        name: Section name.
        model: Absolute model path.
        items: Key/value pairs from `preset_items`.

    Returns:
        The section text, ending in a newline.

    Raises:
        ValueError: When the model path cannot be written (it holds ``;`` or ``#``).
    """
    check_ini_value("model", str(model))
    lines = [f"[{name}]", f"model = {model}"]
    lines += [f"{k} = {v}" for k, v in items]
    return "\n".join(lines) + "\n"


def render_presets(cfg: Config, models: list[Path] | None = None,
                   problems: list[str] | None = None,
                   blocked: dict[str, str] | None = None) -> str:
    """Render the router's ``--models-preset`` INI from the config.

    Each model's active preset is served as ``[alias]``; every other preset as
    ``[alias@name]`` (llama.cpp rewrites the text after a ``:`` in a section name, so the
    separator is `PRESET_SEP`). A ``[*]`` section carries ``sleep-idle-seconds`` from the
    gateway settings when it is above 0. Models are sorted by alias and presets by name, so
    the output is stable and diffable. Models whose file no longer exists are skipped, and so
    is any preset the INI format cannot hold (see `problems`), and any model `blocked` names
    (a ``; skipped`` comment in the file keeps it from looking like it vanished).

    Args:
        cfg: Config with the models, presets and gateway settings.
        models: Render only these model files; None renders every model with presets.
        problems: When given, receives one line per skipped preset or blocked model.
            Missing model files are not problems.
        blocked: Model path -> reason the build cannot load it (e.g. an architecture it does
            not support); those models are left out of the INI.

    Returns:
        The INI text.
    """
    wanted = None if models is None else {str(Path(m)) for m in models}
    found = [(ent.alias or derive_alias(path, set()), path, ent) for path, ent in cfg.models.items()
             if ent.presets and (wanted is None or path in wanted) and Path(path).is_file()]
    blocks = ["version = 1\n"]
    if cfg.gateway.sleep_idle > 0:
        blocks.append(f"[*]\nsleep-idle-seconds = {cfg.gateway.sleep_idle}\n")
    seen: set[str] = set()
    for alias, path, ent in sorted(found):
        if blocked and path in blocked:
            line = f"skipped [{alias}]: {blocked[path]}"
            blocks.append(f"; {line}\n")
            if problems is not None:
                problems.append(line)
            continue
        active = ent.active if ent.active in ent.presets else sorted(ent.presets)[0]
        order = [active] + sorted(n for n in ent.presets if n != active)
        for pname in order:
            section = alias if pname == active else f"{alias}{PRESET_SEP}{pname}"
            try:
                if why := _name_problem(section):
                    raise ValueError(f"section name is {why}")
                if section in seen:
                    raise ValueError("section name is used by another model")
                text = _section(section, Path(os.path.abspath(path)),
                                preset_items(ent.presets[pname].profile))
            except ValueError as e:
                line = f"skipped [{section}]: {e}".replace("\n", " ").replace("\r", " ")
                blocks.append(f"; {line}\n")
                if problems is not None:
                    problems.append(line)
                continue
            seen.add(section)
            blocks.append(text)
    return "\n".join(blocks)


def _preset_sections(text: str) -> dict[str, str]:
    """Extract canonical INI sections without comments or the version header.

    Args:
        text: Rendered or previously installed preset file.

    Returns:
        Section name to its complete INI definition.
    """
    sections: dict[str, str] = {}
    name = ""
    for line in text.splitlines():
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1]
            sections[name] = line + "\n"
        elif name and line.strip() and not line.lstrip().startswith((";", "#")):
            sections[name] += line + "\n"
    return sections


def _replace_sections(text: str, preserved: dict[str, str]) -> str:
    """Replace saved definitions with protected runtime definitions.

    Args:
        text: Newly rendered saved configuration, including diagnostic comments.
        preserved: Running definitions that must remain unchanged.

    Returns:
        The effective INI, retaining removed running definitions until stopped.
    """
    if not preserved:
        return text
    blocks: list[str] = []
    seen: set[str] = set()
    current = ""
    skip = False
    for line in text.splitlines(keepends=True):
        if line.startswith("[") and line.rstrip().endswith("]"):
            current = line.strip()[1:-1]
            skip = current in preserved
            if skip:
                blocks.append(preserved[current] + "\n")
                seen.add(current)
        if not skip or line.lstrip().startswith(";"):
            blocks.append(line)
    for name in sorted(preserved.keys() - seen):
        blocks.append("\n" + preserved[name])
    return "".join(blocks)


# --------------------------------------------------------------------------- state and models

@dataclass
class GatewayState:
    """What is remembered about a started gateway, in ``gateway.json``.

    Attributes:
        pid: Process id of the router (also its process group id).
        url: Base URL llamactl connects to, e.g. ``http://127.0.0.1:8080`` (loopback, also for a
            wildcard bind).
        log: Path of the log file.
        started: Unix time of the start.
        host: The listen address the router was started with ("" in a state file written
            before this was recorded).
        effective: Exact installed INI sections, retained across application restarts.
    """

    pid: int
    url: str
    log: str
    started: float = 0.0
    host: str = ""
    effective: dict[str, str] = field(default_factory=dict)
    backend_url: str = ""
    runtime_api_key: str = ""
    control_token: str = ""

    @classmethod
    def load(cls) -> "GatewayState | None":
        """Read the state file.

        Returns:
            The state, or None when the file is missing or unreadable.
        """
        try:
            raw = json.loads(STATE_FILE.read_text())
            effective = raw.get("effective", {})
            if not isinstance(effective, dict) or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in effective.items()):
                return None
            return cls(int(raw["pid"]), str(raw["url"]), str(raw.get("log", "")),
                       float(raw.get("started", 0.0)), str(raw.get("host", "")), effective,
                       str(raw.get("backend_url", "")), str(raw.get("runtime_api_key", "")),
                       str(raw.get("control_token", "")))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def save(self) -> None:
        """Write the state file, creating the config directory."""
        _write_private_json(STATE_FILE, asdict(self))

    @staticmethod
    def clear() -> None:
        """Delete the state file if it exists."""
        STATE_FILE.unlink(missing_ok=True)


@dataclass(frozen=True)
class GatewayStatus:
    """Snapshot of the gateway process.

    Attributes:
        running: The router process exists.
        url: Base URL llamactl connects to (from the running gateway, else from the settings).
        healthy: ``GET /health`` answered 200.
        pid: Router process id, 0 when not running.
        host: The listen address (of the running gateway, else the one it would start with);
            "" when unknown. `network.client_urls_for(url, host)` gives the URLs for clients.
    """

    running: bool
    url: str
    healthy: bool = False
    pid: int = 0
    host: str = ""


@dataclass(frozen=True)
class GatewayModel:
    """One model entry of the router's ``GET /models``.

    Attributes:
        name: Model id clients use: ``alias`` or ``alias@preset``.
        state: "unloaded", "loading", "loaded", "sleeping" or "failed" (the router's other
            values, "downloading" and "downloaded", pass through unchanged).
        exit_code: Exit code of the model process when `state` is "failed".
        args: Command line the router uses (or used) for the model instance.
    """

    name: str
    state: str
    exit_code: int | None = None
    args: tuple[str, ...] = field(default_factory=tuple)


def parse_models(payload: Any) -> list[GatewayModel]:
    """Parse the JSON body of ``GET /models``.

    The ``status`` object is ``{"value": ...}`` plus ``args`` once the model was started; a
    model that exited with an error is ``{"value": "unloaded", "failed": true, "exit_code": N}``.

    Args:
        payload: Decoded JSON.

    Returns:
        The models in the router's order.

    Raises:
        ValueError: When the payload does not look like a model list.
    """
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ValueError("no 'data' list")
    out: list[GatewayModel] = []
    for item in data:
        status = item.get("status") if isinstance(item, dict) else None
        if not isinstance(status, dict) or "id" not in item:
            raise ValueError("model entry without 'id' or 'status'")
        state = str(status.get("value", UNLOADED))
        code = status.get("exit_code")
        code = code if isinstance(code, int) and not isinstance(code, bool) else None
        failed = bool(status.get("failed")) or bool(code)
        if state == UNLOADED and failed:
            state = FAILED
        out.append(GatewayModel(str(item["id"]), state, code if state == FAILED else None,
                                tuple(str(a) for a in status.get("args") or ())))
    return out


# --------------------------------------------------------------------------- process control

def terminate_group(pid: int, timeout: float = STOP_TIMEOUT,
               sleep: Callable[[float], None] = time.sleep) -> None:
    """Stop a process group: SIGTERM, then SIGKILL after `timeout` seconds.

    Only the whole group is signalled when the process leads one (the router is started in
    its own session), so its model processes go with it.

    Args:
        pid: Process id of the group leader.
        timeout: Seconds to wait after SIGTERM.
        sleep: Sleep function (injectable).
    """
    def send(sig: int) -> None:
        try:
            if os.getpgid(pid) == pid:
                os.killpg(pid, sig)
            else:
                os.kill(pid, sig)
        except OSError:
            pass

    send(signal.SIGTERM)
    waited = 0.0
    while pid_alive(pid) and waited < timeout:
        sleep(0.1)
        waited += 0.1
    if pid_alive(pid):
        send(signal.SIGKILL)


def log_reason(log: Path) -> str:
    """The log line that most likely says why a process exited: its last error, else its last line.

    Args:
        log: Log file of the process.

    Returns:
        A one-line message, with the log's timestamp prefix removed.
    """
    try:
        lines = [ln.strip() for ln in log.read_text(errors="replace").splitlines() if ln.strip()]
    except OSError:
        return "no log output"
    errors = [ln for ln in lines if " E " in ln[:30]]
    pick = (errors or lines or ["no log output"])[-1]
    return pick.split(" ", 2)[-1] if pick[:1].isdigit() else pick


class Gateway:
    """The router-mode llama-server and its API.

    Args:
        cfg: Config with the gateway settings and the models.
        popen: Process launcher with the signature of `subprocess.Popen`.
        http: Factory for an HTTP client, called with ``base_url``, ``headers`` and
            ``timeout`` keywords; the client needs ``request(method, path, **kw)`` returning
            an object with ``status_code``, ``text`` and ``json()``, and context-manager
            support (as `httpx.Client` has).
        settle: Seconds `start` watches the new process for an immediate exit (the router
            exits at once on a preset it cannot parse); 0 skips the check.
        sleep: Sleep function used while settling (injectable).
        read_only: Never touch ``gateway.json``: a dead gateway's state is reported as "not
            running" but not deleted (for commands that only look, like ``llamactl ps``).
        learn: Called with a log file when the router exits right away, so an
            ``unknown model architecture`` line can be remembered (or None).
        blocked: Returns model path -> reason the build cannot load it; those models are left
            out of the preset INI (or None).
        supervised: Own the public endpoint through the detached admission supervisor.
            False retains direct-router behavior for existing gateway compatibility/tests.

    Attributes:
        host_override: Listen address for the next `start` instead of the configured one (never
            saved); None uses ``cfg.gateway.host``.
        pending_names: Cached served names whose saved settings differ from the router's
            installed settings; refreshed by model polling and reloading without UI HTTP.
    """

    def __init__(self, cfg: Config, popen: Callable[..., Any] = subprocess.Popen,
                 http: Callable[..., Any] = httpx.Client, settle: float = SETTLE_SECONDS,
                 sleep: Callable[[float], None] = time.sleep, read_only: bool = False,
                 learn: Callable[[Path], None] | None = None,
                 blocked: Callable[[], dict[str, str]] | None = None,
                 supervised: bool = True) -> None:
        self.read_only = read_only
        self.cfg = cfg
        self._popen = popen
        self._http = http
        self._settle = settle
        self._sleep = sleep
        self._learn = learn
        self._blocked = blocked
        self.supervised = supervised
        self.problems: list[str] = []
        self.host_override: str | None = None
        self._backend_mode = False
        try:
            self.pending_names: frozenset[str] = frozenset(self.pending_changes())
        except GatewayError:
            self.pending_names = frozenset()

    # -- process -----------------------------------------------------------------------

    @property
    def host(self) -> str:
        """The listen address a start would use: the override, else the configured host."""
        return self.cfg.gateway.host if self.host_override is None else self.host_override

    @property
    def listen_url(self) -> str:
        """Base URL llamactl connects to (loopback for a wildcard bind)."""
        return f"http://{network.probe_host(self.host)}:{self.cfg.gateway.port}"

    @property
    def running_host(self) -> str:
        """The listen address of the running gateway (the configured one when it is unknown)."""
        st = self.state
        return (st.host if st is not None and st.host else self.cfg.gateway.host)

    @property
    def state(self) -> GatewayState | None:
        """The remembered state of a running gateway, else None (a dead one is forgotten)."""
        st = GatewayState.load()
        if st is not None and not pid_alive(st.pid):
            if not self.read_only:
                GatewayState.clear()
            return None
        return st

    @property
    def running(self) -> bool:
        """True when the router process is alive."""
        return self.state is not None

    def command(self) -> list[str]:
        """The router's argv: no model, the preset file, and the gateway settings.

        Returns:
            ``llama-server --models-preset <ini> --models-max N --host H --port P [--api-key K]``.
        """
        g = self.cfg.gateway
        argv = [str(Path(self.cfg.bin_dir) / "llama-server"), "--models-preset", str(PRESETS_FILE),
                "--models-max", str(max(1, g.models_max)), "--host", self.host, "--port", str(g.port)]
        if g.api_key:
            argv += ["--api-key", g.api_key]
        return argv

    def _saved_presets(self) -> str:
        """Render saved settings and collect unsupported/invalid preset problems.

        Returns:
            The desired INI before protecting running definitions.
        """
        self.problems = []
        return render_presets(self.cfg, problems=self.problems,
                              blocked=self._blocked() if self._blocked else None)

    def _effective_sections(self, state: GatewayState) -> dict[str, str]:
        """Read effective definitions, including migration from older state files.

        Args:
            state: Running router's persisted state.

        Returns:
            Definitions installed in the router; never inferred from saved config.

        Raises:
            GatewayError: When an older router's effective INI is unavailable.
        """
        if state.effective:
            return state.effective.copy()
        try:
            return _preset_sections(PRESETS_FILE.read_text())
        except OSError as error:
            raise GatewayError("cannot recover running gateway settings; keep its preset "
                               "file until the gateway stops") from error

    def pending_changes(self) -> dict[str, tuple[str | None, str | None]]:
        """Compare effective running definitions with saved settings without HTTP.

        Returns:
            Served name to (effective INI, saved INI), omitting unchanged definitions.
            A missing definition is None. The global wildcard is omitted.
        """
        state = self.state
        if state is None:
            return {}
        effective = self._effective_sections(state)
        saved = _preset_sections(render_presets(self.cfg))
        return {name: (effective.get(name), saved.get(name))
                for name in effective.keys() | saved.keys()
                if name != "*" and effective.get(name) != saved.get(name)}

    @_serialized
    def write_presets(self) -> Path:
        """Render the preset INI from the config and write it.

        Presets that could not be written are listed in `problems`.

        Returns:
            The path of the file.
        """
        if self.running:
            self.reload()
            return PRESETS_FILE
        text = self._saved_presets()
        PRESETS_FILE.parent.mkdir(parents=True, exist_ok=True)
        PRESETS_FILE.write_text(text)
        return PRESETS_FILE

    @_serialized
    def start(self) -> GatewayState:
        """Start the router as a detached process (it survives the TUI), unless it runs.

        Writes the preset INI first. The process is not waited for: use `status` to see when
        it is healthy. It does watch the first moments, because the router exits at once when it
        rejects the preset file (for example an extra_args option this llama.cpp does not
        know), and reports the log line that says why. Blocks for up to `settle` seconds.

        Returns:
            The state of the (possibly already running) gateway.

        Raises:
            GatewayError: When the port is taken, the process cannot be started, or it exits
                right away.
        """
        if (st := self.state) is not None:
            return st
        g = self.cfg.gateway
        if not port_free(g.port, self.host):
            raise GatewayError(f"port {g.port} on {self.host} is already in use; "
                               "stop what holds it or change the gateway port")
        self.write_presets()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log = LOG_DIR / LOG_NAME
        argv = self.command()
        launch = STATE_FILE.with_name("gateway-launch.json")
        control_token = secrets.token_urlsafe(32) if self.supervised else ""
        if self.supervised:
            _write_private_json(launch, {"argv": argv, "config": self.cfg.to_dict(),
                                      "host": self.host, "port": g.port,
                                      "control_token": control_token})
            argv = [sys.executable, "-m", "llamactl.gateway_supervisor", str(launch)]
        try:
            with log.open("ab") as fh:
                proc = self._popen(argv, stdout=fh, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError as e:
            raise GatewayError(f"cannot start {argv[0]}: {e}") from e
        waited = 0.0
        while waited < self._settle and pid_alive(proc.pid):
            self._sleep(0.1)
            waited += 0.1
        if self._settle > 0 and not pid_alive(proc.pid):
            if self._learn is not None:
                self._learn(log)
            raise GatewayError(f"the gateway exited at once: {log_reason(log)} (log: {log})")
        st = GatewayState(proc.pid, self.listen_url, str(log), time.time(), self.host,
                          _preset_sections(PRESETS_FILE.read_text()),
                          "managed" if self.supervised else "", g.api_key,
                          control_token)
        existing = GatewayState.load()
        if existing is not None and existing.pid == proc.pid and existing.backend_url:
            st = existing
        st.save()
        self.pending_names = frozenset()
        return st

    @_serialized
    def stop(self) -> bool:
        """Stop the router and its model processes, and forget the state.

        Returns:
            True when a running gateway was stopped, False when there was none.
        """
        st = self.state
        if st is None:
            self.pending_names = frozenset()
            return False
        terminate_group(st.pid)
        GatewayState.clear()
        self.pending_names = frozenset()
        return True

    def status(self) -> GatewayStatus:
        """Process state plus a health probe; never raises.

        Returns:
            The status; ``healthy`` is False when the router does not answer ``/health``.
        """
        st = self.state
        if st is None:
            return GatewayStatus(False, self.listen_url, host=self.host)
        try:
            healthy = self._request("GET", "/health").status_code == 200
        except GatewayError:
            healthy = False
        return GatewayStatus(True, st.url, healthy, st.pid, st.host or self.cfg.gateway.host)

    # -- router API --------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Send a request to the running router.

        The API key, when configured, goes in an ``Authorization: Bearer`` header.

        Args:
            method: HTTP method.
            path: Path with query, e.g. ``/models?reload=1``.
            **kwargs: Passed to the client's ``request`` (e.g. ``json=``).

        Returns:
            The response object (any status code).

        Raises:
            GatewayError: When the gateway is not running or does not answer.
        """
        st = self.state
        if st is None:
            raise GatewayError("the gateway is not running")
        key = st.runtime_api_key if st.backend_url else self.cfg.gateway.api_key
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        if path.startswith("/__llamactl/"):
            headers["X-Llamactl-Control"] = st.control_token
            kwargs.setdefault("timeout", 1850.0)
        try:
            with self._http(base_url=st.url, headers=headers, timeout=HTTP_TIMEOUT) as client:
                return client.request(method, path, **kwargs)
        except httpx.HTTPError as e:
            raise GatewayError(f"the gateway at {st.url} is not reachable: {e}") from e

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        """Like `_request`, but turn an error response into `GatewayError` and return the JSON.

        Raises:
            GatewayError: On a status of 400 or more (with the router's message), or a body
                that is not JSON.
        """
        r = self._request(method, path, **kwargs)
        if r.status_code >= 400:
            raise GatewayError(self._error_text(r))
        try:
            return r.json()
        except ValueError:
            raise GatewayError(f"the gateway answered {method} {path} with something that is "
                               "not JSON") from None

    @staticmethod
    def _error_text(r: Any) -> str:
        """Message for an error response: the router's own text when it has one."""
        if r.status_code in (401, 403):
            return f"the gateway rejected the API key (HTTP {r.status_code})"
        try:
            err = r.json().get("error")
            msg = err.get("message") if isinstance(err, dict) else err
        except (ValueError, AttributeError):
            msg = None
        return f"the gateway answered HTTP {r.status_code}: {msg or r.text.strip()[:200] or 'no details'}"

    def models(self) -> list[GatewayModel]:
        """The router's models and their states (``GET /models``).

        Raises:
            GatewayError: When the gateway is not running, unreachable or answers badly.
        """
        if self.read_only:
            return self._parse(self._call("GET", "/models"))
        return self.synchronize()

    def synchronize(self) -> list[GatewayModel]:
        """Promote pending definitions after observing a model actually stopped.

        Polling this method also covers models unloaded automatically by the router.
        Router HTTP endpoints do not provide an atomic unload/reload transaction; an
        external on-demand request racing this poll can still use the old definition.

        Returns:
            The current router model snapshot, after any required promotion.

        Raises:
            GatewayError: When the router cannot be queried or safely refreshed.
        """
        state = self.state
        if state is not None and state.backend_url and not self._backend_mode:
            result = self._parse(self._call("POST", "/__llamactl/control", json={"action": "sync"}))
            self.pending_names = frozenset(self.pending_changes())
            return result
        if self.read_only:
            return self._parse(self._call("GET", "/models"))
        return self._synchronize()

    @_serialized
    def _synchronize(self) -> list[GatewayModel]:
        """Query the router and promote any stopped definitions under the runtime lock.

        Returns:
            Router model snapshot after any promotion.

        Raises:
            GatewayError: When querying or refreshing the router fails.
        """
        models = self._parse(self._call("GET", "/models"))
        state = self.state
        if state is None:
            return models
        effective = self._effective_sections(state)
        saved = _preset_sections(self._saved_presets())
        live = {model.name for model in models if model.state not in (UNLOADED, FAILED)}
        stopped = (effective.keys() | saved.keys()) - live
        if live:
            stopped.discard("*")
        if any(effective.get(name) != saved.get(name) for name in stopped):
            return self._reload(models)
        self.pending_names = frozenset(self.pending_changes())
        return models

    @staticmethod
    def _parse(payload: Any) -> list[GatewayModel]:
        """`parse_models` with errors as `GatewayError`."""
        try:
            return parse_models(payload)
        except ValueError as e:
            raise GatewayError(f"unexpected answer from the gateway's /models: {e}") from None

    @_serialized
    def load(self, name: str) -> None:
        """Ask the router to load a model (``POST /models/load``); it returns before it is ready.

        Args:
            name: Model id: ``alias`` or ``alias@preset``.

        Raises:
            GatewayError: When the model is unknown or already running, or the call fails.
        """
        try:
            state = self.state
            if state is not None and state.backend_url and not self._backend_mode:
                self._call("POST", "/__llamactl/control", json={"action": "load", "model": name})
                return
            self.synchronize()
            self._call("POST", "/models/load", json={"model": name})
        except GatewayError as e:
            raise GatewayError(f"cannot load '{name}': {e}") from None

    @_serialized
    def unload(self, name: str) -> None:
        """Ask the router to unload a model (``POST /models/unload``).

        Args:
            name: Model id: ``alias`` or ``alias@preset``.

        Raises:
            GatewayError: When the model is unknown or not running, or the call fails.
        """
        try:
            state = self.state
            if state is not None and state.backend_url and not self._backend_mode:
                self._call("POST", "/__llamactl/control", json={"action": "unload", "model": name})
                return
            self._call("POST", "/models/unload", json={"model": name})
            self.synchronize()
        except GatewayError as e:
            raise GatewayError(f"cannot unload '{name}': {e}") from None

    @_serialized
    def reload(self) -> list[GatewayModel]:
        """Rewrite the preset INI and, when the gateway runs, make it re-read the file.

        Loaded, loading and sleeping models retain their installed definition. Their
        saved changes become effective after a stopped state is observed.

        Returns:
            The router's models after the reload; empty when the gateway is not running.

        Raises:
            GatewayError: When the running gateway cannot be reached.
        """
        state = self.state
        if state is not None and state.backend_url and not self._backend_mode:
            result = self._parse(self._call("POST", "/__llamactl/control", json={
                "action": "reload", "presets": self._saved_presets()}))
            self.pending_names = frozenset(self.pending_changes())
            return result
        if not self.running:
            self.write_presets()
            self.pending_names = frozenset()
            return []
        models = self._parse(self._call("GET", "/models"))
        return self._reload(models)

    def _reload(self, models: list[GatewayModel]) -> list[GatewayModel]:
        """Install saved definitions while keeping every observed live section.

        Args:
            models: Fresh model snapshot obtained while holding the runtime lock.

        Returns:
            Router models after reloading the merged definitions.

        Raises:
            GatewayError: When the gateway disappears or rejects the refresh.
        """
        state = self.state
        if state is None:
            raise GatewayError("the gateway is not running")
        effective = self._effective_sections(state)
        live = {model.name for model in models if model.state not in (UNLOADED, FAILED)}
        if live - effective.keys():
            raise GatewayError("cannot recover effective settings for running models: "
                               + ", ".join(sorted(live - effective.keys())))
        protected = {name: text for name, text in effective.items() if name in live}
        # A changed wildcard is inherited by every model; keep it while any model lives.
        if live:
            protected["*"] = effective.get("*", "")
        text = _replace_sections(self._saved_presets(), protected)
        PRESETS_FILE.write_text(text)
        result = self._parse(self._call("GET", "/models?reload=1"))
        state.effective = _preset_sections(text)
        state.save()
        self.pending_names = frozenset(self.pending_changes())
        return result
