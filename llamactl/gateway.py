"""The gateway: one llama-server in router mode serving every model's presets.

`render_presets` turns the config into the router's ``--models-preset`` INI file. `Gateway`
starts and stops the router process (detached, like standalone sessions, so it outlives the
TUI) and talks to its HTTP API. Both the process launcher and the HTTP client factory are
injectable, so tests never start llama.cpp or touch the network.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

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

_ANY_ADDRESS = {"", "0.0.0.0", "::", "[::]"}


class GatewayError(RuntimeError):
    """The gateway could not be started, stopped or reached, or it refused a request."""


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
                   problems: list[str] | None = None) -> str:
    """Render the router's ``--models-preset`` INI from the config.

    Each model's active preset is served as ``[alias]``; every other preset as
    ``[alias@name]`` (llama.cpp rewrites the text after a ``:`` in a section name, so the
    separator is `PRESET_SEP`). A ``[*]`` section carries ``sleep-idle-seconds`` from the
    gateway settings when it is above 0. Models are sorted by alias and presets by name, so
    the output is stable and diffable. Models whose file no longer exists are skipped, and so
    is any preset the INI format cannot hold (see `problems`).

    Args:
        cfg: Config with the models, presets and gateway settings.
        models: Render only these model files; None renders every model with presets.
        problems: When given, receives one line per skipped preset that could not be written
            (e.g. an extra_args token with no flag). Missing model files are not problems.

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


# --------------------------------------------------------------------------- state and models

@dataclass
class GatewayState:
    """What is remembered about a started gateway, in ``gateway.json``.

    Attributes:
        pid: Process id of the router (also its process group id).
        url: Base URL clients use, e.g. ``http://127.0.0.1:8080``.
        log: Path of the log file.
        started: Unix time of the start.
    """

    pid: int
    url: str
    log: str
    started: float = 0.0

    @classmethod
    def load(cls) -> "GatewayState | None":
        """Read the state file.

        Returns:
            The state, or None when the file is missing or unreadable.
        """
        try:
            raw = json.loads(STATE_FILE.read_text())
            return cls(int(raw["pid"]), str(raw["url"]), str(raw.get("log", "")),
                       float(raw.get("started", 0.0)))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def save(self) -> None:
        """Write the state file, creating the config directory."""
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def clear() -> None:
        """Delete the state file if it exists."""
        STATE_FILE.unlink(missing_ok=True)


@dataclass(frozen=True)
class GatewayStatus:
    """Snapshot of the gateway process.

    Attributes:
        running: The router process exists.
        url: Base URL (from the running gateway, else from the settings).
        healthy: ``GET /health`` answered 200.
        pid: Router process id, 0 when not running.
    """

    running: bool
    url: str
    healthy: bool = False
    pid: int = 0


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


def _log_reason(log: Path) -> str:
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


def _client_host(host: str) -> str:
    """Address to connect to for a listen address (wildcards become loopback)."""
    return "127.0.0.1" if host in _ANY_ADDRESS else host


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
    """

    def __init__(self, cfg: Config, popen: Callable[..., Any] = subprocess.Popen,
                 http: Callable[..., Any] = httpx.Client, settle: float = SETTLE_SECONDS,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.cfg = cfg
        self._popen = popen
        self._http = http
        self._settle = settle
        self._sleep = sleep
        self.problems: list[str] = []

    # -- process -----------------------------------------------------------------------

    @property
    def listen_url(self) -> str:
        """Base URL implied by the gateway settings."""
        g = self.cfg.gateway
        return f"http://{_client_host(g.host)}:{g.port}"

    @property
    def state(self) -> GatewayState | None:
        """The remembered state of a running gateway, else None (a dead one is forgotten)."""
        st = GatewayState.load()
        if st is not None and not pid_alive(st.pid):
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
                "--models-max", str(max(1, g.models_max)), "--host", g.host, "--port", str(g.port)]
        if g.api_key:
            argv += ["--api-key", g.api_key]
        return argv

    def write_presets(self) -> Path:
        """Render the preset INI from the config and write it.

        Presets that could not be written are listed in `problems`.

        Returns:
            The path of the file.
        """
        self.problems = []
        text = render_presets(self.cfg, problems=self.problems)
        PRESETS_FILE.parent.mkdir(parents=True, exist_ok=True)
        PRESETS_FILE.write_text(text)
        return PRESETS_FILE

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
        if not port_free(g.port, g.host):
            raise GatewayError(f"port {g.port} on {g.host} is already in use; "
                               "stop what holds it or change the gateway port")
        self.write_presets()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log = LOG_DIR / LOG_NAME
        argv = self.command()
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
            raise GatewayError(f"the gateway exited at once: {_log_reason(log)} (log: {log})")
        st = GatewayState(proc.pid, self.listen_url, str(log), time.time())
        st.save()
        return st

    def stop(self) -> bool:
        """Stop the router and its model processes, and forget the state.

        Returns:
            True when a running gateway was stopped, False when there was none.
        """
        st = self.state
        if st is None:
            return False
        terminate_group(st.pid)
        GatewayState.clear()
        return True

    def status(self) -> GatewayStatus:
        """Process state plus a health probe; never raises.

        Returns:
            The status; ``healthy`` is False when the router does not answer ``/health``.
        """
        st = self.state
        if st is None:
            return GatewayStatus(False, self.listen_url)
        try:
            healthy = self._request("GET", "/health").status_code == 200
        except GatewayError:
            healthy = False
        return GatewayStatus(True, st.url, healthy, st.pid)

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
        headers = {"Authorization": f"Bearer {self.cfg.gateway.api_key}"} if self.cfg.gateway.api_key else {}
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
        return self._parse(self._call("GET", "/models"))

    @staticmethod
    def _parse(payload: Any) -> list[GatewayModel]:
        """`parse_models` with errors as `GatewayError`."""
        try:
            return parse_models(payload)
        except ValueError as e:
            raise GatewayError(f"unexpected answer from the gateway's /models: {e}") from None

    def load(self, name: str) -> None:
        """Ask the router to load a model (``POST /models/load``); it returns before it is ready.

        Args:
            name: Model id: ``alias`` or ``alias@preset``.

        Raises:
            GatewayError: When the model is unknown or already running, or the call fails.
        """
        try:
            self._call("POST", "/models/load", json={"model": name})
        except GatewayError as e:
            raise GatewayError(f"cannot load '{name}': {e}") from None

    def unload(self, name: str) -> None:
        """Ask the router to unload a model (``POST /models/unload``).

        Args:
            name: Model id: ``alias`` or ``alias@preset``.

        Raises:
            GatewayError: When the model is unknown or not running, or the call fails.
        """
        try:
            self._call("POST", "/models/unload", json={"model": name})
        except GatewayError as e:
            raise GatewayError(f"cannot unload '{name}': {e}") from None

    def reload(self) -> list[GatewayModel]:
        """Rewrite the preset INI and, when the gateway runs, make it re-read the file.

        Running models whose preset changed or vanished are unloaded by the router.

        Returns:
            The router's models after the reload; empty when the gateway is not running.

        Raises:
            GatewayError: When the running gateway cannot be reached.
        """
        self.write_presets()
        if not self.running:
            return []
        return self._parse(self._call("GET", "/models?reload=1"))
