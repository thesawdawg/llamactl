"""Session tracking: managed llama-server processes and attached external servers."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from . import network
from .command import build_command
from .config import CONFIG_DIR, LOG_DIR, PRESET_SEP, Config, Profile

STATE_FILE = CONFIG_DIR / "sessions.json"
BASE_PORT = 8080


def pid_alive(pid: int) -> bool:
    """True when the process exists and is not a zombie of ours.

    A child of this process that already exited is reaped here, so it does not look alive.

    Args:
        pid: Process id (0 or negative is never alive).

    Returns:
        Whether the process is running.
    """
    if pid <= 0:
        return False
    try:
        if os.waitpid(pid, os.WNOHANG) != (0, 0):
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@dataclass
class Session:
    """A llama-server we manage (has pid + log) or merely attach to (external).

    Attributes:
        id: Unique id (``name-port``).
        name: Display name (model stem, plus ``@preset`` when not the active one).
        url: Base URL llamactl connects to (loopback for a wildcard bind; never ``0.0.0.0``).
        pid: Process id, 0 for an attached server.
        log: Log file path, "" for an attached server.
        external: Attached, not started by llamactl.
        started: Unix time of the start.
        host: The listen address a managed server was started with; "" for an attached one or
            a session saved before this was recorded.
    """

    id: str
    name: str
    url: str
    pid: int = 0
    log: str = ""
    external: bool = False
    started: float = 0.0
    host: str = ""

    @property
    def urls(self) -> list[str]:
        """The base URLs to give a client (the LAN ones too for a wildcard bind)."""
        return network.client_urls_for(self.url, self.host)

    @property
    def alive(self) -> bool:
        """Managed: process exists. External: assumed present until health fails."""
        return True if self.external else pid_alive(self.pid)


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    """True if the port can be bound (nothing holds it).

    A connect test hangs on firewalled/mirrored ports; a bind test cannot.
    """
    with socket.socket() as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def next_free_port(taken: set[int], host: str = "127.0.0.1") -> int:
    """First free port from BASE_PORT that is not in `taken`.

    Raises:
        RuntimeError: after 200 candidates (avoids scanning forever).
    """
    port = BASE_PORT
    for _ in range(200):
        if port not in taken and port_free(port, host):
            return port
        port += 1
    raise RuntimeError("no free port")


def probe(url: str, timeout: float = 1.0) -> dict | None:
    """Return /props JSON of a llama-server, or None if unreachable."""
    try:
        r = httpx.get(f"{url}/props", timeout=timeout)
        return r.json() if r.status_code == 200 else {}
    except (httpx.HTTPError, ValueError):
        return None


class SessionStore:
    """Persisted list of sessions so the TUI can be closed and reopened.

    Args:
        persist: Write the pruned list back when loading. False for commands that only look
            (``llamactl ps``) and must not touch ``sessions.json``.
    """

    def __init__(self, persist: bool = True) -> None:
        self.sessions: list[Session] = []
        self.persist = persist
        self.load()

    def load(self) -> None:
        """Read state, dropping dead managed sessions (and saving, unless read-only)."""
        raw = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else []
        self.sessions = [s for s in (Session(**r) for r in raw) if s.alive]
        if self.persist:
            self.save()

    def save(self) -> None:
        """Write state to disk."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps([asdict(s) for s in self.sessions], indent=2))

    def _ports(self) -> set[int]:
        return {int(s.url.rsplit(":", 1)[1]) for s in self.sessions if s.url.rsplit(":", 1)[1].isdigit()}

    def launch_server(self, cfg: Config, model: str, preset_name: str | None = None,
                      port: int | None = None, host: str | None = None) -> Session:
        """Start a detached llama-server; it survives TUI exit.

        Args:
            cfg: Global config.
            model: Path to the GGUF file.
            preset_name: Preset to serve; None uses the model's active preset, or the default
                profile when the model has none. The session id and name carry the preset
                (``stem@preset``) when it is not the active one.
            port: Listen port overriding the preset's; None keeps the preset's, else the first
                free port from `BASE_PORT`.
            host: Listen address for this server only (not saved) instead of the preset's;
                None keeps the preset's.

        Returns:
            The new session (also stored and saved).

        Raises:
            LookupError: When `preset_name` is not a preset of the model.
            RuntimeError: When no free port is found, or the requested `port` is in use.
            OSError: When the process or its log cannot be started.
        """
        prof = self._profile(cfg, model, preset_name)
        if host is not None:
            prof = prof.merged(host=host)
        if port is not None:
            if port in self._ports() or not port_free(port, prof.host):
                raise RuntimeError(f"port {port} on {prof.host} is already in use")
        else:
            port = prof.port or next_free_port(self._ports(), prof.host)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ent = cfg.models.get(model)
        explicit = preset_name is not None and (ent is None or preset_name != ent.active)
        name = Path(model).stem + (PRESET_SEP + preset_name if explicit else "")
        sid = f"{name}-{port}"
        log = LOG_DIR / f"{sid}.log"
        with log.open("ab") as fh:
            proc = subprocess.Popen(
                build_command(cfg, model, prof, port),
                stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        s = Session(sid, name, f"http://{network.probe_host(prof.host)}:{port}", proc.pid, str(log),
                    started=time.time(), host=prof.host)
        self.sessions.append(s)
        self.save()
        return s

    @staticmethod
    def _profile(cfg: Config, model: str, preset_name: str | None) -> Profile:
        """Profile to launch: the named preset, else the active one, else the default profile.

        Raises:
            LookupError: When a named preset does not exist.
        """
        if preset_name is None:
            return cfg.profile_for(model)
        preset = cfg.preset(model, preset_name)
        if preset is None:
            have = ", ".join(cfg.models[model].presets) if model in cfg.models else "none"
            raise LookupError(f"{Path(model).stem} has no preset '{preset_name}' (has: {have or 'none'})")
        return preset.profile

    def attach(self, url: str) -> Session:
        """Register an already-running server by URL."""
        url = url.rstrip("/")
        if "://" not in url:
            url = f"http://{url}"
        props = probe(url) or {}
        name = Path(props.get("model_path", "") or "external").stem
        s = Session(f"ext-{url.split('//')[1]}", name, url, external=True, started=time.time())
        self.sessions = [x for x in self.sessions if x.id != s.id] + [s]
        self.save()
        return s

    def stop(self, s: Session) -> None:
        """Terminate a managed process, or just forget an external one."""
        if not s.external and s.alive:
            try:
                os.killpg(os.getpgid(s.pid), signal.SIGTERM)
            except OSError:
                pass
        self.sessions = [x for x in self.sessions if x.id != s.id]
        self.save()
