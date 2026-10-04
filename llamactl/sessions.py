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

from .command import build_command
from .config import CONFIG_DIR, LOG_DIR, Config

STATE_FILE = CONFIG_DIR / "sessions.json"
BASE_PORT = 8080


@dataclass
class Session:
    """A llama-server we manage (has pid + log) or merely attach to (external)."""

    id: str
    name: str
    url: str
    pid: int = 0
    log: str = ""
    external: bool = False
    started: float = 0.0

    @property
    def alive(self) -> bool:
        """Managed: process exists. External: assumed present until health fails."""
        if self.external:
            return True
        try:
            if os.waitpid(self.pid, os.WNOHANG) != (0, 0):
                return False
        except ChildProcessError:
            pass
        try:
            os.kill(self.pid, 0)
            return True
        except OSError:
            return False


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
    """Persisted list of sessions so the TUI can be closed and reopened."""

    def __init__(self) -> None:
        self.sessions: list[Session] = []
        self.load()

    def load(self) -> None:
        """Read state, dropping dead managed sessions."""
        raw = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else []
        self.sessions = [s for s in (Session(**r) for r in raw) if s.alive]
        self.save()

    def save(self) -> None:
        """Write state to disk."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps([asdict(s) for s in self.sessions], indent=2))

    def _ports(self) -> set[int]:
        return {int(s.url.rsplit(":", 1)[1]) for s in self.sessions if s.url.rsplit(":", 1)[1].isdigit()}

    def launch_server(self, cfg: Config, model: str) -> Session:
        """Start a detached llama-server; it survives TUI exit."""
        prof = cfg.profile_for(model)
        port = prof.port or next_free_port(self._ports(), prof.host)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        sid = f"{Path(model).stem}-{port}"
        log = LOG_DIR / f"{sid}.log"
        with log.open("ab") as fh:
            proc = subprocess.Popen(
                build_command(cfg, model, prof, "server", port),
                stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        s = Session(sid, Path(model).stem, f"http://{prof.host}:{port}", proc.pid, str(log), started=time.time())
        self.sessions.append(s)
        self.save()
        return s

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
