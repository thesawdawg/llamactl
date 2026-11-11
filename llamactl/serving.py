"""What is being served right now: one snapshot shared by the Home and Serving views.

`gather` runs off the UI thread (it makes blocking HTTP calls) and returns a `ServingSnapshot`;
the pure helpers turn a snapshot into the texts the views show. Nothing here imports Textual.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import httpx

from .gateway import LOADED, LOADING, SLEEPING, Gateway, GatewayError, GatewayModel, GatewayStatus
from .sessions import Session, SessionStore
from .usage import Usage

HEALTH_PROBE_TIMEOUT = 1.0
STATE_READY, STATE_LOADING, STATE_DOWN, STATE_CRASHED = "ready", "loading", "down", "crashed"
MAX_PROBES = 8                   # standalone servers probed at once

# Gateway states shown on the Home card.
GW_STOPPED, GW_STARTING, GW_RUNNING, GW_UNHEALTHY, GW_STOPPING = (
    "stopped", "starting", "running", "unhealthy", "stopping")
START_GRACE = 30.0               # seconds a freshly started router may take to answer /health

Health = Callable[[str], str]


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


@dataclass(frozen=True)
class ServingSnapshot:
    """The gateway and the standalone servers at one moment.

    Attributes:
        status: Gateway process status.
        models: The router's models, or None when it is stopped or could not be asked.
        error: Why `models` is None while the gateway runs ("" otherwise).
        sessions: Standalone and attached servers.
        health: Health state by session id ("ready", "loading", "down", "crashed").
        started: Unix time the gateway started, 0 when unknown or stopped.
        log: Path of the gateway log, "" when stopped.
        at: Unix time of the snapshot.
        usage: Per-process usage by row key ("gw:<model>", "s:<session id>"); not part of
            `same_as`.
    """

    status: GatewayStatus
    models: tuple[GatewayModel, ...] | None = None
    error: str = ""
    sessions: tuple[Session, ...] = ()
    health: dict[str, str] = field(default_factory=dict)
    started: float = 0.0
    log: str = ""
    at: float = 0.0
    usage: dict[str, Usage] = field(default_factory=dict)

    @property
    def idle(self) -> bool:
        """True when nothing is running (no gateway, no standalone or attached server)."""
        return not self.status.running and not self.sessions

    @property
    def running_models(self) -> list[GatewayModel]:
        """Gateway models that are loaded, loading or sleeping."""
        return [m for m in self.models or () if m.state in (LOADED, LOADING, SLEEPING)]

    def same_as(self, other: "ServingSnapshot | None") -> bool:
        """True when `other` shows the same state (ignoring the timestamp)."""
        return other is not None and (
            self.status, self.models, self.error, self.sessions, self.health, self.started,
            self.log) == (other.status, other.models, other.error, other.sessions, other.health,
                          other.started, other.log)


def _session_state(s: Session, health: Health) -> str:
    """Health of one standalone or attached server."""
    if not s.external and not s.alive:
        return STATE_CRASHED
    return health(s.url)


def gather(gw: Gateway, store: SessionStore, health: Health = http_health,
           now: Callable[[], float] = time.time,
           usage: Callable[[GatewayStatus, tuple[GatewayModel, ...] | None,
                            tuple[Session, ...]], dict[str, Usage]] | None = None,
           ) -> ServingSnapshot:
    """Collect the gateway status, its models and every session's health.

    Blocking: call it from a worker thread. Nothing is probed over HTTP while nothing runs.

    Args:
        gw: The gateway.
        store: Standalone and attached sessions.
        health: Returns "ready", "loading" or "down" for a base URL.
        now: Wall clock.
        usage: Optional per-process usage collector, called only while something runs.

    Returns:
        The snapshot; a gateway that cannot be asked for its models is reported in `error`.
    """
    status = gw.status()
    models: tuple[GatewayModel, ...] | None = None
    error = ""
    state = gw.state if status.running else None
    if status.running:
        try:
            models = tuple(gw.models())
        except GatewayError as e:
            error = str(e)
    sessions = tuple(store.sessions)
    states: dict[str, str] = {}
    if sessions:
        with ThreadPoolExecutor(max_workers=min(MAX_PROBES, len(sessions))) as ex:
            states = dict(zip((s.id for s in sessions),
                              ex.map(lambda s: _session_state(s, health), sessions)))
    used = usage(status, models, sessions) if usage is not None and (
        status.running or sessions) else {}
    return ServingSnapshot(status, models, error, sessions, states,
                           state.started if state else 0.0, state.log if state else "", now(),
                           used)


def gateway_state(snap: ServingSnapshot | None, busy: str = "", now: float | None = None) -> str:
    """The state of the gateway as the Home card words it.

    Args:
        snap: The latest snapshot, or None before the first one.
        busy: "starting" or "stopping" while the app is doing that, else "".
        now: Unix time; defaults to the wall clock.

    Returns:
        "stopped", "starting", "running", "unhealthy" or "stopping".
    """
    if busy:
        return busy
    if snap is None or not snap.status.running:
        return GW_STOPPED
    if snap.status.healthy:
        return GW_RUNNING
    when = time.time() if now is None else now
    return GW_STARTING if snap.started and when - snap.started < START_GRACE else GW_UNHEALTHY


def uptime(started: float, now: float | None = None) -> str:
    """Seconds since `started` as ``45s``, ``12m``, ``1h02m`` or ``2d``; ``-`` when unknown."""
    if not started:
        return "-"
    s = max(0, int((time.time() if now is None else now) - started))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 86400}d"
