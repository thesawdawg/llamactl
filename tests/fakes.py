"""Test doubles for the TUI: gateway, session store and tune runner that never touch a process."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Callable

from llamactl.config import PRESET_SEP, Config, Profile
from llamactl.gateway import (FAILED, LOADED, UNLOADED, GatewayError, GatewayModel,
                              GatewayState, GatewayStatus)
from llamactl.gguf import ModelInfo
from llamactl.hardware import Device, Host
from llamactl.measure import Measurement
from llamactl.sessions import Session
from llamactl.support import ArchSupport, Support, unknown_arch

URL = "http://127.0.0.1:8080"


def sparse_copy(src: Path, dst: Path) -> Path:
    """Copy a (sparse) test GGUF without materialising its padding: header bytes, then the same size.

    `shutil.copy` writes every zero block, which filled the temp volume over a few hundred tests.

    Args:
        src: A synthetic GGUF whose header is within its first 64 KiB.
        dst: Where to put the copy.

    Returns:
        `dst`.
    """
    with src.open("rb") as fh:
        head = fh.read(1 << 16)
    dst.write_bytes(head)
    with dst.open("r+b") as fh:
        fh.truncate(src.stat().st_size)
    return dst


def make_info(path: str | Path = "/m/qwen3.gguf", size: int = int(4e9), pooling: int = -1,
              ctx_train: int = 32768) -> ModelInfo:
    """A small dense model's metadata (about 4 GB of weights)."""
    return ModelInfo(path=Path(path), size=size, architecture="x", name="m", n_layer=32,
                     n_embd=4096, n_head=32, n_head_kv=8, head_dim_k=128, head_dim_v=128,
                     ctx_train=ctx_train, expert_count=0, file_type=0, sharded=1,
                     parameter_count=0, pooling_type=pooling)


def make_host(vram: float = 16e9, free: float | None = None) -> Host:
    """A single-GPU host."""
    return Host(gpus=[Device.of("CUDA0", "GPU", int(vram), int(vram if free is None else free))],
                ram_total=int(24e9), ram_free=int(24e9), cpu_threads=16, cpu_cores=8,
                cpu_name="Test CPU")


class FakeGateway:
    """Stands in for `Gateway`: scripted states, recorded calls, optional injected errors.

    Model names follow the config like the real router's listing (``alias`` for the active
    preset, ``alias@preset`` for the others) and are refreshed by `reload`.

    Args:
        cfg: Config whose presets name the models.
        running: The router process exists from the start.
        healthy: It answers ``/health`` while running.
        load_state: State a model reaches when asked to load ("loaded", "loading", "failed").
    """

    def __init__(self, cfg: Config | None = None, *, running: bool = False, healthy: bool = True,
                 load_state: str = LOADED) -> None:
        self.cfg = cfg or Config()
        self.problems: list[str] = []
        self.healthy = healthy
        self.load_state = load_state
        self.calls: list[tuple] = []
        self.errors: dict[str, Exception] = {}
        self.states: dict[str, GatewayModel] = {}
        self._running = running
        self.host_override: str | None = None
        self.bound_host = self.cfg.gateway.host      # the listen address of the running router
        self.started = time.time() - 120
        self.log = "/nonexistent/gateway.log"
        if running:
            self._sync()

    # -- helpers -----------------------------------------------------------------------

    def fail(self, method: str, error: Exception | None = GatewayError("scripted failure")) -> None:
        """Make `method` raise `error` from now on (None clears it)."""
        if error is None:
            self.errors.pop(method, None)
        else:
            self.errors[method] = error

    def set_state(self, name: str, state: str, exit_code: int | None = None) -> None:
        """Set a model's state as the router would report it."""
        self.states[name] = GatewayModel(name, state, exit_code)

    def names(self) -> list[str]:
        """Model ids the config implies."""
        out = []
        for ent in self.cfg.models.values():
            for pname in ent.presets:
                out.append(ent.alias if pname == ent.active else f"{ent.alias}{PRESET_SEP}{pname}")
        return sorted(out)

    def _sync(self) -> None:
        names = self.names()
        self.states = {n: self.states.get(n, GatewayModel(n, UNLOADED)) for n in names}

    def _check(self, method: str) -> None:
        self.calls.append((method,))
        if method in self.errors:
            raise self.errors[method]

    # -- the Gateway API ---------------------------------------------------------------

    @property
    def listen_url(self) -> str:
        """Base URL implied by the settings."""
        return f"http://127.0.0.1:{self.cfg.gateway.port}"

    @property
    def state(self) -> GatewayState | None:
        """Remembered state while running."""
        return GatewayState(4242, self.listen_url, self.log, self.started, self.bound_host) \
            if self._running else None

    @property
    def running(self) -> bool:
        """The router process exists."""
        return self._running

    def start(self) -> GatewayState:
        """Start (or find running) the router."""
        self._check("start")
        if not self._running:
            self.bound_host = (self.cfg.gateway.host if self.host_override is None
                               else self.host_override)
        self._running = True
        self.started = time.time()
        self._sync()
        return self.state  # type: ignore[return-value]

    def stop(self) -> bool:
        """Stop the router."""
        self._check("stop")
        was, self._running, self.states = self._running, False, {}
        return was

    def status(self) -> GatewayStatus:
        """Process status."""
        self.calls.append(("status",))
        return GatewayStatus(self._running, self.listen_url, self._running and self.healthy,
                             4242 if self._running else 0,
                             self.bound_host if self._running else self.cfg.gateway.host)

    def models(self) -> list[GatewayModel]:
        """The router's models."""
        self.calls.append(("models",))
        if "models" in self.errors:
            raise self.errors["models"]
        if not self._running:
            raise GatewayError("the gateway is not running")
        return list(self.states.values())

    def load(self, name: str) -> None:
        """Load a model (instantly reaches `load_state`)."""
        self.calls.append(("load", name))
        if "load" in self.errors:
            raise self.errors["load"]
        if name not in self.states:
            raise GatewayError(f"cannot load '{name}': unknown model")
        self.states[name] = GatewayModel(name, self.load_state,
                                         1 if self.load_state == FAILED else None)

    def unload(self, name: str) -> None:
        """Unload a model."""
        self.calls.append(("unload", name))
        if "unload" in self.errors:
            raise self.errors["unload"]
        if name not in self.states:
            raise GatewayError(f"cannot unload '{name}': unknown model")
        self.states[name] = GatewayModel(name, UNLOADED)

    def reload(self) -> list[GatewayModel]:
        """Re-read the presets."""
        self.calls.append(("reload",))
        if "reload" in self.errors:
            raise self.errors["reload"]
        if not self._running:
            return []
        self._sync()
        return list(self.states.values())

    def write_presets(self) -> Path:
        """Pretend to write the INI."""
        self.calls.append(("write_presets",))
        return Path("/nonexistent/gateway.ini")

    def called(self, method: str) -> list[tuple]:
        """Recorded calls of one method."""
        return [c for c in self.calls if c[0] == method]


class FakeStore:
    """Stands in for `SessionStore`: sessions in memory, launches recorded."""

    def __init__(self, sessions: list[Session] | None = None) -> None:
        self.sessions: list[Session] = list(sessions or [])
        self.calls: list[tuple] = []
        self.error: Exception | None = None

    def launch_server(self, cfg: Config, model: str, preset_name: str | None = None,
                      port: int | None = None, host: str | None = None) -> Session:
        """Record a launch and add a session that looks alive (this process's pid)."""
        self.calls.append(("launch", model, preset_name, port))
        if self.error:
            raise self.error
        ent = cfg.models.get(model)
        explicit = preset_name is not None and (ent is None or preset_name != ent.active)
        name = Path(model).stem + (PRESET_SEP + preset_name if explicit else "")
        preset = cfg.preset(model, preset_name)
        bind = host if host is not None else (preset.profile if preset else
                                              cfg.profile_for(model)).host
        s = Session(f"{name}-9001", name, "http://127.0.0.1:9001", os.getpid(),
                    "/nonexistent/s.log", started=time.time(), host=bind)
        self.sessions.append(s)
        return s

    def attach(self, url: str) -> Session:
        """Record an attach and add an external session."""
        self.calls.append(("attach", url))
        if self.error:
            raise self.error
        url = url if "://" in url else f"http://{url}"
        s = Session(f"ext-{url.split('//')[1]}", "external", url, external=True,
                    started=time.time())
        self.sessions.append(s)
        return s

    def stop(self, s: Session) -> None:
        """Record a stop and forget the session."""
        self.calls.append(("stop", s.id))
        self.sessions = [x for x in self.sessions if x.id != s.id]


def ok(gen_tps: float = 40.0, vram: int = int(7e9)) -> Measurement:
    """A successful measurement."""
    return Measurement(ok=True, load_s=3.0, vram_used=vram, prompt_tps=900.0, gen_tps=gen_tps,
                       at="2026-11-09T12:00:00+00:00", build="b1")


def oom() -> Measurement:
    """A failed (out of memory) measurement."""
    return Measurement(error="out of memory: cudaMalloc failed", at="2026-11-09T12:00:00+00:00",
                       build="b1")


def cancelled() -> Measurement:
    """What the real runner returns for a trial stopped on request."""
    return Measurement(error="cancelled: stopped while the model was loading",
                       at="2026-11-09T12:00:00+00:00", build="b1")


class FakeRunner:
    """A tune `Runner` returning scripted measurements, one per call (the last repeats).

    Args:
        results: Measurements in order.
        gates: Optional events by 0-based call index; the call blocks until its event is set,
            so a test can cancel while a trial is "running".
        honour_cancel: A gated call returns a "cancelled" measurement as soon as the `cancel`
            callable says so (like `ServerRunner`); off, it waits for its gate regardless.
    """

    def __init__(self, *results: Measurement, gates: dict[int, threading.Event] | None = None,
                 honour_cancel: bool = True) -> None:
        self.results = list(results) or [ok()]
        self.gates = gates or {}
        self.honour_cancel = honour_cancel
        self.cancelled_calls: list[int] = []
        self.calls: list[tuple[Path, Profile]] = []
        self.started = threading.Event()

    def measure(self, model: Path, profile: Profile, cancel=None) -> Measurement:
        """Return the next scripted measurement (or "cancelled" when asked to stop meanwhile)."""
        i = len(self.calls)
        self.calls.append((model, profile))
        self.started.set()
        gate = self.gates.get(i)
        if gate is not None:
            for _ in range(500):                       # at most 10 s
                if gate.wait(0.02):
                    break
                if self.honour_cancel and cancel is not None and cancel():
                    self.cancelled_calls.append(i)
                    return cancelled()
        return self.results[min(i, len(self.results) - 1)]


def runner_factory(runner: FakeRunner) -> Callable[[], FakeRunner]:
    """A factory handing out the same runner."""
    return lambda: runner


async def wait_for(pilot, cond: Callable[[], bool], timeout: float = 10.0,
                   step: float = 0.01) -> bool:
    """Poll `cond` while the app's workers settle; True as soon as it holds, False at `timeout`.

    The condition is only looked at after the app had a moment to settle. When the active screen
    changed while waiting, the new screen also gets a moment to finish composing (a test that
    ends mid-mount trips over half-built widgets).
    """
    deadline = time.monotonic() + timeout
    screen = pilot.app.screen
    while True:
        await pilot.pause(step)
        if cond():
            if pilot.app.screen is not screen:
                await pilot.pause()
            return True
        if time.monotonic() >= deadline:
            return False


def plain(widget) -> str:
    """The text a Static/Label currently shows."""
    return str(widget.render())


class FakeSupport:
    """Stands in for `SupportChecker` in tests: every architecture is OK unless scripted.

    Args:
        statuses: Architecture name -> `ArchSupport` to return for it (others check OK).
    """

    def __init__(self, statuses: dict[str, ArchSupport] | None = None) -> None:
        self.statuses = statuses or {}
        self.recorded: list[tuple[str, str]] = []
        self.refreshed: list[bool] = []

    def check(self, arch: str, build: str = "") -> ArchSupport:
        """The scripted verdict of `arch`, or `ArchSupport(arch, OK, build)`."""
        return self.statuses.get(arch, ArchSupport(arch, Support.OK, build))

    def learned(self, build: str) -> frozenset[str]:
        """No learned failures."""
        return frozenset()

    def record_failure(self, arch: str, build: str) -> None:
        """Remember the call (no state file)."""
        self.recorded.append((arch, build))

    def learn(self, text: str, build: str) -> None:
        """Record the architecture `text` names as unknown, like `SupportChecker.learn`."""
        if arch := unknown_arch(text):
            self.record_failure(arch, build)

    def upstream_names(self) -> frozenset[str] | None:
        """No upstream list."""
        return None

    def upstream_stale(self) -> bool:
        """Always stale, so a refresh is tried when a check blocks."""
        return True

    def refresh_upstream(self, force: bool = False) -> bool:
        """Pretend the upstream list is fresh (no network)."""
        self.refreshed.append(force)
        return True
