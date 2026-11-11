"""Per-process resource usage of what is served: CPU, RAM and VRAM.

Linux reads ``/proc``; macOS shells out to ``ps``. Per-process device memory comes from
``nvidia-smi`` when the driver reports it, else a tuned measurement or a planner estimate is
shown (marked ``~``). Nothing here imports Textual; probing is blocking and belongs off the
UI thread.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from . import hardware, planner
from .calibration import Calibration
from .config import PRESET_SEP, Config
from .gateway import LOADED, LOADING, SLEEPING, GatewayModel, GatewayStatus
from .gguf import ModelInfo
from .hardware import Runner
from .sessions import Session

CPU_SAMPLE = 0.5                # seconds between the two samples `ps` takes

# Values of `Usage.vram_source`.
SRC_PROCESS, SRC_TUNED, SRC_ESTIMATE, SRC_UNKNOWN = "process", "tuned", "estimate", ""

PROC_TIMEOUT = 5.0              # seconds a ps/nvidia-smi call may take


@dataclass(frozen=True)
class Usage:
    """What one serving process uses right now.

    Attributes:
        pid: Process id.
        rss: Resident system memory in bytes (0 when unknown).
        cpu_pct: CPU since the previous sample, in percent of one core (like ``top``; can
            exceed 100), None when there is no earlier sample to compare with.
        vram: Device memory in bytes, None when unknown.
        vram_source: Where `vram` came from: ``"process"``, ``"tuned"``, ``"estimate"`` or
            "" (unknown).
    """

    pid: int
    rss: int = 0
    cpu_pct: float | None = None
    vram: int | None = None
    vram_source: str = SRC_UNKNOWN


def _argv_value(argv: Iterable[str], flag: str) -> str | None:
    """Value of a ``--flag value`` (or ``--flag=value``) option in an argv, or None."""
    args = list(argv)
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _parse_ps_time(text: str) -> float | None:
    """Seconds of CPU time in a macOS ``ps`` TIME value like ``1-02:03:04.56``."""
    m = re.match(r"^(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$", text.strip())
    if not m:
        return None
    days, hours, minutes, seconds = m.groups()
    return int(days or 0) * 86400 + int(hours or 0) * 3600 + int(minutes) * 60 + float(seconds)


class UsageProbe:
    """Reads process CPU time, RSS and per-process GPU memory.

    `sample` keeps the last reading per pid; the returned percent is the delta since the
    previous sample of that pid (None on first sight), so it measures current usage, not a
    lifetime average. Every read failure — a process exiting mid-read, permissions — is
    skipped silently.

    Args:
        runner: ``subprocess.run`` compatible callable for ``ps`` and ``nvidia-smi``.
        proc: The procfs mount point.
        platform: ``sys.platform`` value to probe as; None uses the real one.
        clock: Monotonic clock the CPU deltas are computed against.
    """

    def __init__(self, runner: Runner = subprocess.run, proc: Path = Path("/proc"),
                 platform: str | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._runner = runner
        self._proc = proc
        self._platform = platform if platform is not None else sys.platform
        self._clock = clock
        self._prev: dict[int, tuple[float, float]] = {}    # pid -> (clock, cpu seconds)

    def sample(self, pids: Iterable[int]) -> dict[int, tuple[int, float | None]]:
        """RSS and current CPU of each live pid.

        Args:
            pids: Process ids to read.

        Returns:
            pid -> (rss bytes, cpu percent since the previous sample, None on first sight).
            Pids that vanished are dropped and forgotten.
        """
        now = self._clock()
        raw = self._sample_darwin(list(pids)) if self._platform == "darwin" else {
            p: r for p in pids if (r := self._sample_linux(p)) is not None}
        out: dict[int, tuple[int, float | None]] = {}
        prev, self._prev = self._prev, {}
        for pid, (rss, cpu_s) in raw.items():
            cpu_pct: float | None = None
            if pid in prev and now > prev[pid][0]:
                cpu_pct = max(0.0, (cpu_s - prev[pid][1]) / (now - prev[pid][0]) * 100.0)
            out[pid] = (rss, cpu_pct)
            self._prev[pid] = (now, cpu_s)
        return out

    def children(self, pid: int) -> dict[int, list[str]]:
        """Direct child processes of `pid`.

        Args:
            pid: Parent process id.

        Returns:
            Child pid -> argv; empty when there are none or they cannot be listed.
        """
        if self._platform == "darwin":
            return self._children_darwin(pid)
        out: dict[int, list[str]] = {}
        try:
            entries = list(self._proc.iterdir())
        except OSError:
            return out
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text()
                # comm is in parentheses and may itself hold spaces and parens: split at
                # the LAST ')'; the field after it is state, then ppid.
                fields = stat.rsplit(")", 1)[1].split()
                if int(fields[1]) != pid:
                    continue
                argv = (entry / "cmdline").read_bytes().split(b"\0")
                out[int(entry.name)] = [a.decode("utf-8", "replace") for a in argv if a]
            except (OSError, ValueError, IndexError):
                continue
        return out

    def gpu_memory(self) -> dict[int, int]:
        """Device memory per process from nvidia-smi: pid -> bytes, {} when unavailable."""
        return hardware.nvidia_smi_process_memory(self._runner)

    # -- platform internals ------------------------------------------------------------

    def _sample_linux(self, pid: int) -> tuple[int, float] | None:
        """(rss bytes, cpu seconds) of one pid from procfs, or None when unreadable."""
        try:
            stat = (self._proc / str(pid) / "stat").read_text()
            fields = stat.rsplit(")", 1)[1].split()
            cpu_s = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
            rss = 0
            for line in (self._proc / str(pid) / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) << 10
                    break
            return rss, cpu_s
        except (OSError, ValueError, IndexError):
            return None

    def _ps(self, *args: str) -> str:
        """One ``ps`` call; its stdout, or "" on any failure."""
        try:
            r = self._runner(["ps", *args], capture_output=True, text=True,
                             timeout=PROC_TIMEOUT)
            return str(r.stdout or "")
        except (OSError, subprocess.SubprocessError, AttributeError):
            return ""

    def _sample_darwin(self, pids: list[int]) -> dict[int, tuple[int, float]]:
        """`sample` via one ps call; rss is KiB, TIME holds centiseconds."""
        if not pids:
            return {}
        text = self._ps("-o", "pid=,rss=,time=", "-p", ",".join(str(p) for p in pids))
        out: dict[int, tuple[int, float]] = {}
        for line in text.splitlines():
            parts = line.split()
            if len(parts) != 3:
                continue
            try:
                pid, rss_kb = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            cpu_s = _parse_ps_time(parts[2])
            if cpu_s is not None:
                out[pid] = (rss_kb << 10, cpu_s)
        return out

    def _children_darwin(self, pid: int) -> dict[int, list[str]]:
        """`children` via one ps process listing (macOS has no procfs)."""
        out: dict[int, list[str]] = {}
        for line in self._ps("-axo", "pid=,ppid=,args=").splitlines():
            try:
                parts = line.split()
                if len(parts) > 1 and int(parts[1]) == pid:
                    out[int(parts[0])] = parts[2:]
            except (ValueError, IndexError):
                continue
        return out


def estimator(cfg: Config, host: hardware.Host, calibration: Calibration,
              info_for: Callable[[str], ModelInfo | None]) \
        -> Callable[[str], tuple[int | None, str]]:
    """VRAM guesser for a served name, for when the driver reports no per-process memory.

    The preset's tuned measurement wins when it is ok; otherwise the planner estimate, capped
    at the host's total device memory.

    Args:
        cfg: Config with the models and their presets.
        host: Host snapshot; without GPUs nothing is estimated.
        calibration: Measured trials scaling estimates.
        info_for: Model metadata by config model path, or None.

    Returns:
        A function mapping ``alias``/``alias@preset`` to (bytes, source) or (None, "") when
        the name or its preset is unknown, the model info is missing, or there is no GPU.
    """
    def estimate(name: str) -> tuple[int | None, str]:
        """(vram bytes, source) for one served name."""
        alias, _, preset_name = name.partition(PRESET_SEP)
        for path, ent in cfg.models.items():
            if ent.alias != alias:
                continue
            preset = ent.presets.get(preset_name or ent.active)
            if preset is None:
                return None, SRC_UNKNOWN
            measured = preset.measured
            if measured is not None and measured.ok and measured.vram_used > 0:
                return measured.vram_used, SRC_TUNED
            info = info_for(path)
            if info is None or not host.vram_total:
                return None, SRC_UNKNOWN
            budget = planner.estimate_budget(info, preset.profile, host, calibration).total()
            return min(budget, host.vram_total), SRC_ESTIMATE
        return None, SRC_UNKNOWN
    return estimate


def served_usage(probe: UsageProbe, status: GatewayStatus,
                 models: Iterable[GatewayModel] | None, sessions: Iterable[Session],
                 estimate: Callable[[str], tuple[int | None, str]]) -> dict[str, Usage]:
    """CPU, RAM and VRAM of every served process, keyed like the Serving view rows.

    Gateway models are matched to the router's child processes by their ``--port`` argument,
    falling back to ``--alias``; only loaded, loading and sleeping models count. A sleeping
    model keeps its process but has released its weights, so without a driver report its VRAM
    is unknown — never estimated. Managed sessions count by their pid; attached sessions
    (pid 0) are skipped.

    Args:
        probe: Process probe; consecutive `sample` calls give the CPU deltas.
        status: Gateway status; children are inspected only while it runs.
        models: The router's models, or None.
        sessions: Standalone and attached servers.
        estimate: Fallback (bytes, source) by served name, e.g. from `estimator`.

    Returns:
        ``gw:<model name>`` / ``s:<session id>`` -> Usage.
    """
    targets: dict[int, tuple[str, str, bool]] = {}    # pid -> (key, name, sleeping)
    if status.running and status.pid:
        live = [m for m in models or () if m.state in (LOADED, LOADING, SLEEPING)]
        for cpid, argv in probe.children(status.pid).items():
            port, alias = _argv_value(argv, "--port"), _argv_value(argv, "--alias")
            model = next((m for m in live if port not in (None, "0")
                          and _argv_value(m.args, "--port") == port), None)
            if model is None and alias:
                model = next((m for m in live if m.name == alias), None)
            if model is not None:
                targets[cpid] = (f"gw:{model.name}", model.name, model.state == SLEEPING)
    for s in sessions:
        if s.pid:
            targets[s.pid] = (f"s:{s.id}", s.name, False)
    if not targets:
        return {}
    sampled = probe.sample(targets)
    vram = probe.gpu_memory()
    out: dict[str, Usage] = {}
    for pid, (key, name, sleeping) in targets.items():
        rss, cpu_pct = sampled.get(pid, (0, None))
        mem = vram.get(pid)
        source = SRC_PROCESS if mem is not None else SRC_UNKNOWN
        if mem is None and not sleeping:
            mem, source = estimate(name)
        out[key] = Usage(pid, rss, cpu_pct, mem, source)
    return out


def format_cpu(cpu_pct: float | None) -> str:
    """CPU percent like ``12%``; ``-`` when unknown."""
    return "-" if cpu_pct is None else f"{cpu_pct:.0f}%"


def format_mem(nbytes: int | None, source: str = SRC_PROCESS) -> str:
    """Bytes like ``420 MB`` or ``1.2 GB``.

    Tuned and estimated values get a ``~`` prefix; unknown is ``-``.
    """
    if not nbytes:
        return "-"
    text = f"{nbytes / 1e9:.1f} GB" if nbytes >= 1e9 else f"{nbytes / 1e6:.0f} MB"
    return f"~{text}" if source in (SRC_TUNED, SRC_ESTIMATE) else text
