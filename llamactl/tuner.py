"""Measured tuning: launch candidate plans for real, keep the best, and learn from the numbers.

The pure part (`SearchSpace`, `next_candidate`, `choose_best`, `tune`) decides which profile to
try next from what the previous trials measured. The only parts that touch processes or HTTP
are behind the `Runner` protocol; `ServerRunner` is the real one, with every subprocess,
HTTP, clock and device-memory call injectable so tests never start llama.cpp.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

import httpx

from . import hardware, planner
from .calibration import TUNED, Calibration, Trial
from .command import build_command
from .config import LOG_DIR, Config, Preset, Profile
from .gateway import terminate_group
from .gguf import ModelInfo
from .hardware import Device, Host
from .intents import KV_NEVER, Intent
from .measure import (ERR_ARGS, ERR_BENCH, ERR_CANCELLED, ERR_LOAD, ERR_OOM, ERR_TIMEOUT,
                      Measurement)
from .planner import F16, GB, Q8, VERDICT_CPU, VERDICT_GPU, VERDICT_NO, VERDICT_SPILL, Plan
from .sessions import next_free_port, pid_alive

LOOPBACK = "127.0.0.1"
SPEED_INTENT = "speed"

# How a candidate was derived (shown in progress events).
HOW_PLAN, HOW_CTX_DOWN, HOW_KV_Q8, HOW_SPILL = "plan", "ctx down", "q8_0 KV", "spill"
HOW_CTX_UP, HOW_THREADS = "ctx up", "threads"

# ServerRunner timing (seconds).
HEALTH_BASE = 60.0        # health timeout = HEALTH_BASE + HEALTH_PER_GB per GB of model file,
HEALTH_PER_GB = 20.0      # capped at HEALTH_MAX: ~50 MB/s covers a cold read from a slow disk
HEALTH_MAX = 1800.0
POLL_INTERVAL = 0.5
REQUEST_TIMEOUT = 5.0
BENCH_TIMEOUT = 300.0
STOP_GRACE = 15.0         # SIGTERM first; SIGKILL when the group is still alive after this
GONE_WAIT = 10.0          # after SIGKILL, how long to wait for the process to disappear
VERSION_TIMEOUT = 20.0
LOG_TAIL_BYTES = 65536
PREFLIGHT_MARGIN = 0.9    # refuse only when free memory is below 90% of the budget

BENCH_PROMPT = (
    "The lighthouse keeper climbed the spiral stairs each evening, counting the hundred and "
    "twelve steps out of habit, a lantern swinging from one hand and a thermos of tea in the "
    "other. Far below, the harbour lights flickered on one by one, and the fishing boats "
    "turned for home against a sky the colour of old copper. He had kept this light for "
    "thirty-one winters, and in all that time the sea had never once repeated itself. Tonight "
    "the barometer was falling, the gulls had gone quiet, and a long low swell was rolling in "
    "from the north-west. Write the next three paragraphs of the story, describing the storm "
    "and what the keeper does when the power fails.")
EMBED_TEXT = " ".join(["The quick brown fox jumps over the lazy dog near the quiet riverbank."] * 8)

_OOM_RE = re.compile(
    r"out of memory|failed to allocate|unable to allocate|cudamalloc failed|"
    r"insufficient memory|not enough memory|bad_alloc|outofdevicememory|"
    r"failed to fit params|cannot allocate memory|alloc_buffer", re.I)
_ARGS_RE = re.compile(r"invalid argument|unknown argument|unrecognized (?:argument|option)|"
                      r"unknown option|error while handling argument", re.I)


# --------------------------------------------------------------------------- data

@dataclass(frozen=True)
class Candidate:
    """A profile to try, with the context and verdict it implies.

    Attributes:
        profile: The settings to launch.
        ctx_per_slot: Context each slot gets (``ctx_size`` divided by the slots).
        verdict: "GPU" when the whole model is on the GPU, "GPU+CPU" when layers or experts
            may sit in RAM, "CPU" for a GPU-less run.
        how: How it was derived (``HOW_*``).
    """

    profile: Profile
    ctx_per_slot: int
    verdict: str
    how: str = HOW_PLAN

    @property
    def summary(self) -> str:
        """One line for progress displays, e.g. ``ctx down: 8192 ctx, f16 KV, GPU``."""
        threads = f", {self.profile.threads} threads" if self.how == HOW_THREADS else ""
        return (f"{self.how}: {self.ctx_per_slot} ctx, {self.profile.cache_type_k} KV, "
                f"{self.verdict}{threads}")


@dataclass(frozen=True)
class TrialEvent:
    """Progress report of the search.

    Attributes:
        index: 1-based number of the trial.
        total: Most trials that can still happen (``max_trials``); the search may stop earlier.
        phase: "start" before the launch, "done" after it.
        summary: Profile summary (`Candidate.summary`).
        measurement: The result on "done", else None.
    """

    index: int
    total: int
    phase: str
    summary: str
    measurement: Measurement | None = None


@dataclass(frozen=True)
class TuneResult:
    """Outcome of a tune.

    Attributes:
        intent: Intent key that was tuned.
        best: The winning candidate, None when nothing loaded.
        measurement: The winner's measurement, None when nothing loaded.
        trials: Every trial that ran, in order (also on failure), ready for `Calibration`.
        ok: A winner exists.
        message: Human readable outcome, or why there is no winner.
        cancelled: The search was cancelled before it finished.
    """

    intent: str
    best: Candidate | None
    measurement: Measurement | None
    trials: list[Trial] = field(default_factory=list)
    ok: bool = False
    message: str = ""
    cancelled: bool = False


class Runner(Protocol):
    """The only part of tuning that touches processes and HTTP."""

    def measure(self, model: Path, profile: Profile,
                cancel: Callable[[], bool] | None = None) -> Measurement:
        """Launch the model with a profile, benchmark it and stop it again.

        Args:
            model: GGUF file.
            profile: Settings to launch with.
            cancel: Polled while waiting; when it returns True the launch is stopped and a
                measurement whose error starts with ``cancelled`` is returned.

        Returns:
            The measurement; a failed load is ``ok=False`` with an ``error``, not an exception.
        """
        ...


History = list[tuple[Candidate, Measurement]]


# --------------------------------------------------------------------------- candidates

def profile_key(profile: Profile) -> tuple:
    """Hashable identity of a profile, to avoid trying the same one twice."""
    return tuple(sorted(profile.to_dict().items()))


def trial_verdict(profile: Profile, host: Host) -> str:
    """Where a concrete profile puts the model, by its settings (not by the estimate).

    Args:
        profile: The profile.
        host: Host snapshot.

    Returns:
        "CPU" without a GPU or with ``gpu_layers`` 0; "GPU" when every layer is on the GPU
        and no MoE experts are kept in RAM; otherwise "GPU+CPU" (layers placed by ``--fit``,
        a layer count, or expert offload).
    """
    if not host.gpus or profile.gpu_layers == "0":
        return VERDICT_CPU
    if profile.gpu_layers == "all" and profile.n_cpu_moe == 0:
        return VERDICT_GPU
    return VERDICT_SPILL


@dataclass(frozen=True)
class SearchSpace:
    """Everything candidate generation needs, with the step functions as methods.

    Attributes:
        intent: Intent being tuned.
        info: Model metadata.
        host: Host snapshot taken before the tune.
        pct: Headroom percentage.
    """

    intent: Intent
    info: ModelInfo
    host: Host
    pct: float = 10

    def candidate(self, profile: Profile, how: str) -> Candidate:
        """Wrap a profile with its context per slot and verdict."""
        slots = max(1, profile.parallel)
        ctx = (profile.ctx_size or planner.ctx_bounds(self.intent, self.info)[1]) // slots
        return Candidate(profile, ctx, trial_verdict(profile, self.host), how)

    def from_plan(self, plan: Plan) -> Candidate:
        """The first candidate: the planner's own profile."""
        return self.candidate(plan.profile, HOW_PLAN)

    def with_ctx(self, base: Candidate, ctx: int, how: str) -> Candidate:
        """`base` with another context per slot."""
        slots = max(1, base.profile.parallel)
        return self.candidate(base.profile.merged(ctx_size=ctx * slots), how)

    def down(self, c: Candidate) -> Candidate | None:
        """The next, more modest candidate after `c` failed to load.

        In order: the next lower context tier (not below the intent's minimum); then q8_0 K/V
        if the intent allows quantised KV and `c` is still f16; then letting llama.cpp place
        the layers (``gpu_layers`` auto with ``fit`` on) when the model is still fully on
        the GPU and there is a second memory pool.

        Args:
            c: The candidate that failed.

        Returns:
            The next candidate, or None when nothing less demanding is left.
        """
        cands, _, floor = planner.ctx_bounds(self.intent, self.info)
        lower = [t for t in cands if floor <= t < c.ctx_per_slot]
        if lower:
            return self.with_ctx(c, lower[-1], HOW_CTX_DOWN)
        p = c.profile
        if self.intent.kv_quant != KV_NEVER and p.cache_type_k == F16:
            return self.candidate(p.merged(cache_type_k=Q8, cache_type_v=Q8, flash_attn="on"),
                                  HOW_KV_Q8)
        if self.host.gpus and not self.host.unified and p.gpu_layers == "all" \
                and p.n_cpu_moe == 0:
            mib = max(planner.headroom_bytes(g.total, self.pct) for g in self.host.gpus) // planner.MiB
            return self.candidate(p.merged(gpu_layers="auto", fit="on", fit_target=mib), HOW_SPILL)
        return None

    def up(self, c: Candidate, m: Measurement) -> Candidate | None:
        """The next larger context tier when the measurement shows room for it.

        Only for a candidate fully on the GPU and below the intent's target context. The
        measured slack is the free VRAM at the start, minus what the model took, minus the
        headroom; the step happens when that exceeds the extra KV cache of one more tier.

        Args:
            c: A candidate that loaded.
            m: Its measurement.

        Returns:
            The next tier's candidate, or None.
        """
        p = c.profile
        if (not m.ok or c.verdict != VERDICT_GPU or not self.host.gpus or self.host.unified
                or p.gpu_layers != "all" or p.n_cpu_moe):
            return None
        cands, target, _ = planner.ctx_bounds(self.intent, self.info)
        higher = [t for t in cands if c.ctx_per_slot < t <= target]
        if not higher:
            return None
        nxt = higher[0]
        slots = max(1, p.parallel)
        extra = planner.kv_per_token(self.info, p.cache_type_k, p.cache_type_v) * (
            nxt - c.ctx_per_slot) * slots
        headroom = sum(planner.headroom_bytes(g.total, self.pct) for g in self.host.gpus)
        slack = self.host.vram_free - m.vram_used - headroom
        return self.with_ctx(c, nxt, HOW_CTX_UP) if slack > extra else None

    def threads(self, c: Candidate) -> Candidate | None:
        """`c` with ``threads`` set to every logical core, for a GPU+CPU candidate.

        Returns:
            The variant, or None when `c` is not GPU+CPU or already uses that many threads.
        """
        n = self.host.cpu_threads
        if c.verdict != VERDICT_SPILL or n <= 0 or c.profile.threads == n:
            return None
        return self.candidate(c.profile.merged(threads=n), HOW_THREADS)

    def score(self, c: Candidate, m: Measurement) -> tuple[float, float]:
        """Ranking key of a loaded candidate (higher is better).

        Normally the context per slot (capped at the intent's target), then the speed; the
        ``speed`` intent ranks speed first. Speed is generation tokens/s, or prompt tokens/s
        for embedding models, which generate nothing.
        """
        target = planner.ctx_bounds(self.intent, self.info)[1]
        ctx, tps = float(min(c.ctx_per_slot, target)), m.gen_tps or m.prompt_tps
        return (tps, ctx) if self.intent.key == SPEED_INTENT else (ctx, tps)


def choose_best(space: SearchSpace, history: History) -> tuple[Candidate, Measurement] | None:
    """The winning trial among those that loaded.

    Trials reaching the intent's minimum context are preferred; when none does, the best of
    the others is returned (the caller reports that it is below the minimum). Ties go to the
    earlier trial.

    Args:
        space: Search space (for the intent's bounds and scoring).
        history: Trials so far as (candidate, measurement).

    Returns:
        The winner, or None when nothing loaded.
    """
    loaded = [(c, m) for c, m in history if m.ok]
    floor = planner.ctx_bounds(space.intent, space.info)[2]
    usable = [(c, m) for c, m in loaded if c.ctx_per_slot >= floor] or loaded
    return max(usable, key=lambda cm: space.score(*cm), default=None)


def next_candidate(space: SearchSpace, history: History) -> Candidate | None:
    """Decide the next trial from the ones so far; never a profile already tried.

    - The last trial failed with no success yet: step down, but only for a lack of memory
      or time (other failures, such as an unknown architecture, will not improve).
    - The last trial loaded: step up one context tier if the measurement shows room.
    - Otherwise (a step up failed, or no room): for the best GPU+CPU candidate, one trial with
      all logical cores as threads. A threads trial is always the last.

    Args:
        space: Search space.
        history: Trials so far (not empty).

    Returns:
        The next candidate, or None when the search is finished.
    """
    seen = {profile_key(c.profile) for c, _ in history}
    last, m = history[-1]
    if last.how == HOW_THREADS:
        return None
    nxt: Candidate | None
    if not m.ok and not any(mm.ok for _, mm in history):
        nxt = space.down(last) if m.resource_failure else None
    else:
        nxt = space.up(last, m) if m.ok else None
        if nxt is None or profile_key(nxt.profile) in seen:
            best = choose_best(space, history)
            nxt = space.threads(best[0]) if best else None
    return nxt if nxt and profile_key(nxt.profile) not in seen else None


# --------------------------------------------------------------------------- tune

def _backend(host: Host) -> str:
    """Backend name of the host's first device, or "cpu"."""
    return host.gpus[0].backend if host.gpus else "cpu"


def _mtime(path: Path) -> float:
    """Modification time of the model file, 0.0 when it cannot be read."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _verdict_message(space: SearchSpace, best: Candidate, m: Measurement, n: int) -> str:
    """Summary of a successful tune."""
    floor = planner.ctx_bounds(space.intent, space.info)[2]
    below = f"; below the {space.intent.key} minimum of {floor} ctx" if best.ctx_per_slot < floor else ""
    speed = f"{m.gen_tps:.1f} tok/s" if m.gen_tps else f"{m.prompt_tps:.1f} prompt tok/s"
    return (f"{best.ctx_per_slot} ctx per slot, {best.profile.cache_type_k} KV, {best.verdict}, "
            f"{speed}, {m.vram_used / GB:.1f} GB VRAM after {n} trial{'s' * (n != 1)}{below}")


def tune(intent: Intent, info: ModelInfo, host: Host, base: Profile, runner: Runner, *,
         headroom_pct: float = 10, calibration: Calibration | None = None, max_trials: int = 4,
         progress: Callable[[TrialEvent], None] | None = None,
         cancel: Callable[[], bool] | None = None) -> TuneResult:
    """Find the best measured profile for an intent by launching real candidates.

    Starts from `planner.plan` and follows `next_candidate` until it has nothing left to try,
    `max_trials` launches happened, or `cancel` returns True (checked before each trial and
    handed to the runner, which stops a trial in flight; that trial is reported through
    `progress` but is not recorded and does not count as a failure to step down from). A
    Ctrl-C (``KeyboardInterrupt`` out of the runner) ends the search the same way: the
    interrupted trial is dropped and the trials before it are kept. A plan that cannot fit this
    machine at all is not launched.

    Args:
        intent: What the model is for.
        info: Model metadata (``info.path`` is launched).
        host: Host snapshot; free memory should be current.
        base: Profile planning starts from (usually the default profile).
        runner: Does the launching and measuring.
        headroom_pct: Percentage of each device kept free.
        calibration: Measured data correcting the planner's estimates, or None.
        max_trials: Most launches to make.
        progress: Called with a `TrialEvent` before and after every trial.
        cancel: Polled between trials and by the runner during one; True stops the search.

    Returns:
        The result; ``trials`` holds every launch, ``best`` the winner (None if nothing loaded).
    """
    space = SearchSpace(intent, info, host, headroom_pct)
    plan = planner.plan(intent, info, host, base, headroom_pct, calibration)
    if plan.verdict == VERDICT_NO:
        why = plan.warnings[0] if plan.warnings else "does not fit this machine"
        return TuneResult(intent.key, None, None, [], False, f"not tried: {why}")
    mtime = _mtime(info.path)
    history: History = []
    trials: list[Trial] = []
    cand: Candidate | None = space.from_plan(plan)
    cancelled = False
    while cand is not None and len(history) < max_trials:
        if cancel is not None and cancel():
            cancelled = True
            break
        index = len(history) + 1
        if progress:
            progress(TrialEvent(index, max_trials, "start", cand.summary))
        try:
            m = runner.measure(info.path, cand.profile, cancel)
        except KeyboardInterrupt:      # Ctrl-C: the runner has stopped its server; keep what we have
            cancelled = True
            break
        if m.cancelled:                # stopped in flight: not a load failure, not a trial
            cancelled = True
            if progress:
                progress(TrialEvent(index, max_trials, "done", cand.summary, m))
            break
        history.append((cand, m))
        trials.append(Trial(
            model=str(info.path), mtime=mtime, intent=intent.key, profile=cand.profile.to_dict(),
            estimated_total=planner.estimate_budget(info, cand.profile, host).raw_total(),
            verdict=cand.verdict, backend=_backend(host), measurement=m))
        if progress:
            progress(TrialEvent(index, max_trials, "done", cand.summary, m))
        cand = next_candidate(space, history)
    best = choose_best(space, history)
    if best is None:
        last = history[-1][1].error if history else ""
        why = ("cancelled before any trial ran" if cancelled and not history else
               f"nothing loaded after {len(history)} trial{'s' * (len(history) != 1)}: {last}")
        return TuneResult(intent.key, None, None, trials, False, why, cancelled)
    msg = _verdict_message(space, best[0], best[1], len(history))
    return TuneResult(intent.key, best[0], best[1], trials, True,
                      f"cancelled; best so far: {msg}" if cancelled else msg, cancelled)


def save_result(cfg: Config, model: str | Path, result: TuneResult, calibration: Calibration,
                *, activate: bool = False) -> None:
    """Persist a tune: every trial to tuning.json, and the winner as the model's preset.

    The trials are recorded even when the tune failed or was cancelled. Only a successful
    result touches the config: the winner is stored as ``Preset(intent, profile, "tuned",
    measurement)`` under the intent's name, replacing any preset of that name.

    Args:
        cfg: Config to update and save.
        model: Model file the tune was for.
        result: What `tune` returned.
        calibration: Store receiving the trials.
        activate: Also make the preset the model's active one (it becomes active anyway when
            the model has no active preset).

    Raises:
        OSError: When tuning.json or the config cannot be written.
    """
    for trial in result.trials:
        calibration.record(trial)
    calibration.save()
    if not result.ok or result.best is None or result.measurement is None:
        return
    cfg.set_preset(str(model), Preset(result.intent, result.best.profile, TUNED,
                                      result.measurement), activate=activate)
    cfg.save()


def preflight(plan: Plan, host: Host) -> str | None:
    """Refuse a tune when free memory is clearly below what the plan needs.

    "Clearly" means below ``PREFLIGHT_MARGIN`` (90%) of the plan's budget, since the
    estimate is not exact. Fully-GPU plans are checked against free VRAM, GPU+CPU plans
    against free VRAM plus free RAM, CPU plans (and unified memory) against free RAM.

    Args:
        plan: The plan about to be tuned.
        host: A fresh host snapshot (the caller refreshes free memory).

    Returns:
        A human readable refusal, or None when the tune may go ahead.
    """
    need = plan.budget.total()
    if plan.verdict == VERDICT_NO:
        return f"the plan needs {need / GB:.1f} GB, more than this machine has"
    if plan.verdict == VERDICT_CPU or not host.gpus or host.unified:
        pool, where = host.ram_free, "free RAM"
    elif plan.verdict == VERDICT_GPU:
        pool, where = host.vram_free, "free VRAM"
    else:
        pool, where = host.vram_free + host.ram_free, "free VRAM + RAM"
    if pool < need * PREFLIGHT_MARGIN:
        return (f"needs about {need / GB:.1f} GB but only {pool / GB:.1f} GB {where} is "
                "available; stop or unload other models first")
    return None


# --------------------------------------------------------------------------- the real runner

def _now() -> str:
    """Current UTC time as an ISO timestamp (seconds)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def health_timeout(size: int) -> float:
    """Seconds to wait for ``/health`` while a model of `size` bytes loads.

    ``HEALTH_BASE`` (60 s) plus ``HEALTH_PER_GB`` (20 s) per GB of model file, capped at
    ``HEALTH_MAX`` (30 min): loading a cold file from a slow disk at ~50 MB/s still fits.

    Args:
        size: Model file size in bytes.

    Returns:
        The timeout in seconds.
    """
    return min(HEALTH_MAX, HEALTH_BASE + HEALTH_PER_GB * size / GB)


def parse_build(text: str) -> str:
    """The version from ``llama-server --version`` output (``version: 0.5.0-dev (build 1, ...)``).

    Args:
        text: Combined stdout and stderr.

    Returns:
        The text after ``version:``, or "" when there is no such line.
    """
    for line in text.splitlines():
        if line.strip().lower().startswith("version:"):
            return line.split(":", 1)[1].strip()
    return ""


_PREFIX_RE = re.compile(r"^\d+(?:\.\d+)*\s+[A-Z]\s+")      # "0.00.175.656 E " log prefix
_ERROR_LEVEL_RE = re.compile(r"^\S+\s+E\s")
_ROOT_CAUSE_RE = re.compile(r"error loading model|unknown model architecture|^error:", re.I)


def classify_failure(log_tail: str) -> str:
    """Name why llama-server failed to come up, from the end of its log.

    Args:
        log_tail: The last part of the server's output.

    Returns:
        ``"out of memory: <line>"``, ``"unknown argument: <line>"`` or ``"load failed: <line>"``.
        The line is the first one matching the kind (the root cause comes before its
        follow-up errors); for "load failed" the first line naming a loading error, else the
        first error-level line, else the last line. The timestamp/level prefix is removed.
    """
    lines = [ln.strip() for ln in log_tail.splitlines() if ln.strip()]

    def show(line: str) -> str:
        return _PREFIX_RE.sub("", line)[:200]

    for kind, pattern in ((ERR_OOM, _OOM_RE), (ERR_ARGS, _ARGS_RE)):
        hits = [ln for ln in lines if pattern.search(ln)]
        if hits:
            return f"{kind}: {show(hits[0])}"
    errors = [ln for ln in lines if _ERROR_LEVEL_RE.match(ln) or re.search(r"\berror\b", ln, re.I)]
    cause = [ln for ln in errors if _ROOT_CAUSE_RE.search(_PREFIX_RE.sub("", ln))]
    pick = (cause or errors or lines[-1:] or ["no log output"])[0]
    return f"{ERR_LOAD}: {show(pick)}"


def live_devices(bin_dir: str) -> list[Device]:
    """Devices with current free memory: nvidia-smi when it works, else ``--list-devices``."""
    return hardware.nvidia_smi_devices() or hardware.list_devices(bin_dir)


class ServerRunner:
    """Measure a profile by launching a real llama-server (``Runner`` implementation).

    One `measure` call: pick a free loopback port, snapshot device and system memory, start
    the server in its own process group with the profile (host forced to 127.0.0.1, port
    chosen, no API key, no idle sleep), wait for ``/health``, snapshot memory again, send the
    fixed benchmark request, and always stop the process group and wait for it to be gone.

    Args:
        cfg: Config (for ``bin_dir``).
        popen: Process launcher with the signature of `subprocess.Popen`.
        http: Factory for an HTTP client, called with ``base_url`` and ``timeout`` keywords;
            the client needs ``request(method, path, **kw)`` and context-manager support.
        devices: Returns the devices with their *current* free memory.
        run: ``subprocess.run`` compatible callable, used for ``--version``.
        ram_free: Returns the available system RAM in bytes.
        clock: Monotonic clock in seconds.
        sleep: Sleep function.
    """

    def __init__(self, cfg: Config, *, popen: Callable[..., Any] = subprocess.Popen,
                 http: Callable[..., Any] = httpx.Client,
                 devices: Callable[[], list[Device]] | None = None,
                 run: Callable[..., Any] = subprocess.run,
                 ram_free: Callable[[], int] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.cfg = cfg
        self._popen = popen
        self._http = http
        self._devices = devices or (lambda: live_devices(cfg.bin_dir))
        self._run = run
        self._ram_free = ram_free or (lambda: hardware.read_memory()[1])
        self._clock = clock
        self._sleep = sleep
        self._build: str | None = None

    # -- helpers ---------------------------------------------------------------------

    def build(self) -> str:
        """The llama-server version string, asked once and cached.

        Returns:
            E.g. ``0.5.0-dev (build 1, commit 5fc4f3c)``; "" when it cannot be determined.
        """
        if self._build is None:
            argv = [str(Path(self.cfg.bin_dir) / "llama-server"), "--version"]
            try:
                proc = self._run(argv, capture_output=True, text=True, timeout=VERSION_TIMEOUT)
            except (OSError, subprocess.SubprocessError):
                return ""
            text = "\n".join(str(getattr(proc, a, "") or "") for a in ("stdout", "stderr"))
            self._build = parse_build(text)
        return self._build

    def _snapshot(self) -> tuple[int, int]:
        """(free device memory summed over devices, available RAM), in bytes."""
        return sum(d.free for d in self._devices()), self._ram_free()

    @staticmethod
    def _new_log(model: Path) -> Path:
        """A fresh log file in the llamactl log directory."""
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f"tune-{model.stem}-", suffix=".log", dir=LOG_DIR)
        os.close(fd)
        return Path(name)

    @staticmethod
    def _tail(log: Path) -> str:
        """The last part of a log file ("" when unreadable)."""
        try:
            with log.open("rb") as fh:
                fh.seek(0, os.SEEK_END)
                fh.seek(max(0, fh.tell() - LOG_TAIL_BYTES))
                return fh.read().decode(errors="replace")
        except OSError:
            return ""

    def _stop(self, pid: int) -> None:
        """Stop the process group (SIGTERM, then SIGKILL after the grace period) and wait.

        Does nothing when the process is already gone, so a recycled pid is never signalled.
        """
        if not pid_alive(pid):
            return
        terminate_group(pid, timeout=STOP_GRACE, sleep=self._sleep)
        waited = 0.0
        while pid_alive(pid) and waited < GONE_WAIT:
            self._sleep(0.1)
            waited += 0.1

    # -- measuring -------------------------------------------------------------------

    def measure(self, model: Path, profile: Profile,
                cancel: Callable[[], bool] | None = None) -> Measurement:
        """Launch, benchmark and stop; see the class description.

        Args:
            model: GGUF file.
            profile: Settings to launch with.
            cancel: Checked before the launch, on every health poll and before the benchmark
                request; when it returns True the server is stopped and a failed measurement
                whose error starts with ``cancelled`` is returned (the single benchmark
                request, once sent, is waited for).

        Returns:
            The measurement. Every failure to start, load or answer is ``ok=False`` with an
            ``error`` (``out of memory``, ``timeout``, ``unknown argument``, ``load failed``
            or ``benchmark failed``, then a detail and the log path). A bug in the code
            around the call raises, after the process has been stopped.
        """
        build = self.build()
        if cancel is not None and cancel():
            return Measurement(error=f"{ERR_CANCELLED}: before the launch", at=_now(), build=build)
        try:
            port = next_free_port(set(), LOOPBACK)
        except RuntimeError as e:
            return Measurement(error=f"{ERR_LOAD}: {e}", at=_now(), build=build)
        prof = profile.merged(host=LOOPBACK, port=port, api_key="", sleep_idle=0)
        argv = build_command(self.cfg, str(model), prof, port)
        log = self._new_log(model)
        vram_before, ram_before = self._snapshot()
        start = self._clock()
        try:
            with log.open("ab") as fh:
                proc = self._popen(argv, stdout=fh, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError as e:
            return Measurement(error=f"{ERR_LOAD}: cannot start {argv[0]}: {e}", at=_now(),
                               build=build)
        try:
            m = self._measure_running(proc.pid, prof, port, model, log, start, vram_before,
                                      ram_before, build, cancel)
        finally:
            self._stop(proc.pid)
        if m.ok or m.cancelled:
            log.unlink(missing_ok=True)
        return m

    def _measure_running(self, pid: int, prof: Profile, port: int, model: Path, log: Path,
                         start: float, vram_before: int, ram_before: int, build: str,
                         cancel: Callable[[], bool] | None = None) -> Measurement:
        """Everything after the process started: wait, snapshot, benchmark.

        `cancel` is checked on every health poll and once more before the benchmark request.
        """
        def failed(error: str, load_s: float = 0.0) -> Measurement:
            return Measurement(error=f"{error} (log: {log})", load_s=load_s, at=_now(), build=build)

        try:
            size = model.stat().st_size
        except OSError:
            size = 0
        with self._http(base_url=f"http://{LOOPBACK}:{port}", timeout=REQUEST_TIMEOUT) as client:
            while True:
                if cancel is not None and cancel():
                    return failed(f"{ERR_CANCELLED}: stopped while the model was loading",
                                  self._clock() - start)
                if not pid_alive(pid):
                    return failed(classify_failure(self._tail(log)), self._clock() - start)
                try:
                    if client.request("GET", "/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if self._clock() - start >= health_timeout(size):
                    return failed(f"{ERR_TIMEOUT}: /health not ready after "
                                  f"{self._clock() - start:.0f} s", self._clock() - start)
                self._sleep(POLL_INTERVAL)
            load_s = self._clock() - start
            if cancel is not None and cancel():
                return failed(f"{ERR_CANCELLED}: stopped before the benchmark", load_s)
            vram_load, ram_load = self._snapshot()
            try:
                prompt_tps, gen_tps = (self._embed(client) if prof.embedding
                                       else self._complete(client))
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
                return failed(f"{ERR_BENCH}: {e}", load_s)
            vram_end, ram_end = self._snapshot()
        return Measurement(
            ok=True, load_s=load_s, vram_used=max(0, vram_before - min(vram_load, vram_end)),
            ram_used=max(0, ram_before - min(ram_load, ram_end)), prompt_tps=prompt_tps,
            gen_tps=gen_tps, at=_now(), build=build)

    @staticmethod
    def _complete(client: Any) -> tuple[float, float]:
        """Benchmark a generative model with ``POST /completion``.

        Returns:
            (prompt tokens/s, generation tokens/s) from the response's ``timings``.

        Raises:
            ValueError: On an HTTP error status, a body without ``timings``, or no speed.
        """
        r = client.request("POST", "/completion", timeout=BENCH_TIMEOUT, json={
            "prompt": BENCH_PROMPT, "n_predict": 128, "cache_prompt": False, "seed": 42,
            "temperature": 0, "stream": False})
        if r.status_code != 200:
            raise ValueError(f"HTTP {r.status_code}: {str(r.text).strip()[:120]}")
        timings = r.json()["timings"]
        gen = float(timings["predicted_per_second"])
        if gen <= 0:
            raise ValueError("the server generated no tokens")
        return float(timings.get("prompt_per_second", 0.0)), gen

    def _embed(self, client: Any) -> tuple[float, float]:
        """Benchmark an embedding model with ``POST /v1/embeddings``.

        Returns:
            (prompt tokens/s, 0.0). The speed is the response's ``usage.prompt_tokens`` over
            the request's wall time; 0.0 when the response carries no usage.

        Raises:
            ValueError: On an HTTP error status or a body without ``data``.
        """
        t0 = self._clock()
        r = client.request("POST", "/v1/embeddings", timeout=BENCH_TIMEOUT,
                           json={"input": EMBED_TEXT, "model": "tune"})
        elapsed = self._clock() - t0
        if r.status_code != 200:
            raise ValueError(f"HTTP {r.status_code}: {str(r.text).strip()[:120]}")
        body = r.json()
        if not body["data"]:
            raise ValueError("no embedding returned")
        usage = body.get("usage") if isinstance(body, dict) else None
        tokens = float(usage.get("prompt_tokens", 0)) if isinstance(usage, dict) else 0.0
        return (tokens / elapsed if tokens > 0 and elapsed > 0 else 0.0), 0.0
