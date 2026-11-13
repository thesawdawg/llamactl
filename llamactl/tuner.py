"""Measured tuning: launch candidate plans for real, keep the best, and learn from the numbers.

The pure part (`SearchSpace`, `next_candidate`, `choose_best`, `tune`) decides which profile to
try next from what the previous trials measured: after the plan, an all-GPU trial when
llama.cpp had to split the model, then a context fill that probes toward the largest context
fitting fully on the GPU (predicted from the measured bytes per token, guarded by a headroom
check plus a spill check on the generation speed, so a driver silently falling back to
system memory does not count as fitting), and a threads variant for a GPU+CPU best as the
last resort. For a MoE model on a discrete GPU it also walks the expert split: failures move
more experts to RAM (`--n-cpu-moe`, auto-fit off), and once a split fits it probes smaller
counts back toward the GPU since fewer experts in RAM is faster. The only parts that touch
processes or HTTP are behind the `Runner` protocol;
`ServerRunner` is the real one, with every subprocess, HTTP, clock and device-memory call
injectable so tests never start llama.cpp.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, replace
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
from .goals import TuningGoals
from .hardware import Device, Host
from .intents import KV_NEVER, Intent
from .measure import (ERR_ARGS, ERR_BENCH, ERR_CANCELLED, ERR_LOAD, ERR_OOM, ERR_TIMEOUT,
                      Measurement, WorkloadMeasurement)
from .planner import F16, GB, Q8, VERDICT_CPU, VERDICT_GPU, VERDICT_NO, VERDICT_SPILL, Plan
from .sessions import next_free_port, pid_alive
from .support import SupportChecker, unknown_arch

LOOPBACK = "127.0.0.1"
SPEED_INTENT = "speed"

# How a candidate was derived (shown in progress events).
HOW_PLAN, HOW_CTX_DOWN, HOW_KV_Q8, HOW_SPILL = "plan", "ctx down", "q8_0 KV", "spill"
HOW_ALL_GPU, HOW_CTX_UP, HOW_THREADS = "all GPU", "ctx up", "threads"
HOW_MORE_RAM, HOW_MOE_BACK = "more experts to RAM", "experts back to GPU"
HOW_BISECT = "ctx bisect"

CTX_STEP = 4096             # context probes are multiples of this
SPEED_TOLERANCE = 0.05      # extra context beyond the target may cost at most ~5% speed
SPILL_RATIO = 1 / 3         # below this share of the family best a trial probably spilled
PROMPT_TOLERANCE = 0.30     # a trial this far below the tier's best prompt speed is dropped
MOE_EXTRA_TRIALS = 3        # MoE models get this many launches beyond --trials for the split

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

# Constrained-tune workload model: a turn is its new input (delta) plus generation over
# a prefix already in the KV cache — how chat and coding harnesses actually behave.
TURN_TOKENS = 2048        # turn delta over a cached prefix; at/below it a turn is cold
WARMUP_TOKENS = 2048      # one discarded request so kernel init and clock ramp stay out
WARMUP_PREDICT = 8
PRIME_PREDICT = 1         # prefix priming/eviction generates one token to settle the slot
NEAR_FLOOR = 1.15         # within 15% of an enabled floor: retest, keep the median of 3
SETTLE_MAX = 4.0          # seconds to wait for device memory release after a stop
SETTLE_POLL = 0.25
CTX_BISECT_TOL = 0.05     # stop bisecting once the open interval is within ~5% of hi

BENCH_PROMPT_TOKENS = 2048  # benchmark prompt length cap; shorter contexts get half of theirs
_BENCH_PARA = (
    "Field note {n}: the survey party reached marker {n} shortly after dawn, found frost on "
    "the tripod and fresh mud on the level, argued briefly about the missing sandwiches, "
    "checked the theodolite twice for safety, logged the bearing, the time and the weather, "
    "and agreed the rest of the ridge could wait until the fog lifted off the valley.")
EMBED_TEXT = " ".join(["The quick brown fox jumps over the lazy dog near the quiet riverbank."] * 8)


def bench_prompt(ctx_per_slot: int) -> str:
    """A deterministic benchmark prompt for a generative trial.

    Args:
        ctx_per_slot: Context each slot gets; the prompt aims for about
            ``min(BENCH_PROMPT_TOKENS, ctx_per_slot // 2)`` tokens.

    Returns:
        Varied numbered paragraphs of ~80 tokens each, joined by blank lines (tokens are
        approximated as a quarter of the character count).
    """
    want = max(1, min(BENCH_PROMPT_TOKENS, ctx_per_slot // 2))
    parts: list[str] = []
    while sum(len(p) for p in parts) // 4 < want:
        parts.append(_BENCH_PARA.format(n=len(parts) + 1))
    return "\n\n".join(parts)


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
        total: Most trials that can still happen (the trial budget); the search may stop
            earlier.
        phase: "start" before the launch, "done" after it.
        summary: Profile summary (`Candidate.summary`).
        measurement: The result on "done", else None.
        note: On "done", why the trial did not count as fitting (`SearchSpace.reject_reason`);
            "" when it did, and always "" for a trial that did not load.
        n_cpu_moe: The candidate's ``--n-cpu-moe`` (0 when not a MoE split).
    """

    index: int
    total: int
    phase: str
    summary: str
    measurement: Measurement | None = None
    note: str = ""
    n_cpu_moe: int = 0


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
        notes: The final `SearchSpace.reject_reason` of every trial, in order, computed
            against the full history (a later faster trial can turn an earlier one into a
            spill); "" for trials that did not load.
        why: For a winner with experts in RAM, why its ``n_cpu_moe`` won over the lower
            counts that were tried; "" otherwise.
    """

    intent: str
    best: Candidate | None
    measurement: Measurement | None
    trials: list[Trial] = field(default_factory=list)
    ok: bool = False
    message: str = ""
    cancelled: bool = False
    notes: tuple[str, ...] = ()
    why: str = ""


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

    `down` walks the failure ladder (on a MoE model first more experts to RAM, then smaller
    context, q8_0 KV, llama.cpp placement); `all_gpu` re-tries a split model whole on the
    GPU; `fill` probes the largest context that still fits fully on the GPU using the
    measured bytes-per-token slope; `moe_fill` probes a lower ``n_cpu_moe`` from the best
    fitting expert split; `fits` is the headroom and spill guard a loaded trial must pass
    to count (`reject_reason` says why it failed); `threads` is the last-resort variant for
    a GPU+CPU best.

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

    def _moe(self) -> bool:
        """True when this search can place MoE experts: a MoE model with a known tensor
        table on a non-unified GPU host."""
        return (self.info.expert_count > 0 and bool(self.info.expert_bytes)
                and bool(self.host.gpus) and not self.host.unified)

    def down(self, c: Candidate, m: Measurement | None = None) -> Candidate | None:
        """The next, more modest candidate after `c` failed to load.

        On a MoE model a memory failure first moves more experts to RAM: `n_cpu_moe` rises
        by enough layers to cover the measured overflow (`m` loaded but rejected, +1 layer
        minimum) or the estimated one (OOM, timeout, unknown). The estimate usually still
        claims the failed trial fits, so after a failure without a usable measurement the
        step is at least ``max(1, n_layer // 8)`` layers; `moe_fill` bisects back down
        between the failure and the first fit. Only at ``n_cpu_moe == n_layer`` does the
        search fall through to the usual ladder: the next lower context tier (not below the
        intent's minimum); then q8_0 K/V if the intent allows quantised KV and `c` is still
        f16; then letting llama.cpp place the layers (``gpu_layers`` auto with ``fit`` on)
        when the model is still fully on the GPU and there is a second memory pool.

        Args:
            c: The candidate that failed.
            m: Its measurement when there is one (used for the MoE overflow).

        Returns:
            The next candidate, or None when nothing less demanding is left.
        """
        if self._moe() and c.profile.gpu_layers == "all" \
                and c.profile.n_cpu_moe < self.info.n_layer:
            if m is not None and m.ok:
                overflow = self._headroom() - (self.host.vram_free - m.vram_used)
                step = 1
            else:
                est = planner.estimate_budget(self.info, c.profile, self.host).total()
                overflow = (est - planner.expert_ram_bytes(self.info, c.profile)
                            - planner.gpu_limit(self.host, self.pct))
                step = max(1, self.info.n_layer // 8)
            target = planner.expert_ram_bytes(self.info, c.profile) + max(overflow, 1)
            split = planner.moe_split(self.info, target)
            n = max(c.profile.n_cpu_moe + step, split[0] if split else self.info.n_layer)
            n = min(n, self.info.n_layer)
            return self.candidate(c.profile.merged(n_cpu_moe=n, fit="off"), HOW_MORE_RAM)
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

    def _headroom(self) -> int:
        """Headroom the planner keeps free, summed over the host's GPUs."""
        return sum(planner.headroom_bytes(g.total, self.pct) for g in self.host.gpus)

    def all_gpu(self, c: Candidate) -> Candidate | None:
        """`c` forced fully onto the GPU, for a loaded candidate llama.cpp had to split.

        Applies only when the placement was llama.cpp's (``gpu_layers auto`` with ``fit``
        on), no experts sit in RAM, the host has a real GPU, and the model file plus the
        headroom fits the free VRAM.

        Args:
            c: A candidate that loaded.

        Returns:
            The same profile with ``gpu_layers="all", fit="off"`` at the same context,
            or None when the step does not apply.
        """
        p = c.profile
        if (p.gpu_layers != "auto" or p.fit != "on" or p.n_cpu_moe
                or not self.host.gpus or self.host.unified):
            return None
        if self.info.size + self._headroom() >= self.host.vram_free:
            return None
        return self.candidate(p.merged(gpu_layers="all", fit="off", fit_target=0),
                              HOW_ALL_GPU)

    def _guarded(self, c: Candidate) -> bool:
        """True when the headroom and spill guards apply to `c`: every layer meant for the
        GPU (``gpu_layers == "all"``, experts possibly in RAM) on a non-unified GPU host."""
        return (c.profile.gpu_layers == "all" and bool(self.host.gpus)
                and not self.host.unified)

    def reject_reason(self, c: Candidate, m: Measurement, history: History = ()) -> str:
        """Why a trial does not count as fitting, or "".

        A loaded all-GPU trial on a non-unified GPU host is rejected when it did not leave
        the headroom free — on Windows/WSL the NVIDIA driver can satisfy an oversized CUDA
        allocation with system memory instead of failing, so a load that ate into the
        headroom does not prove the model really is on the GPU — or when it generated far
        slower than the best loaded trial of the same profile family (context aside) that
        kept at least as many expert layers in RAM, which suggests the same silent spill
        (more experts on the GPU should never be that much slower).

        Args:
            c: The tried candidate.
            m: Its measurement.
            history: Trials so far as (candidate, measurement); the family the spill check
                compares against comes from it.

        Returns:
            "" when the trial counts as fitting, else a short reason (60 chars at most,
            shown in a table cell); "not loaded" for a failed trial.
        """
        if not m.ok:
            return "not loaded"
        if not self._guarded(c):
            return ""
        free = self.host.vram_free - m.vram_used
        need = self._headroom()
        if free < need:
            return f"left less than the headroom free: {free / GB:.1f} of {need / GB:.1f} GB"
        if m.gen_tps > 0:
            key = profile_key(c.profile.merged(ctx_size=0, n_cpu_moe=0))
            family = [mm for cc, mm in history if mm.ok
                      and cc.profile.n_cpu_moe >= c.profile.n_cpu_moe
                      and profile_key(cc.profile.merged(ctx_size=0, n_cpu_moe=0)) == key]
            best = max((mm.gen_tps for mm in family), default=0.0)
            if best > 0 and m.gen_tps < SPILL_RATIO * best:
                return f"spilled to system RAM? {m.gen_tps:.1f} vs {best:.1f} tok/s"
        return ""

    def fits(self, c: Candidate, m: Measurement, history: History = ()) -> bool:
        """True when a trial's measurement counts as "fits on this machine".

        Args:
            c: The tried candidate.
            m: Its measurement.
            history: Trials so far; the spill check compares against them when given.

        Returns:
            Whether it loaded without a reject reason (see `reject_reason`).
        """
        return m.ok and not self.reject_reason(c, m, history)

    def fill(self, history: History) -> Candidate | None:
        """The next context probe toward the largest that still fits fully on the GPU.

        Only trials of the best fitting all-GPU profile family count (same profile, context
        aside). `lo` is the largest context that fitted, `hi` the smallest that provably did
        not (a memory or timeout failure, or a load that failed the headroom guard). The
        prediction is `lo` plus the slack left at `lo` divided by the bytes per token —
        measured between the two largest fitting points when possible, else the planner's
        estimate — rounded down to a `CTX_STEP` multiple, capped at the intent's `ctx_max`
        ceiling and at `hi - CTX_STEP`. When the prediction cannot pass `lo`, the midpoint
        of `lo` and `hi` is probed instead. Embedding models are never filled.

        Args:
            history: Trials so far as (candidate, measurement).

        Returns:
            The next candidate above `lo`, or None.
        """
        fitting = [(c, m) for c, m in history
                   if c.verdict == VERDICT_GPU and self.fits(c, m, history)]
        if not fitting:
            return None
        best = max(fitting, key=lambda cm: self.score(*cm))[0]
        if best.profile.embedding:
            return None
        key = profile_key(best.profile.merged(ctx_size=0))
        same = [(c, m) for c, m in history
                if profile_key(c.profile.merged(ctx_size=0)) == key]
        family = [(c, m) for c, m in same if self.fits(c, m, history)]
        lo = max(c.ctx_per_slot for c, _ in family)
        his = [c.ctx_per_slot for c, m in same
               if (m.ok and not self.fits(c, m, history))
               or (not m.ok and m.resource_failure)]
        hi = min(his, default=None)
        ceiling = planner.ctx_ceiling(self.intent, self.info)
        if lo >= ceiling:
            return None
        pts = sorted(c.ctx_per_slot for c, _ in family)
        slope = 0.0
        if len(pts) >= 2:
            v = {c.ctx_per_slot: m.vram_used for c, m in family}
            slope = (v[pts[-1]] - v[pts[-2]]) / (pts[-1] - pts[-2])
        if slope <= 0:
            slope = planner.kv_per_token(self.info, best.profile.cache_type_k,
                                         best.profile.cache_type_v) * max(1, best.profile.parallel)
        m_lo = next(m for c, m in family if c.ctx_per_slot == lo)
        slack = self.host.vram_free - m_lo.vram_used - self._headroom()
        nxt = (lo + int(slack / slope)) // CTX_STEP * CTX_STEP
        nxt = min(nxt, ceiling)
        if hi is not None:
            nxt = min(nxt, hi - CTX_STEP)
            if nxt <= lo:
                nxt = (lo + hi) // 2 // CTX_STEP * CTX_STEP
        seen = {profile_key(c.profile) for c, _ in history}
        if nxt <= lo:
            return None
        cand = self.with_ctx(best, nxt, HOW_CTX_UP)
        return cand if profile_key(cand.profile) not in seen else None

    def moe_fill(self, history: History) -> Candidate | None:
        """The lowest expert count that should still fit, probed below the best MoE split.

        From the best fitting trial with experts in RAM, the VRAM it left free beyond the
        headroom is slack that could hold more experts: lowering ``n_cpu_moe`` by k puts the
        expert bytes of the last k RAM-placed layers back on the GPU, and fewer layers in
        RAM is faster. `f` is the largest count below it that provably did not fit (failed
        to load, or loaded but rejected), so when both sides are known the probe goes
        halfway between them instead. Never a profile already tried.

        Args:
            history: Trials so far as (candidate, measurement).

        Returns:
            The next candidate, or None.
        """
        if not self._moe():
            return None
        fitting = [(c, m) for c, m in history
                   if c.profile.gpu_layers == "all" and c.profile.n_cpu_moe > 0
                   and self.fits(c, m, history)]
        if not fitting:
            return None
        best = max(fitting, key=lambda cm: (self.score(*cm), -cm[0].profile.n_cpu_moe))[0]
        key = profile_key(best.profile.merged(n_cpu_moe=0))
        family = [(c, m) for c, m in history
                  if profile_key(c.profile.merged(n_cpu_moe=0)) == key]
        fit_family = [(c, m) for c, m in family if self.fits(c, m, history)]
        c_b, m_b = min(fit_family, key=lambda cm: cm[0].profile.n_cpu_moe)
        n_b = c_b.profile.n_cpu_moe
        f = max((c.profile.n_cpu_moe for c, m in family
                 if c.profile.n_cpu_moe < n_b and not self.fits(c, m, history)), default=None)
        slack = self.host.vram_free - m_b.vram_used - self._headroom()
        k = n_b
        while k > 0 and self.info.expert_bytes[k - 1] <= slack:
            slack -= self.info.expert_bytes[k - 1]
            k -= 1
        if f is not None and k <= f:
            k = (f + n_b + 1) // 2 if n_b - f >= 2 else None
        if k is None or k == n_b:
            return None
        seen = {profile_key(c.profile) for c, _ in history}
        cand = self.candidate(best.profile.merged(n_cpu_moe=k, fit="off"), HOW_MOE_BACK)
        return cand if profile_key(cand.profile) not in seen else None

    def threads(self, c: Candidate) -> Candidate | None:
        """`c` with ``threads`` and ``threads_batch`` set to every logical core, for a
        GPU+CPU candidate (prompt speed with experts on the CPU depends on both).

        Returns:
            The variant, or None when `c` is not GPU+CPU or already uses that many threads.
        """
        n = self.host.cpu_threads
        if c.verdict != VERDICT_SPILL or n <= 0 \
                or (c.profile.threads == n and c.profile.threads_batch == n):
            return None
        return self.candidate(c.profile.merged(threads=n, threads_batch=n), HOW_THREADS)

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
    """The winning trial among those that loaded and fit.

    Trials reaching the intent's minimum context are preferred; when none does, the best of
    the others is returned (the caller reports that it is below the minimum). The ``speed``
    intent ranks tokens/s first. For the others the largest context capped at the target
    wins; within that tier a trial more than ``PROMPT_TOLERANCE`` below the tier's best
    prompt speed is dropped first (experts in RAM slow prompt processing far more than
    generation), then a trial more than ``SPEED_TOLERANCE`` slower than the tier's fastest
    — extra context beyond the target wins unless it costs more than ~5 % speed. Ties go
    to the higher speed, then the earlier trial.

    Args:
        space: Search space (for the intent's bounds and scoring).
        history: Trials so far as (candidate, measurement).

    Returns:
        The winner, or None when nothing loaded and fit.
    """
    floor = planner.ctx_bounds(space.intent, space.info)[2]
    fit = [(c, m) for c, m in history if space.fits(c, m, history)]
    usable = [(c, m) for c, m in fit if c.ctx_per_slot >= floor] or fit
    if not usable:
        return None
    if space.intent.key == SPEED_INTENT:
        return max(usable, key=lambda cm: space.score(*cm))
    target = planner.ctx_bounds(space.intent, space.info)[1]
    capped = max(min(c.ctx_per_slot, target) for c, _ in usable)
    group = [(i, c, m) for i, (c, m) in enumerate(usable)
             if min(c.ctx_per_slot, target) == capped]
    top_prompt = max(m.prompt_tps for _, _, m in group)
    group = [icm for icm in group
             if not icm[2].gen_tps
             or icm[2].prompt_tps >= (1 - PROMPT_TOLERANCE) * top_prompt]
    top_tps = max(m.gen_tps or m.prompt_tps for _, _, m in group)
    group = [icm for icm in group
             if (icm[2].gen_tps or icm[2].prompt_tps) >= (1 - SPEED_TOLERANCE) * top_tps]
    return max(group, key=lambda icm: (
        icm[1].ctx_per_slot, icm[2].gen_tps or icm[2].prompt_tps, -icm[0]))[1:]


def next_candidate(space: SearchSpace, history: History) -> Candidate | None:
    """Decide the next trial from the ones so far; never a profile already tried.

    - The last trial failed with no success yet: step down, but only for a lack of memory
      or time (other failures, such as an unknown architecture, will not improve).
    - Otherwise: the all-GPU variant of the best fitting candidate when it exists, then a
      larger context from `SearchSpace.fill`, then — always last — the threads trial for a
      GPU+CPU best.

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
    if not space.fits(last, m, history) and not any(space.fits(c, mm, history)
                                                  for c, mm in history):
        nxt = space.down(last, m) if (m.resource_failure or m.ok) else None
        return nxt if nxt and profile_key(nxt.profile) not in seen else None
    best = choose_best(space, history)
    if best is None:
        return None
    for cand in (space.all_gpu(best[0]), space.moe_fill(history), space.fill(history)):
        if cand is not None and profile_key(cand.profile) not in seen:
            return cand
    nxt = space.threads(best[0])
    return nxt if nxt and profile_key(nxt.profile) not in seen else None


# --------------------------------------------------------------------------- tune

def _backend(host: Host) -> str:
    """Backend name of the host's first device, or "cpu"."""
    return host.gpus[0].backend if host.gpus else "cpu"


def trial_budget(info: ModelInfo, max_trials: int) -> int:
    """How many launches a tune may make: MoE models with a known expert split get extra
    trials for the ``n_cpu_moe`` search.

    Args:
        info: Model metadata.
        max_trials: The requested budget (``--trials``).
    """
    return (max_trials + MOE_EXTRA_TRIALS
            if info.expert_count > 0 and info.expert_bytes else max_trials)


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


def _why(space: SearchSpace, best: Candidate, m: Measurement,
         history: History) -> str:
    """Why the winner's ``n_cpu_moe`` is what it is, against the lower counts tried.

    Only for a winner with experts in RAM: the headroom it kept, then — for every other
    trial of its family that kept fewer experts in RAM, ascending — why it lost (out of
    memory, its error kind, too little headroom left, a suspected spill, or slower), and
    finally the winner's prompt speed, which is what the CPU experts cost.

    Args:
        space: Search space (for the family key and reject reasons).
        best: The winning candidate.
        m: Its measurement.
        history: All trials, in order.
    """
    n = best.profile.n_cpu_moe
    if n <= 0:
        return ""
    free = (space.host.vram_free - m.vram_used) / GB
    out = f"n_cpu_moe {n}: fastest that kept {free:.1f} GB free"
    key = profile_key(best.profile.merged(ctx_size=0, n_cpu_moe=0))
    losers = sorted(
        ((c, mm) for c, mm in history
         if c.profile.n_cpu_moe < n
         and profile_key(c.profile.merged(ctx_size=0, n_cpu_moe=0)) == key),
        key=lambda cm: cm[0].profile.n_cpu_moe)
    for c, mm in losers:
        if not mm.ok:
            reason = ("ran out of memory" if mm.resource_failure
                      else mm.error.split(":")[0])
        elif r := space.reject_reason(c, mm, history):
            reason = ("left too little headroom" if r.startswith("left")
                      else "spilled to system RAM?")
        else:
            reason = "was slower"
        out += f"; {c.profile.n_cpu_moe} {reason}"
    return out + f"; prompt {m.prompt_tps:.0f} tok/s"


def tune(intent: Intent, info: ModelInfo, host: Host, base: Profile, runner: Runner, *,
         headroom_pct: float = 10, calibration: Calibration | None = None, max_trials: int = 6,
         progress: Callable[[TrialEvent], None] | None = None,
         cancel: Callable[[], bool] | None = None,
         goals: TuningGoals | None = None) -> TuneResult:
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
    if goals is not None:
        return tune_goals(intent, info, host, base, runner, goals, headroom_pct=headroom_pct,
                          calibration=calibration, max_trials=max_trials, progress=progress,
                          cancel=cancel)
    space = SearchSpace(intent, info, host, headroom_pct)
    plan = planner.plan(intent, info, host, base, headroom_pct, calibration)
    if plan.verdict == VERDICT_NO:
        why = plan.warnings[0] if plan.warnings else "does not fit this machine"
        return TuneResult(intent.key, None, None, [], False, f"not tried: {why}")
    mtime = _mtime(info.path)
    budget = trial_budget(info, max_trials)
    history: History = []
    trials: list[Trial] = []
    cand: Candidate | None = space.from_plan(plan)
    cancelled = False
    while cand is not None and len(history) < budget:
        if cancel is not None and cancel():
            cancelled = True
            break
        index = len(history) + 1
        if progress:
            progress(TrialEvent(index, budget, "start", cand.summary,
                                n_cpu_moe=cand.profile.n_cpu_moe))
        try:
            m = runner.measure(info.path, cand.profile, cancel)
        except KeyboardInterrupt:      # Ctrl-C: the runner has stopped its server; keep what we have
            cancelled = True
            break
        if m.cancelled:                # stopped in flight: not a load failure, not a trial
            cancelled = True
            if progress:
                progress(TrialEvent(index, budget, "done", cand.summary, m,
                                    n_cpu_moe=cand.profile.n_cpu_moe))
            break
        history.append((cand, m))
        trials.append(Trial(
            model=str(info.path), mtime=mtime, intent=intent.key, profile=cand.profile.to_dict(),
            estimated_total=planner.estimate_budget(info, cand.profile, host).raw_total(),
            verdict=cand.verdict, backend=_backend(host), measurement=m))
        if progress:
            note = space.reject_reason(cand, m, history) if m.ok else ""
            progress(TrialEvent(index, budget, "done", cand.summary, m, note,
                                cand.profile.n_cpu_moe))
        cand = next_candidate(space, history)
    best = choose_best(space, history)
    notes = tuple(space.reject_reason(c, m, history) if m.ok else "" for c, m in history)
    if best is None:
        last = history[-1][1].error if history else ""
        why = ("cancelled before any trial ran" if cancelled and not history else
               f"nothing loaded after {len(history)} trial{'s' * (len(history) != 1)}: {last}")
        return TuneResult(intent.key, None, None, trials, False, why, cancelled, notes)
    msg = _verdict_message(space, best[0], best[1], len(history))
    return TuneResult(intent.key, best[0], best[1], trials, True,
                      f"cancelled; best so far: {msg}" if cancelled else msg, cancelled,
                      notes, _why(space, best[0], best[1], history))


def goal_failure(goals: TuningGoals, measurement: Measurement) -> str:
    """Explain a failed throughput constraint without averaging distinct workloads.

    Args:
        goals: Preset throughput floors.
        measurement: Measured trial, possibly with individual workload results.

    Returns:
        First violated workload/floor, or an empty string when every floor passes.
    """
    if not measurement.ok:
        return measurement.error
    workloads = measurement.workloads or (measurement,)
    for workload in workloads:
        label = getattr(workload, "label", "benchmark")
        for name, value, floor in (("effective", workload.effective_tps, goals.min_effective_tps),
                                   ("input", workload.prompt_tps, goals.min_prompt_tps),
                                   ("generation", workload.gen_tps, goals.min_gen_tps)):
            if floor and (not math.isfinite(value) or value < floor):
                return f"{label}: {name} {value:.1f} TPS below {floor:g} minimum"
    return ""


def _bisect_next(lo: int, hi: int) -> int | None:
    """Next context probe between a proven pass and a proven failure, or None.

    Args:
        lo: Largest context per slot that passed every floor (0 for none yet).
        hi: Smallest context that provably failed.

    Returns:
        The interval midpoint aligned down to a `CTX_STEP` multiple (the raw midpoint
        when the aligned one would not advance), or None when the open interval is
        within ~`CTX_BISECT_TOL` of `hi` or `hi` is at the 256-token minimum.
    """
    if hi <= 256 or hi - lo <= max(1024, int(hi * CTX_BISECT_TOL)):
        return None
    mid = (lo + hi) // 2
    aligned = mid // CTX_STEP * CTX_STEP
    return aligned if lo < aligned < hi else mid


def tune_goals(intent: Intent, info: ModelInfo, host: Host, base: Profile, runner: Runner,
               goals: TuningGoals, *, headroom_pct: float = 10,
               calibration: Calibration | None = None, max_trials: int = 6,
               progress: Callable[[TrialEvent], None] | None = None,
               cancel: Callable[[], bool] | None = None) -> TuneResult:
    """Search for the largest tested context whose measured turns satisfy every floor.

    Candidates grow toward the ceiling while they pass. A failure bounds the search:
    memory failures first try the placement ladder at the same context (more experts
    to RAM, q8_0 KV, llama.cpp placement), then the context is bisected against the
    largest proven pass until the open interval is within ~5%. Placement at each
    probed context is corrected with measured VRAM: a measured slack that proves the
    model fits fully on the GPU overrides a plan that chose a spill, and measured
    bytes that prove an all-GPU plan cannot hold the context step straight to a MoE
    expert split or llama.cpp's own placement instead of spending a launch on an OOM.

    Args:
        intent: Purpose inherited by the preset.
        info: Parsed model metadata.
        host: Available hardware before tuning.
        base: Base launch settings.
        runner: Injectable measurement implementation.
        goals: Persistent speed floors, context ceiling and benchmark suite.
        headroom_pct: Device memory reserve.
        calibration: Existing memory calibration.
        max_trials: Maximum candidate launches before reporting incomplete search.
        progress: Optional trial event callback.
        cancel: Optional cancellation predicate.

    Returns:
        Best passing candidate, or an unmet-goal outcome; no configuration is saved.
    """
    intent = goals.resolve_intent(intent)
    if info.is_embedding:
        goals = replace(goals, min_effective_tps=0, min_gen_tps=0)
    space = SearchSpace(intent, info, host, headroom_pct)
    plan = planner.plan(intent, info, host, base, headroom_pct, calibration)
    if plan.verdict == VERDICT_NO:
        return TuneResult(intent.key, None, None, message="goals unmet: model does not fit available memory")
    if isinstance(runner, ServerRunner):
        runner.goals = goals
    ceiling = planner.ctx_ceiling(intent, info)
    history: History = []
    trials: list[Trial] = []
    notes: list[str] = []
    passing: list[tuple[Candidate, Measurement]] = []
    seen: set[tuple] = set()
    lo = 0                      # largest context per slot that passed every floor
    hi: int | None = None       # smallest context that provably failed
    cancelled = complete = False
    last_failure = ""
    budget = trial_budget(info, max_trials)

    def predicted_gpu_bytes(prof: Profile, context: int) -> int | None:
        """Measured-VRAM prediction for an all-GPU launch of `prof` at `context`.

        Anchored on the largest matching all-GPU trial (same KV cache types and slot
        count, no experts in RAM); the bytes-per-token slope is measured between the
        two largest such points, else the planner's KV estimate.

        Returns:
            Predicted device bytes used, or None without a measured anchor.
        """
        pts = sorted({c.ctx_per_slot: m.vram_used for c, m in history
                      if m.ok and c.profile.gpu_layers == "all" and c.profile.n_cpu_moe == 0
                      and c.profile.cache_type_k == prof.cache_type_k
                      and c.profile.cache_type_v == prof.cache_type_v
                      and c.profile.parallel == prof.parallel and m.vram_used > 0}.items())
        if not pts:
            return None
        slope = 0.0
        if len(pts) >= 2 and pts[-1][0] > pts[-2][0]:
            slope = (pts[-1][1] - pts[-2][1]) / (pts[-1][0] - pts[-2][0])
        if slope <= 0:
            slope = (planner.kv_per_token(info, prof.cache_type_k, prof.cache_type_v)
                     * max(1, prof.parallel))
        if slope <= 0:
            return None
        return pts[-1][1] + int(slope * (context - pts[-1][0]))

    def degrade(cand: Candidate, need: int, how: str) -> Candidate:
        """Placement for a context measured not to fit all-GPU: more experts to RAM on a
        MoE model, else llama.cpp's own placement (``fit`` on).

        Args:
            cand: The all-GPU candidate predicted not to fit.
            need: Its predicted device bytes.
            how: Reason for the probe.

        Returns:
            The replacement candidate, or `cand` unchanged when no alternative exists
            (it will then fail honestly) or it was already tried.
        """
        p = cand.profile
        out = cand
        if space._moe():
            overflow = need - (host.vram_free - space._headroom())
            target = planner.expert_ram_bytes(info, p) + max(overflow, 1)
            split = planner.moe_split(info, target)
            n = min(info.n_layer, max(p.n_cpu_moe + 1, split[0] if split else info.n_layer))
            out = space.candidate(p.merged(n_cpu_moe=n, fit="off"), how)
        elif host.gpus and not host.unified:
            mib = max(planner.headroom_bytes(g.total, headroom_pct)
                      for g in host.gpus) // planner.MiB
            out = space.candidate(p.merged(gpu_layers="auto", fit="on", fit_target=mib), how)
        return cand if profile_key(out.profile) in seen else out

    def at_context(context: int, how: str) -> Candidate:
        """Replan placement for the candidate's actual context, keeping unrelated edits,
        then correct it with measured VRAM where the estimate disagrees.

        Args:
            context: Requested context per slot for this trial.
            how: Human-readable reason for the next trial.

        Returns:
            A candidate with placement chosen at this context, rather than at the ceiling.
        """
        fixed = replace(intent, ctx_min=context, ctx_target=context, ctx_max=context)
        proposed = planner.plan(fixed, info, host, base, headroom_pct, calibration)
        cand = space.with_ctx(space.from_plan(proposed), context, how)
        p = cand.profile
        if cand.verdict == VERDICT_SPILL:
            upgrade = p.merged(gpu_layers="all", fit="off", fit_target=0, n_cpu_moe=0)
            need = predicted_gpu_bytes(upgrade, context)
            if (need is not None and host.vram_free - need >= space._headroom()
                    and profile_key(upgrade) not in seen):
                cand = space.candidate(upgrade, how)
        elif p.gpu_layers == "all" and p.n_cpu_moe == 0:
            need = predicted_gpu_bytes(p, context)
            if need is not None and host.vram_free - need < space._headroom():
                cand = degrade(cand, need, how)
        return cand

    candidate = at_context(min(plan.ctx_per_slot, 32768, ceiling), HOW_PLAN)
    while len(trials) < budget:
        if cancel and cancel():
            cancelled = True
            break
        index = len(trials) + 1
        if progress:
            progress(TrialEvent(index, budget, "start", candidate.summary,
                                n_cpu_moe=candidate.profile.n_cpu_moe))
        try:
            measurement = runner.measure(info.path, candidate.profile, cancel)
        except KeyboardInterrupt:
            cancelled = True
            break
        if measurement.cancelled:
            cancelled = True
            if progress:
                progress(TrialEvent(index, budget, "done", candidate.summary, measurement,
                                    n_cpu_moe=candidate.profile.n_cpu_moe))
            break
        history.append((candidate, measurement))
        seen.add(profile_key(candidate.profile))
        reject = space.reject_reason(candidate, measurement, history) if measurement.ok else ""
        failure = (reject if measurement.ok else measurement.error) or goal_failure(goals, measurement)
        last_failure = failure
        notes.append(failure)
        trials.append(Trial(model=str(info.path), mtime=_mtime(info.path), intent=intent.key,
                            profile=candidate.profile.to_dict(),
                            estimated_total=planner.estimate_budget(info, candidate.profile,
                                                                    host).raw_total(),
                            verdict=candidate.verdict, backend=_backend(host),
                            measurement=measurement))
        if progress:
            progress(TrialEvent(index, budget, "done", candidate.summary, measurement,
                                failure, candidate.profile.n_cpu_moe))
        if not failure:
            passing.append((candidate, measurement))
            lo = candidate.ctx_per_slot
            if lo >= ceiling:
                complete = True
                break
            nxt = (min(ceiling, max(lo + 1, lo * 2)) if hi is None
                   else _bisect_next(lo, hi))
            if nxt is None or nxt <= lo:
                complete = True
                break
            candidate = at_context(nxt, HOW_CTX_UP if hi is None else HOW_BISECT)
            continue
        # Memory failures get the placement ladder first (same context); anything else
        # that cannot be retried at this context bounds the bisection.
        resource = (not measurement.ok and measurement.resource_failure) or bool(reject)
        if not measurement.ok and not resource:
            complete = True            # e.g. unknown argument: more trials will not help
            break
        if resource:
            smaller = space.down(candidate, measurement)
            if (smaller is not None and profile_key(smaller.profile) not in seen
                    and smaller.ctx_per_slot > lo):
                if smaller.ctx_per_slot < candidate.ctx_per_slot:
                    hi = min(hi, candidate.ctx_per_slot) if hi is not None \
                        else candidate.ctx_per_slot
                candidate = smaller
                continue
        hi = min(hi, candidate.ctx_per_slot) if hi is not None else candidate.ctx_per_slot
        nxt = _bisect_next(lo, hi)
        if nxt is None:
            complete = True
            break
        candidate = at_context(nxt, HOW_BISECT)
    status = "cancelled; " if cancelled else "" if complete else "trial budget reached; "
    if not passing:
        return TuneResult(intent.key, None, None, trials=trials, ok=False,
                          message=f"{status}goals unmet: {last_failure or 'no passing trial'}",
                          cancelled=cancelled, notes=tuple(notes))
    best, measurement = max(passing, key=lambda cm: cm[0].ctx_per_slot)
    why = (f"Largest tested context meeting every enabled speed floor: {best.ctx_per_slot} "
           f"tokens/request; {measurement.effective_tps:.1f} effective output TPS per turn "
           f"(cached prefix + new input). {goals.benchmark.capitalize()} workload suite.")
    reduced = [w.label for w in measurement.workloads if "reduced" in w.label]
    if reduced:
        why += " Reduced workload coverage: " + "; ".join(reduced) + "."
    message = f"{status}{why}"
    return TuneResult(intent.key, best, measurement, trials, True, message, cancelled,
                      tuple(notes), why)


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
    previous = cfg.preset(str(model), result.intent)
    cfg.set_preset(str(model), Preset(result.intent, result.best.profile, TUNED,
                                      result.measurement, previous.goals if previous else TuningGoals()),
                   activate=activate)
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
        support: Records a failed load's ``unknown model architecture`` for this build
            (or None).
        goals: Optional workload suite and goals for measured requests.
    """

    def __init__(self, cfg: Config, *, popen: Callable[..., Any] = subprocess.Popen,
                 http: Callable[..., Any] = httpx.Client,
                 devices: Callable[[], list[Device]] | None = None,
                 run: Callable[..., Any] = subprocess.run,
                 ram_free: Callable[[], int] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 support: SupportChecker | None = None,
                 goals: TuningGoals | None = None) -> None:
        self.cfg = cfg
        self.goals = goals
        self._popen = popen
        self._http = http
        self._devices = devices or (lambda: live_devices(cfg.bin_dir))
        self._run = run
        self._ram_free = ram_free or (lambda: hardware.read_memory()[1])
        self._clock = clock
        self._sleep = sleep
        self._support = support
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

    def _settle(self, baseline: int) -> None:
        """Wait for the stopped server's device memory to come back.

        Drivers (WDDM in particular) release device memory lazily; the next trial's
        pre-launch snapshot would otherwise see less free memory than it really has,
        and a ``fit`` placement would under-offload. Polls until the free total is back
        at `baseline` (free bytes before the launch), stops changing, or `SETTLE_MAX`
        passes; never raises.
        """
        last, stable = -1, 0
        deadline = self._clock() + SETTLE_MAX
        while self._clock() < deadline and stable < 2:
            try:
                free = sum(d.free for d in self._devices())
            except Exception:
                return
            if free >= baseline:
                return
            stable = stable + 1 if 0 <= last and free <= last else 0
            last = free
            self._sleep(SETTLE_POLL)

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
            self._settle(vram_before)
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
                    tail = self._tail(log)
                    if self._support is not None and (arch := unknown_arch(tail)):
                        self._support.record_failure(arch, build)
                    return failed(classify_failure(tail), self._clock() - start)
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
                workloads: tuple[WorkloadMeasurement, ...] = ()
                if self.goals is not None and not prof.embedding:
                    workloads = self._workloads(client, (prof.ctx_size or 4096) //
                                                max(1, prof.parallel), cancel)
                    prompt_tps = min(w.prompt_tps for w in workloads)
                    gen_tps = min(w.gen_tps for w in workloads)
                else:
                    prompt_tps, gen_tps = (self._embed(client) if prof.embedding else
                                           self._complete(client, (prof.ctx_size or 4096) //
                                                           max(1, prof.parallel)))
            except (httpx.HTTPError, ValueError, KeyError, TypeError, OverflowError) as e:
                error = str(e)
                return failed(error if error.startswith(ERR_CANCELLED)
                              else f"{ERR_BENCH}: {error}", load_s)
            vram_end, ram_end = self._snapshot()
        return Measurement(
            ok=True, load_s=load_s, vram_used=max(0, vram_before - min(vram_load, vram_end)),
            ram_used=max(0, ram_before - min(ram_load, ram_end)), prompt_tps=prompt_tps,
            gen_tps=gen_tps, at=_now(), build=build,
            effective_tps=min((w.effective_tps for w in workloads), default=0.0),
            request_s=sum(w.request_s for w in workloads),
            prompt_tokens=sum(w.prompt_tokens for w in workloads),
            generated_tokens=sum(w.generated_tokens for w in workloads),
            cached_tokens=sum(w.cached_tokens for w in workloads), workloads=workloads)

    def _completion(self, client: Any, prompt: list[int], n_predict: int,
                    cache_prompt: bool, timeout: float) -> tuple[dict, float]:
        """One ``POST /completion`` pinned to slot 0; returns (body, wall seconds).

        Args:
            client: Trial server HTTP client.
            prompt: Prompt as token IDs.
            n_predict: Tokens to generate.
            cache_prompt: Keep the prompt in the KV cache for reuse (the realistic
                setting); False only marks the cold turns.
            timeout: Request timeout in seconds.

        Returns:
            (response JSON, request wall time).

        Raises:
            ValueError: On a non-200 status.
        """
        started = self._clock()
        response = client.request("POST", "/completion", timeout=timeout, json={
            "id_slot": 0, "prompt": prompt, "n_predict": n_predict,
            "cache_prompt": cache_prompt, "seed": 42, "temperature": 0, "stream": False,
            "ignore_eos": True})
        elapsed = self._clock() - started
        if response.status_code != 200:
            raise ValueError(f"completion HTTP {response.status_code}")
        return response.json(), elapsed

    def _prime(self, client: Any, prompt: list[int], n_predict: int = PRIME_PREDICT) -> None:
        """Seed slot 0's KV cache with `prompt` without measuring it.

        Used for the discarded warmup (which also generates a few tokens to warm the
        decode path) and to place the cached prefix — or evict the previous sequence —
        before a turn request. Only the HTTP status is checked.

        Raises:
            ValueError: On a non-200 status.
        """
        self._completion(client, prompt, n_predict, True,
                         max(BENCH_TIMEOUT, len(prompt) / 20 + 30))

    def _turn(self, client: Any, label: str, prompt: list[int], output: int,
              cache_prefix: int) -> WorkloadMeasurement:
        """One measured turn at `cache_prefix` expected cached tokens.

        `cache_prefix` 0 is a cold turn (no cache coverage allowed); larger values
        require the server to report that prefix served from the cache (up to one
        turn's worth of boundary loss). Turn duration is the server's own prompt +
        predicted time when reported, else the request's wall time — serializing and
        transporting the prompt is client cost, not turn time.

        Args:
            client: Trial server HTTP client.
            label: Workload label for error messages and reports.
            prompt: Turn prompt as token IDs (prefix + delta).
            output: ``n_predict`` for the turn.
            cache_prefix: Tokens of `prompt`'s head expected already in the KV cache.

        Returns:
            The measured workload.

        Raises:
            ValueError: On truncation, unverified cache coverage, or invalid timings.
        """
        body, elapsed = self._completion(client, prompt, output, cache_prefix > 0,
                                         max(BENCH_TIMEOUT, len(prompt) / 20 + 30))
        if body.get("truncated"):
            raise ValueError(f"{label}: server truncated benchmark context")
        timings = body["timings"]
        generated = int(timings["predicted_n"])
        processed = int(timings["prompt_n"])
        cached = int(timings.get("cache_n", 0))
        prompt_rate = float(timings["prompt_per_second"])
        gen_rate = float(timings["predicted_per_second"])
        prompt_ms, predicted_ms = timings.get("prompt_ms"), timings.get("predicted_ms")
        server_s = ((float(prompt_ms) + float(predicted_ms)) / 1000
                    if isinstance(prompt_ms, (int, float))
                    and isinstance(predicted_ms, (int, float)) else elapsed)
        count = len(prompt)
        if (generated <= 0 or generated > output or not math.isfinite(server_s)
                or server_s <= 0):
            raise ValueError(f"{label}: invalid output count or request duration")
        # The measured turn must be the turn it claims: every prompt token accounted
        # for, nothing cached on a cold turn, and a cached-prefix turn must actually
        # have reused its prefix (a little boundary loss is tolerated).
        if (cached + processed != count or processed <= 0
                or (cache_prefix == 0 and cached != 0)
                or cached < max(0, cache_prefix - TURN_TOKENS)):
            raise ValueError(f"{label}: cached-prefix coverage not verified "
                             f"({cached} cached + {processed} processed of {count})")
        if any(not math.isfinite(rate) or rate <= 0 for rate in (prompt_rate, gen_rate)):
            raise ValueError(f"{label}: invalid prompt/generation speed")
        return WorkloadMeasurement(label, processed, generated, server_s,
                                   prompt_rate, gen_rate, cached)

    def _near_floor(self, w: WorkloadMeasurement) -> bool:
        """True when the sample is not comfortably above every enabled floor.

        A sample within `NEAR_FLOOR` of any floor is measured twice more and the
        median stands, so one noisy request cannot pass or fail a candidate alone.
        """
        goals = self.goals
        assert goals is not None
        return any(floor > 0 and value < floor * NEAR_FLOOR
                   for value, floor in ((w.effective_tps, goals.min_effective_tps),
                                        (w.prompt_tps, goals.min_prompt_tps),
                                        (w.gen_tps, goals.min_gen_tps)))

    def _workloads(self, client: Any, context: int,
                   cancel: Callable[[], bool] | None = None) -> tuple[WorkloadMeasurement, ...]:
        """Measure practical turns at increasing context occupancy.

        A workload is the interaction a real client produces: a prefix already in the
        KV cache plus a turn's new input and generation (`TURN_TOKENS` delta + the
        generated answer). Occupancies at or under `TURN_TOKENS` have no prefix to
        reuse and are measured cold. One warmup request runs first so kernel
        initialization and clock ramp are not charged to the first workload.

        Args:
            client: Trial server HTTP client.
            context: Actual context per request.
            cancel: Cancellation check between workload requests.

        Returns:
            Individual workloads, each including actual tokens and turn duration.

        Raises:
            ValueError: If tokenization or completion fails or no tokens are generated.
        """
        assert self.goals is not None
        output = min(128, max(1, context // 20))
        available = max(1, int(context * 0.95) - output - 8)
        if self.goals.benchmark == "full":
            requested = [(f"{pct}% context", int(context * pct / 100))
                         for pct in (5, 25, 50, 75, 95)]
        else:
            requested = [("Quick 2k", 2048)]
            if self.goals.benchmark == "normal":
                requested.append(("Harness 25k", 25000))
        maximum = max(min(tokens, available) for _, tokens in requested)
        # Tokenize once for all workloads, then send token IDs to avoid character estimates
        # being mistaken for the model's actual input occupancy.
        paragraph = _BENCH_PARA.format(n=1)
        text = " ".join(paragraph for _ in range(maximum // 60 + 1))
        tokens: list[int] = []
        for _ in range(8):
            response = client.request("POST", "/tokenize", json={
                "content": text, "add_special": True})
            if response.status_code != 200:
                raise ValueError(f"tokenization HTTP {response.status_code}")
            tokens = response.json()["tokens"]
            if not isinstance(tokens, list) or any(type(t) is not int for t in tokens):
                raise ValueError("tokenizer did not return token IDs")
            if len(tokens) >= maximum:
                break
            text += " " + text
        if len(tokens) < maximum:
            raise ValueError("tokenizer returned too few tokens for requested workloads")
        evict = next(([t] for t in tokens if t != tokens[0]), tokens[:1])
        # Warmup: the first inference pays kernel/graph init, clock ramp and page-in;
        # its result is discarded so those costs land outside the measured turns.
        self._prime(client, tokens[:min(WARMUP_TOKENS, maximum)], WARMUP_PREDICT)
        results: list[WorkloadMeasurement] = []
        for label, wanted in requested:
            count = min(max(1, wanted), available)
            if count < wanted:
                label += f" (reduced to {count} input tokens for this context)"
            prefix = max(0, count - TURN_TOKENS)
            samples: list[WorkloadMeasurement] = []
            for attempt in range(3):
                if attempt and not self._near_floor(samples[0]):
                    break               # comfortably above every floor: one sample is enough
                if cancel and cancel():
                    raise ValueError(f"{ERR_CANCELLED}: stopped between workloads")
                if prefix:
                    self._prime(client, tokens[:prefix])        # cached-prefix turn
                else:
                    self._prime(client, evict)                  # make the next turn cold
                samples.append(self._turn(client, label, tokens[:count], output, prefix))
            keep = sorted(samples, key=lambda w: w.effective_tps)[len(samples) // 2]
            results.append(replace(keep, samples=len(samples)))
        return tuple(results)

    @staticmethod
    def _complete(client: Any, ctx_per_slot: int) -> tuple[float, float]:
        """Benchmark a generative model with ``POST /completion``.

        Args:
            client: HTTP client for the trial server.
            ctx_per_slot: Context each slot gets; the prompt takes about half of it (capped
                at ``BENCH_PROMPT_TOKENS``) so prompt processing exercises a realistic part
                of the window.

        Returns:
            (prompt tokens/s, generation tokens/s) from the response's ``timings``.

        Raises:
            ValueError: On an HTTP error status, a body without ``timings``, or no speed.
        """
        r = client.request("POST", "/completion", timeout=BENCH_TIMEOUT, json={
            "prompt": bench_prompt(ctx_per_slot), "n_predict": 128, "cache_prompt": False,
            "seed": 42, "temperature": 0, "stream": False})
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
