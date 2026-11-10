"""Memory budget model and intent-driven launch planning.

Everything here is pure arithmetic over ``ModelInfo``, ``Host`` and ``Profile``: no I/O, no
subprocesses. ``plan`` turns an intent into concrete settings for one model on one host and
explains every setting it changes; ``plan_all`` does so for every applicable intent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NamedTuple

from .calibration import Calibration
from .config import Profile
from .gguf import ModelInfo
from .hardware import Host
from .intents import KV_IF_NEEDED, KV_NEVER, KV_PREFER, SPILL_ALLOW, SPILL_AVOID, Intent, applicable

MiB = 1 << 20
GB = 1e9
OVERHEAD = 512 * MiB          # fixed runtime overhead
COMPUTE_MIN = 256 * MiB       # minimum compute/scratch buffer
HEADROOM_MIN = 512 * MiB      # smallest memory margin kept free per device
CTX_TIERS = tuple(1 << n for n in range(12, 19))     # 4096 .. 262144

BYTES_PER_ELEM = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32,
                  "q5_0": 22 / 32, "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32}
F16, Q8 = "f16", "q8_0"

VERDICT_GPU, VERDICT_SPILL, VERDICT_CPU, VERDICT_NO = "GPU", "GPU+CPU", "CPU", "NO"

# placement modes of a layout
_GPU, _SPILL, _MOE, _CPU = "gpu", "spill", "moe", "cpu"


@dataclass(frozen=True)
class Budget:
    """Memory a launch needs, split by purpose (bytes).

    Attributes:
        weights: Model weights (ModelInfo.size).
        kv: KV cache for the configured context and cache types.
        compute: Activation/scratch buffer heuristic.
        overhead: Fixed runtime overhead.
        scale: Calibration factor applied to the total (1.0 = uncalibrated).
    """

    weights: int
    kv: int
    compute: int
    overhead: int
    scale: float = 1.0

    def raw_total(self) -> int:
        """All components summed, before calibration."""
        return self.weights + self.kv + self.compute + self.overhead

    def total(self) -> int:
        """All components summed and multiplied by the calibration factor."""
        return round(self.raw_total() * self.scale)

    def fits_vram(self, host: Host) -> bool:
        """True when the total fits in currently free VRAM."""
        return self.total() <= host.vram_free

    def fits_ram(self, host: Host) -> bool:
        """True when the total fits in currently free RAM."""
        return self.total() <= host.ram_free


def kv_per_token(info: ModelInfo, ctk: str, ctv: str) -> int:
    """Bytes of KV cache per token for the given cache types.

    Args:
        info: Model metadata.
        ctk: K cache type.
        ctv: V cache type.

    Returns:
        Bytes per token.
    """
    return int(info.n_layer * info.n_head_kv *
               (info.head_dim_k * BYTES_PER_ELEM[ctk] + info.head_dim_v * BYTES_PER_ELEM[ctv]))


def placeholder_info(size: int) -> ModelInfo:
    """Size-only ModelInfo for files whose header has not been parsed yet.

    Args:
        size: File size in bytes.

    Returns:
        ModelInfo with all architecture fields zeroed (kv = 0).
    """
    return ModelInfo(path=Path(), size=size, architecture="", name="", n_layer=0, n_embd=0,
                     n_head=0, n_head_kv=0, head_dim_k=0, head_dim_v=0, ctx_train=0,
                     expert_count=0, file_type=0, sharded=1, parameter_count=0)


def calibration_factor(host: Host, calibration: Calibration | None) -> float:
    """Correction factor for the host's backend (its first GPU's).

    Args:
        host: Host snapshot.
        calibration: Measured data, or None.

    Returns:
        The factor; 1.0 without a GPU or without data.
    """
    if calibration is None or not host.gpus:
        return 1.0
    return calibration.factor(host.gpus[0].backend)


def _budget(info: ModelInfo, prof: Profile, factor: float) -> Budget:
    """Budget of a profile with an explicit calibration factor."""
    ctx = prof.ctx_size or info.ctx_train
    # -c is the total KV pool: parallel slots each get ctx // np, not ctx each
    kv = kv_per_token(info, prof.cache_type_k, prof.cache_type_v) * ctx
    compute = max(COMPUTE_MIN, info.n_embd * (prof.ubatch_size or 512) * 4 * 8)
    return Budget(weights=info.size, kv=kv, compute=compute, overhead=OVERHEAD, scale=factor)


def estimate_budget(info: ModelInfo, prof: Profile, host: Host,
                    calibration: Calibration | None = None) -> Budget:
    """Memory budget for a model under a profile on a host.

    ctx_size 0 means the model's trained context; ubatch_size 0 means 512.

    Args:
        info: Model metadata (placeholder allowed: kv comes out 0).
        prof: Profile to evaluate.
        host: Host snapshot; its first GPU's backend selects the calibration factor.
        calibration: Measured data scaling the total, or None for the raw estimate.

    Returns:
        The budget; ``total()`` includes the calibration factor.
    """
    return _budget(info, prof, calibration_factor(host, calibration))


def headroom_bytes(total: int, headroom_pct: float) -> int:
    """Memory kept free on a device or pool.

    Args:
        total: Device memory in bytes.
        headroom_pct: Percentage of `total` to keep free.

    Returns:
        ``max(headroom_pct% of total, 512 MiB)``.
    """
    return max(int(headroom_pct / 100 * total), HEADROOM_MIN)


def device_usable(total: int, headroom_pct: float) -> int:
    """Usable memory of one device: its total minus headroom (never negative)."""
    return max(0, total - headroom_bytes(total, headroom_pct))


def gpu_limit(host: Host, headroom_pct: float = 10) -> int:
    """Bytes a launch may occupy when everything stays in GPU memory.

    The sum of each device's total minus its own headroom. On a unified host the GPU memory
    is system RAM, so the pool is also capped by RAM total minus headroom.

    Args:
        host: Host snapshot.
        headroom_pct: Percentage of each device kept free (min 512 MiB).

    Returns:
        Usable bytes; 0 without a GPU.
    """
    usable = sum(device_usable(g.total, headroom_pct) for g in host.gpus)
    if host.unified and host.ram_total:
        usable = min(usable, device_usable(host.ram_total, headroom_pct))
    return usable


def spill_limit(host: Host, headroom_pct: float = 10) -> int:
    """Bytes a launch may occupy when weights may spill from the GPU into RAM.

    Usable GPU memory plus free RAM; on a unified host they are one pool, so just the former.

    Args:
        host: Host snapshot.
        headroom_pct: Percentage of each device kept free (min 512 MiB).

    Returns:
        Usable bytes.
    """
    limit = gpu_limit(host, headroom_pct)
    return limit if host.unified else limit + host.ram_free


def verdict(budget: Budget, host: Host, headroom_pct: float = 10) -> str:
    """Verdict label for a budget on a host: "GPU", "GPU+CPU", "CPU" or "NO".

    Args:
        budget: Memory budget of the launch.
        host: Host resource snapshot.
        headroom_pct: Fraction of each device kept free (min 512 MiB).

    Returns:
        "GPU" when it fits GPU memory with headroom, "GPU+CPU" when it needs RAM too,
        "CPU" on a GPU-less host, "NO" when nothing fits.
    """
    total = budget.total()
    if not host.gpus:
        return VERDICT_CPU if total <= host.ram_free else VERDICT_NO
    if total <= gpu_limit(host, headroom_pct):
        return VERDICT_GPU
    if total <= spill_limit(host, headroom_pct):
        return VERDICT_SPILL
    return VERDICT_NO


@dataclass(frozen=True)
class Plan:
    """Settings for one intent on one model and host, with the reasoning.

    Attributes:
        intent: Intent key.
        profile: The planned profile.
        reasons: key -> one line with the numbers used, only for keys that differ from base.
        budget: Memory budget of the planned profile (calibrated).
        verdict: "GPU", "GPU+CPU", "CPU" or "NO".
        ctx_per_slot: Context each slot gets (total ``-c`` is this times the slots).
        warnings: Things the user should know, e.g. a context below the intent's minimum.
    """

    intent: str
    profile: Profile
    reasons: dict[str, str]
    budget: Budget
    verdict: str
    ctx_per_slot: int
    warnings: tuple[str, ...] = field(default_factory=tuple)


class _Pick(NamedTuple):
    """A context per slot together with the KV cache type that reaches it."""

    ctx: int
    kv: str


@dataclass(frozen=True)
class _Layout:
    """The planner's decision: context, KV type and where the weights go.

    Attributes:
        ctx: Context per slot.
        kv: KV cache type for K and V.
        mode: Weight placement ("gpu", "spill", "moe" or "cpu").
        floor_spill: True when weights were moved off the GPU only to reach the intent's
            minimum context (a spill-"avoid" intent whose GPU-only context was too small).
    """

    ctx: int
    kv: str
    mode: str
    floor_spill: bool = False


@dataclass(frozen=True)
class _Env:
    """Everything the search needs about one planning problem.

    Attributes:
        intent: Intent being planned.
        info: Model metadata.
        host: Host snapshot.
        base: Base profile with the intent's slots and extras applied.
        pct: Headroom percentage.
        factor: Calibration factor.
        cands: Candidate contexts per slot, ascending.
        target: Context per slot the intent aims for (resolved, capped by the trained ctx).
        floor: Smallest acceptable context per slot (resolved).
        gpu_limit: Bytes available with everything in GPU memory.
        spill_limit: Bytes available when weights may leave the GPU.
    """

    intent: Intent
    info: ModelInfo
    host: Host
    base: Profile
    pct: float
    factor: float
    cands: tuple[int, ...]
    target: int
    floor: int
    gpu_limit: int
    spill_limit: int

    def budget(self, ctx: int, kv: str) -> Budget:
        """Budget of `ctx` tokens per slot with K and V both stored as `kv`."""
        prof = self.base.merged(ctx_size=ctx * self.intent.slots, cache_type_k=kv,
                                cache_type_v=kv)
        return _budget(self.info, prof, self.factor)

    def fits(self, ctx: int, kv: str, limit: int) -> bool:
        """True when `ctx` per slot with `kv` cache fits in `limit` bytes.

        On a MoE model on a GPU host, the attention weights, KV cache and buffers must also fit
        GPU memory, because only the experts can move to RAM.
        """
        b = self.budget(ctx, kv)
        if b.total() > limit:
            return False
        if self.info.expert_count and self.gpu_limit:
            return b.total() - round(b.weights * b.scale) <= self.gpu_limit
        return True


def _trained(info: ModelInfo) -> int:
    """The model's trained context, or the largest tier when the header has none."""
    return info.ctx_train or CTX_TIERS[-1]


def ctx_bounds(intent: Intent, info: ModelInfo) -> tuple[tuple[int, ...], int, int]:
    """Candidate contexts, resolved target and resolved minimum for an intent.

    Candidates are the power-of-two tiers from 4096 up to the target, plus the target itself
    when it is not a tier (a trained ctx like 40960, or 512 for a small embedding model).

    Args:
        intent: The intent.
        info: Model metadata.

    Returns:
        (candidates ascending, target, floor), all per slot.
    """
    cap = _trained(info)
    target = min(cap, intent.ctx_target or cap)
    floor = min(intent.ctx_min or cap, target)
    tiers = [t for t in CTX_TIERS if t <= target]
    if target not in tiers:
        tiers.append(target)
    return tuple(tiers), target, floor


def _make_env(intent: Intent, info: ModelInfo, host: Host, base: Profile, pct: float,
              calibration: Calibration | None) -> _Env:
    """Assemble the planning environment."""
    cands, target, floor = ctx_bounds(intent, info)
    prof = base.merged(parallel=intent.slots, **intent.extra)
    return _Env(intent, info, host, prof, pct, calibration_factor(host, calibration),
                cands, target, floor, gpu_limit(host, pct), spill_limit(host, pct))


def _largest(env: _Env, limit: int, kv: str) -> int:
    """Largest candidate context per slot that fits `limit` with `kv` cache; 0 when none."""
    fitting = [c for c in env.cands if env.fits(c, kv, limit)]
    return fitting[-1] if fitting else 0


def _search(env: _Env, limit: int) -> _Pick | None:
    """Best context and KV type within `limit` bytes under the intent's KV policy.

    "never" keeps f16. "prefer" uses q8_0 K and V. "if_needed" starts with f16 and switches to
    q8_0 only when that reaches a larger context than f16 does, and f16 missed the target.

    Args:
        env: Planning environment.
        limit: Bytes available.

    Returns:
        The pick, or None when not even the smallest candidate fits.
    """
    policy = env.intent.kv_quant
    if policy == KV_PREFER:
        ctx = _largest(env, limit, Q8)
        return _Pick(ctx, Q8) if ctx else None
    f16 = _largest(env, limit, F16)
    if policy == KV_IF_NEEDED and f16 < env.target:
        q8 = _largest(env, limit, Q8)
        if q8 > f16:
            return _Pick(q8, Q8)
    return _Pick(f16, F16) if f16 else None


def _fallback(env: _Env, mode: str) -> _Layout:
    """Best-effort layout when nothing fits: the smallest context, quantised KV if allowed."""
    kv = F16 if env.intent.kv_quant == KV_NEVER else Q8
    return _Layout(env.cands[0], kv, mode)


def _choose_cpu(env: _Env) -> _Layout:
    """Layout for a host without a GPU: context against free RAM."""
    pick = _search(env, env.host.ram_free)
    return _Layout(pick.ctx, pick.kv, _CPU) if pick else _fallback(env, _CPU)


def _choose_spill(env: _Env, gpu_pick: _Pick | None) -> _Layout:
    """Layout that lets weights leave the GPU.

    Args:
        env: Planning environment.
        gpu_pick: What fits GPU memory alone, if anything.

    Returns:
        A layout in spill (or MoE) mode when that gains context over `gpu_pick`; otherwise the
        GPU layout, or a best-effort one when nothing fits at all.
    """
    if env.spill_limit <= env.gpu_limit:       # unified memory: there is no second pool
        return _Layout(gpu_pick.ctx, gpu_pick.kv, _GPU) if gpu_pick else _fallback(env, _GPU)
    mode = _MOE if env.info.expert_count else _SPILL
    pick = _search(env, env.spill_limit)
    if pick is None:
        return _fallback(env, mode)
    if gpu_pick and pick.ctx <= gpu_pick.ctx:
        return _Layout(gpu_pick.ctx, gpu_pick.kv, _GPU)
    return _Layout(pick.ctx, pick.kv, mode)


def _spill_to_floor(env: _Env) -> _Layout | None:
    """Layout that spills weights only far enough to reach the intent's minimum context.

    Args:
        env: Planning environment.

    Returns:
        A spill (or MoE) layout at the smallest candidate context of at least the minimum, or
        None when there is no second memory pool or even spilling cannot reach the minimum.
    """
    if env.spill_limit <= env.gpu_limit:
        return None
    at_floor = min(c for c in env.cands if c >= env.floor)
    pick = _search(replace(env, cands=(at_floor,), target=at_floor), env.spill_limit)
    if pick is None:
        return None
    return _Layout(pick.ctx, pick.kv, _MOE if env.info.expert_count else _SPILL, True)


def _choose(env: _Env) -> _Layout:
    """Decide context, KV type and placement (SPEC 5.3 steps 2 to 5)."""
    if not env.host.gpus:
        return _choose_cpu(env)
    pick = _search(env, env.gpu_limit)
    if pick and pick.ctx >= env.target:
        return _Layout(pick.ctx, pick.kv, _GPU)
    if env.intent.spill == SPILL_AVOID:
        if pick and pick.ctx >= env.floor:
            return _Layout(pick.ctx, pick.kv, _GPU)
        # no usable GPU-only context (or none at all): spill, but only far enough for the floor
        return _spill_to_floor(env) or (_Layout(pick.ctx, pick.kv, _GPU) if pick
                                        else _choose_spill(env, None))
    return _choose_spill(env, pick)


def _gb(n: float) -> str:
    """Bytes as a short GB string."""
    return f"{n / GB:.1f} GB"


class _Draft:
    """A profile under construction that remembers why each key differs from the base.

    Args:
        base: The profile planning started from.
    """

    def __init__(self, base: Profile) -> None:
        self.base = base
        self.profile = base.merged()
        self.reasons: dict[str, str] = {}

    def set(self, key: str, value: object, why: str) -> None:
        """Set a key; it gets a reason only while it differs from the base profile."""
        self.profile = self.profile.merged(**{key: value})
        if getattr(self.base, key) == value:
            self.reasons.pop(key, None)
        else:
            self.reasons[key] = why


def _apply_fixed(d: _Draft, env: _Env) -> None:
    """Threads, parallel slots and the intent's extra settings."""
    host, intent = env.host, env.intent
    cores = host.cpu_cores or host.cpu_threads // 2
    if cores:
        d.set("threads", cores, f"host has {cores} physical cores")
    slots = intent.slots
    d.set("parallel", slots, f"{slots} slot{'s' * (slots != 1)} for the {intent.key} intent")
    for key, value in intent.extra.items():
        d.set(key, value, f"the {intent.key} intent sets {key} = {value}")


def _where(env: _Env, mode: str) -> str:
    """Phrase naming the memory a layout was fitted against."""
    if mode == _CPU:
        return f"{_gb(env.host.ram_free)} free RAM"
    if env.host.unified:
        return f"{_gb(env.gpu_limit)} usable unified memory"
    if mode == _GPU:
        return f"{_gb(env.gpu_limit)} usable VRAM"
    return f"{_gb(env.gpu_limit)} usable VRAM + {_gb(env.host.ram_free)} free RAM"


def _apply_ctx(d: _Draft, env: _Env, lay: _Layout) -> None:
    """Context size, KV cache types and flash attention."""
    slots = env.intent.slots
    total_ctx, b = lay.ctx * slots, env.budget(lay.ctx, lay.kv)
    calib = f" (x{env.factor:.2f} calibration)" if env.factor != 1.0 else ""
    d.set("ctx_size", total_ctx,
          f"{lay.ctx} ctx x {slots} slot{'s' * (slots != 1)} = {total_ctx} -c; "
          f"KV {_gb(b.kv)} ({lay.kv}); needs {_gb(b.total())} of {_where(env, lay.mode)}{calib}")
    f16_kv = env.budget(lay.ctx, F16).kv
    why = (f"{lay.kv} K/V: KV {_gb(b.kv)} instead of {_gb(f16_kv)} at f16" if lay.kv != F16
           else f"f16 K/V: the {env.intent.key} intent keeps the cache unquantised or it fits")
    d.set("cache_type_k", lay.kv, why)
    d.set("cache_type_v", lay.kv, why if lay.kv == F16 else "quantised V requires flash_attn on")
    if env.host.gpus:
        d.set("flash_attn", "on", "GPU present")
    elif lay.kv != F16:
        d.set("flash_attn", "on", "quantised V cache requires flash_attn on")
    else:
        d.set("flash_attn", "auto", "no GPU")


def _moe_layers(env: _Env, lay: _Layout) -> int:
    """Layers whose experts must stay in RAM so the rest fits usable VRAM."""
    info = env.info
    if info.n_layer <= 0 or info.size <= 0:
        return 0
    over = env.budget(lay.ctx, lay.kv).total() - env.gpu_limit
    return max(1, min(info.n_layer, math.ceil(over / (info.size / info.n_layer))))


def _fit_target_mib(env: _Env) -> int:
    """Per-device free-memory margin for ``--fit``, in MiB (the largest device headroom)."""
    return max(headroom_bytes(g.total, env.pct) for g in env.host.gpus) // MiB


def _apply_placement(d: _Draft, env: _Env, lay: _Layout) -> None:
    """GPU layers, fit, MoE expert placement and the multi-GPU split."""
    total = env.budget(lay.ctx, lay.kv).total()
    if lay.mode == _CPU:
        d.set("gpu_layers", "0", "no GPU detected")
        return
    if lay.mode == _GPU:
        d.set("gpu_layers", "all", f"whole model fits {_where(env, _GPU)}")
        d.set("fit", "off", "everything fits, so llama.cpp need not place layers")
        d.set("n_cpu_moe", 0, "no experts need to leave the GPU")
    elif lay.mode == _MOE:
        n = _moe_layers(env, lay)
        d.set("gpu_layers", "all", "all layers on GPU; MoE experts kept in RAM")
        d.set("n_cpu_moe", n,
              f"experts of {n}/{env.info.n_layer} layers stay in RAM: needs {_gb(total)} > "
              f"{_gb(env.gpu_limit)} usable VRAM")
        d.set("fit", "off", "expert placement is explicit (n_cpu_moe)")
    else:
        d.set("gpu_layers", "auto",
              f"needs {_gb(total)} but only {_gb(env.gpu_limit)} usable VRAM; "
              "llama.cpp places the layers")
        d.set("fit", "on", "llama.cpp fits the layers to free memory")
        mib = _fit_target_mib(env)
        d.set("fit_target", mib, f"keep {mib} MiB free per device")
    if len(env.host.gpus) > 1:
        shares = [device_usable(g.total, env.pct) for g in env.host.gpus]
        d.set("tensor_split", ",".join(f"{s / GB:.1f}" for s in shares),
              f"proportional to usable memory of {len(shares)} devices "
              f"({', '.join(_gb(s) for s in shares)})")


def _warnings(env: _Env, lay: _Layout, v: str) -> tuple[str, ...]:
    """Warnings for the user about a layout."""
    out: list[str] = []
    if v == VERDICT_NO:
        out.append(f"does not fit memory ({_where(env, lay.mode)}); values are best effort")
    if lay.ctx < env.floor:
        out.append(f"ctx {lay.ctx} per slot is below the {env.intent.key} minimum of {env.floor}")
    if lay.floor_spill:
        out.append(f"part of the model runs on the CPU (slower) to keep a usable context: "
                   f"{lay.ctx} per slot, the {env.intent.key} minimum is {env.floor}")
    elif v != VERDICT_NO and lay.mode in (_SPILL, _MOE) and env.intent.spill == SPILL_AVOID:
        out.append("does not fit usable VRAM; part of the model runs on the CPU (slower)")
    return tuple(out)


def plan(intent: Intent, info: ModelInfo, host: Host, base: Profile, headroom_pct: float = 10,
         calibration: Calibration | None = None) -> Plan:
    """Plan settings for an intent on a host.

    Sampling and server identity fields (host, port, api_key, alias) are never changed.
    Usable VRAM is per device total minus headroom; the calibration factor of the host's
    backend multiplies every estimated total.

    Args:
        intent: What the model is for.
        info: Model metadata.
        host: Host resource snapshot.
        base: Profile to start from (usually the default profile).
        headroom_pct: Percentage of each device kept free (min 512 MiB).
        calibration: Measured data correcting the estimates, or None.

    Returns:
        The plan; ``reasons`` lists exactly the keys that differ from `base`.
    """
    env = _make_env(intent, info, host, base, headroom_pct, calibration)
    lay = _choose(env)
    d = _Draft(base)
    _apply_fixed(d, env)
    _apply_ctx(d, env, lay)
    _apply_placement(d, env, lay)
    budget = _budget(info, d.profile, env.factor)
    v = verdict(budget, host, headroom_pct)
    return Plan(intent.key, d.profile, d.reasons, budget, v, lay.ctx, _warnings(env, lay, v))


def plan_all(info: ModelInfo, host: Host, base: Profile, headroom_pct: float = 10,
             calibration: Calibration | None = None) -> dict[str, Plan]:
    """Plan every intent that applies to the model.

    Args:
        info: Model metadata.
        host: Host resource snapshot.
        base: Profile to start from.
        headroom_pct: Percentage of each device kept free (min 512 MiB).
        calibration: Measured data correcting the estimates, or None.

    Returns:
        Plans by intent key, in intent table order.
    """
    return {i.key: plan(i, info, host, base, headroom_pct, calibration)
            for i in applicable(info)}
