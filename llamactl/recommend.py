"""Memory budget model and launch-setting recommendations."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from .config import Profile
from .gguf import ModelInfo
from .hostinfo import Host

MiB = 1 << 20
GB = 1e9
OVERHEAD = 512 * MiB          # fixed runtime overhead
COMPUTE_MIN = 256 * MiB       # minimum compute/scratch buffer
CTX_TIERS = (4096, 8192, 16384, 32768, 65536, 131072)

BYTES_PER_ELEM = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32,
                  "q5_0": 22 / 32, "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32}


@dataclass(frozen=True)
class Budget:
    """Memory a launch needs, split by purpose (bytes).

    Attributes:
        weights: Model weights (ModelInfo.size).
        kv: KV cache for the configured context and cache types.
        compute: Activation/scratch buffer heuristic.
        overhead: Fixed runtime overhead.
    """

    weights: int
    kv: int
    compute: int
    overhead: int

    def total(self) -> int:
        """All components summed."""
        return self.weights + self.kv + self.compute + self.overhead

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


def estimate_budget(info: ModelInfo, prof: Profile, host: Host) -> Budget:
    """Memory budget for a model under a profile on a host.

    ctx_size 0 means the model's trained context; ubatch_size 0 means 512.

    Args:
        info: Model metadata (placeholder allowed: kv comes out 0).
        prof: Profile to evaluate.
        host: Host snapshot (unused, kept for the impact callback signature).
    """
    ctx = prof.ctx_size or info.ctx_train
    # -c is the total KV pool: parallel slots each get ctx // np, not ctx each
    kv = kv_per_token(info, prof.cache_type_k, prof.cache_type_v) * ctx
    compute = max(COMPUTE_MIN, info.n_embd * (prof.ubatch_size or 512) * 4 * 8)
    return Budget(weights=info.size, kv=kv, compute=compute, overhead=OVERHEAD)


@dataclass(frozen=True)
class Recommendation:
    """A suggested profile plus why each field differs from the base.

    Attributes:
        profile: The recommended profile.
        reasons: key -> one-line reason, only for changed keys.
        budget: Memory budget of the recommended profile.
        verdict: "GPU", "GPU+CPU", "CPU" or "NO".
    """

    profile: Profile
    reasons: dict[str, str]
    budget: Budget
    verdict: str


def verdict(budget: Budget, host: Host, headroom_pct: float = 10) -> str:
    """Verdict label for a budget on a host: "GPU", "GPU+CPU", "CPU" or "NO".

    Args:
        budget: Memory budget of the launch.
        host: Host resource snapshot.
        headroom_pct: Fraction of total VRAM kept free (min 512 MiB).

    Returns:
        "GPU" when it fits VRAM with headroom, "GPU+CPU" when it needs
        RAM too, "CPU" on a GPU-less host, "NO" when nothing fits.
    """
    if not host.gpus:
        return "CPU" if budget.total() <= host.ram_free else "NO"
    headroom = max(int(headroom_pct / 100 * host.vram_total), 512 * MiB)
    if budget.total() <= host.vram_total - headroom:
        return "GPU"
    if budget.total() <= host.vram_total + host.ram_free:
        return "GPU+CPU"
    return "NO"


def _gb(n: float) -> str:
    return f"{n / GB:.1f} GB"


def _fits_ctx(info: ModelInfo, prof: Profile, host: Host, ctx: int, avail: int,
              ctk: str = "f16", ctv: str = "f16") -> bool:
    """True when weights on GPU plus ctx of KV stay under `avail` bytes."""
    p = prof.merged(ctx_size=ctx, cache_type_k=ctk, cache_type_v=ctv)
    return estimate_budget(info, p, host).total() <= avail


def _pick_ctx(info: ModelInfo, prof: Profile, host: Host, avail: int,
              kv_only: bool = False, ctk: str = "f16", ctv: str = "f16") -> int:
    """Largest ctx tier (<= trained ctx) that fits `avail` bytes; 0 when none.

    Args:
        kv_only: Count only KV + compute + overhead against `avail` (weights
            already accounted for elsewhere).
    """
    best = 0
    for ctx in CTX_TIERS:
        if info.ctx_train and ctx > info.ctx_train:
            break
        p = prof.merged(ctx_size=ctx, cache_type_k=ctk, cache_type_v=ctv)
        b = estimate_budget(info, p, host)
        if (b.total() - b.weights if kv_only else b.total()) <= avail:
            best = ctx
    return best


def recommend(info: ModelInfo, host: Host, base: Profile,
              headroom_pct: float = 10) -> Recommendation:
    """Recommend a profile for a model on a host.

    Picks threads, flash_attn, the largest ctx tier that fits, cache types and
    an offload strategy. Sampling and server fields are never changed.

    Args:
        info: Model metadata.
        host: Host resource snapshot.
        base: Profile to start from (usually the default profile).
        headroom_pct: Fraction of total VRAM kept free (min 512 MiB).

    Returns:
        Recommendation with changed keys explained in `reasons`.
    """
    prof = base.merged()
    reasons: dict[str, str] = {}

    def set_(key: str, val, reason: str) -> None:
        nonlocal prof
        if getattr(prof, key) != val:
            prof = prof.merged(**{key: val})
            reasons[key] = reason

    cores = host.cpu_cores or (os.cpu_count() or 2) // 2
    set_("threads", cores, f"host has {cores} physical cores")
    set_("flash_attn", "on" if host.gpus else "auto", "GPU present" if host.gpus else "no GPU")

    headroom = max(int(headroom_pct / 100 * host.vram_total), 512 * MiB)
    vram_avail = host.vram_total - headroom
    verdict = "NO"

    if not host.gpus:
        set_("gpu_layers", "0", "no GPU detected")
        ctx = _pick_ctx(info, prof, host, host.ram_free - info.size, kv_only=True)
        set_("ctx_size", ctx, f"{ctx} ctx -> {_gb(kv_per_token(info, 'f16', 'f16') * ctx)} KV fits free RAM")
        verdict = "CPU"
    elif info.size <= vram_avail:
        ctx = _pick_ctx(info, prof, host, vram_avail)
        if ctx:
            kvt = kv_per_token(info, "f16", "f16")
            set_("ctx_size", ctx,
                 f"{ctx} ctx -> {_gb(kvt * ctx)} KV (f16); total {_gb(info.size + kvt * ctx + COMPUTE_MIN + OVERHEAD)} "
                 f"fits {_gb(host.vram_total)} VRAM with {_gb(headroom)} headroom")
            nxt = next((t for t in CTX_TIERS if t > ctx and (not info.ctx_train or t <= info.ctx_train)), 0)
            if nxt and prof.flash_attn == "on" and _fits_ctx(info, prof, host, nxt, vram_avail, "q8_0", "q8_0"):
                set_("cache_type_k", "q8_0", "q8_0 KV reaches the next ctx tier with minor quality loss")
                set_("cache_type_v", "q8_0", "quantised V requires flash_attn on")
                set_("ctx_size", nxt,
                     f"{nxt} ctx -> {_gb(kv_per_token(info, 'q8_0', 'q8_0') * nxt)} KV (q8_0) fits VRAM")
            verdict = "GPU"
        # fall through to offload when not even 4096 fits
        else:
            verdict = _offload(info, host, base, prof, reasons, set_, vram_avail)
    else:
        verdict = _offload(info, host, base, prof, reasons, set_, vram_avail)

    return Recommendation(prof, reasons, estimate_budget(info, prof, host), verdict)


def _offload(info: ModelInfo, host: Host, base: Profile, prof: Profile,
             reasons: dict, set_, vram_avail: int) -> str:
    """Partial-offload path when weights alone do not fit VRAM minus headroom.

    Returns:
        The verdict string: "GPU+CPU" or "NO".
    """
    verdict = "NO" if info.size > vram_avail + host.ram_free else "GPU+CPU"
    if info.expert_count > 0:
        set_("gpu_layers", "all", "all layers on GPU; MoE experts kept in RAM")
        n = max(1, min(info.n_layer, math.ceil(info.n_layer * (1 - vram_avail / info.size))))
        set_("n_cpu_moe", n,
             f"experts of {n}/{info.n_layer} layers stay in RAM: weights {_gb(info.size)} > "
             f"{_gb(vram_avail)} usable VRAM")
        ctx = _pick_ctx(info, prof, host, vram_avail, kv_only=True)
    else:
        gl = max(0, int(info.n_layer * vram_avail / info.size))
        set_("gpu_layers", str(gl),
             f"{gl}/{info.n_layer} layers fit {_gb(vram_avail)} usable VRAM; rest on CPU")
        ctx = _pick_ctx(info, prof, host, vram_avail - int(info.size * gl / info.n_layer), kv_only=True)
    if ctx:
        set_("ctx_size", ctx, f"{ctx} ctx fits the memory left after partial offload")
    return verdict
