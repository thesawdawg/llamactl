"""Planner: budget maths, per-intent plans on synthetic models and hosts, calibration."""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from llamactl import intents
from llamactl.calibration import Calibration, Trial
from llamactl.config import Profile
from llamactl.gguf import ModelInfo
from llamactl.hardware import Device, Host
from llamactl.planner import (Budget, Plan, estimate_budget, gpu_limit, kv_per_token, plan,
                              plan_all, spill_limit, verdict)

GB = 1e9
SPILL_WARNING = "does not fit usable VRAM; part of the model runs on the CPU (slower)"


def _info(size: float, n_layer: int, n_head_kv: int, ctx: int, experts: int = 0,
          pooling: int = -1, n_embd: int = 4096, head_dim: int = 128) -> ModelInfo:
    return ModelInfo(path=Path("m.gguf"), size=int(size), architecture="x", name="m",
                     n_layer=n_layer, n_embd=n_embd, n_head=32, n_head_kv=n_head_kv,
                     head_dim_k=head_dim, head_dim_v=head_dim, ctx_train=ctx,
                     expert_count=experts, file_type=0, sharded=1, parameter_count=0,
                     pooling_type=pooling)


def _gpu_host(*vram_gb: float, ram_gb: float = 24, backend_id: str = "CUDA",
              unified: bool = False) -> Host:
    gpus = [Device.of(f"{backend_id}{n}", f"G{n}", int(v * GB), int(v * GB))
            for n, v in enumerate(vram_gb)]
    return Host(gpus=gpus, ram_total=int(ram_gb * GB), ram_free=int(ram_gb * GB),
                cpu_threads=16, cpu_cores=8, unified=unified)


@pytest.fixture
def dense8b() -> ModelInfo:
    return _info(8.5e9, 32, 8, 131072)


@pytest.fixture
def dense27b() -> ModelInfo:
    return _info(12e9, 64, 8, 131072, n_embd=5120)


@pytest.fixture
def moe35b() -> ModelInfo:
    return _info(20e9, 48, 4, 262144, experts=128)


@pytest.fixture
def small1b() -> ModelInfo:
    return _info(1.5e9, 16, 8, 131072)


@pytest.fixture
def embed_model() -> ModelInfo:
    return _info(0.3e9, 12, 12, 512, pooling=1, n_embd=768, head_dim=64)


@pytest.fixture
def gpu_host() -> Host:
    return _gpu_host(16)


@pytest.fixture
def cpu_host() -> Host:
    return Host(ram_total=int(32 * GB), ram_free=int(32 * GB), cpu_threads=16, cpu_cores=8)


@pytest.fixture
def two_gpu_host() -> Host:
    return _gpu_host(24, 12, ram_gb=32)


@pytest.fixture
def unified_host() -> Host:
    h = _gpu_host(32, ram_gb=32, backend_id="Metal", unified=True)
    h.ram_free = int(20 * GB)
    return h


@pytest.fixture
def tiny_host() -> Host:
    return _gpu_host(6, ram_gb=16)


def _changed(p: Plan, base: Profile | None = None) -> set[str]:
    base = base or Profile()
    return {f.name for f in dataclasses.fields(Profile)
            if getattr(p.profile, f.name) != getattr(base, f.name)}


def _by(intent: str, info: ModelInfo, host: Host, base: Profile | None = None, **kw) -> Plan:
    return plan(intents.by_key[intent], info, host, base or Profile(), **kw)


# --------------------------------------------------------------------------- budget maths

def test_kv_per_token(dense8b: ModelInfo) -> None:
    assert kv_per_token(dense8b, "f16", "f16") == 32 * 8 * (128 * 2 + 128 * 2)
    assert kv_per_token(dense8b, "q8_0", "q8_0") == int(32 * 8 * (128 * 34 / 32 * 2))


def test_verdict_honours_headroom(gpu_host: Host) -> None:
    """A budget inside 10% headroom is GPU at 0% but GPU+CPU at 10%."""
    b = Budget(weights=int(15e9), kv=0, compute=0, overhead=0)
    assert verdict(b, gpu_host, headroom_pct=0) == "GPU"
    assert verdict(b, gpu_host, headroom_pct=10) == "GPU+CPU"
    assert verdict(Budget(int(50e9), 0, 0, 0), gpu_host) == "NO"


def test_verdict_without_gpu(cpu_host: Host) -> None:
    assert verdict(Budget(int(8e9), 0, 0, 0), cpu_host) == "CPU"
    assert verdict(Budget(int(40e9), 0, 0, 0), cpu_host) == "NO"


def test_headroom_is_per_device(two_gpu_host: Host) -> None:
    # 10% of 24 GB and 10% of 12 GB, not 10% of the sum (same here) and never under 512 MiB
    assert gpu_limit(two_gpu_host, 10) == int(21.6e9) + int(10.8e9)
    mib512 = 512 * (1 << 20)      # 10% of these small cards is under the 512 MiB floor
    assert gpu_limit(_gpu_host(2, 1), 10) == (int(2e9) - mib512) + (int(1e9) - mib512)
    assert gpu_limit(_gpu_host(2), 0) == int(2e9) - mib512
    assert spill_limit(two_gpu_host, 10) == gpu_limit(two_gpu_host, 10) + two_gpu_host.ram_free


def test_unified_is_one_pool(unified_host: Host) -> None:
    assert spill_limit(unified_host) == gpu_limit(unified_host) == int(28.8e9)
    b = Budget(int(30e9), 0, 0, 0)
    assert verdict(b, unified_host) == "NO"      # VRAM + RAM would have said GPU+CPU
    assert verdict(Budget(int(28e9), 0, 0, 0), unified_host) == "GPU"


def test_estimate_budget_scales_total_only(dense8b: ModelInfo, gpu_host: Host) -> None:
    cal = Calibration([_trial(1.25, "cuda")])
    raw = estimate_budget(dense8b, Profile(ctx_size=8192), gpu_host)
    cald = estimate_budget(dense8b, Profile(ctx_size=8192), gpu_host, cal)
    assert (cald.weights, cald.kv) == (raw.weights, raw.kv)
    assert cald.raw_total() == raw.total() and cald.total() == round(raw.total() * 1.25)
    # a different backend, or no GPU, leaves the estimate alone
    assert estimate_budget(dense8b, Profile(), _gpu_host(16, backend_id="Vulkan"), cal).scale == 1.0
    assert estimate_budget(dense8b, Profile(), Host(), cal).scale == 1.0


# --------------------------------------------------------------------------- 16 GB GPU grid

# (model, intent) -> (ctx/slot, total -c, slots, kv, gpu_layers, fit, fit_target, n_cpu_moe,
#                     verdict, number of warnings), on 16 GB VRAM / 24 GB RAM.
GRID = {
    ("1b", "chat"): (16384, 16384, 1, "f16", "all", "off", 0, 0, "GPU", 0),
    ("1b", "agent"): (65536, 65536, 1, "q8_0", "all", "off", 0, 0, "GPU", 0),
    ("1b", "context"): (131072, 131072, 1, "q8_0", "all", "off", 0, 0, "GPU", 0),
    ("1b", "speed"): (8192, 8192, 1, "f16", "all", "off", 0, 0, "GPU", 0),
    ("1b", "api"): (8192, 32768, 4, "f16", "all", "off", 0, 0, "GPU", 0),
    ("8b", "chat"): (16384, 16384, 1, "f16", "all", "off", 0, 0, "GPU", 0),
    ("8b", "agent"): (65536, 65536, 1, "q8_0", "all", "off", 0, 0, "GPU", 0),
    ("8b", "context"): (131072, 131072, 1, "q8_0", "auto", "on", 1525, 0, "GPU+CPU", 0),
    ("8b", "speed"): (8192, 8192, 1, "f16", "all", "off", 0, 0, "GPU", 0),
    ("8b", "api"): (8192, 32768, 4, "f16", "all", "off", 0, 0, "GPU", 0),
    ("27b", "chat"): (8192, 8192, 1, "q8_0", "all", "off", 0, 0, "GPU", 0),
    ("27b", "agent"): (65536, 65536, 1, "q8_0", "auto", "on", 1525, 0, "GPU+CPU", 0),
    ("27b", "context"): (131072, 131072, 1, "q8_0", "auto", "on", 1525, 0, "GPU+CPU", 0),
    ("27b", "speed"): (4096, 4096, 1, "f16", "all", "off", 0, 0, "GPU", 0),
    ("27b", "api"): (4096, 16384, 4, "f16", "auto", "on", 1525, 0, "GPU+CPU", 1),
    ("moe", "chat"): (8192, 8192, 1, "f16", "all", "off", 0, 18, "GPU+CPU", 1),
    ("moe", "agent"): (65536, 65536, 1, "q8_0", "all", "off", 0, 24, "GPU+CPU", 0),
    ("moe", "context"): (131072, 131072, 1, "q8_0", "all", "off", 0, 32, "GPU+CPU", 0),
    ("moe", "speed"): (4096, 4096, 1, "f16", "all", "off", 0, 17, "GPU+CPU", 1),
    ("moe", "api"): (4096, 16384, 4, "f16", "all", "off", 0, 20, "GPU+CPU", 1),
    ("embed", "embed"): (512, 2048, 4, "f16", "all", "off", 0, 0, "GPU", 0),
}


@pytest.fixture
def models(small1b, dense8b, dense27b, moe35b, embed_model) -> dict[str, ModelInfo]:
    return {"1b": small1b, "8b": dense8b, "27b": dense27b, "moe": moe35b, "embed": embed_model}


@pytest.mark.parametrize("model,intent", list(GRID))
def test_grid_on_16gb_gpu(models, gpu_host: Host, model: str, intent: str) -> None:
    p = _by(intent, models[model], gpu_host)
    ctx, total, slots, kv, ngl, fit, fitt, moe, v, nwarn = GRID[(model, intent)]
    pr = p.profile
    assert (p.ctx_per_slot, pr.ctx_size, pr.parallel) == (ctx, total, slots)
    assert (pr.cache_type_k, pr.cache_type_v) == (kv, kv)
    assert (pr.gpu_layers, pr.fit, pr.fit_target, pr.n_cpu_moe) == (ngl, fit, fitt, moe)
    assert (p.verdict, len(p.warnings), p.intent) == (v, nwarn, intent)
    assert (pr.threads, pr.flash_attn, pr.tensor_split) == (8, "on", "")
    assert set(p.reasons) == _changed(p)
    assert all(r and "\n" not in r for r in p.reasons.values())


def test_dense8b_agent_numbers(dense8b: ModelInfo, gpu_host: Host) -> None:
    p = _by("agent", dense8b, gpu_host)
    assert p.profile.cache_reuse == 256 and "cache_reuse" in p.reasons
    # 8.5 GB weights + 65536 x 69632 B q8_0 KV + 268 MB compute + 537 MB overhead
    assert p.budget.total() == 8_500_000_000 + 65536 * 69632 + 268_435_456 + 536_870_912
    assert p.budget.total() <= int(14.4e9)
    assert "65536 ctx x 1 slot = 65536 -c" in p.reasons["ctx_size"]
    assert "usable VRAM" in p.reasons["ctx_size"]


def test_intent_extras_and_slots(dense8b: ModelInfo, gpu_host: Host, embed_model: ModelInfo) -> None:
    assert _by("api", dense8b, gpu_host).profile.metrics is True
    assert _by("embed", embed_model, gpu_host).profile.embedding is True
    p = _by("api", dense8b, gpu_host)
    assert p.profile.parallel == 4 and p.profile.ctx_size == p.ctx_per_slot * 4
    assert p.reasons["parallel"].startswith("4 slots")


def test_trained_ctx_is_reachable_exactly(gpu_host: Host) -> None:
    info = _info(4e9, 28, 4, 40960)      # not a power of two
    assert _by("context", info, gpu_host).ctx_per_slot == 40960
    assert _by("agent", info, gpu_host).ctx_per_slot == 40960      # target 64k capped by ctx_train
    # a short-context model caps every tier
    assert _by("chat", _info(4e9, 28, 4, 8192), gpu_host).ctx_per_slot == 8192


def test_embedding_with_tiny_trained_ctx(embed_model: ModelInfo, gpu_host: Host) -> None:
    p = _by("embed", embed_model, gpu_host)
    assert p.ctx_per_slot == 512 and p.profile.ctx_size == 2048
    assert p.warnings == ()
    assert set(plan_all(embed_model, gpu_host, Profile())) == {"embed"}


def test_plan_all_covers_applicable_intents(dense8b: ModelInfo, gpu_host: Host) -> None:
    plans = plan_all(dense8b, gpu_host, Profile())
    assert list(plans) == ["chat", "agent", "context", "speed", "api"]
    assert all(p.intent == k for k, p in plans.items())


# --------------------------------------------------------------------------- spill and warnings

def test_allow_spill_uses_fit(dense27b: ModelInfo, gpu_host: Host) -> None:
    p = _by("agent", dense27b, gpu_host)
    assert (p.profile.gpu_layers, p.profile.fit, p.profile.fit_target) == ("auto", "on", 1525)
    assert p.verdict == "GPU+CPU" and p.warnings == ()
    assert "llama.cpp places" in p.reasons["gpu_layers"]
    assert p.budget.total() > gpu_limit(gpu_host) and p.budget.total() <= spill_limit(gpu_host)


def test_moe_spill_uses_n_cpu_moe(moe35b: ModelInfo, gpu_host: Host) -> None:
    p = _by("agent", moe35b, gpu_host)
    assert p.profile.gpu_layers == "all" and p.profile.fit == "off"
    assert p.profile.n_cpu_moe == 24 and p.verdict == "GPU+CPU"
    # the KV cache and buffers stay on the GPU
    assert p.budget.total() - moe35b.size <= gpu_limit(gpu_host)
    # the experts that stay in RAM free enough VRAM
    per_layer = moe35b.size / moe35b.n_layer
    assert p.budget.total() - p.profile.n_cpu_moe * per_layer <= gpu_limit(gpu_host)
    assert p.budget.total() - (p.profile.n_cpu_moe - 1) * per_layer > gpu_limit(gpu_host)


FLOOR_WARNING = ("part of the model runs on the CPU (slower) to keep a usable context: "
                 "{ctx} per slot, the {intent} minimum is {floor}")


def test_avoid_spills_to_the_minimum_when_the_gpu_tier_is_too_small(dense27b: ModelInfo) -> None:
    """12 GB of weights on a 15 GB card leave only 4096 ctx on the GPU: spill for the 8192 minimum."""
    p = _by("chat", dense27b, _gpu_host(15))
    assert p.verdict == "GPU+CPU" and p.profile.gpu_layers == "auto" and p.profile.fit == "on"
    assert p.ctx_per_slot == 8192 and p.profile.ctx_size == 8192   # the minimum, not the 16384 target
    assert p.warnings == (FLOOR_WARNING.format(ctx=8192, intent="chat", floor=8192),)
    assert p.budget.total() <= spill_limit(_gpu_host(15))
    assert set(p.reasons) == _changed(p)


def test_avoid_keeps_gpu_tier_below_min_when_spilling_cannot_reach_it(dense27b: ModelInfo) -> None:
    """With almost no free RAM, spilling gains nothing: old behaviour, below-minimum warning."""
    p = _by("chat", dense27b, _gpu_host(15, ram_gb=0.1))
    assert p.verdict == "GPU" and p.profile.gpu_layers == "all"
    assert (p.ctx_per_slot, p.profile.cache_type_k) == (4096, "q8_0")
    assert p.warnings == ("ctx 4096 per slot is below the chat minimum of 8192",)


def test_avoid_on_unified_memory_never_spills_to_the_minimum(dense27b: ModelInfo) -> None:
    """One memory pool: there is nothing to spill into."""
    p = _by("chat", dense27b, _gpu_host(15, unified=True))
    assert p.verdict == "GPU" and p.ctx_per_slot == 4096
    assert p.warnings == ("ctx 4096 per slot is below the chat minimum of 8192",)


def test_avoid_at_or_above_minimum_stays_on_gpu(dense27b: ModelInfo) -> None:
    """16 GB reaches the 8192 minimum on the GPU alone, so nothing spills."""
    p = _by("chat", dense27b, _gpu_host(16))
    assert p.verdict == "GPU" and p.ctx_per_slot == 8192 and p.warnings == ()


def test_avoid_spill_to_minimum_on_moe_moves_experts() -> None:
    """A MoE reaches the minimum by keeping experts in RAM, not by fit."""
    p = _by("chat", _info(12e9, 48, 8, 131072, experts=64), _gpu_host(15))
    assert p.verdict == "GPU+CPU" and p.ctx_per_slot == 8192
    assert p.profile.n_cpu_moe > 0 and p.profile.gpu_layers == "all"
    assert p.warnings == (FLOOR_WARNING.format(ctx=8192, intent="chat", floor=8192),)


def test_avoid_with_nothing_fitting_the_gpu_spills(dense27b: ModelInfo) -> None:
    """On 14 GB the weights alone leave no room for any KV: spill, but only to the minimum."""
    p = _by("chat", dense27b, _gpu_host(14))
    assert p.verdict == "GPU+CPU" and p.profile.gpu_layers == "auto"
    assert p.ctx_per_slot == 8192      # the chat minimum, not its 16384 target
    assert p.warnings == (FLOOR_WARNING.format(ctx=8192, intent="chat", floor=8192),)


def test_avoid_spills_when_slots_do_not_fit(dense27b: ModelInfo, gpu_host: Host) -> None:
    """Weights fit 16 GB but four slots of KV do not."""
    p = _by("api", dense27b, gpu_host)
    assert p.verdict == "GPU+CPU" and p.ctx_per_slot == 4096      # the api minimum
    assert p.warnings == (FLOOR_WARNING.format(ctx=4096, intent="api", floor=4096),)


def test_weights_exceed_vram_falls_back_to_spill(dense8b: ModelInfo, tiny_host: Host) -> None:
    p = _by("speed", dense8b, tiny_host)
    assert p.verdict == "GPU+CPU" and p.profile.gpu_layers == "auto"
    assert p.profile.fit_target == 572
    assert p.ctx_per_slot == 4096      # the speed minimum, not its 8192 target
    assert p.warnings == (FLOOR_WARNING.format(ctx=4096, intent="speed", floor=4096),)


def test_avoid_spill_short_of_the_minimum_keeps_both_warnings(dense8b: ModelInfo) -> None:
    """Spilling can only reach 4096 here, below the 8192 minimum: say so, and that it spills."""
    p = _by("chat", dense8b, _gpu_host(6, ram_gb=4.2))
    assert p.verdict == "GPU+CPU" and p.ctx_per_slot == 4096
    assert p.warnings == ("ctx 4096 per slot is below the chat minimum of 8192", SPILL_WARNING)


def test_verdict_no_has_best_effort_values(moe35b: ModelInfo, tiny_host: Host) -> None:
    p = _by("api", moe35b, tiny_host)
    assert p.verdict == "NO"
    assert p.ctx_per_slot == 4096 and p.profile.cache_type_k == "q8_0"
    assert p.warnings[0].startswith("does not fit memory")
    assert p.profile.parallel == 4 and p.profile.threads == 8
    nokv = _by("speed", _info(80e9, 80, 8, 131072), tiny_host)
    assert nokv.verdict == "NO" and nokv.profile.cache_type_k == "f16"


# --------------------------------------------------------------------------- no GPU

def test_no_gpu(small1b: ModelInfo, cpu_host: Host) -> None:
    p = _by("chat", small1b, cpu_host)
    assert p.verdict == "CPU" and p.profile.gpu_layers == "0" and p.profile.flash_attn == "auto"
    assert p.ctx_per_slot == 16384 and p.profile.cache_type_k == "f16"
    assert set(p.reasons) == _changed(p) and p.warnings == ()


def test_no_gpu_quantised_kv_needs_flash_attn(dense8b: ModelInfo, cpu_host: Host) -> None:
    p = _by("agent", dense8b, cpu_host)
    assert p.profile.cache_type_k == "q8_0" and p.profile.flash_attn == "on"
    assert p.budget.total() <= cpu_host.ram_free


def test_no_gpu_model_too_big(cpu_host: Host) -> None:
    p = _by("chat", _info(40e9, 64, 8, 131072), cpu_host)
    assert p.verdict == "NO" and p.profile.gpu_layers == "0" and p.ctx_per_slot == 4096
    assert p.warnings and p.warnings[0].startswith("does not fit")


# --------------------------------------------------------------------------- multi-GPU

def test_tensor_split_proportional_to_usable(dense27b: ModelInfo, two_gpu_host: Host) -> None:
    p = _by("chat", dense27b, two_gpu_host)
    assert p.profile.tensor_split == "21.6,10.8" and p.verdict == "GPU"
    ratio = [float(x) for x in p.profile.tensor_split.split(",")]
    assert ratio[0] / ratio[1] == pytest.approx(2.0)
    assert "tensor_split" in p.reasons


def test_tensor_split_only_with_several_devices(dense8b: ModelInfo, gpu_host: Host,
                                                cpu_host: Host) -> None:
    assert _by("chat", dense8b, gpu_host).profile.tensor_split == ""
    assert _by("chat", dense8b, cpu_host).profile.tensor_split == ""


def test_two_gpus_pool_memory(moe35b: ModelInfo, two_gpu_host: Host) -> None:
    """20 GB of weights do not fit one 16 GB card but fit 24 + 12 GB."""
    assert _by("agent", moe35b, two_gpu_host).verdict == "GPU"
    assert _by("agent", moe35b, _gpu_host(16)).verdict == "GPU+CPU"
    assert _by("context", moe35b, two_gpu_host).profile.n_cpu_moe == 6


# --------------------------------------------------------------------------- unified memory

def test_unified_does_not_double_count(dense27b: ModelInfo, unified_host: Host) -> None:
    p = _by("context", dense27b, unified_host)
    assert p.ctx_per_slot == 65536 and p.verdict == "GPU" and p.profile.fit == "off"
    assert p.budget.total() <= int(28.8e9)
    # a 40 GB model would "fit" VRAM + RAM (32 + 20) but not one 32 GB pool
    big = _by("chat", _info(40e9, 80, 8, 131072), unified_host)
    assert big.verdict == "NO"
    assert _by("chat", _info(40e9, 80, 8, 131072), _gpu_host(32, ram_gb=32)).verdict == "GPU+CPU"


def test_unified_never_spills(moe35b: ModelInfo, unified_host: Host) -> None:
    for p in plan_all(moe35b, unified_host, Profile()).values():
        assert p.profile.n_cpu_moe == 0 and p.profile.fit == "off"


# --------------------------------------------------------------------------- reasons and base

def test_reasons_only_changed(dense8b: ModelInfo, gpu_host: Host) -> None:
    p = _by("chat", dense8b, gpu_host, Profile(temp=0.5))
    assert p.profile.temp == 0.5
    assert not {"temp", "host", "port", "api_key", "alias", "seed"} & set(p.reasons)
    assert set(p.reasons) == _changed(p, Profile(temp=0.5))


def test_server_identity_and_sampling_untouched(dense8b: ModelInfo, gpu_host: Host) -> None:
    base = Profile(host="0.0.0.0", port=9000, api_key="k", alias="mine", temp=0.2, top_k=10,
                   top_p=0.8, min_p=0.1, repeat_penalty=1.1, seed=7)
    for i in intents.applicable(dense8b):
        p = plan(i, dense8b, gpu_host, base)
        for k in ("host", "port", "api_key", "alias", "temp", "top_k", "top_p", "min_p",
                  "repeat_penalty", "seed"):
            assert getattr(p.profile, k) == getattr(base, k)


def test_reasons_match_diff_for_every_case(models) -> None:
    bases = [Profile(), Profile(ctx_size=1234, cache_type_k="q8_0", cache_type_v="q8_0",
                                gpu_layers="auto", fit="off", n_cpu_moe=5, threads=8,
                                parallel=1, tensor_split="1,1")]
    hosts = [_gpu_host(16), _gpu_host(24, 12, ram_gb=32), _gpu_host(6, ram_gb=16),
             Host(ram_total=int(32 * GB), ram_free=int(32 * GB), cpu_threads=16, cpu_cores=8)]
    for base in bases:
        for host in hosts:
            for info in models.values():
                for p in plan_all(info, host, base).values():
                    assert set(p.reasons) == _changed(p, base), (p.intent, host.summary())


def test_base_ctx_and_cache_are_replaced(dense8b: ModelInfo, gpu_host: Host) -> None:
    base = Profile(ctx_size=999, cache_type_k="q4_0", cache_type_v="q4_0")
    p = _by("chat", dense8b, gpu_host, base)
    assert p.profile.ctx_size == 16384 and p.profile.cache_type_k == "f16"
    assert {"ctx_size", "cache_type_k", "cache_type_v"} <= set(p.reasons)


def test_threads_left_alone_when_cores_unknown(dense8b: ModelInfo) -> None:
    host = _gpu_host(16)
    host.cpu_cores = host.cpu_threads = 0
    assert _by("chat", dense8b, host).profile.threads == 0


def test_plan_budget_matches_estimate(models, gpu_host: Host) -> None:
    cal = Calibration([_trial(1.2, "cuda")])
    for info in models.values():
        for p in plan_all(info, gpu_host, Profile(), calibration=cal).values():
            assert p.budget == estimate_budget(info, p.profile, gpu_host, cal)


# --------------------------------------------------------------------------- sanity properties

VRAM_STEPS = (4, 6, 8, 12, 16, 24, 48, 96)


def test_more_vram_never_shrinks_ctx(models) -> None:
    """For every intent, spill-"avoid" included, more VRAM never gives a smaller context."""
    steps = [n / 2 for n in range(4, 100)]      # 2 .. 49.5 GB in half-GB steps
    for ram in (8, 24, 64):
        for info in models.values():
            for i in intents.applicable(info):
                ctx = [_by(i.key, info, _gpu_host(v, ram_gb=ram)).ctx_per_slot for v in steps]
                assert ctx == sorted(ctx), (ram, info.size, i.key)


def test_context_reaches_the_minimum_whenever_memory_allows(models) -> None:
    """With plenty of RAM no fitting plan is left below its intent's minimum context."""
    for vram in VRAM_STEPS:
        host = _gpu_host(vram, ram_gb=64)
        for info in models.values():
            for i in intents.applicable(info):
                p = _by(i.key, info, host)
                floor = min(i.ctx_min or info.ctx_train, info.ctx_train)
                if p.verdict != "NO":
                    assert p.ctx_per_slot >= floor, (vram, info.size, i.key, p.ctx_per_slot)


def test_more_ram_never_shrinks_ctx(dense27b: ModelInfo) -> None:
    for i in intents.applicable(dense27b):
        ctx = [_by(i.key, dense27b, _gpu_host(8, ram_gb=r)).ctx_per_slot for r in (8, 16, 32, 64)]
        assert ctx == sorted(ctx), i.key


def test_speed_never_quantises_kv(models, gpu_host: Host, tiny_host: Host,
                                  two_gpu_host: Host, cpu_host: Host) -> None:
    for host in (gpu_host, tiny_host, two_gpu_host, cpu_host):
        for info in models.values():
            if "speed" in plan_all(info, host, Profile()):
                p = _by("speed", info, host)
                assert (p.profile.cache_type_k, p.profile.cache_type_v) == ("f16", "f16")


def test_kv_k_and_v_always_match_and_never_below_q8(models, gpu_host: Host, tiny_host: Host) -> None:
    for host in (gpu_host, tiny_host):
        for info in models.values():
            for p in plan_all(info, host, Profile()).values():
                assert p.profile.cache_type_k == p.profile.cache_type_v in ("f16", "q8_0")


def test_agent_total_fits_usable_vram_when_gpu(models) -> None:
    for vram in VRAM_STEPS:
        host = _gpu_host(vram, ram_gb=64)
        for info in models.values():
            if "agent" not in plan_all(info, host, Profile()):
                continue
            p = _by("agent", info, host)
            if p.verdict == "GPU":
                assert p.budget.total() <= gpu_limit(host), (vram, info.size)


def test_every_gpu_verdict_fits_and_ctx_within_trained(models) -> None:
    for vram in VRAM_STEPS:
        host = _gpu_host(vram, ram_gb=32)
        for info in models.values():
            for p in plan_all(info, host, Profile()).values():
                assert p.ctx_per_slot <= max(info.ctx_train, 4096)
                if p.verdict == "GPU":
                    assert p.budget.total() <= gpu_limit(host)
                if p.verdict == "GPU+CPU":
                    assert p.budget.total() <= spill_limit(host)


# --------------------------------------------------------------------------- calibration

def _trial(ratio: float, backend: str) -> Trial:
    return Trial("/m.gguf", 1.0, "chat", {}, 1000, "GPU", backend,
                 {"ok": True, "vram_used": 1000 * ratio})


def _borderline_host(info: ModelInfo) -> Host:
    """A GPU whose usable memory is 12% above the raw 8192-ctx f16 estimate."""
    raw8k = estimate_budget(info, Profile(ctx_size=8192), _gpu_host(16)).total()
    return _gpu_host(raw8k * 1.12 / 0.9 / GB)


def test_calibration_can_push_a_borderline_plan_down_a_tier(dense8b: ModelInfo) -> None:
    host = _borderline_host(dense8b)
    assert _by("speed", dense8b, host).ctx_per_slot == 8192
    p = _by("speed", dense8b, host, calibration=Calibration([_trial(1.14, "cuda")]))
    assert p.ctx_per_slot == 4096 and p.verdict == "GPU" and p.warnings == ()
    assert "x1.14 calibration" in p.reasons["ctx_size"]
    # a factor below 1 frees memory, but the intent's target still caps the tier
    low = Calibration([_trial(0.8, "cuda")])
    assert _by("speed", dense8b, host, calibration=low).ctx_per_slot == 8192


def test_calibration_ignores_other_backends(dense8b: ModelInfo) -> None:
    host = _borderline_host(dense8b)
    other = Calibration([_trial(1.5, "vulkan")])
    assert _by("speed", dense8b, host, calibration=other).ctx_per_slot == 8192
    assert _by("speed", dense8b, host, calibration=Calibration()).ctx_per_slot == 8192


def test_calibration_changes_verdict_and_budget(dense8b: ModelInfo, gpu_host: Host) -> None:
    cal = Calibration([_trial(1.5, "cuda")])
    plain = _by("chat", dense8b, gpu_host)
    scaled = _by("chat", dense8b, gpu_host, calibration=cal)
    assert scaled.budget.total() == round(scaled.budget.raw_total() * 1.5)
    assert scaled.budget.total() > plain.budget.total()
    assert math.isclose(scaled.budget.scale, 1.5)
