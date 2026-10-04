"""Recommender fixtures and assertions per spec 7.3."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from llamactl.config import Profile
from llamactl.gguf import ModelInfo
from llamactl.hostinfo import Gpu, Host
from llamactl.recommend import kv_per_token, recommend


def _info(size: float, n_layer: int, n_head_kv: int, ctx: int, experts: int = 0) -> ModelInfo:
    return ModelInfo(path=Path("m.gguf"), size=int(size), architecture="x", name="m",
                     n_layer=n_layer, n_embd=4096, n_head=32, n_head_kv=n_head_kv,
                     head_dim_k=128, head_dim_v=128, ctx_train=ctx, expert_count=experts,
                     file_type=0, sharded=1)


@pytest.fixture
def dense8b() -> ModelInfo:
    return _info(8.5e9, 32, 8, 131072)


@pytest.fixture
def moe35b() -> ModelInfo:
    return _info(20e9, 48, 4, 262144, 128)


@pytest.fixture
def small1b() -> ModelInfo:
    return _info(1.5e9, 16, 8, 131072)


@pytest.fixture
def gpu_host() -> Host:
    return Host(gpus=[Gpu("G", int(16e9), int(16e9))], ram_total=int(24e9), ram_free=int(24e9),
                cpu_threads=16, cpu_cores=8)


@pytest.fixture
def cpu_host() -> Host:
    return Host(ram_total=int(32e9), ram_free=int(32e9), cpu_threads=16, cpu_cores=8)


def _changed(rec) -> set[str]:
    return {f.name for f in dataclasses.fields(Profile)
            if getattr(rec.profile, f.name) != getattr(Profile(), f.name)}


def test_kv_per_token(dense8b: ModelInfo) -> None:
    assert kv_per_token(dense8b, "f16", "f16") == 32 * 8 * (128 * 2 + 128 * 2)
    assert kv_per_token(dense8b, "q8_0", "q8_0") == int(32 * 8 * (128 * 34 / 32 * 2))


def test_dense_on_gpu(dense8b: ModelInfo, gpu_host: Host) -> None:
    rec = recommend(dense8b, gpu_host, Profile())
    assert rec.verdict == "GPU"
    assert rec.profile.flash_attn == "on"
    assert rec.profile.threads == 8
    # 32768 fits f16, 65536 needs q8_0 -> both cache types recommended
    assert rec.profile.ctx_size == 65536
    assert rec.profile.cache_type_k == rec.profile.cache_type_v == "q8_0"
    assert set(rec.reasons) == _changed(rec)


def test_moe_offload(moe35b: ModelInfo, gpu_host: Host) -> None:
    rec = recommend(moe35b, gpu_host, Profile())
    assert rec.verdict == "GPU+CPU"
    assert rec.profile.gpu_layers == "all"
    assert rec.profile.n_cpu_moe == 14  # ceil(48 * (1 - 14.4/20))
    assert set(rec.reasons) == _changed(rec)


def test_no_gpu(small1b: ModelInfo, cpu_host: Host) -> None:
    rec = recommend(small1b, cpu_host, Profile())
    assert rec.verdict == "CPU"
    assert rec.profile.gpu_layers == "0"
    assert rec.profile.flash_attn == "auto"
    assert rec.profile.ctx_size == 131072
    assert set(rec.reasons) == _changed(rec)


def test_reasons_only_changed(dense8b: ModelInfo, gpu_host: Host) -> None:
    rec = recommend(dense8b, gpu_host, Profile(temp=0.5))
    assert "temp" not in rec.reasons and rec.profile.temp == 0.5
    assert "host" not in rec.reasons and "seed" not in rec.reasons
