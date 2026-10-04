"""GGUF header parsing on synthetic blobs."""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

from llamactl.gguf import GGUFError, ModelInfo, read_model_info

_U32, _F32, _STR, _ARR, _I32 = 4, 6, 8, 9, 5


def _kv_str(key: str, val: str) -> bytes:
    k, v = key.encode(), val.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", _STR) + struct.pack("<Q", len(v)) + v


def _kv_u32(key: str, val: int) -> bytes:
    k = key.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", _U32) + struct.pack("<I", val)


def _kv_arr_i32(key: str, vals: list[int]) -> bytes:
    k = key.encode()
    return (struct.pack("<Q", len(k)) + k + struct.pack("<I", _ARR) +
            struct.pack("<I", _I32) + struct.pack("<Q", len(vals)) +
            struct.pack(f"<{len(vals)}i", *vals))


def _kv_arr_str(key: str, vals: list[str]) -> bytes:
    k = key.encode()
    out = struct.pack("<Q", len(k)) + k + struct.pack("<I", _ARR) + struct.pack("<I", _STR)
    out += struct.pack("<Q", len(vals))
    for v in vals:
        out += struct.pack("<Q", len(v)) + v.encode()
    return out


def _blob(kvs: list[bytes]) -> bytes:
    return b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs)) + b"".join(kvs)


def _write(tmp_path: Path, kvs: list[bytes]) -> Path:
    p = tmp_path / "test.gguf"
    p.write_bytes(_blob(kvs))
    return p


def test_parse(tmp_path: Path) -> None:
    p = _write(tmp_path, [
        _kv_str("general.architecture", "testarch"),
        _kv_str("general.name", "TestModel"),
        _kv_u32("general.file_type", 15),
        _kv_u32("testarch.block_count", 4),
        _kv_u32("testarch.embedding_length", 256),
        _kv_u32("testarch.attention.head_count", 8),
        _kv_u32("testarch.attention.head_count_kv", 2),
        _kv_u32("testarch.attention.key_length", 32),
        _kv_u32("testarch.attention.value_length", 64),
        _kv_u32("testarch.context_length", 8192),
        _kv_u32("testarch.expert_count", 3),
        _kv_arr_str("tokenizer.ggml.tokens", ["a", "b", "c", "dd"]),  # skipped, not materialised
        _kv_arr_i32("tokenizer.ggml.scores", [1, 2]),                 # skipped
    ])
    info = read_model_info(p)
    assert info.architecture == "testarch" and info.name == "TestModel"
    assert (info.n_layer, info.n_embd, info.n_head, info.n_head_kv) == (4, 256, 8, 2)
    assert (info.head_dim_k, info.head_dim_v) == (32, 64)
    assert (info.ctx_train, info.expert_count, info.file_type) == (8192, 3, 15)
    assert info.sharded == 1 and info.size == p.stat().st_size


def test_per_layer_head_count_kv_mean(tmp_path: Path) -> None:
    p = _write(tmp_path, [
        _kv_str("general.architecture", "testarch"),
        _kv_u32("testarch.block_count", 4),
        _kv_u32("testarch.embedding_length", 256),
        _kv_u32("testarch.attention.head_count", 8),
        _kv_arr_i32("testarch.attention.head_count_kv", [8, 8, 4, 4]),
        _kv_u32("testarch.context_length", 8192),
    ])
    info = read_model_info(p)
    assert info.n_head_kv == 6
    assert info.head_dim_k == info.head_dim_v == 32  # fallback n_embd // n_head


def test_bad_magic(tmp_path: Path) -> None:
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"NOPE")
    with pytest.raises(GGUFError):
        read_model_info(p)
