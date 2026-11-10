"""GGUF header parsing on synthetic blobs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from llamactl.gguf import GGUFError, ModelInfoCache, read_model_info

from conftest import gguf_blob, kv_arr_i32, kv_arr_str, kv_str, kv_u32


def _write(tmp_path: Path, kvs: list[bytes]) -> Path:
    p = tmp_path / "test.gguf"
    p.write_bytes(gguf_blob(kvs))
    return p


def test_parse(tmp_path: Path) -> None:
    p = _write(tmp_path, [
        kv_str("general.architecture", "testarch"),
        kv_str("general.name", "TestModel"),
        kv_u32("general.file_type", 15),
        kv_u32("testarch.block_count", 4),
        kv_u32("testarch.embedding_length", 256),
        kv_u32("testarch.attention.head_count", 8),
        kv_u32("testarch.attention.head_count_kv", 2),
        kv_u32("testarch.attention.key_length", 32),
        kv_u32("testarch.attention.value_length", 64),
        kv_u32("testarch.context_length", 8192),
        kv_u32("testarch.expert_count", 3),
        kv_arr_str("tokenizer.ggml.tokens", ["a", "b", "c", "dd"]),  # skipped, not materialised
        kv_arr_i32("tokenizer.ggml.scores", [1, 2]),                 # skipped
    ])
    info = read_model_info(p)
    assert info.architecture == "testarch" and info.name == "TestModel"
    assert (info.n_layer, info.n_embd, info.n_head, info.n_head_kv) == (4, 256, 8, 2)
    assert (info.head_dim_k, info.head_dim_v) == (32, 64)
    assert (info.ctx_train, info.expert_count, info.file_type) == (8192, 3, 15)
    assert info.sharded == 1 and info.size == p.stat().st_size
    assert info.parameter_count == 0


def test_per_layer_head_count_kv_mean(tmp_path: Path) -> None:
    p = _write(tmp_path, [
        kv_str("general.architecture", "testarch"),
        kv_u32("testarch.block_count", 4),
        kv_u32("testarch.embedding_length", 256),
        kv_u32("testarch.attention.head_count", 8),
        kv_arr_i32("testarch.attention.head_count_kv", [8, 8, 4, 4]),
        kv_u32("testarch.context_length", 8192),
    ])
    info = read_model_info(p)
    assert info.n_head_kv == 6
    assert info.head_dim_k == info.head_dim_v == 32  # fallback n_embd // n_head


def test_bad_magic(tmp_path: Path) -> None:
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"NOPE")
    with pytest.raises(GGUFError):
        read_model_info(p)


def _pooling_file(tmp_path: Path, pooling: int | None) -> Path:
    kvs = [
        kv_str("general.architecture", "testarch"),
        kv_u32("testarch.block_count", 4),
        kv_u32("testarch.embedding_length", 256),
        kv_u32("testarch.attention.head_count", 8),
        kv_u32("testarch.context_length", 8192),
    ]
    if pooling is not None:
        kvs.append(kv_u32("testarch.pooling_type", pooling))
    return _write(tmp_path, kvs)


def test_pooling_absent_is_not_embedding(tmp_path: Path) -> None:
    info = read_model_info(_pooling_file(tmp_path, None))
    assert info.pooling_type == -1 and not info.is_embedding


@pytest.mark.parametrize("pooling,embedding", [(0, False), (1, True), (2, True), (3, True), (4, True)])
def test_pooling_detection(tmp_path: Path, pooling: int, embedding: bool) -> None:
    info = read_model_info(_pooling_file(tmp_path, pooling))
    assert info.pooling_type == pooling and info.is_embedding is embedding


def test_cache_round_trips_pooling(tmp_path: Path) -> None:
    info = read_model_info(_pooling_file(tmp_path, 1))
    cache = ModelInfoCache(tmp_path / "cache.json")
    cache.put(info)
    got = ModelInfoCache(tmp_path / "cache.json").get(info.path)
    assert got == info and got.is_embedding


def test_old_cache_entry_is_reread(tmp_path: Path) -> None:
    info = read_model_info(_pooling_file(tmp_path, 1))
    cache_file = tmp_path / "cache.json"
    cache = ModelInfoCache(cache_file)
    cache.put(info)
    raw = json.loads(cache_file.read_text())
    del raw[str(info.path)]["pooling_type"]  # shape written by earlier versions
    cache_file.write_text(json.dumps(raw))
    assert ModelInfoCache(cache_file).get(info.path) is None


def test_read_only_cache_never_writes_and_corrupt_cache_is_empty(tmp_path: Path) -> None:
    info = read_model_info(_pooling_file(tmp_path, 1))
    cache_file = tmp_path / "cache.json"
    cache_file.write_text("{not json")
    cache = ModelInfoCache(cache_file, writable=False)
    assert cache.get(info.path) is None
    cache.put(info)
    assert cache.get(info.path) == info                  # remembered in memory
    assert cache_file.read_text() == "{not json"         # but the file is untouched
