"""Shared test fixtures: synthetic GGUF blobs and an isolated environment."""
from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from llamactl import config, gguf, hostinfo, sessions
from llamactl.hostinfo import Gpu, Host

_U32, _F32, _STR, _ARR, _I32 = 4, 6, 8, 9, 5


def kv_str(key: str, val: str) -> bytes:
    """String metadata pair."""
    k, v = key.encode(), val.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", _STR) + struct.pack("<Q", len(v)) + v


def kv_u32(key: str, val: int) -> bytes:
    """u32 metadata pair."""
    k = key.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", _U32) + struct.pack("<I", val)


def kv_arr_i32(key: str, vals: list[int]) -> bytes:
    """i32 array metadata pair."""
    k = key.encode()
    return (struct.pack("<Q", len(k)) + k + struct.pack("<I", _ARR) +
            struct.pack("<I", _I32) + struct.pack("<Q", len(vals)) +
            struct.pack(f"<{len(vals)}i", *vals))


def kv_arr_str(key: str, vals: list[str]) -> bytes:
    """String array metadata pair (skipped by the reader)."""
    k = key.encode()
    out = struct.pack("<Q", len(k)) + k + struct.pack("<I", _ARR) + struct.pack("<I", _STR)
    out += struct.pack("<Q", len(vals))
    for v in vals:
        out += struct.pack("<Q", len(v)) + v.encode()
    return out


def gguf_blob(kvs: list[bytes]) -> bytes:
    """A minimal GGUF v3 blob with no tensors."""
    return b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs)) + b"".join(kvs)


MINI_KVS = [
    kv_str("general.architecture", "testarch"),
    kv_str("general.name", "TestModel"),
    kv_u32("general.file_type", 15),
    kv_u32("testarch.block_count", 4),
    kv_u32("testarch.embedding_length", 256),
    kv_u32("testarch.attention.head_count", 8),
    kv_u32("testarch.attention.head_count_kv", 2),
    kv_u32("testarch.context_length", 8192),
]


@pytest.fixture
def gguf_file(tmp_path: Path) -> Path:
    """A synthetic 8B-ish GGUF file in tmp_path (>= MIN_MODEL_BYTES is faked by size)."""
    p = tmp_path / "test.gguf"
    # pad so it passes discover_models' 50 MB floor
    p.write_bytes(gguf_blob(MINI_KVS) + b"\0" * (50 * 1024 * 1024))
    return p


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Redirect all llamactl state into tmp_path and pin a fixed host.

    Returns:
        Namespace with cfg_dir, model_dir and host.
    """
    cfg_dir = tmp_path / "cfg"
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for mod, attrs in (
        (config, {"CONFIG_DIR": cfg_dir, "CONFIG_FILE": cfg_dir / "config.json",
                  "LOG_DIR": cfg_dir / "logs", "SEEN_FILE": cfg_dir / "seen.json",
                  "DEFAULT_MODEL_DIRS": [str(model_dir)]}),
        (sessions, {"CONFIG_DIR": cfg_dir, "STATE_FILE": cfg_dir / "sessions.json",
                    "LOG_DIR": cfg_dir / "logs"}),
        (gguf, {"CONFIG_DIR": cfg_dir, "CACHE_FILE": cfg_dir / "gguf_cache.json"}),
    ):
        for name, val in attrs.items():
            monkeypatch.setattr(mod, name, val)
    host = Host(gpus=[Gpu("GPU", int(16e9), int(16e9))], ram_total=int(24e9),
                ram_free=int(24e9), cpu_threads=16, cpu_cores=8, cpu_name="Test CPU")
    monkeypatch.setattr(hostinfo, "detect", lambda: host)
    return SimpleNamespace(cfg_dir=cfg_dir, model_dir=model_dir, host=host)
