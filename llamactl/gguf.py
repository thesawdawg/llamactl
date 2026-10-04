"""GGUF header reader: model metadata without loading tensors, stdlib only."""
from __future__ import annotations

import json
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import CONFIG_DIR

CACHE_FILE = CONFIG_DIR / "gguf_cache.json"
SHARD_RE = re.compile(r"^(?P<prefix>.*)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")

# GGUF value types: fixed byte size for scalars; strings/arrays carry lengths.
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STR, _ARR, _U64, _I64, _F64 = range(13)
_SIZE = {_U8: 1, _I8: 1, _U16: 2, _I16: 2, _U32: 4, _I32: 4, _F32: 4, _BOOL: 1, _U64: 8, _I64: 8, _F64: 8}
_FMT = {_U8: "B", _I8: "b", _U16: "H", _I16: "h", _U32: "I", _I32: "i", _F32: "f",
        _BOOL: "?", _U64: "Q", _I64: "q", _F64: "d"}


class GGUFError(Exception):
    """Unreadable or unsupported GGUF file."""


@dataclass(frozen=True)
class ModelInfo:
    """Metadata parsed from a GGUF header.

    Attributes:
        path: Model file (first shard when split).
        size: Sum of all shards on disk, in bytes.
        architecture: e.g. "qwen3".
        name: general.name or the file stem.
        n_layer: Transformer block count.
        n_embd: Embedding width.
        n_head: Attention head count.
        n_head_kv: KV head count (mean when stored per layer).
        head_dim_k / head_dim_v: Per-head key/value dims (fallback n_embd // n_head).
        ctx_train: Training context length.
        expert_count: Routed experts (0 for dense models).
        file_type: GGUF file type (quantisation) id.
        sharded: split.count or 1.
    """

    path: Path
    size: int
    architecture: str
    name: str
    n_layer: int
    n_embd: int
    n_head: int
    n_head_kv: int
    head_dim_k: int
    head_dim_v: int
    ctx_train: int
    expert_count: int
    file_type: int
    sharded: int


def _read_kv(fh) -> dict:
    """Read all metadata key/value pairs from a positioned GGUF file handle.

    Numeric arrays are materialised only for non-tokenizer keys; string arrays
    (vocab, merges) are skipped by seeking. Scalars are decoded per the spec.

    Args:
        fh: Binary file positioned after the kv_count field.

    Returns:
        Dict of decoded metadata.
    """
    (kv_count,) = struct.unpack("<Q", fh.read(8))
    out: dict = {}
    for _ in range(kv_count):
        (klen,) = struct.unpack("<Q", fh.read(8))
        key = fh.read(klen).decode("utf-8", errors="replace")
        (vtype,) = struct.unpack("<I", fh.read(4))
        # anything outside tokenizer.* is cheap: general.*, {arch}.*, split.*
        wanted = not key.startswith("tokenizer.")
        if vtype == _ARR:
            (etype,) = struct.unpack("<I", fh.read(4))
            (count,) = struct.unpack("<Q", fh.read(8))
            if etype == _STR:
                for _i in range(count):
                    (slen,) = struct.unpack("<Q", fh.read(8))
                    fh.seek(slen, 1)
            elif wanted:
                out[key] = list(struct.unpack(f"<{count}{_FMT[etype]}", fh.read(count * _SIZE[etype])))
            else:
                fh.seek(count * _SIZE[etype], 1)
        elif vtype == _STR:
            (slen,) = struct.unpack("<Q", fh.read(8))
            data = fh.read(slen)
            if wanted:
                out[key] = data.decode("utf-8", errors="replace")
        else:
            (val,) = struct.unpack(f"<{_FMT[vtype]}", fh.read(_SIZE[vtype]))
            if wanted:
                out[key] = val
    return out


def _mean(v) -> int:
    """Scalar or per-layer list to a single int (mean for lists)."""
    return int(sum(v) / len(v) + 0.5) if isinstance(v, list) else int(v)


def _shards(path: Path) -> list[Path]:
    """All shards of a split model (or just `path` when not split), sorted."""
    m = SHARD_RE.match(path.name)
    if not m:
        return [path]
    return sorted(path.parent.glob(f"{m['prefix']}-*-of-{m['total']}.gguf"))


def read_model_info(path: Path) -> ModelInfo:
    """Parse the GGUF header of a model file.

    Args:
        path: Path to the .gguf file (first shard when split).

    Returns:
        The parsed ModelInfo.

    Raises:
        GGUFError: On a bad magic, unsupported version, or missing keys.
    """
    try:
        with path.open("rb") as fh:
            if fh.read(4) != b"GGUF":
                raise GGUFError(f"{path.name}: not a GGUF file")
            (version,) = struct.unpack("<I", fh.read(4))
            if version not in (2, 3):
                raise GGUFError(f"{path.name}: unsupported GGUF version {version}")
            fh.seek(8, 1)  # tensor_count, unused
            kv = _read_kv(fh)
    except OSError as e:
        raise GGUFError(f"{path}: {e}") from e
    arch = kv.get("general.architecture", "")
    a = f"{arch}." if arch else ""
    try:
        n_layer = int(kv[a + "block_count"])
        n_embd = int(kv[a + "embedding_length"])
        n_head = int(kv[a + "attention.head_count"])
    except KeyError as e:
        raise GGUFError(f"{path.name}: missing metadata key {e}") from e
    return ModelInfo(
        path=path,
        size=sum(p.stat().st_size for p in _shards(path)),
        architecture=arch,
        name=str(kv.get("general.name") or path.stem),
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        n_head_kv=_mean(kv.get(a + "attention.head_count_kv", n_head)),
        head_dim_k=int(kv.get(a + "attention.key_length", n_embd // n_head)),
        head_dim_v=int(kv.get(a + "attention.value_length", n_embd // n_head)),
        ctx_train=int(kv.get(a + "context_length", 0)),
        expert_count=int(kv.get(a + "expert_count", 0)),
        file_type=int(kv.get("general.file_type", 0)),
        sharded=int(kv.get("split.count", 1)),
    )


class ModelInfoCache:
    """Disk cache of ModelInfo keyed by path, invalidated on mtime change.

    Args:
        cache_file: JSON file used for persistence.
    """

    def __init__(self, cache_file: Path = CACHE_FILE) -> None:
        self.cache_file = cache_file
        self._data: dict = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    def get(self, path: Path) -> ModelInfo | None:
        """Cached info for `path`, or None when missing or stale.

        Args:
            path: Model file path.
        """
        ent = self._data.get(str(path))
        if not ent:
            return None
        try:
            if ent["mtime"] != path.stat().st_mtime:
                return None
            ent = {**ent, "path": Path(ent["path"])}
            return ModelInfo(**{k: v for k, v in ent.items() if k != "mtime"})
        except (OSError, TypeError, KeyError):
            return None

    def put(self, info: ModelInfo) -> None:
        """Store an entry and persist the cache.

        Args:
            info: Parsed model info.
        """
        self._data[str(info.path)] = {**asdict(info), "path": str(info.path),
                                      "mtime": info.path.stat().st_mtime}
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(self._data, indent=2))
