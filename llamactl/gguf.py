"""GGUF header reader: model metadata without loading tensors, stdlib only."""
from __future__ import annotations

import json
import re
import struct
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import CONFIG_DIR

CACHE_FILE = CONFIG_DIR / "gguf_cache.json"
SHARD_RE = re.compile(r"^(?P<prefix>.*)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")
# Tensors ``--n-cpu-moe`` moves to RAM (routed experts; shared ``_shexp`` experts stay on the
# GPU). Group 1 is the layer index. Same pattern llama.cpp uses (common/common.h).
EXPERT_RE = re.compile(r"blk\.(\d+)\.ffn_(up|down|gate|gate_up)_(ch|)exps")

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
        parameter_count: general.parameter_count or 0 when absent.
        pooling_type: {arch}.pooling_type (0 none, 1 mean, 2 cls, 3 last, 4 rank), -1 when absent.
        expert_bytes: Bytes of the expert tensors ``--n-cpu-moe`` can keep in RAM, per layer
            (0 for dense layers); empty for dense models and when the tensor table could not
            be read.
        dense_bytes: Bytes of every other tensor (0 without the tensor table).
        expert_used: {arch}.expert_used_count (routed experts used per token), 0 when absent.
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
    parameter_count: int
    pooling_type: int = -1
    expert_bytes: tuple[int, ...] = ()
    dense_bytes: int = 0
    expert_used: int = 0

    @property
    def is_embedding(self) -> bool:
        """True for embedding/reranker models: the GGUF declares a pooling type."""
        return self.pooling_type > 0


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


def _header(fh, name: str) -> tuple[int, dict]:
    """Read magic, version, tensor count and the metadata kv block of a GGUF file.

    Args:
        fh: Binary file at position 0, left right after the kv block.
        name: File name for error messages.

    Returns:
        (tensor_count, metadata dict).

    Raises:
        GGUFError: On a bad magic or unsupported version.
    """
    if fh.read(4) != b"GGUF":
        raise GGUFError(f"{name}: not a GGUF file")
    (version,) = struct.unpack("<I", fh.read(4))
    if version not in (2, 3):
        raise GGUFError(f"{name}: unsupported GGUF version {version}")
    (count,) = struct.unpack("<Q", fh.read(8))
    return count, _read_kv(fh)


def _tensor_sizes(path: Path) -> list[tuple[str, int]]:
    """Byte size of every tensor of one GGUF shard, from the info table (never the data).

    Tensor offsets are relative to the data section, which starts at the position after
    the last tensor info rounded up to ``general.alignment`` (default 32), and tensors are
    laid out back to back: a size is the gap to the next larger offset, the last tensor
    runs to the end of the file.

    Args:
        path: One shard file.

    Returns:
        (tensor name, byte size) pairs, in table order.

    Raises:
        GGUFError: On a bad header, a truncated table or offsets that do not add up.
        OSError: When the file cannot be read.
    """
    total = path.stat().st_size
    with path.open("rb") as fh:
        count, kv = _header(fh, path.name)
        infos: list[tuple[str, int]] = []
        for _ in range(count):
            (nlen,) = struct.unpack("<Q", fh.read(8))
            name = fh.read(nlen).decode("utf-8", errors="replace")
            (n_dims,) = struct.unpack("<I", fh.read(4))
            fh.seek(8 * n_dims + 4, 1)                      # dims + ggml type, both unused
            (offset,) = struct.unpack("<Q", fh.read(8))
            infos.append((name, offset))
        align = max(1, int(kv.get("general.alignment", 32)))
        end = total - (fh.tell() + align - 1) // align * align
        ordered = sorted(infos, key=lambda t: t[1])
        out: list[tuple[str, int]] = []
        for i, (name, off) in enumerate(ordered):
            nxt = ordered[i + 1][1] if i + 1 < len(ordered) else end
            if nxt - off < 0:
                raise GGUFError(f"{path.name}: tensor offsets do not add up")
            out.append((name, nxt - off))
    return out


def tensor_table(path: Path) -> list[tuple[str, int]]:
    """(name, byte size) of every tensor of a model, over all its shards.

    Reads only the tensor info tables, never tensor data.

    Args:
        path: First shard of the model.

    Returns:
        (tensor name, byte size) pairs.

    Raises:
        GGUFError: On a bad header, a truncated table or offsets that do not add up.
        OSError: When a file cannot be read.
    """
    return [t for shard in _shards(path) for t in _tensor_sizes(shard)]


def _expert_split(path: Path, n_layer: int) -> tuple[tuple[int, ...], int]:
    """Expert vs dense tensor bytes of a MoE model, summed over every shard.

    Args:
        path: First shard of the model.
        n_layer: Transformer block count.

    Returns:
        (expert bytes per layer, bytes of every other tensor); ``((), 0)`` when a shard's
        tensor table cannot be read or no expert tensor is found.
    """
    layers = [0] * max(0, n_layer)
    dense = 0
    try:
        for shard in _shards(path):
            for name, size in _tensor_sizes(shard):
                m = EXPERT_RE.search(name)
                if m and int(m.group(1)) < n_layer:
                    layers[int(m.group(1))] += size
                else:
                    dense += size
    except (GGUFError, OSError, struct.error):
        return (), 0
    return (tuple(layers), dense) if any(layers) else ((), 0)


def read_model_info(path: Path) -> ModelInfo:
    """Parse the GGUF header of a model file.

    Args:
        path: Path to the .gguf file (first shard when split).

    Returns:
        The parsed ModelInfo. For a MoE model the tensor table of every shard is read too
        (never tensor data) to split expert from dense bytes per layer; an unreadable table
        leaves `expert_bytes` empty.

    Raises:
        GGUFError: On a bad magic, unsupported version, or missing keys.
    """
    try:
        with path.open("rb") as fh:
            _count, kv = _header(fh, path.name)
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
    expert_count = int(kv.get(a + "expert_count", 0))
    expert_bytes: tuple[int, ...] = ()
    dense_bytes = 0
    if expert_count:
        expert_bytes, dense_bytes = _expert_split(path, n_layer)
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
        expert_count=expert_count,
        file_type=int(kv.get("general.file_type", 0)),
        sharded=int(kv.get("split.count", 1)),
        parameter_count=int(kv.get("general.parameter_count", 0)),
        pooling_type=int(kv.get(a + "pooling_type", -1)),
        expert_bytes=expert_bytes,
        dense_bytes=dense_bytes,
        expert_used=int(kv.get(a + "expert_used_count", 0)),
    )


class ModelInfoCache:
    """Disk cache of ModelInfo keyed by path, invalidated on mtime change.

    Args:
        cache_file: JSON file used for persistence. A missing or corrupt file is an empty cache.
        writable: When False, `put` remembers entries in memory only and never writes the file
            (for read-only commands).
    """

    def __init__(self, cache_file: Path | None = None, writable: bool = True) -> None:
        self.cache_file = cache_file or CACHE_FILE
        self.writable = writable
        try:
            data = json.loads(self.cache_file.read_text())
        except (OSError, ValueError):
            data = {}
        self._data: dict = data if isinstance(data, dict) else {}
        self._lock = threading.Lock()          # the app scan and the setup scan may both `put`

    def get(self, path: Path) -> ModelInfo | None:
        """Cached info for `path`, or None when missing, stale or written before
        pooling_type/expert_bytes.

        Args:
            path: Model file path.
        """
        ent = self._data.get(str(path))
        if not ent or "pooling_type" not in ent or "expert_bytes" not in ent:
            return None                                    # entries from older versions are re-read
        try:
            if ent["mtime"] != path.stat().st_mtime:
                return None
            ent = {**ent, "path": Path(ent["path"]),
                   "expert_bytes": tuple(ent["expert_bytes"])}
            return ModelInfo(**{k: v for k, v in ent.items() if k != "mtime"})
        except (OSError, TypeError, KeyError):
            return None

    def put(self, info: ModelInfo) -> None:
        """Store an entry and persist the cache.

        Args:
            info: Parsed model info.
        """
        with self._lock:
            self._data[str(info.path)] = {**asdict(info), "path": str(info.path),
                                          "mtime": info.path.stat().st_mtime}
            if not self.writable:
                return
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps(self._data, indent=2))
