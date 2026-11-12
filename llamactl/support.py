"""Architecture support: can this llama.cpp build load this model's architecture?

Three checks, from cheapest to most remote (SPEC 13.2):

- learned failures: a load whose log said ``unknown model architecture: '<x>'`` is recorded
  per build and is the final word;
- the build check: ``name + b"\\0"`` searched in ``libllama.so`` (or ``llama-server``), a
  tail match because the linker may store a name only inside a longer one;
- upstream: master's ``src/llama-arch.cpp``, cached for a day, tells a missing name apart
  from one llama.cpp will never run.

Everything that touches the filesystem, the clock or the network is injectable.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable

import httpx

from .config import CONFIG_DIR


class Support(StrEnum):
    """Support of a model architecture by a llama.cpp build."""

    OK = "ok"                    # the build knows the architecture
    UPDATE = "update"            # not in this build, but in llama.cpp master (or master unknown)
    UNSUPPORTED = "unsupported"  # in neither: llama.cpp does not run it
    NOT_LLM = "not_llm"          # not a standalone text model (projector, draft, audio)
    UNKNOWN = "unknown"          # nothing could be checked; treated as ok


# Architectures llama.cpp builds but that are not standalone text models: mmproj projectors,
# draft-only models, audio/TTS decoders.
NOT_LLM_ARCHS = frozenset({"clip", "eagle3", "dflash", "wavtokenizer-dec", "qwen3tts",
                           "pockettts"})

UPSTREAM_URL = ("https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/"
                "llama-arch.cpp")
UPSTREAM_TTL = 24 * 3600       # seconds the upstream arch list is trusted
SUPPORT_FILE = "arch_support.json"
UPSTREAM_FILE = "arch_upstream.json"

_UNKNOWN_ARCH_RE = re.compile(r"unknown model architecture:\s*'([^']+)'")
_ARCH_NAME_RE = re.compile(r'\{\s*LLM_ARCH_(\w+),\s*"([^"]*)"\s*\}')


def _fetch_text(url: str) -> str:
    """GET `url` and return its body (the default `fetch` of `SupportChecker`)."""
    r = httpx.get(url, timeout=10, follow_redirects=True)
    r.raise_for_status()
    return r.text


def unknown_arch(text: str) -> str | None:
    """The architecture name in ``unknown model architecture: '<x>'`` anywhere in `text`.

    Args:
        text: Log output of llama.cpp.

    Returns:
        The name, or None when the line is absent.
    """
    m = _UNKNOWN_ARCH_RE.search(text)
    return m.group(1) if m else None


def parse_upstream(source: str) -> frozenset[str]:
    """The architecture names of master's ``LLM_ARCH_NAMES`` table.

    Args:
        source: Text of ``src/llama-arch.cpp``.

    Returns:
        The string values of the table, without the ``LLM_ARCH_UNKNOWN`` "(unknown)" entry.
    """
    start = source.find("LLM_ARCH_NAMES")
    block = source[start:] if start >= 0 else source
    end = block.find("};")
    if end >= 0:
        block = block[:end]
    return frozenset(name for enum, name in _ARCH_NAME_RE.findall(block)
                     if enum != "UNKNOWN")


def rebuild_commands(bin_dir: str) -> str:
    """Shell commands that rebuild the llama.cpp a `bin_dir` belongs to.

    ``<src>/build/bin`` two levels up is the source tree; when it holds ``CMakeLists.txt`` the
    commands pull and build it there, else a generic pointer to the build documentation.

    Args:
        bin_dir: The configured llama.cpp binary folder.

    Returns:
        One line a user can paste into a shell.
    """
    src = Path(bin_dir).expanduser()
    if len(src.parents) >= 2 and (src.parents[1] / "CMakeLists.txt").is_file():
        return (f"cd {src.parents[1]} && git pull && "
                "cmake --build build --config Release -j")
    return ("rebuild llama.cpp at the current master; see "
            "https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md")


@dataclass(frozen=True)
class ArchSupport:
    """The support verdict for one architecture.

    Attributes:
        arch: The GGUF ``general.architecture`` name.
        status: The verdict.
        build: llama-server version the check ran against ("" when unknown).
        upstream: llama.cpp master lists the architecture (False when master is unknown or
            lacks it).
    """

    arch: str
    status: Support
    build: str = ""
    upstream: bool = False

    @property
    def blocks(self) -> bool:
        """True when serving must refuse the model: update, unsupported or not a text model."""
        return self.status in (Support.UPDATE, Support.UNSUPPORTED, Support.NOT_LLM)

    def label(self) -> str:
        """The short label verdict cells show: ``update llama.cpp`` / ``unsupported`` /
        ``not a text model``."""
        return {Support.UPDATE: "update llama.cpp",
                Support.UNSUPPORTED: "unsupported",
                Support.NOT_LLM: "not a text model"}.get(self.status, "")

    def reason(self) -> str:
        """One line for the user, "" for ok/unknown."""
        in_build = f" ({self.build})" if self.build else ""
        if self.status == Support.UPDATE:
            if self.upstream:
                return (f"architecture '{self.arch}' is not in your llama.cpp{in_build} "
                        "but is in llama.cpp master: rebuild llama.cpp")
            return (f"architecture '{self.arch}' is not in your llama.cpp{in_build}; "
                    "a newer build may support it")
        if self.status == Support.UNSUPPORTED:
            return (f"llama.cpp does not run architecture '{self.arch}'; it may be meant "
                    "for another runtime such as vLLM")
        if self.status == Support.NOT_LLM:
            return f"'{self.arch}' is not a text model (projector / draft / audio)"
        return ""


class SupportChecker:
    """Decides whether a llama.cpp build can load a model architecture.

    Args:
        bin_dir: Folder of the build (its ``libllama.so``, else ``llama-server``, is searched).
        state_dir: Folder for ``arch_support.json`` and ``arch_upstream.json`` (default: the
            llamactl state directory).
        read_bytes: Reads a file's bytes (injectable).
        fetch: GETs a URL and returns its text (injectable; default: httpx).
        now: Wall clock in Unix seconds (injectable).
        writable: Write the state files (False for read-only commands; the state is still
            kept in memory).

    Attributes:
        bin_dir: The build folder, ``~`` expanded.
        state_dir: Where the state files live.
        writable: Whether `record_failure` and `refresh_upstream` persist.
    """

    def __init__(self, bin_dir: str, state_dir: Path | None = None,
                 read_bytes: Callable[[Path], bytes] = Path.read_bytes,
                 fetch: Callable[[str], str] | None = None,
                 now: Callable[[], float] = time.time,
                 writable: bool = True) -> None:
        self.bin_dir = Path(bin_dir).expanduser()
        self.state_dir = state_dir or CONFIG_DIR
        self.writable = writable
        self._read_bytes = read_bytes
        self._fetch = fetch or _fetch_text
        self._now = now
        self._failures = self._read_json(SUPPORT_FILE)     # {"build": str, "unknown": [names]}
        self._upstream = self._read_json(UPSTREAM_FILE)    # {"fetched": ts, "names": [...]}
        self._blob_key: tuple[str, int, int] | None = None  # (path, size, mtime_ns) scanned
        self._blob: bytes | None = None
        self._lock = threading.Lock()

    # -- checks ------------------------------------------------------------------------

    def check(self, arch: str, build: str = "") -> ArchSupport:
        """The support verdict of `arch` for this build.

        Order: no arch is unknown; the not-a-text-model table; a learned failure for this
        exact build; the tail match in the build's library. Upstream only decides between
        "update" and "unsupported", and ``unknown`` never blocks. A file stat plus at most
        one read per binary change, never network.

        Args:
            arch: The GGUF ``general.architecture`` name.
            build: llama-server version string ("" matches no learned failure).

        Returns:
            The verdict; `ArchSupport.blocks` says whether serving must refuse.
        """
        if not arch:
            return ArchSupport(arch, Support.UNKNOWN, build)
        if arch in NOT_LLM_ARCHS:
            return ArchSupport(arch, Support.NOT_LLM, build)
        upstream = self.upstream_names()
        in_upstream = None if upstream is None else arch in upstream
        if build and arch in self.learned(build):
            # the learned failure is the final word: it came from this exact build
            status = Support.UNSUPPORTED if in_upstream is False else Support.UPDATE
            return ArchSupport(arch, status, build, bool(in_upstream))
        blob = self._build_bytes()
        if blob is None:
            return ArchSupport(arch, Support.UNKNOWN, build)
        if arch.encode() + b"\0" in blob:
            return ArchSupport(arch, Support.OK, build)
        status = Support.UNSUPPORTED if in_upstream is False else Support.UPDATE
        return ArchSupport(arch, status, build, bool(in_upstream))

    def _binary(self) -> Path | None:
        """``libllama.so`` of the build, else ``llama-server``, else None."""
        libs = sorted(self.bin_dir.glob("libllama.so*"))
        if libs:
            return libs[0]
        server = self.bin_dir / "llama-server"
        return server if server.is_file() else None

    def _build_bytes(self) -> bytes | None:
        """Bytes of the build's library or binary, cached by (path, size, mtime_ns).

        A rebuild is picked up without a restart because the cache key carries the file's
        size and mtime.
        """
        path = self._binary()
        if path is None:
            return None
        try:
            st = path.stat()
        except OSError:
            return None
        key = (str(path), st.st_size, st.st_mtime_ns)
        with self._lock:
            if key != self._blob_key:
                try:
                    self._blob = self._read_bytes(path)
                except OSError:
                    self._blob = None
                self._blob_key = key
            return self._blob

    # -- learned failures ---------------------------------------------------------------

    def learned(self, build: str) -> frozenset[str]:
        """Architecture names recorded as failing on `build` ("" or another build: none)."""
        if self._failures.get("build") != build:
            return frozenset()
        names = self._failures.get("unknown")
        return frozenset(str(n) for n in names) if isinstance(names, list) else frozenset()

    def record_failure(self, arch: str, build: str) -> None:
        """Record that `arch` failed to load on `build` (the final word over the build check).

        Records of a different build are replaced; an empty `arch` or `build` records
        nothing. Never raises.

        Args:
            arch: The name from an ``unknown model architecture`` log line.
            build: The llama-server version the failure happened on.
        """
        if not arch or not build:
            return
        with self._lock:
            if self._failures.get("build") != build:
                self._failures = {"build": build, "unknown": []}
            names = set(str(n) for n in self._failures.get("unknown") or [])
            if arch in names:
                return
            names.add(arch)
            self._failures["unknown"] = sorted(names)
            self._write_json(SUPPORT_FILE, self._failures)

    def learn(self, text: str, build: str) -> None:
        """Record the architecture when `text` names an unknown one; else do nothing.

        Args:
            text: Log output of a failed load.
            build: The llama-server version the failure happened on.
        """
        if arch := unknown_arch(text):
            self.record_failure(arch, build)

    # -- upstream ------------------------------------------------------------------------

    def upstream_names(self) -> frozenset[str] | None:
        """Architecture names of master's table, or None when never fetched."""
        names = self._upstream.get("names")
        return frozenset(str(n) for n in names) if isinstance(names, list) else None

    def upstream_stale(self) -> bool:
        """True when the cached upstream list is older than `UPSTREAM_TTL` or absent."""
        fetched = self._upstream.get("fetched", 0)
        return not isinstance(fetched, (int, float)) or self._now() - fetched >= UPSTREAM_TTL

    def refresh_upstream(self, force: bool = False) -> bool:
        """Fetch master's ``llama-arch.cpp`` when the cache is stale (or `force`).

        A failed fetch or a parse with no names keeps the old cache. Never raises.

        Args:
            force: Fetch even when the cache is fresh.

        Returns:
            True when a (fresh or fetched) list is now known.
        """
        if not force and not self.upstream_stale():
            return True
        try:
            names = parse_upstream(self._fetch(UPSTREAM_URL))
        except Exception:
            return self.upstream_names() is not None
        if not names:
            return self.upstream_names() is not None
        with self._lock:
            self._upstream = {"fetched": self._now(), "names": sorted(names)}
            self._write_json(UPSTREAM_FILE, self._upstream)
        return True

    # -- state files --------------------------------------------------------------------

    def _read_json(self, name: str) -> dict:
        """A state file as a dict ({} when missing or corrupt)."""
        try:
            data = json.loads((self.state_dir / name).read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_json(self, name: str, data: dict) -> None:
        """Persist a state file unless read-only; a failed write loses the data, silently."""
        if not self.writable:
            return
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            (self.state_dir / name).write_text(json.dumps(data, indent=2))
        except OSError:
            pass
