"""Minimal Hugging Face Hub client: search GGUF repos, list files, resumable download."""
from __future__ import annotations

import os
import re
import threading
from collections.abc import Callable
from pathlib import Path

import httpx

from .config import CONFIG_DIR

HF_URL = "https://huggingface.co"
TOKEN_FILE = CONFIG_DIR / "hf_token"
CHUNK = 1 << 20
QUANT_RE = re.compile(r"(?<![A-Za-z0-9])((?:IQ|Q)\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4)(?![A-Za-z0-9])", re.I)
SHARD_RE = re.compile(r"^(?P<prefix>.*)-(?P<idx>\d+)-of-(?P<total>\d+)\.gguf$")


class HFError(Exception):
    """Readable error for the UI."""


def load_token() -> str | None:
    """Token from env, llamactl's own file, or the official hf CLI cache (first found)."""
    if tok := os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return tok
    for f in (TOKEN_FILE, Path.home() / ".cache" / "huggingface" / "token"):
        if f.exists() and (tok := f.read_text().strip()):
            return tok
    return None


def save_token(token: str) -> None:
    """Store the token with owner-only permissions; empty string deletes it."""
    if not token:
        TOKEN_FILE.unlink(missing_ok=True)
        return
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.touch(mode=0o600)
    TOKEN_FILE.write_text(token)


def _headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _check(r: httpx.Response, repo: str = "") -> None:
    """Turn auth/not-found statuses into friendly errors."""
    if r.status_code in (401, 403):
        raise HFError(f"{repo or 'This'} needs a valid token and accepted terms on its HF page (press k to set a token).")
    if r.status_code == 404:
        raise HFError(f"{repo or 'Resource'} not found (private repos also return 404 without a token).")
    r.raise_for_status()


def search(query: str, token: str | None, limit: int = 30) -> list[dict]:
    """Search GGUF model repos, most downloaded first."""
    try:
        r = httpx.get(f"{HF_URL}/api/models", headers=_headers(token), timeout=15,
                      params=[("search", query), ("filter", "gguf"), ("sort", "downloads"), ("limit", limit),
                        *[("expand[]", f) for f in ("downloads", "likes", "gguf", "gated", "lastModified")]])
        _check(r)
    except httpx.HTTPError as e:
        raise HFError(f"Search failed: {e}") from e
    return r.json()


def quant_of(path: str) -> str:
    """Quantization tag guessed from the file name ('' if none found)."""
    m = QUANT_RE.search(Path(path).name)
    return m[1].upper() if m else ""


def gguf_sizes(repo: str, token: str | None) -> dict[str, int]:
    """Map of every .gguf path in the repo to its size in bytes."""
    try:
        r = httpx.get(f"{HF_URL}/api/models/{repo}/tree/main", headers=_headers(token), timeout=15,
                      params={"recursive": "true"})
        _check(r, repo)
    except httpx.HTTPError as e:
        raise HFError(f"Could not list files: {e}") from e
    return {f["path"]: (f.get("lfs") or {}).get("size", f.get("size", 0))
            for f in r.json() if f["type"] == "file" and f["path"].endswith(".gguf")}


def list_gguf(repo: str, token: str | None) -> list[dict]:
    """GGUF files as [{'path', 'size', 'parts', 'quant'}]; sharded models appear once (first shard, total size)."""
    files = gguf_sizes(repo, token)
    out = []
    for path, size in sorted(files.items()):
        m = SHARD_RE.match(path)
        if m and int(m["idx"]) != 1:
            continue
        parts = shard_group(path, list(files))
        out.append({"path": path, "size": sum(files[p] for p in parts), "parts": len(parts), "quant": quant_of(path)})
    return out


def shard_group(path: str, all_paths: list[str]) -> list[str]:
    """All files that must be downloaded together with `path` (itself if not sharded)."""
    m = SHARD_RE.match(path)
    if not m:
        return [path]
    return sorted(p for p in all_paths if (sm := SHARD_RE.match(p)) and sm["prefix"] == m["prefix"])


def download(repo: str, paths: list[str], dest: Path, token: str | None, cancel: threading.Event,
             progress: Callable[[int, int, str], None]) -> list[Path]:
    """Download files to `dest` with resume support. progress(done_bytes, total_bytes, filename)."""
    dest.mkdir(parents=True, exist_ok=True)
    total = sum(_remote_size(repo, p, token) for p in paths)
    done = 0
    out = []
    for p in paths:
        final, part = dest / Path(p).name, dest / (Path(p).name + ".part")
        if final.exists():
            done += final.stat().st_size
            out.append(final)
            continue
        have = part.stat().st_size if part.exists() else 0
        headers = {**_headers(token), **({"Range": f"bytes={have}-"} if have else {})}
        try:
            with httpx.stream("GET", f"{HF_URL}/{repo}/resolve/main/{p}", headers=headers,
                              follow_redirects=True, timeout=30) as r:
                if r.status_code == 416:
                    have = 0
                    part.unlink(missing_ok=True)
                    raise HFError("Partial file was invalid; press Enter again to restart the download.")
                _check(r, repo)
                if have and r.status_code != 206:
                    have = 0
                with part.open("ab" if have else "wb") as fh:
                    done += have
                    for chunk in r.iter_bytes(CHUNK):
                        if cancel.is_set():
                            raise HFError("Download cancelled (partial file kept, will resume).")
                        fh.write(chunk)
                        done += len(chunk)
                        progress(done, total, final.name)
        except httpx.HTTPError as e:
            raise HFError(f"Download failed: {e} (partial file kept, will resume).") from e
        part.rename(final)
        out.append(final)
    return out


def _remote_size(repo: str, path: str, token: str | None) -> int:
    """Size in bytes from a HEAD request, 0 if unknown."""
    try:
        r = httpx.head(f"{HF_URL}/{repo}/resolve/main/{path}", headers=_headers(token), follow_redirects=True, timeout=15)
        _check(r, repo)
        return int(r.headers.get("content-length", 0))
    except httpx.HTTPError:
        return 0
