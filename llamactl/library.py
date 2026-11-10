"""Model discovery, shared by the TUI and the CLI."""
from __future__ import annotations

from pathlib import Path

MIN_MODEL_BYTES = 50 * 1024 * 1024


def discover_models(dirs: list[str]) -> list[Path]:
    """Find GGUF files in the given dirs, skipping vocab stubs and non-first shards.

    Args:
        dirs: Directories to scan recursively; missing ones are skipped.

    Returns:
        Sorted model paths.
    """
    found: dict[Path, None] = {}
    for d in dirs:
        root = Path(d).expanduser()
        if not root.is_dir():
            continue
        for p in root.rglob("*.gguf"):
            if p.stat().st_size < MIN_MODEL_BYTES or "mmproj" in p.name:
                continue
            if "-of-" in p.stem and "-00001-of-" not in p.stem:
                continue
            found[p] = None
    return sorted(found)
