"""Model discovery and name resolution, shared by the TUI and the CLI."""
from __future__ import annotations

from pathlib import Path

from .config import Config, derive_alias

MIN_MODEL_BYTES = 50 * 1024 * 1024
MAX_LISTED = 12      # candidates named in a "not found" message


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


def aliases_for(models: list[Path], cfg: Config) -> dict[Path, str]:
    """Alias of every model: the configured one, or the one `Config.entry` would derive.

    Read-only: nothing is added to the config. Models without an entry get derived aliases
    that avoid every configured alias and each other.

    Args:
        models: Model paths.
        cfg: Config holding the entries.

    Returns:
        Alias by model path.
    """
    taken = {e.alias for e in cfg.models.values() if e.alias}
    out: dict[Path, str] = {}
    for p in models:
        ent = cfg.models.get(str(p))
        if ent is not None and ent.alias:
            out[p] = ent.alias
    for p in models:
        if p not in out:
            out[p] = derive_alias(p, taken)
            taken.add(out[p])
    return out


def resolve_model(query: str, models: list[Path], cfg: Config) -> Path:
    """Find the model a user means.

    Matching order: exact alias, exact file stem, then a case-insensitive substring of the
    alias or the stem. Each step must give exactly one model to succeed.

    Args:
        query: What the user typed.
        models: Candidate model paths, usually from `discover_models`.
        cfg: Config holding the aliases.

    Returns:
        The matching path (an element of `models`).

    Raises:
        LookupError: When nothing matches, or several models do; the message lists candidates.
    """
    aliases = aliases_for(models, cfg)
    q = query.strip()
    if not q:
        raise LookupError("empty model name" + _available(models, aliases))
    steps = (
        [p for p in models if aliases[p] == q],
        [p for p in models if p.stem == q],
        [p for p in models if q.lower() in aliases[p].lower() or q.lower() in p.stem.lower()],
    )
    for hits in steps:
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            names = ", ".join(sorted(f"{aliases[p]} ({p})" for p in hits))
            raise LookupError(f"'{query}' is ambiguous; it matches: {names}")
    raise LookupError(f"no model matches '{query}'" + _available(models, aliases))


def _available(models: list[Path], aliases: dict[Path, str]) -> str:
    """Suffix listing known aliases for an error message."""
    if not models:
        return " (no models found; check the model directories)"
    names = sorted(aliases[p] for p in models)
    more = f", ... ({len(names) - MAX_LISTED} more)" if len(names) > MAX_LISTED else ""
    return ". Available: " + ", ".join(names[:MAX_LISTED]) + more
