"""Build the llama-server command line from a profile."""
from __future__ import annotations

import re
import shlex
from pathlib import Path

from .config import Config, Profile
from .schema import SETTINGS, Kind

# Flags that place experts or tensors by hand; with any of them llama.cpp's --fit cannot
# adjust around the user's placement (common/fit.cpp aborts with "already set by user",
# logs a warning and loads unfitted), so llamactl turns fit off instead.
PLACEMENT_FLAGS = frozenset(
    {"-ot", "--override-tensor", "-cmoe", "--cpu-moe", "-ncmoe", "--n-cpu-moe"})


def places_tensors(prof: Profile) -> bool:
    """True when the profile places experts or tensors by hand.

    That is ``n_cpu_moe`` above 0, an ``override_tensor`` setting (arrives with phase 15;
    read with getattr so this works already), or one of `PLACEMENT_FLAGS` among the
    ``extra_args`` tokens.

    Args:
        prof: Profile to inspect.

    Returns:
        Whether a manual tensor placement is set.
    """
    if prof.n_cpu_moe > 0 or getattr(prof, "override_tensor", ""):
        return True
    try:
        tokens = shlex.split(prof.extra_args)
    except ValueError:
        return False
    return any(t.partition("=")[0] in PLACEMENT_FLAGS for t in tokens)


def fit_conflict(prof: Profile) -> bool:
    """True when the profile places tensors by hand while auto-fit is not off.

    Args:
        prof: Profile to inspect.

    Returns:
        Whether llama.cpp's --fit would conflict with the manual placement.
    """
    return places_tensors(prof) and prof.fit != "off"


def effective_profile(prof: Profile) -> Profile:
    """The profile as it is launched: fit forced off under a manual tensor placement.

    Args:
        prof: The stored or edited profile.

    Returns:
        ``prof.merged(fit="off")`` when `fit_conflict` applies, else `prof` unchanged.
    """
    return prof.merged(fit="off") if fit_conflict(prof) else prof


def build_command(cfg: Config, model: str, prof: Profile, port: int) -> list[str]:
    """Build the llama-server argv for a model + profile.

    Iterates the settings registry in order: values equal to their unset value
    are skipped, port uses the resolved port, extra_args is appended last. When the
    profile places tensors by hand while auto-fit is on, fit is forced off (see
    `effective_profile`): llama.cpp's --fit aborts with "already set by user" when
    tensor overrides or layers are set, logs a warning and loads unfitted
    (common/fit.cpp).

    Args:
        cfg: Global config (for bin_dir).
        model: Path to the GGUF file.
        prof: Profile to apply.
        port: Resolved listen port.

    Returns:
        The argv list.
    """
    prof = effective_profile(prof)
    argv = [str(Path(cfg.bin_dir) / "llama-server"), "-m", model]
    for s in SETTINGS:
        if not s.flag:
            continue
        v = getattr(prof, s.key)
        if v == s.unset:
            continue
        if s.kind == Kind.BOOL:
            # v != unset, so a default-off BOOL is True here and a default-on one is False
            argv.append(s.flag if v else s.negate_flag)
        elif s.key == "port":
            argv += [s.flag, str(port)]
        else:
            argv += [s.flag, str(v)]
    return argv + shlex.split(prof.extra_args)


# What llama.cpp's preset INI parser accepts as a key, and the characters that end a value.
INI_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")
INI_VALUE_FORBIDDEN = ";#\n\r"
_NUMBER_RE = re.compile(r"-\d+(\.\d+)?\Z")


def check_ini_value(key: str, value: str) -> None:
    """Check that a value survives llama.cpp's preset INI parser unchanged.

    The parser ends a value at the first ``;`` or ``#`` (even with no space before it) and at
    a line break, and there is no quoting, so such values cannot be written.

    Args:
        key: Preset key, for the message.
        value: Value to write.

    Raises:
        ValueError: When the value contains ``;``, ``#`` or a line break.
    """
    bad = next((c for c in value if c in INI_VALUE_FORBIDDEN), "")
    if bad:
        shown = {"\n": "a line break", "\r": "a line break"}.get(bad, repr(bad))
        raise ValueError(f"{key}: {value!r} contains {shown}, which a gateway preset (INI) "
                         "cannot hold")


def _is_flag(token: str) -> bool:
    """True for a command-line flag token ("-c", "--jinja"), false for values like "-1"."""
    return token.startswith("-") and len(token) > 1 and not _NUMBER_RE.match(token)


def extra_args_items(extra_args: str) -> list[tuple[str, str]]:
    """Map a free-form extra_args string to preset INI key/value pairs.

    ``--ctx-checkpoints 4`` becomes ``("ctx-checkpoints", "4")``, ``--key=value`` the same,
    and a flag with no value (``--jinja``, ``--no-mmap``) becomes ``(key, "true")``; llama.cpp
    inverts a ``no-`` key itself. Leading dashes are stripped.

    Args:
        extra_args: The profile's extra_args text (shell syntax).

    Returns:
        Key/value pairs in order.

    Raises:
        ValueError: When the text is not valid shell syntax, holds a token with no flag in
            front of it, a key the INI cannot hold, or the same key twice.
    """
    try:
        tokens = shlex.split(extra_args)
    except ValueError as e:
        raise ValueError(f"extra_args: {e}") from None
    items: list[tuple[str, str]] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        i += 1
        if not _is_flag(tok):
            raise ValueError(f"extra_args: cannot express {tok!r} in a gateway preset: it is "
                             "not a flag or the value of one")
        key, eq, value = tok.lstrip("-").partition("=")
        if not eq:
            if i < len(tokens) and not _is_flag(tokens[i]):
                value, i = tokens[i], i + 1
            else:
                value = "true"
        if not INI_KEY_RE.match(key):
            raise ValueError(f"extra_args: cannot express {tok!r} in a gateway preset "
                             f"({key!r} is not a valid option name)")
        if any(k == key for k, _ in items):
            raise ValueError(f"extra_args: {tok!r} is given twice; a gateway preset holds each "
                             "option once")
        check_ini_value(key, value)
        items.append((key, value))
    return items


def preset_items(prof: Profile) -> list[tuple[str, str]]:
    """Render a profile as key/value pairs of a llama-server router preset (INI section).

    Uses the same registry as `build_command`: unset values are skipped, as are settings the
    router owns (host, port, api_key, alias) and settings with no preset key. Booleans are
    written ``true`` / ``false``: llama.cpp turns ``cont-batching = false`` into
    ``--no-cont-batching`` itself. extra_args is parsed into further pairs; an option named
    there replaces the registry's value for the same option. Like `build_command`, a
    manual tensor placement forces ``fit = off`` (see `effective_profile`).

    Args:
        prof: Profile to render.

    Returns:
        Key/value pairs in registry order, then those from extra_args.

    Raises:
        ValueError: When extra_args cannot be expressed as options, or a value holds a
            character the INI format cannot carry.
    """
    prof = effective_profile(prof)
    items: list[tuple[str, str]] = []
    for s in SETTINGS:
        if not s.preset_key or s.router_owned:
            continue
        v = getattr(prof, s.key)
        if v == s.unset:
            continue
        text = ("true" if v else "false") if s.kind == Kind.BOOL else str(v)
        check_ini_value(s.preset_key, text)
        items.append((s.preset_key, text))
    own = {s.preset_key: s.preset_key for s in SETTINGS if s.preset_key}
    own |= {s.flag.lstrip("-"): s.preset_key for s in SETTINGS if s.flag and s.preset_key}
    for key, value in extra_args_items(prof.extra_args):
        key = own.get(key, key)
        items = [kv for kv in items if kv[0] != key]
        items.append((key, value))
    return items


def command_string(argv: list[str]) -> str:
    """Render an argv as a shell-quoted string for the preview pane."""
    return shlex.join(argv)
