"""Build the llama-server / llama-cli command line from a profile."""
from __future__ import annotations

import shlex
from pathlib import Path

from .config import Config, Profile
from .schema import Kind, for_mode


def build_command(cfg: Config, model: str, prof: Profile, mode: str, port: int) -> list[str]:
    """Build the llama-server or llama-cli argv for a model + profile.

    Iterates the settings registry in order: values equal to their unset value
    are skipped, port uses the resolved port, extra_args is appended last.

    Args:
        cfg: Global config (for bin_dir).
        model: Path to the GGUF file.
        prof: Profile to apply.
        mode: "server" or "cli".
        port: Resolved listen port (server mode).

    Returns:
        The argv list.
    """
    argv = [str(Path(cfg.bin_dir) / f"llama-{mode}"), "-m", model]
    for s in for_mode(mode):
        if not s.flag:
            continue
        v = getattr(prof, s.key)
        if v == s.unset:
            continue
        if s.kind == Kind.BOOL:
            if s.key == "cont_batching":
                argv.append("-nocb")
            else:
                argv.append(s.flag)
        elif s.key == "port":
            argv += [s.flag, str(port)]
        else:
            argv += [s.flag, str(v)]
    return argv + shlex.split(prof.extra_args)


def command_string(argv: list[str]) -> str:
    """Render an argv as a shell-quoted string for the preview pane."""
    return shlex.join(argv)
