"""``llamactl update``: report what is pending, ask, then pull and rebuild both checkouts.

Two parts: llamactl itself (the local clone it was installed from, by ``uv tool`` or by
living in a checkout) and llama.cpp (the source tree the configured ``bin_dir`` was built
from, found through its ``CMakeCache.txt``). Every command goes through an injected `run`
(``subprocess.run`` compatible) or `stream` (a command whose output goes to the terminal);
tests script both and never touch a real checkout.
"""
from __future__ import annotations

import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple

import llamactl
from .support import rebuild_commands
from .tuner import parse_build

Run = Callable[..., "subprocess.CompletedProcess[str]"]
Stream = Callable[[list[str]], int]
Confirm = Callable[[str], bool]

_COMMIT_RE = re.compile(r"commit\s+([0-9a-f]+)", re.IGNORECASE)


class Step(NamedTuple):
    """One command of an update plan; `stream` ones run with output on the terminal."""

    argv: list[str]
    stream: bool = False


@dataclass(frozen=True)
class RepoStatus:
    """What a git checkout looks like against its upstream.

    Attributes:
        path: The working tree.
        head: Short sha of HEAD ("" when unreadable).
        branch: Current branch ("" when detached or unreadable).
        upstream: Tracking ref ("" when the branch has none).
        ahead: Commits on HEAD the upstream lacks.
        behind: Commits on the upstream HEAD lacks.
        dirty: Tracked files changed (untracked files do not count).
        error: Why the fetch failed, "" when it ran or was skipped.
    """

    path: Path
    head: str = ""
    branch: str = ""
    upstream: str = ""
    ahead: int = 0
    behind: int = 0
    dirty: bool = False
    error: str = ""


def _tail(r: subprocess.CompletedProcess) -> str:
    """The last non-empty line of a failed command's stderr, else stdout."""
    for text in (r.stderr, r.stdout):
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            return lines[-1]
    return f"exit {r.returncode}"


def repo_status(path: Path, run: Run, fetch: bool = True) -> RepoStatus:
    """Describe a checkout: fetch, then HEAD, branch, upstream, ahead/behind and dirty.

    Args:
        path: The working tree.
        run: ``subprocess.run`` compatible callable (called with capture text).
        fetch: Run ``git fetch --quiet`` first (a failure keeps local fields but sets
            `error`).

    Returns:
        The status; fields a command could not answer stay at their defaults.
    """
    error = ""
    if fetch:
        r = run(["git", "-C", str(path), "fetch", "--quiet"], capture_output=True, text=True)
        if r.returncode != 0:
            error = f"could not fetch: {_tail(r)}"

    def out(*args: str) -> str:
        r = run(["git", "-C", str(path), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ""

    head = out("rev-parse", "--short", "HEAD")
    branch = out("rev-parse", "--abbrev-ref", "HEAD")
    upstream = out("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    ahead = behind = 0
    if upstream:
        r = run(["git", "-C", str(path), "rev-list", "--left-right", "--count", "HEAD...@{u}"],
                capture_output=True, text=True)
        if r.returncode == 0:
            a, _, b = r.stdout.partition("\t")
            ahead, behind = int(a or 0), int(b.strip() or 0)
    r = run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True)
    dirty = bool(r.stdout.strip()) if r.returncode == 0 else False
    return RepoStatus(Path(path), head, branch, upstream, ahead, behind, dirty, error)


def same_commit(a: str, b: str) -> bool:
    """Two shas name the same commit when one is a prefix of the other (either direction)."""
    return bool(a and b) and (a.startswith(b) or b.startswith(a))


def built_commit(version: str) -> str:
    """The commit sha in an ``llama-server --version`` string ("" when absent).

    Args:
        version: E.g. ``0.9.0-dev (build 975, commit 09338b85)``.
    """
    m = _COMMIT_RE.search(version)
    return m.group(1) if m else ""


def llamactl_source(run: Run) -> tuple[Path | None, bool]:
    """The local clone llamactl was installed from, if any.

    First the uv tool receipt (``uv tool dir`` -> ``llama-tui/uv-receipt.toml`` -> the
    ``directory`` requirement named llama-tui); then the imported package itself when it
    lives inside a checkout (``.git`` and ``pyproject.toml`` two levels up).

    Args:
        run: ``subprocess.run`` compatible callable; a missing/failing ``uv`` is skipped.

    Returns:
        (source directory, installed_by_uv); (None, False) for other install kinds.
    """
    try:
        r = run(["uv", "tool", "dir"], capture_output=True, text=True)
    except OSError:
        r = None
    if r is not None and r.returncode == 0:
        receipt = Path(r.stdout.strip()) / "llama-tui" / "uv-receipt.toml"
        try:
            data = tomllib.loads(receipt.read_text())
        except (OSError, ValueError):
            data = {}
        reqs = data.get("tool", {}).get("requirements", data.get("requirements", []))
        for req in reqs:
            if req.get("name") == "llama-tui" and req.get("directory"):
                d = Path(req["directory"]).expanduser()
                if (d / ".git").is_dir():
                    return d, True
    pkg = Path(llamactl.__file__).resolve().parents[1]
    if (pkg / ".git").is_dir() and (pkg / "pyproject.toml").is_file():
        return pkg, False
    return None, False


def llama_source(bin_dir: str) -> tuple[Path, Path] | None:
    """(source tree, build dir) the configured ``llama-server`` was built in.

    The build dir is `bin_dir`'s parent when it holds ``CMakeCache.txt``; the source dir
    comes from its ``CMAKE_HOME_DIRECTORY`` and must be a checkout.

    Args:
        bin_dir: The configured llama.cpp binary folder.

    Returns:
        (src, build_dir), or None when that layout does not name the source.
    """
    build_dir = Path(bin_dir).expanduser().parent
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return None
    src = None
    try:
        for line in cache.read_text(errors="replace").splitlines():
            if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
                src = Path(line.split("=", 1)[1])
                break
    except OSError:
        return None
    if src is None or not (src / ".git").is_dir():
        return None
    return src, build_dir


def _repo_line(st: RepoStatus, name: str, where: str) -> tuple[str, str]:
    """(report line, outcome) for a checkout; outcome is ``pull``, ``ok`` or ``stop``.

    ``stop`` covers dirty, diverged and upstream-less checkouts: a human must act first.
    A failed fetch is appended to the line but does not hide the local state.
    """
    suffix = f"; {st.error}" if st.error else ""
    head = f"{name} ({where})"
    if st.dirty:
        return f"{head}: local changes: commit or stash them first{suffix}", "stop"
    if not st.upstream:
        return f"{head}: no upstream branch{suffix}", "stop"
    if st.ahead and st.behind:
        return (f"{head}: diverged ({st.ahead} ahead, {st.behind} behind): pull it "
                f"yourself{suffix}", "stop")
    if st.behind:
        return f"{head}: {st.behind} new commit{'s' * (st.behind != 1)}{suffix}", "pull"
    return f"{head}: up to date{suffix}", "ok"


def plan_llamactl(st: RepoStatus | None, installed_by_uv: bool) -> tuple[list[str], list[Step], str]:
    """The report lines, command steps and prompt of the llamactl part.

    Args:
        st: Status of the clone llamactl was installed from, or None for other installs.
        installed_by_uv: Reinstall with ``uv tool install --reinstall`` after the pull.

    Returns:
        (lines to print, steps to run, the [y/N] question; "" when there is nothing to do).
    """
    if st is None:
        return (["llamactl was not installed from a local clone; "
                 "update it the way you installed it"], [], "")
    line, outcome = _repo_line(st, "llamactl", f"{st.path}, {st.branch} -> {st.upstream}")
    if outcome != "pull":
        return [line], [], ""
    steps = [Step(["git", "-C", str(st.path), "pull", "--ff-only"])]
    if installed_by_uv:
        steps.append(Step(["uv", "tool", "install", "--reinstall", str(st.path)]))
    return [line], steps, "Update llamactl?"


def plan_llama(st: RepoStatus, build_dir: Path, built: str,
               serving: bool) -> tuple[list[str], list[Step], str]:
    """The report lines, command steps and prompt of the llama.cpp part.

    A rebuild is planned when the checkout is behind (it gets pulled first) or the built
    binary's commit does not match HEAD ("pulled but not rebuilt"). The build runs
    ``cmake --build`` in the existing build dir, keeping the configured flags (CUDA etc.).

    Args:
        st: Status of the llama.cpp checkout.
        build_dir: Its configured build directory.
        built: Commit sha of the installed binary ("" when unreadable).
        serving: A gateway or standalone session is running; steps are refused then.

    Returns:
        (lines to print, steps to run, the [y/N] question; "" when there is nothing to do).
    """
    line, outcome = _repo_line(st, "llama.cpp", str(st.path))
    if outcome == "stop":
        return [line], [], ""
    head = f"llama.cpp ({st.path})"
    needs_build = (bool(built) and not same_commit(built, st.head)) or st.behind > 0
    if not needs_build:
        return [line], [], ""
    if serving:
        return [f"{head}: serving is running; stop it first (llamactl down)"], [], ""
    steps = [Step(["git", "-C", str(st.path), "pull", "--ff-only"])] if st.behind else []
    steps.append(Step(["cmake", "--build", str(build_dir), "--config", "Release", "-j"],
                      stream=True))
    line = (f"{line} (rebuild after pull)" if outcome == "pull"
            else f"{head}: pulled but not rebuilt: build needed (built {built}, HEAD {st.head})")
    return [line], steps, "Pull and rebuild llama.cpp?"


def apply_steps(run: Run, stream: Stream, steps: list[Step],
                say: Callable[[str], None], note: Callable[[str], None]) -> bool:
    """Run the planned steps in order, stopping at the first failure.

    Args:
        run: ``subprocess.run`` compatible callable for quick commands.
        stream: Runs a command with output on the terminal (the build).
        steps: The plan's steps.
        say: Result-stream printer (the commands, echoed as ``$ ...``).
        note: Message-stream printer (failures with the stderr tail).

    Returns:
        True when every step ran (or there were none).
    """
    for s in steps:
        say(f"  $ {' '.join(s.argv)}")
        if s.stream:
            rc = stream(s.argv)
            if rc != 0:
                note(f"  failed (exit {rc}): {' '.join(s.argv)}")
                return False
        else:
            r = run(s.argv, capture_output=True, text=True)
            if r.returncode != 0:
                note(f"  failed (exit {r.returncode}): {' '.join(s.argv)}: {_tail(r)}")
                return False
    return True


def new_version(run: Run, bin_dir: str) -> str:
    """The ``llama-server --version`` of `bin_dir` right now ("" when unreadable).

    Args:
        run: ``subprocess.run`` compatible callable.
        bin_dir: The configured llama.cpp binary folder.
    """
    binary = Path(bin_dir).expanduser() / "llama-server"
    try:
        r = run([str(binary), "--version"], capture_output=True, text=True)
    except OSError:
        return ""
    return parse_build(r.stdout + r.stderr) if r.returncode == 0 else ""


def no_source_line(bin_dir: str) -> str:
    """The llama.cpp part's line when the source tree cannot be found."""
    return f"llama.cpp: no source checkout found for {bin_dir}: {rebuild_commands(bin_dir)}"
