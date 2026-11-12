"""`llamactl update`: the git status parse, the two plans and a no-change --check run."""
from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import llamactl
from llamactl import cli, update
from llamactl.config import Config


def fake_run(script: dict[str, tuple[int, str, str]],
             calls: list[list[str]] | None = None) -> Callable[..., Any]:
    """A `run` answering ``script["git -C /x rev-parse --short HEAD"] -> (rc, out, err)``.

    Args:
        script: Full argv string -> (returncode, stdout, stderr); absent -> (0, "", "").
        calls: When given, every argv is appended here.
    """
    def run(argv: list[str], **kw: Any) -> SimpleNamespace:
        key = " ".join(str(a) for a in argv)
        if calls is not None:
            calls.append([str(a) for a in argv])
        rc, out, err = script.get(key, (0, "", ""))
        return SimpleNamespace(returncode=rc, stdout=out, stderr=err)
    return run


def _repo_script(path: Path, *, head: str = "09338b85", branch: str = "master",
                 upstream: str = "origin/master", ahead: int = 0, behind: int = 0,
                 dirty: str = "") -> dict[str, tuple[int, str, str]]:
    """The git answers of a checkout at `path`, scripted for `fake_run`."""
    g = f"git -C {path}"
    return {
        f"{g} fetch --quiet": (0, "", ""),
        f"{g} rev-parse --short HEAD": (0, head + "\n", ""),
        f"{g} rev-parse --abbrev-ref HEAD": (0, branch + "\n", ""),
        f"{g} rev-parse --abbrev-ref --symbolic-full-name @{{u}}":
            (0, upstream + "\n", "") if upstream else (1, "", "no upstream"),
        f"{g} rev-list --left-right --count HEAD...@{{u}}": (0, f"{ahead}\t{behind}\n", ""),
        f"{g} status --porcelain --untracked-files=no": (0, dirty, ""),
    }


def test_llama_needs_rebuild_on_a_commit_mismatch_and_dirty_tree_stops_llamactl(
        tmp_path: Path) -> None:
    """behind=0 + a different built commit says "pulled but not rebuilt"; a dirty tree has
    no pull step."""
    llama = tmp_path / "llama.cpp"
    st = update.repo_status(llama, fake_run(_repo_script(llama)))
    assert (st.head, st.behind, st.dirty, st.error) == ("09338b85", 0, False, "")
    lines, steps, q = update.plan_llama(st, llama / "build", "5fc4f3c", serving=False)
    assert "pulled but not rebuilt: build needed" in lines[0]
    assert [s.argv[:2] for s in steps] == [["cmake", "--build"]] and steps[0].stream
    assert q == "Pull and rebuild llama.cpp?"

    src = tmp_path / "llamactl-src"
    st = update.repo_status(src, fake_run(_repo_script(src, behind=3,
                                                       dirty=" M llamactl/cli.py\n")))
    assert st.dirty
    lines, steps, q = update.plan_llamactl(st, installed_by_uv=True)
    assert "local changes" in lines[0] and steps == [] and q == ""


def test_update_check_reports_and_runs_nothing(isolated: SimpleNamespace,
                                               tmp_path: Path) -> None:
    """`update --check` prints both parts' reports; no pull, install or build is run."""
    repo = Path(llamactl.__file__).resolve().parents[1]
    script = {"uv tool dir": (1, "", "no such command")}
    script.update(_repo_script(repo, behind=3))
    calls: list[list[str]] = []
    out, err = io.StringIO(), io.StringIO()
    sv = cli.Services(Config(bin_dir=str(tmp_path / "bin")), out=out, err=err,
                      run=fake_run(script, calls), stream=lambda argv: 0,
                      confirm=lambda q: True, tty=lambda: True)
    code = cli.main(["update", "--check"], sv)
    assert code == 0
    text = out.getvalue()
    assert f"llamactl ({repo}" in text and "3 new commits" in text
    assert "llama.cpp" in text and "no source checkout" in text
    used = " ".join(" ".join(a) for a in calls)
    assert " pull " not in used and "uv tool install" not in used and "cmake" not in used
