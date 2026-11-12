"""Architecture support: the build check on fake library bytes, learned failures, parsing."""
from __future__ import annotations

import json
from pathlib import Path

from llamactl.support import (ArchSupport, Support, SupportChecker, parse_upstream,
                              rebuild_commands, unknown_arch)

BLOB = b"...modern-bert\0qwen35\0gemma3\0clip\0..."
LOG_LINE = ("llama_model_load: error loading model: unknown model architecture: 'clef'"
            "\nsome other line")


def _checker(tmp_path: Path, blob: bytes | None = BLOB,
             names: list[str] | None = None) -> SupportChecker:
    """A checker over a fake libllama.so in `tmp_path`; `names` is the upstream list."""
    (tmp_path / "libllama.so").write_bytes(blob or b"")
    check = SupportChecker(str(tmp_path), state_dir=tmp_path)
    if names is not None:
        check._upstream = {"fetched": 1.0, "names": names}
    return check


def test_build_check_and_upstream(tmp_path: Path) -> None:
    """bert is found as the tail of modern-bert; upstream splits update from unsupported."""
    check = _checker(tmp_path, names=["clef", "qwen35", "modern-bert"])
    assert check.check("bert").status == Support.OK                 # tail match
    assert check.check("qwen35").status == Support.OK
    assert check.check("clef").status == Support.UPDATE             # missing, in master
    assert check.check("foo").status == Support.UNSUPPORTED         # in neither
    assert check.check("clip").status == Support.NOT_LLM            # table wins over the blob


def test_unknown_upstream_means_update(tmp_path: Path) -> None:
    """A missing name stays `update` (never `unsupported`) while master is unknown."""
    check = _checker(tmp_path)
    s = check.check("clef")
    assert s.status == Support.UPDATE and s.blocks
    assert "a newer build may support it" in s.reason()


def test_no_binary_is_unknown(tmp_path: Path) -> None:
    """Neither libllama.so nor llama-server: nothing could be checked, nothing blocks."""
    check = SupportChecker(str(tmp_path / "nope"), state_dir=tmp_path)
    assert check.check("qwen35").status == Support.UNKNOWN
    assert not check.check("").blocks


def test_unknown_arch_parses_the_real_log_line() -> None:
    """The name is taken from ``unknown model architecture: '<x>'`` anywhere in the log."""
    assert unknown_arch(LOG_LINE) == "clef"
    assert unknown_arch("all is well") is None


def test_record_failure_overrides_ok_for_that_build_only(tmp_path: Path) -> None:
    """A learned failure beats the build check on the same build, not on another build."""
    check = _checker(tmp_path, names=["qwen35"])
    assert check.check("qwen35", "b1").status == Support.OK
    check.record_failure("qwen35", "b1")
    assert check.check("qwen35", "b1").status == Support.UPDATE      # in master: update
    assert check.check("qwen35", "b2").status == Support.OK          # different build: cleared
    assert json.loads((tmp_path / "arch_support.json").read_text()) == {
        "build": "b1", "unknown": ["qwen35"]}


def test_parse_upstream_drops_unknown() -> None:
    """The LLM_ARCH_NAMES table is parsed; the LLM_ARCH_UNKNOWN row is excluded."""
    src = ('static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {\n'
           '    { LLM_ARCH_LLAMA,            "llama"            },\n'
           '    { LLM_ARCH_CLEF,             "clef"             },\n'
           '    { LLM_ARCH_UNKNOWN,          "(unknown)"        },\n'
           '};\n')
    assert parse_upstream(src) == frozenset({"llama", "clef"})


def test_rebuild_commands(tmp_path: Path) -> None:
    """A source tree two levels up gives git pull + cmake; anything else the docs pointer."""
    src = tmp_path / "src"
    (src / "build" / "bin").mkdir(parents=True)
    (src / "CMakeLists.txt").write_text("cmake_minimum_required()")
    cmd = rebuild_commands(str(src / "build" / "bin"))
    assert cmd == f"cd {src} && git pull && cmake --build build --config Release -j"
    assert "docs/build.md" in rebuild_commands(str(tmp_path / "bin"))
