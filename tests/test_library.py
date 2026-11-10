"""Model discovery and name resolution."""
from __future__ import annotations

from pathlib import Path

import pytest

from llamactl import library
from llamactl.config import Config, Preset, Profile
from llamactl.library import discover_models, resolve_model

from conftest import gguf_blob


def _paths(*names: str) -> list[Path]:
    return [Path("/models") / n for n in names]


@pytest.fixture
def models() -> list[Path]:
    return _paths("Qwen3-8B-Q4_K_M.gguf", "Qwen3-14B-Q4_K_M.gguf", "llama-3.1-8b.gguf",
                  "Mistral-7B-Instruct.gguf")


def test_discover(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(library, "MIN_MODEL_BYTES", 10)
    sub = tmp_path / "sub"
    sub.mkdir()
    big = gguf_blob([]) + b"\0" * 64
    (tmp_path / "a.gguf").write_bytes(big)
    (sub / "b.gguf").write_bytes(big)
    (tmp_path / "tiny.gguf").write_bytes(b"x")                       # below the size floor
    (tmp_path / "vision-mmproj-f16.gguf").write_bytes(big)            # projector, not a model
    (tmp_path / "m-00001-of-00002.gguf").write_bytes(big)
    (tmp_path / "m-00002-of-00002.gguf").write_bytes(big)             # non-first shard
    (tmp_path / "notes.txt").write_bytes(big)
    found = discover_models([str(tmp_path), str(tmp_path / "missing")])
    assert [p.name for p in found] == ["a.gguf", "m-00001-of-00002.gguf", "b.gguf"]


def test_exact_alias_wins_over_stem_and_substring(models: list[Path]) -> None:
    cfg = Config()
    cfg.entry(str(models[2])).alias = "qwen3-8b"      # alias equals the prefix of another stem
    assert resolve_model("qwen3-8b", models, cfg) == models[2]


def test_exact_stem(models: list[Path]) -> None:
    assert resolve_model("Qwen3-14B-Q4_K_M", models, Config()) == models[1]


def test_derived_alias_matches_without_entry(models: list[Path]) -> None:
    cfg = Config()
    assert resolve_model("qwen3-14b-q4_k_m", models, cfg) == models[1]
    assert cfg.models == {}                            # resolution never mutates the config


def test_configured_alias(models: list[Path]) -> None:
    cfg = Config()
    cfg.entry(str(models[3])).alias = "tiny"
    assert resolve_model("tiny", models, cfg) == models[3]


def test_unique_substring_case_insensitive(models: list[Path]) -> None:
    assert resolve_model("MISTRAL", models, Config()) == models[3]
    assert resolve_model("3.1", models, Config()) == models[2]
    assert resolve_model("14b", models, Config()) == models[1]


def test_substring_matches_alias_too(models: list[Path]) -> None:
    cfg = Config()
    cfg.entry(str(models[3])).alias = "writer"
    assert resolve_model("writ", models, cfg) == models[3]


def test_ambiguous_lists_candidates(models: list[Path]) -> None:
    with pytest.raises(LookupError) as e:
        resolve_model("qwen3", models, Config())
    msg = str(e.value)
    assert "ambiguous" in msg and "qwen3-8b-q4_k_m" in msg and "qwen3-14b-q4_k_m" in msg
    assert "mistral" not in msg


def test_ambiguous_same_stem_in_two_dirs() -> None:
    ms = [Path("/a/m.gguf"), Path("/b/m.gguf")]
    cfg = Config()
    cfg.entry(str(ms[0])).alias, cfg.entry(str(ms[1])).alias = "x", "y"
    with pytest.raises(LookupError) as e:
        resolve_model("m", ms, cfg)
    assert "/a/m.gguf" in str(e.value) and "/b/m.gguf" in str(e.value)


def test_missing_lists_available(models: list[Path]) -> None:
    with pytest.raises(LookupError) as e:
        resolve_model("gemma", models, Config())
    msg = str(e.value)
    assert "no model matches 'gemma'" in msg and "mistral-7b-instruct" in msg


def test_missing_truncates_long_lists() -> None:
    ms = [Path(f"/m/model{i:02d}.gguf") for i in range(30)]
    with pytest.raises(LookupError) as e:
        resolve_model("zzz", ms, Config())
    assert "(18 more)" in str(e.value)


def test_no_models_and_empty_query() -> None:
    with pytest.raises(LookupError, match="no models found"):
        resolve_model("x", [], Config())
    with pytest.raises(LookupError, match="empty"):
        resolve_model("  ", _paths("a.gguf"), Config())


def test_aliases_avoid_configured_ones() -> None:
    ms = [Path("/a/m.gguf"), Path("/b/m.gguf")]
    cfg = Config()
    cfg.set_preset(str(ms[1]), Preset("custom", Profile()))      # /b/m.gguf owns alias "m"
    assert library.aliases_for(ms, cfg) == {ms[0]: "m-2", ms[1]: "m"}
    assert resolve_model("m", ms, cfg) == ms[1]
