"""Config load/save: legacy shape, coercion, round-trip."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from llamactl import config
from llamactl.config import Config, Profile

LEGACY = {
    "bin_dir": "/bin",
    "model_dirs": ["/models"],
    "default_profile": {"ctx_size": 0, "gpu_layers": 99, "threads": 0, "parallel": 0,
                        "flash_attn": "auto", "port": 0, "host": "127.0.0.1", "extra_args": ""},
    "profiles": {"/models/a.gguf": {"ctx_size": 64000, "gpu_layers": 24, "threads": 0,
                                    "parallel": 0, "flash_attn": "auto", "port": 0,
                                    "host": "127.0.0.1", "extra_args": ""}},
}


@pytest.fixture
def cfg_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    return tmp_path


def test_legacy_loads(cfg_dir: Path) -> None:
    (cfg_dir / "config.json").write_text(json.dumps(LEGACY))
    cfg = Config.load()
    assert cfg.default_profile.gpu_layers == "all"      # legacy 99 -> "all"
    assert cfg.profiles["/models/a.gguf"].gpu_layers == "24"  # other ints -> str
    assert cfg.profiles["/models/a.gguf"].ctx_size == 64000


def test_from_dict_ignores_unknown() -> None:
    p = Profile.from_dict({"ctx_size": "8192", "bogus": 1, "temp": 0.5})
    assert p.ctx_size == 8192 and p.temp == 0.5 and not hasattr(p, "bogus")


def test_round_trip_and_version(cfg_dir: Path) -> None:
    cfg = Config(bin_dir="/b", profiles={"/m": Profile(ctx_size=4096)})
    cfg.save()
    raw = json.loads((cfg_dir / "config.json").read_text())
    assert raw["version"] == 2
    assert Config.load().profiles["/m"].ctx_size == 4096


def test_is_set_and_merged() -> None:
    p = Profile()
    assert not p.is_set("ctx_size") and not p.is_set("temp")
    p2 = p.merged(ctx_size=8192, top_k=0)
    assert p2.is_set("ctx_size") and p2.is_set("top_k") and p.ctx_size == 0


def test_has_profile() -> None:
    cfg = Config(profiles={"/m": Profile()})
    assert cfg.has_profile("/m") and not cfg.has_profile("/n")
