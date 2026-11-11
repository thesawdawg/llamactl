"""Config v3: round trip, v2 migration and backup, presets, aliases."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from llamactl import config
from llamactl.config import (Config, GatewaySettings, ModelEntry, Preset, Profile,
                             derive_alias)
from llamactl.measure import Measurement

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


def _v2(cfg_dir: Path, **extra: object) -> str:
    text = json.dumps({"version": 2, **LEGACY, "headroom_pct": 15, **extra}, indent=2)
    (cfg_dir / "config.json").write_text(text)
    return text


def test_legacy_loads(cfg_dir: Path) -> None:
    (cfg_dir / "config.json").write_text(json.dumps(LEGACY))   # no version key at all
    cfg = Config.load()
    assert cfg.default_profile.gpu_layers == "all"      # legacy 99 -> "all"
    prof = cfg.profile_for("/models/a.gguf")
    assert prof.gpu_layers == "24" and prof.ctx_size == 64000   # other ints -> str


def test_from_dict_ignores_unknown() -> None:
    p = Profile.from_dict({"ctx_size": "8192", "bogus": 1, "temp": 0.5})
    assert p.ctx_size == 8192 and p.temp == 0.5 and not hasattr(p, "bogus")


def test_defaults() -> None:
    cfg = Config()
    assert cfg.models == {} and cfg.default_intent == "chat" and cfg.onboarded is False
    assert cfg.gateway == GatewaySettings("127.0.0.1", 8080, "", 1, 600, False)


def test_v3_round_trip(cfg_dir: Path) -> None:
    cfg = Config(bin_dir="/b", default_intent="agent", onboarded=True,
                 gateway=GatewaySettings(host="0.0.0.0", port=9000, api_key="k", models_max=2,
                                         sleep_idle=0, autostart=True))
    cfg.set_preset("/m/a.gguf", Preset("agent", Profile(ctx_size=65536, fit="off"), "tuned",
                                       Measurement(ok=True, gen_tps=41.5, vram_used=9_000_000_000,
                                                   at="2026-11-09T10:00:00+00:00", build="b1")))
    cfg.set_preset("/m/a.gguf", Preset("chat", Profile(ctx_size=8192), "estimated"), activate=False)
    cfg.entry("/m/a.gguf").alias = "my-a"
    cfg.save()
    raw = json.loads((cfg_dir / "config.json").read_text())
    assert raw["version"] == 3
    assert list(raw) == ["version", "bin_dir", "model_dirs", "headroom_pct", "gauge_interval",
                         "default_profile", "default_intent", "models", "gateway", "onboarded"]
    assert raw["models"]["/m/a.gguf"]["active"] == "agent"
    assert raw["models"]["/m/a.gguf"]["presets"]["agent"]["measured"]["gen_tps"] == 41.5
    assert not (cfg_dir / "config.v2.bak.json").exists()   # nothing to back up

    back = Config.load()
    assert back == cfg
    ent = back.models["/m/a.gguf"]
    assert ent.alias == "my-a" and ent.active == "agent" and set(ent.presets) == {"agent", "chat"}
    assert ent.presets["agent"].measured == Measurement(
        ok=True, gen_tps=41.5, vram_used=9_000_000_000, at="2026-11-09T10:00:00+00:00", build="b1")
    assert ent.presets["chat"].measured is None
    assert back.gateway.api_key == "k" and back.gateway.autostart is True
    assert back.onboarded is True and back.default_intent == "agent"


def test_unknown_and_bad_values_tolerated(cfg_dir: Path) -> None:
    (cfg_dir / "config.json").write_text(json.dumps({
        "version": 3, "mystery": 1,
        "gateway": {"port": "9001", "models_max": "x", "autostart": "true", "extra": 1},
        "models": {"/m/a.gguf": {"presets": {"edit": {"profile": {"ctx_size": "4096"}, "new": 1,
                                                       "measured": "nope"}}, "zzz": 2},
                   "/m/b.gguf": "garbage"},
        "onboarded": "yes"}))
    cfg = Config.load()
    assert cfg.gateway.port == 9001 and cfg.gateway.models_max == 1 and cfg.gateway.autostart
    ent = cfg.models["/m/a.gguf"]
    assert ent.alias == "a"                       # missing alias is derived
    p = ent.presets["edit"]
    assert (p.intent, p.source, p.measured, p.profile.ctx_size) == ("edit", "edited", None, 4096)
    assert "/m/b.gguf" not in cfg.models and cfg.onboarded is True


def test_v2_migration(cfg_dir: Path) -> None:
    _v2(cfg_dir)
    cfg = Config.load()
    ent = cfg.models["/models/a.gguf"]
    assert ent.active == "custom" and list(ent.presets) == ["custom"]
    pre = ent.presets["custom"]
    assert (pre.intent, pre.source, pre.measured) == ("custom", "edited", None)
    assert pre.profile.ctx_size == 64000 and pre.profile.gpu_layers == "24"
    assert ent.alias == "a"
    assert cfg.headroom_pct == 15 and cfg.bin_dir == "/bin" and cfg.model_dirs == ["/models"]
    assert cfg.profile_for("/models/a.gguf").ctx_size == 64000
    assert cfg.has_profile("/models/a.gguf") and not cfg.has_profile("/models/b.gguf")
    assert cfg.profile_for("/models/b.gguf") is cfg.default_profile


def test_v2_backup_made_once_before_first_v3_save(cfg_dir: Path) -> None:
    original = _v2(cfg_dir)
    backup = cfg_dir / "config.v2.bak.json"
    cfg = Config.load()
    assert not backup.exists()                      # loading never writes
    cfg.save()
    assert backup.read_text() == original
    assert json.loads((cfg_dir / "config.json").read_text())["version"] == 3
    cfg.headroom_pct = 20
    cfg.save()                                      # second save leaves the backup alone
    assert backup.read_text() == original
    assert json.loads((cfg_dir / "config.json").read_text())["headroom_pct"] == 20


def test_unversioned_and_corrupt_files_are_backed_up(cfg_dir: Path) -> None:
    (cfg_dir / "config.json").write_text("{not json")
    Config().save()
    assert (cfg_dir / "config.v2.bak.json").read_text() == "{not json"


def test_first_ever_save_makes_no_backup(cfg_dir: Path) -> None:
    Config().save()
    assert not (cfg_dir / "config.v2.bak.json").exists()


def test_roundtrip_of_migrated_config_is_stable(cfg_dir: Path) -> None:
    _v2(cfg_dir)
    first = Config.load()
    first.save()
    assert Config.load() == first


def test_is_set_and_merged() -> None:
    p = Profile()
    assert not p.is_set("ctx_size") and not p.is_set("temp")
    p2 = p.merged(ctx_size=8192, top_k=0)
    assert p2.is_set("ctx_size") and p2.is_set("top_k") and p.ctx_size == 0


def test_preset_helpers() -> None:
    cfg = Config()
    m = "/m/x.gguf"
    assert cfg.preset(m) is None and not cfg.has_profile(m) and m not in cfg.models
    assert cfg.profile_for(m) is cfg.default_profile
    agent = Preset("agent", Profile(ctx_size=65536), "tuned")
    chat = Preset("chat", Profile(ctx_size=8192), "estimated")
    cfg.set_preset(m, agent)
    cfg.set_preset(m, chat, activate=False)
    assert cfg.preset(m) is agent and cfg.preset(m, "chat") is chat and cfg.preset(m, "nope") is None
    assert cfg.profile_for(m).ctx_size == 65536 and cfg.has_profile(m)
    cfg.set_preset(m, chat)                          # activate
    assert cfg.entry(m).active == "chat"
    assert cfg.delete_preset(m, "chat") is True
    assert cfg.entry(m).active == "agent"            # falls back to a remaining preset
    assert cfg.delete_preset(m, "chat") is False and cfg.delete_preset("/none", "x") is False
    assert cfg.delete_preset(m, "agent") and cfg.entry(m).active == ""
    assert not cfg.has_profile(m) and cfg.profile_for(m) is cfg.default_profile
    assert m in cfg.models                           # the entry (and its alias) stays


def test_first_preset_becomes_active_even_without_activate() -> None:
    cfg = Config()
    cfg.set_preset("/m/x.gguf", Preset("chat"), activate=False)
    assert cfg.entry("/m/x.gguf").active == "chat"


@pytest.mark.parametrize("path,alias", [
    ("/m/Qwen3-8B-Q4_K_M.gguf", "qwen3-8b-q4_k_m"),
    ("/m/My Model (v2) [final].gguf", "my-model--v2---final-"),
    ("/m/ünï.gguf", "-n-"),
    ("/m/big-00001-of-00003.gguf", "big"),
    ("/m/a.b_c-d.gguf", "a.b_c-d"),
])
def test_derive_alias(path: str, alias: str) -> None:
    assert derive_alias(path, set()) == alias


def test_alias_uniqueness() -> None:
    assert derive_alias("/a/m.gguf", {"m"}) == "m-2"
    assert derive_alias("/a/m.gguf", {"m", "m-2"}) == "m-3"
    cfg = Config()
    aliases = [cfg.entry(p).alias for p in ("/a/m.gguf", "/b/M.gguf", "/c/m.gguf")]
    assert aliases == ["m", "m-2", "m-3"]
    assert cfg.entry("/b/M.gguf").alias == "m-2"     # stable on repeat calls
    assert isinstance(cfg.entry("/a/m.gguf"), ModelEntry)


def test_migrated_aliases_are_unique(cfg_dir: Path) -> None:
    profiles = {f"/{d}/m.gguf": {"ctx_size": 1} for d in "abc"}
    (cfg_dir / "config.json").write_text(json.dumps({"version": 2, "profiles": profiles}))
    cfg = Config.load()
    assert sorted(e.alias for e in cfg.models.values()) == ["m", "m-2", "m-3"]


def test_legacy_dict_measured_still_loads(cfg_dir: Path) -> None:
    """A measured dict from before Measurement existed (extra keys, missing keys) loads."""
    (cfg_dir / "config.json").write_text(json.dumps({"version": 3, "models": {"/m/a.gguf": {
        "active": "agent", "presets": {
            "agent": {"intent": "agent", "source": "tuned",
                      "measured": {"ok": True, "gen_tps": "41.5", "nested": {"a": 1}}},
            "chat": {"intent": "chat", "measured": None}}}}}))
    ent = Config.load().models["/m/a.gguf"]
    m = ent.presets["agent"].measured
    assert m == Measurement(ok=True, gen_tps=41.5)
    assert ent.presets["chat"].measured is None
    cfg = Config.load()
    cfg.save()
    assert json.loads((cfg_dir / "config.json").read_text())["models"]["/m/a.gguf"][
        "presets"]["agent"]["measured"]["gen_tps"] == 41.5


# ---------------------------------------------------------------------------- state directory

def test_state_dir_prefers_llamactl_home(tmp_path: Path) -> None:
    env = {"LLAMACTL_HOME": str(tmp_path / "mine"), "XDG_CONFIG_HOME": str(tmp_path / "xdg")}
    assert config.state_dir(env, tmp_path / "home") == tmp_path / "mine"


def test_state_dir_falls_back_to_xdg_config_home(tmp_path: Path) -> None:
    env = {"XDG_CONFIG_HOME": str(tmp_path / "xdg")}
    assert config.state_dir(env, tmp_path / "home") == tmp_path / "xdg" / "llamactl"


def test_state_dir_defaults_to_dot_config_and_ignores_empty_variables(tmp_path: Path) -> None:
    default = tmp_path / "home" / ".config" / "llamactl"
    assert config.state_dir({}, tmp_path / "home") == default
    assert config.state_dir({"LLAMACTL_HOME": " ", "XDG_CONFIG_HOME": ""}, tmp_path / "home") == default


def test_state_dir_expands_a_tilde() -> None:
    assert config.state_dir({"LLAMACTL_HOME": "~/state"}, Path("/x")) == Path.home() / "state"


def test_session_default_state_dir_is_a_temp_dir_not_the_real_one() -> None:
    """The conftest session fixture keeps un-isolated tests away from ~/.config/llamactl."""
    assert config.CONFIG_DIR != Path.home() / ".config" / "llamactl"
    assert config.CONFIG_DIR.name.startswith("llamactl-test-home-")
    from llamactl import calibration, gateway, gguf, hf, sessions
    for mod, name in ((gguf, "CACHE_FILE"), (calibration, "TUNING_FILE"), (gateway, "STATE_FILE"),
                      (gateway, "PRESETS_FILE"), (sessions, "STATE_FILE"), (hf, "TOKEN_FILE")):
        assert getattr(mod, name).parent == config.CONFIG_DIR, f"{mod.__name__}.{name}"
    assert config.LOG_DIR.parent == config.CONFIG_DIR


# ---------------------------------------------------------------------------- onboarded

def test_v2_config_with_presets_counts_as_onboarded_without_a_save(cfg_dir: Path) -> None:
    text = _v2(cfg_dir)
    cfg = Config.load()
    assert cfg.onboarded
    assert (cfg_dir / "config.json").read_text() == text and not (cfg_dir / config.BACKUP_NAME).exists()


def test_v2_config_with_custom_dirs_only_counts_as_onboarded(cfg_dir: Path) -> None:
    (cfg_dir / "config.json").write_text(json.dumps({"version": 2, "model_dirs": ["/elsewhere"]}))
    assert Config.load().onboarded
    (cfg_dir / "config.json").write_text(json.dumps({"version": 2, "bin_dir": "/opt/llama/bin"}))
    assert Config.load().onboarded


def test_v2_config_with_only_defaults_is_not_onboarded(cfg_dir: Path) -> None:
    (cfg_dir / "config.json").write_text(json.dumps({"version": 2, "headroom_pct": 12}))
    assert not Config.load().onboarded


def test_v3_config_keeps_its_own_flag(cfg_dir: Path) -> None:
    (cfg_dir / "config.json").write_text(json.dumps(
        {"version": 3, "model_dirs": ["/elsewhere"], "onboarded": False}))
    assert not Config.load().onboarded
    (cfg_dir / "config.json").write_text(json.dumps({"version": 3, "onboarded": True}))
    assert Config.load().onboarded
    assert not Config().onboarded and not Config.from_dict({}).onboarded
