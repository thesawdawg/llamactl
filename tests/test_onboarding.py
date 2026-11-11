"""The first-run flow's logic: location, hardware sentence, scan summary, choices, finish."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from fakes import make_host, make_info
from llamactl import onboarding as ob
from llamactl.config import Config, Preset, Profile
from llamactl.hardware import Device, Host
from llamactl.intents import by_key


def bin_with_server(tmp_path: Path, name: str = "bin") -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "llama-server").touch()
    return d


# ---------------------------------------------------------------------------- step 1

def test_check_bin_dir_messages(tmp_path: Path) -> None:
    good = bin_with_server(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert ob.check_bin_dir(str(good)) == ""
    assert "holds llama-server" in ob.check_bin_dir("  ")
    assert "not a folder" in ob.check_bin_dir(str(tmp_path / "nope"))
    assert "no llama-server inside" in ob.check_bin_dir(str(empty))


def test_locate_prefers_the_configured_folder_then_path(tmp_path: Path) -> None:
    cfg_bin = bin_with_server(tmp_path, "cfg")
    path_bin = bin_with_server(tmp_path, "onpath")
    which = lambda name: str(path_bin / name)          # noqa: E731
    loc = ob.locate_llama(str(cfg_bin), which)
    assert (loc.bin_dir, loc.source, loc.found) == (str(cfg_bin), ob.SRC_CONFIG, True)
    assert loc.binary == str(cfg_bin / "llama-server")
    loc = ob.locate_llama(str(tmp_path / "missing"), which)
    assert (loc.bin_dir, loc.source) == (str(path_bin), ob.SRC_PATH)
    none = ob.locate_llama("", lambda name: None)
    assert not none.found and none.source == ob.SRC_NONE and none.binary == ""


def test_llama_text_for_found_and_missing(tmp_path: Path) -> None:
    d = bin_with_server(tmp_path)
    found = ob.llama_text(ob.LlamaLocation(str(d), ob.SRC_CONFIG), "0.5.0-dev (build 1)")
    assert str(d / "llama-server") in found and "0.5.0-dev (build 1)" in found
    assert "reading the version" in ob.llama_text(ob.LlamaLocation(str(d), ob.SRC_PATH), "")
    missing = ob.llama_text(ob.LlamaLocation(), "")
    assert "not found" in missing and "github.com/ggml-org/llama.cpp" in missing


# ---------------------------------------------------------------------------- step 2

def test_hardware_sentence_for_a_16gb_gpu() -> None:
    s = ob.hardware_sentence(make_host(16e9), by_key["chat"])
    assert s.startswith("With GPU (16.0 GB VRAM)")
    assert "Chat intent's 16k context" in s
    assert "spill into your 24.0 GB of RAM" in s
    room = ob.fit_bytes(make_host(16e9), by_key["chat"], 10)
    assert 11e9 < room < 13e9 and f"{room / 1e9:.1f} GB" in s


def test_bigger_context_intents_leave_less_room() -> None:
    host = make_host(16e9)
    assert ob.fit_bytes(host, by_key["agent"], 10) < ob.fit_bytes(host, by_key["chat"], 10)
    # "trained context" intents assume 32k
    assert ob.intent_ctx(by_key["context"]) == 32768
    assert ob.fit_bytes(host, by_key["speed"], 10) > ob.fit_bytes(host, by_key["chat"], 10)


def test_hardware_sentence_without_a_gpu_uses_ram() -> None:
    host = Host(ram_total=int(32e9), ram_free=int(30e9), cpu_threads=8, cpu_cores=4)
    s = ob.hardware_sentence(host, by_key["chat"])
    assert s.startswith("No GPU was found, so models run on the CPU")
    assert "32.0 GB of RAM" in s and "few tokens per second" in s


def test_hardware_sentence_tiny_gpu_and_multi_gpu_and_unified_and_unknown() -> None:
    tiny = make_host(1e9)
    assert "too small to hold even a small model" in ob.hardware_sentence(tiny, by_key["chat"])
    two = Host(gpus=[Device.of("CUDA0", "A", int(12e9), int(12e9)),
                     Device.of("CUDA1", "B", int(12e9), int(12e9))], ram_total=int(32e9),
               ram_free=int(30e9))
    s = ob.hardware_sentence(two, by_key["chat"])
    assert s.startswith("With 2 GPUs (24.0 GB VRAM)")
    uni = Host(gpus=[Device.of("Metal", "M3", int(32e9), int(32e9))], ram_total=int(32e9),
               ram_free=int(20e9), unified=True)
    s = ob.hardware_sentence(uni, by_key["chat"])
    assert "unified memory" in s and "spill" not in s
    assert ob.hardware_sentence(Host(), by_key["chat"]) == "Still detecting the hardware..."


# ---------------------------------------------------------------------------- step 3

def test_scan_summary_counts_sizes_and_unreadable(tmp_path: Path) -> None:
    a, b, c = tmp_path / "a.gguf", tmp_path / "b.gguf", tmp_path / "c.gguf"
    c.write_bytes(b"x" * 1000)
    infos = {str(a): make_info(a, int(4e9)), str(b): make_info(b, int(6e9))}
    s = ob.scan_summary([a, b, c], infos)
    assert (s.count, s.unreadable, s.total) == (3, 1, int(10e9) + 1000)
    assert s.text(["/m"]) == "3 models, 10.00 GB in total (1 unreadable)"
    assert ob.scan_summary([a], infos).text(["/m"]) == "1 model, 4.00 GB in total"
    assert ob.ScanSummary().text(["/m"]) == "No models found in these folders."
    assert "add one" in ob.ScanSummary().text([])


# ---------------------------------------------------------------------------- step 4

def test_intent_blurb_has_no_model_numbers() -> None:
    assert ob.intent_blurb(by_key["chat"]) == (
        "Chat: interactive chat frontends (about 16k context, 1 slot)")
    assert "4 slots" in ob.intent_blurb(by_key["api"])
    assert "trained context" in ob.intent_blurb(by_key["context"])


def test_model_choices_give_each_models_verdict_for_the_intent(tmp_path: Path) -> None:
    small, big, bad, emb = (tmp_path / n for n in ("small.gguf", "big.gguf", "bad.gguf", "emb.gguf"))
    infos = {str(small): make_info(small, int(4e9)), str(big): make_info(big, int(40e9)),
             str(emb): make_info(emb, int(1e9), pooling=1, ctx_train=8192)}
    got = ob.model_choices([small, big, bad, emb], infos, make_host(16e9), Config(), None, "chat")
    assert [c.alias for c in got] == ["small", "big", "emb"]          # the unreadable one is left out
    assert [c.verdict for c in got] == ["GPU", "NO", "GPU"]
    assert [c.intent for c in got] == ["chat", "chat", "embed"]       # embedding model: its own intent
    assert got[0].ctx == 16384 and got[0].size == int(4e9)


# ---------------------------------------------------------------------------- the flow

def test_flow_walks_the_five_steps_with_progress(tmp_path: Path) -> None:
    cfg = Config(bin_dir="/b", model_dirs=["/m"], default_intent="agent")
    f = ob.Flow(cfg)
    assert f.progress() == "Step 1 of 5: llama.cpp" and f.first and not f.last
    assert f.draft.default_intent == "agent" and f.draft.model_dirs == ["/m"]
    assert not f.back()
    seen = [f.step]
    while f.next():
        seen.append(f.step)
    assert seen == list(ob.STEPS) and f.last and f.progress() == "Step 5 of 5: Finish"
    assert not f.next() and f.back() and f.step == "purpose"


def test_flow_copies_the_settings_it_may_change() -> None:
    cfg = Config(model_dirs=["/m"], default_intent="nonsense")
    f = ob.Flow(cfg)
    f.draft.model_dirs.append("/n")
    assert cfg.model_dirs == ["/m"] and f.draft.default_intent == "chat"


# ---------------------------------------------------------------------------- finish

@pytest.fixture
def env(isolated: SimpleNamespace, tmp_path: Path):
    d = bin_with_server(tmp_path)
    model = tmp_path / "alpha.gguf"
    info = make_info(model)
    cfg = Config(bin_dir="/old", model_dirs=["/old-models"])
    draft = ob.Draft(str(d), [str(isolated.model_dir)], "agent", str(model), tune=True,
                     start_gateway=True)
    return SimpleNamespace(cfg=cfg, draft=draft, model=model, info=info, bin=d, iso=isolated)


def test_finish_applies_the_draft_creates_the_preset_and_saves(env) -> None:
    out = ob.finish(env.cfg, env.draft, {str(env.model): env.info}, make_host(16e9), None,
                    "http://127.0.0.1:8080")
    cfg = env.cfg
    assert cfg.onboarded and cfg.bin_dir == str(env.bin) and cfg.default_intent == "agent"
    assert cfg.model_dirs == [str(env.iso.model_dir)]
    p = cfg.preset(str(env.model))
    assert p is not None and p.source == "estimated" and p.intent == "agent"
    assert (out.preset, out.created, out.served) == ("agent", True, "alpha")
    assert out.url == "http://127.0.0.1:8080/v1"
    assert "/v1/chat/completions" in out.snippet and '"model": "alpha"' in out.snippet
    assert out.start_gateway and out.tune and out.bin_changed and out.dirs_changed
    saved = Config.load()
    assert saved.onboarded and saved.preset(str(env.model)).source == "estimated"


def test_finish_without_a_model_still_onboards_and_starts_nothing(env) -> None:
    env.draft.model = ""
    out = ob.finish(env.cfg, env.draft, {}, make_host(), None, "http://127.0.0.1:8080/")
    assert env.cfg.onboarded and env.cfg.models == {}
    assert out.model == "" and out.snippet == "" and out.url == "http://127.0.0.1:8080/v1"
    assert not out.start_gateway and not out.tune


def test_finish_does_not_replace_or_reactivate_existing_presets(env) -> None:
    path = str(env.model)
    env.cfg.set_preset(path, Preset("chat", Profile(ctx_size=4096), "tuned"))
    ob.finish(env.cfg, env.draft, {path: env.info}, make_host(), None, "http://x:1")
    ent = env.cfg.models[path]
    assert ent.active == "chat" and set(ent.presets) == {"chat", "agent"}
    assert ent.presets["chat"].source == "tuned" and ent.presets["agent"].source == "estimated"
    ob.finish(env.cfg, env.draft, {path: env.info}, make_host(), None, "http://x:1")
    assert ent.presets["agent"].source == "estimated" and len(ent.presets) == 2


def test_finish_without_a_working_llama_cpp_skips_gateway_and_tune(env) -> None:
    env.draft.bin_dir = "/nowhere"
    out = ob.finish(env.cfg, env.draft, {str(env.model): env.info}, make_host(), None, "http://x:1")
    assert not out.start_gateway and not out.tune and out.preset == "agent"


def test_finish_with_an_unread_model_header_changes_nothing(env) -> None:
    with pytest.raises(LookupError):
        ob.finish(env.cfg, env.draft, {}, make_host(), None, "http://x:1")
    assert not env.cfg.onboarded and env.cfg.bin_dir == "/old"


def test_skip_marks_onboarded_once_and_applies_nothing(isolated: SimpleNamespace) -> None:
    cfg = Config(bin_dir="/old")
    assert ob.skip(cfg) and cfg.onboarded and Config.load().onboarded
    assert Config.load().bin_dir == "/old"
    assert ob.skip(cfg) is False
