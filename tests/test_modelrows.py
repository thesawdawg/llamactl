"""Summaries behind the Models table and the intent picker."""
from __future__ import annotations

from pathlib import Path

import pytest

from fakes import make_host, make_info
from llamactl import modelrows, planner
from llamactl.calibration import Calibration
from llamactl.config import ModelEntry, Preset, Profile
from llamactl.measure import Measurement
from llamactl.modelrows import best_for, default_intent_key, intent_rows, preset_state, speed_of


def _plans(vram: float = 16e9, size: int = int(4e9)):
    info, host = make_info(size=size), make_host(vram)
    return planner.plan_all(info, host, Profile(), 10), host


def test_best_for_lists_intents_with_the_best_verdict() -> None:
    plans, _ = _plans()
    b = best_for(plans)
    assert b.verdict == "GPU" and b.keys == ("chat", "agent", "context", "speed", "api")
    assert b.text() == "chat agent context speed +1"


def test_best_for_collapses_after_four() -> None:
    b = modelrows.BestFor("GPU", ("chat", "agent", "context", "speed", "api"))
    assert b.text() == "chat agent context speed +1"


def test_best_for_when_nothing_fits() -> None:
    plans, _ = _plans(vram=2e9, size=int(80e9))
    b = best_for(plans)
    assert b.verdict == "NO" and b.text() == "-"
    assert best_for({}).text() == "-"


def test_best_for_prefers_gpu_over_partial() -> None:
    plans, _ = _plans(vram=6e9, size=int(4e9))
    b = best_for(plans)
    assert b.verdict in ("GPU", "GPU+CPU")
    assert all(plans[k].verdict == b.verdict for k in b.keys)


def test_speed_of_only_for_tuned_ok() -> None:
    m = Measurement(ok=True, gen_tps=41.5, prompt_tps=800.0)
    assert speed_of(Preset("chat", Profile(), "tuned", m)) == 41.5
    assert speed_of(Preset("chat", Profile(), "edited", m)) is None
    assert speed_of(Preset("chat", Profile(), "tuned", Measurement(error="x"))) is None
    assert speed_of(Preset("embed", Profile(), "tuned", Measurement(ok=True, prompt_tps=900.0))) == 900.0
    assert speed_of(None) is None


def test_preset_state_new_estimated_tuned_edited_and_extra_count() -> None:
    path, cal = Path("/m/a.gguf"), Calibration()
    assert preset_state(None, path, cal, "").text() == "New"
    assert preset_state(ModelEntry(), path, cal, "").new
    ent = ModelEntry("a", "chat", {"chat": Preset("chat", Profile(), "estimated")})
    st = preset_state(ent, path, cal, "")
    assert (st.name, st.source, st.measured, st.text()) == ("chat", "estimated", None,
                                                            "chat estimated")
    ent.presets["agent"] = Preset("agent", Profile(), "tuned",
                                  Measurement(ok=True, gen_tps=33.0, at="t", build="old"))
    ent.active = "agent"
    st = preset_state(ent, path, cal, "old")
    assert (st.source, st.measured, st.stale, st.count) == ("tuned", 33.0, False, 2)
    assert st.text() == "agent tuned (+1)"
    st = preset_state(ent, path, cal, "new")
    assert st.stale and st.text() == "agent tuned stale (+1)"


def test_intent_rows_have_plan_and_preset_facts() -> None:
    plans, host = _plans()
    path = Path("/m/a.gguf")
    ent = ModelEntry("a", "agent", {
        "agent": Preset("agent", plans["agent"].profile, "tuned",
                        Measurement(ok=True, gen_tps=28.0, at="t", build="b")),
        "chat": Preset("chat", plans["chat"].profile, "estimated")})
    rows = {r.key: r for r in intent_rows(plans, ent, path, host, 10, Calibration(), "b")}
    assert list(rows) == list(plans)
    agent, chat, speed = rows["agent"], rows["chat"], rows["speed"]
    assert (agent.source, agent.active, agent.tps) == ("tuned", True, 28.0)
    assert (chat.source, chat.active, chat.tps) == ("estimated", False, None)
    assert (speed.source, speed.has_preset, speed.tps) == ("", False, None)
    assert agent.ctx_per_slot == plans["agent"].ctx_per_slot and agent.slots == 1
    assert agent.needs == plans["agent"].budget.total()
    assert agent.usable == planner.gpu_limit(host, 10) > 0
    assert agent.kv == plans["agent"].profile.cache_type_k
    assert rows["api"].slots == 4


def test_top_reason_prefers_the_context_line() -> None:
    plans, _ = _plans()
    r = modelrows.top_reason(plans["chat"])
    assert r == plans["chat"].reasons["ctx_size"]


def test_default_intent_key_order() -> None:
    plans, _ = _plans()
    ent = ModelEntry("a", "speed", {"speed": Preset("speed")})
    assert default_intent_key(ent, plans, "chat") == "speed"
    assert default_intent_key(None, plans, "agent") == "agent"
    assert default_intent_key(None, plans, "embed") == "chat"      # not planned for this model
    assert default_intent_key(None, {}, "chat") == ""
