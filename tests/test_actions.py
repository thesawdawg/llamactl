"""Shared actions: preset choice, served names, waiting (fake clock, no sleeping)."""
from __future__ import annotations

from pathlib import Path

import pytest

from llamactl import actions
from llamactl.actions import ActionError
from llamactl.calibration import Calibration
from llamactl.config import Config, Preset, Profile
from llamactl.gateway import FAILED, LOADED, LOADING, UNLOADED, GatewayError, GatewayModel
from llamactl.gguf import ModelInfo
from llamactl.hardware import Device, Host


def _info(pooling: int = -1) -> ModelInfo:
    return ModelInfo(path=Path("/m/qwen3.gguf"), size=int(4e9), architecture="x", name="m",
                     n_layer=32, n_embd=4096, n_head=32, n_head_kv=8, head_dim_k=128,
                     head_dim_v=128, ctx_train=32768, expert_count=0, file_type=0, sharded=1,
                     parameter_count=0, pooling_type=pooling)


HOST = Host(gpus=[Device.of("CUDA0", "G", int(16e9), int(16e9))], ram_total=int(24e9),
            ram_free=int(24e9), cpu_threads=16, cpu_cores=8)


class Clock:
    """Fake monotonic clock; sleeping advances it."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


class ScriptedGateway:
    """A gateway whose model listing follows a script of states, then repeats the last."""

    def __init__(self, *states: tuple[str, int | None]) -> None:
        self.script, self.state = list(states), None
        self.polls = 0

    def models(self) -> list[GatewayModel]:
        st, code = self.script[min(self.polls, len(self.script) - 1)]
        self.polls += 1
        return [GatewayModel("m", st, code)]


def test_choose_intent() -> None:
    assert actions.choose_intent(None, "agent", _info()).key == "agent"
    assert actions.choose_intent(None, "agent", _info(pooling=1)).key == "embed"
    assert actions.choose_intent("speed", "chat", _info()).key == "speed"
    with pytest.raises(LookupError, match="does not apply"):
        actions.choose_intent("embed", "chat", _info())
    with pytest.raises(LookupError, match="no such intent"):
        actions.choose_intent("zzz", "chat", _info())


def test_pick_preset_default_plans_stores_and_activates() -> None:
    cfg = Config()
    c = actions.pick_preset(cfg, _info(), HOST, Calibration(), save=False)
    assert (c.name, c.created, c.preset.source) == ("chat", True, "estimated")
    assert cfg.models["/m/qwen3.gguf"].active == "chat"
    again = actions.pick_preset(cfg, _info(), HOST, Calibration(), save=False)
    assert again.created is False and again.preset is c.preset


def test_pick_preset_keeps_existing_and_never_replans() -> None:
    cfg = Config()
    mine = Preset("agent", Profile(ctx_size=777), "tuned")
    cfg.set_preset("/m/qwen3.gguf", mine)
    cfg.set_preset("/m/qwen3.gguf", Preset("chat", Profile(), "estimated"), activate=False)
    assert actions.pick_preset(cfg, _info(), HOST, None, save=False).preset is mine
    c = actions.pick_preset(cfg, _info(), HOST, None, "chat", activate=False, save=False)
    assert not c.created and cfg.models["/m/qwen3.gguf"].active == "agent"
    c = actions.pick_preset(cfg, _info(), HOST, None, "chat", save=False)
    assert cfg.models["/m/qwen3.gguf"].active == "chat" and not c.created


def test_served_name_uses_the_separator_for_non_active_presets() -> None:
    cfg = Config()
    cfg.set_preset("/m/qwen3.gguf", Preset("chat", Profile(), "estimated"))
    cfg.set_preset("/m/qwen3.gguf", Preset("agent", Profile(), "estimated"), activate=False)
    assert actions.served_name(cfg, "/m/qwen3.gguf", "chat") == "qwen3"
    assert actions.served_name(cfg, "/m/qwen3.gguf", "agent") == "qwen3@agent"


def test_store_plan_protects_tuned_presets() -> None:
    cfg = Config()
    plan = actions.plan_for(cfg, _info(), HOST, actions.choose_intent("chat", "chat", _info()))
    cfg.set_preset("/m/qwen3.gguf", Preset("chat", Profile(), "tuned"))
    with pytest.raises(ActionError, match="tuned"):
        actions.store_plan(cfg, "/m/qwen3.gguf", plan, save=False)
    actions.store_plan(cfg, "/m/qwen3.gguf", plan, overwrite=True, save=False)
    assert cfg.preset("/m/qwen3.gguf", "chat").source == "estimated"


def test_wait_loaded_progress_and_ticks() -> None:
    clock, ticks = Clock(), []
    gw = ScriptedGateway((LOADING, None), (LOADING, None), (LOADED, None))
    m = actions.wait_loaded(gw, "m", 60, clock=clock, sleep=clock.sleep, tick=lambda: ticks.append(1))
    assert m.state == LOADED and len(ticks) == 2 and clock.t == 2.0


def test_wait_loaded_failure_timeout_and_vanished_model() -> None:
    clock = Clock()
    with pytest.raises(GatewayError, match=r"failed to load \(exit code 7\)"):
        actions.wait_loaded(ScriptedGateway((FAILED, 7)), "m", 60, clock=clock, sleep=clock.sleep)
    with pytest.raises(GatewayError, match="still loading after 5 s"):
        actions.wait_loaded(ScriptedGateway((LOADING, None)), "m", 5, clock=clock, sleep=clock.sleep)
    with pytest.raises(GatewayError, match="no longer lists"):
        actions.wait_loaded(ScriptedGateway((LOADING, None)), "other", 5, clock=clock, sleep=clock.sleep)


def test_wait_loaded_gives_up_on_a_model_that_stays_unloaded() -> None:
    clock = Clock()
    with pytest.raises(GatewayError, match="reports 'm' as unloaded"):
        actions.wait_loaded(ScriptedGateway((UNLOADED, None)), "m", 600, clock=clock, sleep=clock.sleep)
    assert clock.t >= actions.UNLOADED_GRACE


def test_curl_snippet() -> None:
    c = actions.curl_snippet("http://h:1/", "m@x", api_key=True)
    assert c.startswith("curl http://h:1/v1/chat/completions") and "Bearer YOUR_API_KEY" in c
    assert '"model": "m@x"' in c
    e = actions.curl_snippet("http://h:1", None, embedding=True)
    assert "/v1/embeddings" in e and "model" not in e
