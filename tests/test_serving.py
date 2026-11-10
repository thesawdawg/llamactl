"""The shared serving snapshot: gathering, gateway wording, uptime."""
from __future__ import annotations

import os

import httpx
import pytest

from fakes import FakeGateway, FakeStore
from llamactl import serving
from llamactl.config import Config, Preset, Profile
from llamactl.gateway import LOADED, GatewayError, GatewayStatus
from llamactl.serving import ServingSnapshot, gather, gateway_state, uptime
from llamactl.sessions import Session


def _cfg() -> Config:
    cfg = Config()
    cfg.set_preset("/m/a.gguf", Preset("chat", Profile(), "estimated"))
    cfg.set_preset("/m/a.gguf", Preset("agent", Profile(), "estimated"), activate=False)
    return cfg


def test_gather_idle_makes_no_calls() -> None:
    gw, store = FakeGateway(_cfg()), FakeStore()
    snap = gather(gw, store, health=lambda url: pytest.fail("no probe expected"))
    assert snap.idle and snap.models is None and snap.error == ""
    assert not snap.status.running and snap.running_models == []
    assert gw.called("models") == []


def test_gather_running_gateway_and_sessions() -> None:
    gw = FakeGateway(_cfg(), running=True)
    gw.set_state("a", LOADED)
    ext = Session("ext-x", "external", "http://x:1", external=True)
    dead = Session("dead-1", "dead", "http://127.0.0.1:1", pid=2**22 + 7, log="/l")
    snap = gather(gw, FakeStore([ext, dead]), health=lambda url: "ready")
    assert [m.name for m in snap.models] == ["a", "a@agent"]
    assert [m.name for m in snap.running_models] == ["a"]
    assert snap.health == {"ext-x": "ready", "dead-1": "crashed"}
    assert snap.log and snap.started > 0 and not snap.idle


def test_gather_reports_a_gateway_that_cannot_list_models() -> None:
    gw = FakeGateway(_cfg(), running=True)
    gw.fail("models", GatewayError("boom"))
    snap = gather(gw, FakeStore())
    assert snap.models is None and snap.error == "boom" and snap.status.running


def test_snapshot_same_as_ignores_the_timestamp() -> None:
    st = GatewayStatus(False, "u")
    assert ServingSnapshot(st, at=1.0).same_as(ServingSnapshot(st, at=2.0))
    assert not ServingSnapshot(st).same_as(ServingSnapshot(GatewayStatus(True, "u")))
    assert not ServingSnapshot(st).same_as(None)


def test_gateway_state_words() -> None:
    assert gateway_state(None) == "stopped"
    off = ServingSnapshot(GatewayStatus(False, "u"))
    assert gateway_state(off) == "stopped"
    up = ServingSnapshot(GatewayStatus(True, "u", True, 5), started=100.0)
    assert gateway_state(up, now=1000.0) == "running"
    young = ServingSnapshot(GatewayStatus(True, "u", False, 5), started=100.0)
    assert gateway_state(young, now=110.0) == "starting"
    assert gateway_state(young, now=100.0 + serving.START_GRACE + 1) == "unhealthy"
    assert gateway_state(off, busy="starting") == "starting"
    assert gateway_state(up, busy="stopping") == "stopping"


@pytest.mark.parametrize("seconds,text", [(0, "-"), (5, "5s"), (125, "2m"), (3700, "1h01m"),
                                          (200000, "2d")])
def test_uptime(seconds: int, text: str) -> None:
    assert uptime(0 if seconds == 0 else 1000.0, 1000.0 + seconds) == text


def test_http_health_maps_status_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    class R:
        def __init__(self, code: int) -> None:
            self.status_code = code

    for code, want in ((200, "ready"), (503, "loading"), (404, "down")):
        monkeypatch.setattr(serving.httpx, "get", lambda url, timeout=0, c=code: R(c))
        assert serving.http_health("http://x/") == want

    def refuse(url, timeout=0):
        raise httpx.ConnectError("no")

    monkeypatch.setattr(serving.httpx, "get", refuse)
    assert serving.http_health("http://x") == "down"
