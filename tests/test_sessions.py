"""Unit tests for port probing and the async launch + loading banner."""
from __future__ import annotations

import socket
from types import SimpleNamespace

import pytest

from llamactl import sessions
from llamactl.sessions import SessionStore, next_free_port, port_free


def test_port_free() -> None:
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        taken = held.getsockname()[1]
        assert not port_free(taken)
    assert port_free(taken)  # freed once the socket closes


def test_next_free_port_skips_taken() -> None:
    assert next_free_port({sessions.BASE_PORT}) == sessions.BASE_PORT + 1


def test_next_free_port_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sessions, "port_free", lambda p, host="127.0.0.1": False)
    with pytest.raises(RuntimeError):
        next_free_port(set())


async def _wait(pilot, cond, tries: int = 60) -> bool:
    for _ in range(tries):
        await pilot.pause(0.1)
        if cond():
            return True
    return False


# --------------------------------------------------------------------------- launch_server presets

class _FakePopen:
    """Records the argv instead of starting a process."""

    calls: list[list[str]] = []

    def __init__(self, argv: list[str], **kwargs: object) -> None:
        type(self).calls.append(argv)
        assert kwargs["start_new_session"] is True
        self.pid = 4242


@pytest.fixture
def two_presets(isolated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch):
    from llamactl.config import Config, Preset, Profile
    _FakePopen.calls = []
    monkeypatch.setattr(sessions.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(sessions, "next_free_port", lambda taken, host="127.0.0.1": 8123)
    cfg = Config(bin_dir="/bin")
    model = "/models/Qwen3.gguf"
    cfg.set_preset(model, Preset("chat", Profile(ctx_size=16384), "estimated"), activate=True)
    cfg.set_preset(model, Preset("agent", Profile(ctx_size=65536, cache_reuse=256), "estimated"),
                   activate=False)
    return cfg, model


def _ctx(argv: list[str]) -> str:
    return argv[argv.index("-c") + 1]


def test_launch_defaults_to_the_active_preset(two_presets) -> None:
    cfg, model = two_presets
    s = SessionStore().launch_server(cfg, model)
    assert _ctx(_FakePopen.calls[-1]) == "16384" and "-m" in _FakePopen.calls[-1]
    assert (s.id, s.name, s.url, s.pid) == ("Qwen3-8123", "Qwen3", "http://127.0.0.1:8123", 4242)


def test_launch_explicit_non_active_preset_is_named_in_the_session(two_presets) -> None:
    cfg, model = two_presets
    s = SessionStore().launch_server(cfg, model, "agent")
    argv = _FakePopen.calls[-1]
    assert _ctx(argv) == "65536" and argv[argv.index("--cache-reuse") + 1] == "256"
    assert (s.id, s.name) == ("Qwen3@agent-8123", "Qwen3@agent")
    assert s.log.endswith("Qwen3@agent-8123.log")


def test_launch_explicit_active_preset_keeps_the_plain_name(two_presets) -> None:
    cfg, model = two_presets
    s = SessionStore().launch_server(cfg, model, "chat")
    assert s.name == "Qwen3" and _ctx(_FakePopen.calls[-1]) == "16384"


def test_launch_unknown_preset(two_presets) -> None:
    cfg, model = two_presets
    with pytest.raises(LookupError, match="no preset 'speed'.*chat, agent"):
        SessionStore().launch_server(cfg, model, "speed")
    assert not _FakePopen.calls


def test_launch_model_without_presets_uses_default_profile(two_presets) -> None:
    from llamactl.config import Profile
    cfg, _ = two_presets
    cfg.default_profile = Profile(ctx_size=2048)
    s = SessionStore().launch_server(cfg, "/models/other.gguf")
    assert _ctx(_FakePopen.calls[-1]) == "2048" and s.name == "other"
    with pytest.raises(LookupError, match="no preset 'agent'"):
        SessionStore().launch_server(cfg, "/models/other.gguf", "agent")


def test_launch_port_override_and_busy_port(two_presets) -> None:
    cfg, model = two_presets
    store = SessionStore()
    s = store.launch_server(cfg, model, "agent", port=48123)
    assert s.url.endswith(":48123") and "48123" in _FakePopen.calls[-1]
    with pytest.raises(RuntimeError, match="port 48123.*in use"):
        store.launch_server(cfg, model, port=48123)     # held by the first session


def test_load_without_persist_does_not_write(isolated: SimpleNamespace) -> None:
    SessionStore(persist=False)
    assert not sessions.STATE_FILE.exists()
    SessionStore()
    assert sessions.STATE_FILE.exists()


def test_pid_alive() -> None:
    import os
    assert sessions.pid_alive(os.getpid()) and not sessions.pid_alive(0)
    assert not sessions.pid_alive(2 ** 22 + 12345)
