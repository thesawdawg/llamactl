"""Gateway: preset INI rendering, process control and the router API client (all faked)."""
from __future__ import annotations

import json
import signal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from llamactl import gateway
from llamactl.config import Config, GatewaySettings, Preset, Profile
from llamactl.gateway import (FAILED, Gateway, GatewayError, GatewayModel, GatewayState,
                              parse_models, render_presets)


# --------------------------------------------------------------------------- fixtures

@pytest.fixture
def two_models(tmp_path: Path) -> tuple[Config, Path, Path]:
    """A config with two model files: qwen (chat active + agent) and a one-preset llama."""
    qwen, llama = tmp_path / "Qwen3-8B-Q4_K_M.gguf", tmp_path / "llama-3.2-1b.gguf"
    qwen.write_bytes(b"x")
    llama.write_bytes(b"x")
    cfg = Config(bin_dir="/opt/llama/bin")
    cfg.set_preset(str(qwen), Preset("chat", Profile(ctx_size=16384, threads=8, flash_attn="on"),
                                     "estimated"))
    cfg.set_preset(str(qwen), Preset("agent", Profile(
        ctx_size=65536, cache_type_k="q8_0", cache_type_v="q8_0", cache_reuse=256,
        extra_args="--jinja"), "tuned"), activate=False)
    cfg.set_preset(str(llama), Preset("speed", Profile(ctx_size=8192, parallel=4, metrics=True,
                                                       cont_batching=False), "estimated"))
    cfg.models[str(qwen)].alias = "qwen3-8b"
    cfg.models[str(llama)].alias = "llama-3.2-1b"
    return cfg, qwen, llama


GOLDEN = """\
version = 1

[*]
sleep-idle-seconds = 600

[llama-3.2-1b]
model = {llama}
ctx-size = 8192
n-gpu-layers = all
flash-attn = auto
parallel = 4
cont-batching = false
metrics = true

[qwen3-8b]
model = {qwen}
ctx-size = 16384
n-gpu-layers = all
flash-attn = on
threads = 8

[qwen3-8b@agent]
model = {qwen}
ctx-size = 65536
n-gpu-layers = all
cache-type-k = q8_0
cache-type-v = q8_0
flash-attn = auto
cache-reuse = 256
jinja = true
"""


# --------------------------------------------------------------------------- render_presets

def test_render_golden(two_models) -> None:
    cfg, qwen, llama = two_models
    assert render_presets(cfg) == GOLDEN.format(qwen=qwen, llama=llama)


def test_render_is_deterministic_whatever_the_insertion_order(two_models) -> None:
    cfg, qwen, llama = two_models
    first = render_presets(cfg)
    cfg.models = dict(reversed(list(cfg.models.items())))
    ent = cfg.models[str(qwen)]
    ent.presets = dict(reversed(list(ent.presets.items())))
    assert render_presets(cfg) == first


def test_active_preset_is_the_plain_alias_and_others_are_suffixed(two_models) -> None:
    cfg, qwen, _ = two_models
    cfg.models[str(qwen)].active = "agent"
    names = [ln for ln in render_presets(cfg).splitlines() if ln.startswith("[")]
    assert names == ["[*]", "[llama-3.2-1b]", "[qwen3-8b]", "[qwen3-8b@chat]"]
    assert "cache-reuse = 256" in render_presets(cfg).split("[qwen3-8b]")[1].split("[")[0]


def test_no_section_name_uses_a_colon(two_models) -> None:
    """Regression: the router rewrites the text after ':' ("qwen:agent" became "qwen:AGENT")."""
    cfg, qwen, _ = two_models
    cfg.models[str(qwen)].presets["my-preset"] = Preset("my-preset", Profile())
    sections = [ln for ln in render_presets(cfg).splitlines() if ln.startswith("[")]
    assert "[qwen3-8b@my-preset]" in sections and not any(":" in s for s in sections)


def test_invalid_active_falls_back_to_the_first_preset_by_name(two_models) -> None:
    cfg, qwen, _ = two_models
    cfg.models[str(qwen)].active = ""
    names = [ln for ln in render_presets(cfg).splitlines() if ln.startswith("[")]
    assert "[qwen3-8b]" in names and "[qwen3-8b@chat]" in names and "[qwen3-8b@agent]" not in names


def test_missing_model_file_is_skipped(two_models) -> None:
    cfg, qwen, llama = two_models
    llama.unlink()
    text = render_presets(cfg)
    assert "llama" not in text and "[qwen3-8b]" in text


def test_models_filter_and_models_without_presets(two_models, tmp_path: Path) -> None:
    cfg, qwen, llama = two_models
    text = render_presets(cfg, [llama])
    assert "[llama-3.2-1b]" in text and "qwen3" not in text
    cfg.entry(str(tmp_path / "bare.gguf"))          # an entry with no presets
    assert "bare" not in render_presets(cfg)


def test_sleep_idle_global(two_models) -> None:
    cfg, *_ = two_models
    cfg.gateway.sleep_idle = 0
    assert "[*]" not in render_presets(cfg) and "sleep-idle" not in render_presets(cfg)
    cfg.gateway.sleep_idle = 45
    assert "[*]\nsleep-idle-seconds = 45\n" in render_presets(cfg)


def test_router_owned_keys_never_appear(two_models) -> None:
    cfg, qwen, _ = two_models
    cfg.models[str(qwen)].presets["chat"].profile = Profile(
        host="0.0.0.0", port=9000, api_key="secret", alias="other")
    text = render_presets(cfg)
    for key in ("host", "port", "api-key", "alias", "secret", "9000"):
        assert key not in text


def test_mmproj_and_draft_are_ordinary_keys(two_models) -> None:
    cfg, qwen, _ = two_models
    cfg.models[str(qwen)].presets["chat"].profile = Profile(mmproj="/m/p.gguf", draft_model="/m/d.gguf")
    text = render_presets(cfg)
    assert "mmproj = /m/p.gguf\n" in text and "spec-draft-model = /m/d.gguf\n" in text


def test_unrepresentable_preset_is_skipped_with_a_problem(two_models) -> None:
    cfg, qwen, _ = two_models
    cfg.models[str(qwen)].presets["agent"].profile = Profile(extra_args="stray")
    problems: list[str] = []
    text = render_presets(cfg, problems=problems)
    assert "[qwen3-8b@agent]\n" not in text and "[qwen3-8b]" in text
    assert "; skipped [qwen3-8b@agent]: extra_args: cannot express 'stray'" in text
    assert len(problems) == 1 and problems[0].startswith("skipped [qwen3-8b@agent]")


def test_bad_names_and_paths_are_skipped(two_models, tmp_path: Path) -> None:
    cfg, qwen, llama = two_models
    cfg.models[str(qwen)].presets["a]b"] = Preset("a]b", Profile())
    odd = tmp_path / "a#1.gguf"
    odd.write_bytes(b"x")
    cfg.set_preset(str(odd), Preset("chat", Profile()))
    cfg.models[str(odd)].alias = "odd"
    problems: list[str] = []
    text = render_presets(cfg, problems=problems)
    sections = [ln for ln in text.splitlines() if ln.startswith("[")]
    assert "[qwen3-8b@a]b]" not in sections and "[odd]" not in sections
    assert len(problems) == 2 and all(ln in text for ln in ("; " + p for p in problems))


def test_duplicate_section_names_are_skipped(two_models) -> None:
    cfg, qwen, llama = two_models
    cfg.models[str(llama)].alias = "qwen3-8b"
    problems: list[str] = []
    text = render_presets(cfg, problems=problems)
    assert text.count("[qwen3-8b]\n") == 1 and len(problems) == 1


def test_rendered_ini_parses_back(two_models) -> None:
    """configparser reads our output; sections and keys round-trip."""
    import configparser
    cfg, qwen, llama = two_models
    cp = configparser.ConfigParser(interpolation=None, delimiters=("=",), comment_prefixes=(";", "#"),
                                   strict=True)
    cp.optionxform = str
    cp.read_string(render_presets(cfg).split("\n", 1)[1])      # drop the top-level `version`
    assert cp.sections() == ["*", "llama-3.2-1b", "qwen3-8b", "qwen3-8b@agent"]
    assert cp["qwen3-8b@agent"]["model"] == str(qwen) and cp["llama-3.2-1b"]["metrics"] == "true"


# --------------------------------------------------------------------------- fakes

class FakePopen:
    """Records launches; never starts a process."""

    calls: list[tuple[list[str], dict[str, Any]]] = []
    next_pid = 777

    def __init__(self, argv: list[str], **kwargs: Any) -> None:
        type(self).calls.append((argv, kwargs))
        self.pid = type(self).next_pid


class FakeResponse:
    def __init__(self, status: int = 200, body: Any = None, text: str = "") -> None:
        self.status_code, self._body = status, body
        self.text = text or (json.dumps(body) if body is not None else "")

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeHttp:
    """Stands in for httpx.Client: records requests, answers from a route table."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.routes: dict[tuple[str, str], Any] = {}
        self.clients: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> "FakeHttp":
        self.clients.append(kwargs)
        return self

    def __enter__(self) -> "FakeHttp":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def request(self, method: str, path: str, **kwargs: Any) -> FakeResponse:
        self.requests.append({"method": method, "path": path, **kwargs})
        answer = self.routes.get((method, path), FakeResponse(200, {"success": True}))
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def env(isolated: SimpleNamespace, two_models, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A Gateway over fakes, with process liveness controlled through `alive`."""
    cfg, qwen, llama = two_models
    FakePopen.calls = []
    http = FakeHttp()
    alive: set[int] = set()
    monkeypatch.setattr(gateway, "pid_alive", lambda pid: pid in alive)
    monkeypatch.setattr(gateway, "port_free", lambda port, host="127.0.0.1": True)
    gw = Gateway(cfg, popen=FakePopen, http=http, settle=0)
    return SimpleNamespace(gw=gw, cfg=cfg, http=http, alive=alive, cfg_dir=isolated.cfg_dir,
                           qwen=qwen, llama=llama)


def started(env: SimpleNamespace) -> GatewayState:
    st = env.gw.start()
    env.alive.add(st.pid)
    return st


# --------------------------------------------------------------------------- command / start

def test_command_line(env) -> None:
    env.cfg.gateway = GatewaySettings(host="0.0.0.0", port=9090, models_max=2)
    argv = env.gw.command()
    assert argv == ["/opt/llama/bin/llama-server", "--models-preset", str(env.cfg_dir / "gateway.ini"),
                    "--models-max", "2", "--host", "0.0.0.0", "--port", "9090"]
    assert "-m" not in argv
    env.cfg.gateway.api_key = "k3y"
    assert env.gw.command()[-2:] == ["--api-key", "k3y"]


def test_start_writes_ini_spawns_detached_and_saves_state(env) -> None:
    st = env.gw.start()
    argv, kwargs = FakePopen.calls[0]
    assert argv == env.gw.command() and kwargs["start_new_session"] is True
    assert kwargs["stdin"] is not None
    assert (env.cfg_dir / "gateway.ini").read_text() == render_presets(env.cfg)
    assert st.pid == 777 and st.url == "http://127.0.0.1:8080"
    assert st.log == str(env.cfg_dir / "logs" / "gateway.log") and st.started > 0
    assert GatewayState.load() == st


def test_state_round_trip_and_garbage(env) -> None:
    st = GatewayState(5, "http://127.0.0.1:1", "/l", 1.5)
    st.save()
    assert GatewayState.load() == st
    GatewayState.clear()
    assert GatewayState.load() is None
    (env.cfg_dir / "gateway.json").write_text("not json")
    assert GatewayState.load() is None
    (env.cfg_dir / "gateway.json").write_text('{"pid": "x"}')
    assert GatewayState.load() is None


def test_start_refuses_a_busy_port(env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway, "port_free", lambda port, host="127.0.0.1": False)
    with pytest.raises(GatewayError, match="port 8080.*already in use"):
        env.gw.start()
    assert not FakePopen.calls and GatewayState.load() is None


def test_start_is_idempotent_while_running(env) -> None:
    first = started(env)
    assert env.gw.start() == first and len(FakePopen.calls) == 1


def test_start_reports_an_immediate_exit_with_thelog_reason(env) -> None:
    """Regression: the real router exits at once on an option it does not know."""
    def dies(argv: list[str], stdout: Any = None, **kw: Any) -> SimpleNamespace:
        stdout.write(b"0.00.001.116 I srv  llama_server: initializing ...\n"
                     b"0.00.082.119 E srv  llama_server: failed to initialize router models: "
                     b"option 'no-mmap' not recognized in preset 'qwen3-8b@agent'\n")
        stdout.flush()
        return SimpleNamespace(pid=999)
    gw = Gateway(env.cfg, popen=dies, http=env.http, settle=0.3, sleep=lambda s: None)
    with pytest.raises(GatewayError, match=r"exited at once: srv  llama_server: failed to "
                                           r"initialize router models: option 'no-mmap' not "
                                           r"recognized.*gateway\.log"):
        gw.start()
    assert GatewayState.load() is None


def test_start_settles_when_the_process_stays_up(env) -> None:
    env.alive.add(777)
    gw = Gateway(env.cfg, popen=FakePopen, http=env.http, settle=0.3, sleep=lambda s: None)
    assert gw.start().pid == 777


def testlog_reason_fallbacks(tmp_path: Path) -> None:
    from llamactl.gateway import log_reason
    assert log_reason(tmp_path / "missing.log") == "no log output"
    empty = tmp_path / "e.log"
    empty.write_text("\n")
    assert log_reason(empty) == "no log output"
    plain = tmp_path / "p.log"
    plain.write_text("first\nlast words\n")
    assert log_reason(plain) == "last words"


def test_start_reports_a_launch_failure(env) -> None:
    def boom(argv: list[str], **kw: Any) -> None:
        raise FileNotFoundError(2, "No such file or directory")
    gw = Gateway(env.cfg, popen=boom, http=env.http, settle=0)
    with pytest.raises(GatewayError, match="cannot start /opt/llama/bin/llama-server"):
        gw.start()
    assert GatewayState.load() is None


def test_running_forgets_a_dead_gateway(env) -> None:
    st = env.gw.start()                       # pid never added to `alive`: it died at once
    assert not env.gw.running and env.gw.state is None
    assert not (env.cfg_dir / "gateway.json").exists()
    env.alive.add(st.pid)
    assert not env.gw.running                  # state was cleared, nothing to find


def test_read_only_gateway_does_not_forget_a_dead_state(env) -> None:
    env.gw.start()
    ro = Gateway(env.cfg, popen=FakePopen, http=FakeHttp(), settle=0, read_only=True)
    assert not ro.running
    assert (env.cfg_dir / "gateway.json").exists()


def test_stop_signals_the_group_and_clears_state(env, monkeypatch: pytest.MonkeyPatch) -> None:
    st = started(env)
    sent: list[tuple[str, int, int]] = []

    def killpg(pgid: int, sig: int) -> None:
        sent.append(("killpg", pgid, sig))
        env.alive.discard(pgid)

    monkeypatch.setattr(gateway.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(gateway.os, "killpg", killpg)
    assert env.gw.stop() is True
    assert sent == [("killpg", st.pid, signal.SIGTERM)]
    assert not (env.cfg_dir / "gateway.json").exists()
    assert env.gw.stop() is False


def test_stop_escalates_to_sigkill(env, monkeypatch: pytest.MonkeyPatch) -> None:
    st = started(env)
    sent: list[int] = []
    monkeypatch.setattr(gateway.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(gateway.os, "killpg", lambda pgid, sig: sent.append(sig))
    gateway.terminate_group(st.pid, timeout=0.2, sleep=lambda s: None)
    assert sent == [signal.SIGTERM, signal.SIGKILL]


def test_stop_without_group_leadership_signals_only_the_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(gateway, "pid_alive", lambda pid: False)
    monkeypatch.setattr(gateway.os, "getpgid", lambda pid: 1)
    monkeypatch.setattr(gateway.os, "kill", lambda pid, sig: calls.append(("kill", pid)))
    monkeypatch.setattr(gateway.os, "killpg", lambda *a: calls.append(("killpg", 0)))
    gateway.terminate_group(55)
    assert calls == [("kill", 55)]


# --------------------------------------------------------------------------- status

def test_status_not_running_makes_no_request(env) -> None:
    s = env.gw.status()
    assert (s.running, s.healthy, s.pid, s.url) == (False, False, 0, "http://127.0.0.1:8080")
    assert env.http.requests == []


def test_status_healthy_and_unhealthy(env) -> None:
    started(env)
    env.http.routes[("GET", "/health")] = FakeResponse(200, {"status": "ok"})
    s = env.gw.status()
    assert (s.running, s.healthy, s.pid) == (True, True, 777)
    env.http.routes[("GET", "/health")] = FakeResponse(503, {"error": {"message": "Loading"}})
    assert env.gw.status().healthy is False
    env.http.routes[("GET", "/health")] = httpx.ConnectError("refused")
    s = env.gw.status()
    assert s.running and not s.healthy


def test_wildcard_host_connects_via_loopback(env) -> None:
    env.cfg.gateway.host = "0.0.0.0"
    assert env.gw.listen_url == "http://127.0.0.1:8080"


# --------------------------------------------------------------------------- models()

def entry(name: str, status: dict[str, Any]) -> dict[str, Any]:
    return {"id": name, "object": "model", "owned_by": "llamacpp", "status": status}


def test_parse_every_documented_status_shape() -> None:
    args = ["llama-server", "-ctx", "4096"]
    payload = {"object": "list", "data": [
        entry("a", {"value": "unloaded"}),
        entry("b", {"value": "loading", "args": args}),
        entry("c", {"value": "loaded", "args": args}),
        entry("d", {"value": "sleeping", "args": args}),
        entry("e", {"value": "unloaded", "args": args, "failed": True, "exit_code": 1}),
        entry("f", {"value": "downloading", "progress": {"https://x/m.gguf": {"done": 1, "total": 2}}}),
        entry("g", {"value": "unloaded", "args": args, "exit_code": 0}),
    ]}
    got = parse_models(payload)
    assert got == [
        GatewayModel("a", "unloaded"),
        GatewayModel("b", "loading", None, tuple(args)),
        GatewayModel("c", "loaded", None, tuple(args)),
        GatewayModel("d", "sleeping", None, tuple(args)),
        GatewayModel("e", FAILED, 1, tuple(args)),
        GatewayModel("f", "downloading"),
        GatewayModel("g", "unloaded", None, tuple(args)),
    ]


def test_failed_without_the_flag_but_with_a_nonzero_exit_code() -> None:
    m = parse_models({"data": [entry("x", {"value": "unloaded", "exit_code": 137})]})[0]
    assert (m.state, m.exit_code) == (FAILED, 137)


@pytest.mark.parametrize("payload", [None, [], {"data": "x"}, {"data": [{"id": "x"}]},
                                     {"data": [{"status": {"value": "loaded"}}]}])
def test_parse_rejects_garbage(payload: Any) -> None:
    with pytest.raises(ValueError):
        parse_models(payload)


def test_models_request(env) -> None:
    started(env)
    env.http.routes[("GET", "/models")] = FakeResponse(200, {"data": [
        entry("qwen3-8b", {"value": "loaded", "args": ["llama-server"]})]})
    assert env.gw.models() == [GatewayModel("qwen3-8b", "loaded", None, ("llama-server",))]
    assert env.http.requests == [{"method": "GET", "path": "/models"}]
    assert env.http.clients[0]["base_url"] == "http://127.0.0.1:8080"
    assert "Authorization" not in env.http.clients[0]["headers"]


def test_models_bad_answer(env) -> None:
    started(env)
    env.http.routes[("GET", "/models")] = FakeResponse(200, {"data": 5})
    with pytest.raises(GatewayError, match="unexpected answer"):
        env.gw.models()
    env.http.routes[("GET", "/models")] = FakeResponse(200, None, text="<html>")
    with pytest.raises(GatewayError, match="not JSON"):
        env.gw.models()


# --------------------------------------------------------------------------- load / unload / reload

def test_load_and_unload_post_the_model_name(env) -> None:
    started(env)
    env.gw.load("qwen3-8b@agent")
    env.gw.unload("qwen3-8b")
    assert env.http.requests == [
        {"method": "POST", "path": "/models/load", "json": {"model": "qwen3-8b@agent"}},
        {"method": "POST", "path": "/models/unload", "json": {"model": "qwen3-8b"}}]


def test_api_key_is_sent_as_a_bearer_header(env) -> None:
    env.cfg.gateway.api_key = "s3cret"
    started(env)
    env.http.routes[("GET", "/models")] = FakeResponse(200, {"data": []})
    env.gw.models()
    env.gw.load("x")
    env.gw.status()
    assert {c["headers"]["Authorization"] for c in env.http.clients} == {"Bearer s3cret"}
    assert env.http.clients[0]["timeout"] == gateway.HTTP_TIMEOUT


def test_router_errors_become_gateway_errors(env) -> None:
    started(env)
    env.http.routes[("POST", "/models/load")] = FakeResponse(
        404, {"error": {"message": "model is not found", "type": "not_found_error"}})
    with pytest.raises(GatewayError, match="cannot load 'nope': .*HTTP 404: model is not found"):
        env.gw.load("nope")
    env.http.routes[("POST", "/models/unload")] = FakeResponse(
        400, {"error": {"message": "model is not running"}})
    with pytest.raises(GatewayError, match="cannot unload 'qwen3-8b': .*model is not running"):
        env.gw.unload("qwen3-8b")
    env.http.routes[("GET", "/models")] = FakeResponse(
        401, {"error": {"message": "Invalid API Key", "type": "authentication_error"}})
    with pytest.raises(GatewayError, match="rejected the API key"):
        env.gw.models()
    env.http.routes[("GET", "/models")] = FakeResponse(500, None, text="boom")
    with pytest.raises(GatewayError, match="HTTP 500: boom"):
        env.gw.models()


def test_unreachable_and_not_running(env) -> None:
    with pytest.raises(GatewayError, match="not running"):
        env.gw.models()
    started(env)
    env.http.routes[("GET", "/models")] = httpx.ConnectError("refused")
    with pytest.raises(GatewayError, match=r"http://127.0.0.1:8080 is not reachable: refused"):
        env.gw.models()
    env.http.routes[("POST", "/models/load")] = httpx.ReadTimeout("slow")
    with pytest.raises(GatewayError, match="not reachable"):
        env.gw.load("x")


def test_reload_rewrites_the_ini_then_asks_the_router(env) -> None:
    started(env)
    env.cfg.models[str(env.qwen)].presets["chat"].profile.ctx_size = 12345
    env.http.routes[("GET", "/models?reload=1")] = FakeResponse(200, {"data": [
        entry("qwen3-8b", {"value": "unloaded"})]})
    assert env.gw.reload() == [GatewayModel("qwen3-8b", "unloaded")]
    assert "ctx-size = 12345" in (env.cfg_dir / "gateway.ini").read_text()
    assert env.http.requests == [{"method": "GET", "path": "/models?reload=1"}]


def test_reload_when_stopped_only_rewrites_the_ini(env) -> None:
    assert env.gw.reload() == [] and env.http.requests == []
    assert (env.cfg_dir / "gateway.ini").exists()


def test_write_presets_collects_problems(env) -> None:
    env.cfg.models[str(env.qwen)].presets["agent"].profile = Profile(extra_args="oops")
    path = env.gw.write_presets()
    assert path == env.cfg_dir / "gateway.ini" and len(env.gw.problems) == 1
    env.cfg.models[str(env.qwen)].presets["agent"].profile = Profile()
    env.gw.write_presets()
    assert env.gw.problems == []
