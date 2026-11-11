"""ServerRunner over fake popen / http / devices / clock: no llama.cpp, no network, no sleeping."""
from __future__ import annotations

import os
import signal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import httpx
import pytest

from llamactl import gateway, tuner
from llamactl.config import Config, Profile
from llamactl.hardware import Device
from llamactl.measure import ERR_ARGS, ERR_BENCH, ERR_LOAD, ERR_OOM, ERR_TIMEOUT
from llamactl.tuner import ServerRunner, classify_failure, health_timeout, parse_build

GB = 1e9
PID = 4242

TIMINGS = {"timings": {"prompt_n": 120, "prompt_per_second": 1850.5, "predicted_n": 128,
                       "predicted_per_second": 61.25}}


class Resp:
    def __init__(self, status: int = 200, body: Any = None, text: str = "") -> None:
        self.status_code, self._body, self.text = status, body, text

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no json")
        return self._body


class World:
    """One fake machine: process table, clock, device memory, HTTP routes."""

    def __init__(self) -> None:
        self.alive: set[int] = set()
        self.now = 0.0
        self.signals: list[tuple[int, int]] = []
        self.popens: list[tuple[list[str], dict[str, Any]]] = []
        self.requests: list[dict[str, Any]] = []
        self.clients: list[dict[str, Any]] = []
        self.versions = 0
        self.free_seq: list[float] = [16 * GB]       # successive free-VRAM readings
        self.ram_seq: list[float] = [20 * GB]
        self.ignore_sigterm = False
        self.ready_after = 3.0                         # seconds until /health answers 200
        self.exit_text: str | None = None              # process dies at once, leaving this log
        self.routes: dict[tuple[str, str], Any] = {}
        self.sleeps = 0

    # fakes handed to ServerRunner ----------------------------------------------------
    def popen(self, argv: list[str], **kw: Any) -> SimpleNamespace:
        self.popens.append((argv, kw))
        if self.exit_text is not None:
            Path(kw["stdout"].name).write_text(self.exit_text)
        else:
            self.alive.add(PID)
        return SimpleNamespace(pid=PID)

    def http(self, **kw: Any) -> "World":
        self.clients.append(kw)
        return self

    def __enter__(self) -> "World":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def request(self, method: str, path: str, **kw: Any) -> Any:
        self.requests.append({"method": method, "path": path, **kw})
        route = self.routes.get((method, path))
        if route is not None:
            return route(kw) if callable(route) else route
        if path == "/health":
            if self.now < self.ready_after:
                return Resp(503, {"error": {"message": "Loading model"}})
            return Resp(200, {"status": "ok"})
        raise AssertionError(f"unexpected request {method} {path}")

    def devices(self) -> list[Device]:
        free = self.free_seq.pop(0) if len(self.free_seq) > 1 else self.free_seq[0]
        return [Device.of("CUDA0", "G0", int(16 * GB), int(free))]

    def ram(self) -> int:
        return int(self.ram_seq.pop(0) if len(self.ram_seq) > 1 else self.ram_seq[0])

    def run(self, argv: list[str], **kw: Any) -> SimpleNamespace:
        self.versions += 1
        return SimpleNamespace(stdout="", stderr="version: 0.5.0-dev (build 1, commit 5fc4f3c)\n"
                                                 "built with GNU 15.2.0\n")

    def sleep(self, s: float) -> None:
        self.sleeps += 1
        self.now += s

    def clock(self) -> float:
        return self.now


@pytest.fixture
def world(isolated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> World:
    w = World()
    monkeypatch.setattr(tuner, "pid_alive", lambda pid: pid in w.alive)
    monkeypatch.setattr(gateway, "pid_alive", lambda pid: pid in w.alive)
    monkeypatch.setattr(tuner, "next_free_port", lambda taken, host="127.0.0.1": w.ports.append(
        (set(taken), host)) or 18080)
    w.ports = []                                                # type: ignore[attr-defined]
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)

    def killpg(pgid: int, sig: int) -> None:
        w.signals.append((pgid, sig))
        if sig == signal.SIGKILL or not w.ignore_sigterm:
            w.alive.discard(pgid)

    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(tuner, "_now", lambda: "2026-11-09T10:00:00+00:00")
    return w


@pytest.fixture
def model(tmp_path: Path) -> Path:
    p = tmp_path / "Tiny-Q8_0.gguf"
    p.write_bytes(b"x" * 1000)
    return p


def make(world: World, **kw: Any) -> ServerRunner:
    cfg = Config(bin_dir="/opt/llama/bin")
    return ServerRunner(cfg, popen=world.popen, http=world.http, devices=world.devices,
                        run=world.run, ram_free=world.ram, clock=world.clock, sleep=world.sleep,
                        **kw)


def completion_ok(world: World, body: Any = None) -> None:
    world.routes[("POST", "/completion")] = Resp(200, body or TIMINGS)


# --------------------------------------------------------------------------- happy path

def test_happy_path_records_numbers(world: World, model: Path) -> None:
    completion_ok(world)
    world.free_seq = [16 * GB, 7 * GB, 6.5 * GB]       # before, after load, after benchmark
    world.ram_seq = [20 * GB, 19.5 * GB, 19.4 * GB]
    m = make(world).measure(model, Profile(ctx_size=8192))
    assert m.ok and m.error == ""
    assert (m.prompt_tps, m.gen_tps) == (1850.5, 61.25)
    assert m.vram_used == int(9.5 * GB)                 # the larger of the two deltas
    assert m.ram_used == int(0.6 * GB)
    assert m.load_s == pytest.approx(3.0)               # 0.5 s polls until 3 s
    assert m.build == "0.5.0-dev (build 1, commit 5fc4f3c)"
    assert m.at == "2026-11-09T10:00:00+00:00"


def test_benchmark_request_is_fixed_and_deterministic(world: World, model: Path) -> None:
    completion_ok(world)
    make(world).measure(model, Profile())
    (post,) = [r for r in world.requests if r["method"] == "POST"]
    assert post["path"] == "/completion"
    body = post["json"]
    assert body["n_predict"] == 128 and body["cache_prompt"] is False
    assert body["seed"] == 42 and body["temperature"] == 0 and body["stream"] is False
    # ctx_size 0 -> safe 4096 per slot -> ~2k-token deterministic prompt
    assert body["prompt"] == tuner.bench_prompt(4096) and len(body["prompt"]) > 4000


def test_launch_overrides_host_port_key_and_runs_detached(world: World, model: Path) -> None:
    completion_ok(world)
    prof = Profile(host="0.0.0.0", port=9999, api_key="secret", sleep_idle=300, ctx_size=4096,
                   alias="mine")
    make(world).measure(model, prof)
    argv, kw = world.popens[0]
    assert argv[:3] == ["/opt/llama/bin/llama-server", "-m", str(model)]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "18080"
    assert "--api-key" not in argv and "secret" not in argv and "--sleep-idle-seconds" not in argv
    assert argv[argv.index("-c") + 1] == "4096"
    assert kw["start_new_session"] is True and kw["stdin"] is not None
    assert world.ports == [(set(), "127.0.0.1")]
    assert world.clients[0]["base_url"] == "http://127.0.0.1:18080"
    assert prof.host == "0.0.0.0" and prof.port == 9999          # the caller's profile is untouched


def test_log_lives_in_the_log_dir_and_is_removed_on_success(world: World, model: Path,
                                                            isolated: SimpleNamespace) -> None:
    completion_ok(world)
    make(world).measure(model, Profile())
    log = Path(world.popens[0][1]["stdout"].name)
    assert log.parent == isolated.cfg_dir / "logs" and log.name.startswith("tune-Tiny-Q8_0-")
    assert not log.exists()


def test_version_is_asked_once(world: World, model: Path) -> None:
    completion_ok(world)
    runner = make(world)
    runner.measure(model, Profile())
    world.alive.clear()
    runner.measure(model, Profile())
    assert world.versions == 1 and runner.build().startswith("0.5.0-dev")


def test_unknown_version_is_empty_not_fatal(world: World, model: Path) -> None:
    completion_ok(world)

    def boom(argv: list[str], **kw: Any) -> None:
        raise FileNotFoundError("llama-server")

    runner = ServerRunner(Config(), popen=world.popen, http=world.http, devices=world.devices,
                          run=boom, ram_free=world.ram, clock=world.clock, sleep=world.sleep)
    m = runner.measure(model, Profile())
    assert m.ok and m.build == ""


# --------------------------------------------------------------------------- stopping

def test_process_group_is_stopped_after_success(world: World, model: Path) -> None:
    completion_ok(world)
    make(world).measure(model, Profile())
    assert world.signals == [(PID, signal.SIGTERM)] and PID not in world.alive


def test_stubborn_process_gets_sigkill_and_is_waited_for(world: World, model: Path) -> None:
    completion_ok(world)
    world.ignore_sigterm = True
    make(world).measure(model, Profile())
    assert world.signals == [(PID, signal.SIGTERM), (PID, signal.SIGKILL)]
    assert PID not in world.alive


def test_process_is_stopped_when_the_http_call_raises(world: World, model: Path) -> None:
    def explode(kw: dict) -> Resp:
        raise httpx.ReadTimeout("slow")

    world.routes[("POST", "/completion")] = explode
    m = make(world).measure(model, Profile())
    assert not m.ok and m.error.startswith(ERR_BENCH) and "slow" in m.error
    assert PID not in world.alive and world.signals[0] == (PID, signal.SIGTERM)


def test_process_is_stopped_even_for_an_unexpected_exception(world: World, model: Path) -> None:
    def explode(kw: dict) -> Resp:
        raise RuntimeError("bug")

    world.routes[("POST", "/completion")] = explode
    with pytest.raises(RuntimeError):
        make(world).measure(model, Profile())
    assert PID not in world.alive and world.signals == [(PID, signal.SIGTERM)]


def test_process_is_stopped_on_keyboard_interrupt(world: World, model: Path) -> None:
    def interrupt(kw: dict) -> Resp:
        raise KeyboardInterrupt

    world.routes[("POST", "/completion")] = interrupt
    with pytest.raises(KeyboardInterrupt):
        make(world).measure(model, Profile())
    assert PID not in world.alive


def test_an_already_dead_process_is_not_signalled(world: World, model: Path) -> None:
    """A recycled pid must never be signalled."""
    world.exit_text = "boom"
    make(world).measure(model, Profile())
    assert world.signals == []


# --------------------------------------------------------------------------- failures

def test_health_timeout(world: World, model: Path) -> None:
    world.ready_after = 1e9
    m = make(world).measure(model, Profile())
    assert not m.ok and m.error.startswith(ERR_TIMEOUT) and m.resource_failure
    assert m.load_s >= health_timeout(1000) and m.vram_used == 0
    assert PID not in world.alive and world.signals[0] == (PID, signal.SIGTERM)
    assert all(r["path"] == "/health" for r in world.requests)


def test_connection_errors_while_starting_are_retried(world: World, model: Path) -> None:
    completion_ok(world)
    calls = {"n": 0}

    def health(kw: dict) -> Resp:
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("refused")
        return Resp(200, {"status": "ok"})

    world.routes[("GET", "/health")] = health
    assert make(world).measure(model, Profile()).ok and calls["n"] == 3


OOM_LOG = """\
load_tensors: offloading 32 repeating layers to GPU
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9500.00 MiB on device 0: cudaMalloc failed: out of memory
llama_model_load: error loading model: unable to allocate CUDA0 buffer
srv    load_model: failed to load model, '/m/x.gguf'
"""


def test_early_exit_with_oom_log_is_classified(world: World, model: Path) -> None:
    world.exit_text = OOM_LOG
    m = make(world).measure(model, Profile())
    assert not m.ok and m.error.startswith(f"{ERR_OOM}: ") and "cudaMalloc failed" in m.error
    assert "(log: " in m.error and m.resource_failure
    assert not [r for r in world.requests if r["method"] == "POST"]
    assert Path(world.popens[0][1]["stdout"].name).exists()       # failure logs are kept


def test_early_exit_with_unknown_argument(world: World, model: Path) -> None:
    world.exit_text = "error: invalid argument: --bogus\n"
    m = make(world).measure(model, Profile())
    assert m.error.startswith(f"{ERR_ARGS}: ") and "--bogus" in m.error and not m.resource_failure


def test_early_exit_with_other_reason(world: World, model: Path) -> None:
    world.exit_text = ("llama_model_load: error loading model architecture: unknown model "
                       "architecture: 'clef'\nsrv load_model: failed to load model\n")
    m = make(world).measure(model, Profile())
    assert m.error.startswith(f"{ERR_LOAD}: ") and not m.resource_failure
    assert "unknown model architecture: 'clef'" in m.error


REAL_UNKNOWN_ARCH_LOG = """\
0.00.078.575 I srv  llama_server: initializing ...
0.00.078.895 W srv  llama_server: security: no API key is set and CORS allows all origins
0.00.082.327 I srv    load_model: loading model '/home/u/models/clef-flash-Q8_0.gguf'
0.00.175.656 E llama_model_load: error loading model: unknown model architecture: 'clef'
0.00.175.675 E llama_model_load_from_file_impl: failed to load model
0.00.175.687 E srv    load_model: failed to load model, '/home/u/models/clef-flash-Q8_0.gguf'
0.00.175.689 I srv    operator(): operator(): cleaning up before exit...
0.00.176.331 E srv  llama_server: exiting due to model loading error
"""


def test_real_unknown_architecture_log_names_the_root_cause(world: World, model: Path) -> None:
    """Seen with llama.cpp 5fc4f3c: the last error line is generic, the first says why."""
    world.exit_text = REAL_UNKNOWN_ARCH_LOG
    m = make(world).measure(model, Profile())
    assert m.error.startswith(f"{ERR_LOAD}: llama_model_load: error loading model: "
                              "unknown model architecture: 'clef' (log: ")


def test_empty_log_still_gives_a_reason(world: World, model: Path) -> None:
    world.exit_text = ""
    assert make(world).measure(model, Profile()).error.startswith(f"{ERR_LOAD}: no log output")


def test_process_dying_while_loading(world: World, model: Path) -> None:
    """It starts, then dies before /health answers (killed by the OOM killer, say)."""
    world.ready_after = 1e9
    real_sleep = world.sleep

    def sleep(s: float) -> None:
        real_sleep(s)
        if world.now >= 1.0:
            Path(world.popens[0][1]["stdout"].name).write_text("terminate called: std::bad_alloc\n")
            world.alive.discard(PID)

    runner = ServerRunner(Config(), popen=world.popen, http=world.http, devices=world.devices,
                          run=world.run, ram_free=world.ram, clock=world.clock, sleep=sleep)
    m = runner.measure(model, Profile())
    assert m.error.startswith(ERR_OOM) and m.load_s == pytest.approx(1.0)


def test_cannot_start_the_binary(world: World, model: Path) -> None:
    def popen(argv: list[str], **kw: Any) -> None:
        raise FileNotFoundError("no such file")

    runner = ServerRunner(Config(bin_dir="/nope"), popen=popen, http=world.http,
                          devices=world.devices, run=world.run, ram_free=world.ram,
                          clock=world.clock, sleep=world.sleep)
    m = runner.measure(model, Profile())
    assert not m.ok and m.error.startswith(f"{ERR_LOAD}: cannot start /nope/llama-server")
    assert world.signals == []


@pytest.mark.parametrize("resp", [Resp(500, None, "boom"), Resp(200, {"content": "x"}),
                                  Resp(200, {"timings": {"predicted_per_second": 0}}),
                                  Resp(200, None)])
def test_bad_benchmark_answers(world: World, model: Path, resp: Resp) -> None:
    world.routes[("POST", "/completion")] = resp
    m = make(world).measure(model, Profile())
    assert not m.ok and m.error.startswith(ERR_BENCH) and not m.resource_failure
    assert PID not in world.alive


def test_missing_prompt_speed_is_tolerated(world: World, model: Path) -> None:
    completion_ok(world, {"timings": {"predicted_per_second": 33.0}})
    m = make(world).measure(model, Profile())
    assert m.ok and m.gen_tps == 33.0 and m.prompt_tps == 0.0


# --------------------------------------------------------------------------- embeddings

def test_embedding_models_use_the_embeddings_endpoint(world: World, model: Path) -> None:
    def embed(kw: dict) -> Resp:
        world.now += 0.5                                    # the request takes half a second
        return Resp(200, {"data": [{"embedding": [0.1, 0.2]}], "usage": {"prompt_tokens": 150,
                                                                         "total_tokens": 150}})

    world.routes[("POST", "/v1/embeddings")] = embed
    m = make(world).measure(model, Profile(embedding=True, parallel=4))
    assert m.ok and m.prompt_tps == pytest.approx(300.0) and m.gen_tps == 0.0
    paths = [r["path"] for r in world.requests if r["method"] == "POST"]
    assert paths == ["/v1/embeddings"]
    body = world.requests[-1]["json"]
    assert body["input"] == tuner.EMBED_TEXT


def test_embedding_without_usage_just_times_the_request(world: World, model: Path) -> None:
    world.routes[("POST", "/v1/embeddings")] = Resp(200, {"data": [{"embedding": [0.1]}]})
    m = make(world).measure(model, Profile(embedding=True))
    assert m.ok and m.prompt_tps == 0.0 and m.gen_tps == 0.0


def test_embedding_error_status(world: World, model: Path) -> None:
    world.routes[("POST", "/v1/embeddings")] = Resp(400, None, "pooling none")
    m = make(world).measure(model, Profile(embedding=True))
    assert not m.ok and m.error.startswith(ERR_BENCH) and "400" in m.error


# --------------------------------------------------------------------------- pure helpers

def test_health_timeout_scales_with_model_size() -> None:
    assert health_timeout(0) == 60.0
    assert health_timeout(int(10 * GB)) == pytest.approx(260.0)
    assert health_timeout(int(1e6 * GB)) == tuner.HEALTH_MAX


def test_parse_build() -> None:
    text = "ggml_cuda_init: found 1 CUDA devices\nversion: 0.5.0-dev (build 1, commit 5fc4f3c)\nbuilt"
    assert parse_build(text) == "0.5.0-dev (build 1, commit 5fc4f3c)"
    assert parse_build("") == "" and parse_build("hello") == ""


@pytest.mark.parametrize("log,kind", [
    ("cudaMalloc failed: out of memory", ERR_OOM),
    ("0.06.1 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 500000.00 MiB on device 0: "
     "cudaMalloc failed: out of memory", ERR_OOM),
    ("llama_kv_cache: failed to allocate buffer for kv cache", ERR_OOM),
    ("main: failed to fit params to free device memory", ERR_OOM),
    ("vk::Device::allocateMemory: ErrorOutOfDeviceMemory", ERR_OOM),
    ("error: invalid argument: -zz", ERR_ARGS),
    ("error: unknown option --foo", ERR_ARGS),
    ("something odd happened", ERR_LOAD),
    ("", ERR_LOAD),
])
def test_classify_failure(log: str, kind: str) -> None:
    assert classify_failure(log).startswith(kind + ": ")


def test_classify_prefers_memory_over_generic_failure_and_truncates() -> None:
    log = "srv: failed to load model\n" + "x" * 500 + " out of memory\nfailed again"
    out = classify_failure(log)
    assert out.startswith(ERR_OOM) and len(out) < 260


# --------------------------------------------------------------------------- cancel

def test_cancel_before_launch_starts_nothing(world: World, model: Path) -> None:
    m = make(world).measure(model, Profile(), cancel=lambda: True)
    assert m.cancelled and not m.ok and not m.resource_failure
    assert world.popens == [] and world.signals == []


def test_cancel_while_waiting_for_health_stops_the_server(world: World, model: Path) -> None:
    world.ready_after = 100.0                              # never ready on its own
    polls = {"n": 0}

    def cancel() -> bool:
        polls["n"] += 1
        return polls["n"] > 3

    m = make(world).measure(model, Profile(), cancel=cancel)
    assert m.cancelled and m.error.startswith("cancelled: stopped while the model was loading")
    assert (PID, signal.SIGTERM) in world.signals and PID not in world.alive
    assert [r for r in world.requests if r["method"] == "POST"] == []
    log = Path(world.popens[0][1]["stdout"].name)
    assert not log.exists()                                # nothing worth keeping


def test_cancel_after_load_skips_the_benchmark(world: World, model: Path) -> None:
    completion_ok(world)
    state = {"ready": False}
    orig = world.request

    def request(method: str, path: str, **kw: Any) -> Any:
        r = orig(method, path, **kw)
        state["ready"] = state["ready"] or (path == "/health" and r.status_code == 200)
        return r

    world.request = request                                # type: ignore[method-assign]
    m = make(world).measure(model, Profile(), cancel=lambda: state["ready"])
    assert m.cancelled and "before the benchmark" in m.error
    assert [r for r in world.requests if r["method"] == "POST"] == []
    assert PID not in world.alive


def test_a_cancel_that_never_fires_changes_nothing(world: World, model: Path) -> None:
    completion_ok(world)
    assert make(world).measure(model, Profile(), cancel=lambda: False).ok
