"""CLI tests: `cli.main` driven with synthetic GGUF files and fakes for the gateway, the session
store and the tune runner. No llama.cpp, no network, no real sleeping."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from conftest import MINI_KVS, gguf_blob, kv_u32
from llamactl import cli, gateway, render, sessions
from llamactl.calibration import Calibration
from llamactl.config import Config, Preset, Profile
from llamactl.gateway import FAILED, LOADED, LOADING, UNLOADED, GatewayError, GatewayModel, GatewayState, GatewayStatus
from llamactl.hardware import Device, Host
from llamactl.measure import ERR_OOM, Measurement
from llamactl.sessions import Session

GB = 1e9
BUILD = "0.5.0-test (build 1, commit abc)"


# --------------------------------------------------------------------------- fakes

class FakeClock:
    """A monotonic clock that only moves when the code sleeps."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps = 0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds
        self.sleeps += 1


class FakeGateway:
    """Gateway double: renders the real preset INI, models move loading -> `outcome`."""

    def __init__(self, cfg: Config, log: Path) -> None:
        self.cfg, self.log = cfg, log
        self.is_running = False
        self.calls: list[Any] = []
        self.problems: list[str] = []
        self.states: dict[str, str] = {}
        self.outcome = LOADED            # what a loading model becomes after `polls` model listings
        self.polls = 2
        self._left: dict[str, int] = {}
        self.start_error: str = ""
        self.healthy = True
        self.listen_url = "http://127.0.0.1:8080"
        self.exit_code = 1
        self.unload_polls = 0            # listings that still show the model after an unload
        self._unloading: dict[str, int] = {}
        self.host_override: str | None = None
        self.bound_host = ""             # the listen address of the running router

    @property
    def state(self) -> GatewayState | None:
        return GatewayState(4242, self.listen_url, str(self.log), 1000.0, self.bound_host) \
            if self.is_running else None

    @property
    def running(self) -> bool:
        return self.is_running

    def write_presets(self) -> Path:
        self.problems = []
        gateway.PRESETS_FILE.parent.mkdir(parents=True, exist_ok=True)
        gateway.PRESETS_FILE.write_text(gateway.render_presets(self.cfg, problems=self.problems))
        return gateway.PRESETS_FILE

    def start(self) -> GatewayState:
        if self.is_running:
            return self.state
        if self.start_error:
            raise GatewayError(self.start_error)
        self.calls.append("start")
        self.bound_host = self.cfg.gateway.host if self.host_override is None else self.host_override
        self.write_presets()
        self.log.parent.mkdir(parents=True, exist_ok=True)
        if not self.log.exists():
            self.log.write_text("gateway up\n")
        self.is_running = True
        return self.state

    def stop(self) -> bool:
        was, self.is_running = self.is_running, False
        self.calls.append("stop")
        return was

    def status(self) -> GatewayStatus:
        return GatewayStatus(self.is_running, self.listen_url, self.is_running and self.healthy,
                             4242 if self.is_running else 0, self.bound_host)

    def names(self) -> list[str]:
        text = gateway.PRESETS_FILE.read_text() if gateway.PRESETS_FILE.exists() else ""
        return [n for n in re.findall(r"^\[(.+)\]$", text, re.M) if n != "*"]

    def models(self) -> list[GatewayModel]:
        out = []
        for n in self.names():
            if n in self._unloading:
                self._unloading[n] -= 1
                if self._unloading[n] <= 0:
                    self.states[n] = UNLOADED
                    del self._unloading[n]
            if self.states.get(n) == LOADING:
                self._left[n] = self._left.get(n, self.polls) - 1
                if self._left[n] <= 0:
                    self.states[n] = self.outcome
            code = self.exit_code if self.states.get(n) == FAILED else None
            out.append(GatewayModel(n, self.states.get(n, UNLOADED), code))
        return out

    def reload(self) -> list[GatewayModel]:
        self.calls.append("reload")
        self.write_presets()
        return self.models() if self.is_running else []

    def load(self, name: str) -> None:
        if name not in self.names():
            raise GatewayError(f"cannot load '{name}': unknown model")
        self.calls.append(("load", name))
        self.states[name] = LOADING
        self._left[name] = self.polls

    def unload(self, name: str) -> None:
        self.calls.append(("unload", name))
        if self.unload_polls:
            self._unloading[name] = self.unload_polls
        else:
            self.states[name] = UNLOADED


class FakeStore:
    """Session store double."""

    def __init__(self, log_dir: Path, alive: set[int]) -> None:
        self.alive, self.dies = alive, False
        self.sessions: list[Session] = []
        self.launched: list[tuple] = []
        self.stopped: list[str] = []
        self.hosts: list[str | None] = []
        self.log_dir = log_dir

    def launch_server(self, cfg: Config, model: str, preset_name: str | None = None,
                      port: int | None = None, host: str | None = None) -> Session:
        port = port or 8081
        self.launched.append((model, preset_name, port))
        self.hosts.append(host)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = self.log_dir / f"{Path(model).stem}-{port}.log"
        log.write_text("server log\n")
        s = Session(f"{Path(model).stem}-{port}", Path(model).stem, f"http://127.0.0.1:{port}",
                    pid=5000 + port, log=str(log), started=900.0,
                    host=host if host is not None else cfg.profile_for(model).host)
        self.sessions.append(s)
        if not self.dies:
            self.alive.add(s.pid)
        return s

    def stop(self, s: Session) -> None:
        self.stopped.append(s.id)
        self.sessions = [x for x in self.sessions if x.id != s.id]


class FakeRunner:
    """Tune runner double: answers in order, repeating the last; an Exception instance is raised."""

    def __init__(self, *answers: Measurement | BaseException) -> None:
        self.answers, self.profiles = list(answers), []

    def measure(self, model: Path, profile: Profile, cancel=None) -> Measurement:
        self.profiles.append(profile)
        a = self.answers[min(len(self.profiles), len(self.answers)) - 1]
        if isinstance(a, BaseException):
            raise a
        return a


def good(vram: float = 4.0, gen: float = 60.0) -> Measurement:
    return Measurement(ok=True, vram_used=int(vram * GB), gen_tps=gen, prompt_tps=900.0,
                       load_s=4.0, at="2026-11-09T10:00:00+00:00", build=BUILD)


def oom() -> Measurement:
    return Measurement(error=f"{ERR_OOM}: cudaMalloc failed", build=BUILD)


class Tty:
    """A writable stream that claims to be a terminal."""

    def __init__(self) -> None:
        self.parts: list[str] = []

    def write(self, s: str) -> int:
        self.parts.append(s)
        return len(s)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return True

    @property
    def text(self) -> str:
        return "".join(self.parts)


# --------------------------------------------------------------------------- world

def _model_file(path: Path, kvs: list[bytes]) -> Path:
    """A sparse >50 MB file starting with a valid GGUF header."""
    with path.open("wb") as fh:
        fh.write(gguf_blob(kvs))
        fh.truncate(60 * 1024 * 1024)
    return path


@pytest.fixture
def world(isolated: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Two text models, a fake llama.cpp folder, a config and the fakes, all inside tmp_path."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    server = bin_dir / "llama-server"
    server.write_text(f"#!/bin/sh\necho 'version: {BUILD}'\n")
    server.chmod(0o755)
    qwen = _model_file(isolated.model_dir / "Qwen3-8B-Q4_K_M.gguf", MINI_KVS)
    llama = _model_file(isolated.model_dir / "llama-3-q8_0.gguf", MINI_KVS)
    cfg = Config(bin_dir=str(bin_dir), model_dirs=[str(isolated.model_dir)])
    alive: set[int] = set()
    monkeypatch.setattr(sessions, "pid_alive", lambda pid: pid in alive)
    clock = FakeClock()
    gw = FakeGateway(cfg, isolated.cfg_dir / "logs" / "gateway.log")
    store = FakeStore(isolated.cfg_dir / "logs", alive)
    w = SimpleNamespace(cfg=cfg, gw=gw, store=store, clock=clock, alive=alive, qwen=qwen,
                        llama=llama, bin_dir=bin_dir, cfg_dir=isolated.cfg_dir,
                        model_dir=isolated.model_dir, host=isolated.host, runner=FakeRunner(good()),
                        health="ready", build=BUILD)

    def services(**kw: Any) -> cli.Services:
        opts: dict[str, Any] = dict(
            colour=False, detect=lambda: w.host, refresh=lambda h: w.host,
            cache=None, calibration=Calibration(), gateway=w.gw, store=w.store,
            runner_factory=lambda: w.runner, build=lambda: w.build,
            health=lambda url: w.health, sleep=clock.sleep, clock=clock, now=lambda: 1060.0)
        opts.update(kw)
        return cli.Services(w.cfg, **opts)

    def run(capsys: pytest.CaptureFixture[str], *argv: str, **kw: Any) -> SimpleNamespace:
        try:
            read_only = cli.is_read_only(cli.build_parser().parse_args(list(argv)))
        except SystemExit:
            read_only = False
        code = cli.main(list(argv), services(read_only=read_only, **kw))
        cap = capsys.readouterr()
        return SimpleNamespace(code=code, out=cap.out, err=cap.err)

    w.services, w.run = services, run
    return w


def saved_config(w: SimpleNamespace) -> dict:
    """The config as saved on disk."""
    return json.loads((w.cfg_dir / "config.json").read_text())


def preset_names(w: SimpleNamespace, path: Path) -> dict[str, str]:
    """Preset name -> source for a model, from the saved config."""
    ent = saved_config(w)["models"][str(path)]
    return {n: p["source"] for n, p in ent["presets"].items()}


# --------------------------------------------------------------------------- basics

def test_no_command_launches_the_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    import llamactl.app
    calls: list[str] = []
    monkeypatch.setattr(llamactl.app, "main", lambda: calls.append("tui"))
    assert cli.main([]) == 0 and calls == ["tui"]


def test_version_and_usage_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.startswith("llamactl ")
    assert cli.main(["nonsense"]) == 2
    assert cli.main(["plan"]) == 2                       # MODEL is required
    assert cli.main(["tune", "m", "--for", "bogus"]) == 2
    assert cli.main(["tune", "m", "--trials", "0"]) == 2
    capsys.readouterr()


def test_importing_the_cli_does_not_import_textual() -> None:
    code = ("import sys, llamactl.cli; "
            "sys.exit(int('textual' in sys.modules or 'rich' in sys.modules))")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_help_for_every_command(capsys: pytest.CaptureFixture[str]) -> None:
    for n in ("hw", "ls", "plan", "tune", "serve", "up", "down", "config", "ps", "load", "unload",
              "stop", "logs"):
        assert cli.main([n, "--help"]) == 0
        assert capsys.readouterr().out.startswith(f"usage: llamactl {n}")


# --------------------------------------------------------------------------- hw

def test_hw_text(world, capsys) -> None:
    r = world.run(capsys, "hw")
    assert r.code == 0 and "\x1b" not in r.out
    for want in ("CUDA0", "cuda", "GPU", "16.0 GB", "Test CPU", "8 cores", "x1.00",
                 "nothing measured", BUILD, str(world.bin_dir)):
        assert want in r.out


def test_hw_json_is_one_document(world, capsys) -> None:
    r = world.run(capsys, "hw", "--json")
    d = json.loads(r.out)                                # whole stdout parses: nothing else there
    assert r.code == 0 and r.err == ""
    assert d["devices"][0] == {"id": "CUDA0", "backend": "cuda", "name": "GPU",
                               "total": int(16e9), "free": int(16e9)}
    assert d["ram"]["total"] == int(24e9) and d["unified"] is False
    assert d["cpu"] == {"name": "Test CPU", "threads": 16, "cores": 8}
    assert d["calibration"]["cuda"]["factor"] == 1.0
    assert d["llama_cpp"] == {"bin_dir": str(world.bin_dir), "found": True, "build": BUILD}


def test_hw_without_llama_cpp_still_works(world, capsys) -> None:
    world.cfg.bin_dir = str(world.bin_dir / "nowhere")
    r = world.run(capsys, "hw")
    assert r.code == 0 and "llama-server not found" in r.out


# --------------------------------------------------------------------------- ls

def test_ls_rows_fits_presets_and_errors(world, capsys) -> None:
    _model_file(world.model_dir / "broken.gguf", [])      # valid size, header without metadata
    (world.model_dir / "junk.gguf").write_bytes(b"\0" * (60 * 1024 * 1024))
    q = str(world.qwen)
    world.cfg.set_preset(q, Preset("chat", Profile(ctx_size=8192), "tuned",
                                   Measurement(ok=True, gen_tps=50, at="t", build="old-build")))
    world.cfg.set_preset(q, Preset("agent", Profile(), "estimated"), activate=False)
    r = world.run(capsys, "ls")
    assert r.code == 0
    lines = {ln.split()[0]: ln for ln in r.out.splitlines()[1:]}
    assert set(lines) == {"broken", "junk", "llama-3-q8_0", "qwen3-8b-q4_k_m"}
    assert "chat GPU 8k" in lines["qwen3-8b-q4_k_m"] and "speed GPU" in lines["qwen3-8b-q4_k_m"]
    assert "Q4_K_M" in lines["qwen3-8b-q4_k_m"]
    assert "*chat (tuned, stale)" in lines["qwen3-8b-q4_k_m"]      # build differs from BUILD
    assert "agent (estimated)" in lines["qwen3-8b-q4_k_m"]
    assert "unreadable" in lines["junk"] and "not a GGUF" in lines["junk"]
    assert "unreadable" in lines["broken"]


def test_ls_json(world, capsys) -> None:
    (world.model_dir / "junk.gguf").write_bytes(b"\0" * (60 * 1024 * 1024))
    r = world.run(capsys, "ls", "--json")
    rows = {m["alias"]: m for m in json.loads(r.out)["models"]}
    q = rows["qwen3-8b-q4_k_m"]
    assert q["error"] == "" and q["quant"] == "Q4_K_M" and q["size"] > 50e6
    assert [f["intent"] for f in q["fits"]] == ["chat", "agent", "context", "speed", "api"]
    assert q["fits"][0]["verdict"] == "GPU" and q["presets"] == []
    assert rows["junk"]["error"] and rows["junk"]["fits"] == []


def test_ls_with_no_models(world, capsys) -> None:
    world.cfg.model_dirs = [str(world.cfg_dir / "empty")]
    r = world.run(capsys, "ls")
    assert r.code == 0 and "no models found" in r.out
    assert json.loads(world.run(capsys, "ls", "--json").out) == {"models": []}


def test_ls_marks_embedding_models(world, capsys) -> None:
    _model_file(world.model_dir / "embed-f16.gguf", MINI_KVS + [kv_u32("testarch.pooling_type", 1)])
    rows = {m["alias"]: m for m in json.loads(world.run(capsys, "ls", "--json").out)["models"]}
    assert [f["intent"] for f in rows["embed-f16"]["fits"]] == ["embed"]


# --------------------------------------------------------------------------- model resolution

@pytest.mark.parametrize("argv", [["plan", "nope"], ["tune", "nope"], ["serve", "nope"],
                                  ["plan", "q", "--all"]])
def test_unknown_and_ambiguous_models_exit_2(world, capsys, argv: list[str]) -> None:
    _model_file(world.model_dir / "Qwen3-14B-Q4_K_M.gguf", MINI_KVS)    # "q" now matches two
    r = world.run(capsys, *argv)
    assert r.code == 2 and r.out == ""
    assert r.err.startswith("llamactl: error: ")
    assert "Available" in r.err or "ambiguous" in r.err


def test_model_resolves_by_alias_stem_and_substring(world, capsys) -> None:
    for q in ("qwen3-8b-q4_k_m", "Qwen3-8B-Q4_K_M", "qwen3"):
        r = world.run(capsys, "plan", q, "--json")
        assert r.code == 0 and json.loads(r.out)["model"]["alias"] == "qwen3-8b-q4_k_m"


def test_unreadable_gguf_is_a_runtime_error(world, capsys) -> None:
    (world.model_dir / "junk.gguf").write_bytes(b"\0" * (60 * 1024 * 1024))
    r = world.run(capsys, "plan", "junk")
    assert r.code == 1 and "cannot read junk.gguf" in r.err


# --------------------------------------------------------------------------- plan

def test_plan_text(world, capsys) -> None:
    r = world.run(capsys, "plan", "qwen3")
    assert r.code == 0
    for want in ("qwen3-8b-q4_k_m for chat", "verdict  GPU", "8192 tokens per slot x 1 slot",
                 "Memory", "weights", "KV cache", "needs", "usable VRAM", "Why", "SETTING",
                 "ctx_size", "Command", f"{world.bin_dir}/llama-server -m {world.qwen}"):
        assert want in r.out, want
    assert "-c 8192" in r.out and "\x1b" not in r.out


def test_plan_uses_the_default_intent_and_for_overrides(world, capsys) -> None:
    world.cfg.default_intent = "speed"
    assert json.loads(world.run(capsys, "plan", "qwen3", "--json").out)["intent"]["key"] == "speed"
    d = json.loads(world.run(capsys, "plan", "qwen3", "--for", "api", "--json").out)
    assert d["intent"]["key"] == "api" and d["slots"] == 4
    assert d["ctx_total"] == d["ctx_per_slot"] * 4


def test_plan_json_shape(world, capsys) -> None:
    r = world.run(capsys, "plan", "qwen3", "--for", "agent", "--json")
    d = json.loads(r.out)
    assert r.err == ""
    assert set(d) == {"model", "intent", "verdict", "ctx_per_slot", "slots", "ctx_total", "budget",
                      "memory", "reasons", "warnings", "profile", "existing_preset", "command"}
    assert d["budget"]["total"] == sum(d["budget"][k] for k in ("weights", "kv", "compute", "overhead"))
    assert d["memory"]["vram_usable"] < d["memory"]["vram_total"]
    assert d["profile"]["cache_reuse"] == 256 and "cache_reuse" in d["reasons"]
    assert d["existing_preset"] is None and d["command"].startswith(str(world.bin_dir))


def test_plan_all(world, capsys) -> None:
    r = world.run(capsys, "plan", "qwen3", "--all")
    assert r.code == 0
    for k in ("chat", "agent", "context", "speed", "api"):
        assert re.search(rf"^{k}\s+GPU", r.out, re.M), k
    assert "embed" not in r.out
    d = json.loads(world.run(capsys, "plan", "qwen3", "--all", "--json").out)
    assert [p["intent"] for p in d["plans"]] == ["chat", "agent", "context", "speed", "api"]


def test_plan_embedding_model_defaults_to_embed(world, capsys) -> None:
    _model_file(world.model_dir / "embed-f16.gguf", MINI_KVS + [kv_u32("testarch.pooling_type", 1)])
    assert json.loads(world.run(capsys, "plan", "embed", "--json").out)["intent"]["key"] == "embed"
    assert world.run(capsys, "plan", "embed", "--for", "chat").code == 2
    capsys.readouterr()
    assert world.run(capsys, "plan", "qwen3", "--for", "embed").code == 2


def test_plan_shows_warnings_and_an_existing_preset(world, capsys) -> None:
    world.host = Host(gpus=[Device.of("CUDA0", "GPU", int(0.9 * GB), int(0.9 * GB))],
                      ram_total=int(24e9), ram_free=int(24e9), cpu_threads=16, cpu_cores=8)
    world.cfg.set_preset(str(world.qwen), Preset("chat", Profile(), "tuned"))
    r = world.run(capsys, "plan", "qwen3")
    assert "Warnings" in r.out and "already has a 'chat' preset (tuned)" in r.out


def test_plan_changes_nothing_but_save_stores_an_estimate(world, capsys) -> None:
    assert world.run(capsys, "plan", "qwen3").code == 0
    assert not world.cfg_dir.exists() or list(world.cfg_dir.iterdir()) == []
    r = world.run(capsys, "plan", "qwen3", "--for", "agent", "--save", "--json")
    assert r.code == 0 and json.loads(r.out)["intent"]["key"] == "agent"     # stdout stays pure JSON
    assert "saved as the 'agent' preset" in r.err
    assert preset_names(world, world.qwen) == {"agent": "estimated"}
    assert saved_config(world)["models"][str(world.qwen)]["active"] == "agent"


def test_plan_save_keeps_a_tuned_preset_unless_forced(world, capsys) -> None:
    world.cfg.set_preset(str(world.qwen), Preset("chat", Profile(ctx_size=1234), "tuned"))
    r = world.run(capsys, "plan", "qwen3", "--for", "chat", "--save")
    assert r.code == 1 and "tuned" in r.err and "--force" in r.err
    assert world.cfg.preset(str(world.qwen), "chat").profile.ctx_size == 1234
    assert world.run(capsys, "plan", "qwen3", "--for", "chat", "--save", "--force").code == 0
    assert preset_names(world, world.qwen) == {"chat": "estimated"}


def test_plan_all_with_save_is_a_usage_error(world, capsys) -> None:
    assert world.run(capsys, "plan", "qwen3", "--all", "--save").code == 2


# --------------------------------------------------------------------------- colour

def test_colour_only_on_a_tty_and_never_with_no_color(world, capsys, monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    tty = Tty()
    assert cli.main(["plan", "qwen3"], world.services(out=tty, colour=None)) == 0
    assert "\x1b[" in tty.text
    plain = re.sub(r"\x1b\[[0-9;]*m", "", tty.text)
    assert "verdict  GPU" in plain
    monkeypatch.setenv("NO_COLOR", "1")
    quiet = Tty()
    assert cli.main(["plan", "qwen3"], world.services(out=quiet, colour=None)) == 0
    assert "\x1b" not in quiet.text
    monkeypatch.delenv("NO_COLOR")
    assert cli.main(["plan", "qwen3"], world.services(colour=None)) == 0      # captured: not a TTY
    assert "\x1b" not in capsys.readouterr().out


def test_coloured_tables_stay_aligned(world) -> None:
    world.cfg.set_preset(str(world.qwen), Preset("chat", Profile(), "tuned"))
    coloured, plain = Tty(), Tty()
    assert cli.main(["ls"], world.services(out=coloured, colour=True)) == 0
    assert cli.main(["ls"], world.services(out=plain, colour=False)) == 0
    assert "\x1b[" in coloured.text and "\x1b" not in plain.text
    assert re.sub(r"\x1b\[[0-9;]*m", "", coloured.text) == plain.text


def test_use_colour_rules() -> None:
    tty = Tty()
    assert render.use_colour(tty, {}) and not render.use_colour(tty, {"NO_COLOR": "1"})
    assert not render.use_colour(tty, {"TERM": "dumb"}) and not render.use_colour(object(), {})
    assert not render.Style(False).paint("x", "red") != "x"


# --------------------------------------------------------------------------- tune

def test_tune_saves_the_tuned_preset_and_records_trials(world, capsys) -> None:
    world.gw.is_running = True
    world.gw.write_presets()
    r = world.run(capsys, "tune", "qwen3", "--for", "chat")
    assert r.code == 0
    assert "tuned: qwen3-8b-q4_k_m for chat" in r.out and "saved as the 'chat' preset" in r.out
    assert "gateway reloaded" in r.out and "reload" in world.gw.calls
    assert "[1/4] trying" in r.err and "[1/4] ok" in r.err and "60.0 tok/s" in r.err
    assert preset_names(world, world.qwen) == {"chat": "tuned"}
    trials = json.loads((world.cfg_dir / "tuning.json").read_text())["trials"]
    assert len(trials) == 1 and trials[0]["measurement"]["ok"] and trials[0]["intent"] == "chat"
    assert len(world.runner.profiles) == 1


def test_tune_json_is_pure(world, capsys) -> None:
    r = world.run(capsys, "tune", "qwen3", "--json")
    d = json.loads(r.out)
    assert r.code == 0 and d["ok"] and d["intent"] == "chat" and not d["cancelled"]
    assert d["best"]["measurement"]["gen_tps"] == 60.0 and len(d["trials"]) == 1
    assert d["gateway_reloaded"] is False and "[1/4]" in r.err


def test_tune_without_a_running_gateway_does_not_reload(world, capsys) -> None:
    assert world.run(capsys, "tune", "qwen3").code == 0
    assert "reload" not in world.gw.calls


def test_tune_steps_down_and_respects_trials(world, capsys) -> None:
    world.runner = FakeRunner(oom(), oom(), good())
    r = world.run(capsys, "tune", "qwen3", "--trials", "2")
    assert len(world.runner.profiles) == 2 and r.code == 1
    assert "nothing loaded after 2 trials" in r.out and "out of memory" in r.err
    trials = json.loads((world.cfg_dir / "tuning.json").read_text())["trials"]
    assert len(trials) == 2                                  # failed trials are recorded too
    assert not (world.cfg_dir / "config.json").exists()      # no winner, no preset
    world.runner = FakeRunner(oom(), good())
    assert world.run(capsys, "tune", "qwen3").code == 0


def test_tune_preflight_refuses_and_force_overrides(world, capsys) -> None:
    world.host = Host(gpus=[Device.of("CUDA0", "GPU", int(16e9), int(0.1e9))],
                      ram_total=int(24e9), ram_free=int(24e9), cpu_threads=16, cpu_cores=8)
    r = world.run(capsys, "tune", "qwen3")
    assert r.code == 1 and "cannot tune qwen3-8b-q4_k_m" in r.err and "--force" in r.err
    assert world.runner.profiles == []
    assert world.run(capsys, "tune", "qwen3", "--force").code == 0
    assert len(world.runner.profiles) == 1


def test_tune_ctrl_c_stops_cleanly_and_records_the_trials_so_far(world, capsys) -> None:
    world.runner = FakeRunner(oom(), KeyboardInterrupt())
    r = world.run(capsys, "tune", "qwen3")
    assert r.code == 1 and "cancelled" in r.out
    trials = json.loads((world.cfg_dir / "tuning.json").read_text())["trials"]
    assert len(trials) == 1 and not trials[0]["measurement"]["ok"]
    assert len(world.runner.profiles) == 2


def test_cancelled_tune_text_does_not_repeat_itself(world, capsys) -> None:
    world.runner = FakeRunner(KeyboardInterrupt())
    r = world.run(capsys, "tune", "qwen3")
    assert r.code == 1
    assert r.out.splitlines()[0] == "qwen3-8b-q4_k_m for chat: cancelled before any trial ran"


def test_tune_needs_the_binary(world, capsys) -> None:
    world.cfg.bin_dir = str(world.bin_dir / "missing")
    r = world.run(capsys, "tune", "qwen3")
    assert r.code == 1 and r.err.count("\n") == 1
    assert "llama-server not found in" in r.err and "bin_dir" in r.err
    assert world.runner.profiles == []


# --------------------------------------------------------------------------- serve (gateway)

def test_serve_creates_a_missing_estimated_preset_and_loads_the_alias(world, capsys) -> None:
    r = world.run(capsys, "serve", "qwen3")
    assert r.code == 0
    assert preset_names(world, world.qwen) == {"chat": "estimated"}
    assert world.gw.calls == ["start", "reload", ("load", "qwen3-8b-q4_k_m")]
    assert "preset 'chat' (estimated, planned just now)" in r.out
    assert "serving qwen3-8b-q4_k_m" in r.out and "http://127.0.0.1:8080/v1" in r.out
    assert re.search(r"model\s+qwen3-8b-q4_k_m", r.out)
    assert "curl http://127.0.0.1:8080/v1/chat/completions" in r.out
    assert '"model": "qwen3-8b-q4_k_m"' in r.out and "Bearer" not in r.out
    assert "loading qwen3-8b-q4_k_m ." in r.err                   # progress dots on stderr
    assert world.gw.states["qwen3-8b-q4_k_m"] == LOADED


def test_serve_uses_an_existing_preset_as_is(world, capsys) -> None:
    custom = Profile(ctx_size=2048, threads=3)
    world.cfg.set_preset(str(world.qwen), Preset("agent", custom, "tuned", good()))
    r = world.run(capsys, "serve", "qwen3")
    assert r.code == 0 and "preset 'agent' (tuned, existing)" in r.out
    ent = world.cfg.models[str(world.qwen)]
    assert set(ent.presets) == {"agent"} and ent.presets["agent"].profile == custom
    assert not (world.cfg_dir / "config.json").exists()          # nothing changed, nothing saved
    assert ("load", "qwen3-8b-q4_k_m") in world.gw.calls          # the active preset is the plain alias


def test_serve_for_makes_the_preset_active_and_keeps_others(world, capsys) -> None:
    world.cfg.set_preset(str(world.qwen), Preset("chat", Profile(ctx_size=1111), "tuned"))
    r = world.run(capsys, "serve", "qwen3", "--for", "agent")
    assert r.code == 0 and "planned just now" in r.out
    ent = saved_config(world)["models"][str(world.qwen)]
    assert ent["active"] == "agent" and set(ent["presets"]) == {"chat", "agent"}
    assert ent["presets"]["chat"]["profile"]["ctx_size"] == 1111
    assert ("load", "qwen3-8b-q4_k_m") in world.gw.calls


def test_serve_for_an_existing_non_active_preset_activates_it_without_replanning(world, capsys) -> None:
    world.cfg.set_preset(str(world.qwen), Preset("chat", Profile(), "estimated"))
    world.cfg.set_preset(str(world.qwen), Preset("agent", Profile(ctx_size=4242), "tuned"),
                         activate=False)
    r = world.run(capsys, "serve", "qwen3", "--for", "agent")
    assert r.code == 0 and "(tuned, existing)" in r.out
    ent = saved_config(world)["models"][str(world.qwen)]
    assert ent["active"] == "agent" and ent["presets"]["agent"]["profile"]["ctx_size"] == 4242


def test_serve_no_activate_uses_alias_at_intent(world, capsys) -> None:
    world.cfg.set_preset(str(world.qwen), Preset("chat", Profile(), "estimated"))
    r = world.run(capsys, "serve", "qwen3", "--for", "agent", "--no-activate")
    assert r.code == 0
    assert saved_config(world)["models"][str(world.qwen)]["active"] == "chat"
    assert ("load", "qwen3-8b-q4_k_m@agent") in world.gw.calls
    assert re.search(r"model\s+qwen3-8b-q4_k_m@agent", r.out)


def test_serve_with_a_running_gateway_does_not_start_it(world, capsys) -> None:
    world.gw.is_running = True
    r = world.run(capsys, "serve", "qwen3")
    assert r.code == 0 and "start" not in world.gw.calls and "reload" in world.gw.calls


def test_serve_skips_load_when_the_model_is_already_loaded(world, capsys) -> None:
    world.gw.is_running = True
    world.cfg.set_preset(str(world.qwen), Preset("chat", Profile(), "estimated"))
    world.gw.write_presets()
    world.gw.states["qwen3-8b-q4_k_m"] = LOADED
    assert world.run(capsys, "serve", "qwen3").code == 0
    assert not any(isinstance(c, tuple) for c in world.gw.calls)


def test_serve_reports_a_failed_model_with_the_reason_and_log(world, capsys) -> None:
    world.gw.outcome, world.gw.exit_code = FAILED, 3
    world.gw.log.parent.mkdir(parents=True, exist_ok=True)
    world.gw.log.write_text("12.3 I loading\n12.4 E error loading model: tensor 'x' missing\n")
    r = world.run(capsys, "serve", "qwen3")
    assert r.code == 1
    assert "failed to load (exit code 3)" in r.err and "tensor 'x' missing" in r.err
    assert str(world.gw.log) in r.err


def test_serve_times_out_without_real_sleeping(world, capsys) -> None:
    world.gw.polls = 10 ** 9                              # never finishes loading
    r = world.run(capsys, "serve", "qwen3")
    assert r.code == 1 and "still loading" in r.err
    assert 60 <= world.clock.t < 120                      # 60 s base + 20 s/GB of a tiny file


def test_serve_no_wait_does_not_poll(world, capsys) -> None:
    before = world.clock.sleeps
    r = world.run(capsys, "serve", "qwen3", "--no-wait")
    assert r.code == 0 and "loading qwen3-8b-q4_k_m" in r.out
    assert world.gw.states["qwen3-8b-q4_k_m"] == LOADING and world.clock.sleeps == before


def test_serve_reports_a_gateway_that_does_not_start(world, capsys) -> None:
    world.gw.start_error = "port 8080 on 127.0.0.1 is already in use; stop what holds it"
    r = world.run(capsys, "serve", "qwen3")
    assert r.code == 1 and "already in use" in r.err


def test_serve_embedding_model_prints_an_embeddings_call(world, capsys) -> None:
    _model_file(world.model_dir / "embed-f16.gguf", MINI_KVS + [kv_u32("testarch.pooling_type", 1)])
    r = world.run(capsys, "serve", "embed")
    assert r.code == 0 and "/v1/embeddings" in r.out and '"input"' in r.out


def test_serve_with_an_api_key_mentions_the_header_but_not_the_key(world, capsys) -> None:
    world.cfg.gateway.api_key = "sekrit"
    r = world.run(capsys, "serve", "qwen3")
    assert "Authorization: Bearer YOUR_API_KEY" in r.out and "sekrit" not in r.out


def test_serve_needs_the_binary_and_changes_nothing_then(world, capsys) -> None:
    world.cfg.bin_dir = str(world.bin_dir / "missing")
    r = world.run(capsys, "serve", "qwen3")
    assert r.code == 1 and "llama-server not found in" in r.err
    assert not (world.cfg_dir / "config.json").exists() and world.gw.calls == []


def test_serve_port_needs_standalone(world, capsys) -> None:
    assert world.run(capsys, "serve", "qwen3", "--port", "9000").code == 2


def test_serve_migrates_a_v2_config_with_a_backup(world, capsys) -> None:
    world.cfg_dir.mkdir(parents=True, exist_ok=True)
    (world.cfg_dir / "config.json").write_text(json.dumps({"version": 2, "profiles": {}}))
    assert world.run(capsys, "serve", "qwen3").code == 0
    assert (world.cfg_dir / "config.v2.bak.json").exists()
    assert saved_config(world)["version"] == 3


# --------------------------------------------------------------------------- serve --standalone

def test_serve_standalone(world, capsys) -> None:
    r = world.run(capsys, "serve", "qwen3", "--standalone", "--port", "9100", "--for", "agent")
    assert r.code == 0
    assert world.store.launched == [(str(world.qwen), "agent", 9100)]
    assert world.gw.calls == []                           # the gateway is not involved
    assert "http://127.0.0.1:9100/v1" in r.out and "curl http://127.0.0.1:9100/v1/chat" in r.out
    assert '"model"' not in r.out.split("try")[1]
    assert preset_names(world, world.qwen) == {"agent": "estimated"}


def test_serve_standalone_waits_for_health_and_reports_a_dead_server(world, capsys) -> None:
    world.health = "loading"
    r = world.run(capsys, "serve", "qwen3", "--standalone")
    assert r.code == 1 and "did not become ready" in r.err
    world.store.dies = True
    world.alive.clear()
    r = world.run(capsys, "serve", "qwen3", "--standalone")
    assert r.code == 1 and "the server exited" in r.err


def test_serve_standalone_no_wait_and_launch_failure(world, capsys, monkeypatch) -> None:
    world.health = "loading"
    r = world.run(capsys, "serve", "qwen3", "--standalone", "--no-wait")
    assert r.code == 0 and "started Qwen3-8B-Q4_K_M" in r.out

    def busy(*a: Any, **k: Any) -> Session:
        raise RuntimeError("port 9100 on 127.0.0.1 is already in use")

    monkeypatch.setattr(world.store, "launch_server", busy)
    r = world.run(capsys, "serve", "qwen3", "--standalone", "--port", "9100")
    assert r.code == 1 and "already in use" in r.err


# --------------------------------------------------------------------------- up / down

def test_up_starts_the_gateway_and_is_idempotent(world, capsys) -> None:
    world.cfg.set_preset(str(world.qwen), Preset("chat", Profile(), "estimated"))
    world.cfg.set_preset(str(world.llama), Preset("chat", Profile(), "estimated"))
    r = world.run(capsys, "up")
    assert r.code == 0 and world.gw.calls == ["start"]
    assert "gateway running at http://127.0.0.1:8080" in r.out and "2 presets written" in r.out
    r = world.run(capsys, "up")
    assert r.code == 0 and "already running" in r.out and world.gw.calls == ["start"]


def test_up_lists_problems_failures_and_needs_the_binary(world, capsys) -> None:
    world.cfg.set_preset(str(world.qwen), Preset("chat", Profile(extra_args="oops"), "estimated"))
    r = world.run(capsys, "up")
    assert r.code == 0 and "warning: skipped [qwen3-8b-q4_k_m]" in r.err and "0 presets" in r.out
    world.gw.is_running = False
    world.gw.start_error = "the gateway exited at once: bad option (log: x)"
    r = world.run(capsys, "up")
    assert r.code == 1 and "bad option" in r.err
    world.cfg.bin_dir = str(world.bin_dir / "missing")
    assert "llama-server not found" in world.run(capsys, "up").err


def test_up_reports_a_gateway_that_never_answers(world, capsys) -> None:
    world.gw.healthy = False
    r = world.run(capsys, "up")
    assert r.code == 1 and "did not answer within 30 s" in r.err


def test_down_is_idempotent(world, capsys) -> None:
    world.gw.is_running = True
    r = world.run(capsys, "down")
    assert r.code == 0 and "gateway stopped" in r.out and not world.gw.is_running
    r = world.run(capsys, "down")
    assert r.code == 0 and "was not running" in r.out


# --------------------------------------------------------------------------- ps

def test_ps_gateway_down_and_no_servers(world, capsys) -> None:
    r = world.run(capsys, "ps")
    assert r.code == 0 and "not running" in r.out and "no standalone or attached servers" in r.out
    d = json.loads(world.run(capsys, "ps", "--json").out)
    assert d["gateway"]["running"] is False and d["gateway"]["models"] is None
    assert d["sessions"] == []


def test_ps_lists_gateway_models_and_sessions(world, capsys) -> None:
    world.cfg.set_preset(str(world.qwen), Preset("chat", Profile(), "estimated"))
    world.cfg.set_preset(str(world.qwen), Preset("agent", Profile(), "estimated"), activate=False)
    world.gw.is_running = True
    world.gw.write_presets()
    world.gw.states["qwen3-8b-q4_k_m"] = LOADED
    world.store.sessions = [
        Session("m-9000", "m", "http://127.0.0.1:9000", pid=1, log="x", started=1000.0 - 7500),
        Session("ext-h:1", "ext", "http://h:1", external=True, started=0.0)]
    r = world.run(capsys, "ps")
    assert r.code == 0
    assert "running at http://127.0.0.1:8080 (pid 4242, up 60s), healthy" in r.out
    assert re.search(r"qwen3-8b-q4_k_m\s+loaded", r.out) and re.search(r"qwen3-8b-q4_k_m@agent\s+unloaded", r.out)
    assert re.search(r"m\s+http://127.0.0.1:9000\s+ready\s+2h06m\s+standalone", r.out)
    assert "attached" in r.out
    d = json.loads(world.run(capsys, "ps", "--json").out)
    assert d["gateway"]["models"] == [
        {"name": "qwen3-8b-q4_k_m", "state": "loaded", "exit_code": None},
        {"name": "qwen3-8b-q4_k_m@agent", "state": "unloaded", "exit_code": None}]
    assert [s["state"] for s in d["sessions"]] == ["ready", "ready"]
    assert d["sessions"][0]["uptime_s"] == 7560 and d["sessions"][1]["uptime_s"] is None


def test_ps_survives_a_gateway_that_cannot_list_models(world, capsys, monkeypatch) -> None:
    world.gw.is_running = True

    def boom() -> list:
        raise GatewayError("the gateway at x is not reachable")

    monkeypatch.setattr(world.gw, "models", boom)
    r = world.run(capsys, "ps")
    assert r.code == 0 and "models unavailable: the gateway at x is not reachable" in r.out


# --------------------------------------------------------------------------- load / unload

def _gateway_with_two_presets(w: SimpleNamespace) -> None:
    w.cfg.set_preset(str(w.qwen), Preset("chat", Profile(), "estimated"))
    w.cfg.set_preset(str(w.qwen), Preset("agent", Profile(), "estimated"), activate=False)
    w.cfg.set_preset(str(w.llama), Preset("chat", Profile(), "estimated"))
    w.gw.is_running = True
    w.gw.write_presets()


def test_load_and_unload_need_a_running_gateway(world, capsys) -> None:
    for cmd in ("load", "unload"):
        r = world.run(capsys, cmd, "x")
        assert r.code == 1 and "gateway is not running" in r.err and "llamactl up" in r.err


def test_load_waits_and_unload_works(world, capsys) -> None:
    _gateway_with_two_presets(world)
    r = world.run(capsys, "load", "llama-3-q8_0")
    assert r.code == 0 and "llama-3-q8_0 is loaded" in r.out and ("load", "llama-3-q8_0") in world.gw.calls
    r = world.run(capsys, "load", "llama-3-q8_0")
    assert r.code == 0 and "already loaded" in r.out
    r = world.run(capsys, "unload", "llama-3")                 # unique substring
    assert r.code == 0 and ("unload", "llama-3-q8_0") in world.gw.calls


def test_unload_waits_for_the_model_to_leave(world, capsys) -> None:
    _gateway_with_two_presets(world)
    world.gw.states["llama-3-q8_0"] = LOADED
    world.gw.unload_polls = 3
    r = world.run(capsys, "unload", "llama-3-q8_0")
    assert r.code == 0 and "llama-3-q8_0 unloaded" in r.out and world.clock.sleeps == 2
    world.gw.states["llama-3-q8_0"] = LOADED
    world.gw.unload_polls = 10 ** 9
    r = world.run(capsys, "unload", "llama-3-q8_0")
    assert r.code == 0 and "still lists it as running" in r.out


def test_load_by_preset_name_no_wait_and_failure(world, capsys) -> None:
    _gateway_with_two_presets(world)
    r = world.run(capsys, "load", "qwen3-8b-q4_k_m@agent", "--no-wait")
    assert r.code == 0 and "is loading" in r.out
    world.gw.outcome = FAILED
    r = world.run(capsys, "load", "llama-3-q8_0")
    assert r.code == 1 and "failed to load" in r.err


def test_load_unknown_and_ambiguous_names_exit_2(world, capsys) -> None:
    _gateway_with_two_presets(world)
    r = world.run(capsys, "load", "zzz")
    assert r.code == 2 and "unknown gateway model 'zzz'" in r.err and "llama-3-q8_0" in r.err
    r = world.run(capsys, "unload", "q8")                      # llama-3-q8_0 only: fine
    assert r.code == 0
    r = world.run(capsys, "unload", "q")
    assert r.code == 2 and "ambiguous" in r.err


# --------------------------------------------------------------------------- stop

def _two_sessions(w: SimpleNamespace) -> None:
    w.store.sessions = [
        Session("a-9000", "a", "http://127.0.0.1:9000", pid=1, log="x"),
        Session("a-9001", "a", "http://127.0.0.1:9001", pid=2, log="x"),
        Session("b-9002", "b", "http://127.0.0.1:9002", pid=3, log="x"),
        Session("ext-h:1", "ext", "http://h:1", external=True)]


def test_stop_by_id_name_and_all(world, capsys) -> None:
    _two_sessions(world)
    assert "stopped b-9002" in world.run(capsys, "stop", "b").out
    r = world.run(capsys, "stop", "a")                          # two servers named a
    assert r.code == 2 and "a-9000, a-9001" in r.err
    assert world.run(capsys, "stop", "a-9001").code == 0
    r = world.run(capsys, "stop", "--all")
    assert r.code == 0 and "stopped a-9000" in r.out and "detached ext-h:1" in r.out
    assert world.store.sessions == []
    assert "no standalone" in world.run(capsys, "stop", "--all").out


def test_stop_never_touches_the_gateway_and_says_so(world, capsys) -> None:
    _two_sessions(world)
    world.gw.is_running = True
    r = world.run(capsys, "stop", "--all")
    assert "the gateway was not touched" in r.out and "stop" not in world.gw.calls
    assert world.gw.is_running
    r = world.run(capsys, "stop", "gateway")
    assert r.code == 2 and "llamactl down" in r.err


def test_stop_usage(world, capsys) -> None:
    _two_sessions(world)
    for argv in (["stop"], ["stop", "x", "--all"], ["stop", "nothing-like-it"]):
        assert world.run(capsys, *argv).code == 2
    assert len(world.store.sessions) == 4


# --------------------------------------------------------------------------- logs

def test_logs_tail_for_gateway_and_sessions(world, capsys) -> None:
    world.gw.log.parent.mkdir(parents=True, exist_ok=True)
    world.gw.log.write_text("".join(f"line {i}\n" for i in range(100)))
    r = world.run(capsys, "logs", "gateway")
    assert r.code == 0 and r.out.splitlines() == [f"line {i}" for i in range(60, 100)]
    r = world.run(capsys, "logs", "gateway", "-n", "3")
    assert r.out.splitlines() == ["line 97", "line 98", "line 99"]
    assert world.run(capsys, "logs", "gateway", "-n", "0").out == ""
    log = world.cfg_dir / "logs" / "m-9000.log"
    log.write_text("hello\nworld\n")
    world.store.sessions = [Session("m-9000", "m", "http://x:1", pid=1, log=str(log))]
    assert world.run(capsys, "logs", "m").out == "hello\nworld\n"
    assert world.run(capsys, "logs", "m-9000", "-n", "1").out == "world\n"


def test_logs_errors(world, capsys) -> None:
    r = world.run(capsys, "logs", "gateway")
    assert r.code == 1 and "no log at" in r.err
    r = world.run(capsys, "logs", "nothing")
    assert r.code == 2
    world.store.sessions = [Session("ext-h:1", "ext", "http://h:1", external=True)]
    r = world.run(capsys, "logs", "ext")
    assert r.code == 1 and "attached server" in r.err


def test_logs_follow_prints_new_lines_until_ctrl_c(world, capsys) -> None:
    world.gw.log.parent.mkdir(parents=True, exist_ok=True)
    world.gw.log.write_text("old\n")
    steps = iter(["a\nb\n", "TRUNC", ""])

    def sleep(seconds: float) -> None:
        step = next(steps, None)
        if step is None:
            raise KeyboardInterrupt
        if step == "TRUNC":
            world.gw.log.write_text("new\n")
        elif step:
            with world.gw.log.open("a") as fh:
                fh.write(step)

    r = world.run(capsys, "logs", "gateway", "-f", sleep=sleep)
    assert r.code == 0 and r.out.splitlines() == ["old", "a", "b", "new"]


def test_tail_lines_reads_from_the_end(tmp_path: Path) -> None:
    p = tmp_path / "big.log"
    p.write_text("".join(f"{i}\n" for i in range(5000)))
    assert cli.tail_lines(p, 2, block=64) == ["4998", "4999"]
    assert len(cli.tail_lines(p, 6000)) == 5000
    p.write_text("no newline")
    assert cli.tail_lines(p, 3) == ["no newline"]


# --------------------------------------------------------------------------- read-only

def _snapshot(root: Path) -> dict[str, tuple[str, int]]:
    return {str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()} | {
        str(p.relative_to(root)) + "/": ("dir", 0) for p in root.rglob("*") if p.is_dir()}


def test_read_only_commands_write_nothing(world, capsys, isolated, monkeypatch) -> None:
    """hw, ls, plan, ps and logs through the real Services leave the config dir byte-identical."""
    cfg_dir = world.cfg_dir
    (cfg_dir / "logs").mkdir(parents=True)
    (cfg_dir / "logs" / "gateway.log").write_text("log line\n")
    (cfg_dir / "config.json").write_text(json.dumps({
        "version": 2, "bin_dir": str(world.bin_dir), "model_dirs": [str(world.model_dir)],
        "profiles": {str(world.qwen): {"ctx_size": 4096}}}))
    dead = {"id": "x-1", "name": "x", "url": "http://127.0.0.1:9", "pid": 2 ** 22 + 99,
            "log": "", "external": False, "started": 1.0}
    (cfg_dir / "sessions.json").write_text(json.dumps([dead]))
    (cfg_dir / "gateway.json").write_text(json.dumps(
        {"pid": 2 ** 22 + 98, "url": "http://127.0.0.1:8080", "log": "", "started": 1.0}))
    (cfg_dir / "tuning.json").write_text('{"version": 1, "trials": []}')
    (cfg_dir / "gguf_cache.json").write_text("{}")
    before = _snapshot(cfg_dir)
    for argv in (["hw"], ["hw", "--json"], ["ls"], ["ls", "--json"], ["plan", "qwen3"],
                 ["plan", "qwen3", "--all"], ["plan", "llama", "--for", "agent", "--json"],
                 ["ps"], ["ps", "--json"], ["logs", "gateway"], ["config"], ["config", "--json"]):
        assert cli.main(argv) == 0, argv
        capsys.readouterr()
    assert _snapshot(cfg_dir) == before
    assert cli.main(["plan", "qwen3", "--save"]) == 0            # ...whereas --save does write
    assert _snapshot(cfg_dir) != before
    assert (cfg_dir / "config.v2.bak.json").exists()




# --------------------------------------------------------------------------- network access

LAN_IP = "192.168.50.7"        # the address the autouse `fake_network` fixture reports (eth1)


def test_up_lan_binds_every_interface_for_this_start_only(world, capsys) -> None:
    r = world.run(capsys, "up", "--lan")
    assert r.code == 0 and world.gw.bound_host == "0.0.0.0"
    assert f"gateway running at http://127.0.0.1:8080, http://{LAN_IP}:8080 (pid 4242)" in r.out
    assert "0.0.0.0:" not in r.out
    assert "warning: listening on 0.0.0.0 without an API key" in r.err
    assert world.cfg.gateway.host == "127.0.0.1" and not (world.cfg_dir / "config.json").exists()


def test_up_lan_with_a_key_does_not_warn(world, capsys) -> None:
    world.cfg.gateway.api_key = "k"
    r = world.run(capsys, "up", "--lan")
    assert r.code == 0 and "warning" not in r.err


def test_up_host_binds_one_address(world, capsys) -> None:
    r = world.run(capsys, "up", "--host", LAN_IP)
    assert r.code == 0 and world.gw.bound_host == LAN_IP
    assert f"gateway running at http://{LAN_IP}:8080 (pid" in r.out and "127.0.0.1" not in r.out
    assert "without an API key" in r.err


def test_up_host_loopback_name_is_private_and_quiet(world, capsys) -> None:
    r = world.run(capsys, "up", "--host", "local")
    assert r.code == 0 and world.gw.bound_host == "127.0.0.1" and r.err == ""
    assert "gateway running at http://127.0.0.1:8080 (pid" in r.out


def test_lan_and_host_are_mutually_exclusive(world, capsys) -> None:
    for cmd in (["up"], ["serve", "qwen3"]):
        r = world.run(capsys, *cmd, "--lan", "--host", "10.0.0.5")
        assert r.code == 2 and "not allowed with" in r.err
    assert world.gw.calls == []


def test_a_bad_host_is_a_usage_error(world, capsys) -> None:
    r = world.run(capsys, "up", "--host", "two words")
    assert r.code == 2 and "--host: expected a host name or address" in r.err and world.gw.calls == []


def test_up_fails_when_the_running_gateway_has_another_bind(world, capsys) -> None:
    assert world.run(capsys, "up").code == 0                   # loopback
    r = world.run(capsys, "up", "--lan")
    assert r.code == 1 and "listens on 127.0.0.1, not 0.0.0.0" in r.err
    assert "llamactl down" in r.err and world.gw.calls == ["start"]
    assert world.run(capsys, "down").code == 0
    assert world.run(capsys, "up", "--lan").code == 0 and world.gw.bound_host == "0.0.0.0"
    # the same bind again, and a plain `up`, are both fine
    r = world.run(capsys, "up", "--lan")
    assert r.code == 0 and "already running at http://127.0.0.1:8080, http://192.168.50.7:8080" in r.out
    assert world.run(capsys, "up").code == 0


def test_serve_lan_starts_the_gateway_on_every_interface(world, capsys) -> None:
    r = world.run(capsys, "serve", "qwen3", "--lan")
    assert r.code == 0 and world.gw.bound_host == "0.0.0.0"
    assert re.search(r"base URL\s+http://127.0.0.1:8080/v1", r.out)
    assert re.search(rf"on network\s+http://{LAN_IP}:8080/v1", r.out)
    assert f"curl http://{LAN_IP}:8080/v1/chat/completions" in r.out        # the curl uses the LAN URL
    assert "warning: listening on 0.0.0.0 without an API key" in r.err
    assert world.cfg.gateway.host == "127.0.0.1"


def test_serve_lan_fails_against_a_loopback_gateway(world, capsys) -> None:
    assert world.run(capsys, "up").code == 0
    r = world.run(capsys, "serve", "qwen3", "--lan")
    assert r.code == 1 and "llamactl down" in r.err
    assert "load" not in [c[0] for c in world.gw.calls if isinstance(c, tuple)]


def test_serve_without_flags_shows_one_url(world, capsys) -> None:
    r = world.run(capsys, "serve", "qwen3")
    assert "on network" not in r.out and r.err.count("warning") == 0


def test_serve_standalone_lan(world, capsys) -> None:
    r = world.run(capsys, "serve", "qwen3", "--standalone", "--lan", "--port", "9100")
    assert r.code == 0 and world.store.hosts == ["0.0.0.0"]
    assert "0.0.0.0:" not in r.out
    assert re.search(r"base URL\s+http://127.0.0.1:9100/v1", r.out)
    assert re.search(rf"on network\s+http://{LAN_IP}:9100/v1", r.out)
    assert f"curl http://{LAN_IP}:9100/v1/chat/completions" in r.out
    assert "warning: listening on 0.0.0.0 without an API key" in r.err


def test_serve_standalone_without_flags_keeps_the_preset_host(world, capsys) -> None:
    r = world.run(capsys, "serve", "qwen3", "--standalone")
    assert world.store.hosts == [None] and "on network" not in r.out and "warning" not in r.err


def test_serve_standalone_warns_for_an_exposed_preset_without_a_key(world, capsys) -> None:
    world.cfg.set_preset(str(world.qwen), Preset("chat", Profile(host="0.0.0.0"), "estimated"))
    r = world.run(capsys, "serve", "qwen3", "--standalone")
    assert "warning: listening on 0.0.0.0 without an API key" in r.err
    world.cfg.set_preset(str(world.qwen), Preset("chat", Profile(host="0.0.0.0", api_key="k"),
                                                 "estimated"))
    assert "warning" not in world.run(capsys, "serve", "qwen3", "--standalone").err


def test_ps_lists_client_urls(world, capsys) -> None:
    world.run(capsys, "up", "--lan")
    world.store.sessions = [Session("m-9000", "m", "http://127.0.0.1:9000", pid=1, log="x",
                                    started=1000.0, host="0.0.0.0"),
                            Session("ext-h:1", "ext", "http://h:1", external=True)]
    r = world.run(capsys, "ps")
    assert f"running at http://127.0.0.1:8080, http://{LAN_IP}:8080 (pid" in r.out
    assert f"http://127.0.0.1:9000, http://{LAN_IP}:9000" in r.out and "0.0.0.0" not in r.out
    d = json.loads(world.run(capsys, "ps", "--json").out)
    assert d["gateway"]["urls"] == ["http://127.0.0.1:8080", f"http://{LAN_IP}:8080"]
    assert d["gateway"]["url"] == "http://127.0.0.1:8080"
    assert d["sessions"][0]["urls"] == ["http://127.0.0.1:9000", f"http://{LAN_IP}:9000"]
    assert d["sessions"][1]["urls"] == ["http://h:1"]


def test_ps_for_a_loopback_gateway_has_one_url(world, capsys) -> None:
    world.run(capsys, "up")
    assert json.loads(world.run(capsys, "ps", "--json").out)["gateway"]["urls"] == [
        "http://127.0.0.1:8080"]


# --------------------------------------------------------------------------- config

def test_config_shows_the_settings_with_the_key_masked(world, capsys) -> None:
    world.cfg.gateway.api_key = "super-secret-key"
    r = world.run(capsys, "config")
    assert r.code == 0 and "super-secret-key" not in r.out and r.err == ""
    assert re.search(r"gateway.host\s+127.0.0.1  \(local\)", r.out)
    assert re.search(r"gateway.api_key\s+\*{8} \(set\)", r.out)
    for want in ("gateway.port", "8080", "gateway.models_max", "gateway.sleep_idle 600",
                 "gateway.autostart false", "default_intent", "chat", "headroom_pct", "10",
                 "Clients connect to", "http://127.0.0.1:8080/v1"):
        assert want in re.sub(r" +", " ", r.out), want
    assert "note:" not in r.out and "!" not in r.out


def test_config_json_masks_the_key_too(world, capsys) -> None:
    world.cfg.gateway.api_key = "super-secret-key"
    d = json.loads(world.run(capsys, "config", "--json").out)
    assert "super-secret-key" not in json.dumps(d)
    assert d["gateway"] == {"host": "127.0.0.1", "mode": "local", "port": 8080,
                            "api_key": "********", "models_max": 1, "sleep_idle": 600,
                            "autostart": False}
    assert d["urls"] == ["http://127.0.0.1:8080"] and d["exposed"] is False
    assert d["warnings"] == [] and d["notes"] == [] and d["default_intent"] == "chat"


def test_config_without_a_key_says_none(world, capsys) -> None:
    assert re.search(r"gateway.api_key\s+none", world.run(capsys, "config").out)
    assert json.loads(world.run(capsys, "config", "--json").out)["gateway"]["api_key"] == ""


def test_config_set_validates_and_saves(world, capsys) -> None:
    r = world.run(capsys, "config", "gateway.port=9090", "gateway.sleep_idle=0",
                  "gateway.autostart=yes", "default_intent=agent", "headroom_pct=15",
                  "gateway.models_max=2", "gateway.api_key=abc123")
    assert r.code == 0 and "saved gateway.port, gateway.sleep_idle" in r.err
    g = saved_config(world)["gateway"]
    assert (g["port"], g["sleep_idle"], g["autostart"], g["models_max"], g["api_key"]) == (
        9090, 0, True, 2, "abc123")
    assert saved_config(world)["default_intent"] == "agent" and saved_config(world)["headroom_pct"] == 15
    assert "abc123" not in r.out and "http://127.0.0.1:9090/v1" in r.out
    r = world.run(capsys, "config", "gateway.port=9090")
    assert r.code == 0 and "nothing changed" in r.err


def test_config_gateway_host_accepts_lan_local_and_an_address(world, capsys) -> None:
    r = world.run(capsys, "config", "gateway.host=lan")
    assert r.code == 0 and saved_config(world)["gateway"]["host"] == "0.0.0.0"
    assert re.search(r"gateway.host\s+0.0.0.0  \(lan\)", r.out)
    assert f"http://{LAN_IP}:8080/v1" in r.out and "0.0.0.0:" not in r.out
    assert "! listening on 0.0.0.0 without an API key" in r.out
    assert world.run(capsys, "config", "gateway.host=local").code == 0
    assert saved_config(world)["gateway"]["host"] == "127.0.0.1"
    r = world.run(capsys, "config", f"gateway.host={LAN_IP}")
    assert saved_config(world)["gateway"]["host"] == LAN_IP and "(custom)" in r.out
    d = json.loads(world.run(capsys, "config", "gateway.host=lan", "--json").out)
    assert d["exposed"] is True and d["warnings"] and d["urls"][1] == f"http://{LAN_IP}:8080"


def test_config_shows_the_wsl_notes_for_an_exposed_bind(world, capsys, monkeypatch) -> None:
    from llamactl import network

    monkeypatch.setattr(network, "read_osrelease", lambda: "5.15-microsoft-standard-WSL2")
    monkeypatch.setattr(network, "read_wslinfo", lambda: "mirrored")
    network.reset_cache()
    world.cfg.gateway.host = "0.0.0.0"
    assert "Hyper-V firewall" in world.run(capsys, "config").out
    r = world.run(capsys, "up")
    assert "note: WSL uses mirrored networking" in r.err


@pytest.mark.parametrize("pair, message", [
    ("gateway.nope=1", "unknown setting 'gateway.nope'"),
    ("port=1", "unknown setting 'port'"),
    ("gateway.port=0", "gateway.port: expected a whole number from 1 to 65535"),
    ("gateway.port=abc", "gateway.port: expected a whole number"),
    ("gateway.port=70000", "from 1 to 65535"),
    ("gateway.models_max=0", "gateway.models_max: expected a whole number, 1 or more"),
    ("gateway.sleep_idle=-1", "gateway.sleep_idle"),
    ("gateway.autostart=maybe", "expected true or false"),
    ("gateway.host=", "gateway.host: expected a host name or address"),
    ("gateway.host=a b", "gateway.host"),
    ("default_intent=bogus", "default_intent: expected one of chat"),
    ("headroom_pct=60", "headroom_pct: expected a whole number from 0 to 50"),
    ("gateway.port", "'gateway.port' is not KEY=VALUE"),
])
def test_config_rejects_bad_keys_and_values_and_saves_nothing(world, capsys, pair, message) -> None:
    r = world.run(capsys, "config", "gateway.models_max=3", pair)
    assert r.code == 2 and message in r.err and r.out == ""
    assert not (world.cfg_dir / "config.json").exists() and world.cfg.gateway.models_max == 1


def test_config_new_api_key_is_printed_once_and_saved(world, capsys) -> None:
    r = world.run(capsys, "config", "--new-api-key")
    assert r.code == 0
    key = saved_config(world)["gateway"]["api_key"]
    assert len(key) >= 43 and r.out.count(key) == 1
    assert f"new API key (saved; shown only now): {key}" in r.out
    assert re.search(r"gateway.api_key\s+\*{8} \(set\)", r.out)
    again = world.run(capsys, "config")
    assert key not in again.out
    other = world.run(capsys, "config", "--new-api-key")
    assert saved_config(world)["gateway"]["api_key"] != key and key not in other.out


def test_config_new_api_key_json_and_with_a_host(world, capsys) -> None:
    r = world.run(capsys, "config", "gateway.host=lan", "--new-api-key", "--json")
    d = json.loads(r.out)
    key = saved_config(world)["gateway"]["api_key"]
    assert d["new_api_key"] == key and d["gateway"]["api_key"] == "********"
    assert d["warnings"] == []                                  # exposed, but it has a key now
    assert r.out.count(key) == 1


def test_config_new_api_key_conflicts_with_an_explicit_one(world, capsys) -> None:
    r = world.run(capsys, "config", "gateway.api_key=x", "--new-api-key")
    assert r.code == 2 and "not both" in r.err and not (world.cfg_dir / "config.json").exists()


def test_config_clears_the_key_with_an_empty_value(world, capsys) -> None:
    world.cfg.gateway.api_key = "old"
    assert world.run(capsys, "config", "gateway.api_key=").code == 0
    assert saved_config(world)["gateway"]["api_key"] == ""


def test_config_says_a_running_gateway_needs_down_and_up(world, capsys) -> None:
    world.run(capsys, "up")
    r = world.run(capsys, "config", "gateway.host=lan", "gateway.autostart=true")
    assert "keeps its old gateway.host until you run `llamactl down` and `llamactl up`" in r.err
    assert "autostart" not in r.err.split("keeps its old")[1]
    world.gw.is_running = False
    assert "keeps its old" not in world.run(capsys, "config", "gateway.port=9999").err


def test_config_show_is_read_only() -> None:
    parse = cli.build_parser().parse_args
    assert cli.is_read_only(parse(["config"])) and cli.is_read_only(parse(["config", "--json"]))
    assert not cli.is_read_only(parse(["config", "gateway.port=1"]))
    assert not cli.is_read_only(parse(["config", "--new-api-key"]))
