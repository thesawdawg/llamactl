"""Tuner search: candidate stepping, scoring, progress/cancel, saving, preflight (fake Runner)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

from llamactl import intents, planner
from llamactl.calibration import Calibration, Trial
from llamactl.config import Config, Preset, Profile
from llamactl.gguf import ModelInfo
from llamactl.hardware import Device, Host
from llamactl.measure import ERR_ARGS, ERR_OOM, Measurement
from llamactl.tuner import (HOW_CTX_DOWN, HOW_CTX_UP, HOW_KV_Q8, HOW_PLAN, HOW_SPILL, HOW_THREADS,
                            Candidate, SearchSpace, TrialEvent, TuneResult, choose_best,
                            next_candidate, preflight, save_result, tune)

GB = 1e9


def _info(size: float, n_layer: int, n_head_kv: int, ctx: int, path: str = "m.gguf",
          pooling: int = -1, n_embd: int = 4096, head_dim: int = 128) -> ModelInfo:
    return ModelInfo(path=Path(path), size=int(size), architecture="x", name="m", n_layer=n_layer,
                     n_embd=n_embd, n_head=32, n_head_kv=n_head_kv, head_dim_k=head_dim,
                     head_dim_v=head_dim, ctx_train=ctx, expert_count=0, file_type=0, sharded=1,
                     parameter_count=0, pooling_type=pooling)


def _host(vram_gb: float = 16, ram_gb: float = 24, threads: int = 16) -> Host:
    gpus = [Device.of("CUDA0", "G0", int(vram_gb * GB), int(vram_gb * GB))] if vram_gb else []
    return Host(gpus=gpus, ram_total=int(ram_gb * GB), ram_free=int(ram_gb * GB),
                cpu_threads=threads, cpu_cores=8)


def ok(vram: float = 9.0, gen: float = 50.0, prompt: float = 1000.0, **kw) -> Measurement:
    return Measurement(ok=True, vram_used=int(vram * GB), gen_tps=gen, prompt_tps=prompt,
                       load_s=5.0, at="2026-11-09T10:00:00+00:00", build="b1", **kw)


def oom() -> Measurement:
    return Measurement(error=f"{ERR_OOM}: cudaMalloc failed", build="b1")


class Scripted:
    """A Runner whose answers come from a function of the profile; records every call."""

    def __init__(self, answer: Callable[[Profile], Measurement]) -> None:
        self.answer, self.profiles = answer, []

    def measure(self, model: Path, profile: Profile, cancel=None) -> Measurement:
        self.profiles.append(profile)
        return self.answer(profile)

    @property
    def shape(self) -> list[tuple[int, str, str]]:
        """(ctx_size, cache type, gpu_layers) per call."""
        return [(p.ctx_size, p.cache_type_k, p.gpu_layers) for p in self.profiles]


@pytest.fixture
def dense8b() -> ModelInfo:
    return _info(8.5e9, 32, 8, 131072)


@pytest.fixture
def dense27b() -> ModelInfo:
    return _info(12e9, 64, 8, 131072, n_embd=5120)


def _run(intent: str, info: ModelInfo, host: Host, runner: Scripted, **kw) -> TuneResult:
    return tune(intents.by_key[intent], info, host, Profile(), runner, **kw)


# --------------------------------------------------------------------------- stepping down

def test_plan_loads_first_try_and_gpu_plan_stops(dense8b: ModelInfo) -> None:
    runner = Scripted(lambda p: ok(10))
    r = _run("chat", dense8b, _host(), runner)
    assert r.ok and len(r.trials) == 1 and r.best.how == HOW_PLAN
    assert runner.shape == [(16384, "f16", "all")]       # target reached: nothing to step up


def test_failure_steps_down_ctx_then_q8_then_spill(dense8b: ModelInfo) -> None:
    """16384 -> 8192 (the minimum) -> q8_0 KV -> let llama.cpp place layers."""
    def answer(p: Profile) -> Measurement:
        return ok(8) if p.gpu_layers == "auto" else oom()
    runner = Scripted(answer)
    r = _run("chat", dense8b, _host(), runner)
    assert runner.shape == [(16384, "f16", "all"), (8192, "f16", "all"),
                            (8192, "q8_0", "all"), (8192, "q8_0", "auto")]
    assert r.ok and r.best.profile.gpu_layers == "auto" and r.best.profile.fit == "on"
    assert r.best.profile.fit_target > 0 and r.best.verdict == "GPU+CPU"
    assert [t.measurement.ok for t in r.trials] == [False, False, False, True]


def test_step_down_candidates_directly(dense8b: ModelInfo) -> None:
    host = _host()
    space = SearchSpace(intents.by_key["chat"], dense8b, host)
    c = space.from_plan(planner.plan(space.intent, dense8b, host, Profile()))
    seq = [c]
    while (n := space.down(seq[-1])) is not None:
        seq.append(n)
    assert [x.how for x in seq] == [HOW_PLAN, HOW_CTX_DOWN, HOW_KV_Q8, HOW_SPILL]
    assert [x.ctx_per_slot for x in seq] == [16384, 8192, 8192, 8192]
    assert seq[2].profile.cache_type_v == "q8_0" and seq[2].profile.flash_attn == "on"
    assert seq[3].verdict == "GPU+CPU"


def test_no_q8_step_for_speed_and_no_spill_without_second_pool(dense8b: ModelInfo) -> None:
    host = _host()
    speed = SearchSpace(intents.by_key["speed"], dense8b, host)
    c = speed.from_plan(planner.plan(speed.intent, dense8b, host, Profile()))
    assert [x.how for x in _chain(speed, c)] == [HOW_PLAN, HOW_SPILL]   # 8192 -> 4096 -> ...
    uni = _host()
    uni.gpus = [Device.of("Metal", "M", int(16 * GB), int(16 * GB))]
    uni.unified = True
    sp = SearchSpace(intents.by_key["chat"], dense8b, uni)
    c = sp.from_plan(planner.plan(sp.intent, dense8b, uni, Profile()))
    assert HOW_SPILL not in [x.how for x in _chain(sp, c)]


def _chain(space: SearchSpace, first: Candidate) -> list[Candidate]:
    """Distinct step kinds on the way down (ctx steps collapsed)."""
    out, cur = [first], first
    while (n := space.down(cur)) is not None:
        if n.how != HOW_CTX_DOWN:
            out.append(n)
        cur = n
    return out


def test_failure_that_is_not_about_memory_stops_at_one_trial(dense8b: ModelInfo) -> None:
    runner = Scripted(lambda p: Measurement(error=f"{ERR_ARGS}: -fit"))
    r = _run("chat", dense8b, _host(), runner)
    assert len(runner.profiles) == 1 and not r.ok and r.best is None and r.measurement is None
    assert len(r.trials) == 1 and "unknown argument" in r.message

    runner = Scripted(lambda p: Measurement(error="load failed: unknown model architecture"))
    assert len(_run("chat", dense8b, _host(), runner).trials) == 1


def test_everything_fails_gives_nothing_loaded(dense8b: ModelInfo) -> None:
    runner = Scripted(lambda p: oom())
    r = _run("chat", dense8b, _host(), runner, max_trials=10)
    assert not r.ok and r.best is None and r.measurement is None
    assert len(r.trials) == 4                       # the ladder ends: ctx, q8, spill, nothing left
    assert r.message.startswith("nothing loaded after 4 trials") and "out of memory" in r.message


# --------------------------------------------------------------------------- stepping up

def test_step_up_when_the_measurement_shows_slack(dense27b: ModelInfo) -> None:
    """16 GB card: the plan is 8192; the model took 11 GB, leaving room for 16384."""
    host = _host(16)
    assert planner.plan(intents.by_key["chat"], dense27b, host, Profile()).ctx_per_slot == 8192
    runner = Scripted(lambda p: ok(11))
    r = _run("chat", dense27b, host, runner)
    assert [s[0] for s in runner.shape] == [8192, 16384]
    assert r.best.how == HOW_CTX_UP and r.best.ctx_per_slot == 16384 and r.ok


def test_no_step_up_without_enough_slack(dense27b: ModelInfo) -> None:
    runner = Scripted(lambda p: ok(14))       # 16 GB - 14 GB - 1.6 GB headroom < one tier of KV
    r = _run("chat", dense27b, _host(16), runner)
    assert len(runner.profiles) == 1 and r.best.ctx_per_slot == 8192


def test_failed_step_up_keeps_the_last_good_trial(dense27b: ModelInfo) -> None:
    runner = Scripted(lambda p: ok(11) if p.ctx_size == 8192 else oom())
    r = _run("chat", dense27b, _host(16), runner)
    assert [s[0] for s in runner.shape] == [8192, 16384]      # no step down afterwards
    assert r.ok and r.best.ctx_per_slot == 8192 and r.measurement.ok
    assert [t.measurement.ok for t in r.trials] == [True, False]


def test_step_up_never_exceeds_the_target_or_leaves_the_gpu(dense8b: ModelInfo) -> None:
    host = _host()
    space = SearchSpace(intents.by_key["chat"], dense8b, host)
    top = space.from_plan(planner.plan(space.intent, dense8b, host, Profile()))
    assert space.up(top, ok(1)) is None                         # already at the target
    spilled = space.candidate(top.profile.merged(gpu_layers="auto"), HOW_SPILL)
    assert space.up(spilled, ok(1)) is None                     # GPU+CPU: VRAM is full by design
    assert space.up(top, oom()) is None                         # a failure shows no slack


# --------------------------------------------------------------------------- threads trial

def test_gpu_cpu_plan_gets_one_threads_trial_and_keeps_the_faster(dense27b: ModelInfo) -> None:
    host = _host(16, threads=16)
    plan = planner.plan(intents.by_key["agent"], dense27b, host, Profile())
    assert plan.verdict == "GPU+CPU" and plan.profile.threads == 8
    runner = Scripted(lambda p: ok(14, gen=20.0 if p.threads == 8 else 26.0))
    r = _run("agent", dense27b, host, runner)
    assert [p.threads for p in runner.profiles] == [8, 16]
    assert r.best.how == HOW_THREADS and r.measurement.gen_tps == 26.0
    # the slower threads variant loses
    runner = Scripted(lambda p: ok(14, gen=20.0 if p.threads == 8 else 18.0))
    r = _run("agent", dense27b, host, runner)
    assert [p.threads for p in runner.profiles] == [8, 16]
    assert r.best.how == HOW_PLAN and r.measurement.gen_tps == 20.0


def test_no_threads_trial_when_already_at_all_cores_or_all_gpu(dense27b: ModelInfo,
                                                                dense8b: ModelInfo) -> None:
    host = _host(16, threads=8)               # physical == logical
    runner = Scripted(lambda p: ok(14))
    assert len(_run("agent", dense27b, host, runner).trials) == 1
    runner = Scripted(lambda p: ok(9))
    assert len(_run("agent", dense8b, _host(), runner).trials) == 1


def test_never_repeats_an_identical_profile(dense27b: ModelInfo) -> None:
    host = _host(16, threads=16)
    runner = Scripted(lambda p: ok(14))
    _run("agent", dense27b, host, runner, max_trials=10)
    keys = [json.dumps(p.to_dict(), sort_keys=True) for p in runner.profiles]
    assert len(keys) == len(set(keys))


def test_next_candidate_ends_after_a_threads_trial(dense27b: ModelInfo) -> None:
    host = _host(16, threads=16)
    space = SearchSpace(intents.by_key["agent"], dense27b, host)
    first = space.from_plan(planner.plan(space.intent, dense27b, host, Profile()))
    hist = [(first, ok(14))]
    second = next_candidate(space, hist)
    assert second.how == HOW_THREADS
    assert next_candidate(space, [*hist, (second, ok(14, gen=99))]) is None
    assert next_candidate(space, [*hist, (second, oom())]) is None


def test_trial_cap(dense8b: ModelInfo) -> None:
    runner = Scripted(lambda p: oom())
    assert len(_run("chat", dense8b, _host(), runner, max_trials=2).trials) == 2
    runner = Scripted(lambda p: oom())
    r = _run("chat", dense8b, _host(), runner, max_trials=0)
    assert runner.profiles == [] and not r.ok and r.trials == []


# --------------------------------------------------------------------------- scoring

def _cand(space: SearchSpace, ctx: int, kv: str = "f16") -> Candidate:
    prof = Profile(ctx_size=ctx, parallel=1, cache_type_k=kv, cache_type_v=kv)
    return space.candidate(prof, HOW_PLAN)


def test_score_prefers_context_then_speed(dense8b: ModelInfo) -> None:
    space = SearchSpace(intents.by_key["chat"], dense8b, _host())
    a, b, c = _cand(space, 8192), _cand(space, 16384), _cand(space, 16384, "q8_0")
    assert choose_best(space, [(a, ok(gen=90)), (b, ok(gen=40))])[0] is b
    assert choose_best(space, [(b, ok(gen=40)), (c, ok(gen=44))])[0] is c
    assert choose_best(space, [(b, ok(gen=40)), (c, ok(gen=40))])[0] is b       # tie: earlier


def test_speed_intent_ranks_tokens_per_second_first(dense8b: ModelInfo) -> None:
    space = SearchSpace(intents.by_key["speed"], dense8b, _host())
    slow_big, fast_small = _cand(space, 8192), _cand(space, 4096)
    hist = [(slow_big, ok(gen=40)), (fast_small, ok(gen=70))]
    assert choose_best(space, hist)[0] is fast_small
    chat = SearchSpace(intents.by_key["chat"], dense8b, _host())
    big, small = _cand(chat, 16384), _cand(chat, 8192)
    assert choose_best(chat, [(big, ok(gen=40)), (small, ok(gen=70))])[0] is big


def test_failed_trials_never_win_and_below_minimum_is_a_last_resort(dense8b: ModelInfo) -> None:
    space = SearchSpace(intents.by_key["chat"], dense8b, _host())
    small, big = _cand(space, 4096), _cand(space, 16384)      # chat minimum is 8192
    assert choose_best(space, [(big, oom())]) is None
    assert choose_best(space, []) is None
    assert choose_best(space, [(small, ok(gen=90)), (big, oom())])[0] is small
    assert choose_best(space, [(small, ok(gen=90)), (_cand(space, 8192), ok(gen=10))])[0].ctx_per_slot == 8192


def test_embedding_scores_by_prompt_speed() -> None:
    info = _info(0.3e9, 12, 12, 512, pooling=1, n_embd=768, head_dim=64)
    space = SearchSpace(intents.by_key["embed"], info, _host())
    a = space.candidate(Profile(ctx_size=2048, parallel=4), HOW_PLAN)
    b = space.candidate(Profile(ctx_size=2048, parallel=4, threads=3), HOW_PLAN)
    hist = [(a, ok(gen=0, prompt=900)), (b, ok(gen=0, prompt=1500))]
    assert choose_best(space, hist)[0] is b


def test_below_minimum_winner_is_reported() -> None:
    """A plan already below the minimum still tunes; the message says so."""
    big = _info(12e9, 64, 8, 131072, n_embd=5120)
    r = _run("chat", big, _host(15, ram_gb=0.1), Scripted(lambda p: ok(14.2)))
    assert r.ok and r.best.ctx_per_slot == 4096 and "below the chat minimum of 8192" in r.message


def test_plan_that_cannot_fit_is_not_launched(dense8b: ModelInfo) -> None:
    runner = Scripted(lambda p: ok())
    r = _run("chat", _info(80e9, 80, 8, 131072), _host(6, ram_gb=16), runner)
    assert not r.ok and r.trials == [] and runner.profiles == []
    assert r.message.startswith("not tried")


# --------------------------------------------------------------------------- trials, progress, cancel

def test_trials_record_the_uncalibrated_estimate(dense8b: ModelInfo) -> None:
    """The factor must never compound: estimated_total is Budget.raw_total()."""
    cal = Calibration([Trial("/x", 1.0, "chat", {}, 1000, "GPU", "cuda",
                             Measurement(ok=True, vram_used=1400))])
    assert cal.factor("cuda") == pytest.approx(1.4)
    host = _host()
    runner = Scripted(lambda p: ok(10))
    r = _run("chat", dense8b, host, runner, calibration=cal)
    t = r.trials[0]
    raw = planner.estimate_budget(dense8b, runner.profiles[0], host).raw_total()
    plan = planner.plan(intents.by_key["chat"], dense8b, host, Profile(), calibration=cal)
    assert t.estimated_total == raw and plan.budget.total() == round(raw * 1.4)
    assert (t.model, t.intent, t.backend, t.verdict) == ("m.gguf", "chat", "cuda", "GPU")
    assert t.profile == runner.profiles[0].to_dict() and t.measurement == r.measurement


def test_trial_carries_the_model_mtime(tmp_path: Path, dense8b: ModelInfo) -> None:
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x")
    info = _info(8.5e9, 32, 8, 131072, path=str(f))
    r = _run("chat", info, _host(), Scripted(lambda p: ok()))
    assert r.trials[0].mtime == f.stat().st_mtime and r.trials[0].model == str(f)


def test_progress_events(dense8b: ModelInfo) -> None:
    events: list[TrialEvent] = []
    runner = Scripted(lambda p: ok(8) if p.ctx_size == 8192 else oom())
    r = _run("chat", dense8b, _host(), runner, max_trials=3, progress=events.append)
    assert [(e.index, e.phase, e.total) for e in events] == [
        (1, "start", 3), (1, "done", 3), (2, "start", 3), (2, "done", 3)]
    assert events[0].measurement is None and events[1].measurement.error.startswith(ERR_OOM)
    assert events[2].summary.startswith("ctx down: 8192 ctx") and events[3].measurement.ok
    assert events[0].summary == "plan: 16384 ctx, f16 KV, GPU"
    assert r.ok


def test_cancel_between_trials(dense27b: ModelInfo) -> None:
    runner = Scripted(lambda p: ok(11))            # would step up to 16384 next
    polled: list[int] = []

    def cancel() -> bool:
        polled.append(len(runner.profiles))
        return len(runner.profiles) >= 1           # after the first trial

    r = _run("chat", dense27b, _host(16), runner, cancel=cancel, max_trials=10)
    assert len(runner.profiles) == 1 and polled == [0, 1]
    assert r.cancelled and r.ok and r.message.startswith("cancelled; best so far")
    assert r.best.ctx_per_slot == 8192


def test_cancel_before_anything_ran(dense8b: ModelInfo) -> None:
    runner = Scripted(lambda p: ok())
    r = _run("chat", dense8b, _host(), runner, cancel=lambda: True)
    assert runner.profiles == [] and r.cancelled and not r.ok
    assert r.message == "cancelled before any trial ran"


def test_keyboard_interrupt_keeps_the_trials_so_far(dense27b: ModelInfo) -> None:
    calls: list[Profile] = []

    def answer(p: Profile) -> Measurement:
        calls.append(p)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return ok(11)

    r = _run("chat", dense27b, _host(16), Scripted(answer), max_trials=10)
    assert len(calls) == 2 and len(r.trials) == 1       # the interrupted trial is dropped
    assert r.cancelled and r.ok and r.message.startswith("cancelled; best so far")


def test_keyboard_interrupt_in_the_first_trial(dense8b: ModelInfo) -> None:
    def answer(p: Profile) -> Measurement:
        raise KeyboardInterrupt

    r = _run("chat", dense8b, _host(), Scripted(answer))
    assert r.cancelled and not r.ok and r.trials == []
    assert r.message == "cancelled before any trial ran"


# --------------------------------------------------------------------------- save_result

def test_save_result_stores_the_tuned_preset_and_all_trials(isolated: SimpleNamespace,
                                                             dense8b: ModelInfo) -> None:
    runner = Scripted(lambda p: ok(8) if p.ctx_size == 8192 else oom())
    r = _run("chat", dense8b, _host(), runner)
    cfg, cal = Config(), Calibration()
    cfg.set_preset("m.gguf", Preset("chat", Profile(ctx_size=1), "estimated"))
    save_result(cfg, "m.gguf", r, cal)
    p = cfg.preset("m.gguf", "chat")
    assert p.source == "tuned" and p.intent == "chat" and p.measured == r.measurement
    assert p.profile == r.best.profile and cfg.models["m.gguf"].active == "chat"
    assert len(cal.trials) == 2 and len(Calibration.load().trials) == 2
    assert Config.load().preset("m.gguf", "chat").measured == r.measurement
    assert Calibration.load().trials_for("m.gguf", "chat")[1].measurement.ok is True


def test_save_result_does_not_activate_unless_asked(isolated: SimpleNamespace,
                                                    dense8b: ModelInfo) -> None:
    r = _run("agent", dense8b, _host(), Scripted(lambda p: ok(8)))
    cfg = Config()
    cfg.set_preset("m.gguf", Preset("chat", Profile(), "estimated"))
    save_result(cfg, "m.gguf", r, Calibration())
    assert cfg.models["m.gguf"].active == "chat" and "agent" in cfg.models["m.gguf"].presets
    save_result(cfg, "m.gguf", r, Calibration(), activate=True)
    assert cfg.models["m.gguf"].active == "agent"


def test_failed_tune_records_trials_but_changes_no_preset(isolated: SimpleNamespace,
                                                          dense8b: ModelInfo) -> None:
    r = _run("chat", dense8b, _host(), Scripted(lambda p: oom()), max_trials=2)
    cfg, cal = Config(), Calibration()
    cfg.set_preset("m.gguf", Preset("chat", Profile(ctx_size=1), "edited"))
    before = cfg.preset("m.gguf", "chat")
    save_result(cfg, "m.gguf", r, cal)
    assert cfg.preset("m.gguf", "chat") == before and before.source == "edited"
    assert not (isolated.cfg_dir / "config.json").exists()
    assert len(Calibration.load().trials) == 2


def test_tune_then_calibrate_feeds_the_next_plan(isolated: SimpleNamespace,
                                                 dense8b: ModelInfo) -> None:
    host = _host()
    est = planner.plan(intents.by_key["chat"], dense8b, host, Profile()).budget.raw_total()
    r = _run("chat", dense8b, host, Scripted(lambda p: ok(est * 1.2 / GB)))
    cal = Calibration()
    save_result(Config(), "m.gguf", r, cal)
    assert Calibration.load().factor("cuda") == pytest.approx(1.2, rel=1e-3)


# --------------------------------------------------------------------------- preflight

def _plan(info: ModelInfo, host: Host, intent: str = "chat"):
    return planner.plan(intents.by_key[intent], info, host, Profile())


def test_preflight_ok_and_refusals(dense8b: ModelInfo, dense27b: ModelInfo) -> None:
    host = _host()
    plan = _plan(dense8b, host)
    assert preflight(plan, host) is None
    busy = _host()
    busy.gpus = [Device.of("CUDA0", "G0", int(16 * GB), int(4 * GB))]          # another model loaded
    msg = preflight(plan, busy)
    assert msg and "free VRAM" in msg and "4.0 GB" in msg and "unload" in msg
    # a small shortfall is tolerated: the estimate is not exact
    need = plan.budget.total()
    almost = _host()
    almost.gpus = [Device.of("CUDA0", "G0", int(16 * GB), int(need * 0.95))]
    assert preflight(plan, almost) is None


def test_preflight_by_verdict(dense27b: ModelInfo) -> None:
    host = _host()
    spill = _plan(dense27b, host, "agent")
    assert spill.verdict == "GPU+CPU" and preflight(spill, host) is None
    low = _host(ram_gb=2)
    low.gpus = [Device.of("CUDA0", "G0", int(16 * GB), int(6 * GB))]
    assert "free VRAM + RAM" in preflight(spill, low)
    cpu = _host(0)
    cplan = _plan(_info(4e9, 32, 8, 8192), cpu)
    assert cplan.verdict == "CPU" and preflight(cplan, cpu) is None
    cpu.ram_free = int(2 * GB)
    assert "free RAM" in preflight(cplan, cpu)
    no = _plan(_info(80e9, 80, 8, 131072), _host(6, ram_gb=16))
    assert no.verdict == "NO" and "more than this machine has" in preflight(no, _host(6, ram_gb=16))


def test_a_trial_cancelled_in_flight_is_neither_recorded_nor_stepped_down_from(
        dense27b: ModelInfo) -> None:
    from llamactl.measure import ERR_CANCELLED

    runner = Scripted(lambda p: Measurement(error=f"{ERR_CANCELLED}: stopped while loading"))
    events: list = []
    r = _run("chat", dense27b, _host(16), runner, progress=events.append, cancel=lambda: False,
             max_trials=10)
    assert len(runner.profiles) == 1                    # no step-down trial followed
    assert r.cancelled and not r.ok and r.trials == []
    assert [e.phase for e in events] == ["start", "done"] and events[1].measurement.cancelled


def test_cancel_callable_is_handed_to_the_runner(dense8b: ModelInfo) -> None:
    seen: list = []

    class Spy(Scripted):
        def measure(self, model, profile, cancel=None):
            seen.append(cancel)
            return super().measure(model, profile, cancel)

    cancel = lambda: False  # noqa: E731
    _run("chat", dense8b, _host(), Spy(lambda p: ok()), cancel=cancel)
    assert seen and all(c is cancel for c in seen)
