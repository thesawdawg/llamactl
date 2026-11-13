"""Preset goal persistence, constrained search and real runner workload measurements."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llamactl import tuner
from llamactl.config import Preset, Profile
from llamactl.goals import TuningGoals
from llamactl.intents import by_key
from llamactl.measure import Measurement, WorkloadMeasurement
from llamactl.tuner import ServerRunner, goal_failure, tune
from test_server_runner import Resp, World, kv_slot, make, model, world
from test_tuner import Scripted, _host, _info, ok


def test_preset_goals_survive_json_without_changing_legacy_defaults() -> None:
    """Goals belong to each saved preset and old configs receive safe defaults."""
    goals = TuningGoals(25, 100, 30, 64000, "full")
    preset = Preset("agent", Profile(ctx_size=12345), goals=goals)
    assert Preset.from_dict(json.loads(json.dumps(preset.to_dict()))) == preset
    assert Preset.from_dict({}).goals == TuningGoals()
    intent = goals.resolve_intent(by_key["agent"])
    assert intent.ctx_target == intent.ctx_max == 64000


@pytest.mark.parametrize("kwargs", [{"min_effective_tps": float("nan")},
                                   {"min_prompt_tps": -1}, {"max_context": -2},
                                   {"benchmark": "missing"}])
def test_invalid_goals_rejected(kwargs: dict[str, Any]) -> None:
    """Reject unsafe target values before launching a candidate."""
    with pytest.raises(ValueError):
        TuningGoals(**kwargs)


def test_each_workload_must_pass_instead_of_hiding_failure_in_an_average() -> None:
    """A fast short request cannot mask a slow full-window request."""
    fast = WorkloadMeasurement("2k", 2048, 128, 1, 3000, 100)
    slow = WorkloadMeasurement("95%", 50000, 128, 20, 500, 30)
    measured = Measurement(ok=True, effective_tps=100, workloads=(fast, slow))
    assert "95%" in goal_failure(TuningGoals(), measured)
    assert Measurement.from_dict(json.loads(json.dumps(measured.to_dict()))) == measured


def test_a_failing_context_bisects_against_the_largest_pass() -> None:
    """A throughput failure bounds the search; bisection keeps the largest reliable pass."""
    info = _info(1e9, 4, 1, 131072)
    runner = Scripted(lambda profile: ok(vram=2, effective_tps=(19 if profile.ctx_size > 61000 else 25)))
    result = tune(by_key["agent"], info, _host(), Profile(), runner,
                  goals=TuningGoals(max_context=65536, benchmark="quick"), max_trials=10)
    contexts = [p.ctx_size for p in runner.profiles]
    assert contexts == [32768, 65536, 49152, 57344, 61440, 59392]
    assert result.ok and result.best is not None
    assert result.best.ctx_per_slot == 59392           # largest pass within ~5% of the failure
    assert result.measurement is not None and result.measurement.effective_tps >= 20


def test_goal_failure_never_saves_a_too_slow_candidate() -> None:
    """Unmet goals remain an explicit failure, not an accepted best effort."""
    runner = Scripted(lambda p: ok(vram=2, effective_tps=5))
    result = tune(by_key["chat"], _info(1e9, 4, 1, 32768), _host(), Profile(), runner,
                  goals=TuningGoals(), max_trials=2)
    assert not result.ok and result.best is None
    assert "goals unmet" in result.message and "trial budget" in result.message


def test_reaching_context_ceiling_finishes_without_forcing_failure() -> None:
    """A requested smaller context ceiling is a hard upper bound."""
    runner = Scripted(lambda p: ok(vram=2, effective_tps=25))
    result = tune(by_key["agent"], _info(1e9, 4, 1, 131072), _host(), Profile(), runner,
                  goals=TuningGoals(max_context=8192))
    assert result.ok and [p.ctx_size for p in runner.profiles] == [8192]
    assert "trial budget" not in result.message


@pytest.mark.parametrize("level, expected", [("quick", 1), ("normal", 2), ("full", 5)])
def test_workloads_use_real_token_ids_and_whole_request_duration(
        world: World, model: Path, level: str, expected: int) -> None:
    """Tokenizer latency and startup are excluded while the turn's input wait counts."""
    world.routes[("POST", "/tokenize")] = Resp(200, {"tokens": list(range(40000))})
    kv_slot(world, gen=100)
    measured = make(world, goals=TuningGoals(benchmark=level)).measure(
        model, Profile(ctx_size=32768))
    assert measured.ok, measured.error
    assert len(measured.workloads) == expected
    assert measured.effective_tps == 25
    assert all(w.generated_tokens == 100 and w.request_s == 4 for w in measured.workloads)
    requests = [r for r in world.requests if r["path"] == "/completion"]
    assert all(len(r["json"]["prompt"]) + r["json"]["n_predict"] < .95 * 32768
               for r in requests)
    assert requests[0]["json"]["n_predict"] == tuner.WARMUP_PREDICT   # warmup is discarded
    assert measured.request_s == 4 * expected
    if expected > 1:
        big = measured.workloads[-1]
        assert big.cached_tokens > 0 and big.prompt_tokens <= tuner.TURN_TOKENS


def test_normal_small_context_labels_reduced_harness_workload(world: World, model: Path) -> None:
    """A small model cannot be falsely described as having passed a 25k workload."""
    world.routes[("POST", "/tokenize")] = Resp(200, {"tokens": list(range(10000))})
    kv_slot(world, gen=100, advance=1)
    measured = make(world, goals=TuningGoals()).measure(model, Profile(ctx_size=8192))
    assert measured.ok and "reduced" in measured.workloads[-1].label
    assert measured.workloads[-1].occupancy_tokens < 8192


def test_embedding_has_no_output_speed_floor() -> None:
    """Embeddings honor optional input floors without requiring nonexistent output."""
    info = _info(1e9, 4, 1, 2048, pooling=1)
    runner = Scripted(lambda p: ok(vram=2, gen=0, prompt=100))
    result = tune(by_key["embed"], info, _host(), Profile(), runner,
                  goals=TuningGoals(min_prompt_tps=50))
    assert result.ok
    failed = tune(by_key["embed"], info, _host(), Profile(), runner,
                  goals=TuningGoals(min_prompt_tps=200), max_trials=1)
    assert not failed.ok


def test_nonfinite_speed_cannot_pass_constraint() -> None:
    """Malformed measurements never turn a constrained trial into a winner."""
    assert goal_failure(TuningGoals(), Measurement(ok=True, effective_tps=float("nan")))


@pytest.mark.parametrize("value", [1024.5, "1024.5", True, float("inf")])
def test_loading_goals_never_truncates_fractional_context(value: Any) -> None:
    """Invalid stored ceilings fall back rather than silently selecting a smaller one."""
    assert TuningGoals.from_dict({"max_context": value}) == TuningGoals()


def test_cancelled_trial_emits_done_without_becoming_a_saved_trial() -> None:
    """The UI can mark the current row cancelled instead of leaving it running."""
    runner = Scripted(lambda p: Measurement(error="cancelled: stopped"))
    events = []
    result = tune(by_key["chat"], _info(1e9, 4, 1, 32768), _host(), Profile(), runner,
                  goals=TuningGoals(), progress=events.append)
    assert result.cancelled and not result.ok and not result.trials
    assert [event.phase for event in events] == ["start", "done"]
    assert events[-1].measurement.cancelled


@pytest.mark.parametrize("change", [{"prompt_n": 12}, {"cache_n": 128},
                                   {"prompt_per_second": float("nan")},
                                   {"predicted_per_second": 0}, {"predicted_n": 9999}])
def test_benchmark_rejects_cached_incomplete_or_invalid_measurements(
        world: World, model: Path, change: dict[str, Any]) -> None:
    """Unaccounted prompt tokens and invalid rates cannot win tuning."""
    world.routes[("POST", "/tokenize")] = Resp(200, {"tokens": list(range(10000))})
    kv_slot(world, gen=100, advance=1, change=change)
    measured = make(world, goals=TuningGoals(benchmark="quick")).measure(
        model, Profile(ctx_size=8192))
    assert not measured.ok and "benchmark failed" in measured.error


def test_saved_sampling_settings_seed_constrained_trials() -> None:
    """Memory search preserves the supplied preset's unrelated sampling settings."""
    runner = Scripted(lambda p: ok(vram=2, effective_tps=25))
    result = tune(by_key["chat"], _info(1e9, 4, 1, 32768), _host(),
                  Profile(temp=.3, top_k=17, extra_args="--jinja"), runner,
                  goals=TuningGoals(max_context=8192))
    assert result.ok and result.best.profile.temp == .3
    assert all(p.top_k == 17 and p.extra_args == "--jinja" for p in runner.profiles)


def test_each_context_replans_dense_placement_instead_of_retaining_ceiling_spill() -> None:
    """A large context ceiling must not force smaller trials to keep unnecessary CPU work."""
    info = _info(8.5e9, 32, 8, 131072)
    runner = Scripted(lambda profile: ok(vram=10, effective_tps=(25 if profile.gpu_layers == "all" else 5)))
    result = tune(by_key["agent"], info, _host(), Profile(temp=.4), runner,
                  goals=TuningGoals(max_context=131072), max_trials=3)
    contexts = [p.ctx_size for p in runner.profiles]
    assert contexts == [32768, 65536, 131072]
    assert [p.gpu_layers for p in runner.profiles] == ["all", "all", "auto"]
    assert result.ok and result.best.ctx_per_slot == 65536
    assert result.best.profile.temp == .4


def test_bisection_uses_measured_vram_to_replan_dense_placement() -> None:
    """After a spill fails the floor, bisection probes run fully on the GPU when the
    measured bytes at lower contexts prove they fit — not the planner's static split."""
    info = _info(8.5e9, 32, 8, 131072)
    runner = Scripted(lambda profile: ok(vram=10, effective_tps=(25 if profile.gpu_layers == "all" else 5)))
    result = tune(by_key["agent"], info, _host(), Profile(top_k=17), runner,
                  goals=TuningGoals(max_context=131072), max_trials=32)
    assert [(p.ctx_size, p.gpu_layers) for p in runner.profiles] == [
        (32768, "all"), (65536, "all"), (131072, "auto"),
        (98304, "all"), (114688, "all"), (122880, "all"), (126976, "all")]
    assert result.ok and result.best.profile.gpu_layers == "all"
    assert result.best.ctx_per_slot == 126976
    assert result.best.profile.top_k == 17
