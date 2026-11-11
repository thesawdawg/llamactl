"""Pilot tests: TuneScreen with a fake runner (no llama.cpp)."""
from __future__ import annotations

import threading
from types import SimpleNamespace

from textual.widgets import Button, DataTable, Label, OptionList, Static

from fakes import FakeGateway, FakeRunner, ok, oom, plain, wait_for
from llamactl.config import Config
from llamactl.gateway import LOADED
from llamactl.screens.tune import TuneScreen, split_summary, trial_cells
from test_ui_shell import SIZE, add_preset, make_app
from test_ui_models import ready


def shown(app, bid: str) -> bool:
    """True when a button of the tune screen is visible."""
    return app.screen.query_one(f"#{bid}", Button).display


async def open_tune(pilot, app) -> TuneScreen:
    """Press `t` in the Models view and wait for the screen."""
    app.views["models"].model_table.focus()
    await pilot.press("t")
    assert await wait_for(pilot, lambda: isinstance(app.screen, TuneScreen))
    return app.screen


def test_split_summary_and_cells() -> None:
    assert split_summary("ctx down: 8192 ctx, f16 KV, GPU") == ("ctx down, f16 KV, GPU", "8192")
    assert split_summary("threads: 4096 ctx, q8_0 KV, GPU+CPU, 16 threads") == (
        "threads, q8_0 KV, GPU+CPU, 16 threads", "4096")
    assert split_summary("weird") == ("weird", "-")
    cells = trial_cells(2, "plan: 8192 ctx, f16 KV, GPU", ok(41.2, int(7.5e9)))
    assert [str(c) for c in cells] == ["2", "plan, f16 KV, GPU", "8192", "3", "7.5 GB", "900",
                                       "41.2", "ok"]
    assert str(trial_cells(1, "plan: 8192 ctx, f16 KV, GPU", None)[7]) == "running"
    assert str(trial_cells(1, "plan: 8192 ctx, f16 KV, GPU", oom())[7]).startswith("out of memory")


async def test_tune_shows_trials_saves_the_winner_and_use_activates(
        ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    add_preset(ui_env, m, "edited", "chat")                  # active preset is another one
    runner = FakeRunner(oom(), ok(37.0, int(2e9)))
    gw = FakeGateway(ui_env.cfg, running=True)
    app = make_app(ui_env, gateway=gw, runner=runner)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        mv = app.views["models"]
        mv.intent_table.move_cursor(row=mv.intent_table.get_row_index("speed"))
        assert await wait_for(pilot, lambda: mv.selected_intent() == "speed")
        scr = await open_tune(pilot, app)
        t = scr.query_one("#t_trials", DataTable)
        assert await wait_for(pilot, lambda: scr.state == "done")
        assert t.row_count == 2
        assert str(t.get_cell("1", "out")).startswith("out of memory")
        assert str(t.get_cell("2", "out")) == "ok" and str(t.get_cell("2", "tg")) == "37.0"
        assert "Pre-flight ok" in plain(scr.query_one("#t_pre", Static))
        summary = plain(scr.query_one("#t_summary", Static))
        assert "Winner" in summary and "37.0 tok/s" in summary and "Estimate" in summary
        assert "Not active yet" in summary
        p = ui_env.cfg.preset(str(m), "speed")
        assert p.source == "tuned" and p.measured.gen_tps == 37.0
        assert ui_env.cfg.models[str(m)].active == "chat"         # tune alone does not activate
        assert gw.called("reload")                                 # the running gateway re-read
        assert shown(app, "use") and shown(app, "keep") and shown(app, "close")
        assert not shown(app, "cancel")
        assert Config.load().preset(str(m), "speed").source == "tuned"
        scr.query_one("#use", Button).press()
        assert await wait_for(pilot, lambda: not isinstance(app.screen, TuneScreen))
        assert ui_env.cfg.models[str(m)].active == "speed"
        assert await wait_for(pilot, lambda: "speed tuned" in str(
            mv.model_table.get_cell(str(m), "preset")))
        assert str(mv.model_table.get_cell(str(m), "measured")) == "37.0 t/s"


async def test_keep_as_is_leaves_the_active_preset(ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    add_preset(ui_env, m, "edited", "chat")
    app = make_app(ui_env, runner=FakeRunner(ok()))
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        mv = app.views["models"]
        mv.intent_table.move_cursor(row=mv.intent_table.get_row_index("api"))
        assert await wait_for(pilot, lambda: mv.selected_intent() == "api")
        scr = await open_tune(pilot, app)
        assert await wait_for(pilot, lambda: scr.state == "done")
        scr.query_one("#keep", Button).press()
        assert await wait_for(pilot, lambda: not isinstance(app.screen, TuneScreen))
        assert ui_env.cfg.models[str(m)].active == "chat"
        assert ui_env.cfg.preset(str(m), "api").source == "tuned"


async def test_tuning_the_active_preset_needs_no_use_button(ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    add_preset(ui_env, m, "estimated", "chat")
    app = make_app(ui_env, runner=FakeRunner(ok(50.0)))
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        scr = await open_tune(pilot, app)
        assert await wait_for(pilot, lambda: scr.state == "done")
        assert "This preset is active" in plain(scr.query_one("#t_summary", Static))
        assert not shown(app, "use") and shown(app, "keep")


async def test_cancel_interrupts_the_trial_in_flight(ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    gate = threading.Event()                                # never set: only cancel ends trial 2
    runner = FakeRunner(oom(), oom(), ok(33.0), gates={1: gate})
    app = make_app(ui_env, runner=runner)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        scr = await open_tune(pilot, app)
        t = scr.query_one("#t_trials", DataTable)
        assert await wait_for(pilot, lambda: t.row_count == 2 and str(t.get_cell("2", "out")) == "running")
        assert shown(app, "cancel") and not shown(app, "close")
        scr.query_one("#cancel", Button).press()
        assert await wait_for(pilot, lambda: scr.state == "done")
        assert runner.cancelled_calls == [1] and len(runner.calls) == 2   # no third trial
        assert str(t.get_cell("2", "out")) == "cancelled"
        assert scr.outcome.result.cancelled and not scr.outcome.result.ok
        assert "nothing loaded after 1 trial" in plain(scr.query_one("#t_summary", Static))
        assert ui_env.cfg.preset(str(m), "chat") is None
        assert len(app.calibration.trials) == 1             # the cancelled trial is not recorded
        assert shown(app, "close") and not shown(app, "keep")


async def test_cancel_between_trials_for_a_runner_that_ignores_it(ui_env: SimpleNamespace) -> None:
    gate = threading.Event()
    runner = FakeRunner(oom(), oom(), ok(33.0), gates={1: gate}, honour_cancel=False)
    ui_env.add_model()
    app = make_app(ui_env, runner=runner)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        scr = await open_tune(pilot, app)
        assert await wait_for(pilot, lambda: len(runner.calls) == 2)
        scr.query_one("#cancel", Button).press()
        assert await wait_for(pilot, lambda: "cancelling" in plain(scr.query_one("#t_progress", Label)))
        assert scr._cancel.is_set() and not shown(app, "cancel")
        gate.set()
        assert await wait_for(pilot, lambda: scr.state == "done")
        assert len(runner.calls) == 2                       # the third trial never started
        assert "nothing loaded after 2 trials" in plain(scr.query_one("#t_summary", Static))
        assert len(app.calibration.trials) == 2             # finished trials are recorded


async def test_closing_while_running_cancels_and_still_records(ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    gate = threading.Event()
    runner = FakeRunner(oom(), oom(), ok(33.0), gates={1: gate}, honour_cancel=False)
    app = make_app(ui_env, runner=runner)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        scr = await open_tune(pilot, app)
        assert await wait_for(pilot, lambda: len(runner.calls) == 2)
        await pilot.press("escape")
        assert await wait_for(pilot, lambda: not isinstance(app.screen, TuneScreen))
        assert scr._cancel.is_set()
        gate.set()                                          # the trial in flight now finishes
        assert await wait_for(pilot, lambda: app.cancel_flags == set())
        assert len(runner.calls) == 2 and len(app.calibration.trials) == 2
        assert ui_env.cfg.preset(str(m), "chat") is None
        assert app.is_running


async def test_tune_result_is_saved_on_the_ui_thread(ui_env: SimpleNamespace, monkeypatch) -> None:
    """The worker only searches: the config and tuning.json are written on the UI thread."""
    from llamactl import tuner

    m = ui_env.add_model()
    threads: list[tuple[str, int]] = []
    real = tuner.save_result

    def spy(*a, **kw):
        threads.append(("save", threading.get_ident()))
        return real(*a, **kw)

    monkeypatch.setattr(tuner, "save_result", spy)
    runner = FakeRunner(ok(41.0))
    orig = runner.measure
    runner.measure = lambda *a, **kw: (threads.append(("measure", threading.get_ident())),
                                       orig(*a, **kw))[1]
    gw = FakeGateway(ui_env.cfg, running=True)
    app = make_app(ui_env, gateway=gw, runner=runner)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        ui_thread = threading.get_ident()
        scr = await open_tune(pilot, app)
        assert await wait_for(pilot, lambda: scr.state == "done")
        kinds = dict(threads)
        assert kinds["save"] == ui_thread and kinds["measure"] != ui_thread
        assert ui_env.cfg.preset(str(m), "chat").source == "tuned"
        assert gw.called("reload") and scr.outcome.reloaded
        assert "re-read" in plain(scr.query_one("#t_summary", Static))


def test_run_tune_without_save_touches_nothing(ui_env: SimpleNamespace) -> None:
    from llamactl import actions, intents
    from llamactl.calibration import Calibration
    from fakes import make_host, make_info

    m = ui_env.add_model()
    cfg, gw = ui_env.cfg, FakeGateway(ui_env.cfg, running=True)
    cal = Calibration()
    info = make_info(m)
    out = actions.run_tune(cfg, info, make_host(), intents.by_key["chat"], FakeRunner(ok()), cal,
                           gw, save=False)
    assert out.result.ok and not out.reloaded
    assert cfg.preset(str(m), "chat") is None and not cal.trials and gw.called("reload") == []
    assert not (ui_env.cfg_dir / "config.json").exists()
    actions.apply_tune(cfg, info, out.result, cal)         # the explicit apply step
    assert cfg.preset(str(m), "chat").source == "tuned" and cal.trials
    assert actions.reload_after_tune(gw, out.result).reloaded


async def test_preflight_refusal_offers_unload_and_retry(ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    host = ui_env.host
    host.gpus[0] = host.gpus[0].__class__.of("CUDA0", "GPU", host.gpus[0].total, int(1e6))
    host.ram_free = int(1e6)
    gw = FakeGateway(ui_env.cfg, running=True)
    add_preset(ui_env, m, "edited", "chat")
    gw.set_state("alpha", LOADED)
    runner = FakeRunner(ok())
    app = make_app(ui_env, gateway=gw, runner=runner)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        scr = await open_tune(pilot, app)
        assert await wait_for(pilot, lambda: scr.state == "refused")
        pre = plain(scr.query_one("#t_pre", Static))
        assert "Pre-flight refused" in pre and "alpha" in pre
        assert shown(app, "unload") and shown(app, "anyway") and shown(app, "close")
        assert runner.calls == []
        # unloading frees memory: the retry passes the pre-flight and tunes
        host.gpus[0] = host.gpus[0].__class__.of("CUDA0", "GPU", host.gpus[0].total, host.gpus[0].total)
        host.ram_free = int(24e9)
        scr.query_one("#unload", Button).press()
        assert await wait_for(pilot, lambda: ("unload", "alpha") in gw.calls)
        assert await wait_for(pilot, lambda: scr.state == "done")
        assert len(runner.calls) == 1


async def test_tune_anyway_skips_the_check(ui_env: SimpleNamespace) -> None:
    ui_env.add_model()
    host = ui_env.host
    host.gpus[0] = host.gpus[0].__class__.of("CUDA0", "GPU", host.gpus[0].total, int(1e6))
    host.ram_free = int(1e6)
    runner = FakeRunner(ok())
    app = make_app(ui_env, runner=runner)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        scr = await open_tune(pilot, app)
        assert await wait_for(pilot, lambda: scr.state == "refused")
        assert not shown(app, "unload")                     # nothing loaded in the gateway
        scr.query_one("#anyway", Button).press()
        assert await wait_for(pilot, lambda: scr.state == "done")
        assert len(runner.calls) == 1


async def test_nothing_loaded_is_reported_without_a_preset(ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    app = make_app(ui_env, runner=FakeRunner(oom()))
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        scr = await open_tune(pilot, app)
        assert await wait_for(pilot, lambda: scr.state == "done")
        assert "nothing loaded" in plain(scr.query_one("#t_summary", Static))
        assert not shown(app, "use") and not shown(app, "keep") and shown(app, "close")
        assert ui_env.cfg.preset(str(m), "chat") is None
        assert app.calibration.trials                       # but the trials are recorded


async def test_tune_from_the_home_suggestion(ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    add_preset(ui_env, m, "estimated", "chat")
    runner = FakeRunner(ok())
    app = make_app(ui_env, runner=runner, running=True)
    async with app.run_test(size=SIZE) as pilot:
        home = app.views["home"]
        assert await wait_for(pilot, lambda: home.query_one("#steps", OptionList).option_count == 1
                              and not app.scanning)
        assert "estimate" in str(home.query_one("#steps", OptionList).get_option_at_index(0).prompt)
        home.query_one("#steps", OptionList).focus()
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: isinstance(app.screen, TuneScreen))
        assert await wait_for(pilot, lambda: app.screen.state == "done")
        assert len(runner.calls) == 1


async def test_a_runner_bug_is_a_message_not_a_crash(ui_env: SimpleNamespace) -> None:
    class Broken:
        def measure(self, model, profile, cancel=None):
            raise OSError("cannot launch llama-server: no such file")

    ui_env.add_model()
    app = make_app(ui_env, runner=Broken())
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        scr = await open_tune(pilot, app)
        assert await wait_for(pilot, lambda: scr.state == "done")
        assert "cannot launch" in plain(scr.query_one("#t_summary", Static))
        assert app.is_running
