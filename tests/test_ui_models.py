"""Pilot tests: the Models view and the flows it starts (picker, plan, edit, serve)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from textual.widgets import Button, Select, Static

from fakes import FakeGateway, FakeStore, plain, wait_for
from llamactl.config import Config, Preset, Profile
from llamactl.gateway import LOADED
from llamactl.measure import Measurement
from llamactl.screens.confirm import ConfirmScreen
from llamactl.screens.intent_picker import IntentPicker
from llamactl.screens.model_workspace import ModelWorkspace, ProfileReview
from llamactl.screens.plan import PlanScreen
from llamactl.screens.preset_editor import PresetEditor
from test_ui_shell import SIZE, add_preset, make_app


def tuned(gen: float = 42.5, build: str = "b1") -> Measurement:
    """A good tuned measurement."""
    return Measurement(ok=True, gen_tps=gen, vram_used=int(5e9), at="t1", build=build)


def cells(app, path: Path) -> list[str]:
    """Plain text of a model row."""
    t = app.views["models"].model_table
    return [str(c) for c in t.get_row(str(path))]


async def ready(pilot, app, n: int = 1) -> None:
    """Wait until n models are listed, read and planned."""
    mv = app.views["models"]
    assert await wait_for(pilot, lambda: len(app.infos) == n and not app.scanning
                          and len(mv.plans) == n and mv.selected_intent() != "")
    await pilot.press("2")
    assert await wait_for(pilot, lambda: mv.intent_table.row_count > 0)


async def test_table_shows_best_for_and_new(ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        row = cells(app, m)
        assert row[0] == "alpha" and row[2] == ""                 # alias, no quant in the name
        assert row[3].startswith("chat agent") and row[4] == "New" and row[5] == "-"
        t = app.views["models"].model_table
        assert [str(c.label) for c in t.columns.values()] == [
            "Model (alias)", "Size", "Quant", "Best for", "Preset", "Measured"]


async def test_preset_states_estimated_tuned_stale_and_measured(ui_env: SimpleNamespace) -> None:
    a = ui_env.add_model("a.gguf")
    b = ui_env.add_model("b.gguf")
    c = ui_env.add_model("c.gguf")
    add_preset(ui_env, a, "estimated", "chat")
    ui_env.cfg.set_preset(str(b), Preset("agent", Profile(), "tuned", tuned(42.5)))
    ui_env.cfg.set_preset(str(c), Preset("speed", Profile(), "tuned", tuned(30.0, build="old")))
    ui_env.cfg.set_preset(str(c), Preset("chat", Profile(), "edited"), activate=False)
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app, 3)
        assert cells(app, a)[4:] == ["chat estimated", "-"]
        assert cells(app, b)[4:] == ["agent tuned", "42.5 t/s"]
        assert cells(app, c)[4:] == ["speed tuned stale (+1)", "30.0 t/s"]


async def test_detail_has_one_row_per_intent_with_measured_speed(ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    ui_env.cfg.set_preset(str(m), Preset("agent", Profile(), "tuned", tuned(42.5)))
    ui_env.cfg.set_preset(str(m), Preset("chat", Profile(), "estimated"), activate=False)
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        mv = app.views["models"]
        it = mv.intent_table
        assert await wait_for(pilot, lambda: it.row_count == 5)
        rows = {str(k.value): [str(c) for c in it.get_row(k)] for k in it.rows}
        assert list(rows) == ["chat", "agent", "context", "speed", "api"]
        assert rows["agent"][1] == "GPU" and rows["agent"][5] == "42.5" and rows["agent"][6] == "tuned *"
        assert rows["chat"][5] == "-" and rows["chat"][6] == "estimated"
        assert rows["speed"][6] == "-"
        assert rows["api"][2].endswith("x4")
        assert " / " in rows["agent"][4]
        assert mv.selected_intent() == "agent"                       # the active preset's intent
        assert "Preset command (agent)" in plain(mv.query_one("#m_command", Static))
        assert "llama-server" in plain(mv.query_one("#m_command", Static))
        # moving to another intent shows its plan
        it.move_cursor(row=it.get_row_index("speed"))
        await pilot.pause(0.1)
        assert mv.selected_intent() == "speed"
        assert "Plan command (speed)" in plain(mv.query_one("#m_command", Static))


async def test_enter_without_presets_opens_workspace(ui_env: SimpleNamespace) -> None:
    """New models open controls without creating presets or loading a server."""
    model = ui_env.add_model()
    gateway = FakeGateway(ui_env.cfg)
    app = make_app(ui_env, gateway=gateway)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        app.views["models"].model_table.focus()
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: isinstance(app.screen, ModelWorkspace))
        assert ui_env.cfg.preset(str(model), "chat") is None
        assert not gateway.called("load") and not gateway.called("start")


async def test_picking_another_intent_serves_alias_at_intent(ui_env: SimpleNamespace) -> None:
    model = ui_env.add_model()
    add_preset(ui_env, model, "edited", "chat")
    add_preset(ui_env, model, "estimated", "speed", active=False)
    gateway = FakeGateway(ui_env.cfg)
    app = make_app(ui_env, gateway=gateway)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        app.views["models"].model_table.focus()
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: isinstance(app.screen, ModelWorkspace))
        app.screen.query_one("#workspace_purpose", Select).value = "speed"
        await pilot.pause()
        app.screen.query_one("#workspace_serve", Button).press()
        assert await wait_for(pilot, lambda: ("load", "alpha@speed") in gateway.calls)
        assert ui_env.cfg.models[str(model)].active == "chat"


async def test_enter_with_one_preset_opens_workspace(ui_env: SimpleNamespace) -> None:
    model = ui_env.add_model()
    add_preset(ui_env, model, "edited", "chat")
    gateway = FakeGateway(ui_env.cfg, running=True)
    app = make_app(ui_env, gateway=gateway)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        app.views["models"].model_table.focus()
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: isinstance(app.screen, ModelWorkspace))
        assert not gateway.called("load")


async def test_a_loaded_model_is_not_loaded_twice(ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    add_preset(ui_env, m, "edited", "chat")
    gw = FakeGateway(ui_env.cfg, running=True)
    gw.set_state("alpha", LOADED)
    app = make_app(ui_env, gateway=gw)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        app.views["models"].model_table.focus()
        await pilot.press("v")
        assert await wait_for(pilot, lambda: "is loaded" in plain(app._status))
        assert gw.called("load") == []


async def test_standalone_serve_launches_a_session(ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    add_preset(ui_env, m, "edited", "chat")
    store = FakeStore()
    app = make_app(ui_env, store=store, health=lambda url: "ready")
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        app.views["models"].model_table.focus()
        await pilot.press("s")
        assert await wait_for(pilot, lambda: store.calls and store.calls[0][:3] == (
            "launch", str(m), "chat"))
        assert await wait_for(pilot, lambda: "is ready at" in plain(app._status))


async def test_serve_failure_shows_a_message(ui_env: SimpleNamespace) -> None:
    from llamactl.gateway import GatewayError

    m = ui_env.add_model()
    add_preset(ui_env, m, "edited", "chat")
    gw = FakeGateway(ui_env.cfg)
    gw.fail("start", GatewayError("the gateway exited at once: bad option"))
    app = make_app(ui_env, gateway=gw)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        app.views["models"].model_table.focus()
        await pilot.press("v")
        assert await wait_for(pilot, lambda: "bad option" in plain(app._status))
        assert app.is_running


async def test_missing_binary_blocks_serving_with_a_message(ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    add_preset(ui_env, m, "edited", "chat")
    ui_env.cfg.bin_dir = str(ui_env.bin_dir / "nowhere")
    gw = FakeGateway(ui_env.cfg)
    app = make_app(ui_env, gateway=gw)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        app.views["models"].model_table.focus()
        await pilot.press("v")
        assert await wait_for(pilot, lambda: "llama-server not found" in plain(app._status))
        assert gw.calls.count(("start",)) == 0


async def test_plan_screen_save_makes_a_preset_and_reloads_the_gateway(
        ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    gw = FakeGateway(ui_env.cfg, running=True)
    app = make_app(ui_env, gateway=gw)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        app.views["models"].model_table.focus()
        await pilot.press("p")
        assert await wait_for(pilot, lambda: isinstance(app.screen, PlanScreen))
        await pilot.press("s")
        assert await wait_for(pilot, lambda: ui_env.cfg.preset(str(m)) is not None)
        assert await wait_for(pilot, lambda: gw.called("reload"))
        assert await wait_for(pilot, lambda: cells(app, m)[4] == "chat estimated")
        assert Config.load().preset(str(m), "chat").source == "estimated"


async def test_edit_reviews_saved_preset_and_invalidates_old_measurement(
        ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    ui_env.cfg.set_preset(str(m), Preset("chat", Profile(ctx_size=4096), "tuned", tuned()))
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        app.views["models"].model_table.focus()
        await pilot.press("e")
        assert await wait_for(pilot, lambda: isinstance(app.screen, PresetEditor))
        ed = app.screen
        f = ed.fields["ctx_size"]
        f.input.value = "2048"
        assert await wait_for(pilot, lambda: ed._draft.ctx_size == 2048)
        await pilot.press("ctrl+s")
        assert await wait_for(pilot, lambda: isinstance(app.screen, ProfileReview))
        assert ui_env.cfg.preset(str(m)).source == "tuned"
        app.screen.query_one("#review_save", Button).press()
        assert await wait_for(pilot, lambda: ui_env.cfg.preset(str(m)).source == "edited")
        p = ui_env.cfg.preset(str(m), "chat")
        assert p.profile.ctx_size == 2048 and p.measured is None
        assert await wait_for(pilot, lambda: cells(app, m)[4] == "chat edited")
        assert cells(app, m)[5] == "-"                          # not tuned any more


async def test_edit_without_changes_leaves_the_preset_alone(ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    ui_env.cfg.set_preset(str(m), Preset("chat", Profile(ctx_size=4096), "tuned", tuned()))
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        app.views["models"].model_table.focus()
        await pilot.press("e")
        assert await wait_for(pilot, lambda: isinstance(app.screen, PresetEditor))
        await pilot.press("ctrl+s")
        assert await wait_for(pilot, lambda: not isinstance(app.screen, PresetEditor))
        assert ui_env.cfg.preset(str(m)).source == "tuned"


async def test_edit_of_a_missing_preset_starts_from_the_plan(ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        app.views["models"].model_table.focus()
        await pilot.press("e")
        assert await wait_for(pilot, lambda: isinstance(app.screen, PresetEditor))
        assert await wait_for(pilot, lambda: app.screen._draft.ctx_size > 0)   # the plan was applied
        await pilot.press("ctrl+s")
        assert await wait_for(pilot, lambda: isinstance(app.screen, ProfileReview))
        assert ui_env.cfg.preset(str(m), "chat") is None
        app.screen.query_one("#review_save", Button).press()
        assert await wait_for(pilot, lambda: ui_env.cfg.preset(str(m), "chat") is not None)
        assert ui_env.cfg.preset(str(m), "chat").source == "edited"


async def test_set_active_and_delete_preset(ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    add_preset(ui_env, m, "edited", "chat")
    add_preset(ui_env, m, "estimated", "speed", active=False)
    gw = FakeGateway(ui_env.cfg, running=True)
    app = make_app(ui_env, gateway=gw)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        mv = app.views["models"]
        mv.model_table.focus()
        mv.intent_table.move_cursor(row=mv.intent_table.get_row_index("speed"))
        assert await wait_for(pilot, lambda: mv.selected_intent() == "speed")
        await pilot.press("a")
        assert await wait_for(pilot, lambda: ui_env.cfg.models[str(m)].active == "speed")
        assert await wait_for(pilot, lambda: gw.called("reload"))
        assert await wait_for(pilot, lambda: cells(app, m)[4].startswith("speed estimated"))
        await pilot.press("delete")
        assert await wait_for(pilot, lambda: isinstance(app.screen, ConfirmScreen))
        await pilot.press("n")
        await pilot.pause(0.1)
        assert "speed" in ui_env.cfg.models[str(m)].presets
        await pilot.press("delete")
        assert await wait_for(pilot, lambda: isinstance(app.screen, ConfirmScreen))
        await pilot.press("y")
        assert await wait_for(pilot, lambda: "speed" not in ui_env.cfg.models[str(m)].presets)
        assert ui_env.cfg.models[str(m)].active == "chat"
        assert await wait_for(pilot, lambda: cells(app, m)[4] == "chat edited")


async def test_actions_on_an_intent_without_a_preset_explain(ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    add_preset(ui_env, m, "edited", "chat")
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        mv = app.views["models"]
        mv.model_table.focus()
        mv.intent_table.move_cursor(row=mv.intent_table.get_row_index("api"))
        assert await wait_for(pilot, lambda: mv.selected_intent() == "api")
        await pilot.press("a")
        assert "No 'api' preset" in plain(app._status)
        await pilot.press("delete")
        assert "No 'api' preset" in plain(app._status)


async def test_default_profile_editor_replans(ui_env: SimpleNamespace) -> None:
    ui_env.add_model()
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        app.views["models"].model_table.focus()
        await pilot.press("D")
        assert await wait_for(pilot, lambda: isinstance(app.screen, PresetEditor))
        app.screen.fields["threads"].input.value = "3"
        assert await wait_for(pilot, lambda: app.screen._draft.threads == 3)
        await pilot.press("ctrl+s")
        assert await wait_for(pilot, lambda: ui_env.cfg.default_profile.threads == 3)
        assert Config.load().default_profile.threads == 3


async def test_empty_and_unreadable_states(ui_env: SimpleNamespace) -> None:
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        mv = app.views["models"]
        assert await wait_for(pilot, lambda: not app.scanning)
        await pilot.press("2")
        assert "No models found" in plain(mv.query_one("#m_empty", Static))
        await pilot.press("t")
        assert "No model selected" in plain(app._status)
    bad = ui_env.model_dir / "broken.gguf"
    bad.write_bytes(b"not a gguf")
    with bad.open("r+b") as fh:
        fh.truncate(60 * 1024 * 1024)
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        assert await wait_for(pilot, lambda: not app.scanning and app.views["models"].model_table.row_count == 1)
        assert cells(app, bad)[3] == "unreadable"


async def test_rescan_picks_up_new_files(ui_env: SimpleNamespace) -> None:
    ui_env.add_model("one.gguf")
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        ui_env.add_model("two.gguf")
        app.screen.set_focus(None)
        await pilot.press("r")
        assert await wait_for(pilot, lambda: app.views["models"].model_table.row_count == 2)
