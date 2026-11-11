"""Pilot tests: the shell, the Home view and the Serving view (fake gateway, sessions and runner)."""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from textual.widgets import DataTable, Input, OptionList, Static

from fakes import FakeGateway, FakeRunner, FakeStore, plain, runner_factory, wait_for
from llamactl.app import LlamaCtl, ServingChanged
from llamactl.config import Config, Preset, Profile
from llamactl.gateway import FAILED, LOADED, UNLOADED, GatewayError
from llamactl.screens.confirm import ConfirmScreen
from llamactl.screens.log import LogScreen
from llamactl.screens.prompt import PromptScreen
from llamactl.serving import gather
from llamactl.sessions import Session

SIZE = (140, 45)
VIEW_KEYS = {"1": "home", "2": "models", "3": "serving", "4": "discover", "5": "settings"}


def make_app(env: SimpleNamespace, *, running: bool = False, store: FakeStore | None = None,
             gateway: FakeGateway | None = None, runner: FakeRunner | None = None,
             **kw) -> LlamaCtl:
    """An app over fakes (including the health probe: no test reaches for the network); the
    config is the environment's."""
    gw = gateway or FakeGateway(env.cfg, running=running)
    kw.setdefault("health", lambda url: "ready")
    return LlamaCtl(cfg=env.cfg, gateway=gw, store=store or FakeStore(),
                    runner_factory=runner_factory(runner or FakeRunner()),
                    build=lambda: "b1", **kw)


def current(app: LlamaCtl) -> str:
    """Id of the visible view."""
    return app.current_view


def add_preset(env: SimpleNamespace, path: Path, source: str = "edited", name: str = "chat",
               active: bool = True) -> None:
    """Give a model a preset (and save the config so the app sees it)."""
    env.cfg.set_preset(str(path), Preset(name, Profile(), source), activate=active)


# ---------------------------------------------------------------------------- shell

async def test_starts_on_home_and_every_view_is_reachable_by_key(ui_env: SimpleNamespace) -> None:
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        assert await wait_for(pilot, lambda: app.snapshot is not None)
        assert current(app) == "home"
        for key, name in VIEW_KEYS.items():
            app.screen.set_focus(None)         # Discover's search box would swallow the digit
            await pilot.press(key)
            await pilot.pause(0.05)
            assert current(app) == name
        app.screen.set_focus(None)
        await pilot.press("1")
        assert current(app) == "home"


async def test_status_bar_shows_messages_and_gateway_state(ui_env: SimpleNamespace) -> None:
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        assert await wait_for(pilot, lambda: app.snapshot is not None)
        assert "gateway stopped" in plain(app._gwstate)
        app.say("literal [brackets] stay literal")
        await pilot.pause()
        assert plain(app._status) == "literal [brackets] stay literal"
        # a modal on top must not break status updates
        app.push_screen(ConfirmScreen("x?"))
        await pilot.pause()
        app.say("still works")
        assert plain(app._status) == "still works"


async def test_one_serving_message_feeds_home_and_serving(ui_env: SimpleNamespace) -> None:
    app = make_app(ui_env, running=True)
    seen: list[str] = []
    async with app.run_test(size=SIZE) as pilot:
        for name in ("home", "serving"):
            orig = app.views[name].on_serving
            app.views[name].on_serving = lambda snap, n=name, o=orig: (seen.append(n), o(snap))
        app.post_message(ServingChanged(gather(app.gateway, app.store)))
        await pilot.pause(0.1)
        assert "home" in seen and "serving" in seen


async def test_idle_poll_does_not_hit_the_gateway(ui_env: SimpleNamespace) -> None:
    """Nothing running: the poll gathers a snapshot without asking the router anything."""
    gw = FakeGateway(ui_env.cfg)
    app = make_app(ui_env, gateway=gw)
    async with app.run_test(size=SIZE) as pilot:
        assert await wait_for(pilot, lambda: app.snapshot is not None)
        app.refresh_serving()
        await pilot.pause(0.2)
        assert gw.called("models") == []


# ---------------------------------------------------------------------------- home

async def test_home_empty_states(ui_env: SimpleNamespace) -> None:
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        home = app.views["home"]
        assert await wait_for(pilot, lambda: app.snapshot is not None and not app.scanning)
        assert "stopped" in plain(home.query_one("#gw_info", Static))
        assert "Nothing is loaded" in plain(home.query_one("#loaded_empty", Static))
        steps = home.query_one("#steps", OptionList)
        assert steps.option_count == 1                       # no models found
        assert "No models found" in str(steps.get_option_at_index(0).prompt)
        assert "CUDA0" in plain(home.query_one("#hw_card", Static))


async def test_enter_on_no_models_suggestion_opens_discover(ui_env: SimpleNamespace) -> None:
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        home = app.views["home"]
        assert await wait_for(pilot, lambda: home.query_one("#steps", OptionList).option_count == 1
                              and not app.scanning)
        home.query_one("#steps", OptionList).focus()
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: current(app) == "discover")


async def test_enter_on_no_preset_suggestion_selects_the_model(ui_env: SimpleNamespace) -> None:
    ui_env.add_model("alpha.gguf")
    beta = ui_env.add_model("beta.gguf")
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        home = app.views["home"]
        assert await wait_for(pilot, lambda: len(app.infos) == 2 and not app.scanning)
        assert await wait_for(pilot, lambda: home.query_one("#steps", OptionList).option_count == 2)
        home.query_one("#steps", OptionList).focus()
        await pilot.press("down", "enter")                    # second suggestion: beta
        assert await wait_for(pilot, lambda: current(app) == "models")
        assert app.views["models"].selected_model() == str(beta)


async def test_enter_on_start_gateway_suggestion_starts_it(ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    add_preset(ui_env, m)
    gw = FakeGateway(ui_env.cfg)
    app = make_app(ui_env, gateway=gw)
    async with app.run_test(size=SIZE) as pilot:
        home = app.views["home"]
        assert await wait_for(pilot, lambda: home.query_one("#steps", OptionList).option_count == 1
                              and not app.scanning)
        assert "Start the gateway" in str(home.query_one("#steps", OptionList)
                                          .get_option_at_index(0).prompt)
        home.query_one("#steps", OptionList).focus()
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: gw.called("start"))
        assert await wait_for(pilot, lambda: "running" in plain(home.query_one("#gw_info", Static)))


async def test_enter_on_missing_binary_suggestion_opens_settings(ui_env: SimpleNamespace) -> None:
    ui_env.cfg.bin_dir = str(ui_env.bin_dir / "nowhere")
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        home = app.views["home"]
        assert await wait_for(pilot, lambda: home.query_one("#steps", OptionList).option_count >= 1)
        assert "llama-server not found" in str(home.query_one("#steps", OptionList)
                                               .get_option_at_index(0).prompt)
        home.query_one("#steps", OptionList).focus()
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: current(app) == "settings")


async def test_gateway_start_and_stop_from_home_update_the_card(ui_env: SimpleNamespace) -> None:
    gw = FakeGateway(ui_env.cfg)
    app = make_app(ui_env, gateway=gw)
    async with app.run_test(size=SIZE) as pilot:
        home = app.views["home"]
        assert await wait_for(pilot, lambda: app.snapshot is not None)
        await pilot.press("u")
        assert await wait_for(pilot, lambda: "running" in plain(home.query_one("#gw_info", Static)))
        assert gw.running and "gateway running" in plain(app._gwstate)
        await pilot.press("d")
        assert await wait_for(pilot, lambda: isinstance(app.screen, ConfirmScreen))
        await pilot.press("y")
        assert await wait_for(pilot, lambda: not gw.running)
        assert await wait_for(pilot, lambda: "stopped" in plain(home.query_one("#gw_info", Static)))


async def test_stop_from_home_can_be_declined(ui_env: SimpleNamespace) -> None:
    gw = FakeGateway(ui_env.cfg, running=True)
    app = make_app(ui_env, gateway=gw)
    async with app.run_test(size=SIZE) as pilot:
        assert await wait_for(pilot, lambda: app.snapshot is not None and app.snapshot.status.running)
        await pilot.press("d")
        assert await wait_for(pilot, lambda: isinstance(app.screen, ConfirmScreen))
        await pilot.press("n")
        await pilot.pause(0.1)
        assert gw.running and not gw.called("stop")


async def test_gateway_failure_is_a_message_not_a_traceback(ui_env: SimpleNamespace) -> None:
    gw = FakeGateway(ui_env.cfg)
    gw.fail("start", GatewayError("port 8080 on 127.0.0.1 is already in use"))
    app = make_app(ui_env, gateway=gw)
    async with app.run_test(size=SIZE) as pilot:
        assert await wait_for(pilot, lambda: app.snapshot is not None)
        await pilot.press("u")
        assert await wait_for(pilot, lambda: "already in use" in plain(app._status))
        assert not gw.running and app.gateway_state() == "stopped"
        assert app.is_running


async def test_home_lists_what_is_loaded_and_builds_a_snippet(ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    add_preset(ui_env, m, "edited", "chat")
    ext = Session("ext-x", "other", "http://127.0.0.1:9999", external=True, started=time.time())
    gw = FakeGateway(ui_env.cfg, running=True, load_state=LOADED)
    gw.set_state("alpha", LOADED)
    app = make_app(ui_env, gateway=gw, store=FakeStore([ext]))
    async with app.run_test(size=SIZE) as pilot:
        home = app.views["home"]
        loaded = home.query_one("#loaded", DataTable)
        assert await wait_for(pilot, lambda: loaded.row_count == 2)
        assert "alpha" in plain(home.query_one("#gw_snippet", Static))
        assert "/v1/chat/completions" in plain(home.query_one("#gw_snippet", Static))
        assert "attached" in [str(c) for c in loaded.get_row_at(1)]
        assert await wait_for(pilot, lambda: app.focused is not None)       # keys need a focus
        await pilot.press("y")
        assert app.clipboard == f"{gw.listen_url}/v1"
        await pilot.press("c")
        assert app.clipboard.startswith("curl ") and '"model": "alpha"' in app.clipboard


# ---------------------------------------------------------------------------- serving

def _serving_env(env: SimpleNamespace) -> tuple[FakeGateway, FakeStore, Session]:
    m = env.add_model()
    add_preset(env, m, "estimated", "chat")
    add_preset(env, m, "estimated", "agent", active=False)
    gw = FakeGateway(env.cfg, running=True)
    gw.set_state("alpha", LOADED)
    gw.set_state("alpha@agent", FAILED, 1)
    sess = Session("beta-9001", "beta", "http://127.0.0.1:9001", pid=0, log="/nonexistent",
                   started=time.time() - 90, external=False)
    return gw, FakeStore([sess]), sess


async def test_serving_view_shows_gateway_models_and_sessions(ui_env: SimpleNamespace) -> None:
    gw, store, sess = _serving_env(ui_env)
    app = make_app(ui_env, gateway=gw, store=store, health=lambda url: "ready")
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("3")
        t = app.views["serving"].table
        assert await wait_for(pilot, lambda: t.row_count == 3)
        rows = {str(r.value): [str(c) for c in t.get_row(r)] for r in t.rows}
        assert rows["gw:alpha"][:3] == ["alpha", "loaded", "gateway"]
        assert rows["gw:alpha@agent"][1] == "failed (exit 1)"
        assert rows["s:beta-9001"][:3] == ["beta", "crashed", "standalone"]   # pid 0 is not alive
        assert rows["s:beta-9001"][3:6] == ["-", "-", "-"]                   # no usage sampled
        assert rows["s:beta-9001"][7] == "1m"


async def test_serving_state_changes_update_rows_in_place(ui_env: SimpleNamespace) -> None:
    gw, store, _ = _serving_env(ui_env)
    app = make_app(ui_env, gateway=gw, store=store)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("3")
        sv = app.views["serving"]
        assert await wait_for(pilot, lambda: sv.table.row_count == 3)
        key_before = {r.value: r for r in sv.table.rows}["gw:alpha"]
        gw.set_state("alpha", UNLOADED)
        app.refresh_serving()
        assert await wait_for(pilot, lambda: str(sv.table.get_cell("gw:alpha", "state")) == "unloaded")
        assert sv.table.row_count == 3
        assert {r.value: r for r in sv.table.rows}["gw:alpha"] is key_before


async def test_serving_load_and_unload_call_the_gateway(ui_env: SimpleNamespace) -> None:
    gw, store, _ = _serving_env(ui_env)
    app = make_app(ui_env, gateway=gw, store=store)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("3")
        sv = app.views["serving"]
        assert await wait_for(pilot, lambda: sv.table.row_count == 3)
        sv.table.move_cursor(row=sv.table.get_row_index("gw:alpha@agent"))
        await pilot.press("l")
        assert await wait_for(pilot, lambda: ("load", "alpha@agent") in gw.calls)
        assert await wait_for(pilot, lambda: str(sv.table.get_cell("gw:alpha@agent", "state")) == "loaded")
        sv.table.move_cursor(row=sv.table.get_row_index("gw:alpha"))
        await pilot.press("x")
        assert await wait_for(pilot, lambda: ("unload", "alpha") in gw.calls)
        assert await wait_for(pilot, lambda: str(sv.table.get_cell("gw:alpha", "state")) == "unloaded")
        await pilot.press("l")
        assert await wait_for(pilot, lambda: ("load", "alpha") in gw.calls)


async def test_serving_stop_standalone_asks_first(ui_env: SimpleNamespace) -> None:
    gw, store, sess = _serving_env(ui_env)
    app = make_app(ui_env, gateway=gw, store=store)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("3")
        sv = app.views["serving"]
        assert await wait_for(pilot, lambda: sv.table.row_count == 3)
        sv.table.move_cursor(row=sv.table.get_row_index("s:beta-9001"))
        await pilot.press("s")
        assert await wait_for(pilot, lambda: isinstance(app.screen, ConfirmScreen))
        await pilot.press("n")
        await pilot.pause(0.1)
        assert store.calls == []
        await pilot.press("s")
        assert await wait_for(pilot, lambda: isinstance(app.screen, ConfirmScreen))
        await pilot.press("y")
        assert await wait_for(pilot, lambda: ("stop", "beta-9001") in store.calls)
        assert await wait_for(pilot, lambda: sv.table.row_count == 2)


async def test_serving_stop_on_a_gateway_model_explains(ui_env: SimpleNamespace) -> None:
    gw, store, _ = _serving_env(ui_env)
    app = make_app(ui_env, gateway=gw, store=store)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("3")
        sv = app.views["serving"]
        assert await wait_for(pilot, lambda: sv.table.row_count == 3)
        sv.table.move_cursor(row=sv.table.get_row_index("gw:alpha"))
        await pilot.press("s")
        await pilot.pause(0.1)
        assert "x unloads" in plain(app._status) and store.calls == []


async def test_serving_attach_url(ui_env: SimpleNamespace) -> None:
    gw = FakeGateway(ui_env.cfg)
    store = FakeStore()
    app = make_app(ui_env, gateway=gw, store=store)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("3")
        await pilot.press("a")
        assert await wait_for(pilot, lambda: isinstance(app.screen, PromptScreen))
        app.screen.query_one("#value", Input).value = "10.0.0.5:8081"
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: ("attach", "10.0.0.5:8081") in store.calls)
        sv = app.views["serving"]
        assert await wait_for(pilot, lambda: sv.table.row_count == 1)
        assert str(sv.table.get_cell("s:ext-10.0.0.5:8081", "kind")) == "attached"


async def test_serving_gateway_up_down_and_copy(ui_env: SimpleNamespace) -> None:
    gw, store, _ = _serving_env(ui_env)
    gw._running = False
    gw.states = {}
    app = make_app(ui_env, gateway=gw, store=store)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("3")
        sv = app.views["serving"]
        assert await wait_for(pilot, lambda: app.snapshot is not None)
        assert "stopped" in plain(sv.query_one("#sv_gateway", Static))
        await pilot.press("u")
        assert await wait_for(pilot, lambda: gw.running)
        assert await wait_for(pilot, lambda: sv.table.row_count == 3)     # alpha, alpha@agent, beta
        sv.table.move_cursor(row=sv.table.get_row_index("gw:alpha"))
        await pilot.press("y")
        assert app.clipboard == f"{gw.listen_url}/v1"
        await pilot.press("c")
        assert '"model": "alpha"' in app.clipboard
        sv.table.move_cursor(row=sv.table.get_row_index("s:beta-9001"))
        await pilot.press("c")
        assert "model" not in app.clipboard.split("-d ")[1]
        await pilot.press("d")
        assert await wait_for(pilot, lambda: isinstance(app.screen, ConfirmScreen))
        await pilot.press("y")
        assert await wait_for(pilot, lambda: not gw.running)


async def test_serving_log_pane_follows_the_selection_and_opens_full_log(
        ui_env: SimpleNamespace, tmp_path: Path) -> None:
    glog = tmp_path / "gateway.log"
    glog.write_text("router line 1\nrouter line 2\n")
    slog = tmp_path / "s.log"
    slog.write_text("session line\n")
    gw, store, sess = _serving_env(ui_env)
    store.sessions[0].log = str(slog)
    gw.log = str(glog)
    app = make_app(ui_env, gateway=gw, store=store)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("3")
        sv = app.views["serving"]
        assert await wait_for(pilot, lambda: sv.table.row_count == 3)
        sv.table.move_cursor(row=sv.table.get_row_index("gw:alpha"))
        await pilot.pause(0.1)
        assert "router line 2" in sv.detail_text()
        sv.table.move_cursor(row=sv.table.get_row_index("s:beta-9001"))
        await pilot.pause(0.1)
        assert sv.detail_text() == "session line"
        await pilot.press("o")
        assert await wait_for(pilot, lambda: isinstance(app.screen, LogScreen))


async def test_serving_empty_state_is_a_helpful_line(ui_env: SimpleNamespace) -> None:
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("3")
        sv = app.views["serving"]
        assert await wait_for(pilot, lambda: "gateway is stopped" in plain(sv.query_one("#sv_empty", Static)))


async def test_a_failing_load_is_reported(ui_env: SimpleNamespace) -> None:
    gw, store, _ = _serving_env(ui_env)
    gw.fail("load", GatewayError("cannot load 'alpha@agent': no memory"))
    app = make_app(ui_env, gateway=gw, store=store)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("3")
        sv = app.views["serving"]
        assert await wait_for(pilot, lambda: sv.table.row_count == 3)
        sv.table.move_cursor(row=sv.table.get_row_index("gw:alpha@agent"))
        await pilot.press("l")
        assert await wait_for(pilot, lambda: "no memory" in plain(app._status))
        assert app.is_running


async def test_attached_server_shows_its_props(ui_env: SimpleNamespace, monkeypatch) -> None:
    from llamactl.views import serving as serving_view

    monkeypatch.setattr(serving_view, "probe", lambda url: {"model_path": "/x/m.gguf", "n_ctx": 4096})
    ext = Session("ext-box-8081", "m", "http://10.0.0.5:8081", external=True, started=time.time())
    app = make_app(ui_env, store=FakeStore([ext]), health=lambda url: "ready")
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("3")
        sv = app.views["serving"]
        assert await wait_for(pilot, lambda: sv.table.row_count == 1)
        assert await wait_for(pilot, lambda: "n_ctx" in sv.detail_text())
        assert "http://10.0.0.5:8081" in sv.detail_text()
        await pilot.press("o")
        await pilot.pause(0.1)
        assert "attached server has no log" in plain(app._status)


async def test_unwritable_config_is_a_message(ui_env: SimpleNamespace, monkeypatch) -> None:
    m = ui_env.add_model()
    app = make_app(ui_env, running=True)
    async with app.run_test(size=SIZE) as pilot:
        assert await wait_for(pilot, lambda: len(app.infos) == 1)

        def boom(self) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(Config, "save", boom)
        ui_env.cfg.set_preset(str(m), Preset("chat", Profile(), "edited"))
        app.presets_changed(str(m))
        await pilot.pause(0.1)
        assert "cannot save the config: disk full" in plain(app._status)
        assert app.is_running


async def test_ini_rendering_error_while_reloading_is_a_message(ui_env: SimpleNamespace) -> None:
    gw = FakeGateway(ui_env.cfg, running=True)
    gw.fail("reload", ValueError("extra_args: a bare token cannot be written"))
    app = make_app(ui_env, gateway=gw)
    async with app.run_test(size=SIZE) as pilot:
        assert await wait_for(pilot, lambda: app.snapshot is not None)
        app.presets_changed()
        assert await wait_for(pilot, lambda: "bare token" in plain(app._status))
