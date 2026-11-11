"""Pilot tests: the Settings view, gateway settings, autostart."""
from __future__ import annotations

from types import SimpleNamespace

from textual.widgets import Input, ListView, Select, Static, Switch

from fakes import FakeGateway, plain, wait_for
from llamactl import hf
from llamactl.config import Config
from llamactl.gateway import GatewayError
from test_ui_shell import SIZE, add_preset, make_app


async def open_settings(pilot, app):
    """Show the Settings view (focus is cleared first: Discover's box would eat the digit)."""
    app.screen.set_focus(None)
    await pilot.press("5")
    await pilot.pause(0.1)
    return app.views["settings"]


def setv(view, wid: str, value: str) -> None:
    """Type a value into an input."""
    view.query_one(f"#{wid}", Input).value = value


async def test_save_writes_gateway_settings_default_intent_and_reloads(
        ui_env: SimpleNamespace, monkeypatch) -> None:
    monkeypatch.setattr(hf, "TOKEN_FILE", ui_env.cfg_dir / "hf_token")
    m = ui_env.add_model()
    add_preset(ui_env, m, "edited", "chat")
    gw = FakeGateway(ui_env.cfg, running=True)
    app = make_app(ui_env, gateway=gw)
    async with app.run_test(size=SIZE) as pilot:
        sv = await open_settings(pilot, app)
        setv(sv, "gw_port", "9090")
        setv(sv, "gw_host", "0.0.0.0")
        setv(sv, "gw_api_key", "s3cret")
        setv(sv, "gw_models_max", "2")
        setv(sv, "gw_sleep_idle", "0")
        setv(sv, "headroom_pct", "15")
        sv.query_one("#gw_autostart", Switch).value = True
        sv.query_one("#default_intent", Select).value = "agent"
        assert sv.query_one("#gw_api_key", Input).password
        assert await wait_for(pilot, sv._fields_valid)
        await pilot.press("ctrl+s")
        assert await wait_for(pilot, lambda: Config.load().gateway.port == 9090)
        saved = Config.load()
        g = saved.gateway
        assert (g.host, g.api_key, g.models_max, g.sleep_idle, g.autostart) == (
            "0.0.0.0", "s3cret", 2, 0, True)
        assert saved.default_intent == "agent" and saved.headroom_pct == 15
        assert await wait_for(pilot, lambda: gw.called("reload"))        # presets rewritten + reloaded
        msg = plain(sv.query_one("#set_msg", Static))
        assert "restarted" in msg and "host" in msg and "port" in msg and "API key" in msg
        assert "models-max" in msg


async def test_changes_that_need_no_restart_say_plain_saved(ui_env: SimpleNamespace) -> None:
    gw = FakeGateway(ui_env.cfg, running=True)
    app = make_app(ui_env, gateway=gw)
    async with app.run_test(size=SIZE) as pilot:
        sv = await open_settings(pilot, app)
        setv(sv, "gw_sleep_idle", "120")
        assert await wait_for(pilot, sv._fields_valid)
        await pilot.press("ctrl+s")
        assert await wait_for(pilot, lambda: plain(sv.query_one("#set_msg", Static)) == "Settings saved.")
        assert ui_env.cfg.gateway.sleep_idle == 120
        assert await wait_for(pilot, lambda: gw.called("reload"))


async def test_stopped_gateway_needs_no_restart_message(ui_env: SimpleNamespace) -> None:
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        sv = await open_settings(pilot, app)
        setv(sv, "gw_port", "9191")
        assert await wait_for(pilot, sv._fields_valid)
        await pilot.press("ctrl+s")
        assert await wait_for(pilot, lambda: plain(sv.query_one("#set_msg", Static)) == "Settings saved.")


async def test_invalid_values_block_the_save(ui_env: SimpleNamespace) -> None:
    gw = FakeGateway(ui_env.cfg, running=True)
    app = make_app(ui_env, gateway=gw)
    async with app.run_test(size=SIZE) as pilot:
        sv = await open_settings(pilot, app)
        for wid, bad in (("gw_port", "99999"), ("gw_models_max", "0"), ("gw_sleep_idle", "-3"),
                         ("gw_host", "  "), ("headroom_pct", "abc")):
            old = sv.query_one(f"#{wid}", Input).value
            setv(sv, wid, bad)
            assert await wait_for(pilot, lambda: not sv.query_one(f"#{wid}", Input).is_valid), wid
            await pilot.press("ctrl+s")
            assert await wait_for(pilot, lambda: "Fix the invalid fields" in plain(
                sv.query_one("#set_msg", Static)))
            setv(sv, wid, old)
            assert await wait_for(pilot, sv._fields_valid)
        assert ui_env.cfg.gateway.port == 8080 and not (ui_env.cfg_dir / "config.json").exists()


async def test_model_dirs_are_added_removed_and_rescanned(ui_env: SimpleNamespace, tmp_path) -> None:
    extra = tmp_path / "extra"
    extra.mkdir()
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        sv = await open_settings(pilot, app)
        add = sv.query_one("#add_dir", Input)
        add.value = str(extra)
        add.focus()
        assert await wait_for(pilot, lambda: add.is_valid)
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: len(sv.query_one("#dir_list", ListView).children) == 2)
        await pilot.press("ctrl+s")
        assert await wait_for(pilot, lambda: str(extra) in Config.load().model_dirs)
        assert app.cfg.model_dirs == [str(ui_env.model_dir), str(extra)]
        ui_env.add_model("late.gguf")
        # remove the first dir again: the model there disappears from the list after the rescan
        sv.query_one("#dir_list", ListView).index = 0
        sv.query_one("#dir_list", ListView).focus()
        await pilot.press("delete")
        assert await wait_for(pilot, lambda: len(sv.query_one("#dir_list", ListView).children) == 1)
        await pilot.press("ctrl+s")
        assert await wait_for(pilot, lambda: Config.load().model_dirs == [str(extra)])
        assert await wait_for(pilot, lambda: not app.scanning and app.models == [])


async def test_edit_default_profile_button_opens_the_editor(ui_env: SimpleNamespace) -> None:
    from llamactl.screens.preset_editor import PresetEditor
    from textual.widgets import Button

    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        sv = await open_settings(pilot, app)
        sv.query_one("#edit_default", Button).press()
        assert await wait_for(pilot, lambda: isinstance(app.screen, PresetEditor))


async def test_autostart_starts_the_gateway_on_mount(ui_env: SimpleNamespace) -> None:
    ui_env.cfg.gateway.autostart = True
    gw = FakeGateway(ui_env.cfg)
    app = make_app(ui_env, gateway=gw)
    async with app.run_test(size=SIZE) as pilot:
        assert await wait_for(pilot, lambda: gw.called("start"))
        assert await wait_for(pilot, lambda: gw.running and "running" in plain(app._gwstate))


async def test_autostart_without_llama_server_says_so(ui_env: SimpleNamespace) -> None:
    ui_env.cfg.gateway.autostart = True
    ui_env.cfg.bin_dir = str(ui_env.bin_dir / "nowhere")
    gw = FakeGateway(ui_env.cfg)
    app = make_app(ui_env, gateway=gw)
    async with app.run_test(size=SIZE) as pilot:
        assert await wait_for(pilot, lambda: "autostart: llama-server not found" in app._last_failure)
        await pilot.pause(0.05)
        assert gw.called("start") == []


async def test_a_failing_reload_after_save_is_a_message(ui_env: SimpleNamespace) -> None:
    gw = FakeGateway(ui_env.cfg, running=True)
    gw.fail("reload", GatewayError("the gateway at http://127.0.0.1:8080 is not reachable"))
    app = make_app(ui_env, gateway=gw)
    async with app.run_test(size=SIZE) as pilot:
        sv = await open_settings(pilot, app)
        setv(sv, "gw_sleep_idle", "30")
        assert await wait_for(pilot, sv._fields_valid)
        await pilot.press("ctrl+s")
        assert await wait_for(pilot, lambda: "not reachable" in plain(app._status))
        assert Config.load().gateway.sleep_idle == 30 and app.is_running
