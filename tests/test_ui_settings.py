"""Pilot tests: the Settings view, gateway settings, autostart."""
from __future__ import annotations

from types import SimpleNamespace

from textual.widgets import Button, Input, ListView, Select, Static, Switch

from conftest import FAKE_LAN
from fakes import FakeGateway, plain, wait_for
from llamactl import hf, network
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
        sv.query_one("#gw_listen", Select).value = "lan"
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
        sv.query_one("#gw_listen", Select).value = "custom"
        assert await wait_for(pilot, lambda: not sv.query_one("#gw_host", Input).disabled)
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


# ---------------------------------------------------------------------------- network access

def urls_line(sv) -> str:
    """The text of the line under the API key row."""
    return plain(sv.query_one("#gw_urls", Static))


async def test_listen_on_select_starts_from_the_configured_host(ui_env: SimpleNamespace) -> None:
    for host, mode, editable in (("127.0.0.1", "local", False), ("0.0.0.0", "lan", False),
                                 ("10.0.0.5", "custom", True), ("localhost", "custom", True)):
        ui_env.cfg.gateway.host = host
        app = make_app(ui_env)
        async with app.run_test(size=SIZE) as pilot:
            sv = await open_settings(pilot, app)
            assert sv.query_one("#gw_listen", Select).value == mode, host
            box = sv.query_one("#gw_host", Input)
            assert box.value == host and box.disabled is (not editable), host


async def test_custom_address_is_editable_only_for_custom(ui_env: SimpleNamespace) -> None:
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        sv = await open_settings(pilot, app)
        box, listen = sv.query_one("#gw_host", Input), sv.query_one("#gw_listen", Select)
        assert box.disabled and box.value == "127.0.0.1"
        listen.value = "lan"
        assert await wait_for(pilot, lambda: box.value == "0.0.0.0" and box.disabled)
        listen.value = "custom"
        assert await wait_for(pilot, lambda: not box.disabled)
        setv(sv, "gw_host", "192.168.50.7")
        assert await wait_for(pilot, sv._fields_valid)
        listen.value = "local"                       # leaving Custom discards the typed address
        assert await wait_for(pilot, lambda: box.value == "127.0.0.1" and box.disabled)


async def test_saving_each_listen_mode_writes_the_host(ui_env: SimpleNamespace) -> None:
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        sv = await open_settings(pilot, app)
        listen = sv.query_one("#gw_listen", Select)
        for mode, custom, want in (("lan", "", "0.0.0.0"), ("custom", "192.168.50.7", "192.168.50.7"),
                                   ("local", "", "127.0.0.1")):
            listen.value = mode
            if mode == "custom":
                assert await wait_for(pilot, lambda: not sv.query_one("#gw_host", Input).disabled)
                setv(sv, "gw_host", custom)
            assert await wait_for(pilot, sv._fields_valid)
            await pilot.press("ctrl+s")
            assert await wait_for(pilot, lambda: Config.load().gateway.host == want), mode
        assert app.cfg.gateway.host == "127.0.0.1"


async def test_url_line_follows_the_form(ui_env: SimpleNamespace) -> None:
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        sv = await open_settings(pilot, app)
        assert urls_line(sv).strip() == "Clients connect to  http://127.0.0.1:8080/v1"
        sv.query_one("#gw_listen", Select).value = "lan"
        assert await wait_for(pilot, lambda: f"http://{FAKE_LAN}:8080/v1" in urls_line(sv))
        assert "http://127.0.0.1:8080/v1" in urls_line(sv) and "0.0.0.0:" not in urls_line(sv)
        setv(sv, "gw_port", "9090")
        assert await wait_for(pilot, lambda: f"http://{FAKE_LAN}:9090/v1" in urls_line(sv))
        sv.query_one("#gw_listen", Select).value = "custom"
        assert await wait_for(pilot, lambda: not sv.query_one("#gw_host", Input).disabled)
        setv(sv, "gw_host", "10.9.8.7")
        assert await wait_for(pilot, lambda: "Clients connect to  http://10.9.8.7:9090/v1" in urls_line(sv))
        assert "127.0.0.1" not in urls_line(sv)
        setv(sv, "gw_port", "oops")                    # an unfinished port keeps the saved one
        assert await wait_for(pilot, lambda: "http://10.9.8.7:8080/v1" in urls_line(sv))


async def test_exposed_bind_without_a_key_warns_and_a_key_clears_it(ui_env: SimpleNamespace) -> None:
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        sv = await open_settings(pilot, app)
        assert "without an API key" not in urls_line(sv)
        sv.query_one("#gw_listen", Select).value = "lan"
        assert await wait_for(pilot, lambda: "without an API key" in urls_line(sv))
        setv(sv, "gw_api_key", "k")
        assert await wait_for(pilot, lambda: "without an API key" not in urls_line(sv))
        setv(sv, "gw_api_key", "")
        assert await wait_for(pilot, lambda: "without an API key" in urls_line(sv))
        sv.query_one("#gw_listen", Select).value = "local"
        assert await wait_for(pilot, lambda: "without an API key" not in urls_line(sv))


async def test_generate_button_fills_reveals_copies_and_clears_the_warning(
        ui_env: SimpleNamespace) -> None:
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        sv = await open_settings(pilot, app)
        sv.query_one("#gw_listen", Select).value = "lan"
        assert await wait_for(pilot, lambda: "without an API key" in urls_line(sv))
        box = sv.query_one("#gw_api_key", Input)
        assert box.password and box.value == ""
        sv.query_one("#gw_gen_key", Button).press()
        assert await wait_for(pilot, lambda: len(box.value) >= 43)
        assert not box.password and app.clipboard == box.value
        assert await wait_for(pilot, lambda: "without an API key" not in urls_line(sv))
        assert "Generated an API key" in plain(sv.query_one("#set_msg", Static))
        first = box.value
        sv.query_one("#gw_gen_key", Button).press()
        assert await wait_for(pilot, lambda: box.value != first)
        assert await wait_for(pilot, sv._fields_valid)
        await pilot.press("ctrl+s")
        assert await wait_for(pilot, lambda: Config.load().gateway.api_key == box.value)
        assert Config.load().gateway.host == "0.0.0.0"


async def test_url_line_shows_the_wsl_notes_for_an_exposed_bind(ui_env: SimpleNamespace,
                                                                monkeypatch) -> None:
    monkeypatch.setattr(network, "read_osrelease", lambda: "5.15-microsoft-standard-WSL2")
    monkeypatch.setattr(network, "read_wslinfo", lambda: "nat")
    network.reset_cache()
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        sv = await open_settings(pilot, app)
        assert "NAT" not in urls_line(sv)                       # private bind: nothing to explain
        sv.query_one("#gw_listen", Select).value = "lan"
        assert await wait_for(pilot, lambda: "NAT networking" in urls_line(sv))
        assert "portproxy" in urls_line(sv)


async def test_changing_the_bind_while_running_says_to_restart(ui_env: SimpleNamespace) -> None:
    gw = FakeGateway(ui_env.cfg, running=True)
    app = make_app(ui_env, gateway=gw)
    async with app.run_test(size=SIZE) as pilot:
        sv = await open_settings(pilot, app)
        sv.query_one("#gw_listen", Select).value = "lan"
        assert await wait_for(pilot, sv._fields_valid)
        await pilot.press("ctrl+s")
        assert await wait_for(pilot, lambda: "keeps its old host" in plain(sv.query_one("#set_msg", Static)))
