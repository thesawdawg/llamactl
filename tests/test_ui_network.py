"""Pilot tests: client URLs and the no-key warning in Home, Serving and the start flows."""
from __future__ import annotations

from types import SimpleNamespace

from textual.widgets import DataTable, Static

from conftest import FAKE_LAN
from fakes import FakeGateway, FakeStore, plain, wait_for
from llamactl import network
from llamactl.config import Preset, Profile
from llamactl.gateway import LOADED
from test_ui_models import ready
from test_ui_shell import SIZE, add_preset, make_app

LAN_URL = f"http://{FAKE_LAN}:8080"


def record_notices(app) -> list[tuple[str, str]]:
    """Collect (message, severity) of every notification the app raises."""
    seen: list[tuple[str, str]] = []
    original = app.notify

    def notify(message, *args, **kwargs):
        seen.append((str(message), kwargs.get("severity", "information")))
        return original(message, *args, **kwargs)

    app.notify = notify
    return seen


async def test_home_card_lists_every_client_url_and_warns_without_a_key(
        ui_env: SimpleNamespace) -> None:
    ui_env.cfg.gateway.host = "0.0.0.0"
    gw = FakeGateway(ui_env.cfg, running=True)
    app = make_app(ui_env, gateway=gw)
    async with app.run_test(size=SIZE) as pilot:
        info = app.views["home"].query_one("#gw_info", Static)
        assert await wait_for(pilot, lambda: LAN_URL in plain(info))
        text = plain(info)
        assert "http://127.0.0.1:8080/v1" in text and f"{LAN_URL}/v1" in text
        assert "0.0.0.0:" not in text
        assert "without an API key" in text
        ui_env.cfg.gateway.api_key = "k"
        app.config_changed()
        assert await wait_for(pilot, lambda: "without an API key" not in plain(info))
        assert "required" in plain(info)


async def test_home_card_for_a_loopback_gateway_has_one_url_and_no_warning(
        ui_env: SimpleNamespace) -> None:
    app = make_app(ui_env, gateway=FakeGateway(ui_env.cfg, running=True))
    async with app.run_test(size=SIZE) as pilot:
        info = app.views["home"].query_one("#gw_info", Static)
        assert await wait_for(pilot, lambda: "running" in plain(info))
        assert FAKE_LAN not in plain(info) and "without an API key" not in plain(info)


async def test_home_card_shows_the_wsl_notes_when_exposed(ui_env: SimpleNamespace,
                                                          monkeypatch) -> None:
    monkeypatch.setattr(network, "read_osrelease", lambda: "5.15-microsoft-standard-WSL2")
    monkeypatch.setattr(network, "read_wslinfo", lambda: "mirrored")
    network.reset_cache()
    ui_env.cfg.gateway.host = "0.0.0.0"
    app = make_app(ui_env, gateway=FakeGateway(ui_env.cfg, running=True))
    async with app.run_test(size=SIZE) as pilot:
        info = app.views["home"].query_one("#gw_info", Static)
        assert await wait_for(pilot, lambda: "Hyper-V firewall" in plain(info))


async def test_home_copy_and_snippet_use_the_lan_url(ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    add_preset(ui_env, m, "edited", "chat")
    ui_env.cfg.gateway.host = "0.0.0.0"
    gw = FakeGateway(ui_env.cfg, running=True)
    gw.set_state("alpha", LOADED)
    app = make_app(ui_env, gateway=gw)
    async with app.run_test(size=SIZE) as pilot:
        home = app.views["home"]
        assert await wait_for(pilot, lambda: LAN_URL in plain(home.query_one("#gw_info", Static)))
        assert await wait_for(pilot, lambda: "alpha" in plain(home.query_one("#gw_snippet", Static)))
        assert LAN_URL in plain(home.query_one("#gw_snippet", Static))
        assert await wait_for(pilot, lambda: app.focused is not None)
        await pilot.press("y")
        assert app.clipboard == f"{LAN_URL}/v1"
        await pilot.press("c")
        assert app.clipboard.startswith(f"curl {LAN_URL}/v1/chat/completions")


async def test_serving_view_shows_the_lan_url_in_the_line_and_the_table(
        ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    add_preset(ui_env, m, "estimated", "chat")
    ui_env.cfg.gateway.host = "0.0.0.0"
    gw = FakeGateway(ui_env.cfg, running=True)
    gw.set_state("alpha", LOADED)
    app = make_app(ui_env, gateway=gw)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("3")
        view = app.views["serving"]
        line = view.query_one("#sv_gateway", Static)
        assert await wait_for(pilot, lambda: LAN_URL in plain(line))
        assert "http://127.0.0.1:8080/v1" in plain(line) and "0.0.0.0:" not in plain(line)
        t = view.query_one(DataTable)
        assert await wait_for(pilot, lambda: t.row_count == 1)
        assert LAN_URL in [str(c) for c in t.get_row_at(0)]


async def test_starting_an_open_gateway_without_a_key_warns(ui_env: SimpleNamespace) -> None:
    ui_env.cfg.gateway.host = "0.0.0.0"
    gw = FakeGateway(ui_env.cfg)
    app = make_app(ui_env, gateway=gw)
    notices = record_notices(app)
    async with app.run_test(size=SIZE) as pilot:
        assert await wait_for(pilot, lambda: app.snapshot is not None)
        await pilot.press("u")
        assert await wait_for(pilot, lambda: any("without an API key" in m for m, _ in notices))
        assert [sev for m, sev in notices if "without an API key" in m] == ["warning"]
        assert await wait_for(pilot, lambda: f"Gateway running at http://127.0.0.1:8080, {LAN_URL}"
                              in plain(app._status))


async def test_starting_a_private_or_keyed_gateway_does_not_warn(ui_env: SimpleNamespace) -> None:
    for host, key in (("127.0.0.1", ""), ("0.0.0.0", "k")):
        ui_env.cfg.gateway.host, ui_env.cfg.gateway.api_key = host, key
        gw = FakeGateway(ui_env.cfg)
        app = make_app(ui_env, gateway=gw)
        notices = record_notices(app)
        async with app.run_test(size=SIZE) as pilot:
            assert await wait_for(pilot, lambda: app.snapshot is not None)
            await pilot.press("u")
            assert await wait_for(pilot, lambda: gw.running and app.gateway_state() == "running")
            await pilot.pause(0.1)
            assert not any("API key" in m for m, _ in notices), (host, key)


async def test_serving_a_model_that_starts_an_open_gateway_warns(ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    add_preset(ui_env, m, "edited", "chat")
    ui_env.cfg.gateway.host = "0.0.0.0"
    gw = FakeGateway(ui_env.cfg)
    app = make_app(ui_env, gateway=gw)
    notices = record_notices(app)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        app.views["models"].model_table.focus()
        await pilot.press("v")
        assert await wait_for(pilot, lambda: f"alpha is loaded at {LAN_URL}/v1" in plain(app._status))
        assert await wait_for(pilot, lambda: any("without an API key" in m for m, _ in notices))


async def test_standalone_server_on_a_wildcard_shows_the_lan_url_and_warns(
        ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    ui_env.cfg.set_preset(str(m), Preset("chat", Profile(host="0.0.0.0"), "edited"))
    store = FakeStore()
    app = make_app(ui_env, store=store, health=lambda url: "ready")
    notices = record_notices(app)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        app.views["models"].model_table.focus()
        await pilot.press("s")
        assert await wait_for(pilot, lambda: f"is ready at http://{FAKE_LAN}:9001" in plain(app._status))
        assert await wait_for(pilot, lambda: any("without an API key" in m for m, _ in notices))
        await pilot.press("3")
        t = app.views["serving"].query_one(DataTable)
        assert await wait_for(pilot, lambda: t.row_count == 1)
        assert f"http://{FAKE_LAN}:9001" in [str(c) for c in t.get_row_at(0)]
