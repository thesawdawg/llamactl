"""Pilot tests: the first-run setup flow over fakes (no llama.cpp, no network)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from textual.widgets import Button, Checkbox, Input, Label, OptionList, Static

from fakes import FakeGateway, FakeRunner, ok, plain, sparse_copy, wait_for
from llamactl.config import Config
from llamactl.screens.onboarding import OnboardingScreen
from llamactl.screens.tune import TuneScreen
from test_ui_shell import SIZE, current, make_app


def fresh(env: SimpleNamespace) -> Config:
    """The environment's config as a first run sees it."""
    env.cfg.onboarded = False
    return env.cfg


def shown(app, wid: str) -> bool:
    """True when a widget of the setup screen is displayed."""
    return app.screen.query_one(f"#{wid}").display


def progress(app) -> str:
    """The setup's progress line."""
    return plain(app.screen.query_one("#ob_progress", Label))


async def open_setup(pilot, app) -> OnboardingScreen:
    """Wait for the setup screen and for its first detection."""
    assert await wait_for(pilot, lambda: isinstance(app.screen, OnboardingScreen))
    return app.screen


async def press(pilot, app, wid: str) -> None:
    """Press a button of the setup screen and let the screen settle."""
    app.screen.query_one(f"#{wid}", Button).press()
    await pilot.pause(0.05)


async def to_step(pilot, app, n: int) -> None:
    """Press Next until the progress line says 'Step n'."""
    while f"Step {n} of" not in progress(app):
        await press(pilot, app, "ob_next")


async def test_fresh_config_walks_all_five_steps_and_ends_onboarded(
        ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model("alpha.gguf")
    cfg = fresh(ui_env)
    gw = FakeGateway(cfg)
    app = make_app(ui_env, gateway=gw, which=lambda name: None)
    async with app.run_test(size=SIZE) as pilot:
        scr = await open_setup(pilot, app)
        # 1: llama.cpp
        assert progress(app) == "Step 1 of 5: llama.cpp"
        assert await wait_for(pilot, lambda: "b1" in plain(scr.query_one("#ob_llama", Static)))
        assert str(ui_env.bin_dir / "llama-server") in plain(scr.query_one("#ob_llama", Static))
        assert scr.query_one("#ob_back", Button).disabled
        await press(pilot, app, "ob_next")
        # 2: hardware
        assert progress(app) == "Step 2 of 5: Hardware"
        assert await wait_for(pilot, lambda: "CUDA0" in plain(scr.query_one("#ob_hw", Static)))
        sentence = plain(scr.query_one("#ob_sentence", Static))
        assert "fully on the GPU" in sentence and "Chat intent" in sentence
        await press(pilot, app, "ob_next")
        # 3: models
        assert progress(app) == "Step 3 of 5: Models"
        assert await wait_for(pilot, lambda: "1 model" in plain(scr.query_one("#ob_scan", Static)))
        assert not shown(app, "ob_find")
        await press(pilot, app, "ob_next")
        # 4: purpose
        assert progress(app) == "Step 4 of 5: Purpose"
        intents = scr.query_one("#ob_intents", OptionList)
        models = scr.query_one("#ob_models", OptionList)
        assert intents.option_count == 5
        assert await wait_for(pilot, lambda: models.option_count == 2)
        assert "GPU" in str(models.get_option_at_index(1).prompt)
        intents.highlighted = [str(intents.get_option_at_index(i).id)
                               for i in range(5)].index("agent")
        await pilot.pause(0.1)
        models.highlighted = 1
        await pilot.pause(0.1)
        assert scr.draft.default_intent == "agent" and scr.draft.model == str(m)
        await press(pilot, app, "ob_next")
        # 5: finish
        assert progress(app) == "Step 5 of 5: Finish"
        assert str(scr.query_one("#ob_next", Button).label) == "Finish"
        assert "Agent" in plain(scr.query_one("#ob_summary", Static))
        assert scr.query_one("#ob_gateway", Checkbox).value and not scr.query_one("#ob_tune", Checkbox).value
        await press(pilot, app, "ob_next")
        assert await wait_for(pilot, lambda: gw.called("start"))
        assert cfg.onboarded and Config.load().onboarded
        assert cfg.default_intent == "agent"
        p = cfg.preset(str(m))
        assert p is not None and p.source == "estimated" and p.intent == "agent"
        assert await wait_for(pilot, lambda: "Gateway running at" in plain(scr.query_one("#ob_result", Static)))
        result = plain(scr.query_one("#ob_result", Static))
        assert "http://127.0.0.1:8080/v1" in result and "curl http://127.0.0.1:8080/v1/chat/completions" in result
        assert '"model": "alpha"' in result
        await press(pilot, app, "ob_close")
        assert await wait_for(pilot, lambda: not isinstance(app.screen, OnboardingScreen))
        assert Config.load().preset(str(m)).intent == "agent"


async def test_skip_marks_onboarded_and_changes_nothing_else(ui_env: SimpleNamespace) -> None:
    ui_env.add_model()
    cfg = fresh(ui_env)
    gw = FakeGateway(cfg)
    app = make_app(ui_env, gateway=gw, which=lambda name: None)
    async with app.run_test(size=SIZE) as pilot:
        await open_setup(pilot, app)
        await press(pilot, app, "ob_skip")
        assert await wait_for(pilot, lambda: not isinstance(app.screen, OnboardingScreen))
        saved = Config.load()
        assert saved.onboarded and saved.bin_dir == str(ui_env.bin_dir) and saved.models == {}
        assert gw.called("start") == []


async def test_escape_skips_too_and_closing_without_finishing_asks_again(
        ui_env: SimpleNamespace) -> None:
    cfg = fresh(ui_env)
    app = make_app(ui_env, which=lambda name: None)
    async with app.run_test(size=SIZE) as pilot:
        await open_setup(pilot, app)
        await pilot.press("ctrl+n")
        await pilot.pause(0.05)
        assert progress(app).startswith("Step 2")
    assert not cfg.onboarded and not (ui_env.cfg_dir / "config.json").exists()   # quit mid-way
    app = make_app(ui_env, which=lambda name: None)
    async with app.run_test(size=SIZE) as pilot:
        await open_setup(pilot, app)
        await pilot.press("escape")
        assert await wait_for(pilot, lambda: not isinstance(app.screen, OnboardingScreen))
        assert Config.load().onboarded


async def test_missing_llama_cpp_is_explained_and_a_good_folder_is_accepted(
        ui_env: SimpleNamespace, tmp_path: Path) -> None:
    cfg = fresh(ui_env)
    cfg.bin_dir = str(tmp_path / "nowhere")
    good = tmp_path / "good"
    good.mkdir()
    (good / "llama-server").touch()
    app = make_app(ui_env, which=lambda name: None)
    app.build_for = lambda folder: f"v-{Path(folder).name}"            # type: ignore[method-assign]
    async with app.run_test(size=SIZE) as pilot:
        scr = await open_setup(pilot, app)
        text = lambda wid: plain(scr.query_one(wid, Static))           # noqa: E731
        assert await wait_for(pilot, lambda: "is not a folder" in text("#ob_bin_msg"))
        assert "not found" in text("#ob_llama") and "github.com/ggml-org/llama.cpp" in text("#ob_llama")
        inp = scr.query_one("#ob_bin", Input)
        inp.value = str(tmp_path)                                      # a folder without llama-server
        await pilot.pause(0.1)
        assert "no llama-server inside" in text("#ob_bin_msg")
        inp.value = str(good)
        assert await wait_for(pilot, lambda: "v-good" in text("#ob_llama"))
        assert text("#ob_bin_msg") == "folder ok" and "folder you entered" in text("#ob_llama")
        assert cfg.bin_dir == str(tmp_path / "nowhere")                # the draft only; nothing applied yet
        await to_step(pilot, app, 5)
        assert str(good) in plain(scr.query_one("#ob_summary", Static))
        await press(pilot, app, "ob_next")
        assert cfg.bin_dir == str(good) and cfg.onboarded


async def test_llama_server_is_found_on_path_when_the_configured_folder_is_wrong(
        ui_env: SimpleNamespace, tmp_path: Path) -> None:
    cfg = fresh(ui_env)
    cfg.bin_dir = str(tmp_path / "nowhere")
    onpath = tmp_path / "onpath"
    onpath.mkdir()
    (onpath / "llama-server").touch()
    app = make_app(ui_env, which=lambda name: str(onpath / name))
    app.build_for = lambda folder: "pathbuild"                         # type: ignore[method-assign]
    async with app.run_test(size=SIZE) as pilot:
        scr = await open_setup(pilot, app)
        assert await wait_for(pilot, lambda: scr.query_one("#ob_bin", Input).value == str(onpath))
        assert await wait_for(pilot, lambda: "pathbuild" in plain(scr.query_one("#ob_llama", Static)))
        assert "from PATH" in plain(scr.query_one("#ob_llama", Static))


async def test_no_models_offers_find_a_model_which_lands_on_discover(
        ui_env: SimpleNamespace) -> None:
    cfg = fresh(ui_env)                                                # the model folder is empty
    gw = FakeGateway(cfg)
    app = make_app(ui_env, gateway=gw, which=lambda name: None)
    async with app.run_test(size=SIZE) as pilot:
        scr = await open_setup(pilot, app)
        await to_step(pilot, app, 3)
        assert await wait_for(pilot, lambda: "No models found" in plain(scr.query_one("#ob_scan", Static)))
        assert await wait_for(pilot, lambda: shown(app, "ob_find"))
        await press(pilot, app, "ob_find")
        assert await wait_for(pilot, lambda: not isinstance(app.screen, OnboardingScreen))
        assert await wait_for(pilot, lambda: current(app) == "discover")
        assert Config.load().onboarded and gw.called("start") == []


async def test_adding_a_folder_rescans_in_step_three(ui_env: SimpleNamespace, tmp_path: Path) -> None:
    cfg = fresh(ui_env)
    extra = tmp_path / "extra"
    extra.mkdir()
    sparse_copy(ui_env.add_model("seed.gguf"), extra / "other.gguf")
    (ui_env.model_dir / "seed.gguf").unlink()
    app = make_app(ui_env, which=lambda name: None)
    async with app.run_test(size=SIZE) as pilot:
        scr = await open_setup(pilot, app)
        await to_step(pilot, app, 3)
        assert await wait_for(pilot, lambda: "No models found" in plain(scr.query_one("#ob_scan", Static)))
        add = scr.query_one("#add_dir", Input)
        add.value = str(extra)
        add.focus()
        await pilot.pause(0.1)
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: "1 model" in plain(scr.query_one("#ob_scan", Static)))
        assert not shown(app, "ob_find")
        assert cfg.model_dirs == [str(ui_env.model_dir)]               # still only the draft
        await to_step(pilot, app, 5)
        await press(pilot, app, "ob_next")
        assert await wait_for(pilot, lambda: Config.load().model_dirs == [str(ui_env.model_dir), str(extra)])
        assert await wait_for(pilot, lambda: len(app.models) == 1)


async def test_tune_now_opens_the_tune_screen_after_closing(ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    cfg = fresh(ui_env)
    app = make_app(ui_env, gateway=FakeGateway(cfg), runner=FakeRunner(ok(55.0)),
                   which=lambda name: None)
    async with app.run_test(size=SIZE) as pilot:
        scr = await open_setup(pilot, app)
        await to_step(pilot, app, 4)
        models = scr.query_one("#ob_models", OptionList)
        assert await wait_for(pilot, lambda: models.option_count == 2)
        models.highlighted = 1
        await pilot.pause(0.1)
        await press(pilot, app, "ob_next")
        scr.query_one("#ob_tune", Checkbox).value = True
        await pilot.pause(0.05)
        await press(pilot, app, "ob_next")
        assert await wait_for(pilot, lambda: scr.result is not None)
        assert str(scr.query_one("#ob_close", Button).label) == "Close and tune"
        assert cfg.preset(str(m)).source == "estimated"
        await press(pilot, app, "ob_close")
        assert await wait_for(pilot, lambda: isinstance(app.screen, TuneScreen))
        assert await wait_for(pilot, lambda: app.screen.state == "done")
        assert cfg.preset(str(m)).source == "tuned"


async def test_finishing_without_a_model_starts_nothing(ui_env: SimpleNamespace) -> None:
    ui_env.add_model()
    cfg = fresh(ui_env)
    gw = FakeGateway(cfg)
    app = make_app(ui_env, gateway=gw, which=lambda name: None)
    async with app.run_test(size=SIZE) as pilot:
        scr = await open_setup(pilot, app)
        await to_step(pilot, app, 5)
        assert scr.query_one("#ob_gateway", Checkbox).disabled and not scr.query_one("#ob_gateway", Checkbox).value
        assert "none" in plain(scr.query_one("#ob_summary", Static))
        await press(pilot, app, "ob_next")
        assert await wait_for(pilot, lambda: scr.result is not None)
        assert "No model yet" in plain(scr.query_one("#ob_result", Static))
        assert cfg.onboarded and cfg.models == {} and gw.called("start") == []


async def test_a_gateway_that_cannot_start_is_reported_in_the_result(
        ui_env: SimpleNamespace) -> None:
    from llamactl.gateway import GatewayError

    ui_env.add_model()
    cfg = fresh(ui_env)
    gw = FakeGateway(cfg)
    gw.fail("start", GatewayError("port 8080 is already in use"))
    app = make_app(ui_env, gateway=gw, which=lambda name: None)
    async with app.run_test(size=SIZE) as pilot:
        scr = await open_setup(pilot, app)
        await to_step(pilot, app, 4)
        models = scr.query_one("#ob_models", OptionList)
        assert await wait_for(pilot, lambda: models.option_count == 2)
        models.highlighted = 1
        await pilot.pause(0.1)
        await to_step(pilot, app, 5)
        await press(pilot, app, "ob_next")
        assert await wait_for(pilot, lambda: "did not start" in plain(scr.query_one("#ob_result", Static)))
        assert "already in use" in plain(scr.query_one("#ob_result", Static))
        assert cfg.onboarded and app.is_running


async def test_back_returns_to_the_previous_step_and_keeps_choices(ui_env: SimpleNamespace) -> None:
    cfg = fresh(ui_env)
    app = make_app(ui_env, which=lambda name: None)
    async with app.run_test(size=SIZE) as pilot:
        scr = await open_setup(pilot, app)
        await to_step(pilot, app, 4)
        intents = scr.query_one("#ob_intents", OptionList)
        intents.highlighted = 3
        await pilot.pause(0.1)
        choice = scr.draft.default_intent
        await press(pilot, app, "ob_back")
        assert progress(app).startswith("Step 3")
        await press(pilot, app, "ob_next")
        assert scr.query_one("#ob_intents", OptionList).highlighted == 3
        assert scr.draft.default_intent == choice and cfg.default_intent == "chat"


async def test_migrated_v2_config_with_presets_does_not_show_the_setup(
        ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    ui_env.cfg_dir.mkdir(parents=True, exist_ok=True)
    (ui_env.cfg_dir / "config.json").write_text(json.dumps({
        "version": 2, "bin_dir": str(ui_env.bin_dir), "model_dirs": [str(ui_env.model_dir)],
        "profiles": {str(m): {"ctx_size": 8192}}}))
    cfg = Config.load()
    assert cfg.onboarded
    app = make_app(ui_env)
    app.cfg = cfg
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause(0.3)
        assert not isinstance(app.screen, OnboardingScreen)
        assert (ui_env.cfg_dir / "config.v2.bak.json").exists() is False       # nothing was saved


async def test_run_setup_again_from_settings_and_cancel_changes_nothing(
        ui_env: SimpleNamespace) -> None:
    ui_env.add_model()
    app = make_app(ui_env, which=lambda name: None)                           # onboarded already
    async with app.run_test(size=SIZE) as pilot:
        assert await wait_for(pilot, lambda: app.snapshot is not None)
        assert not isinstance(app.screen, OnboardingScreen)
        app.screen.set_focus(None)
        await pilot.press("5")
        await pilot.pause(0.1)
        app.views["settings"].query_one("#run_setup", Button).press()
        scr = await open_setup(pilot, app)
        assert await wait_for(pilot, lambda: scr.flow is not None)          # mounted
        assert str(scr.query_one("#ob_skip", Button).label) == "Cancel"
        assert scr.query_one("#ob_bin", Input).value == str(ui_env.bin_dir)    # prefilled
        await press(pilot, app, "ob_skip")
        assert await wait_for(pilot, lambda: not isinstance(app.screen, OnboardingScreen))
        assert not (ui_env.cfg_dir / "config.json").exists()                    # no save on cancel


# ---------------------------------------------------------------------------- reachable

from conftest import FAKE_LAN  # noqa: E402


async def test_reachable_checkbox_is_off_by_default_and_changes_nothing(
        ui_env: SimpleNamespace) -> None:
    ui_env.add_model()
    cfg = fresh(ui_env)
    app = make_app(ui_env, gateway=FakeGateway(cfg), which=lambda name: None)
    async with app.run_test(size=SIZE) as pilot:
        scr = await open_setup(pilot, app)
        await to_step(pilot, app, 5)
        box = scr.query_one("#ob_reachable", Checkbox)
        assert not box.value and not box.disabled         # usable even without a model
        await press(pilot, app, "ob_next")
        assert await wait_for(pilot, lambda: scr.result is not None)
        assert cfg.gateway.host == "127.0.0.1" and cfg.gateway.api_key == ""
        result = plain(scr.query_one("#ob_result", Static))
        assert FAKE_LAN not in result and "API key" not in result
        assert not shown(app, "ob_reachable")


async def test_reachable_sets_lan_generates_a_key_and_shows_the_urls_once(
        ui_env: SimpleNamespace) -> None:
    m = ui_env.add_model()
    cfg = fresh(ui_env)
    gw = FakeGateway(cfg)
    app = make_app(ui_env, gateway=gw, which=lambda name: None)
    async with app.run_test(size=SIZE) as pilot:
        scr = await open_setup(pilot, app)
        await to_step(pilot, app, 4)
        models = scr.query_one("#ob_models", OptionList)
        assert await wait_for(pilot, lambda: models.option_count == 2)
        models.highlighted = 1
        assert await wait_for(pilot, lambda: scr.draft.model == str(m))
        await to_step(pilot, app, 5)
        scr.query_one("#ob_reachable", Checkbox).value = True
        await press(pilot, app, "ob_next")
        assert await wait_for(pilot, lambda: gw.called("start"))
        assert cfg.gateway.host == "0.0.0.0" and len(cfg.gateway.api_key) >= 43
        saved = Config.load().gateway
        assert (saved.host, saved.api_key) == ("0.0.0.0", cfg.gateway.api_key)
        assert await wait_for(pilot, lambda: "Gateway running at" in plain(scr.query_one("#ob_result", Static)))
        result = plain(scr.query_one("#ob_result", Static))
        assert f"Endpoint  http://{FAKE_LAN}:8080/v1" in result
        assert "http://127.0.0.1:8080/v1" in result and "0.0.0.0:" not in result
        assert cfg.gateway.api_key in result and "shown only now" in result
        assert f"curl http://{FAKE_LAN}:8080/v1/chat/completions" in result and "Bearer" in result
        assert gw.bound_host == "0.0.0.0"


async def test_reachable_keeps_an_existing_key(ui_env: SimpleNamespace) -> None:
    ui_env.add_model()
    cfg = fresh(ui_env)
    cfg.gateway.api_key = "mine"
    app = make_app(ui_env, gateway=FakeGateway(cfg), which=lambda name: None)
    async with app.run_test(size=SIZE) as pilot:
        scr = await open_setup(pilot, app)
        await to_step(pilot, app, 5)
        scr.query_one("#ob_reachable", Checkbox).value = True
        await press(pilot, app, "ob_next")
        assert await wait_for(pilot, lambda: scr.result is not None)
        assert cfg.gateway.api_key == "mine" and cfg.gateway.host == "0.0.0.0"
        assert "mine" not in plain(scr.query_one("#ob_result", Static))


async def test_rerunning_setup_shows_the_current_choice_and_unchecking_goes_local(
        ui_env: SimpleNamespace) -> None:
    ui_env.add_model()
    cfg = fresh(ui_env)
    cfg.gateway.host = "0.0.0.0"
    app = make_app(ui_env, gateway=FakeGateway(cfg), which=lambda name: None)
    async with app.run_test(size=SIZE) as pilot:
        scr = await open_setup(pilot, app)
        await to_step(pilot, app, 5)
        box = scr.query_one("#ob_reachable", Checkbox)
        assert box.value
        box.value = False
        await press(pilot, app, "ob_next")
        assert await wait_for(pilot, lambda: scr.result is not None)
        assert cfg.gateway.host == "127.0.0.1"


async def test_reachable_warns_when_the_running_gateway_has_the_old_bind(
        ui_env: SimpleNamespace) -> None:
    ui_env.add_model()
    cfg = fresh(ui_env)
    gw = FakeGateway(cfg, running=True)                # runs on 127.0.0.1
    app = make_app(ui_env, gateway=gw, which=lambda name: None)
    async with app.run_test(size=SIZE) as pilot:
        scr = await open_setup(pilot, app)
        await to_step(pilot, app, 5)
        scr.query_one("#ob_reachable", Checkbox).value = True
        await press(pilot, app, "ob_next")
        assert await wait_for(pilot, lambda: scr.result is not None)
        assert "already runs on 127.0.0.1" in plain(scr.query_one("#ob_result", Static))
