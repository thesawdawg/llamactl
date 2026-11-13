"""Regression coverage for model navigation, goal drafts and explicit acceptance."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from textual.widgets import Button, Input, Select, Static

from fakes import FakeGateway, plain, wait_for
from llamactl.config import Config, Preset, Profile
from llamactl.goals import TuningGoals
from llamactl.screens.model_workspace import ModelWorkspace, ProfileReview
from test_ui_models import ready
from test_ui_shell import SIZE, add_preset, make_app


async def test_enter_opens_model_without_loading(ui_env: SimpleNamespace) -> None:
    """Opening a model never starts or loads a gateway.

    Args:
        ui_env: Isolated application environment.

    Returns:
        None.
    """
    model = ui_env.add_model()
    gateway = FakeGateway(ui_env.cfg)
    app = make_app(ui_env, gateway=gateway)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        app.views["models"].model_table.focus()
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: isinstance(app.screen, ModelWorkspace))
        assert not gateway.called("load") and not gateway.called("start")
        assert app.screen.model == str(model)
        assert ui_env.cfg.preset(str(model), "chat") is None


async def test_workspace_goals_are_drafts_until_review_accepted(ui_env: SimpleNamespace) -> None:
    """Keeping a proposed plan preserves both launch settings and tuning goals.

    Args:
        ui_env: Isolated application environment.

    Returns:
        None.
    """
    model = ui_env.add_model()
    goals = TuningGoals(max_context=8192, min_effective_tps=25, benchmark="quick")
    saved = Preset("chat", Profile(ctx_size=4096), "edited", None, goals)
    ui_env.cfg.set_preset(str(model), saved)
    ui_env.cfg.save()
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        app.views["models"].action_open_workspace()
        assert await wait_for(pilot, lambda: isinstance(app.screen, ModelWorkspace))
        workspace = app.screen
        assert workspace.query_one("#goal_context", Input).value == "8192"
        workspace.query_one("#goal_effective", Input).value = "30"
        workspace.query_one("#workspace_plan", Button).press()
        assert await wait_for(pilot, lambda: isinstance(app.screen, ProfileReview))
        assert ui_env.cfg.preset(str(model), "chat") == saved
        app.screen.query_one("#review_keep", Button).press()
        assert await wait_for(pilot, lambda: isinstance(app.screen, ModelWorkspace))
        assert Config.load().preset(str(model), "chat") == saved


async def test_workspace_accept_saves_goals_without_loading(ui_env: SimpleNamespace) -> None:
    """Acceptance saves only the reviewed preset; serving remains explicit.

    Args:
        ui_env: Isolated application environment.

    Returns:
        None.
    """
    model = ui_env.add_model()
    add_preset(ui_env, model, name="chat")
    gateway = FakeGateway(ui_env.cfg, running=True)
    app = make_app(ui_env, gateway=gateway)
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        app.views["models"].action_open_workspace()
        assert await wait_for(pilot, lambda: isinstance(app.screen, ModelWorkspace))
        app.screen.query_one("#goal_effective", Input).value = "31"
        app.screen.query_one("#goal_benchmark", Select).value = "full"
        app.screen.query_one("#workspace_plan", Button).press()
        assert await wait_for(pilot, lambda: isinstance(app.screen, ProfileReview))
        app.screen.query_one("#review_save", Button).press()
        assert await wait_for(pilot, lambda: isinstance(app.screen, ModelWorkspace))
        preset = Config.load().preset(str(model), "chat")
        assert preset.goals.min_effective_tps == 31 and preset.goals.benchmark == "full"
        assert not gateway.called("load")
        assert "next load" in plain(app.screen.query_one("#workspace_status", Static))


@pytest.mark.parametrize("size", [(60, 20), (80, 24), (100, 32)])
async def test_workspace_review_layout_and_mouse_selection(
        ui_env: SimpleNamespace, size: tuple[int, int]) -> None:
    """A click opens controls and both guided screens remain reachable in small terminals.

    Args:
        ui_env: Isolated application environment.
        size: Terminal columns and rows.

    Returns:
        None.
    """
    from test_ui_layout import violations

    ui_env.add_model()
    app = make_app(ui_env)
    async with app.run_test(size=size) as pilot:
        await ready(pilot, app)
        await pilot.click("#model_table", offset=(3, 1))
        assert await wait_for(pilot, lambda: isinstance(app.screen, ModelWorkspace))
        assert violations(app.screen) == []
        app.screen.query_one("#workspace_plan", Button).press()
        assert await wait_for(pilot, lambda: isinstance(app.screen, ProfileReview))
        assert violations(app.screen) == []
