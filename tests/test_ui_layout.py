"""Pilot tests: controls stay fully visible (or scrollable) at small terminal sizes."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.app import App
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Input, Select, Switch

from fakes import FakeGateway, FakeRunner, make_host, make_info, ok, wait_for
from llamactl import intents
from llamactl.config import Config, Profile
from llamactl.gateway import LOADED
from llamactl.screens.plan import PlanScreen
from llamactl.screens.preset_editor import PresetEditor
from test_ui_models import ready
from test_ui_onboarding import open_setup, to_step
from test_ui_shell import add_preset, make_app
from test_ui_tune import open_tune

SIZES = [(60, 20), (80, 24)]
CONTROLS = (Button, Input, Select, Switch, Checkbox)


def _displayed(widget: Widget) -> bool:
    """True when the widget and all its ancestors have `display` on."""
    node = widget
    while isinstance(node, Widget):
        if not node.display:
            return False
        node = node.parent
    return True


def _fully_visible(widget: Widget) -> bool:
    """True when the compositor paints the widget's whole region on screen."""
    screen = widget.screen
    entry = screen._compositor.visible_widgets.get(widget)
    if entry is None:
        return False
    region, clip = entry
    return (region.area > 0
            and region == region.intersection(clip).intersection(screen._compositor.size.region))


def _can_scroll_into_view(widget: Widget) -> bool:
    """True when a scrollable ancestor could show the whole widget after scrolling."""
    parent = widget.parent
    while isinstance(parent, Widget):
        if ((parent.allow_vertical_scroll or parent.allow_horizontal_scroll)
                and parent.scrollable_content_region.area):
            region = parent.scrollable_content_region
            if parent.allow_vertical_scroll and not parent.allow_horizontal_scroll:
                return widget.outer_size.width <= region.width
            if parent.allow_horizontal_scroll and not parent.allow_vertical_scroll:
                return widget.outer_size.height <= region.height
            return (widget.outer_size.width <= region.width
                    and widget.outer_size.height <= region.height)
        parent = parent.parent
    return False


def violations(screen) -> list[str]:
    """Displayed controls that are clipped or off-screen with no way to scroll them in.

    Action buttons are docked outside the scroll areas, so a clipped or unreachable control of
    any kind is a bug; a control inside a scrollable container is fine when the container can
    show it whole.
    """
    bad = []
    for widget in screen.walk_children():
        if (not isinstance(widget, CONTROLS) or not _displayed(widget)
                or _fully_visible(widget)):
            continue
        if not _can_scroll_into_view(widget):
            bad.append(f"{type(widget).__name__}#{widget.id}")
    return bad


class Harness(App[None]):
    """Bare app that pushes one screen so it can be probed without the shell."""

    def __init__(self, screen) -> None:
        super().__init__()
        self.screen_under_test = screen

    def on_mount(self) -> None:
        self.push_screen(self.screen_under_test)


@pytest.mark.parametrize("size", SIZES)
async def test_shell_views(ui_env: SimpleNamespace, size: tuple[int, int]) -> None:
    """All five views of the shell, with a model so detail and budget render."""
    m = ui_env.add_model()
    add_preset(ui_env, m, "estimated", "chat")
    app = make_app(ui_env)
    async with app.run_test(size=size) as pilot:
        await ready(pilot, app)
        for key, name in (("1", "home"), ("3", "serving"),
                          ("4", "discover"), ("5", "settings"), ("2", "models")):
            app.screen.set_focus(None)
            await pilot.press(key)
            await pilot.pause(0.05)
            assert violations(app.screen) == [], f"{name} @ {size}"


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("builder", ["plan", "editor"])
async def test_plan_and_editor(builder: str, size: tuple[int, int],
                               ui_env: SimpleNamespace) -> None:
    """PlanScreen and PresetEditor on a bare Harness."""
    m = ui_env.add_model()
    info = make_info(str(m))
    if builder == "plan":
        scr = PlanScreen(Config(), str(m), "alpha", info, make_host(),
                         intents.by_key["chat"])
    else:
        scr = PresetEditor(Config(), Profile(), ui_env.host, info, "alpha - chat")
    app = Harness(scr)
    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.1)
        assert violations(app.screen) == [], f"{builder} @ {size}"


@pytest.mark.parametrize("size", SIZES)
async def test_tune_done(ui_env: SimpleNamespace, size: tuple[int, int]) -> None:
    """TuneScreen after a successful tune (use/keep/close buttons)."""
    m = ui_env.add_model()
    add_preset(ui_env, m, "edited", "chat")
    app = make_app(ui_env, runner=FakeRunner(ok()))
    async with app.run_test(size=size) as pilot:
        await ready(pilot, app)
        scr = await open_tune(pilot, app)
        assert await wait_for(pilot, lambda: scr.state == "done")
        await pilot.pause(0.05)
        assert violations(app.screen) == []


@pytest.mark.parametrize("size", SIZES)
async def test_tune_refused(ui_env: SimpleNamespace, size: tuple[int, int]) -> None:
    """TuneScreen refusing the tune (unload/anyway/close buttons)."""
    m = ui_env.add_model()
    host = ui_env.host
    host.gpus[0] = host.gpus[0].__class__.of("CUDA0", "GPU", host.gpus[0].total, int(1e6))
    host.ram_free = int(1e6)
    gw = FakeGateway(ui_env.cfg, running=True)
    add_preset(ui_env, m, "edited", "chat")
    gw.set_state("alpha", LOADED)
    app = make_app(ui_env, gateway=gw, runner=FakeRunner(ok()))
    async with app.run_test(size=size) as pilot:
        await ready(pilot, app)
        scr = await open_tune(pilot, app)
        assert await wait_for(pilot, lambda: scr.state == "refused")
        await pilot.pause(0.05)
        assert violations(app.screen) == []


@pytest.mark.parametrize("size", SIZES)
async def test_onboarding(ui_env: SimpleNamespace, size: tuple[int, int]) -> None:
    """The setup flow on every step, including the finish checkboxes."""
    ui_env.add_model("alpha.gguf")
    ui_env.cfg.onboarded = False
    app = make_app(ui_env, gateway=FakeGateway(ui_env.cfg), which=lambda name: None)
    async with app.run_test(size=size) as pilot:
        scr = await open_setup(pilot, app)
        for step in range(1, 6):
            await to_step(pilot, app, step)
            await pilot.pause(0.1)
            assert violations(app.screen) == [], f"step {step} @ {size}"
