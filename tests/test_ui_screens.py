"""Pilot tests: IntentPicker and PlanScreen on a bare app (no shell needed)."""
from __future__ import annotations

from pathlib import Path

from textual.app import App
from textual.widgets import Button, DataTable, Label, OptionList

from fakes import make_host, make_info, plain, wait_for
from llamactl import intents, planner
from llamactl.calibration import Calibration
from llamactl.config import Config, Preset, Profile
from llamactl.measure import Measurement
from llamactl.modelrows import intent_rows
from llamactl.screens.confirm import ConfirmScreen
from llamactl.screens.intent_picker import IntentPicker, card
from llamactl.screens.plan import PlanScreen

MODEL = "/m/qwen3.gguf"


class Harness(App[None]):
    """Pushes one screen and records what it dismisses with."""

    def __init__(self, screen) -> None:
        super().__init__()
        self.screen_under_test = screen
        self.result: list = []

    def on_mount(self) -> None:
        self.push_screen(self.screen_under_test, self.result.append)


def _rows(cfg: Config):
    info, host = make_info(MODEL), make_host()
    plans = planner.plan_all(info, host, cfg.default_profile, cfg.headroom_pct)
    return intent_rows(plans, cfg.models.get(MODEL), Path(MODEL), host, cfg.headroom_pct,
                       Calibration(), ""), plans


# ---------------------------------------------------------------------------- IntentPicker

def test_card_carries_verdict_numbers_reason_and_preset_status() -> None:
    cfg = Config()
    cfg.set_preset(MODEL, Preset("agent", Profile(), "tuned",
                                 Measurement(ok=True, gen_tps=33.0, at="t", build="b")))
    rows = {r.key: r for r in _rows(cfg)[0]}
    text = card(rows["agent"]).plain
    assert "Agent" in text and "GPU" in text and "ctx x 1 slot" in text and "KV" in text
    assert "tuned, active, 33 tok/s" in text
    assert "none yet" in card(rows["chat"]).plain
    assert "4 slots" in card(rows["api"]).plain


async def test_picker_returns_the_chosen_intent_key() -> None:
    rows, _ = _rows(Config())
    app = Harness(IntentPicker("qwen3", rows, current="agent"))
    async with app.run_test(size=(120, 50)) as pilot:
        lst = app.screen.query_one("#intents", OptionList)
        assert lst.option_count == len(rows) and lst.highlighted == 1     # agent is highlighted
        await pilot.press("down", "enter")
        await pilot.pause()
        assert app.result == ["context"]


async def test_picker_cancel_and_default_highlight() -> None:
    rows, _ = _rows(Config())
    app = Harness(IntentPicker("qwen3", rows))
    async with app.run_test(size=(120, 50)) as pilot:
        assert app.screen.query_one("#intents", OptionList).highlighted == 0
        await pilot.press("escape")
        await pilot.pause()
        assert app.result == [None]


# ---------------------------------------------------------------------------- PlanScreen

def _plan_screen(cfg: Config, intent: str = "chat") -> PlanScreen:
    return PlanScreen(cfg, MODEL, "qwen3", make_info(MODEL), make_host(), intents.by_key[intent])


async def test_plan_screen_shows_the_reasons_table_and_verdict() -> None:
    cfg = Config()
    scr = _plan_screen(cfg)
    app = Harness(scr)
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        t = scr.query_one("#plan_table", DataTable)
        assert t.row_count == len(scr.plan.reasons) > 0
        assert [str(c.label) for c in t.columns.values()] == ["Setting", "Base", "Planned", "Why"]
        assert "GPU" in plain(scr.query_one(".verdict", Label))


async def test_plan_screen_save_stores_an_estimated_preset() -> None:
    cfg = Config()
    scr = _plan_screen(cfg, "agent")
    app = Harness(scr)
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        assert app.result == ["saved"]
    p = cfg.preset(MODEL, "agent")
    assert (p.source, p.profile) == ("estimated", scr.plan.profile)
    assert cfg.models[MODEL].active == "agent"


async def test_plan_screen_refuses_to_overwrite_a_tuned_preset_without_confirmation() -> None:
    cfg = Config()
    tuned = Preset("chat", Profile(ctx_size=1234), "tuned", Measurement(ok=True, gen_tps=9.0))
    cfg.set_preset(MODEL, tuned)
    scr = _plan_screen(cfg)
    app = Harness(scr)
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        await pilot.press("s")
        assert await wait_for(pilot, lambda: isinstance(app.screen, ConfirmScreen))
        assert cfg.preset(MODEL, "chat") is tuned and app.result == []      # nothing replaced yet
        await pilot.press("n")
        await pilot.pause(0.1)
        assert cfg.preset(MODEL, "chat") is tuned and app.result == []
        assert "Not saved" in plain(scr.query_one("#plan_msg", Label))
        await pilot.press("s")
        assert await wait_for(pilot, lambda: isinstance(app.screen, ConfirmScreen))
        await pilot.press("y")
        await pilot.pause(0.1)
        assert app.result == ["saved"]
    assert cfg.preset(MODEL, "chat").source == "estimated"


async def test_plan_screen_buttons_dismiss_with_their_action() -> None:
    for key, want in (("e", "edit"), ("t", "tune"), ("v", "serve"), ("escape", None)):
        app = Harness(_plan_screen(Config()))
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await pilot.press(key)
            await pilot.pause()
            assert app.result == [want]
    app = Harness(_plan_screen(Config()))
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        app.screen.query_one("#tune", Button).press()
        await pilot.pause()
        assert app.result == ["tune"]


async def test_plan_screen_with_nothing_to_change() -> None:
    cfg = Config()
    scr = _plan_screen(cfg)
    cfg.default_profile = scr.plan.profile            # base already equals the plan
    scr2 = _plan_screen(cfg)
    app = Harness(scr2)
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        assert not scr2.query_one("#plan_table", DataTable).display
        assert scr2.query_one("#plan_none").display
