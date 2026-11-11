"""Pilot tests: profile editor and the app shell with Models view."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from textual.app import App
from textual.widgets import Button, DataTable, Input, Label

from fakes import wait_for as _wait
from llamactl import actions, hf, intents, planner
from llamactl.app import LlamaCtl
from llamactl.config import Config, Profile
from llamactl.gguf import ModelInfo
from llamactl.hardware import Device, Host
from llamactl.screens.confirm import ConfirmScreen
from llamactl.screens.preset_editor import PresetEditor
from llamactl.screens.prompt import PromptScreen


def _info() -> ModelInfo:
    return ModelInfo(path=Path("m.gguf"), size=int(8.5e9), architecture="x", name="m",
                     n_layer=32, n_embd=4096, n_head=32, n_head_kv=8,
                     head_dim_k=128, head_dim_v=128, ctx_train=131072,
                     expert_count=0, file_type=0, sharded=1, parameter_count=0)


def _host() -> Host:
    return Host(gpus=[Device.of("CUDA0", "G", int(16e9), int(16e9))], ram_total=int(24e9), ram_free=int(24e9),
                cpu_threads=16, cpu_cores=8)


def onboarded_app() -> LlamaCtl:
    """The real app over the isolated (empty) state, as if the first-run setup had been done."""
    cfg = Config.load()
    cfg.onboarded = True
    return LlamaCtl(cfg=cfg)


class _App(App[None]):
    """Minimal app that pushes a PresetEditor and records its result."""

    def __init__(self, screen: PresetEditor) -> None:
        super().__init__()
        self.editor = screen
        self.result = None

    def on_mount(self) -> None:
        self.push_screen(self.editor, lambda r: setattr(self, "result", r))


def _make(info: ModelInfo | None = None) -> _App:
    return _App(PresetEditor(Config(), Profile(), _host(), info, "test"))


async def test_invalid_input_disables_save() -> None:
    app = _make(_info())
    async with app.run_test() as pilot:
        await pilot.pause()
        f = app.editor.fields["ctx_size"]
        f.input.focus()
        await pilot.press(*"abc")
        assert await _wait(pilot, lambda: app.editor.query_one("#save", Button).disabled)
        assert not f.valid and f.has_class("invalid")
        assert app.editor.query_one("#save", Button).disabled
        # error message shows in the impact line
        assert "integer" in str(f.query_one(".impact", Label).render())


async def test_ctx_updates_impact_and_budget() -> None:
    app = _make(_info())
    async with app.run_test() as pilot:
        await pilot.pause()
        f = app.editor.fields["ctx_size"]
        f.input.focus()
        f.input.value = ""
        await pilot.press(*"65536")
        assert await _wait(pilot, lambda: "KV" in str(f.query_one(".impact", Label).render()))
        assert "KV" in str(f.query_one(".impact", Label).render())
        verdict = str(app.editor.query_one(".verdict", Label).render())
        assert "Verdict: " in verdict


async def test_apply_plan_and_save() -> None:
    info, host = _info(), _host()
    app = _make(info)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.editor.fields["cont_batching"].input.focus()  # 'R' must not go into an Input
        await pilot.press("R")
        assert await _wait(pilot, lambda: app.editor._draft.ctx_size == 16384)
        rec = planner.plan(intents.by_key["chat"], info, host, Profile())
        prof = app.editor._draft
        assert prof.ctx_size == rec.profile.ctx_size == 16384
        assert prof.parallel == 1 and prof.fit == "off" and prof.threads == 8
        ctx_f = app.editor.fields["ctx_size"]
        assert ctx_f._reason and ctx_f.query_one(".impact", Label).has_class("recommended")
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert isinstance(app.result, Profile)
        assert app.result.ctx_size == rec.profile.ctx_size


async def test_apply_plan_uses_the_editors_intent() -> None:
    info, host = _info(), _host()
    app = _App(PresetEditor(Config(), Profile(), host, info, "test", intent=intents.by_key["agent"],
                            apply_plan=True))
    async with app.run_test() as pilot:
        assert await _wait(pilot, lambda: app.editor._draft.cache_reuse == 256)
        want = actions.plan_for(Config(), info, host, intents.by_key["agent"])
        assert app.editor._draft.ctx_size == want.profile.ctx_size
        assert app.editor._draft.cache_reuse == 256
        assert "Agent" in str(app.editor.query_one("#ed_status", Label).render())
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert app.result.cache_reuse == 256


async def test_default_profile_no_model() -> None:
    app = _make(None)
    async with app.run_test() as pilot:
        await pilot.pause()
        ed = app.editor
        assert not ed.fields["n_cpu_moe"].input.disabled  # enabled when info is None
        assert ed.fields["tensor_split"].input.disabled   # one GPU in the fixture
        assert ed.fields["pooling"].value == "" and ed.fields["fit"].value == "on"
        ed.fields["cont_batching"].input.focus()
        await pilot.press("R")
        assert await _wait(pilot, lambda: "No model" in str(ed.query_one("#ed_status", Label).render()))
        assert ed._draft == Profile()  # R is a no-op without a model
        assert "No model" in str(ed.query_one("#ed_status", Label).render())


async def test_confirm_screen() -> None:
    app = App()
    async with app.run_test() as pilot:
        result = []
        app.push_screen(ConfirmScreen("Stop x (pid 1)?"), result.append)
        await pilot.pause()
        await pilot.press("y")
        await pilot.pause()
        assert result == [True]
        app.push_screen(ConfirmScreen("Stop x?"), result.append)
        await pilot.pause()
        await pilot.press("n")
        await pilot.pause()
        app.push_screen(ConfirmScreen("Stop x?"), result.append)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert result == [True, False, False]


async def test_prompt_screen() -> None:
    app = App()
    async with app.run_test() as pilot:
        result = []
        app.push_screen(PromptScreen("URL", default="x:1"), result.append)
        await pilot.pause()
        app.screen.query_one("#value", Input).value = "host:8080"
        await pilot.press("enter")
        await pilot.pause()
        app.push_screen(PromptScreen("URL"), result.append)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert result == ["host:8080", None]


async def test_settings_view(isolated: SimpleNamespace, tmp_path: Path,
                             monkeypatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "llama-server").touch()
    new_dir = tmp_path / "extra"
    new_dir.mkdir()
    monkeypatch.setattr(hf, "TOKEN_FILE", isolated.cfg_dir / "hf_token")
    app = onboarded_app()
    async with app.run_test(size=(140, 45)) as pilot:
        await pilot.press("5")
        sv = app.views["settings"]
        bin_in = sv.query_one("#bin_dir", Input)
        bin_in.value = "/no/such/dir"
        assert await _wait(pilot, lambda: not bin_in.is_valid)
        bin_in.value = str(bin_dir)
        add = sv.query_one("#add_dir", Input)
        add.value = str(new_dir)
        add.focus()
        await pilot.press("enter")
        assert await _wait(pilot, lambda: add.value == "" and bin_in.is_valid)
        sv.action_save()
        await pilot.pause()
        loaded = Config.load()
        assert loaded.bin_dir == str(bin_dir)
        assert str(new_dir) in loaded.model_dirs


async def test_hf_view_tables(isolated: SimpleNamespace, monkeypatch) -> None:
    monkeypatch.setattr(hf, "TOKEN_FILE", isolated.cfg_dir / "hf_token")
    monkeypatch.setattr(hf, "search", lambda q, t, limit=30: [
        {"id": "org/repo", "gated": False, "downloads": 5, "likes": 1,
         "lastModified": "2025-01-01", "gguf": {"total": int(8e9),
         "architecture": "qwen3", "context_length": 32768}}])
    monkeypatch.setattr(hf, "list_gguf", lambda r, t: [
        {"path": "m-Q8_0.gguf", "quant": "Q8_0", "size": int(8e9), "parts": 1}])
    app = onboarded_app()
    async with app.run_test(size=(140, 45)) as pilot:
        await pilot.press("4")
        hv = app.views["discover"]
        hv.query_one("#hf_q", Input).value = "qwen"
        hv.query_one("#hf_q", Input).post_message(Input.Submitted(hv.query_one("#hf_q", Input), "qwen"))
        assert await _wait(pilot, lambda: hv.query_one("#repos", DataTable).row_count == 1)
        hv.repo = "org/repo"
        hv.load_files("org/repo")
        assert await _wait(pilot, lambda: hv.query_one("#files", DataTable).row_count == 1)
        fits = hv.query_one("#files", DataTable).get_cell_at((0, 4))
        assert "GPU" in str(fits)


async def test_pollers_finishing_during_shutdown_do_not_fail(isolated: SimpleNamespace) -> None:
    """Regression for the intermittent `WorkerFailed`: a poll finishing as the app shuts down.

    `call_from_thread` ran the poll's callback while the gauges were being torn down, and the
    `NoMatches` it raised surfaced as `WorkerFailed` from `run_test`. Polling every 10 ms makes
    that overlap near certain.
    """
    isolated.cfg_dir.mkdir(parents=True, exist_ok=True)
    (isolated.cfg_dir / "config.json").write_text(
        json.dumps({"version": 3, "gauge_interval": 0.01}))
    for _ in range(8):
        app = onboarded_app()
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.06)


async def test_ui_helper_is_safe_when_the_target_is_gone(isolated: SimpleNamespace) -> None:
    """`LlamaCtl.ui` drops calls to removed widgets and to a closed app instead of raising."""
    app = onboarded_app()
    calls: list[str] = []
    async with app.run_test(size=(140, 45)) as pilot:
        await pilot.pause(0.1)
        gauges = app.screen_stack[0].query_one("#gauges")
        await gauges.remove()
        assert app.ui(gauges.update_host, _host())          # queued, then dropped: no NoMatches
        assert app.ui(calls.append, "kept")
        await pilot.pause(0.1)
        assert calls == ["kept"]
    assert app.ui(calls.append, "after exit") is False
    assert calls == ["kept"]


async def test_setting_field_impact_set_before_it_is_mounted_is_shown_later() -> None:
    """Regression: the editor's first preview ran while the field's label was not composed yet.

    That raised `NoMatches` when the editor was opened right after startup (the intermittent
    crash of the "edit default profile" test); the text is now kept and painted on mount.
    """
    from llamactl import schema
    from llamactl.widgets.field import SettingField

    app = App()
    async with app.run_test() as pilot:
        field = SettingField(schema.by_key["ctx_size"], 0, None, _host())
        field.set_impact("Needs ~1 GB")             # not mounted: must not raise
        field.show_reason("planned for chat")
        await app.mount(field)
        await pilot.pause(0.1)
        lbl = field.query_one(".impact", Label)
        assert "planned for chat" in str(lbl.render()) and lbl.has_class("recommended")
