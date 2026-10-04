"""ProfileEditor pilot tests: validation, impact lines, recommendations."""
from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Button, Label

from llamactl import recommend
from llamactl.config import Config, Profile
from llamactl.gguf import ModelInfo
from llamactl.hostinfo import Gpu, Host
from llamactl.screens.profile import ProfileEditor
from llamactl.widgets.field import SettingField


def _info() -> ModelInfo:
    return ModelInfo(path=Path("m.gguf"), size=int(8.5e9), architecture="x", name="m",
                     n_layer=32, n_embd=4096, n_head=32, n_head_kv=8,
                     head_dim_k=128, head_dim_v=128, ctx_train=131072,
                     expert_count=0, file_type=0, sharded=1)


def _host() -> Host:
    return Host(gpus=[Gpu("G", int(16e9), int(16e9))], ram_total=int(24e9), ram_free=int(24e9),
                cpu_threads=16, cpu_cores=8)


class _App(App[None]):
    """Minimal app that pushes a ProfileEditor and records its result."""

    def __init__(self, screen: ProfileEditor) -> None:
        super().__init__()
        self.editor = screen
        self.result = None

    def on_mount(self) -> None:
        self.push_screen(self.editor, lambda r: setattr(self, "result", r))


def _make(info: ModelInfo | None = None) -> _App:
    return _App(ProfileEditor(Config(), Profile(), _host(), info, "test"))


async def test_invalid_input_disables_save() -> None:
    app = _make(_info())
    async with app.run_test() as pilot:
        await pilot.pause()
        f = app.editor.fields["ctx_size"]
        f.input.focus()
        await pilot.press(*"abc")
        await pilot.pause(0.4)
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
        await pilot.pause(0.4)
        assert "KV" in str(f.query_one(".impact", Label).render())
        verdict = str(app.editor.query_one(".verdict", Label).render())
        assert "Verdict: " in verdict


async def test_apply_recommended_and_save() -> None:
    info, host = _info(), _host()
    app = _make(info)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.editor.fields["cont_batching"].input.focus()  # 'R' must not go into an Input
        await pilot.press("R")
        await pilot.pause(0.4)
        rec = recommend.recommend(info, host, Profile())
        prof = app.editor._draft
        assert prof.ctx_size == rec.profile.ctx_size
        assert prof.cache_type_k == "q8_0" and prof.threads == 8
        ctx_f = app.editor.fields["ctx_size"]
        assert ctx_f._reason and ctx_f.query_one(".impact", Label).has_class("recommended")
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert isinstance(app.result, Profile)
        assert app.result.ctx_size == rec.profile.ctx_size


async def test_default_profile_no_model() -> None:
    app = _make(None)
    async with app.run_test() as pilot:
        await pilot.pause()
        ed = app.editor
        assert not ed.fields["n_cpu_moe"].input.disabled  # enabled when info is None
        assert ed.fields["tensor_split"].input.disabled   # one GPU in the fixture
        ed.fields["cont_batching"].input.focus()
        await pilot.press("R")
        await pilot.pause(0.4)
        assert ed._draft == Profile()  # R is a no-op without a model
        assert "No model" in str(ed.query_one("#ed_status", Label).render())
