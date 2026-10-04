"""Pilot tests: profile editor and the app shell with Models view."""
from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

from textual.app import App
from textual.widgets import Button, ContentSwitcher, DataTable, Input, Label

from llamactl import hf, recommend
from llamactl.app import LlamaCtl
from llamactl.config import Config, Profile
from llamactl.gguf import ModelInfo
from llamactl.hostinfo import Gpu, Host
from llamactl.screens.confirm import ConfirmScreen
from llamactl.screens.profile import ProfileEditor
from llamactl.screens.prompt import PromptScreen
from llamactl.screens.recommend import RecommendationScreen


def _info() -> ModelInfo:
    return ModelInfo(path=Path("m.gguf"), size=int(8.5e9), architecture="x", name="m",
                     n_layer=32, n_embd=4096, n_head=32, n_head_kv=8,
                     head_dim_k=128, head_dim_v=128, ctx_train=131072,
                     expert_count=0, file_type=0, sharded=1, parameter_count=0)


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


async def _wait(pilot, cond, tries: int = 40) -> bool:
    """Poll `cond` while the app's workers settle."""
    for _ in range(tries):
        await pilot.pause(0.1)
        if cond():
            return True
    return False


async def test_shell_models_view(isolated: SimpleNamespace, gguf_file: Path) -> None:
    shutil.copy(gguf_file, isolated.model_dir / gguf_file.name)
    app = LlamaCtl()
    async with app.run_test(size=(140, 45)) as pilot:
        assert await _wait(pilot, lambda: app.views["models"].model_table.row_count == 1)
        # host poll has landed -> Fits resolved to a verdict
        assert await _wait(pilot, lambda: app.host.vram_total > 0)
        row_key = str(isolated.model_dir / gguf_file.name)
        fits = app.views["models"].model_table.get_cell(row_key, "fits")
        assert str(getattr(fits, "plain", fits)) != "..."
        assert app.query_one("#main", ContentSwitcher).current == "models"

        await pilot.press("2")
        await pilot.pause()
        assert app.query_one("#main", ContentSwitcher).current == "sessions"

        await pilot.press("1")
        await pilot.pause()
        app.views["models"].model_table.focus()
        await pilot.pause()
        await pilot.press("e")
        assert await _wait(pilot, lambda: isinstance(app.screen, ProfileEditor))
        await pilot.press("escape")
        await pilot.pause()


async def test_focus_moves_into_views(isolated: SimpleNamespace, gguf_file: Path) -> None:
    shutil.copy(gguf_file, isolated.model_dir / gguf_file.name)
    app = LlamaCtl()
    async with app.run_test(size=(140, 45)) as pilot:
        assert await _wait(pilot, lambda: app.views["models"].model_table.row_count == 1)
        await pilot.press("2")
        await pilot.pause(0.2)
        assert app.screen.focused is app.views["sessions"].sess_table
        await pilot.press("1")
        await pilot.pause(0.2)
        assert app.screen.focused is app.views["models"].model_table


async def test_sessions_external_down(isolated: SimpleNamespace) -> None:
    app = LlamaCtl()
    async with app.run_test(size=(140, 45)) as pilot:
        app.store.attach("127.0.0.1:59999")  # nothing listens -> probe fails
        await pilot.press("2")
        await pilot.pause()
        sv = app.views["sessions"]
        assert await _wait(pilot, lambda: sv.sess_table.row_count == 1)
        assert await _wait(pilot, lambda: sv.state.get(app.store.sessions[0].id) == "down")


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


async def test_recommendation_screen(isolated: SimpleNamespace) -> None:
    app = LlamaCtl()
    async with app.run_test(size=(140, 45)) as pilot:
        result = []
        scr = RecommendationScreen(app.cfg, "/m.gguf", _info(), app.host)
        app.push_screen(scr, result.append)
        await pilot.pause(0.2)
        assert scr.query_one("#rec_table", DataTable).row_count == len(scr.rec.reasons)
        await pilot.press("enter")
        await pilot.pause()
        assert result == ["accepted"]
        assert "/m.gguf" in app.cfg.profiles
        assert app.cfg.profiles["/m.gguf"].ctx_size == scr.rec.profile.ctx_size

    app2 = LlamaCtl()
    async with app2.run_test(size=(140, 45)) as pilot:
        result2 = []
        app2.push_screen(RecommendationScreen(app2.cfg, "/n.gguf", _info(), app2.host),
                         result2.append)
        await pilot.pause(0.2)
        await pilot.press("escape")
        await pilot.pause()
        assert result2 == ["skipped"]
        assert "/n.gguf" in app2.cfg.seen


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
    app = LlamaCtl()
    async with app.run_test(size=(140, 45)) as pilot:
        await pilot.press("4")
        await pilot.pause(0.2)
        sv = app.views["settings"]
        bin_in = sv.query_one("#bin_dir", Input)
        bin_in.value = "/no/such/dir"
        await pilot.pause(0.2)
        assert not bin_in.is_valid
        bin_in.value = str(bin_dir)
        add = sv.query_one("#add_dir", Input)
        add.value = str(new_dir)
        add.focus()
        await pilot.press("enter")
        await pilot.pause(0.2)
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
    app = LlamaCtl()
    async with app.run_test(size=(140, 45)) as pilot:
        await pilot.press("3")
        await pilot.pause(0.2)
        hv = app.views["hf"]
        hv.query_one("#hf_q", Input).value = "qwen"
        hv.query_one("#hf_q", Input).post_message(Input.Submitted(hv.query_one("#hf_q", Input), "qwen"))
        assert await _wait(pilot, lambda: hv.query_one("#repos", DataTable).row_count == 1)
        hv.repo = "org/repo"
        hv.load_files("org/repo")
        assert await _wait(pilot, lambda: hv.query_one("#files", DataTable).row_count == 1)
        fits = hv.query_one("#files", DataTable).get_cell_at((0, 4))
        assert "GPU" in str(fits)
