"""Pilot tests: Discover's fit-aware file list and the post-download intent picker (fake HF)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from textual.widgets import DataTable, Label

from fakes import FakeGateway, FakeRunner, ok, plain, sparse_copy, wait_for
from llamactl import hf
from llamactl.config import Config
from llamactl.hardware import Host
from llamactl.screens.after_download import AfterDownload
from llamactl.screens.intent_picker import IntentPicker
from llamactl.screens.tune import TuneScreen
from test_ui_shell import SIZE, current, make_app

FILES = [
    {"path": "m-Q4_K_M.gguf", "quant": "Q4_K_M", "size": int(5e9), "parts": 1},
    {"path": "m-Q8_0.gguf", "quant": "Q8_0", "size": int(9e9), "parts": 1},
    {"path": "m-BF16.gguf", "quant": "BF16", "size": int(17e9), "parts": 1},
    {"path": "m-Q6_K.gguf", "quant": "Q6_K", "size": int(7e9), "parts": 1},
]


def fake_hf(monkeypatch, ui_env, files=FILES) -> None:
    """Search and listing answer from memory; the token file lives in the temp state dir."""
    monkeypatch.setattr(hf, "TOKEN_FILE", ui_env.cfg_dir / "hf_token")
    monkeypatch.setattr(hf, "search", lambda q, t, limit=30: [
        {"id": "org/repo", "gated": False, "downloads": 5, "likes": 1, "lastModified": "2025-01-01",
         "gguf": {"total": int(8e9), "architecture": "qwen3", "context_length": 32768}}])
    monkeypatch.setattr(hf, "list_gguf", lambda repo, token: [dict(f) for f in files])


async def open_files(pilot, app, files_expected: int):
    """Switch to Discover and list the (fake) repo's files."""
    app.screen.set_focus(None)
    await pilot.press("4")
    await pilot.pause(0.1)
    hv = app.views["discover"]
    hv.repo = "org/repo"
    hv.load_files("org/repo")
    t = hv.query_one("#files", DataTable)
    assert await wait_for(pilot, lambda: t.row_count == files_expected)
    return hv, t


def col(t: DataTable, i: int) -> list[str]:
    """Plain text of one column."""
    return [str(t.get_row_at(r)[i]) for r in range(t.row_count)]


async def test_file_list_is_sorted_judged_weights_only_and_marks_the_best_fit(
        ui_env: SimpleNamespace, monkeypatch) -> None:
    fake_hf(monkeypatch, ui_env)
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        assert await wait_for(pilot, lambda: app.host.ram_total > 0)
        hv, t = await open_files(pilot, app, 4)
        assert col(t, 0) == ["m-BF16.gguf", "m-Q8_0.gguf", "m-Q6_K.gguf", "m-Q4_K_M.gguf"]
        fits = col(t, 4)
        assert fits[0] == "GPU+CPU (weights only)"
        assert fits[1] == "GPU (weights only)  ★ best fit"
        assert fits[2] == "GPU (weights only)" and "best fit" not in fits[3]
        assert t.cursor_row == 1                                  # the cursor starts on the best fit
        legend = plain(hv.query_one("#hf_legend", Label))
        assert "best fit" in legend and "Chat" in legend and "KV cache" in legend


async def test_a_download_starts_for_the_row_you_see_after_sorting(
        ui_env: SimpleNamespace, monkeypatch) -> None:
    fake_hf(monkeypatch, ui_env)
    started: list[str] = []
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        assert await wait_for(pilot, lambda: app.host.ram_total > 0)
        hv, t = await open_files(pilot, app, 4)
        monkeypatch.setattr(hv, "start_download", started.append)
        t.move_cursor(row=3)
        t.focus()
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: started == ["m-Q4_K_M.gguf"])


async def test_nothing_fits_marks_nothing_and_without_a_gpu_the_legend_says_so(
        ui_env: SimpleNamespace, monkeypatch) -> None:
    fake_hf(monkeypatch, ui_env, [{"path": "huge.gguf", "quant": "", "size": int(90e9), "parts": 1}])
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        assert await wait_for(pilot, lambda: app.host.ram_total > 0)
        hv, t = await open_files(pilot, app, 1)
        assert col(t, 4) == ["NO"]
        # the machine loses its GPU (e.g. another setup): same files, now judged against RAM
        ui_env.host.gpus.clear()
        hv.fill_files()
        assert col(t, 4) == ["NO"] and "No GPU found" in plain(hv.query_one("#hf_legend", Label))


async def test_sharded_models_are_one_row_with_parts_and_total_size(
        ui_env: SimpleNamespace, monkeypatch) -> None:
    shards = [{"path": "m-Q8_0-00001-of-00002.gguf", "quant": "Q8_0", "size": int(15e9), "parts": 2},
              {"path": "m-Q4_K_M.gguf", "quant": "Q4_K_M", "size": int(6e9), "parts": 1}]
    fake_hf(monkeypatch, ui_env, shards)
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        assert await wait_for(pilot, lambda: app.host.ram_total > 0)
        hv, t = await open_files(pilot, app, 2)
        assert col(t, 3) == ["2", "1"] and col(t, 2) == ["15.00 GB", "6.00 GB"]
        assert "best fit" in col(t, 4)[1] and "best fit" not in col(t, 4)[0]


async def test_files_are_rejudged_when_the_host_arrives_late(
        ui_env: SimpleNamespace, monkeypatch) -> None:
    fake_hf(monkeypatch, ui_env)
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        assert await wait_for(pilot, lambda: app.host.ram_total > 0)
        hv, t = await open_files(pilot, app, 4)
        real = app.host
        app.host = Host()                                          # as if detection had not finished
        hv.fill_files()
        assert set(col(t, 4)) == {"detecting..."}
        assert "Detecting" in plain(hv.query_one("#hf_legend", Label))
        app._host_update(real)
        await pilot.pause(0.1)
        assert "best fit" in " ".join(col(t, 4))


# ---------------------------------------------------------------------------- after a download

def fake_download(monkeypatch, gguf_file: Path, name: str = "newone.gguf") -> list[tuple]:
    """A downloader that copies the synthetic GGUF into the destination instead of fetching."""
    calls: list[tuple] = []
    monkeypatch.setattr(hf, "gguf_sizes", lambda repo, token: {name: gguf_file.stat().st_size})

    def download(repo, paths, dest, token, cancel, progress):
        calls.append((repo, paths, dest))
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / Path(paths[0]).name
        sparse_copy(gguf_file, out)
        progress(10, 10, out.name)
        return [out]

    monkeypatch.setattr(hf, "download", download)
    return calls


async def download(pilot, app, name: str = "newone.gguf"):
    """Start the fake download from the Discover view."""
    app.screen.set_focus(None)
    await pilot.press("4")
    await pilot.pause(0.1)
    hv = app.views["discover"]
    hv.repo = "org/repo"
    hv.start_download(name)
    return hv


async def test_download_opens_the_picker_and_a_pick_stores_the_preset_and_can_serve(
        ui_env: SimpleNamespace, monkeypatch, gguf_file: Path) -> None:
    fake_hf(monkeypatch, ui_env)
    calls = fake_download(monkeypatch, gguf_file)
    gw = FakeGateway(ui_env.cfg)
    app = make_app(ui_env, gateway=gw)
    async with app.run_test(size=SIZE) as pilot:
        assert await wait_for(pilot, lambda: app.host.ram_total > 0)
        await download(pilot, app)
        assert await wait_for(pilot, lambda: isinstance(app.screen, IntentPicker))
        path = str(ui_env.model_dir / "newone.gguf")
        assert calls and calls[0][2] == ui_env.model_dir and (ui_env.model_dir / "newone.gguf").exists()
        assert await wait_for(pilot, lambda: path in {str(m) for m in app.models})   # rescanned
        assert app.cfg.preset(path) is None                      # nothing stored before the pick
        assert app.screen.current == "chat"                      # the default intent is highlighted
        await pilot.press("down")                                # agent
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: isinstance(app.screen, AfterDownload))
        p = app.cfg.preset(path)
        assert p is not None and p.intent == "agent" and p.source == "estimated"
        assert app.cfg.models[path].active == "agent"
        assert "newone" in plain(app.screen.query_one("#abox Label", Label))
        await pilot.press("s")                                   # Serve
        assert await wait_for(pilot, lambda: ("load", "newone") in gw.calls)
        assert Config.load().preset(path).intent == "agent"


async def test_cancelling_the_picker_leaves_the_model_as_new(
        ui_env: SimpleNamespace, monkeypatch, gguf_file: Path) -> None:
    fake_hf(monkeypatch, ui_env)
    fake_download(monkeypatch, gguf_file)
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        assert await wait_for(pilot, lambda: app.host.ram_total > 0)
        await download(pilot, app)
        assert await wait_for(pilot, lambda: isinstance(app.screen, IntentPicker))
        await pilot.press("escape")
        assert await wait_for(pilot, lambda: not isinstance(app.screen, IntentPicker))
        path = str(ui_env.model_dir / "newone.gguf")
        assert await wait_for(pilot, lambda: path in {str(m) for m in app.models} and not app.scanning)
        assert app.cfg.preset(path) is None and not (ui_env.cfg_dir / "config.json").exists()
        mv = app.views["models"]
        assert await wait_for(pilot, lambda: "New" in str(mv.model_table.get_row(path)[4]))
        assert "without a preset" in plain(app._status)


async def test_done_keeps_the_preset_and_tune_opens_the_tune_screen(
        ui_env: SimpleNamespace, monkeypatch, gguf_file: Path) -> None:
    fake_hf(monkeypatch, ui_env)
    fake_download(monkeypatch, gguf_file)
    gw = FakeGateway(ui_env.cfg)
    app = make_app(ui_env, gateway=gw, runner=FakeRunner(ok(44.0)))
    async with app.run_test(size=SIZE) as pilot:
        assert await wait_for(pilot, lambda: app.host.ram_total > 0)
        await download(pilot, app)
        assert await wait_for(pilot, lambda: isinstance(app.screen, IntentPicker))
        await pilot.press("enter")                               # chat
        assert await wait_for(pilot, lambda: isinstance(app.screen, AfterDownload))
        await pilot.press("escape")                              # Done
        assert await wait_for(pilot, lambda: not isinstance(app.screen, AfterDownload))
        path = str(ui_env.model_dir / "newone.gguf")
        assert app.cfg.preset(path, "chat").source == "estimated" and gw.called("load") == []
        # a second download of another file: Tune
        sparse_copy(ui_env.model_dir / "newone.gguf", ui_env.model_dir / "other.gguf")
        app.offer_new_model(ui_env.model_dir / "other.gguf")
        assert await wait_for(pilot, lambda: isinstance(app.screen, IntentPicker))
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: isinstance(app.screen, AfterDownload))
        await pilot.press("t")
        assert await wait_for(pilot, lambda: isinstance(app.screen, TuneScreen))
        assert await wait_for(pilot, lambda: app.screen.state == "done")
        assert app.cfg.preset(str(ui_env.model_dir / "other.gguf"), "chat").source == "tuned"


async def test_a_download_that_fails_opens_no_picker(
        ui_env: SimpleNamespace, monkeypatch, gguf_file: Path) -> None:
    fake_hf(monkeypatch, ui_env)
    monkeypatch.setattr(hf, "gguf_sizes", lambda repo, token: {"x.gguf": 1})

    def broken(*a, **kw):
        raise hf.HFError("Download failed: boom (partial file kept, will resume).")

    monkeypatch.setattr(hf, "download", broken)
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        hv = await download(pilot, app, "x.gguf")
        assert await wait_for(pilot, lambda: "boom" in plain(hv.query_one("#hf_note", Label)))
        await pilot.pause(0.2)
        assert not isinstance(app.screen, IntentPicker) and not hv.downloading


async def test_an_unreadable_download_is_reported_not_offered(
        ui_env: SimpleNamespace, monkeypatch, tmp_path: Path) -> None:
    fake_hf(monkeypatch, ui_env)
    junk = tmp_path / "junk.gguf"
    junk.write_bytes(b"not a gguf" * 10)
    fake_download(monkeypatch, junk, "junk.gguf")
    app = make_app(ui_env)
    async with app.run_test(size=SIZE) as pilot:
        await download(pilot, app, "junk.gguf")
        assert await wait_for(pilot, lambda: "cannot read the GGUF header" in app._last_failure)
        assert not isinstance(app.screen, IntentPicker)
