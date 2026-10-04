"""HFView: search Hugging Face GGUF repos and download into the first model dir."""
from __future__ import annotations

import threading
from pathlib import Path

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import DataTable, Input, Label, ProgressBar

from .. import hf
from ..config import Profile
from ..fmt import human, params_str
from ..gguf import GGUFError, read_model_info
from ..recommend import estimate_budget, placeholder_info, verdict
from ..screens.prompt import PromptScreen
from .base import BaseView

TOKEN_HINT = ("Create one at huggingface.co/settings/tokens (read access). Needed for gated "
              "or private models; also accept the model's terms on its page. "
              "Leave empty to remove the saved token.")


class HFView(BaseView):
    """Search box, repo list (left), file list (right), download progress."""

    BINDINGS = [
        Binding("slash", "focus_search", "Search"),
        Binding("k", "token", "Set token"),
        Binding("x", "cancel", "Cancel download"),
    ]
    CSS = """
    #hf_q { margin: 0 1; }
    #hf_body { height: 1fr; }
    #repos { width: 3fr; }
    #files { width: 2fr; }
    #hf_bar { margin: 0 1; display: none; }
    #hf_note { color: $text-muted; padding: 0 1; height: auto; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.token = hf.load_token()
        self.repos: list[dict] = []
        self.files: list[dict] = []
        self.repo = ""
        self.cancel_ev = threading.Event()
        self.downloading = False

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search GGUF models on Hugging Face (Enter), e.g. qwen3 7b",
                    id="hf_q")
        with Horizontal(id="hf_body"):
            yield DataTable(id="repos", cursor_type="row")
            yield DataTable(id="files", cursor_type="row")
        yield ProgressBar(id="hf_bar", total=100, show_eta=True)
        yield Label("", id="hf_note")

    def on_mount(self) -> None:
        self.query_one("#repos", DataTable).add_columns(
            "Repo (Enter = list files)", "Params", "Arch", "Max ctx", "Downloads", "Likes", "Updated")
        t = self.query_one("#files", DataTable)
        for label, key in [("File (Enter = download)", "file"), ("Quant", "quant"),
                           ("Size", "size"), ("Parts", "parts"), ("Fits", "fits")]:
            t.add_column(label, key=key)
        self.query_one("#hf_bar", ProgressBar).display = False
        self.refresh_status()

    def focus_primary(self) -> None:
        self.query_one("#hf_q", Input).focus()

    def on_host(self, host) -> None:
        """Refresh the Fits cells for the fresh host snapshot."""
        t = self.query_one("#files", DataTable)
        for i, key in enumerate(t.rows):
            if i < len(self.files):
                t.update_cell(key, "fits", self._fits(self.files[i]["size"]))
        self.refresh_status()

    def _fits(self, size: int) -> str:
        """Weights-only verdict for a remote file size."""
        b = estimate_budget(placeholder_info(size), Profile(), self.app.host)
        return verdict(b, self.app.host, self.app.cfg.headroom_pct)

    def refresh_status(self, msg: str = "") -> None:
        tok = "token set" if self.token else "no token (public models only)"
        dest = self.app.cfg.model_dirs[0] if self.app.cfg.model_dirs else "?"
        host = self.app.host
        summary = ("detecting host..." if host.ram_total == 0 and not host.gpus
                   else host.summary())
        self.query_one("#hf_note", Label).update(
            f"{msg}  [dim]{summary} | {tok} | saves to {dest} | "
            f"/ search, k token, x cancel[/dim]".lstrip())

    def say_status(self, msg: str) -> None:
        """Thread-safe status update."""
        self.app.call_from_thread(self.refresh_status, msg)

    def action_focus_search(self) -> None:
        self.query_one("#hf_q", Input).focus()

    def action_token(self) -> None:
        def done(res: str | None) -> None:
            if res is not None:
                self.token = res or None
                hf.save_token(res)
                self.refresh_status("Token saved." if self.token else "Token removed.")

        self.app.push_screen(PromptScreen("Hugging Face access token", TOKEN_HINT,
                                          self.token or "", password=True), done)

    def action_cancel(self) -> None:
        self.cancel_ev.set()

    @on(Input.Submitted, "#hf_q")
    def _search(self, event: Input.Submitted) -> None:
        self.refresh_status("Searching...")
        self.do_search(event.value.strip())

    @work(thread=True, exclusive=True, group="hfsearch")
    def do_search(self, query: str) -> None:
        try:
            self.repos = hf.search(query, self.token)
        except hf.HFError as e:
            return self.say_status(f"[red]{e}[/red]")
        self.app.call_from_thread(self.fill_repos)

    def fill_repos(self) -> None:
        t = self.query_one("#repos", DataTable)
        t.clear()
        for r in self.repos:
            g = r.get("gguf") or {}
            t.add_row(r["id"] + (" [gated]" if r.get("gated") else ""),
                      params_str(g.get("total")), g.get("architecture", ""),
                      f"{g['context_length']:,}" if g.get("context_length") else "",
                      f"{r.get('downloads', 0):,}", r.get("likes", 0),
                      r.get("lastModified", "")[:10])
        self.refresh_status(f"{len(self.repos)} result(s)")
        t.focus()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "files" and self.files and not self.downloading:
            f = self.files[event.cursor_row]
            size = human(f["size"])
            self.refresh_status(f"{f['path']}: {size} weights only (KV unknown until downloaded)")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "repos" and self.repos:
            self.repo = self.repos[event.cursor_row]["id"]
            self.refresh_status(f"Listing files of {self.repo}...")
            self.load_files(self.repo)
        elif event.data_table.id == "files" and self.files:
            self.start_download(self.files[event.cursor_row]["path"])

    @work(thread=True, exclusive=True, group="hffiles")
    def load_files(self, repo: str) -> None:
        try:
            self.files = hf.list_gguf(repo, self.token)
        except hf.HFError as e:
            return self.say_status(f"[red]{e}[/red]")
        self.app.call_from_thread(self.fill_files)

    def fill_files(self) -> None:
        t = self.query_one("#files", DataTable)
        t.clear()
        for f in self.files:
            t.add_row(f["path"], f["quant"], human(f["size"]), f["parts"], self._fits(f["size"]))
        self.refresh_status(f"{len(self.files)} GGUF file(s) in {self.repo}"
                            if self.files else f"No GGUF files in {self.repo}")
        t.focus()

    def start_download(self, path: str) -> None:
        if self.downloading:
            return self.refresh_status("A download is already running.")
        if not self.app.cfg.model_dirs:
            return self.refresh_status("[red]No model dir configured (Settings view).[/red]")
        self.downloading = True
        self.cancel_ev.clear()
        bar = self.query_one("#hf_bar", ProgressBar)
        bar.display = True
        bar.update(total=100, progress=0)
        self.download_worker(self.repo, path, Path(self.app.cfg.model_dirs[0]).expanduser())

    @work(thread=True, group="hfdown")
    def download_worker(self, repo: str, path: str, dest: Path) -> None:
        bar = self.query_one("#hf_bar", ProgressBar)

        def progress(done: int, total: int, name: str) -> None:
            self.app.call_from_thread(bar.update, total=total or None, progress=done)
            self.say_status(f"Downloading {name}: {human(done)} / {human(total)}")

        try:
            group = hf.shard_group(path, list(hf.gguf_sizes(repo, self.token)))
            out = hf.download(repo, group, dest, self.token, self.cancel_ev, progress)
            info = None
            try:
                info = read_model_info(out[0])
                self.app.info_cache.put(info)
            except (GGUFError, OSError) as e:
                self.say_status(f"[yellow]Downloaded but could not read header: {e}[/yellow]")
            extra = f" (+{len(out) - 1} shards)" if len(out) > 1 else ""
            self.say_status(f"[green]Done: {out[0].name}{extra}.[/green]")
            self.app.call_from_thread(self._post_download, out[0], info)
        except hf.HFError as e:
            self.say_status(f"[red]{e}[/red]")
        finally:
            self.downloading = False
            self.app.call_from_thread(setattr, bar, "display", False)

    def _post_download(self, path: Path, info) -> None:
        """Rescan models and offer a recommendation for the new file."""
        mv = self.app.views["models"]
        if info is not None:
            mv.infos[str(path)] = info
            mv.rescan()
            mv.open_recommendation(str(path))
        else:
            mv.rescan()
