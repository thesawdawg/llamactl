"""Hugging Face browser: search GGUF repos and download files into the first model dir."""
from __future__ import annotations

import threading
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Label, ProgressBar

from . import hf, hostinfo
from .config import Config
from .fmt import human, params_str
from .forms import FormScreen


class HFScreen(Screen):
    """Search box, repo list (left), file list (right), download progress."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("k", "token", "Set token"),
        Binding("x", "cancel", "Cancel download"),
        Binding("slash", "focus_search", "Search"),
    ]
    CSS = """
    #hf_body { height: 1fr; }
    #repos { width: 3fr; }
    #files { width: 2fr; }
    #hf_status { height: auto; padding: 0 1; }
    #hf_bar { margin: 0 1; display: none; }
    """

    def __init__(self, cfg: Config, host: hostinfo.Host) -> None:
        super().__init__()
        self.cfg = cfg
        self.token = hf.load_token()
        self.repos: list[dict] = []
        self.files: list[dict] = []
        self.repo = ""
        self.host = host
        self.cancel_ev = threading.Event()
        self.downloading = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="Search GGUF models on Hugging Face (Enter), e.g. qwen3 7b", id="q")
        with Horizontal(id="hf_body"):
            yield DataTable(id="repos", cursor_type="row")
            yield DataTable(id="files", cursor_type="row")
        yield ProgressBar(id="hf_bar", total=100, show_eta=True)
        yield Label("", id="hf_status")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = "Hugging Face"
        self.query_one("#repos", DataTable).add_columns("Repo (Enter = list files)", "Params", "Arch", "Max ctx", "Downloads", "Likes", "Updated")
        self.query_one("#files", DataTable).add_columns("File (Enter = download)", "Quant", "Size", "Parts", "Fits?")
        self.query_one("#q", Input).focus()
        self.refresh_status()

    def refresh_status(self, msg: str = "") -> None:
        tok = "token set" if self.token else "no token (public models only)"
        dest = self.cfg.model_dirs[0] if self.cfg.model_dirs else "?"
        self.query_one("#hf_status", Label).update(f"{msg}\n[dim]{self.host.summary()}\n{tok} | saves to {dest} | / search, k token, x cancel[/dim]".lstrip())

    def say(self, msg: str) -> None:
        """Thread-safe status update."""
        self.app.call_from_thread(self.refresh_status, msg)

    def action_focus_search(self) -> None:
        self.query_one("#q", Input).focus()

    def action_token(self) -> None:
        field = [("token", "Hugging Face access token", self.token or "",
                  "Create one at huggingface.co/settings/tokens (read access). Needed for gated or private models; also accept the model's terms on its page. Leave empty to remove the saved token.")]

        def done(res: dict | None) -> None:
            if res is not None:
                self.token = res["token"] or None
                hf.save_token(res["token"])
                self.refresh_status("Token saved." if self.token else "Token removed.")

        self.app.push_screen(FormScreen("Hugging Face token", field), done)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.refresh_status("Searching...")
        self.do_search(event.value.strip())

    @work(thread=True, exclusive=True, group="search")
    def do_search(self, query: str) -> None:
        try:
            self.repos = hf.search(query, self.token)
        except hf.HFError as e:
            return self.say(f"[red]{e}[/red]")
        self.app.call_from_thread(self.fill_repos)

    def fill_repos(self) -> None:
        t = self.query_one("#repos", DataTable)
        t.clear()
        for r in self.repos:
            g = r.get("gguf") or {}
            t.add_row(r["id"] + (" [gated]" if r.get("gated") else ""), params_str(g.get("total")), g.get("architecture", ""),
                      f"{g['context_length']:,}" if g.get("context_length") else "", f"{r.get('downloads', 0):,}",
                      r.get("likes", 0), r.get("lastModified", "")[:10])
        self.refresh_status(f"{len(self.repos)} result(s)")
        t.focus()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "files" and self.files and not self.downloading:
            f = self.files[event.cursor_row]
            self.refresh_status(f"{f['path']}: {hostinfo.estimate(self.host, f['size'])[1]}")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "repos" and self.repos:
            self.repo = self.repos[event.cursor_row]["id"]
            self.refresh_status(f"Listing files of {self.repo}...")
            self.load_files(self.repo)
        elif event.data_table.id == "files" and self.files:
            self.start_download(self.files[event.cursor_row]["path"])

    @work(thread=True, exclusive=True, group="files")
    def load_files(self, repo: str) -> None:
        try:
            self.files = hf.list_gguf(repo, self.token)
        except hf.HFError as e:
            return self.say(f"[red]{e}[/red]")
        self.app.call_from_thread(self.fill_files)

    def fill_files(self) -> None:
        t = self.query_one("#files", DataTable)
        t.clear()
        for f in self.files:
            t.add_row(f["path"], f["quant"], human(f["size"]), f["parts"], hostinfo.estimate(self.host, f["size"])[0])
        self.refresh_status(f"{len(self.files)} GGUF file(s) in {self.repo}" if self.files else f"No GGUF files in {self.repo}")
        t.focus()

    def start_download(self, path: str) -> None:
        if self.downloading:
            return self.refresh_status("A download is already running.")
        if not self.cfg.model_dirs:
            return self.refresh_status("[red]No model dir configured (press g on main screen).[/red]")
        self.downloading = True
        self.cancel_ev.clear()
        bar = self.query_one("#hf_bar", ProgressBar)
        bar.display = True
        bar.update(total=100, progress=0)
        self.download_worker(self.repo, path, Path(self.cfg.model_dirs[0]).expanduser())

    def action_cancel(self) -> None:
        self.cancel_ev.set()

    @work(thread=True, group="download")
    def download_worker(self, repo: str, path: str, dest: Path) -> None:
        bar = self.query_one("#hf_bar", ProgressBar)

        def progress(done: int, total: int, name: str) -> None:
            self.app.call_from_thread(bar.update, total=total or None, progress=done)
            self.say(f"Downloading {name}: {human(done)} / {human(total)}")

        try:
            group = hf.shard_group(path, list(hf.gguf_sizes(repo, self.token)))
            out = hf.download(repo, group, dest, self.token, self.cancel_ev, progress)
            self.say(f"[green]Done: {out[0].name}{f' (+{len(out) - 1} shards)' if len(out) > 1 else ''}. Back (Esc) to launch it.[/green]")
        except hf.HFError as e:
            self.say(f"[red]{e}[/red]")
        finally:
            self.downloading = False
            self.app.call_from_thread(setattr, bar, "display", False)
