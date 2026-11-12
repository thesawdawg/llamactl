"""HFView: search Hugging Face GGUF repos and download into the first model dir."""
from __future__ import annotations

import threading
from pathlib import Path

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import DataTable, Input, Label, ProgressBar

from rich.text import Text

from .. import discover, hf
from ..fmt import FITS_STYLE, human, params_str
from ..gguf import GGUFError, read_model_info
from ..intents import by_key
from ..screens.confirm import ConfirmScreen
from ..screens.prompt import PromptScreen
from .base import BaseView

TOKEN_HINT = ("Create one at huggingface.co/settings/tokens (read access). Needed for gated "
              "or private models; also accept the model's terms on its page. "
              "Leave empty to remove the saved token.")


class HFView(BaseView):
    """Search box, repo list (left), file list (right), download progress.

    The file list judges each file by its size alone (weights only, the KV cache is unknown
    before the download), sorts the largest first and marks the best fit for the default intent.
    After a download the library is rescanned and the intent picker opens for the new model.
    """

    BINDINGS = [
        Binding("slash", "focus_search", "Search"),
        Binding("k", "token", "Set token"),
        Binding("x", "cancel", "Cancel download"),
    ]
    DEFAULT_CSS = """
    #hf_q { margin: 0 1; }
    #hf_body { height: 1fr; }
    #repos { height: 1fr; }
    #files { height: 2fr; }
    #hf_bar { margin: 0 1; display: none; }
    #hf_note { color: $text-muted; padding: 0 1; height: auto; width: 100%; }
    #hf_legend { color: $text-muted; padding: 0 1; height: auto; width: 100%; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.token = hf.load_token()
        self.repos: list[dict] = []
        self.files: list[dict] = []              # as listed by hf.list_gguf, in table order
        self.fits: list[discover.FileFit] = []   # the same files, judged and sorted
        self.repo = ""
        self._repo_blocked = ""                # why the selected repo's arch is blocked, or ""
        self._blocked: dict[str, str] = {}     # repo id -> blocking reason, filled in fill_repos
        self._shown_host: tuple = ()
        self.cancel_ev = threading.Event()
        self.downloading = False

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search GGUF models on Hugging Face (Enter), e.g. qwen3 7b",
                    id="hf_q")
        with Vertical(id="hf_body"):
            yield DataTable(id="repos", cursor_type="row")
            yield DataTable(id="files", cursor_type="row")
        yield Label("", id="hf_legend")
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
        """Judge the files again for the fresh host snapshot (the free memory moved or arrived)."""
        if self.files and (self._host_sig(host) != self._shown_host):
            self.fill_files(keep=True)
        self.refresh_status()

    @staticmethod
    def _host_sig(host) -> tuple:
        """What the verdicts depend on: devices, their totals and the RAM."""
        return (tuple((d.id, d.total) for d in host.gpus), host.ram_total)

    def _intent_label(self) -> str:
        """Label of the default intent."""
        return by_key[self.app.cfg.default_intent].label if self.app.cfg.default_intent in by_key \
            else "Chat"

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
        self.app.ui(self.refresh_status, msg)

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
        self.app.ui(self.fill_repos)

    def fill_repos(self) -> None:
        t = self.query_one("#repos", DataTable)
        a = self.app
        t.clear()
        for r in self.repos:
            g = r.get("gguf") or {}
            arch = g.get("architecture", "")
            s = a.support.check(arch, a.build) if arch else None
            arch_cell: str | Text = arch
            if s is not None and s.blocks:
                self._blocked[r["id"]] = s.reason()
                arch_cell = Text(f"{arch} ⚠ {s.label()}", style="red")
            else:
                self._blocked.pop(r["id"], None)
            t.add_row(r["id"] + (" [gated]" if r.get("gated") else ""),
                      params_str(g.get("total")), arch_cell,
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
            self._repo_blocked = self._blocked.get(self.repo, "")
            self.refresh_status(f"Listing files of {self.repo}...")
            self.load_files(self.repo)
        elif event.data_table.id == "files" and self.files:
            path = self.files[event.cursor_row]["path"]
            if not self._repo_blocked:
                return self.start_download(path)

            def done(ok: bool | None) -> None:
                """Download the file after the user confirmed the blocked architecture."""
                if ok:
                    self.start_download(path)

            self.app.push_screen(
                ConfirmScreen(f"{self.repo}: {self._repo_blocked}\nDownload anyway?"), done)

    @work(thread=True, exclusive=True, group="hffiles")
    def load_files(self, repo: str) -> None:
        try:
            self.files = hf.list_gguf(repo, self.token)
        except hf.HFError as e:
            return self.say_status(f"[red]{e}[/red]")
        self.app.ui(self.fill_files)

    def fill_files(self, keep: bool = False) -> None:
        """Fill the file table: largest first, verdicts weights only, the best fit marked.

        Args:
            keep: Keep the cursor on the same file (a redraw for new hardware data).
        """
        t = self.query_one("#files", DataTable)
        host, pct = self.app.host, self.app.cfg.headroom_pct
        by_path = {f["path"]: f for f in self.files}
        was = self.fits[t.cursor_row].path if keep and t.row_count and self.fits else None
        self.fits = discover.rank_files(self.files, host, pct)
        self.files = [by_path[x.path] for x in self.fits]      # row i of the table is files[i]
        self._shown_host = self._host_sig(host)
        t.clear()
        for x in self.fits:
            t.add_row(Text(x.path, style="bold" if x.best else ""), x.quant, human(x.size), x.parts,
                      self._fit_cell(x, discover.host_ready(host)))
        best = next((i for i, x in enumerate(self.fits) if x.best), None)
        want = next((i for i, x in enumerate(self.fits) if x.path == was), best or 0)
        if self.fits:
            t.move_cursor(row=want)
        self.query_one("#hf_legend", Label).update(discover.legend(host, self._intent_label()))
        if not keep:
            self.refresh_status(f"{len(self.files)} GGUF file(s) in {self.repo}"
                                if self.files else f"No GGUF files in {self.repo}")
            t.focus()

    @staticmethod
    def _fit_cell(x: discover.FileFit, ready: bool) -> Text:
        """The Fits cell: the weights-only verdict, plus the best-fit marker."""
        if not ready:
            return Text("detecting...", style="dim")
        t = Text(x.label, style=FITS_STYLE.get(x.verdict, ""))
        if x.best:
            t.append(f"  \u2605 {discover.BEST_MARK}", style="bold green")
        return t

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
            self.app.ui(bar.update, total=total or None, progress=done)
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
            self.app.ui(self._post_download, out[0], info)
        except hf.HFError as e:
            self.say_status(f"[red]{e}[/red]")
        finally:
            self.downloading = False
            self.app.ui(setattr, bar, "display", False)

    def _post_download(self, path: Path, info) -> None:
        """Rescan so the new file shows up in Models, then ask what the new model is for.

        Args:
            path: The downloaded model (first shard).
            info: Its parsed header, None when it could not be read.
        """
        if info is not None:
            self.app.infos[str(path)] = info
        self.app.rescan()
        self.app.offer_new_model(path)
