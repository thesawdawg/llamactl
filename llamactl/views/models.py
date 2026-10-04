"""ModelsView: model table with metadata, budget, Fits verdict and actions."""
from __future__ import annotations

import subprocess
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, DataTable, Label, Static

from .. import hf
from ..command import build_command, command_string
from ..config import Profile, discover_models
from ..fmt import human, params_str
from ..gguf import GGUFError, ModelInfo, read_model_info
from ..hostinfo import Host
from ..recommend import estimate_budget, verdict
from ..screens.profile import ProfileEditor
from ..widgets.budget import BudgetPanel
from .base import BaseView

FITS_STYLE = {"GPU": "green", "GPU+CPU": "yellow", "CPU": "yellow", "NO": "red"}


class ModelsView(BaseView):
    """Model table (left) plus metadata/budget/command detail pane (right)."""

    BINDINGS = [
        Binding("enter", "launch_server", "Launch server"),
        Binding("l", "launch_server", "Launch", show=False),
        Binding("c", "launch_cli", "CLI chat"),
        Binding("e", "edit_profile", "Edit profile"),
        Binding("d", "edit_default", "Defaults"),
        Binding("f2", "recommend", "Recommend"),
        Binding("R", "recommend", "Recommend", show=False),
        Binding("delete", "delete_profile", "Delete profile"),
    ]
    CSS = """
    #models_row { height: 1fr; }
    #model_table { width: 3fr; }
    #detail { width: 2fr; padding: 0 1; }
    #detail Static { margin-bottom: 1; }
    #m_buttons { height: auto; }
    #m_buttons Button { margin-right: 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.models: list[Path] = []
        self.infos: dict[str, ModelInfo] = {}

    def compose(self) -> ComposeResult:
        with Horizontal(id="models_row"):
            yield DataTable(id="model_table", cursor_type="row")
            with VerticalScroll(id="detail"):
                yield Static("", id="meta")
                yield BudgetPanel()
                yield Static("", id="prof_summary")
                with Horizontal(id="m_buttons"):
                    yield Button("Launch", id="m_launch", variant="primary")
                    yield Button("CLI chat", id="m_cli")
                    yield Button("Edit", id="m_edit")
                    yield Button("Recommend", id="m_rec")
                    yield Button("Delete profile", id="m_del")

    def on_mount(self) -> None:
        t = self.model_table
        for label, key in (("Model", "m"), ("Size", "size"), ("Params", "params"),
                           ("Quant", "quant"), ("Fits", "fits"), ("Ctx", "ctx"),
                           ("Profile", "prof")):
            t.add_column(label, key=key)
        self.rescan()

    @property
    def model_table(self) -> DataTable:
        return self.query_one("#model_table", DataTable)

    def selected_model(self) -> str | None:
        """Path of the highlighted model row."""
        rows = list(self.model_table.rows)  # insertion order == cursor order
        if not rows:
            return None
        return str(rows[min(self.model_table.cursor_row, len(rows) - 1)].value)

    def on_activate(self) -> None:
        self.update_detail()
        super().on_activate()

    def focus_primary(self) -> None:
        self.model_table.focus()

    def on_host(self, host: Host) -> None:
        """Recompute the Fits cells and the detail pane for a fresh host snapshot."""
        for key, row in self.model_table.rows.items():
            model = str(key.value)
            if model in self.infos:
                self.model_table.update_cell(key, "fits",
                                             self._fits(model, self.infos[model]))
        self.update_detail()

    def _fits(self, model: str, info: ModelInfo) -> Text:
        """Coloured verdict for a model under its current profile."""
        v = verdict(estimate_budget(info, self.app.cfg.profile_for(model), self.app.host),
                    self.app.host)
        return Text(v, style=FITS_STYLE.get(v, ""))

    def _profile_tag(self, model: str) -> str:
        """'custom', 'default' or 'New' for the Profile column."""
        if self.app.cfg.has_profile(model):
            return "custom"
        return "default" if model in self.app.cfg.seen else "New"

    def _add_row(self, p: Path, info: ModelInfo | None) -> None:
        """Insert or update one table row; Fits/Params fill in when info arrives."""
        key = str(p)
        prof = self.app.cfg.profile_for(key)
        ctx = (str(prof.ctx_size) if prof.ctx_size
               else f"{info.ctx_train} (max)" if info else "auto")
        row = (p.name, f"{p.stat().st_size / 1e9:.2f} GB",
               params_str(info.parameter_count) if info else "",
               hf.quant_of(p.name), self._fits(key, info) if info else "...",
               ctx, self._profile_tag(key))
        if key in self.model_table.rows:
            r = self.model_table.rows[key]
            for col, v in zip(("m", "size", "params", "quant", "fits", "ctx", "prof"), row):
                self.model_table.update_cell(key, col, v)
        else:
            self.model_table.add_row(*row, key=key)

    @work(thread=True, exclusive=True, group="scan")
    def rescan(self) -> None:
        """Scan model dirs, then parse headers progressively in this worker."""
        models = discover_models(self.app.cfg.model_dirs)
        infos = {}
        for p in models:
            info = self.app.info_cache.get(p)
            if info is None:
                try:
                    info = read_model_info(p)
                    self.app.info_cache.put(info)
                except (GGUFError, OSError):
                    info = None
            if info:
                infos[str(p)] = info
            self.app.call_from_thread(self._add_row, p, info)
        self.app.call_from_thread(self._scan_done, models, infos)

    def _scan_done(self, models: list[Path], infos: dict[str, ModelInfo]) -> None:
        self.models = models
        self.infos = infos
        self.say(f"{len(models)} model(s) in {', '.join(self.app.cfg.model_dirs)}")
        self.update_detail()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table is self.model_table:
            if t := getattr(self, "_hl", None):
                t.stop()
            self._hl = self.set_timer(0.1, self.update_detail)

    def update_detail(self) -> None:
        """Fill the right pane for the highlighted model."""
        model = self.selected_model()
        if model is None:
            return
        info = self.infos.get(model)
        meta = self.query_one("#meta", Static)
        if info is None:
            meta.update("Reading model header...")
            return
        is_new = self._profile_tag(model) == "New"
        meta.update(
            f"[b]{info.name}[/b]\n"
            f"arch {info.architecture} | {info.n_layer} layers | {info.n_head_kv} kv heads | "
            f"trained ctx {info.ctx_train} | experts {info.expert_count} | "
            f"file type {info.file_type} | shards {info.sharded}"
            + ("\n[yellow]No profile yet. Press F2 for a recommendation.[/]" if is_new else ""))
        prof = self.app.cfg.profile_for(model)
        self.query_one(BudgetPanel).update_budget(
            estimate_budget(info, prof, self.app.host), self.app.host)
        changed = prof.non_default()
        cmd = command_string(build_command(self.app.cfg, model, prof, "server",
                                           prof.port or 0))
        if not prof.port:
            cmd = cmd.replace(f"--port 0", "--port <auto>")
        self.query_one("#prof_summary", Static).update(
            ("\n".join(f"{k}={v}" for k, v in changed.items()) or "(all defaults)") + "\n\n" + cmd)

    def _launch(self, model: str, cli: bool) -> None:
        """Launch a server (or CLI chat) for the model and mark it seen.

        Args:
            model: Model path.
            cli: True for llama-cli, False for llama-server.
        """
        self.app.cfg.seen.mark_seen(model)
        if cli:
            prof = self.app.cfg.profile_for(model)
            cmd = build_command(self.app.cfg, model, prof, "cli", 0)
            with self.app.suspend():
                subprocess.run(cmd)
            return
        self.app.sessions.launch(model)
        self.app.switch_view("sessions")

    def _edit(self, model: str | None, auto_recommend: bool) -> None:
        """Open the profile editor, parsing the header first when needed.

        Args:
            model: Model path, or None for the default profile.
            auto_recommend: Apply the recommendation once the editor opens.
        """
        def done(new: Profile | None) -> None:
            if new is not None:
                if model:
                    self.app.cfg.profiles[model] = new
                else:
                    self.app.cfg.default_profile = new
                self.app.cfg.save()
            if model:
                self.app.cfg.seen.mark_seen(model)
            key = str(model) if model else ""
            if key and key in self.model_table.rows:
                self._add_row(Path(key), self.infos.get(key))
            self.update_detail()

        if model is None:
            self.app.push_screen(
                ProfileEditor(self.app.cfg, self.app.cfg.default_profile, self.app.host,
                              None, "Default profile"), done)
            return
        info = self.infos.get(model)
        if info is None:
            return self.say("Still reading the model header - try again.")
        prof = self.app.cfg.profile_for(model)
        self.app.push_screen(
            ProfileEditor(self.app.cfg, prof, self.app.host, info, Path(model).name,
                          auto_recommend=auto_recommend), done)

    def action_launch_server(self) -> None:
        if m := self.selected_model():
            self._launch(m, cli=False)

    def action_launch_cli(self) -> None:
        if m := self.selected_model():
            self._launch(m, cli=True)

    def action_edit_profile(self) -> None:
        if m := self.selected_model():
            self._edit(m, auto_recommend=False)

    def action_edit_default(self) -> None:
        self._edit(None, auto_recommend=False)

    def action_recommend(self) -> None:
        if m := self.selected_model():
            self._edit(m, auto_recommend=True)

    def action_delete_profile(self) -> None:
        m = self.selected_model()
        if m and self.app.cfg.profiles.pop(m, None) is not None:
            self.app.cfg.save()
            self._add_row(Path(m), self.infos.get(m))
            self.update_detail()
            self.say(f"Profile for {Path(m).name} deleted.")

    @on(Button.Pressed, "#m_launch")
    def _b_launch(self) -> None:
        self.action_launch_server()

    @on(Button.Pressed, "#m_cli")
    def _b_cli(self) -> None:
        self.action_launch_cli()

    @on(Button.Pressed, "#m_edit")
    def _b_edit(self) -> None:
        self.action_edit_profile()

    @on(Button.Pressed, "#m_rec")
    def _b_rec(self) -> None:
        self.action_recommend()

    @on(Button.Pressed, "#m_del")
    def _b_del(self) -> None:
        self.action_delete_profile()
