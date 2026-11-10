"""ModelsView: every model with what it suits, its preset, and per-intent detail and actions."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.timer import Timer
from textual.widgets import DataTable, Static

from .. import hf, planner
from ..command import build_command, command_string
from ..config import Preset
from ..fmt import FITS_STYLE, human, params_str
from ..gguf import ModelInfo
from ..hardware import Host
from ..intents import by_key
from ..modelrows import (IntentRow, PresetState, best_for, default_intent_key, intent_rows,
                         preset_state)
from ..planner import Plan
from ..screens.confirm import ConfirmScreen
from ..screens.intent_picker import IntentPicker
from ..screens.plan import EDIT, SAVED, SERVE, TUNE, PlanScreen
from ..screens.preset_editor import PresetEditor
from ..widgets.budget import BudgetPanel
from .base import BaseView

COLUMNS = (("Model (alias)", "model"), ("Size", "size"), ("Quant", "quant"),
           ("Best for", "best"), ("Preset", "preset"), ("Measured", "measured"))
INTENT_COLUMNS = (("Intent", "intent"), ("Verdict", "verdict"), ("Ctx", "ctx"), ("KV", "kv"),
                  ("Needs / usable", "mem"), ("tok/s", "tps"), ("Preset", "preset"))
SOURCE_STYLE = {"tuned": "green", "estimated": "dim", "edited": "cyan"}
DEBOUNCE = 0.1          # seconds: detail pane waits for the cursor to settle; plans for the scan
GB = 1e9
EDITED = "edited"     # Preset.source of a hand-edited preset


def best_text(plans: dict[str, Plan] | None, readable: bool) -> Text:
    """The "Best for" cell: the intents with the best verdict, in that verdict's colour."""
    if not readable:
        return Text("unreadable", style="red")
    if plans is None:
        return Text("...", style="dim")
    b = best_for(plans)
    return Text(b.text() if b.keys else "-", style=FITS_STYLE.get(b.verdict, ""))


def preset_text(st: PresetState) -> Text:
    """The "Preset" cell: ``New``, or the active preset's name, source and a stale flag."""
    if st.new:
        return Text("New", style="bold cyan")
    t = Text(f"{st.name} ")
    t.append(st.source, style=SOURCE_STYLE.get(st.source, ""))
    if st.stale:
        t.append(" stale", style="bold yellow")
    if st.count > 1:
        t.append(f" (+{st.count - 1})", style="dim")
    return t


def intent_cells(r: IntentRow) -> tuple:
    """Cells of one row of the per-intent table."""
    source = Text(r.source or "-", style=SOURCE_STYLE.get(r.source, "dim"))
    if r.stale:
        source.append(" stale", style="bold yellow")
    if r.active:
        source.append(" *", style="bold")
    return (Text(r.label, style="bold" if r.active else ""),
            Text(r.verdict, style=FITS_STYLE.get(r.verdict, "")),
            f"{r.ctx_per_slot} x{r.slots}", r.kv, f"{r.needs / GB:.1f} / {r.usable / GB:.1f} GB",
            f"{r.tps:.1f}" if r.tps else "-", source)


class ModelTable(DataTable):
    """The model table; Enter serves (the footer says so, which a bare DataTable would hide)."""

    BINDINGS = [Binding("enter", "select_cursor", "Serve")]


class IntentTable(DataTable):
    """The per-intent table; Enter serves that intent."""

    BINDINGS = [Binding("enter", "select_cursor", "Serve intent")]


class ModelsView(BaseView):
    """Model table (left); metadata, per-intent rows, budget and command (right)."""

    BINDINGS = [
        Binding("s", "serve_standalone", "Standalone"),
        Binding("t", "tune", "Tune"),
        Binding("p", "plan", "Plan"),
        Binding("e", "edit", "Edit preset"),
        Binding("a", "activate", "Set active"),
        Binding("delete", "delete_preset", "Delete preset"),
        Binding("D", "edit_default", "Defaults"),
    ]
    DEFAULT_CSS = """
    ModelsView { height: 1fr; }
    #m_top { height: auto; max-height: 45%; }
    #model_table { height: auto; max-height: 100%; }
    #m_empty { color: $text-muted; padding: 1 2; height: auto; }
    #m_bottom { height: 1fr; }
    #detail { width: 3fr; padding: 0 1; }
    #detail Static { margin-bottom: 1; height: auto; }
    #intent_table { height: auto; max-height: 12; margin-bottom: 1; }
    #m_side { width: 2fr; padding: 0 1; }
    #m_command { color: $text-muted; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.plans: dict[str, dict[str, Plan]] = {}
        self._intent = ""                    # intent the actions apply to
        self._shown = ""                     # model the detail pane shows
        self._host_sig: tuple = ()
        self._unreadable: set[str] = set()
        self._detail_timer: Timer | None = None
        self._plan_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        """Model table on top; metadata, intent table and command below; budget on the side."""
        with Vertical():
            with VerticalScroll(id="m_top"):
                yield ModelTable(id="model_table", cursor_type="row")
                yield Static("", id="m_empty")
            with Horizontal(id="m_bottom"):
                with VerticalScroll(id="detail"):
                    yield Static("", id="meta")
                    yield IntentTable(id="intent_table", cursor_type="row")
                    yield Static("", id="m_warnings")
                    yield Static("", id="m_command")
                with VerticalScroll(id="m_side"):
                    yield BudgetPanel()

    def on_mount(self) -> None:
        """Add the columns and the rows of the models known so far; start planning."""
        for label, key in COLUMNS:
            self.model_table.add_column(label, key=key)
        for label, key in INTENT_COLUMNS:
            self.query_one("#intent_table", DataTable).add_column(label, key=key)
        for p in self.app.models:
            self._row(p)
        self._empty_state()
        self.plan_async()

    # -- accessors ---------------------------------------------------------------------

    @property
    def model_table(self) -> DataTable:
        """The model table."""
        return self.query_one("#model_table", DataTable)

    @property
    def intent_table(self) -> DataTable:
        """The per-intent table of the detail pane."""
        return self.query_one("#intent_table", DataTable)

    def selected_model(self) -> str | None:
        """Path of the highlighted model row, None when there are none."""
        t = self.model_table
        if not t.row_count:
            return None
        return str(list(t.rows)[min(t.cursor_row, t.row_count - 1)].value)

    def selected_intent(self) -> str:
        """Intent the actions apply to: the highlighted intent row, else the model's default."""
        model = self.selected_model()
        plans = self.plans.get(model or "")
        if not plans:
            return ""
        return self._intent if self._intent in plans else default_intent_key(
            self.app.cfg.models.get(model or ""), plans, self.app.cfg.default_intent)

    def focus_primary(self) -> None:
        """Focus the model table."""
        self.model_table.focus()

    def select(self, path: str) -> None:
        """Move the cursor to a model's row and focus the table.

        Args:
            path: Model path.
        """
        if path in self.model_table.rows:
            self.model_table.move_cursor(row=self.model_table.get_row_index(path))
            self.model_table.focus()

    # -- rows --------------------------------------------------------------------------

    def _state(self, path: str) -> PresetState:
        """Active preset state of a model for the table."""
        a = self.app
        return preset_state(a.cfg.models.get(path), Path(path), a.calibration, a.build)

    def _cells(self, path: str) -> tuple:
        """Cells of one model row."""
        a = self.app
        info = a.infos.get(path)
        st = self._state(path)
        size = info.size if info else _size(Path(path))
        return (a.alias_of(path), human(size) if size else "-", hf.quant_of(Path(path).name),
                best_text(self.plans.get(path), path not in self._unreadable or info is not None),
                preset_text(st), f"{st.measured:.1f} t/s" if st.measured else "-")

    def _row(self, p: Path) -> None:
        """Insert or update one row, in place by key."""
        key, t = str(p), self.model_table
        cells = self._cells(key)
        if key in t.rows:
            for col, v in zip([c for _, c in COLUMNS], cells):
                t.update_cell(key, col, v, update_width=True)
        else:
            t.add_row(*cells, key=key)

    def _empty_state(self) -> None:
        """Show a one-line hint when there are no models."""
        a = self.app
        msg = ""
        if not self.model_table.row_count:
            msg = ("Scanning models..." if a.scanning else
                   f"No models found in {', '.join(a.cfg.model_dirs) or '(no folders set)'}. "
                   "Add a folder in Settings (5) or download one in Discover (4).")
        w = self.query_one("#m_empty", Static)
        w.update(msg)
        w.display = bool(msg)

    # -- hooks -------------------------------------------------------------------------

    def on_models(self, models: list[Path]) -> None:
        """Add rows for new files and drop rows of gone ones."""
        want = {str(p) for p in models}
        for key in [k for k in self.model_table.rows if k.value not in want]:
            self.model_table.remove_row(key)
        self._unreadable.clear()
        for p in models:
            self._row(p)
        self._empty_state()

    def on_info(self, path: Path) -> None:
        """A header was read: fill its row and schedule planning."""
        if str(path) not in self.app.infos:
            self._unreadable.add(str(path))
        self._row(path)
        self._schedule_plan()

    def on_scan_done(self) -> None:
        """The scan finished: plan every model."""
        self._empty_state()
        self.plan_async()

    def on_host(self, host: Host) -> None:
        """Replan when the memory picture changed meaningfully, else redraw the detail."""
        sig = (tuple((d.id, d.total) for d in host.gpus), host.ram_free // (512 << 20))
        if sig != self._host_sig:
            self._host_sig = sig
            self.plan_async()
        else:
            self._schedule_detail()          # free memory moved a little: budget bars only

    def on_presets_changed(self, paths: Iterable[str]) -> None:
        """Refresh the rows of the models whose presets changed."""
        keys = list(paths) or [str(p) for p in self.app.models]
        for k in keys:
            if k in self.model_table.rows:
                self._row(Path(k))
        self.update_detail()

    def on_config_changed(self) -> None:
        """Headroom or the default profile changed: replan."""
        self.plan_async()

    # -- planning (worker) -------------------------------------------------------------

    def _schedule_plan(self) -> None:
        """Plan shortly, once per burst of header reads."""
        if self._plan_timer is not None:
            self._plan_timer.stop()
        self._plan_timer = self.set_timer(DEBOUNCE, self.plan_async)

    def plan_async(self) -> None:
        """Plan every readable model for every applicable intent in a worker."""
        a = self.app
        infos, host, base = dict(a.infos), a.host, a.cfg.default_profile
        pct, cal = a.cfg.headroom_pct, a.calibration
        if not infos or (not host.gpus and not host.ram_total):
            return

        def work() -> dict[str, dict[str, Plan]]:
            return {p: planner.plan_all(i, host, base, pct, cal) for p, i in infos.items()}

        a.run_bg(work, label="Planning models", group="plan", exclusive=True,
                 done=self._plans_ready, quiet=True)

    def _plans_ready(self, plans: dict[str, dict[str, Plan]]) -> None:
        """Plans arrived from the worker: fill the Best for cells and the detail."""
        self.plans = plans
        for key in list(self.model_table.rows):
            self._row(Path(str(key.value)))
        self.update_detail()

    # -- detail pane -------------------------------------------------------------------

    @on(DataTable.RowHighlighted, "#model_table")
    def _model_highlighted(self) -> None:
        """The model cursor moved: the detail follows after a short delay."""
        self._intent = ""
        self._schedule_detail()

    def _schedule_detail(self) -> None:
        """Redraw the detail pane once the cursor settles."""
        if self._detail_timer is not None:
            self._detail_timer.stop()
        self._detail_timer = self.set_timer(DEBOUNCE, self.update_detail)

    @on(DataTable.RowHighlighted, "#intent_table")
    def _intent_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """The intent cursor moved: actions now apply to that intent."""
        if event.row_key is not None and event.row_key.value != self._intent:
            self._intent = str(event.row_key.value)
            self._show_intent()

    def rows_for(self, model: str) -> list[IntentRow]:
        """Per-intent rows of a model from the cached plans (empty while planning)."""
        a = self.app
        plans = self.plans.get(model)
        if not plans:
            return []
        return intent_rows(plans, a.cfg.models.get(model), Path(model), a.host,
                           a.cfg.headroom_pct, a.calibration, a.build)

    def update_detail(self) -> None:
        """Fill the right pane for the highlighted model."""
        model = self.selected_model()
        meta = self.query_one("#meta", Static)
        it = self.intent_table
        if model is None:
            meta.update("")
            it.clear()
            self.query_one(BudgetPanel).display = False
            self.query_one("#m_warnings", Static).update("")
            self.query_one("#m_command", Static).update("")
            return
        info = self.app.infos.get(model)
        if info is None:
            meta.update("Cannot read this file's header." if model in self._unreadable
                        else "Reading model header...")
            it.clear()
            return
        meta.update(self._meta_text(model, info))
        rows = self.rows_for(model)
        want = {r.key for r in rows}
        for key in [k for k in it.rows if k.value not in want]:
            it.remove_row(key)
        for r in rows:
            cells = intent_cells(r)
            if r.key in it.rows:
                for col, v in zip([c for _, c in INTENT_COLUMNS], cells):
                    it.update_cell(r.key, col, v, update_width=True)
            else:
                it.add_row(*cells, key=r.key)
        if model != self._shown:
            self._shown = model
            self._intent = ""
        key = self.selected_intent()
        if key and key in it.rows and it.cursor_row != it.get_row_index(key):
            it.move_cursor(row=it.get_row_index(key))
        self._show_intent()

    def _meta_text(self, model: str, info: ModelInfo) -> Text:
        """Name, file, size, architecture and the no-preset hint of a model."""
        a = self.app
        ent = a.cfg.models.get(model)
        t = Text()
        t.append(f"{info.name}\n", style="bold")
        t.append(f"{Path(model).name}\n", style="dim")
        t.append(f"served as {a.alias_of(model)}  |  {human(info.size)}  |  "
                 f"{params_str(info.parameter_count) or '?'} params  |  "
                 f"{hf.quant_of(Path(model).name) or 'quant ?'}\n")
        t.append(f"{info.architecture}, {info.n_layer} layers, {info.n_head_kv} kv heads, trained "
                 f"ctx {info.ctx_train}" + (f", {info.expert_count} experts" if info.expert_count
                                            else "") + (f", {info.sharded} shards"
                                                        if info.sharded > 1 else "")
                 + ("  |  embedding model" if info.is_embedding else ""))
        if ent is None or not ent.presets:
            t.append("\nNo preset yet: Enter picks what it is for and serves it; t tunes it.",
                     style="yellow")
        return t

    def _show_intent(self) -> None:
        """Budget panel, warnings and command for the highlighted intent."""
        a = self.app
        model, key = self.selected_model(), self.selected_intent()
        panel = self.query_one(BudgetPanel)
        plans = self.plans.get(model or "")
        if not model or not plans or key not in plans:
            panel.display = False
            return
        plan = plans[key]
        panel.display = True
        panel.update_budget(plan.budget, a.host, a.cfg.headroom_pct)
        self.query_one("#m_warnings", Static).update(
            Text("\n".join(f"! {w}" for w in plan.warnings), style="yellow"))
        existing = a.cfg.preset(model, key)
        prof = existing.profile if existing else plan.profile
        argv = build_command(a.cfg, model, prof, prof.port or 0)
        cmd = command_string(argv).replace("--port 0", "--port <auto>")
        self.query_one("#m_command", Static).update(
            Text.assemble((f"{'Preset' if existing else 'Plan'} command ({key}):\n", "bold"), cmd))

    # -- actions -----------------------------------------------------------------------

    def _need_model(self) -> tuple[str, ModelInfo] | None:
        """The highlighted model and its header, or None (with a message)."""
        model = self.selected_model()
        if model is None:
            self.say("No model selected.")
            return None
        info = self.app.infos.get(model)
        if info is None:
            self.say("Still reading the model header - try again in a moment."
                     if model not in self._unreadable else "That file's header cannot be read.")
            return None
        return model, info

    def _need_intent(self) -> tuple[str, ModelInfo, str] | None:
        """The highlighted model, its header and the intent actions apply to, or None."""
        got = self._need_model()
        if got is None:
            return None
        key = self.selected_intent()
        if not key:
            self.say("Still planning - try again in a moment.")
            return None
        return got[0], got[1], key

    @on(DataTable.RowSelected, "#model_table")
    def _row_selected(self) -> None:
        """Enter on a model: serve it."""
        self.action_serve()

    @on(DataTable.RowSelected, "#intent_table")
    def _intent_selected(self) -> None:
        """Enter on an intent row: serve that intent."""
        got = self._need_intent()
        if got:
            self.app.serve_model(got[0], got[2])

    def _pick_intent(self, model: str, then) -> None:
        """Ask what the model is for, then call `then(intent_key)`."""
        rows = self.rows_for(model)
        if not rows:
            return self.say("Still planning - try again in a moment.")
        ent = self.app.cfg.models.get(model)

        def done(key: str | None) -> None:
            if key:
                then(key)

        self.app.push_screen(IntentPicker(self.app.alias_of(model), rows,
                                          self.selected_intent() if ent is None
                                          else (ent.active or self.selected_intent())), done)

    def action_serve(self) -> None:
        """Serve the model: the active preset, or an intent picker when it has none or several."""
        got = self._need_model()
        if got is None:
            return
        model = got[0]
        ent = self.app.cfg.models.get(model)
        if ent is None or len(ent.presets) != 1:
            return self._pick_intent(model, lambda key: self.app.serve_model(model, key))
        self.app.serve_model(model)

    def action_serve_standalone(self) -> None:
        """Serve the model on a llama-server of its own (picker when it has no preset)."""
        got = self._need_model()
        if got is None:
            return
        model = got[0]
        ent = self.app.cfg.models.get(model)
        if ent is None or not ent.presets:
            return self._pick_intent(model, lambda key: self.app.serve_model(
                model, key, standalone=True))
        self.app.serve_model(model, standalone=True)

    def action_tune(self) -> None:
        """Tune the highlighted intent."""
        got = self._need_intent()
        if got:
            self.app.open_tune(got[0], got[2])

    def action_plan(self) -> None:
        """Show the plan of the highlighted intent."""
        got = self._need_intent()
        if got is None:
            return
        model, info, key = got
        a = self.app
        screen = PlanScreen(a.cfg, model, a.alias_of(model), info, a.host, by_key[key],
                            a.calibration)

        def done(result: str | None) -> None:
            if result == SAVED:
                a.presets_changed(model)
            elif result == EDIT:
                self.edit(model, key)
            elif result == TUNE:
                a.open_tune(model, key)
            elif result == SERVE:
                a.serve_model(model, key)

        a.push_screen(screen, done)

    def action_edit(self) -> None:
        """Edit the highlighted intent's preset (a missing one starts from the plan)."""
        got = self._need_intent()
        if got:
            self.edit(got[0], got[2])

    def edit(self, model: str, key: str) -> None:
        """Open the preset editor for a model's preset named `key`.

        Saving stores the profile with ``source="edited"`` (the measurement is kept); an
        unchanged profile changes nothing.

        Args:
            model: Model path.
            key: Preset name (an intent key).
        """
        a = self.app
        info = a.infos.get(model)
        if info is None:
            return self.say("Still reading the model header - try again in a moment.")
        old = a.cfg.preset(model, key)
        prof = old.profile if old else a.cfg.default_profile

        def done(new) -> None:
            if new is None or (old is not None and new == old.profile):
                return
            a.cfg.set_preset(model, Preset(key, new, EDITED, old.measured if old else None),
                             activate=False)
            a.presets_changed(model)
            self.say(f"Saved the '{key}' preset of {a.alias_of(model)} (edited)")

        a.push_screen(PresetEditor(a.cfg, prof, a.host, info, f"{a.alias_of(model)} - {key}",
                                   intent=by_key[key], apply_plan=old is None,
                                   calibration=a.calibration), done)

    def action_edit_default(self) -> None:
        """Edit the default profile every plan starts from."""
        a = self.app

        def done(new) -> None:
            if new is not None and new != a.cfg.default_profile:
                a.cfg.default_profile = new
                try:
                    a.cfg.save()
                except OSError as e:
                    return a.fail(f"cannot save the config: {e}")
                a.config_changed()
                self.say("Default profile saved; plans are recomputed")

        a.push_screen(PresetEditor(a.cfg, a.cfg.default_profile, a.host, None,
                                   "Default profile"), done)

    def action_activate(self) -> None:
        """Make the highlighted intent's preset the one served as plain alias."""
        got = self._need_intent()
        if got is None:
            return
        model, _, key = got
        a = self.app
        if a.cfg.preset(model, key) is None:
            return self.say(f"No '{key}' preset yet: p plans it, t tunes it, Enter serves it.")
        if a.cfg.models[model].active == key:
            return self.say(f"'{key}' is already the active preset.")
        if a.activate_preset(model, key):
            self.say(f"'{key}' is now served as {a.alias_of(model)}")

    def action_delete_preset(self) -> None:
        """Delete the highlighted intent's preset after a confirmation."""
        got = self._need_intent()
        if got is None:
            return
        model, _, key = got
        a = self.app
        if a.cfg.preset(model, key) is None:
            return self.say(f"No '{key}' preset to delete.")

        def done(yes: bool | None) -> None:
            if yes and a.cfg.delete_preset(model, key):
                a.presets_changed(model)
                self.say(f"Deleted the '{key}' preset of {a.alias_of(model)}")

        a.push_screen(ConfirmScreen(f"Delete the '{key}' preset of {a.alias_of(model)}?"), done)


def _size(path: Path) -> int:
    """File size, 0 when unreadable."""
    try:
        return path.stat().st_size
    except OSError:
        return 0
