"""PresetEditor: full-screen editor for one preset (or the default profile) with live impact."""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import Button, Collapsible, Footer, Header, Label, Static

from .. import actions, schema
from ..calibration import Calibration
from ..command import build_command, command_string, fit_conflict
from ..config import Config, Profile
from ..fmt import human
from ..gguf import ModelInfo, tensor_table
from ..hardware import Host
from ..intents import Intent
from ..placement import cpu_bytes, parse_overrides
from ..planner import (estimate_budget, expert_ram_bytes, gpu_limit, placeholder_info,
                      verdict, verdict_label)
from ..planner import Budget
from ..schema import Group
from ..widgets import layout
from ..widgets.budget import BudgetPanel
from ..widgets.buttonbar import ButtonBar
from ..widgets.field import SettingField
from .help import GuideScreen, HelpPanel

GROUP_TITLES = {Group.MEMORY: "Memory & offload", Group.SAMPLING: "Sampling",
                Group.SERVER: "Server & API", Group.GENERAL: "Extra"}
DEBOUNCE = 0.15


class PresetEditor(Screen[Profile | None]):
    """Edit a preset's Profile: typed fields, live impacts, budget and command preview.

    Args:
        cfg: Global config (bin_dir for the preview command).
        prof: Profile to edit.
        host: Host snapshot.
        info: Parsed model metadata, or None for the default profile.
        title: Screen title.
        intent: The preset's intent: what "Apply plan" plans for. None uses the configured
            default intent that fits the model.
        apply_plan: Fill the fields from the plan once the editor opens (a preset that does not
            exist yet starts this way).
        calibration: Measured data correcting the estimates, or None.

    Returns (via dismiss): the edited Profile on save, None on cancel. The caller stores it as
    a preset with ``source="edited"``; without a model (the default profile) there is no plan.
    """

    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "cancel", "Cancel"),
        Binding("f2", "apply_plan", "Apply plan"),
        Binding("f1", "help", "Help"),
        Binding("R", "apply_plan", "Apply plan", show=False),
        Binding("ctrl+r", "reset_field", "Reset field"),
        Binding("question_mark", "help", "Help", show=False),
    ]
    HORIZONTAL_BREAKPOINTS = layout.HORIZONTAL_BREAKPOINTS
    VERTICAL_BREAKPOINTS = layout.VERTICAL_BREAKPOINTS
    CSS = """
    #editor { height: 1fr; }
    #fields { width: 3fr; padding: 0 1; }
    #side { width: 2fr; padding: 0 1; }
    #preview { border: round $primary; padding: 0 1; height: auto; }
    #preview:focus-within { border: round $accent; }
    #ed_status { height: auto; padding: 0 1; color: $text-muted; }
    PresetEditor.-narrow #editor { layout: vertical; }
    PresetEditor.-narrow #fields { width: 1fr; }
    PresetEditor.-narrow #side { width: 1fr; height: auto; max-height: 45%; }
    """

    def __init__(self, cfg: Config, prof: Profile, host: Host,
                 info: ModelInfo | None, title: str, *, intent: Intent | None = None,
                 apply_plan: bool = False, calibration: Calibration | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.host = host
        self.info = info
        self.title_text = title
        self.intent = intent
        self._apply_plan_on_open = apply_plan
        self.calibration = calibration
        self._draft = prof
        self.fields: dict[str, SettingField] = {}
        self.tensors: list[tuple[str, int]] | None = None   # tensor table, read in a worker
        self._tensor_err = ""                               # why that read failed

    def _moe(self) -> bool:
        """True when the edited model is a MoE model."""
        return self.info is not None and self.info.expert_count > 0

    def _disabled_reason(self, s: schema.Setting) -> str:
        """Why a field is disabled for this model/host, or ''."""
        if self.info is not None:
            if s.key in ("n_cpu_moe", "override_tensor") and self.info.expert_count == 0:
                return "not a MoE model"
        if s.key == "tensor_split" and len(self.host.gpus) < 2:
            return "only one GPU"
        return ""

    def compose(self) -> ComposeResult:
        """Grouped setting fields on the left; budget, command and buttons on the right."""
        yield Header()
        with Horizontal(id="editor"):
            with VerticalScroll(id="fields"):
                for group in Group:
                    with Collapsible(title=GROUP_TITLES[group],
                                     collapsed=(group != Group.MEMORY)):
                        for s in schema.for_group(group):
                            f = SettingField(s, getattr(self._draft, s.key), self.info,
                                             self.host, self._disabled_reason(s))
                            self.fields[s.key] = f
                            yield f
                            if s.key == "override_tensor" and self._moe():
                                h = (self.info or placeholder_info(0)).n_layer // 2
                                with Horizontal(id="ot_examples"):
                                    yield Button(f"experts of layers {h}+ to CPU",
                                                 id="ot_layers", compact=True)
                                    yield Button("only down-projection experts to CPU",
                                                 id="ot_down", compact=True)
                                    yield Button("MoE guide", id="moe_guide", compact=True)
                                yield Static("", id="ot_preview", classes="hint")
            with VerticalScroll(id="side"):
                yield BudgetPanel()
                yield Static("", id="preview")
        with ButtonBar(id="buttons", classes="-docked"):
            yield Button("Save", id="save", variant="primary")
            yield Button("Apply plan", id="plan")
            yield Button("Cancel", id="cancel")
        yield Label("", id="ed_status")
        yield Footer()

    def on_mount(self) -> None:
        """Compute the first preview, read the tensor table of a MoE model for the -ot preview
        (and apply the plan when asked)."""
        self.sub_title = self.title_text
        self._recompute()
        if self._moe():
            path = self.info.path
            self.app.run_bg(lambda: tensor_table(path), label="Reading the tensor list",
                            group="tensors", exclusive=True, done=self._tensors_read,
                            failed=self._tensors_failed, quiet=True)
        if self._apply_plan_on_open:
            self.action_apply_plan()

    def _tensors_read(self, tensors: list[tuple[str, int]]) -> None:
        """The tensor table arrived: the -ot preview and simulated CPU bytes can fill in."""
        self.tensors = tensors
        self._recompute()

    def _tensors_failed(self, e: Exception) -> None:
        """The table could not be read: the preview explains why it stays empty."""
        self._tensor_err = str(e)
        self._recompute()

    def _rebuild_draft(self) -> None:
        """Rebuild the draft profile; invalid fields keep the previous value."""
        updates = {}
        for key, f in self.fields.items():
            try:
                updates[key] = f.value
            except ValueError:
                updates[key] = getattr(self._draft, key)
        self._draft = self._draft.merged(**updates)

    @property
    def all_valid(self) -> bool:
        """True when every field parses."""
        return all(f.valid for f in self.fields.values())

    def _recompute(self) -> None:
        """Refresh draft, impacts, budget panel and command preview."""
        self._rebuild_draft()
        for key, f in self.fields.items():
            s = schema.by_key[key]
            if s.impact and f.valid and not f.input.disabled:
                f.set_impact(s.impact(self._draft, self.info, self.host))
        self.fields["fit"].show_warning(
            "Auto-fit is ignored while experts are placed by hand; llamactl turns it off."
            if fit_conflict(self._draft) else "")
        info = self.info or placeholder_info(0)
        budget = estimate_budget(info, self._draft, self.host, self.calibration)
        v = verdict(budget, self.host, self.cfg.headroom_pct)
        expert_ram = expert_ram_bytes(info, self._draft)
        if self.tensors is not None:
            try:
                overrides = parse_overrides(self._draft.override_tensor)
            except ValueError:
                overrides = []
            expert_ram = cpu_bytes(self.tensors, overrides, self._draft.n_cpu_moe)[1]
        self.query_one(BudgetPanel).update_budget(
            budget, self.host, self.cfg.headroom_pct,
            expert_ram=expert_ram,
            verdict_text=verdict_label(v, self._draft, info))
        self._ot_preview(budget)
        model = str(self.info.path) if self.info else "<model>"
        argv = build_command(self.cfg, model, self._draft, self._draft.port or 0)
        if not self._draft.port:
            argv[argv.index("--port") + 1] = "<auto>"
        self.query_one("#preview", Static).update(command_string(argv))
        self.query_one("#save", Button).disabled = not self.all_valid

    def _ot_preview(self, budget: Budget) -> None:
        """One line under the -ot field: which tensors the patterns move to the CPU."""
        try:
            w = self.query_one("#ot_preview", Static)
        except NoMatches:
            return
        text = ""
        if self._draft.override_tensor.strip():
            if self._tensor_err:
                text = f"cannot read the tensor list: {self._tensor_err}"
            elif self.tensors is None:
                text = "loading tensor list..."
            else:
                try:
                    overrides = parse_overrides(self._draft.override_tensor)
                except ValueError:
                    overrides = []
                if overrides:
                    matched, cpu = cpu_bytes(self.tensors, overrides,
                                             self._draft.n_cpu_moe)
                    text = (f"matches {matched} tensors, {human(cpu)} moves to CPU; "
                            f"GPU part now {human(budget.total() - cpu)} of "
                            f"{human(gpu_limit(self.host, self.cfg.headroom_pct))} usable")
        w.update(text)

    @on(SettingField.Changed)
    def _changed(self) -> None:
        """A field changed: recompute after a short delay."""
        if t := getattr(self, "_deb", None):
            t.stop()
        self._deb = self.set_timer(DEBOUNCE, self._recompute)

    def _focused_field(self) -> SettingField | None:
        """The field containing the currently focused widget."""
        w = self.app.focused
        while w is not None:
            if isinstance(w, SettingField):
                return w
            w = w.parent
        return None

    def _say(self, text: str) -> None:
        """Show a line under the buttons."""
        self.query_one("#ed_status", Label).update(text)

    def action_save(self) -> None:
        """Close with the edited profile when every field is valid."""
        if self.all_valid:
            self.dismiss(self._draft)

    def action_cancel(self) -> None:
        """Close without a result."""
        self.dismiss(None)

    def action_apply_plan(self) -> None:
        """Refill the fields the planner changes from `actions.plan_for` for the intent."""
        if self.info is None:
            return self._say("No model - plans need parsed model metadata.")
        intent = self.intent or actions.choose_intent(None, self.cfg.default_intent, self.info)
        rec = actions.plan_for(self.cfg, self.info, self.host, intent, self.calibration)
        for key, reason in rec.reasons.items():
            f = self.fields[key]
            f.set_value(getattr(rec.profile, key))
            f.show_reason(reason)
        self._draft = self._draft.merged(**{k: getattr(rec.profile, k) for k in rec.reasons})
        self._recompute()
        warn = f" Warning: {'; '.join(rec.warnings)}." if rec.warnings else ""
        v = verdict_label(rec.verdict, rec.profile, self.info or placeholder_info(0))
        self._say(f"Plan applied ({intent.label}): {v}.{warn} "
                  "Review, then F1 for help or Ctrl+S to save.")

    def action_reset_field(self) -> None:
        """Reset the focused field to its default."""
        f = self._focused_field()
        if f:
            f.set_value(f.setting.default)
            self._recompute()

    def action_help(self) -> None:
        """Show the help text of the focused field."""
        f = self._focused_field()
        if f:
            self.app.push_screen(HelpPanel(f.setting))

    @on(Button.Pressed, "#save")
    def _save_btn(self) -> None:
        """The Save button."""
        self.action_save()

    @on(Button.Pressed, "#plan")
    def _plan_btn(self) -> None:
        """The Apply plan button."""
        self.action_apply_plan()

    @on(Button.Pressed, "#ot_layers")
    def _ot_layers_btn(self) -> None:
        """Fill -ot with the experts of the second half of the layers."""
        n = (self.info or placeholder_info(0)).n_layer
        self.fields["override_tensor"].set_value(
            "blk\\.(" + "|".join(str(i) for i in range(n // 2, n))
            + ")\\.ffn_.*_exps\\.=CPU")
        self._recompute()

    @on(Button.Pressed, "#ot_down")
    def _ot_down_btn(self) -> None:
        """Fill -ot with only the down-projection experts."""
        self.fields["override_tensor"].set_value("ffn_down_exps=CPU")
        self._recompute()

    @on(Button.Pressed, "#moe_guide")
    def _guide_btn(self) -> None:
        """Open the MoE guide."""
        self.app.push_screen(GuideScreen())

    @on(Button.Pressed, "#cancel")
    def _cancel_btn(self) -> None:
        """The Cancel button."""
        self.action_cancel()
