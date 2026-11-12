"""PlanScreen: what the planner proposes for one intent, and what to do with it."""
from __future__ import annotations

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Label

from .. import actions, schema
from ..actions import ActionError
from ..calibration import Calibration
from ..config import Config
from ..fmt import FITS_STYLE, human
from ..gguf import ModelInfo
from ..hardware import Host
from ..intents import Intent
from ..planner import verdict_label
from ..support import ArchSupport
from ..widgets import layout
from ..widgets.budget import BudgetPanel
from ..widgets.buttonbar import ButtonBar
from .confirm import ConfirmScreen
from .help import GuideScreen

SAVED, EDIT, TUNE, SERVE = "saved", "edit", "tune", "serve"


class PlanScreen(Screen[str | None]):
    """Show a plan and let the user save it as a preset, edit it, tune it, serve it or leave.

    Dismisses with "saved", "edit", "tune", "serve", or None on cancel. "Save as preset"
    stores an ``estimated`` preset in the config (the caller then calls
    `LlamaCtl.presets_changed`); it asks before replacing a tuned or edited one.

    Args:
        cfg: Global config (base profile, headroom, existing presets).
        model: Model path.
        alias: Name clients use for the model.
        info: Parsed model metadata.
        host: Host snapshot.
        intent: What the plan is for.
        calibration: Measured data correcting the estimates, or None.
    """

    BINDINGS = [
        Binding("s", "save", "Save as preset"),
        Binding("e", "edit", "Edit"),
        Binding("t", "tune", "Tune"),
        Binding("v", "serve", "Serve"),
        Binding("escape", "cancel", "Cancel"),
    ]
    HORIZONTAL_BREAKPOINTS = layout.HORIZONTAL_BREAKPOINTS
    VERTICAL_BREAKPOINTS = layout.VERTICAL_BREAKPOINTS
    CSS = """
    #plan_body { padding: 0 1; }
    #plan_table { height: auto; max-height: 50%; margin: 1 0; }
    #plan_msg { height: auto; padding: 0 1; }
    #plan_none { color: $text-muted; margin: 1 0; }
    """

    def __init__(self, cfg: Config, model: str, alias: str, info: ModelInfo, host: Host,
                 intent: Intent, calibration: Calibration | None = None,
                 support: ArchSupport | None = None) -> None:
        super().__init__()
        self.cfg, self.model, self.alias, self.host, self.intent = cfg, model, alias, host, intent
        self.info = info
        self.plan = actions.plan_for(cfg, info, host, intent, calibration, support=support)

    def compose(self) -> ComposeResult:
        """Verdict, budget, planned settings table, message line and action buttons."""
        yield Header()
        v = verdict_label(self.plan.verdict, self.plan.profile, self.info, self.plan.support)
        with VerticalScroll(id="plan_body"):
            yield Label(Text.assemble(
                (self.alias, "b"), (f"  -  {self.intent.label}: {self.intent.description}  -  ", ""),
                (v, FITS_STYLE.get(v, ""))))
            yield Label(f"{self.plan.ctx_per_slot} ctx per slot x {self.intent.slots} slot"
                        f"{'s' * (self.intent.slots != 1)}, {self.plan.profile.cache_type_k} KV")
            for warning in self.plan.warnings:
                yield Label(Text(f"! {warning}", style="yellow"))
            yield BudgetPanel()
            if self.plan.expert_ram:
                yield Label(Text("Expert placement", style="bold"))
                yield Label(f"experts of {self.plan.profile.n_cpu_moe}/{self.info.n_layer} "
                            f"layers in RAM ({human(self.plan.expert_ram)}); "
                            f"GPU part {human(self.plan.budget.total() - self.plan.expert_ram)}")
                yield Label(f"one more layer on the GPU needs {human(self.plan.layer_gain)} "
                            "more VRAM; fewer layers in RAM is faster")
            yield DataTable(id="plan_table")
            yield Label("The plan changes nothing from the base profile.", id="plan_none")
        yield Label("", id="plan_msg")
        with ButtonBar(id="plan_buttons", classes="-docked"):
            yield Button("Save as preset (s)", id="save", variant="success")
            yield Button("Edit (e)", id="edit")
            yield Button("Tune (t)", id="tune")
            yield Button("Serve (v)", id="serve", variant="primary")
            if self.plan.expert_ram:
                yield Button("MoE guide", id="guide")
            yield Button("Cancel (Esc)", id="cancel")
        yield Footer()

    def on_mount(self) -> None:
        """Fill the budget panel and the Setting/Base/Planned/Why table."""
        self.sub_title = f"Plan - {self.intent.label}"
        self.query_one(BudgetPanel).update_budget(
            self.plan.budget, self.host, self.cfg.headroom_pct, expert_ram=self.plan.expert_ram,
            verdict_text=verdict_label(self.plan.verdict, self.plan.profile, self.info,
                                       self.plan.support))
        t = self.query_one("#plan_table", DataTable)
        t.add_columns("Setting", "Base", "Planned", "Why")
        base = self.cfg.default_profile
        for key, reason in self.plan.reasons.items():
            t.add_row(schema.by_key[key].label, str(getattr(base, key)),
                      str(getattr(self.plan.profile, key)), reason)
        t.display = bool(self.plan.reasons)
        self.query_one("#plan_none").display = not self.plan.reasons
        self.query_one("#save", Button).focus()

    def _say(self, text: str, style: str = "") -> None:
        """Show a line under the table."""
        self.query_one("#plan_msg", Label).update(Text(text, style=style))

    def action_save(self) -> None:
        """Store the plan as an estimated preset, asking before replacing a measured one."""
        try:
            actions.store_plan(self.cfg, self.model, self.plan, save=False)
        except ActionError as e:
            def done(yes: bool | None) -> None:
                if yes:
                    actions.store_plan(self.cfg, self.model, self.plan, overwrite=True,
                                       save=False)
                    self.dismiss(SAVED)

            self.app.push_screen(ConfirmScreen(f"{e}. Replace it with this estimate?"), done)
            return self._say(f"Not saved: {e}", "yellow")
        self.dismiss(SAVED)

    def action_edit(self) -> None:
        """Close and open the preset editor."""
        self.dismiss(EDIT)

    def action_tune(self) -> None:
        """Close and open the tune screen."""
        self.dismiss(TUNE)

    def action_serve(self) -> None:
        """Close and serve this intent."""
        self.dismiss(SERVE)

    def action_cancel(self) -> None:
        """Close without doing anything."""
        self.dismiss(None)

    def _guide(self) -> None:
        """Open the MoE guide (shown when the plan parks experts in RAM)."""
        self.app.push_screen(GuideScreen())

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        """Route a button to its action."""
        {"save": self.action_save, "edit": self.action_edit, "tune": self.action_tune,
         "serve": self.action_serve, "guide": self._guide,
         "cancel": self.action_cancel}[str(event.button.id)]()
