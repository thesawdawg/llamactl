"""RecommendationScreen: propose a profile, confirm before saving."""
from __future__ import annotations

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Label

from .. import intents, planner, schema
from ..calibration import Calibration
from ..config import Config, Preset
from ..fmt import FITS_STYLE
from ..gguf import ModelInfo
from ..hardware import Host
from ..widgets.budget import BudgetPanel


class RecommendationScreen(Screen[str]):
    """Show a recommended profile and let the user accept, edit or skip it.

    Dismisses "accepted", "editor" or "skipped".

    Args:
        cfg: Global config.
        model: Model path the recommendation applies to.
        info: Parsed model metadata.
        host: Host snapshot.
        calibration: Measured data correcting the estimates, or None.

    The plan is for the configured default intent when it applies to the model, otherwise
    for the first applicable intent. Accepting saves it as an "estimated" preset.
    """

    BINDINGS = [
        Binding("enter", "accept", "Accept and save", show=False),
        Binding("e", "editor", "Open in editor"),
        Binding("escape", "skip", "Skip"),
    ]
    CSS = """
    #rec_body { padding: 0 1; }
    #rec_table { height: auto; max-height: 50%; margin: 1 0; }
    #rec_buttons { height: auto; }
    #rec_buttons Button { margin-right: 1; }
    #rec_none { color: $text-muted; margin: 1 0; display: none; }
    """

    def __init__(self, cfg: Config, model: str, info: ModelInfo, host: Host,
                 calibration: Calibration | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.host = host
        self.model_name = info.name
        self.intent = intents.resolve(cfg.default_intent, info)
        self.plan = planner.plan(self.intent, info, host, cfg.profile_for(model),
                                 cfg.headroom_pct, calibration)

    def compose(self) -> ComposeResult:
        yield Header()
        v = self.plan.verdict
        with Vertical(id="rec_body"):
            yield Label(Text.assemble(
                (self.model_name, "b"), (f"  -  {self.intent.label}  -  verdict: ", ""),
                (v, FITS_STYLE.get(v, ""))))
            for warning in self.plan.warnings:
                yield Label(Text(warning, style="yellow"))
            yield BudgetPanel()
            t = DataTable(id="rec_table")
            yield t
            yield Label("Current profile already matches the recommendation.", id="rec_none")
            with Horizontal(id="rec_buttons"):
                yield Button("Accept and save", id="accept", variant="success")
                yield Button("Open in editor", id="editor")
                yield Button("Skip", id="skip")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = "Recommendation"
        self.query_one(BudgetPanel).update_budget(self.plan.budget, self.host,
                                                  self.cfg.headroom_pct)
        t = self.query_one("#rec_table", DataTable)
        t.add_columns("Setting", "Current", "Recommended", "Why")
        base = self.cfg.profile_for(self.model)
        for key, reason in self.plan.reasons.items():
            t.add_row(schema.by_key[key].label, str(getattr(base, key)),
                      str(getattr(self.plan.profile, key)), reason)
        if not self.plan.reasons:
            t.display = False
            self.query_one("#rec_none", Label).display = True
            self.query_one("#accept", Button).disabled = True
        self.query_one("#accept", Button).focus()  # Enter lands on the button

    def action_accept(self) -> None:
        if self.plan.reasons:
            self.cfg.set_preset(self.model, Preset(self.plan.intent, self.plan.profile, "estimated"))
            self.cfg.save()
        self.dismiss("accepted")

    def action_editor(self) -> None:
        self.dismiss("editor")

    def action_skip(self) -> None:
        self.dismiss("skipped")

    @on(Button.Pressed, "#accept")
    def _accept(self) -> None:
        self.action_accept()

    @on(Button.Pressed, "#editor")
    def _editor(self) -> None:
        self.action_editor()

    @on(Button.Pressed, "#skip")
    def _skip(self) -> None:
        self.action_skip()
