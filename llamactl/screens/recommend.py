"""RecommendationScreen: propose a profile, confirm before saving."""
from __future__ import annotations

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Label

from .. import recommend, schema
from ..config import Config
from ..fmt import FITS_STYLE
from ..gguf import ModelInfo
from ..hostinfo import Host
from ..widgets.budget import BudgetPanel


class RecommendationScreen(Screen[str]):
    """Show a recommended profile and let the user accept, edit or skip it.

    Dismisses "accepted", "editor" or "skipped".

    Args:
        cfg: Global config.
        model: Model path the recommendation applies to.
        info: Parsed model metadata.
        host: Host snapshot.
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

    def __init__(self, cfg: Config, model: str, info: ModelInfo, host: Host) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.host = host
        self.model_name = info.name
        self.rec = recommend.recommend(info, host, cfg.profile_for(model),
                                       headroom_pct=cfg.headroom_pct)

    def compose(self) -> ComposeResult:
        yield Header()
        v = self.rec.verdict
        with Vertical(id="rec_body"):
            yield Label(Text.assemble(
                (self.model_name, "b"), ("  -  verdict: ", ""), (v, FITS_STYLE.get(v, ""))))
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
        self.query_one(BudgetPanel).update_budget(self.rec.budget, self.host,
                                                  self.cfg.headroom_pct)
        t = self.query_one("#rec_table", DataTable)
        t.add_columns("Setting", "Current", "Recommended", "Why")
        base = self.cfg.profile_for(self.model)
        for key, reason in self.rec.reasons.items():
            t.add_row(schema.by_key[key].label, str(getattr(base, key)),
                      str(getattr(self.rec.profile, key)), reason)
        if not self.rec.reasons:
            t.display = False
            self.query_one("#rec_none", Label).display = True
            self.query_one("#accept", Button).disabled = True
        self.query_one("#accept", Button).focus()  # Enter lands on the button

    def action_accept(self) -> None:
        if self.rec.reasons:
            self.cfg.profiles[self.model] = self.rec.profile
            self.cfg.save()
        self.cfg.seen.mark_seen(self.model)
        self.dismiss("accepted")

    def action_editor(self) -> None:
        self.cfg.seen.mark_seen(self.model)
        self.dismiss("editor")

    def action_skip(self) -> None:
        self.cfg.seen.mark_seen(self.model)
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
