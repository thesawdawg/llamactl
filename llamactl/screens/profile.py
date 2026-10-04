"""ProfileEditor: full-screen profile editor with live impact and budget."""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Collapsible, Footer, Header, Label, Static

from .. import recommend, schema
from ..command import build_command, command_string
from ..config import Config, Profile
from ..gguf import ModelInfo
from ..hostinfo import Host
from ..recommend import estimate_budget, placeholder_info
from ..schema import Group
from ..widgets.budget import BudgetPanel
from ..widgets.field import SettingField
from .help import HelpPanel

GROUP_TITLES = {Group.MEMORY: "Memory & offload", Group.SAMPLING: "Sampling",
                Group.SERVER: "Server & API", Group.GENERAL: "Extra"}
DEBOUNCE = 0.15


class ProfileEditor(Screen[Profile | None]):
    """Edit a Profile: typed fields, live impacts, budget and command preview.

    Args:
        cfg: Global config (bin_dir for the preview command).
        prof: Profile to edit.
        host: Host snapshot.
        info: Parsed model metadata, or None for the default profile.
        title: Screen title.

    Returns (via dismiss): the edited Profile on save, None on cancel.
    """

    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "cancel", "Cancel"),
        Binding("f2", "apply_recommended", "Recommend"),
        Binding("f1", "help", "Help"),
        Binding("R", "apply_recommended", "Apply recommended", show=False),
        Binding("ctrl+r", "reset_field", "Reset field"),
        Binding("question_mark", "help", "Help", show=False),
    ]
    CSS = """
    #editor { height: 1fr; }
    #fields { width: 3fr; padding: 0 1; }
    #side { width: 2fr; padding: 0 1; }
    #preview { border: round $primary; padding: 0 1; height: auto; }
    #buttons { height: auto; margin-top: 1; }
    #buttons Button { margin-right: 1; }
    #ed_status { height: auto; padding: 0 1; color: $text-muted; }
    """

    def __init__(self, cfg: Config, prof: Profile, host: Host,
                 info: ModelInfo | None, title: str, auto_recommend: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.host = host
        self.info = info
        self.title_text = title
        self._auto_recommend = auto_recommend
        self._draft = prof
        self.fields: dict[str, SettingField] = {}

    def _disabled_reason(self, s: schema.Setting) -> str:
        """Why a field is disabled for this model/host, or ''."""
        if self.info is not None:
            if s.key == "n_cpu_moe" and self.info.expert_count == 0:
                return "not a MoE model"
        if s.key == "tensor_split" and len(self.host.gpus) < 2:
            return "only one GPU"
        return ""

    def compose(self) -> ComposeResult:
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
            with Vertical(id="side"):
                yield BudgetPanel()
                yield Static("", id="preview")
                with Horizontal(id="buttons"):
                    yield Button("Save", id="save", variant="primary")
                    yield Button("Apply recommended", id="rec")
                    yield Button("Cancel", id="cancel")
                yield Label("", id="ed_status")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = self.title_text
        self._recompute()
        if self._auto_recommend:
            self.action_apply_recommended()

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
        info = self.info or placeholder_info(0)
        self.query_one(BudgetPanel).update_budget(
            estimate_budget(info, self._draft, self.host), self.host)
        model = str(self.info.path) if self.info else "<model>"
        argv = build_command(self.cfg, model, self._draft, "server",
                             self._draft.port or 0)
        if not self._draft.port:
            argv[argv.index("--port") + 1] = "<auto>"
        self.query_one("#preview", Static).update(command_string(argv))
        self.query_one("#save", Button).disabled = not self.all_valid

    @on(SettingField.Changed)
    def _changed(self) -> None:
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
        self.query_one("#ed_status", Label).update(text)

    def action_save(self) -> None:
        if self.all_valid:
            self.dismiss(self._draft)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_apply_recommended(self) -> None:
        if self.info is None:
            return self._say("No model - recommendations need parsed model metadata.")
        rec = recommend.recommend(self.info, self.host, self._draft,
                                  headroom_pct=self.cfg.headroom_pct)
        for key, reason in rec.reasons.items():
            f = self.fields[key]
            f.set_value(getattr(rec.profile, key))
            f.show_reason(reason)
        self._draft = rec.profile
        self._recompute()
        self._say(f"Recommended: {rec.verdict}. Review, then F1 for help or Ctrl+S to save.")

    def action_reset_field(self) -> None:
        f = self._focused_field()
        if f:
            f.set_value(f.setting.default)
            self._recompute()

    def action_help(self) -> None:
        f = self._focused_field()
        if f:
            self.app.push_screen(HelpPanel(f.setting))

    @on(Button.Pressed, "#save")
    def _save_btn(self) -> None:
        self.action_save()

    @on(Button.Pressed, "#rec")
    def _rec_btn(self) -> None:
        self.action_apply_recommended()

    @on(Button.Pressed, "#cancel")
    def _cancel_btn(self) -> None:
        self.action_cancel()
