"""Guided model controls and explicit review before saving a configuration."""
from __future__ import annotations

from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.validation import Integer, Number
from textual.widgets import Button, Collapsible, DataTable, Footer, Header, Input, Label, Select, Static

from .. import actions, schema
from ..config import Preset, Profile
from ..goals import TuningGoals
from ..gguf import ModelInfo
from ..intents import applicable, by_key
from ..widgets import layout
from ..widgets.buttonbar import ButtonBar
from .preset_editor import PresetEditor


def profile_changes(saved: Profile | None, candidate: Profile) -> list[tuple[str, str, str]]:
    """Build a comparison against the actual saved configuration.

    Args:
        saved: Existing profile, or None for a new preset.
        candidate: Proposed profile.

    Returns:
        Changed setting labels, saved values and proposed values.
    """
    return [(s.label, ("Set" if getattr(saved, s.key) else "Empty") if saved and s.key == "api_key"
             else str(getattr(saved, s.key)) if saved else "Not saved",
             ("Set" if getattr(candidate, s.key) else "Empty") if s.key == "api_key"
             else str(getattr(candidate, s.key))) for s in schema.SETTINGS
            if saved is None or getattr(saved, s.key) != getattr(candidate, s.key)]


def goal_changes(saved: TuningGoals | None, candidate: TuningGoals) -> list[tuple[str, str, str]]:
    """Compare task constraints independently of concrete launch settings.

    Args:
        saved: Existing task goals, if this preset exists.
        candidate: Draft goals.

    Returns:
        Changed goal labels, saved values and proposed values.
    """
    fields = (("max_context", "Goal: maximum context/request"),
              ("min_effective_tps", "Goal: effective output TPS"),
              ("min_prompt_tps", "Goal: input TPS floor"),
              ("min_gen_tps", "Goal: output TPS floor"),
              ("benchmark", "Goal: benchmark level"))
    return [(label, str(getattr(saved, key)) if saved else "Not saved",
             str(getattr(candidate, key))) for key, label in fields
            if saved is None or getattr(saved, key) != getattr(candidate, key)]


class ProfileReview(Screen[bool]):
    """Compare a candidate to saved values before explicit acceptance.

    Args:
        title: Model and preset identity.
        saved: Existing profile, if any.
        candidate: Proposed profile.
        explanation: Source and reason for the proposal.
    """

    BINDINGS = [Binding("escape", "cancel", "Keep current")]
    CSS = """
    ProfileReview #review_body { padding: 1 2; }
    ProfileReview #review_diff { height: auto; max-height: 65%; }
    """

    def __init__(self, title: str, saved: Profile | None, candidate: Profile,
                 explanation: str = "", *, saved_goals: TuningGoals | None = None,
                 goals: TuningGoals | None = None) -> None:
        super().__init__()
        self.title_text, self.saved, self.candidate = title, saved, candidate
        self.explanation = explanation
        self.saved_goals, self.goals = saved_goals, goals

    def compose(self) -> ComposeResult:
        """Show the candidate source, exact changes and explicit save scope.

        Returns:
            Composed widgets.
        """
        yield Header()
        with VerticalScroll(id="review_body"):
            yield Label(self.title_text)
            yield Static(self.explanation, markup=False)
            yield DataTable(id="review_diff")
            yield Static("Save changes only this preset. Running models keep their current "
                         "settings; saved changes apply on the next load. Client default "
                         "selection is unchanged.", markup=False)
        with ButtonBar(classes="-docked"):
            yield Button("Save candidate for next load", id="review_save", variant="primary")
            yield Button("Keep current", id="review_keep")
        yield Footer()

    def on_mount(self) -> None:
        """Populate changed fields and focus the non-mutating choice.

        Returns:
            None.
        """
        table = self.query_one("#review_diff", DataTable)
        table.add_columns("Setting", "Saved", "Candidate")
        rows = profile_changes(self.saved, self.candidate)
        if self.goals is not None:
            rows.extend(goal_changes(self.saved_goals, self.goals))
        for row in rows:
            table.add_row(*row)
        if not rows:
            table.add_row("Launch settings", "Current", "Unchanged; goals may change")
        self.query_one("#review_keep", Button).focus()

    @on(Button.Pressed, "#review_save")
    def save(self) -> None:
        """Accept this candidate.

        Returns:
            None.
        """
        self.dismiss(True)

    @on(Button.Pressed, "#review_keep")
    def action_cancel(self) -> None:
        """Keep the saved configuration.

        Returns:
            None.
        """
        self.dismiss(False)


class ModelWorkspace(Screen[None]):
    """Choose a preset's task goals, review settings, then explicitly serve it.

    Args:
        model: Model path.
        info: Parsed model metadata.
        key: Initial purpose preset.
    """

    BINDINGS = [Binding("escape", "close", "Back to models")]
    HORIZONTAL_BREAKPOINTS = layout.HORIZONTAL_BREAKPOINTS
    VERTICAL_BREAKPOINTS = layout.VERTICAL_BREAKPOINTS
    CSS = """
    ModelWorkspace #workspace_body { padding: 0 2; }
    ModelWorkspace .step { text-style: bold; margin-top: 1; }
    ModelWorkspace Input, ModelWorkspace Select { width: 100%; }
    ModelWorkspace #workspace_status { height: auto; color: $text-muted; }
    """

    def __init__(self, model: str, info: ModelInfo, key: str) -> None:
        super().__init__()
        self.model, self.info, self.key = model, info, key

    def compose(self) -> ComposeResult:
        """Compose the guided controls with advanced settings as an optional action.

        Returns:
            Composed widgets.
        """
        yield Header()
        with VerticalScroll(id="workspace_body"):
            yield Label(self.app.alias_of(self.model), classes="step")
            yield Static("", id="workspace_summary", markup=False)
            yield Label("1. Choose the task and tuning goals", classes="step")
            yield Select([(i.label, i.key) for i in applicable(self.info)], value=self.key,
                         allow_blank=False, id="workspace_purpose")
            yield Label("Maximum context per request (0 = purpose/model default)")
            yield Static(f"Model-reported limit: {self.info.ctx_train:,} tokens/request"
                         if self.info.ctx_train else "Model-reported context limit: unknown",
                         markup=False)
            yield Input("0", id="goal_context", validators=[Integer(minimum=0)])
            yield Label("Minimum effective output tokens/s (per turn: new input + generation)"
                        if not self.info.is_embedding else "Effective output TPS: N/A for embeddings")
            yield Input("20", id="goal_effective", validators=[Number(minimum=0)],
                        disabled=self.info.is_embedding)
            yield Label("Benchmark level")
            yield Select([("Quick — 2k turn", "quick"),
                          ("Normal — 2k turn + turn at 25k occupancy", "normal"),
                          ("Full — 5%, 25%, 50%, 75%, 95% occupancy", "full")],
                         value="normal", allow_blank=False, id="goal_benchmark")
            with Collapsible(title="Advanced speed limits", collapsed=True):
                yield Label("Minimum input-processing tokens/s (0 = off)")
                yield Input("0", id="goal_prompt", validators=[Number(minimum=0)])
                yield Label("Minimum output-generation tokens/s (0 = off)")
                yield Input("0", id="goal_generation", validators=[Number(minimum=0)],
                            disabled=self.info.is_embedding)
            yield Label("2. Review a plan or measure candidates", classes="step")
            yield Static("Tuning changes no saved settings until you accept a candidate. "
                         "Failed speed targets reduce context by 5% and test again.", markup=False)
            with ButtonBar():
                yield Button("Review plan", id="workspace_plan")
                yield Button("Tune and compare", id="workspace_tune", variant="primary")
                yield Button("Advanced settings", id="workspace_advanced")
            yield Label("3. Serve the saved preset", classes="step")
            yield Static("Serving is explicit. Saving leaves running models unchanged; "
                         "accepted changes apply after unload on the next load.", markup=False)
            with ButtonBar():
                yield Button("Serve saved preset", id="workspace_serve", variant="success")
                yield Button("Set as client default", id="workspace_default")
            yield Static("", id="workspace_status", markup=False)
        with ButtonBar(classes="-docked"):
            yield Button("Back to models", id="workspace_back")
        yield Footer()

    def on_screen_resume(self) -> None:
        """Refresh saved status after comparison or tuning returns.

        Returns:
            None.
        """
        if self.is_mounted:
            self.refresh_preset(load_goals=False)

    def on_mount(self) -> None:
        """Load the selected preset's persisted goals.

        Returns:
            None.
        """
        self.sub_title = "Model setup"
        self.refresh_preset()
        self.set_interval(2.0, lambda: self.refresh_preset(load_goals=False))

    def refresh_preset(self, *, load_goals: bool = True) -> None:
        """Display saved goals and the selected preset's scope.

        Args:
            load_goals: Replace draft fields only on initial selection or accepted save.

        Returns:
            None.
        """
        preset = self.app.cfg.preset(self.model, self.key)
        goals = preset.goals if preset else TuningGoals()
        if load_goals:
            for name, value in (("context", goals.max_context), ("effective", goals.min_effective_tps),
                                ("prompt", goals.min_prompt_tps), ("generation", goals.min_gen_tps)):
                self.query_one(f"#goal_{name}", Input).value = str(value)
            self.query_one("#goal_benchmark", Select).value = goals.benchmark
        ent = self.app.cfg.models.get(self.model)
        default = bool(ent and ent.active == self.key)
        desc = (f"Saved: {self.key} ({preset.source}), "
                f"{preset.profile.ctx_size // max(1, preset.profile.parallel)} context/request"
                if preset else f"{self.key}: no saved preset yet")
        pending = getattr(self.app.gateway, "pending_names", ())
        name = actions.served_name(self.app.cfg, self.model, self.key) if preset else ""
        snapshot = self.app.snapshot
        runtime = next((model.state for model in snapshot.models or () if model.name == name),
                       "not loaded") if snapshot else "checking"
        desc += f" | Gateway: {runtime}"
        if snapshot:
            session_name = Path(self.model).stem + ("@" + self.key if not default else "")
            sessions = [s for s in snapshot.sessions if s.name == session_name]
            if sessions:
                desc += f" | {len(sessions)} standalone/attached instance(s)"
        if name in pending:
            desc += " | Saved changes pending next load"
        self.query_one("#workspace_summary", Static).update(
            desc + (" | Client default" if default else " | Not the client default"))
        self.query_one("#workspace_serve", Button).disabled = preset is None
        self.query_one("#workspace_default", Button).disabled = preset is None or default

    @on(Select.Changed, "#workspace_purpose")
    def purpose_changed(self, event: Select.Changed) -> None:
        """Switch the target preset without applying or serving settings.

        Args:
            event: New purpose selection.

        Returns:
            None.
        """
        if isinstance(event.value, str):
            self.key = event.value
            if self.is_mounted:
                self.refresh_preset()

    def draft_goals(self) -> TuningGoals | None:
        """Validate the goal form without mutating the saved preset.

        Returns:
            Draft goals, or None when a field is invalid.
        """
        if not all(w.is_valid for w in self.query(Input)):
            self.query_one("#workspace_status", Static).update("Fix invalid goal fields first.")
            return None
        try:
            return TuningGoals(
                max_context=int(self.query_one("#goal_context", Input).value),
                min_effective_tps=float(self.query_one("#goal_effective", Input).value),
                min_prompt_tps=float(self.query_one("#goal_prompt", Input).value),
                min_gen_tps=float(self.query_one("#goal_generation", Input).value),
                benchmark=str(self.query_one("#goal_benchmark", Select).value))
        except (ValueError, OverflowError) as error:
            self.query_one("#workspace_status", Static).update(str(error))
            return None

    def review(self, candidate: Profile, goals: TuningGoals, explanation: str,
               source: str = "edited") -> None:
        """Compare proposed settings, then save only on explicit acceptance.

        Args:
            candidate: Proposed launch profile.
            goals: Draft goals saved with the accepted profile.
            explanation: Reason/source shown before acceptance.
            source: Configuration provenance.

        Returns:
            None.
        """
        old = self.app.cfg.preset(self.model, self.key)
        key = self.key

        def accepted(yes: bool) -> None:
            """Store the reviewed candidate without changing the client default.

            Args:
                yes: Explicit acceptance.

            Returns:
                None.
            """
            if not yes:
                return
            try:
                actions.save_preset(self.app.cfg, self.model,
                                    Preset(key, candidate, source, None, goals))
            except OSError as error:
                self.app.fail(f"Cannot save candidate: {error}")
                return
            self.app.presets_changed(self.model, save=False)
            self.refresh_preset()
            self.query_one("#workspace_status", Static).update("Saved; applies on next load.")

        self.app.push_screen(ProfileReview(f"{self.app.alias_of(self.model)} / {key}",
                                          old.profile if old else None, candidate, explanation,
                                          saved_goals=old.goals if old else None,
                                          goals=goals), accepted)

    @on(Button.Pressed, "#workspace_plan")
    def review_plan(self) -> None:
        """Build an estimate for the draft goals and compare it before saving.

        Returns:
            None.
        """
        goals = self.draft_goals()
        if goals is None:
            return
        intent = by_key[self.key]
        plan = actions.plan_for(self.app.cfg, self.info, self.app.host, intent,
                                self.app.calibration,
                                support=self.app.support.check(self.info.architecture, self.app.build),
                                goals=goals)
        self.review(plan.profile, goals, "Estimated plan; throughput has not been measured.\n" +
                    "\n".join(plan.reasons.values()) + "\n" + "\n".join(plan.warnings), "estimated")

    @on(Button.Pressed, "#workspace_tune")
    def tune(self) -> None:
        """Measure candidates using these draft goals without saving them first.

        Returns:
            None.
        """
        goals = self.draft_goals()
        if goals is not None:
            self.app.open_tune(self.model, self.key, goals=goals)

    @on(Button.Pressed, "#workspace_advanced")
    def advanced(self) -> None:
        """Edit lower-level settings and review the result before saving.

        Returns:
            None.
        """
        goals = self.draft_goals()
        if goals is None:
            return
        old = self.app.cfg.preset(self.model, self.key)
        profile = old.profile if old else self.app.cfg.default_profile

        def edited(candidate: Profile | None) -> None:
            """Review an advanced draft without an implicit save.

            Args:
                candidate: Editor result or None.

            Returns:
                None.
            """
            if candidate is not None:
                self.review(candidate, goals, "Advanced settings draft.")

        self.app.push_screen(PresetEditor(self.app.cfg, profile, self.app.host, self.info,
                                          f"{self.app.alias_of(self.model)} / {self.key}",
                                          intent=by_key[self.key], apply_plan=old is None,
                                          calibration=self.app.calibration), edited)

    @on(Button.Pressed, "#workspace_serve")
    def serve(self) -> None:
        """Explicitly serve the saved preset, leaving draft goals unapplied.

        Returns:
            None.
        """
        self.dismiss(None)
        self.app.serve_model(self.model, self.key)

    @on(Button.Pressed, "#workspace_default")
    def make_default(self) -> None:
        """Select the saved preset for the plain client alias.

        Returns:
            None.
        """
        self.app.activate_preset(self.model, self.key)
        self.refresh_preset()

    @on(Button.Pressed, "#workspace_back")
    def action_close(self) -> None:
        """Return to the library without applying draft goals.

        Returns:
            None.
        """
        self.dismiss(None)
