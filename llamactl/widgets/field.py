"""SettingField: label + typed input + hint + live impact line for one Setting."""
from __future__ import annotations

from typing import Any

from textual import events, on
from textual.css.query import NoMatches
from textual.message import Message
from textual.validation import Integer, Number, ValidationResult, Validator
from textual.widget import Widget
from textual.widgets import Input, Label, Select, Switch

from .. import network, schema
from ..gguf import ModelInfo
from ..hardware import Host
from ..schema import Kind, Setting


HOST_OPTIONS = [("This machine (127.0.0.1)", network.LOCAL),
                ("Local network (0.0.0.0)", network.LAN), ("Custom address", network.CUSTOM)]


class _StepInput(Input):
    """Input where +/- adjust the value by a step instead of typing.

    Args:
        step: Amount added/subtracted per keypress.
        integer: True for int fields, False for float.
    """

    def __init__(self, *args: Any, step: float = 1.0, integer: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._step = step
        self._integer = integer

    async def _on_key(self, event: events.Key) -> None:
        if event.character in ("+", "-"):
            try:
                cur = int(self.value or 0) if self._integer else float(self.value or 0)
            except ValueError:
                cur = 0
            cur += self._step if event.character == "+" else -self._step
            self.value = str(int(cur) if self._integer else round(cur, 6))
            event.stop()
            event.prevent_default()
            return
        await super()._on_key(event)


class GpuLayersValidator(Validator):
    """Accepts 'auto', 'all' or a non-negative integer."""

    def validate(self, value: str) -> ValidationResult:
        """Check the gpu_layers text."""
        ok = value.strip() in ("auto", "all") or value.strip().isdigit()
        return self.success() if ok else self.failure("'auto', 'all' or an integer")


class SettingField(Widget):
    """One settings row: label, typed input, hint, impact line.

    Args:
        setting: The schema Setting this field edits.
        value: Initial value.
        info: Model metadata (None for the default profile).
        host: Host snapshot.
        disabled_reason: When given, the field is disabled and this reason
            replaces the hint text.
    """

    DEFAULT_CSS = """
    SettingField { height: auto; margin-bottom: 1; }
    SettingField .hint { color: $text-muted; text-style: italic; }
    SettingField .impact { color: $secondary; }
    SettingField .impact.recommended { color: $success; text-style: bold; }
    SettingField .warning { color: $warning; }
    SettingField.invalid Input { border: round $error; }
    """

    class Changed(Message):
        """Posted when the field's input value changes."""

        def __init__(self, field: "SettingField") -> None:
            self.field = field
            super().__init__()

    def __init__(self, setting: Setting, value: Any, info: ModelInfo | None,
                 host: Host, disabled_reason: str = "") -> None:
        super().__init__()
        self.setting = setting
        self.info = info
        self.host = host
        self._disabled_reason = disabled_reason
        self._value = value
        self._reason = ""  # active recommendation reason, kept until user edits
        self._impact = ("", False)  # impact line text and whether it is a plan reason
        self._warning = ""  # warning line text, hidden while empty
        self._pending = 0  # programmatic set_value calls with a Changed event in flight
        if setting.kind == Kind.INT:
            inp: Widget = (_StepInput(str(value), step=setting.step, integer=True,
                                      validators=[Integer(setting.minimum, setting.maximum)])
                           if setting.step else
                           Input(str(value), validators=[Integer(setting.minimum, setting.maximum)]))
        elif setting.kind == Kind.FLOAT:
            inp = (_StepInput(str(value), step=setting.step, integer=False,
                              validators=[Number(setting.minimum, setting.maximum)])
                   if setting.step else
                   Input(str(value), validators=[Number(setting.minimum, setting.maximum)]))
        elif setting.kind == Kind.CHOICE:
            inp = Select([(c or "(default)", c) for c in setting.choices], value=value, allow_blank=False)
        elif setting.kind == Kind.BOOL:
            inp = Switch(value=bool(value))
        elif setting.key == "gpu_layers":
            inp = Input(str(value), validators=[GpuLayersValidator()])
        else:
            inp = Input(str(value), password=(setting.key == "api_key"))
        self.input = inp
        self.mode: Select | None = None          # bind mode picker of the "host" field
        self._syncing = False
        if setting.key == "host":
            mode = network.bind_mode(str(value))
            self.mode = Select(HOST_OPTIONS, allow_blank=False, value=mode, compact=True)
            inp.disabled = mode != network.CUSTOM

    def compose(self):
        """Label with flag, input, hint, empty impact line."""
        s = self.setting
        yield Label(f"{s.label} ({s.flag or 'extra'})")
        if self.mode is not None:
            yield self.mode
        yield self.input
        hint = f"disabled: {self._disabled_reason}" if self._disabled_reason else s.hint
        yield Label(hint, classes="hint")
        yield Label("", classes="impact")
        yield Label("", classes="warning")

    def on_mount(self) -> None:
        """Disable the input when the field is not applicable; paint any impact already set."""
        if self._disabled_reason:
            self.input.disabled = True
            if self.mode is not None:
                self.mode.disabled = True
        self._paint()

    @property
    def value(self) -> Any:
        """The typed value via schema.validate.

        Raises:
            ValueError: When the raw input is invalid.
        """
        if isinstance(self.input, Switch):
            return self.input.value
        if isinstance(self.input, Select):
            v = self.input.value
            return self._value if v is Select.NULL else v     # not mounted yet: the initial value
        return schema.validate(self.setting, self.input.value)

    @property
    def valid(self) -> bool:
        """True when the current input parses to a valid value."""
        try:
            self.value
            return True
        except ValueError:
            return False

    def set_value(self, v: Any) -> None:
        """Set the input widget to a value (programmatic; keeps any reason shown)."""
        self._value = v
        old = self.input.value
        if isinstance(self.input, Switch):
            self.input.value = bool(v)
        elif isinstance(self.input, Select):
            self.input.value = v
        else:
            self.input.value = str(v)
        if self.input.value != old:
            self._pending += 1
        if self.mode is not None:
            self._show_mode(str(v))

    def set_impact(self, text: str) -> None:
        """Update the impact line (no highlighting); skipped while a reason shows."""
        if self._reason:
            return
        self._impact = (text, False)
        self._paint()

    def show_reason(self, text: str) -> None:
        """Show a recommendation reason, highlighted, until the user edits."""
        self._reason = text
        self._impact = (text, True)
        self._paint()

    def show_warning(self, text: str) -> None:
        """Show a warning line under the field ("" clears and hides it)."""
        self._warning = text
        self._paint()

    def _paint(self) -> None:
        """Show the stored impact and warning lines; a field whose children are not mounted
        yet shows them later.

        The editor computes its first preview while the collapsible groups are still being
        composed, so the labels may not exist yet (`on_mount` paints again).
        """
        try:
            lbl = self.query_one(".impact", Label)
            warn = self.query_one(".warning", Label)
        except NoMatches:
            return
        lbl.set_class(self._impact[1], "recommended")
        lbl.update(self._impact[0])
        warn.update(self._warning)
        warn.display = bool(self._warning)

    def _check(self) -> None:
        """Re-validate and mark the field, then notify the editor."""
        if self._pending:
            self._pending -= 1
        else:
            self._reason = ""  # user edit clears the recommendation reason
        if self.valid:
            self.remove_class("invalid")
        else:
            self.add_class("invalid")
            try:
                self.value
            except ValueError as e:
                self._impact = (str(e), False)
                self._paint()
        self.post_message(self.Changed(self))

    def _show_mode(self, host: str) -> None:
        """Point the bind picker at `host` and enable the address box only for Custom."""
        assert self.mode is not None
        mode = network.bind_mode(host)
        self._syncing = True
        try:
            self.mode.value = mode
        finally:
            self._syncing = False
        self.input.disabled = mode != network.CUSTOM

    @on(Select.Changed)
    def _mode_changed(self, event: Select.Changed) -> None:
        """The user picked a bind mode: fill the address box, editable only for Custom."""
        if event.control is not self.mode or self._syncing:
            return
        mode = str(event.value)
        if mode == network.CUSTOM:
            self.input.disabled = False
            self.input.focus()
        else:
            self.input.disabled = True
            self.input.value = network.host_for(mode)

    @on(Input.Changed)
    @on(Select.Changed)
    @on(Switch.Changed)
    def _changed(self, event) -> None:
        if event.control is self.input:
            self._check()
