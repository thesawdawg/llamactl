"""FolderList: the model-folder editor shared by Settings and the first-run setup."""
from __future__ import annotations

from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.validation import Function
from textual.widgets import Input, Label, ListItem, ListView


def is_folder(value: str) -> bool:
    """True for an existing directory; empty input is accepted (and ignored when submitted).

    Args:
        value: What was typed.
    """
    return not value.strip() or Path(value).expanduser().is_dir()


class FolderList(Vertical):
    """A list of folders with an add box: Enter adds a valid path, Delete removes the highlighted one.

    Posts `Changed` after every add or removal; the owner reads `dirs` when it applies them.

    Args:
        dirs: The folders to start with.
        add_label: Label in front of the add box.
    """

    BINDINGS = [Binding("delete", "remove", "Remove folder")]
    DEFAULT_CSS = """
    FolderList { height: auto; }
    FolderList #dir_list { height: auto; max-height: 6; border: round $primary-darken-2; }
    FolderList .add_row { height: auto; }
    FolderList .lbl { width: 14; padding: 0 1 0 0; }
    FolderList #add_dir { width: 1fr; max-width: 50; }
    FolderList #dir_note { color: $text-muted; height: auto; }
    """

    class Changed(Message):
        """The list of folders changed."""

    def __init__(self, dirs: list[str], add_label: str = "Add folder", **kwargs) -> None:
        super().__init__(**kwargs)
        self._initial, self._add_label = list(dirs), add_label

    def compose(self) -> ComposeResult:
        """The list, the add box and a one-line note."""
        yield ListView(*[ListItem(Label(d)) for d in self._initial], id="dir_list")
        with Horizontal(classes="add_row"):
            yield Label(self._add_label, classes="lbl")
            yield Input(placeholder="path, then Enter", id="add_dir", compact=True,
                        validators=[Function(is_folder, "not a directory")])
        yield Label("", id="dir_note")

    def dirs(self) -> list[str]:
        """The folders currently listed."""
        return [item.children[0].content  # type: ignore[attr-defined]
                for item in self.query_one("#dir_list", ListView).children]

    def _note(self, text: str) -> None:
        """Show a line under the add box and in the status bar."""
        self.query_one("#dir_note", Label).update(text)
        self.app.say(text)                      # type: ignore[attr-defined]

    @on(Input.Submitted, "#add_dir")
    def _add(self, event: Input.Submitted) -> None:
        """Enter in the add box: append a valid folder to the list."""
        event.stop()
        inp = event.input
        text = inp.value.strip()
        if not text:
            return
        if not inp.is_valid:
            return self._note("Not a directory.")
        self.query_one("#dir_list", ListView).append(ListItem(Label(text)))
        inp.value = ""
        self._note("")
        self.post_message(self.Changed())

    def action_remove(self) -> None:
        """Remove the highlighted folder from the list."""
        lv = self.query_one("#dir_list", ListView)
        if lv.highlighted_child is not None:
            lv.highlighted_child.remove()
            self.post_message(self.Changed())
