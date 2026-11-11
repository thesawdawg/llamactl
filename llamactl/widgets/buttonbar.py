"""ButtonBar: a row of buttons that wraps into a grid when the row would not fit."""
from __future__ import annotations

from textual import events
from textual.geometry import Size
from textual.widget import Widget
from textual.widgets import Button

GUTTER = 1                   # cells between buttons (margin-right / grid-gutter)
MIN_COLS = 1                 # the grid never goes below one column


class ButtonBar(Widget):
    """A dockable button row that never clips its buttons.

    One horizontal row at natural button widths while the displayed buttons fit the bar's
    content width; otherwise the `-wrapped` class switches to a grid whose column count is
    how many of the widest button fit. Buttons toggled with ``display`` do not take grid
    cells. On `-short` screens the buttons lose their tall borders and take one row.

    Args:
        *buttons: The buttons (passed on like a container's children).
        id: DOM id.
        classes: CSS classes.
    """

    # The -short rule keys off a class of the screen (an ancestor), so this CSS has to
    # stay unscoped.
    SCOPED_CSS = False
    DEFAULT_CSS = """
    ButtonBar { height: auto; layout: horizontal; }
    ButtonBar Button { margin-right: 1; }
    ButtonBar.-wrapped { layout: grid; grid-gutter: 0 1; }
    ButtonBar.-wrapped Button { width: 100%; margin-right: 0; }
    .-short ButtonBar Button { border-top: none !important; border-bottom: none !important;
        height: 1 !important; min-width: 0 !important; }
    ButtonBar.-docked { border-top: solid $panel; }
    """

    def __init__(self, *buttons: Button, id: str | None = None,
                 classes: str | None = None) -> None:
        super().__init__(*buttons, id=id, classes=classes)
        self._last_size = Size(0, 0)

    def on_mount(self) -> None:
        """First reflow once the children are arranged."""
        self.call_after_refresh(self.reflow)

    def on_resize(self, event: events.Resize) -> None:
        """Reflow when the bar's size changed. Height matters too: crossing the `-short`
        breakpoint changes button min-widths, which changes what fits on one row."""
        if event.size != self._last_size:
            self._last_size = event.size
            self.reflow()

    def reflow(self) -> None:
        """Pick row or grid from the current width; call after toggling button `display`."""
        buttons = [b for b in self.children if isinstance(b, Button) and b.display]
        width = self.scrollable_content_region.width
        if not buttons or not width:
            return
        natural = [self._natural(b) for b in buttons]
        if sum(natural) + GUTTER * (len(natural) - 1) <= width:
            self.remove_class("-wrapped")
            return
        cols = max(MIN_COLS, (width + GUTTER) // (max(natural) + GUTTER))
        self.styles.grid_size_columns = cols
        self.add_class("-wrapped")

    def _natural(self, button: Button) -> int:
        """The width a button wants: its label plus gutter (padding and border), honoured
        by its min-width.

        Args:
            button: A displayed button of the bar.

        Returns:
            Natural width in cells.
        """
        w = (button.get_content_width(self.scrollable_content_region.size, Size(0, 0))
             + button.styles.gutter.width)
        if button.styles.min_width is not None:
            w = max(w, int(button.styles.min_width.resolve(self.size, Size(0, 0))))
        return w
