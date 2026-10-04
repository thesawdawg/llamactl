"""BaseView: shared helpers for the four main views."""
from __future__ import annotations

from typing import TYPE_CHECKING

from textual.widget import Widget

from ..hostinfo import Host

if TYPE_CHECKING:
    from ..app import LlamaCtl


class BaseView(Widget):
    """A ContentSwitcher pane with access to the app and the status line."""

    @property
    def app(self) -> "LlamaCtl":
        return super().app  # type: ignore[return-value]

    def say(self, msg: str) -> None:
        """Write a one-line action result to the status bar.

        Args:
            msg: Text to show (persists until the next action).
        """
        self.app.say(msg)

    def on_activate(self) -> None:
        """Called by the shell when this view becomes visible."""
        self.focus_primary()

    def focus_primary(self) -> None:
        """Move focus to the view's primary widget (table, first button)."""

    def on_host(self, host: Host) -> None:
        """Called by the host poller with each fresh snapshot.

        Args:
            host: Detected host resources.
        """
