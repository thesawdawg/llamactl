"""BaseView: what the shell expects of every view, with do-nothing defaults."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from textual.widget import Widget

from ..hardware import Host
from ..serving import ServingSnapshot

if TYPE_CHECKING:
    from ..app import LlamaCtl


class BaseView(Widget):
    """A ContentSwitcher pane with access to the app's services and its status line.

    The shell calls the ``on_*`` hooks on every view (shown or not), always on the UI thread;
    a view overrides the ones it cares about.
    """

    @property
    def app(self) -> "LlamaCtl":
        """The running app (typed)."""
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
        """A fresh host snapshot arrived.

        Args:
            host: Detected host resources.
        """

    def on_models(self, models: list[Path]) -> None:
        """The model folders were scanned; headers are still being read.

        Args:
            models: Model files found.
        """

    def on_info(self, path: Path) -> None:
        """The header of one model was read (or turned out unreadable).

        Args:
            path: The model file; ``app.infos`` has its metadata when it could be read.
        """

    def on_scan_done(self) -> None:
        """Every header of the current scan was read."""

    def on_serving(self, snapshot: ServingSnapshot) -> None:
        """The serving state changed (gateway models, sessions, gateway process).

        Args:
            snapshot: The gateway and sessions now.
        """

    def on_presets_changed(self, paths: Iterable[str]) -> None:
        """Presets (or the build they are judged against) changed.

        Args:
            paths: Models whose presets changed; empty means "any of them".
        """

    def on_config_changed(self) -> None:
        """Settings were saved (folders, headroom, gateway settings, default intent)."""
