"""Textual TUI for launching, monitoring and chatting with llama.cpp servers."""
from __future__ import annotations

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import ContentSwitcher, Footer, Header, Label, ListItem, ListView

from . import hostinfo
from .config import Config
from .gguf import ModelInfoCache
from .sessions import SessionStore
from .views.base import BaseView
from .views.hf import HFView
from .views.models import ModelsView
from .views.sessions import SessionsView
from .views.settings import SettingsView
from .widgets.gauges import ResourceGauges

VIEWS = (("models", "Models"), ("sessions", "Sessions"), ("hf", "Hugging Face"),
         ("settings", "Settings"))


class LlamaCtl(App):
    """Main application: sidebar navigation, gauges, status bar, views."""

    TITLE = "llamactl"
    CSS = """
    #shell { height: 1fr; }
    #nav { width: 14; }
    #main { width: 1fr; height: 1fr; }
    #main > * { height: 1fr; }
    #gauges { height: 1; }
    #status { height: auto; padding: 0 1; }
    """
    BINDINGS = [
        Binding("1", "view('models')", "Models", show=False),
        Binding("2", "view('sessions')", "Sessions", show=False),
        Binding("3", "view('hf')", "Hugging Face", show=False),
        Binding("4", "view('settings')", "Settings", show=False),
        Binding("r", "rescan", "Rescan"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.cfg = Config.load()
        self.store = SessionStore()
        self.info_cache = ModelInfoCache()
        self.host = hostinfo.Host()
        self.views: dict[str, BaseView] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        yield ResourceGauges(id="gauges")
        with Horizontal(id="shell"):
            yield ListView(*[ListItem(Label(label), id=f"nav-{name}") for name, label in VIEWS],
                           id="nav")
            with ContentSwitcher(id="main", initial="models"):
                for name, _ in VIEWS:
                    v = {"models": ModelsView, "sessions": SessionsView,
                         "hf": HFView, "settings": SettingsView}[name]()
                    v.id = name
                    self.views[name] = v
                    yield v
        yield Label("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#nav", ListView).index = 0
        self.poll_host()
        self.set_interval(3.0, self.poll_host)
        self.views["models"].on_activate()

    @property
    def sessions(self) -> SessionsView:
        """The sessions view."""
        return self.views["sessions"]  # type: ignore[return-value]

    def say(self, msg: str) -> None:
        """Show a one-line status message (user actions only)."""
        self.query_one("#status", Label).update(msg)

    def switch_view(self, name: str) -> None:
        """Activate a view by id and sync the sidebar cursor.

        Args:
            name: One of models / sessions / hf / settings.
        """
        self.query_one("#main", ContentSwitcher).current = name
        nav = self.query_one("#nav", ListView)
        idx = [n for n, _ in VIEWS].index(name)
        if nav.index != idx:
            nav.index = idx
        self.views[name].on_activate()

    def action_view(self, name: str) -> None:
        self.switch_view(name)

    @on(ListView.Highlighted, "#nav")
    def _nav(self, event: ListView.Highlighted) -> None:
        name = event.item.id.removeprefix("nav-") if event.item else ""
        if name and name in self.views and self.query_one("#main", ContentSwitcher).current != name:
            self.query_one("#main", ContentSwitcher).current = name
            self.views[name].on_activate()

    @on(ListView.Selected, "#nav")
    def _nav_selected(self, event: ListView.Selected) -> None:
        # Enter on a nav item: also move focus into the view
        if event.item:
            self.switch_view(event.item.id.removeprefix("nav-"))

    @work(thread=True, exclusive=True, group="host")
    def poll_host(self) -> None:
        """Detect host resources off the UI thread and push to gauges + active view."""
        host = hostinfo.detect()
        self.call_from_thread(self._host_update, host)

    def _host_update(self, host: hostinfo.Host) -> None:
        self.host = host
        self.query_one("#gauges", ResourceGauges).update_host(host)
        current = self.query_one("#main", ContentSwitcher).current
        if current in self.views:
            self.views[current].on_host(host)

    def action_rescan(self) -> None:
        self.views["models"].rescan()


def main() -> None:
    """Entry point."""
    LlamaCtl().run()


if __name__ == "__main__":
    main()
