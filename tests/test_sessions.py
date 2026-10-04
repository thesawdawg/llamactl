"""Unit tests for port probing and the async launch + loading banner."""
from __future__ import annotations

import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from llamactl import sessions
from llamactl.app import LlamaCtl
from llamactl.sessions import Session, SessionStore, next_free_port, port_free


def test_port_free() -> None:
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        taken = held.getsockname()[1]
        assert not port_free(taken)
    assert port_free(taken)  # freed once the socket closes


def test_next_free_port_skips_taken() -> None:
    assert next_free_port({sessions.BASE_PORT}) == sessions.BASE_PORT + 1


def test_next_free_port_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sessions, "port_free", lambda p, host="127.0.0.1": False)
    with pytest.raises(RuntimeError):
        next_free_port(set())


async def _wait(pilot, cond, tries: int = 60) -> bool:
    for _ in range(tries):
        await pilot.pause(0.1)
        if cond():
            return True
    return False


async def test_launch_worker_and_banner(isolated: SimpleNamespace, tmp_path: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """A mocked launch registers a pending session and shows the banner."""
    log = tmp_path / "srv.log"
    log.write_text("llama_model_loader: load_tensors: offloading 12/33 tensors\n")
    fake = Session("m-8080", "m", "http://127.0.0.1:59998", pid=999999,
                   log=str(log), started=0.0)
    monkeypatch.setattr(SessionStore, "launch_server",
                        lambda self, cfg, model: (self.sessions.append(fake) or fake))
    # process is "alive" while pending; probe fails until we flip the switch
    import llamactl.views.sessions as sv
    monkeypatch.setattr(sv, "probe", lambda url, timeout=1.0: None)
    alive = {"v": True}
    monkeypatch.setattr(Session, "alive", property(lambda s: alive["v"]))
    app = LlamaCtl()
    async with app.run_test(size=(140, 45)) as pilot:
        app.views["sessions"].launch(str(tmp_path / "m.gguf"))
        assert await _wait(pilot, lambda: app.views["sessions"].sess_table.row_count == 1)
        box = app.views["sessions"].query_one("#load_box")
        banner = app.views["sessions"].query_one("#loading_banner")
        assert await _wait(pilot, lambda: box.display is True)
        assert await _wait(pilot, lambda: "Loading m" in str(banner.render()))
        # progress fraction parsed from the log line
        assert await _wait(
            pilot, lambda: app.views["sessions"].query_one("#load_progress").display)
        # now the server answers /props -> ready, banner hides
        monkeypatch.setattr(sv, "probe",
                            lambda url, timeout=1.0: {"model_path": "/x/m.gguf"})
        assert await _wait(pilot, lambda: box.display is False)
        assert await _wait(
            pilot, lambda: app.views["sessions"].state.get("m-8080") == "ready")
        st = app.views["sessions"].sess_table.get_cell("m-8080", "state")
        assert "ready" in str(getattr(st, "plain", st))
