"""Process usage probing on a fake /proc tree, and `served_usage` attribution."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from llamactl.gateway import LOADED, SLEEPING, UNLOADED, GatewayModel, GatewayStatus
from llamactl.sessions import Session
from llamactl.usage import (SRC_ESTIMATE, SRC_PROCESS, SRC_TUNED, UsageProbe, format_cpu,
                            format_mem, served_usage)

CLK_TCK = os.sysconf("SC_CLK_TCK")


def _stat(pid: int, comm: str, ppid: int, utime: int, stime: int) -> str:
    """A /proc/<pid>/stat line; utime/stime sit at index 11/12 after the comm field."""
    return f"{pid} ({comm}) S {ppid} " + " ".join(["0"] * 9) + f" {utime} {stime} 0 0"


def _proc_tree(root: Path) -> None:
    """A router (pid 100) plus one child (pid 200) whose comm holds a space and parens."""
    for pid, comm, ppid, utime, stime, rss_kb, argv in (
            (100, "llama-router", 1, 100, 20, 51200, ["/bin/router", "--port", "8080"]),
            (200, "llama (srv) x", 100, 300, 100, 409600,
             ["/bin/llama-server", "--port", "45349", "--alias", "qwen"])):
        d = root / str(pid)
        d.mkdir(parents=True)
        (d / "stat").write_text(_stat(pid, comm, ppid, utime, stime))
        (d / "status").write_text(f"Name:\t{comm}\nVmRSS:\t{rss_kb} kB\n")
        (d / "cmdline").write_bytes(("\0".join(argv) + "\0").encode())


class _Clock:
    """Monotonic clock that only moves when the test moves it."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class FakeProbe:
    """UsageProbe double: scripted children, samples and GPU memory."""

    def __init__(self, kids: dict[int, list[str]] | None = None,
                 vram: dict[int, int] | None = None,
                 rss: int = 768 << 20, cpu: float | None = 12.5) -> None:
        self.kids = kids or {}
        self.vram = vram or {}
        self.rss, self.cpu = rss, cpu

    def children(self, pid: int) -> dict[int, list[str]]:
        return dict(self.kids)

    def sample(self, pids: object) -> dict[int, tuple[int, float | None]]:
        return {p: (self.rss, self.cpu) for p in pids}  # type: ignore[union-attr]

    def gpu_memory(self) -> dict[int, int]:
        return dict(self.vram)


def test_children_reads_ppid_and_argv(tmp_path: Path) -> None:
    _proc_tree(tmp_path)
    probe = UsageProbe(proc=tmp_path, platform="linux")
    assert probe.children(100) == {
        200: ["/bin/llama-server", "--port", "45349", "--alias", "qwen"]}
    assert probe.children(200) == {}


def test_sample_reports_rss_then_cpu_delta(tmp_path: Path) -> None:
    _proc_tree(tmp_path)
    clock = _Clock()
    probe = UsageProbe(proc=tmp_path, platform="linux", clock=clock)
    first = probe.sample([100, 200])
    assert first[200] == (409600 << 10, None)          # no earlier sample: cpu unknown
    clock.t += 0.5
    (tmp_path / "200" / "stat").write_text(_stat(200, "llama (srv) x", 100, 350, 100))
    second = probe.sample([100, 200])
    assert second[200][0] == 409600 << 10
    assert second[200][1] == pytest.approx(50 / CLK_TCK / 0.5 * 100)
    assert set(probe.sample([200])) == {200}           # vanished pids are dropped


def _served(kids: dict[int, list[str]], models: list[GatewayModel],
            sessions: list[Session] | None = None,
            estimate=lambda name: (None, ""), vram: dict[int, int] | None = None) -> dict:
    status = GatewayStatus(True, "http://127.0.0.1:8080", True, 100)
    return served_usage(FakeProbe(kids, vram), status, models, sessions or (), estimate)


def test_served_usage_prefers_process_vram() -> None:
    model = GatewayModel("qwen", LOADED, args=("--port", "45349"))
    out = _served({200: ["/bin/llama-server", "--port", "45349"]}, [model],
                  estimate=lambda name: (111, SRC_TUNED), vram={200: 8 << 30})
    u = out["gw:qwen"]
    assert (u.pid, u.vram, u.vram_source) == (200, 8 << 30, SRC_PROCESS)
    assert u.rss == 768 << 20 and u.cpu_pct == 12.5


def test_served_usage_falls_back_to_tuned_then_estimate() -> None:
    model = GatewayModel("qwen", LOADED, args=("--port", "45349"))
    kids = {200: ["/bin/llama-server", "--port", "45349"]}
    u = _served(kids, [model], estimate=lambda name: (7 << 30, SRC_TUNED))["gw:qwen"]
    assert (u.vram, u.vram_source) == (7 << 30, SRC_TUNED)
    u = _served(kids, [model], estimate=lambda name: (5 << 30, SRC_ESTIMATE))["gw:qwen"]
    assert (u.vram, u.vram_source) == (5 << 30, SRC_ESTIMATE)


def test_served_usage_sleeping_model_never_estimates() -> None:
    model = GatewayModel("qwen", SLEEPING, args=("--port", "45349"))
    out = _served({200: ["/bin/llama-server", "--port", "45349"]}, [model],
                  estimate=lambda name: (9, SRC_ESTIMATE))
    u = out["gw:qwen"]
    assert u.vram is None and u.vram_source == "" and u.pid == 200


def test_served_usage_alias_fallback_sessions_and_skips() -> None:
    loaded = GatewayModel("qwen", LOADED)                  # no args: match --alias == name
    parked = GatewayModel("other", UNLOADED)
    out = _served({200: ["--alias", "qwen"], 300: ["--alias", "other"]},
                  [loaded, parked])
    assert set(out) == {"gw:qwen"} and out["gw:qwen"].pid == 200
    managed = Session("m-9000", "m", "http://127.0.0.1:9000", pid=400)
    attached = Session("ext-1", "e", "http://h:1", external=True)
    out = _served({}, [], sessions=[managed, attached])
    assert set(out) == {"s:m-9000"} and out["s:m-9000"].pid == 400


def test_format_helpers() -> None:
    assert format_cpu(12.4) == "12%" and format_cpu(None) == "-"
    assert format_mem(420_000_000) == "420 MB" and format_mem(1_234_000_000) == "1.2 GB"
    assert format_mem(5 << 30, SRC_TUNED) == "~5.4 GB" and format_mem(None) == "-"
