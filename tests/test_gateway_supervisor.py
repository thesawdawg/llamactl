"""Managed gateway admission, deferred settings, forwarding and concurrency."""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import httpx
import pytest

from llamactl import gateway
from llamactl.gateway import Gateway, GatewayError, GatewayState
from llamactl.gateway_supervisor import AdmissionController, BackendGateway, SupervisorServer
from test_gateway import FakePopen, FakeResponse, RouterHttp, env, started, two_models


class ImmediateRouter(RouterHttp):
    """A native-router fake that completes loads and can evict idle models."""

    def request(self, method: str, path: str, **kwargs: Any) -> FakeResponse:
        """Apply native behavior and finish the requested load immediately.

        Args:
            method: HTTP verb.
            path: Router endpoint.
            **kwargs: Native request arguments.

        Returns:
            Native-style model response.
        """
        result = super().request(method, path, **kwargs)
        if path == "/models/load":
            self.states[kwargs["json"]["model"]] = "loaded"
        return result


@pytest.fixture
def controller_env(env: SimpleNamespace) -> SimpleNamespace:
    """Construct the supervisor over scratch config and an in-memory router.

    Args:
        env: Isolated legacy gateway fixture.

    Returns:
        Controller, native router and fixture state.
    """
    started(env)
    router = ImmediateRouter(env.cfg_dir / "gateway.ini")
    backend = BackendGateway(env.cfg, "http://private", "", http=router)
    controller = AdmissionController(backend, env.cfg_dir / "config.json",
                                     (env.cfg_dir / "gateway.ini").read_text())
    return SimpleNamespace(base=env, router=router, backend=backend, controller=controller)


def test_new_gateway_launches_detached_supervisor_with_private_snapshot(env: SimpleNamespace) -> None:
    """New gateways own request admission without exposing credentials on argv.

    Args:
        env: Isolated process-launch fixture.
    """
    env.cfg.gateway.api_key = "launch-secret"
    managed = Gateway(env.cfg, popen=FakePopen, http=env.http, settle=0)
    state = managed.start()
    command = FakePopen.calls[-1][0]
    assert command[1:3] == ["-m", "llamactl.gateway_supervisor"]
    assert "launch-secret" not in " ".join(command)
    assert state.backend_url and state.control_token and state.runtime_api_key == "launch-secret"
    snapshot = Path(command[-1])
    assert snapshot.stat().st_mode & 0o777 == 0o600
    assert gateway.STATE_FILE.stat().st_mode & 0o777 == 0o600
    assert json.loads(snapshot.read_text())["argv"][-2:] == ["--api-key", "launch-secret"]


def test_managed_control_uses_launch_key_after_saved_key_changes(env: SimpleNamespace) -> None:
    """Pending API-key edits do not break control of an existing managed gateway.

    Args:
        env: Isolated HTTP gateway fixture.
    """
    state = started(env)
    state.backend_url = "http://private"
    state.runtime_api_key, state.control_token = "running-key", "local-secret"
    state.save()
    env.cfg.gateway.api_key = "next-key"
    env.http.routes[("POST", "/__llamactl/control")] = FakeResponse(200, {"data": []})
    env.gw.reload()
    assert env.http.requests[-1]["path"] == "/__llamactl/control"
    assert env.http.clients[-1]["headers"] == {
        "Authorization": "Bearer running-key", "X-Llamactl-Control": "local-secret"}
    assert env.http.requests[-1]["timeout"] >= 1800


def test_external_next_load_applies_saved_settings_without_tui(controller_env: SimpleNamespace) -> None:
    """The next admitted request refreshes config after native automatic unload.

    Args:
        controller_env: Supervisor over the deterministic native-router fake.
    """
    env, router, controller = controller_env.base, controller_env.router, controller_env.controller
    router.states["qwen3-8b"] = "loaded"
    old = router.definitions["qwen3-8b"].copy()
    env.cfg.models[str(env.qwen)].presets["chat"].profile.ctx_size = 32768
    env.cfg.save()
    with controller.lease("qwen3-8b"):
        assert router.definitions["qwen3-8b"] == old
    router.states["qwen3-8b"] = "unloaded"
    with controller.lease("qwen3-8b"):
        assert router.definitions["qwen3-8b"]["ctx-size"] == "32768"
        assert router.states["qwen3-8b"] == "loaded"
    assert router.interruptions == []


def test_cold_load_waits_for_admitted_request_before_eviction(controller_env: SimpleNamespace) -> None:
    """An admitted request cannot be evicted in the gap before HTTP forwarding.

    Args:
        controller_env: Supervisor over the deterministic native-router fake.
    """
    controller, router = controller_env.controller, controller_env.router
    router.states["qwen3-8b"] = "loaded"
    trying, admitted = threading.Event(), threading.Event()

    def other_model() -> None:
        """Attempt a competing cold admission while the first lease is active."""
        trying.set()
        with controller.lease("llama-3.2-1b"):
            admitted.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with controller.lease("qwen3-8b"):
            future = executor.submit(other_model)
            assert trying.wait(1)
            assert not admitted.wait(0.1)
            assert router.states["llama-3.2-1b"] == "unloaded"
        future.result(timeout=2)
    assert admitted.is_set() and router.interruptions == []


def test_loaded_models_accept_concurrent_requests(controller_env: SimpleNamespace) -> None:
    """Inference leases do not serialize requests to already-loaded models.

    Args:
        controller_env: Supervisor over an in-memory native router.
    """
    controller, router = controller_env.controller, controller_env.router
    router.states["qwen3-8b"] = router.states["llama-3.2-1b"] = "loaded"

    def nested() -> bool:
        """Admit a second loaded model while the first response is in flight."""
        with controller.lease("llama-3.2-1b"):
            return True

    with ThreadPoolExecutor(max_workers=1) as executor:
        with controller.lease("qwen3-8b"):
            assert executor.submit(nested).result(timeout=1)


def test_missing_model_cannot_bypass_multi_model_admission(controller_env: SimpleNamespace) -> None:
    """Ambiguous requests fail rather than triggering an unmanaged native default.

    Args:
        controller_env: Supervisor with several registered models.
    """
    with pytest.raises(GatewayError, match="select a model explicitly"):
        with controller_env.controller.lease(None):
            raise AssertionError("ambiguous request was admitted")


@contextmanager
def running_server(server: ThreadingHTTPServer) -> Iterator[str]:
    """Serve an isolated HTTP fake on loopback and close it deterministically.

    Args:
        server: Already-bound fake or supervisor HTTP server.

    Yields:
        Loopback base URL.
    """
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


class StreamingBackend(BaseHTTPRequestHandler):
    """A loopback HTTP response fake, never a model process."""

    def log_message(self, format: str, *args: Any) -> None:
        """Keep fake requests out of test output.

        Args:
            format: HTTP logging template.
            *args: Formatting values.
        """

    def do_POST(self) -> None:
        """Return a byte-for-byte SSE payload from the fake private router."""
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        body = b'data: {"token":"hello"}\n\ndata: [DONE]\n\n'
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_proxy_streams_and_authenticates_before_loading(controller_env: SimpleNamespace) -> None:
    """The public endpoint forwards SSE and enforces separate control credentials.

    Args:
        controller_env: Model admission fixture with no real llama.cpp process.
    """
    native = ThreadingHTTPServer(("127.0.0.1", 0), StreamingBackend)
    with running_server(native) as backend_url:
        public = SupervisorServer(("127.0.0.1", 0), controller_env.controller,
                                  "api-secret", "control-secret", backend_url)
        with running_server(public) as url:
            with httpx.Client(base_url=url) as client:
                denied = client.post("/completion", json={"model": "qwen3-8b"})
                assert denied.status_code == 401
                assert controller_env.router.states["qwen3-8b"] == "unloaded"
                client.headers["Authorization"] = "Bearer api-secret"
                # JSON is recognized independently of Content-Type capitalization/presence.
                response = client.post("/completion", content=b'{"model":"qwen3-8b"}')
                assert response.status_code == 200
                assert response.content == b'data: {"token":"hello"}\n\ndata: [DONE]\n\n'
                assert response.headers["content-type"] == "text/event-stream"
                assert client.post("/__llamactl/control", json={"action": "sync"}).status_code == 403
                assert controller_env.router.states["qwen3-8b"] == "loaded"
                assert controller_env.controller.active == {}
