"""Detached admission controller for the public gateway and its private router.

Only this process admits model loads or refreshes the private router's presets.
Inference responses stream outside the admission lock, preserving concurrent clients.
"""
from __future__ import annotations

import hmac
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import parse_qs, urlsplit

import httpx

from . import config, gateway
from .config import Config
from .gateway import FAILED, LOADED, LOADING, SLEEPING, UNLOADED, Gateway, GatewayError

HOP_HEADERS = frozenset({"connection", "keep-alive", "proxy-authenticate",
                         "proxy-authorization", "te", "trailer", "transfer-encoding",
                         "upgrade", "host", "x-llamactl-control"})


class BackendGateway(Gateway):
    """Gateway operations addressed directly to the private router.

    Args:
        cfg: Initial configuration snapshot.
        backend_url: Private router URL.
        api_key: Key captured at launch, independent of later config changes.
    """

    def __init__(self, cfg: Config, backend_url: str, api_key: str,
                 http: Callable[..., Any] = httpx.Client) -> None:
        self.backend_url = backend_url
        self.api_key = api_key
        super().__init__(cfg, http=http)
        self._backend_mode = True
        self.desired_text: str = gateway.render_presets(cfg)

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Request the private router without traversing the public proxy.

        Args:
            method: HTTP verb.
            path: Router endpoint.
            **kwargs: Arguments forwarded to httpx.

        Returns:
            Router response.

        Raises:
            GatewayError: If the private router cannot be reached.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            with self._http(base_url=self.backend_url, headers=headers,
                              timeout=gateway.HTTP_TIMEOUT) as client:
                return client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise GatewayError(f"private router unavailable: {exc}") from exc

    def _saved_presets(self) -> str:
        """Return the controller's current desired configuration.

        Returns:
            Desired INI, without replacing live definitions.
        """
        return self.desired_text


class AdmissionController:
    """Serialize configuration reconciliation and model load admission.

    Args:
        backend: Private router operations, injectable in tests.
        config_file: Persisted user configuration.
        initial: Initial desired preset INI.
        clock: Monotonic timer for bounded model loading.
        sleep: Injectable delay between load checks.
    """

    def __init__(self, backend: BackendGateway, config_file: Path, initial: str,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.backend = backend
        self.config_file = config_file
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.active: dict[str, int] = {}
        self.loading: set[str] = set()
        self._clock, self._sleep = clock, sleep
        self._config_mtime = self._mtime()
        self.backend.desired_text = initial

    def _mtime(self) -> int:
        """Return the current config revision, or zero when absent.

        Returns:
            File modification time in nanoseconds.
        """
        try:
            return self.config_file.stat().st_mtime_ns
        except OSError:
            return 0

    def refresh(self, presets: str | None = None) -> None:
        """Refresh desired definitions without changing a running process.

        Args:
            presets: Explicit parent-rendered INI, preserving support filtering.

        Raises:
            GatewayError: If newly saved configuration cannot be read.
        """
        revision = self._mtime()
        if presets is not None:
            self.backend.desired_text = presets
            self._config_mtime = revision
        elif revision and revision != self._config_mtime:
            try:
                cfg = Config.from_dict(json.loads(self.config_file.read_text()))
            except (OSError, ValueError, TypeError) as exc:
                raise GatewayError(f"cannot read saved presets: {exc}") from exc
            self.backend.cfg = cfg
            self.backend.desired_text = gateway.render_presets(cfg)
            self._config_mtime = revision

    def control(self, action: str, model: str = "", presets: str | None = None) -> dict[str, Any]:
        """Perform a management operation inside the admission boundary.

        Args:
            action: Sync, reload, load or unload.
            model: Served name for model operations.
            presets: Explicit desired definitions for reload.

        Returns:
            Native router model list after the operation.

        Raises:
            GatewayError: On invalid operations or router errors.
        """
        with self.lock:
            self.refresh(presets)
            if action == "reload":
                self.backend.reload()
            elif action == "sync":
                self.backend.synchronize()
            elif action == "load" and model:
                self._prepare(model)
            elif action == "unload" and model:
                self.backend.unload(model)
            else:
                raise GatewayError("invalid gateway control action")
            return self.backend._call("GET", "/models")

    def _prepare(self, model: str | None) -> str:
        """Apply pending definitions before admitting an on-demand model load.

        Args:
            model: Requested served name, or None for router default selection.

        Returns:
            Resolved served name with its model loaded.

        Raises:
            GatewayError: If model load fails or exceeds its deadline.
        """
        with self.condition:
            self.refresh()
            models = self.backend.synchronize()
            if model is None and len(models) == 1:
                model = models[0].name
            if model is None:
                raise GatewayError("select a model explicitly when the gateway has multiple models")
            item = next((item for item in models if item.name == model), None)
            if item is None:
                raise GatewayError(f"unknown model '{model}'")
            if item.state in (LOADED, SLEEPING):
                return model
            deadline = self._clock() + 1800
            # Reserving a loaded model before forwarding closes the admission/HTTP gap.
            # A cold load could evict it, so defer cold loads until leased work finishes.
            while self.active or self.loading:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise GatewayError("model load is waiting for active gateway requests")
                self.condition.wait(timeout=min(remaining, 0.1))
                models = self.backend.synchronize()
                item = next((item for item in models if item.name == model), None)
                if item is not None and item.state in (LOADED, SLEEPING):
                    return model
            self.refresh()
            models = self.backend.synchronize()
            item = next((item for item in models if item.name == model), None)
            if item is None:
                raise GatewayError(f"unknown model '{model}'")
            if item.state in (LOADED, SLEEPING):
                return model
            self.loading.add(model)
            try:
                if item.state != LOADING:
                    self.backend.load(model)
                while self._clock() < deadline:
                    snapshot = self.backend._parse(self.backend._call("GET", "/models"))
                    item = next((item for item in snapshot if item.name == model), None)
                    if item is None or item.state == FAILED:
                        raise GatewayError(f"model '{model}' failed to load")
                    if item.state in (LOADED, SLEEPING):
                        return model
                    self.condition.wait(timeout=0.05)
                raise GatewayError(f"model '{model}' did not load within 1800 seconds")
            finally:
                self.loading.discard(model)
                self.condition.notify_all()

    @contextmanager
    def lease(self, model: str | None) -> Iterator[None]:
        """Reserve a loaded model across forwarding without locking inference.

        Args:
            model: Explicit served name, or a unique default.

        Yields:
            Control while the request owns an active admission reservation.

        Raises:
            GatewayError: If the requested model cannot be admitted.
        """
        with self.condition:
            name = self._prepare(model)
            self.active[name] = self.active.get(name, 0) + 1
        try:
            yield
        finally:
            with self.condition:
                self.active[name] -= 1
                if not self.active[name]:
                    del self.active[name]
                self.condition.notify_all()


class SupervisorServer(ThreadingHTTPServer):
    """Threaded public gateway server with one shared admission controller."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], controller: AdmissionController,
                 api_key: str, control_token: str, backend_url: str) -> None:
        """Bind the gateway's public address.

        Args:
            address: Public host and port.
            controller: Shared model admission state.
            api_key: API key captured at gateway launch.
            control_token: Private local management credential.
            backend_url: Private router URL.
        """
        self.controller = controller
        self.api_key, self.control_token = api_key, control_token
        self.backend_url = backend_url
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, SupervisorHandler)


class SupervisorHandler(BaseHTTPRequestHandler):
    """Authenticate, admit and stream one public gateway request."""

    server: SupervisorServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        """Avoid recording request bodies, query credentials or API keys.

        Args:
            format: Standard HTTP log template.
            *args: Formatting values.
        """

    def _json(self, status: int, body: dict[str, Any]) -> None:
        """Send a framed JSON response.

        Args:
            status: HTTP status code.
            body: Serializable response object.
        """
        payload = json.dumps(body).encode()
        self.response_started = True
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True

    def _body(self) -> bytes:
        """Read a length-framed or chunked body without imposing a prompt-size cap.

        Returns:
            Request body with chunk framing removed.

        Raises:
            ValueError: On malformed framing.
        """
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            pieces: list[bytes] = []
            while True:
                size = int(self.rfile.readline().split(b";", 1)[0], 16)
                if size < 0:
                    raise ValueError("negative chunk size")
                if not size:
                    while self.rfile.readline().strip():
                        pass
                    return b"".join(pieces)
                part = self.rfile.read(size)
                if len(part) != size or self.rfile.read(2) != b"\r\n":
                    raise ValueError("incomplete HTTP chunk")
                pieces.append(part)
        size = int(self.headers.get("Content-Length", "0"))
        if size < 0:
            raise ValueError("negative content length")
        result = self.rfile.read(size)
        if len(result) != size:
            raise ValueError("incomplete HTTP body")
        return result

    def _authorized(self) -> bool:
        """Check launch-time API credentials before any model operation.

        Returns:
            Whether the request may access the gateway.
        """
        expected = f"Bearer {self.server.api_key}"
        return not self.server.api_key or hmac.compare_digest(
            self.headers.get("Authorization", ""), expected)

    def _handle(self) -> None:
        """Route management requests or admit and forward public inference.

        Returns:
            None.
        """
        self.response_started = False
        if not self._authorized():
            return self._json(401, {"error": "invalid API key"})
        try:
            body = self._body()
            parsed = urlsplit(self.path)
            data = json.loads(body) if body else {}
            if not isinstance(data, dict):
                data = {}
            if parsed.path.startswith("/__llamactl/"):
                token = self.headers.get("X-Llamactl-Control", "")
                if not token or not hmac.compare_digest(token, self.server.control_token):
                    return self._json(403, {"error": "private gateway control"})
                if self.command != "POST" or parsed.path != "/__llamactl/control":
                    return self._json(404, {"error": "unknown control endpoint"})
                result = self.server.controller.control(str(data.get("action", "")),
                           str(data.get("model", "")), data.get("presets"))
                return self._json(200, result)
            if parsed.path == "/models" and "reload" in parse_qs(parsed.query):
                return self._json(200, self.server.controller.control("reload"))
            if parsed.path in ("/models/load", "/models/unload") and self.command == "POST":
                action = "load" if parsed.path.endswith("/load") else "unload"
                return self._json(200, self.server.controller.control(action, str(data.get("model", ""))))
            if self.command == "POST":
                model = data.get("model")
                with self.server.controller.lease(model if isinstance(model, str) else None):
                    self._forward(body)
            else:
                self._forward(body)
        except (ValueError, TypeError) as exc:
            if not self.response_started:
                self._json(400, {"error": str(exc)})
        except GatewayError as exc:
            if not self.response_started:
                self._json(503, {"error": str(exc)})
        except (httpx.HTTPError, OSError) as exc:
            if not self.response_started and not self.wfile.closed:
                try:
                    self._json(502, {"error": f"gateway forwarding failed: {type(exc).__name__}"})
                except OSError:
                    pass
            self.close_connection = True

    def _forward(self, body: bytes) -> None:
        """Stream the private response without holding the model admission lock.

        Args:
            body: Decoded HTTP request body.

        Raises:
            httpx.HTTPError: If the upstream request fails.
            OSError: If the client disconnects.
        """
        headers = {key: value for key, value in self.headers.items()
                   if key.lower() not in HOP_HEADERS and key.lower() != "content-length"}
        with httpx.Client(timeout=httpx.Timeout(None, connect=10)) as client:
            with client.stream(self.command, self.server.backend_url + self.path,
                               headers=headers, content=body) as response:
                self.send_response(response.status_code)
                self.response_started = True
                for key, value in response.headers.items():
                    if key.lower() not in HOP_HEADERS:
                        self.send_header(key, value)
                self.send_header("Connection", "close")
                self.end_headers()
                for chunk in response.iter_raw():
                    self.wfile.write(chunk)
                    self.wfile.flush()
        self.close_connection = True

    do_GET = _handle
    do_POST = _handle
    do_PUT = _handle
    do_DELETE = _handle
    do_OPTIONS = _handle
    do_HEAD = _handle


def run(launch_file: Path) -> int:
    """Run the detached public supervisor and its private llama.cpp router.

    Args:
        launch_file: Private startup snapshot created by Gateway.start.

    Returns:
        The router exit status, or zero after requested shutdown.
    """
    launch = json.loads(launch_file.read_text())
    cfg = Config.from_dict(launch["config"])
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    argv = list(launch["argv"])
    argv[argv.index("--host") + 1] = "127.0.0.1"
    argv[argv.index("--port") + 1] = str(port)
    backend_url = f"http://127.0.0.1:{port}"
    backend = BackendGateway(cfg, backend_url, cfg.gateway.api_key)
    initial = gateway.PRESETS_FILE.read_text()
    controller = AdmissionController(backend, config.CONFIG_FILE, initial)
    server = SupervisorServer((launch["host"], int(launch["port"])), controller,
                              cfg.gateway.api_key, launch["control_token"], backend_url)
    try:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL)
    except OSError:
        server.server_close()
        raise
    state = gateway.GatewayState(os.getpid(), f"http://{gateway.network.probe_host(launch['host'])}:"
                                f"{launch['port']}", str(gateway.LOG_DIR / gateway.LOG_NAME),
                                time.time(), launch["host"], gateway._preset_sections(initial),
                                backend_url, cfg.gateway.api_key, launch["control_token"])
    try:
        with gateway._runtime_lock():
            state.save()
    except OSError:
        process.terminate()
        server.server_close()
        raise

    def shutdown(signum: int, frame: Any) -> None:
        """Stop accepting requests after a process-group shutdown signal.

        Args:
            signum: Received signal.
            frame: Interrupted frame.
        """
        threading.Thread(target=server.shutdown, daemon=True).start()

    def watch() -> None:
        """Stop the public gateway if the private router exits."""
        process.wait()
        server.shutdown()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    threading.Thread(target=watch, daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return process.returncode or 0


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1])))
