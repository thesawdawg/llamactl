"""Hardware detection: offload devices from llama.cpp, plus RAM and CPU.

The device list comes from ``llama-server --list-devices`` so it reflects what llama.cpp can
actually offload to, whatever the vendor. ``nvidia-smi`` is the fallback when the binary is
missing or prints no devices. Every subprocess call goes through an injectable ``runner``
(``subprocess.run`` compatible) so tests never start anything.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

GB = 1e9
MiB = 1 << 20
LIST_INTERVAL = 15.0     # seconds between --list-devices runs when refreshing
SMI_TIMEOUT = 5.0
LIST_TIMEOUT = 20.0

Runner = Callable[..., Any]
Clock = Callable[[], float]

# "  CUDA0: NVIDIA GeForce RTX 4060 Ti (16379 MiB, 15223 MiB free)". The name may itself contain
# parentheses, so the memory group is anchored at the end of the line.
_DEVICE_RE = re.compile(
    r"^\s*(?P<id>[A-Za-z][A-Za-z0-9_]*)\s*:\s*(?P<name>.*\S)\s*"
    r"\(\s*(?P<total>\d+)\s*MiB(?:\s*,\s*(?P<free>\d+)\s*MiB\s*free)?\s*\)\s*$")
_NON_GPU_BACKENDS = {"cpu"}


@dataclass(frozen=True)
class Device:
    """One offload device as llama.cpp names it (bytes).

    Attributes:
        id: Identifier printed by llama.cpp, e.g. "CUDA0", "Vulkan1", "Metal".
        backend: The id without trailing digits, lower-cased: "cuda", "vulkan", "metal".
        name: Human readable device name.
        total: Total device memory in bytes.
        free: Free device memory in bytes.
    """

    id: str
    backend: str
    name: str
    total: int
    free: int

    @classmethod
    def of(cls, id: str, name: str, total: int, free: int) -> "Device":
        """Build a device, deriving the backend from the id.

        Args:
            id: Device id as printed by llama.cpp.
            name: Device name.
            total: Total memory in bytes.
            free: Free memory in bytes.

        Returns:
            The device.
        """
        return cls(id, backend_of(id), name, total, free)


@dataclass
class Host:
    """Snapshot of host resources (bytes).

    Attributes:
        gpus: Offload devices (any backend).
        ram_total: System RAM in bytes.
        ram_free: Available system RAM in bytes.
        cpu_name: CPU model string.
        cpu_threads: Logical CPU count.
        cpu_cores: Physical core count.
        unified: True when GPU memory is system RAM (Apple Silicon, integrated GPUs).
    """

    gpus: list[Device] = field(default_factory=list)
    ram_total: int = 0
    ram_free: int = 0
    cpu_name: str = ""
    cpu_threads: int = 0
    cpu_cores: int = 0
    unified: bool = False

    @property
    def vram_total(self) -> int:
        """Total memory over all devices."""
        return sum(g.total for g in self.gpus)

    @property
    def vram_free(self) -> int:
        """Free memory over all devices."""
        return sum(g.free for g in self.gpus)

    def summary(self) -> str:
        """One-line description for the UI."""
        gpu = ", ".join(f"{g.name} {g.total / GB:.1f}G ({g.free / GB:.1f}G free)"
                        for g in self.gpus) or "no GPU"
        return (f"GPU: {gpu} | RAM: {self.ram_total / GB:.1f}G ({self.ram_free / GB:.1f}G free) | "
                f"CPU: {self.cpu_name or '?'} ({self.cpu_threads} threads)")


def backend_of(device_id: str) -> str:
    """Backend name for a device id: trailing digits stripped, lower-cased.

    Args:
        device_id: e.g. "CUDA0".

    Returns:
        e.g. "cuda".
    """
    return re.sub(r"\d+$", "", device_id).lower()


def parse_devices(text: str) -> list[Device]:
    """Parse the output of ``llama-server --list-devices``.

    Lines that do not look like a device (log noise, the "Available devices:" header) are
    ignored, device names may contain parentheses, and the "free" figure is optional.

    Args:
        text: Combined stdout and stderr of the command.

    Returns:
        Devices in the order printed; CPU pseudo-devices are skipped.
    """
    out: list[Device] = []
    for line in text.splitlines():
        m = _DEVICE_RE.match(line)
        if not m:
            continue
        backend = backend_of(m["id"])
        if backend in _NON_GPU_BACKENDS:
            continue
        total = int(m["total"]) * MiB
        free = int(m["free"]) * MiB if m["free"] is not None else total
        out.append(Device(m["id"], backend, m["name"].strip(), total, min(free, total)))
    return out


def _text(proc: Any) -> str:
    """Stdout plus stderr of a finished process, tolerating None and bytes."""
    parts = []
    for attr in ("stdout", "stderr"):
        v = getattr(proc, attr, None)
        if isinstance(v, bytes):
            v = v.decode(errors="replace")
        if v:
            parts.append(v)
    return "\n".join(parts)


def _run(runner: Runner, argv: list[str], timeout: float) -> str:
    """Run a command through `runner`, returning its output or "" on any failure.

    Args:
        runner: subprocess.run compatible callable.
        argv: Command line.
        timeout: Seconds before giving up.

    Returns:
        Combined stdout and stderr, or "" when the command is missing or times out.
    """
    try:
        return _text(runner(argv, capture_output=True, text=True, timeout=timeout))
    except (OSError, subprocess.SubprocessError):
        return ""


def parse_nvidia_smi(text: str) -> list[Device]:
    """Parse ``nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits``.

    Args:
        text: Command output, one GPU per line.

    Returns:
        Devices with ids CUDA0, CUDA1, ... in line order.
    """
    out: list[Device] = []
    for line in text.strip().splitlines():
        try:
            name, total, free = (x.strip() for x in line.rsplit(",", 2))
            out.append(Device.of(f"CUDA{len(out)}", name, int(total) * MiB, int(free) * MiB))
        except ValueError:
            continue
    return out


def nvidia_smi_devices(runner: Runner = subprocess.run) -> list[Device]:
    """Query NVIDIA GPUs with nvidia-smi.

    Args:
        runner: subprocess.run compatible callable.

    Returns:
        Devices named CUDA0..N, or [] when nvidia-smi is missing or fails.
    """
    return parse_nvidia_smi(_run(runner, [
        "nvidia-smi", "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits"], SMI_TIMEOUT))


def _server_bin(bin_dir: str | os.PathLike | None) -> str:
    """Path of llama-server inside `bin_dir`, or the bare name for a PATH lookup."""
    return str(Path(bin_dir) / "llama-server") if bin_dir else "llama-server"


def list_devices(bin_dir: str | os.PathLike | None = None,
                 runner: Runner = subprocess.run) -> list[Device]:
    """Devices llama.cpp can offload to, with an nvidia-smi fallback.

    Args:
        bin_dir: Folder holding llama-server; None looks it up on PATH.
        runner: subprocess.run compatible callable.

    Returns:
        Devices from ``llama-server --list-devices``; when the binary is missing or prints no
        devices, NVIDIA GPUs from nvidia-smi (ids CUDA0..N); otherwise [].
    """
    devices = parse_devices(_run(runner, [_server_bin(bin_dir), "--list-devices"], LIST_TIMEOUT))
    return devices or nvidia_smi_devices(runner)


def physical_cores(cpuinfo: str | None = None) -> int:
    """Count unique (physical id, core id) pairs in /proc/cpuinfo.

    Args:
        cpuinfo: Contents of /proc/cpuinfo; read from disk when None.

    Returns:
        Physical core count; falls back to os.cpu_count() // 2.
    """
    if cpuinfo is None:
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text()
        except OSError:
            cpuinfo = ""
    pairs = set()
    phys = core = None
    for line in cpuinfo.splitlines() + [""]:
        if line.startswith("physical id"):
            phys = line.split(":", 1)[1].strip()
        elif line.startswith("core id"):
            core = line.split(":", 1)[1].strip()
        elif not line.strip():
            if core is not None:
                pairs.add((phys or "0", core))
            phys = core = None
    return len(pairs) or (os.cpu_count() or 2) // 2


def parse_meminfo(text: str) -> tuple[int, int]:
    """Total and available RAM from /proc/meminfo text.

    Args:
        text: Contents of /proc/meminfo.

    Returns:
        (total, available) in bytes; zeros when the keys are missing.
    """
    mem: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            mem[key] = int(parts[0]) * 1024
    return mem.get("MemTotal", 0), mem.get("MemAvailable", mem.get("MemFree", 0))


def parse_vm_stat(text: str) -> int:
    """Reclaimable memory in bytes from macOS ``vm_stat`` output (free + inactive + speculative).

    Args:
        text: Output of vm_stat.

    Returns:
        Bytes, or 0 when the output is not recognised.
    """
    m = re.search(r"page size of (\d+) bytes", text)
    page = int(m[1]) if m else 16384
    pages = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative"):
        m = re.search(rf"{key}:\s+(\d+)", text)
        pages += int(m[1]) if m else 0
    return pages * page


def _sysctl(runner: Runner, name: str) -> str:
    """One ``sysctl -n`` value, or ""."""
    return _run(runner, ["sysctl", "-n", name], SMI_TIMEOUT).strip()


def read_memory(runner: Runner = subprocess.run, platform: str | None = None) -> tuple[int, int]:
    """Total and available system RAM.

    Args:
        runner: subprocess.run compatible callable (macOS only).
        platform: sys.platform override for tests.

    Returns:
        (total, free) bytes; zeros on unsupported platforms or failure.
    """
    platform = platform or sys.platform
    if platform.startswith("linux"):
        try:
            return parse_meminfo(Path("/proc/meminfo").read_text())
        except OSError:
            return 0, 0
    if platform == "darwin":
        total = _sysctl(runner, "hw.memsize")
        free = parse_vm_stat(_run(runner, ["vm_stat"], SMI_TIMEOUT))
        return (int(total) if total.isdigit() else 0), free
    return 0, 0


def read_cpu(runner: Runner = subprocess.run, platform: str | None = None) -> tuple[str, int, int]:
    """CPU name, logical threads and physical cores.

    Args:
        runner: subprocess.run compatible callable (macOS only).
        platform: sys.platform override for tests.

    Returns:
        (name, threads, cores); name "" and cores from os.cpu_count() where unknown.
    """
    platform = platform or sys.platform
    threads = os.cpu_count() or 0
    if platform.startswith("linux"):
        try:
            text = Path("/proc/cpuinfo").read_text()
        except OSError:
            return "", threads, physical_cores("")
        name = next((ln.split(":", 1)[1].strip() for ln in text.splitlines()
                     if ln.startswith("model name")), "")
        return name, threads, physical_cores(text)
    if platform == "darwin":
        cores = _sysctl(runner, "hw.physicalcpu")
        return (_sysctl(runner, "machdep.cpu.brand_string"), threads,
                int(cores) if cores.isdigit() else threads // 2)
    return "", threads, threads // 2


@dataclass
class _Listing:
    """Process-wide memo of one bin_dir's device identity."""

    devices: list[Device]
    at: float


_cache: dict[str, _Listing] = {}


def reset_cache() -> None:
    """Forget cached device identities (tests, or after the llama.cpp build changed)."""
    _cache.clear()


def _key(bin_dir: str | os.PathLike | None) -> str:
    """Cache key for a bin_dir."""
    return str(bin_dir or "")


def _list_cached(bin_dir: str | os.PathLike | None, runner: Runner, clock: Clock) -> list[Device]:
    """Run list_devices and memoise the result with the current time."""
    devices = list_devices(bin_dir, runner)
    _cache[_key(bin_dir)] = _Listing(devices, clock())
    return devices


def detect(bin_dir: str | os.PathLike | None = None, runner: Runner = subprocess.run,
           clock: Clock = time.monotonic) -> Host:
    """Read devices, RAM and CPU.

    The device list (ids, names, totals) is cached per process and bin_dir; later calls only
    refresh free memory, as `refresh` does. An empty result is retried at most every 15 s.

    Args:
        bin_dir: Folder holding llama-server; None looks it up on PATH.
        runner: subprocess.run compatible callable.
        clock: Monotonic clock in seconds (injectable for tests).

    Returns:
        The host snapshot; missing pieces stay zero or empty.
    """
    cached = _cache.get(_key(bin_dir))
    if cached is not None and (cached.devices or clock() - cached.at < LIST_INTERVAL):
        host = _host(list(cached.devices), runner)
        return refresh(host, bin_dir, runner, clock) if cached.devices else host
    return _host(_list_cached(bin_dir, runner, clock), runner)


def _host(devices: list[Device], runner: Runner) -> Host:
    """Assemble a Host from devices plus RAM and CPU readings."""
    total, free = read_memory(runner)
    name, threads, cores = read_cpu(runner)
    return Host(gpus=devices, ram_total=total, ram_free=free, cpu_name=name, cpu_threads=threads,
                cpu_cores=cores, unified=any(d.backend == "metal" for d in devices))


def refresh(host: Host, bin_dir: str | os.PathLike | None = None,
            runner: Runner = subprocess.run, clock: Clock = time.monotonic) -> Host:
    """Update free memory (devices and RAM) on an existing snapshot.

    When every device is CUDA, ``nvidia-smi`` supplies the new free figures (cheap). Otherwise,
    or when nvidia-smi does not match the known devices, ``--list-devices`` runs again, but no
    more than once every 15 s; between runs the device figures are left as they were.

    Args:
        host: Snapshot to refresh; not modified.
        bin_dir: Folder holding llama-server; None looks it up on PATH.
        runner: subprocess.run compatible callable.
        clock: Monotonic clock in seconds (injectable for tests).

    Returns:
        A new Host.
    """
    gpus = host.gpus
    if gpus and all(g.backend == "cuda" for g in gpus):
        smi = nvidia_smi_devices(runner)
        if len(smi) == len(gpus):
            gpus = [replace(g, free=min(s.free, g.total)) for g, s in zip(gpus, smi)]
            return _with(host, gpus, runner)
    entry = _cache.get(_key(bin_dir))
    if entry is None or clock() - entry.at >= LIST_INTERVAL:
        listed = _list_cached(bin_dir, runner, clock)
        if listed or not gpus:
            gpus = listed
    return _with(host, gpus, runner)


def _with(host: Host, gpus: list[Device], runner: Runner) -> Host:
    """Copy of `host` with new devices and freshly read RAM."""
    total, free = read_memory(runner)
    return replace(host, gpus=gpus, ram_total=total or host.ram_total,
                   ram_free=free or host.ram_free,
                   unified=any(d.backend == "metal" for d in gpus))
