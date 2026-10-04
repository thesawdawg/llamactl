"""Host resource detection and a rough 'can this model run here?' estimate."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field

GB = 1e9


@dataclass
class Gpu:
    """One NVIDIA GPU (bytes)."""

    name: str
    total: int
    free: int


@dataclass
class Host:
    """Snapshot of host resources (bytes)."""

    gpus: list[Gpu] = field(default_factory=list)
    ram_total: int = 0
    ram_free: int = 0
    cpu_name: str = ""
    cpu_threads: int = 0
    cpu_cores: int = 0

    @property
    def vram_total(self) -> int:
        return sum(g.total for g in self.gpus)

    @property
    def vram_free(self) -> int:
        return sum(g.free for g in self.gpus)

    def summary(self) -> str:
        """One-line description for the UI."""
        gpu = ", ".join(f"{g.name} {g.total / GB:.1f}G ({g.free / GB:.1f}G free)" for g in self.gpus) or "no NVIDIA GPU"
        return (f"GPU: {gpu} | RAM: {self.ram_total / GB:.1f}G ({self.ram_free / GB:.1f}G free) | "
                f"CPU: {self.cpu_name or '?'} ({self.cpu_threads} threads)")


def _gpus() -> list[Gpu]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    gpus = []
    for line in out.strip().splitlines():
        name, total, free = (x.strip() for x in line.rsplit(",", 2))
        gpus.append(Gpu(name, int(total) * 2**20, int(free) * 2**20))
    return gpus


def physical_cores() -> int:
    """Count unique (physical id, core id) pairs in /proc/cpuinfo.

    Returns:
        Physical core count; falls back to os.cpu_count() // 2.
    """
    pairs = set()
    phys = core = None
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("physical id"):
                phys = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                core = line.split(":", 1)[1].strip()
            elif not line.strip():
                if core is not None:
                    pairs.add((phys or "0", core))
                phys = core = None
    except OSError:
        pass
    return len(pairs) or (os.cpu_count() or 2) // 2


def detect() -> Host:
    """Read GPU, RAM and CPU info (Linux). Missing pieces stay zero/empty."""
    host = Host(gpus=_gpus(), cpu_threads=os.cpu_count() or 0, cpu_cores=physical_cores())
    try:
        mem = {k: int(v.split()[0]) * 1024 for k, v in (l.split(":", 1) for l in open("/proc/meminfo"))}
        host.ram_total, host.ram_free = mem["MemTotal"], mem["MemAvailable"]
        host.cpu_name = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
    except (OSError, KeyError):
        pass
    return host


def estimate(host: Host, size: int) -> tuple[str, str]:
    """Return (short verdict, explanation) for a model whose files total `size` bytes.

    Uses a weights-only budget (KV unknown until the header is parsed).
    Long contexts, MoE offload tricks and multi-GPU splits can change the result.
    """
    from .config import Profile
    from .recommend import estimate_budget, placeholder_info

    need = estimate_budget(placeholder_info(size), Profile(), host).total()
    g = lambda b: f"{b / GB:.1f}G"
    if not host.ram_total and not host.vram_total:
        return "?", "Host resources unknown"
    if host.vram_total >= need:
        if host.vram_free >= need:
            return "GPU", f"Needs ~{g(need)}; fits fully in VRAM ({g(host.vram_free)} free). Fast."
        return "GPU*", f"Needs ~{g(need)}; fits in VRAM ({g(host.vram_total)}) but only {g(host.vram_free)} free now - stop other models first."
    if host.vram_total + host.ram_free >= need and host.vram_total:
        return "GPU+RAM", f"Needs ~{g(need)}; VRAM {g(host.vram_total)} + RAM. Partial offload (lower -ngl): works but slower."
    if host.ram_free >= need:
        return "CPU", f"Needs ~{g(need)}; fits in RAM ({g(host.ram_free)} free) but no GPU offload. Slow."
    return "NO", f"Needs ~{g(need)}; only {g(host.vram_total)} VRAM + {g(host.ram_free)} free RAM. Will not run reliably."
