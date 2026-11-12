"""Device parsing, nvidia-smi fallback, detect/refresh caching. No process is ever started."""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from llamactl import hardware
from llamactl.hardware import Device, Host

MiB = 1 << 20

REAL = """\
Available devices:
  CUDA0: NVIDIA GeForce RTX 4060 Ti (16379 MiB, 15223 MiB free)
"""

NOISY = """\
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 1: NVIDIA RTX A2000 (Laptop GPU), compute capability 8.6, VMM: yes
load_backend: loaded CUDA backend from /x/libggml-cuda.so
Available devices:
  CUDA0: NVIDIA GeForce RTX 3090 (24576 MiB, 20000 MiB free)
  CUDA1: NVIDIA RTX A2000 (Laptop GPU) (8192 MiB, 7000 MiB free)
  Vulkan0: AMD Radeon (RADV NAVI31) (radv) (24560 MiB, 24000 MiB free)
  Metal: Apple M2 Max (49152 MiB, 49151 MiB free)
  MTL1: Apple thing (1024 MiB)
trailing log line: done (it took 3 MiB)
"""

SMI = "NVIDIA GeForce RTX 4060 Ti, 16380, 15000\nNVIDIA, With Comma, 8192, 100\n"


def proc(stdout: str = "", stderr: str = "") -> SimpleNamespace:
    """A fake CompletedProcess."""
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=0)


class FakeRunner:
    """Records calls; answers by the first word of the command."""

    def __init__(self, server: str | Exception = "", smi: str | Exception = "") -> None:
        self.server, self.smi = server, smi
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], **kw: object) -> SimpleNamespace:
        self.calls.append(list(argv))
        answer = self.smi if argv[0] == "nvidia-smi" else self.server
        if argv[0] not in ("nvidia-smi",) and not argv[0].endswith("llama-server"):
            return proc()  # sysctl, vm_stat, ...
        if isinstance(answer, Exception):
            raise answer
        return proc(answer)

    def count(self, name: str) -> int:
        """Number of calls whose executable ends with `name`."""
        return sum(1 for c in self.calls if c[0].endswith(name))


class Clock:
    """Manually advanced clock."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    hardware.reset_cache()


def test_parse_real_output() -> None:
    (d,) = hardware.parse_devices(REAL)
    assert d == Device("CUDA0", "cuda", "NVIDIA GeForce RTX 4060 Ti", 16379 * MiB, 15223 * MiB)


def test_parse_noise_and_odd_names() -> None:
    devs = hardware.parse_devices(NOISY)
    assert [d.id for d in devs] == ["CUDA0", "CUDA1", "Vulkan0", "Metal", "MTL1"]
    assert [d.backend for d in devs] == ["cuda", "cuda", "vulkan", "metal", "mtl"]
    assert devs[1].name == "NVIDIA RTX A2000 (Laptop GPU)"
    assert devs[2].name == "AMD Radeon (RADV NAVI31) (radv)" and devs[2].total == 24560 * MiB
    assert devs[4].free == devs[4].total  # "free" figure missing -> assume all free


def test_parse_empty_and_cpu() -> None:
    assert hardware.parse_devices("") == []
    assert hardware.parse_devices("Available devices:\n  CPU: Some CPU (64000 MiB, 1 MiB free)\n") == []


def test_backend_of() -> None:
    assert hardware.backend_of("CUDA12") == "cuda" and hardware.backend_of("Metal") == "metal"


def test_list_devices_reads_stderr_too() -> None:
    run = lambda argv, **kw: proc("", REAL)
    assert hardware.list_devices("/b", run)[0].id == "CUDA0"


def test_list_devices_uses_bin_dir() -> None:
    r = FakeRunner(server=REAL)
    hardware.list_devices("/opt/llama/bin", r)
    assert r.calls == [["/opt/llama/bin/llama-server", "--list-devices"]]
    r2 = FakeRunner(server=REAL)
    hardware.list_devices(None, r2)
    assert r2.calls == [["llama-server", "--list-devices"]]


@pytest.mark.parametrize("server", [FileNotFoundError("no binary"), "no devices here",
                                    subprocess.TimeoutExpired("x", 1)])
def test_nvidia_smi_fallback(server: str | Exception) -> None:
    r = FakeRunner(server=server, smi=SMI)
    devs = hardware.list_devices("/b", r)
    assert [d.id for d in devs] == ["CUDA0", "CUDA1"]
    assert devs[1].name == "NVIDIA, With Comma" and devs[0].total == 16380 * MiB
    assert r.count("nvidia-smi") == 1


def test_no_gpu_at_all() -> None:
    r = FakeRunner(server=FileNotFoundError(), smi=FileNotFoundError())
    assert hardware.list_devices("/b", r) == []


def test_binary_with_devices_skips_smi() -> None:
    r = FakeRunner(server=REAL, smi=SMI)
    hardware.list_devices("/b", r)
    assert r.count("nvidia-smi") == 0


def test_host_properties_and_summary() -> None:
    h = Host(gpus=[Device.of("CUDA0", "A", 10 * 10**9, 4 * 10**9),
                   Device.of("CUDA1", "B", 6 * 10**9, 1 * 10**9)],
             ram_total=32 * 10**9, ram_free=8 * 10**9, cpu_name="X", cpu_threads=16)
    assert h.vram_total == 16 * 10**9 and h.vram_free == 5 * 10**9
    s = h.summary()
    assert "A 10.0G (4.0G free)" in s and "B 6.0G" in s and "CPU: X (16 threads)" in s
    assert "no GPU" in Host().summary()


def test_detect_caches_identity_and_refreshes_free_with_smi() -> None:
    clock = Clock()
    r = FakeRunner(server=REAL, smi="NVIDIA GeForce RTX 4060 Ti, 16379, 9000\n")
    h1 = hardware.detect("/b", r, clock)
    assert h1.gpus[0].free == 15223 * MiB and not h1.unified
    assert h1.cpu_threads > 0
    h2 = hardware.detect("/b", r, clock)
    assert r.count("llama-server") == 1          # identity cached for the process
    assert h2.gpus[0].free == 9000 * MiB         # but free memory is live
    assert h2.gpus[0].name == h1.gpus[0].name


def test_refresh_cuda_uses_smi_only() -> None:
    clock = Clock()
    host = Host(gpus=[Device.of("CUDA0", "G", 100 * MiB, 100 * MiB)], ram_total=1, ram_free=1)
    r = FakeRunner(server=REAL, smi="G, 100, 40\n")
    for _ in range(3):
        clock.t += 100
        host = hardware.refresh(host, "/b", r, clock)
    assert host.gpus[0].free == 40 * MiB
    assert r.count("llama-server") == 0 and r.count("nvidia-smi") == 3


def test_refresh_smi_free_never_exceeds_total() -> None:
    host = Host(gpus=[Device.of("CUDA0", "G", 100 * MiB, 100 * MiB)])
    out = hardware.refresh(host, "/b", FakeRunner(smi="G, 100, 500\n"), Clock())
    assert out.gpus[0].free == 100 * MiB


def test_refresh_non_cuda_rate_limited() -> None:
    clock = Clock()
    vk = "Available devices:\n  Vulkan0: Card (1000 MiB, {} MiB free)\n"
    r = FakeRunner(server=vk.format(900))
    host = hardware.detect("/b", r, clock)
    assert host.gpus[0].free == 900 * MiB and r.count("llama-server") == 1
    r.server = vk.format(100)
    clock.t += 5
    assert hardware.refresh(host, "/b", r, clock).gpus[0].free == 900 * MiB   # < 15 s: untouched
    assert r.count("llama-server") == 1
    clock.t += 11
    assert hardware.refresh(host, "/b", r, clock).gpus[0].free == 100 * MiB   # >= 15 s since listing
    assert r.count("llama-server") == 2
    assert r.count("nvidia-smi") == 0


def test_refresh_mixed_backends_does_not_use_smi() -> None:
    host = Host(gpus=[Device.of("CUDA0", "A", MiB, MiB), Device.of("Vulkan0", "B", MiB, MiB)])
    r = FakeRunner(server="  CUDA0: A (1 MiB, 0 MiB free)\n  Vulkan0: B (1 MiB, 1 MiB free)\n")
    out = hardware.refresh(host, "/b", r, Clock())
    assert r.count("nvidia-smi") == 0 and out.gpus[0].free == 0


def test_refresh_keeps_devices_when_relisting_fails() -> None:
    host = Host(gpus=[Device.of("Vulkan0", "B", MiB, MiB)])
    out = hardware.refresh(host, "/b", FakeRunner(server=FileNotFoundError(),
                                                  smi=FileNotFoundError()), Clock())
    assert out.gpus == host.gpus


def test_refresh_does_not_mutate_input() -> None:
    host = Host(gpus=[Device.of("CUDA0", "G", 100 * MiB, 100 * MiB)])
    out = hardware.refresh(host, "/b", FakeRunner(smi="G, 100, 1\n"), Clock())
    assert host.gpus[0].free == 100 * MiB and out is not host


def test_empty_detection_retried_only_after_interval() -> None:
    clock = Clock()
    r = FakeRunner(server=FileNotFoundError(), smi=FileNotFoundError())
    assert hardware.detect("/b", r, clock).gpus == []
    clock.t += 3
    hardware.detect("/b", r, clock)
    assert r.count("llama-server") == 1
    clock.t += 20
    hardware.detect("/b", r, clock)
    assert r.count("llama-server") == 2


def test_detect_cache_is_per_bin_dir() -> None:
    r = FakeRunner(server=REAL, smi="x, 16379, 1\n")
    hardware.detect("/a", r, Clock())
    hardware.detect("/b", r, Clock())
    assert r.count("llama-server") == 2


def test_unified_for_metal() -> None:
    r = FakeRunner(server="  Metal: Apple M2 (16384 MiB, 16000 MiB free)\n")
    host = hardware.detect("/b", r, Clock())
    assert host.unified and host.gpus[0].backend == "metal"


def test_parse_meminfo() -> None:
    text = "MemTotal:       32768 kB\nMemFree:  100 kB\nMemAvailable:   16384 kB\n"
    assert hardware.parse_meminfo(text) == (32768 * 1024, 16384 * 1024)
    assert hardware.parse_meminfo("") == (0, 0)


def test_physical_cores() -> None:
    text = ("processor: 0\nphysical id: 0\ncore id: 0\n\nprocessor: 1\nphysical id: 0\n"
            "core id: 0\n\nprocessor: 2\nphysical id: 0\ncore id: 1\n")
    assert hardware.physical_cores(text) == 2
    assert hardware.physical_cores("") >= 1


def test_macos_readings() -> None:
    vm = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
          "Pages free:    100.\nPages inactive:  50.\nPages speculative: 10.\n")
    assert hardware.parse_vm_stat(vm) == 160 * 16384

    def run(argv: list[str], **kw: object) -> SimpleNamespace:
        out = {"hw.memsize": "17179869184", "hw.physicalcpu": "8",
               "machdep.cpu.brand_string": "Apple M2"}.get(argv[-1], "")
        return proc(vm if argv[0] == "vm_stat" else out)

    assert hardware.read_memory(run, "darwin") == (17179869184, 160 * 16384)
    name, threads, cores = hardware.read_cpu(run, "darwin")
    assert (name, cores) == ("Apple M2", 8)
    assert hardware.read_memory(run, "win32") == (0, 0)
    assert hardware.read_cpu(run, "win32")[0] == ""


def test_wsl_advice_suggests_leaving_room_for_windows() -> None:
    """WSL sees 23.5 GiB of a 32 GiB machine: suggest 28 GiB (Windows keeps 4)."""
    host = Host(ram_total=int(23.5 * hardware.GiB), cpu_threads=8)
    advice = hardware.wsl_advice(host, 32 * hardware.GiB)
    assert advice is not None and advice.suggested_gib == 28 and advice.worth
    assert "memory=28GB" in advice.snippet
    assert hardware.wsl_advice(host, None) is None
    assert hardware.wsl_advice(host, 0) is None
