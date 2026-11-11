"""Discover's fit hints for a repo's GGUF files (pure: no Textual, no network).

Only the file size is known before a download, so the verdict is a *weights only* estimate:
`planner.placeholder_info` carries no KV cache, which is why every label says so. The largest
file whose weights fit fully in GPU memory is marked as the best fit.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import Profile
from .hardware import Host
from .planner import VERDICT_GPU, VERDICT_NO, estimate_budget, placeholder_info, verdict

BEST_MARK = "best fit"
WEIGHTS_ONLY = "weights only"


@dataclass(frozen=True)
class FileFit:
    """One downloadable entry of a repo with its weights-only verdict.

    A sharded model is one entry: `path` is its first shard and `size` the sum of all shards.

    Attributes:
        path: Path in the repo (first shard when sharded).
        quant: Quantisation tag guessed from the name ("" when none).
        size: Bytes (all shards).
        parts: Number of files that make up the entry.
        verdict: Planner verdict for the weights alone ("GPU", "GPU+CPU", "CPU", "NO").
        best: The largest file that fully fits the GPU.
    """

    path: str
    quant: str
    size: int
    parts: int
    verdict: str
    best: bool = False

    @property
    def label(self) -> str:
        """The verdict as shown: ``GPU (weights only)``; ``NO`` stays bare (it needs no caveat)."""
        return self.verdict if self.verdict == VERDICT_NO else f"{self.verdict} ({WEIGHTS_ONLY})"


def host_ready(host: Host) -> bool:
    """True once the host has been detected (a GPU or some RAM is known)."""
    return bool(host.gpus or host.ram_total)


def weights_verdict(size: int, host: Host, headroom_pct: float = 10) -> str:
    """Verdict for a file of `size` bytes counting its weights only.

    Args:
        size: File size in bytes.
        host: Host snapshot.
        headroom_pct: Percentage of each device kept free.

    Returns:
        "GPU", "GPU+CPU", "CPU" or "NO".
    """
    budget = estimate_budget(placeholder_info(size), Profile(), host)
    return verdict(budget, host, headroom_pct)


def rank_files(files: list[dict], host: Host, headroom_pct: float = 10) -> list[FileFit]:
    """Judge every file and mark the best fit; largest file first.

    The best fit is the largest entry whose verdict is "GPU". Without a GPU, or before the host
    is known, nothing is marked.

    Args:
        files: Entries as `hf.list_gguf` returns them (``path``, ``size``, ``parts``, ``quant``).
        host: Host snapshot.
        headroom_pct: Percentage of each device kept free.

    Returns:
        One `FileFit` per entry, sorted by size descending (ties by path).
    """
    ranked = sorted(files, key=lambda f: (-int(f.get("size") or 0), str(f["path"])))
    fits = [(f, weights_verdict(int(f.get("size") or 0), host, headroom_pct)) for f in ranked]
    best = next((f["path"] for f, v in fits
                 if v == VERDICT_GPU and host.gpus and int(f.get("size") or 0) > 0), None)
    return [FileFit(f["path"], f.get("quant", ""), int(f.get("size") or 0), int(f.get("parts") or 1),
                    v, f["path"] == best) for f, v in fits]


def legend(host: Host, intent_label: str) -> str:
    """One line explaining the markers of the file list.

    Args:
        host: Host snapshot.
        intent_label: Label of the default intent.

    Returns:
        The legend.
    """
    if not host_ready(host):
        return "Detecting the hardware to judge which files fit..."
    if not host.gpus:
        return ("No GPU found, so no file is marked: sizes are judged against RAM, weights only "
                "(the context cache comes on top).")
    return (f"★ {BEST_MARK}: the largest file whose weights fit fully in GPU memory for "
            f"{intent_label}; context (KV cache) is not counted yet, so leave some room.")
