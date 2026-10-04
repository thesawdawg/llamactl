"""Shared number/formatting helpers."""
from __future__ import annotations


def human(n: float) -> str:
    """Bytes as decimal GB (as shown on Hugging Face)."""
    return f"{n / 1e9:.2f} GB"


def params_str(n: int | None) -> str:
    """Parameter count like '8.2B'."""
    return "" if not n else f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e6:.0f}M"


# Rich style for Fits/verdict cells: GPU green, partial/CPU yellow, NO red.
FITS_STYLE = {"GPU": "green", "GPU+CPU": "yellow", "CPU": "yellow", "NO": "red"}
