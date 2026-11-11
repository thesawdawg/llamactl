"""The result of one measured launch (SPEC 6.1).

Kept in its own module so both ``config`` (presets remember their measurement) and
``calibration`` (trials store one) can use it without importing each other.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

# Failure kinds: `Measurement.error` starts with one of these, then ": detail".
ERR_OOM = "out of memory"
ERR_TIMEOUT = "timeout"
ERR_ARGS = "unknown argument"
ERR_LOAD = "load failed"
ERR_BENCH = "benchmark failed"
ERR_CANCELLED = "cancelled"     # the caller asked to stop; not a failure of the settings


def _number(value: Any, kind: type, default: Any) -> Any:
    """Coerce a JSON value to an int or float, falling back on `default`."""
    if isinstance(value, bool):
        return default
    try:
        return kind(value)
    except (TypeError, ValueError, OverflowError):
        return default


@dataclass(frozen=True)
class Measurement:
    """What one launch of llama-server showed (all memory in bytes).

    Attributes:
        ok: The model loaded and answered the benchmark request.
        error: Why it did not, starting with one of the ``ERR_*`` kinds; "" when `ok`.
        load_s: Seconds from launch until ``/health`` reported ready.
        vram_used: Device memory taken by the loaded model, as a delta against the pre-launch
            snapshot (summed over devices).
        ram_used: System RAM taken, as a delta against the pre-launch snapshot.
        prompt_tps: Prompt processing speed in tokens/s (``timings.prompt_per_second``).
        gen_tps: Generation speed in tokens/s (``timings.predicted_per_second``); 0 for
            embedding models.
        at: ISO timestamp of the measurement.
        build: llama-server version string the measurement was taken with.
    """

    ok: bool = False
    error: str = ""
    load_s: float = 0.0
    vram_used: int = 0
    ram_used: int = 0
    prompt_tps: float = 0.0
    gen_tps: float = 0.0
    at: str = ""
    build: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "Measurement":
        """Build a measurement from its JSON form.

        Unknown keys are ignored and a field with a bad value takes its default, so records
        written by older or newer versions still load.

        Args:
            data: A dict as produced by `to_dict`.

        Returns:
            The measurement.

        Raises:
            ValueError: When `data` is not a dict.
        """
        if not isinstance(data, dict):
            raise ValueError("measurement must be an object")
        base = cls()
        out: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            v, default = data[f.name], getattr(base, f.name)
            if isinstance(default, bool):
                out[f.name] = v if isinstance(v, bool) else default
            elif isinstance(default, str):
                out[f.name] = v if isinstance(v, str) else default
            else:
                out[f.name] = _number(v, type(default), default)
        return cls(**out)

    def to_dict(self) -> dict:
        """Serialise to a JSON-able dict."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @property
    def resource_failure(self) -> bool:
        """True when the launch failed for lack of memory or time, so a smaller one may work."""
        return not self.ok and self.error.startswith((ERR_OOM, ERR_TIMEOUT))

    @property
    def cancelled(self) -> bool:
        """True when the launch was stopped on request: it says nothing about the settings."""
        return not self.ok and self.error.startswith(ERR_CANCELLED)
