"""The result of one measured launch (SPEC 6.1).

Kept in its own module so both ``config`` (presets remember their measurement) and
``calibration`` (trials store one) can use it without importing each other.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
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
class WorkloadMeasurement:
    """Measured throughput of one turn at a given context occupancy.

    Attributes:
        label: Requested workload description.
        prompt_tokens: Actually processed input tokens (the turn delta; the cached
            prefix is not counted).
        generated_tokens: Actual generated output tokens.
        request_s: Measured turn duration (server timings when reported, else the
            request's wall time), excluding server startup.
        prompt_tps: Server input-processing speed over the processed delta.
        gen_tps: Server output-generation speed.
        cached_tokens: Prefix tokens served from the KV cache instead of processed.
        samples: Measured requests folded into this result; >1 means a noisy
            near-floor result was retested and this is the median.
    """

    label: str = ""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    request_s: float = 0.0
    prompt_tps: float = 0.0
    gen_tps: float = 0.0
    cached_tokens: int = 0
    samples: int = 1

    @property
    def effective_tps(self) -> float:
        """Return delivered output tokens per second including the turn input wait."""
        return self.generated_tokens / self.request_s if self.request_s > 0 else 0.0

    @property
    def occupancy_tokens(self) -> int:
        """Context tokens the turn occupied: processed delta plus cached prefix."""
        return self.prompt_tokens + self.cached_tokens

    @classmethod
    def from_dict(cls, data: Any) -> WorkloadMeasurement:
        """Read a stored workload.

        Args:
            data: JSON object.

        Returns:
            Workload with unknown fields ignored and missing new fields defaulted.
        """
        if not isinstance(data, dict):
            return cls()
        return cls(str(data.get("label", "")),
                   _number(data.get("prompt_tokens", 0), int, 0),
                   _number(data.get("generated_tokens", 0), int, 0),
                   _number(data.get("request_s", 0), float, 0.0),
                   _number(data.get("prompt_tps", 0), float, 0.0),
                   _number(data.get("gen_tps", 0), float, 0.0),
                   _number(data.get("cached_tokens", 0), int, 0),
                   _number(data.get("samples", 1), int, 1))

    def to_dict(self) -> dict[str, Any]:
        """Return the workload's JSON-compatible measured values."""
        return asdict(self)


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
        effective_tps: Lowest workload effective output TPS; zero if unavailable.
        request_s: Sum of measured turn durations, excluding model startup.
        prompt_tokens: Total actually processed input tokens across workloads.
        generated_tokens: Total actual generated tokens across workloads.
        cached_tokens: Total prefix tokens served from the KV cache across workloads.
        workloads: Individual benchmark results; empty for legacy measurements.
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
    effective_tps: float = 0.0
    request_s: float = 0.0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    cached_tokens: int = 0
    workloads: tuple[WorkloadMeasurement, ...] = ()

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
            if f.name == "workloads":
                out[f.name] = (tuple(WorkloadMeasurement.from_dict(w) for w in v)
                               if isinstance(v, (list, tuple)) else ())
            elif isinstance(default, bool):
                out[f.name] = v if isinstance(v, bool) else default
            elif isinstance(default, str):
                out[f.name] = v if isinstance(v, str) else default
            else:
                out[f.name] = _number(v, type(default), default)
        return cls(**out)

    def to_dict(self) -> dict:
        """Serialise to a JSON-able dict."""
        return {f.name: ([w.to_dict() for w in self.workloads] if f.name == "workloads"
                         else getattr(self, f.name)) for f in fields(self)}

    @property
    def resource_failure(self) -> bool:
        """True when the launch failed for lack of memory or time, so a smaller one may work."""
        return not self.ok and self.error.startswith((ERR_OOM, ERR_TIMEOUT))

    @property
    def cancelled(self) -> bool:
        """True when the launch was stopped on request: it says nothing about the settings."""
        return not self.ok and self.error.startswith(ERR_CANCELLED)
