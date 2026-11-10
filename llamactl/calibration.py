"""Measured tuning results and the estimate correction factor derived from them.

``tuning.json`` holds one record per measured trial. This module is the read path: it loads
the file and turns the successful all-GPU trials into a per-backend factor the planner applies
to its estimated totals. Writing records is added with the tuner.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import CONFIG_DIR

TUNING_FILE = CONFIG_DIR / "tuning.json"
FACTOR_MIN = 0.8
FACTOR_MAX = 1.5
GPU_VERDICT = "GPU"    # plan verdict of a trial that kept the whole model on the GPU


@dataclass(frozen=True)
class Trial:
    """One measured launch, as stored in tuning.json.

    Attributes:
        model: Model file path.
        mtime: Model file modification time when measured.
        intent: Intent key the profile was planned for.
        profile: The profile that was launched, as a plain dict.
        estimated_total: Planner's uncalibrated (factor 1.0) total estimate in bytes.
        verdict: Plan verdict at launch ("GPU", "GPU+CPU", "CPU", "NO").
        backend: Backend of the first device ("cuda", "vulkan", "metal", ...).
        measurement: The measurement as a plain dict (``ok``, ``vram_used``, ``gen_tps``, ...).
    """

    model: str
    mtime: float
    intent: str
    profile: dict[str, Any]
    estimated_total: int
    verdict: str
    backend: str
    measurement: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "Trial":
        """Build a trial from its JSON form.

        Args:
            data: One record of tuning.json.

        Returns:
            The trial.

        Raises:
            ValueError: When the record is not a dict or a field has the wrong type.
        """
        if not isinstance(data, dict):
            raise ValueError("trial record must be an object")
        try:
            profile, measurement = data.get("profile", {}), data.get("measurement", {})
            if not isinstance(profile, dict) or not isinstance(measurement, dict):
                raise ValueError("profile and measurement must be objects")
            return cls(model=str(data["model"]), mtime=float(data.get("mtime", 0.0)),
                       intent=str(data.get("intent", "")), profile=dict(profile),
                       estimated_total=int(data.get("estimated_total", 0)),
                       verdict=str(data.get("verdict", "")),
                       backend=str(data.get("backend", "")), measurement=dict(measurement))
        except (KeyError, TypeError) as e:
            raise ValueError(f"bad trial record: {e}") from e

    def to_dict(self) -> dict:
        """Serialise to a JSON-able dict."""
        return {"model": self.model, "mtime": self.mtime, "intent": self.intent,
                "profile": dict(self.profile), "estimated_total": self.estimated_total,
                "verdict": self.verdict, "backend": self.backend,
                "measurement": dict(self.measurement)}

    def ratio(self) -> float | None:
        """Measured VRAM over estimated total, for a successful all-GPU trial.

        Returns:
            ``vram_used / estimated_total``, or None when the trial failed, was not an
            all-GPU plan, or carries no usable numbers.
        """
        if self.verdict != GPU_VERDICT or self.measurement.get("ok") is not True:
            return None
        try:
            used = float(self.measurement.get("vram_used", 0))
        except (TypeError, ValueError):
            return None
        return used / self.estimated_total if used > 0 and self.estimated_total > 0 else None


class Calibration:
    """Per-backend correction of the planner's memory estimates.

    Args:
        trials: Measured trials to derive factors from.
    """

    def __init__(self, trials: Iterable[Trial] = ()) -> None:
        self.trials: tuple[Trial, ...] = tuple(trials)

    @classmethod
    def load(cls, path: Path | None = None) -> "Calibration":
        """Read tuning.json; a missing or corrupt file means no data.

        Individual records that cannot be parsed are skipped.

        Args:
            path: File to read; defaults to ``TUNING_FILE``.

        Returns:
            The calibration (empty when nothing could be read).
        """
        path = path or TUNING_FILE
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            return cls()
        records = raw.get("trials") if isinstance(raw, dict) else None
        trials: list[Trial] = []
        for rec in records if isinstance(records, list) else []:
            try:
                trials.append(Trial.from_dict(rec))
            except ValueError:
                continue
        return cls(trials)

    def factor(self, backend: str) -> float:
        """Correction factor for estimated totals on a backend.

        The median of measured VRAM over estimated total across successful all-GPU trials of
        that backend, clamped to 0.8..1.5.

        Args:
            backend: Backend name as in ``Device.backend``.

        Returns:
            The factor; 1.0 when no trial applies.
        """
        ratios = [r for t in self.trials if t.backend == backend
                  if (r := t.ratio()) is not None]
        if not ratios:
            return 1.0
        return min(FACTOR_MAX, max(FACTOR_MIN, statistics.median(ratios)))
