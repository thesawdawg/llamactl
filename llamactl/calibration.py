"""Measured tuning results and the estimate correction factor derived from them.

``tuning.json`` holds one record per measured trial. This module loads the file, turns the
successful all-GPU trials into a per-backend factor the planner applies to its estimated
totals, records new trials (atomic save) and tells whether a tuned preset has gone stale.
"""
from __future__ import annotations

import json
import os
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import CONFIG_DIR, Preset
from .measure import Measurement

TUNING_FILE = CONFIG_DIR / "tuning.json"
FACTOR_MIN = 0.8
FACTOR_MAX = 1.5
GPU_VERDICT = "GPU"    # plan verdict of a trial that kept the whole model on the GPU
TUNING_VERSION = 1
TUNED = "tuned"        # Preset.source of a measured preset


@dataclass(frozen=True)
class Trial:
    """One measured launch, as stored in tuning.json.

    Attributes:
        model: Model file path.
        mtime: Model file modification time when measured.
        intent: Intent key the profile was planned for.
        profile: The profile that was launched, as a plain dict.
        estimated_total: Planner's uncalibrated total estimate in bytes (``Budget.raw_total()``,
            never the calibrated total, so the factor cannot compound).
        verdict: Plan verdict at launch ("GPU", "GPU+CPU", "CPU", "NO").
        backend: Backend of the first device ("cuda", "vulkan", "metal", ...).
        measurement: What the launch showed.
    """

    model: str
    mtime: float
    intent: str
    profile: dict[str, Any]
    estimated_total: int
    verdict: str
    backend: str
    measurement: Measurement = field(default_factory=Measurement)

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
            if not isinstance(profile, dict):
                raise ValueError("profile must be an object")
            return cls(model=str(data["model"]), mtime=float(data.get("mtime", 0.0)),
                       intent=str(data.get("intent", "")), profile=dict(profile),
                       estimated_total=int(data.get("estimated_total", 0)),
                       verdict=str(data.get("verdict", "")),
                       backend=str(data.get("backend", "")),
                       measurement=Measurement.from_dict(measurement))
        except (KeyError, TypeError) as e:
            raise ValueError(f"bad trial record: {e}") from e

    def to_dict(self) -> dict:
        """Serialise to a JSON-able dict."""
        return {"model": self.model, "mtime": self.mtime, "intent": self.intent,
                "profile": dict(self.profile), "estimated_total": self.estimated_total,
                "verdict": self.verdict, "backend": self.backend,
                "measurement": self.measurement.to_dict()}

    def ratio(self) -> float | None:
        """Measured VRAM over estimated total, for a successful all-GPU trial.

        Returns:
            ``vram_used / estimated_total``, or None when the trial failed, was not an
            all-GPU plan, or carries no usable numbers.
        """
        used = self.measurement.vram_used
        if self.verdict != GPU_VERDICT or not self.measurement.ok:
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

    # -- write path ------------------------------------------------------------------

    def record(self, trial: Trial) -> None:
        """Add a trial (in memory; call `save` to persist it).

        Args:
            trial: The measured launch to remember.
        """
        self.trials = (*self.trials, trial)

    def save(self, path: Path | None = None) -> None:
        """Write every trial to tuning.json atomically.

        The data goes to a temporary file in the same folder, which then replaces the real
        one, so a crash mid-write leaves the previous file intact.

        Args:
            path: File to write; defaults to ``TUNING_FILE``.

        Raises:
            OSError: When the folder or file cannot be written.
        """
        path = path or TUNING_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps({"version": TUNING_VERSION, "trials": [t.to_dict() for t in self.trials]},
                          indent=2)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # -- queries ---------------------------------------------------------------------

    def trials_for(self, model: str | Path, intent: str | None = None) -> list[Trial]:
        """Trials of one model, oldest first.

        Args:
            model: Model file path.
            intent: Only trials planned for this intent; None for all of them.

        Returns:
            The matching trials.
        """
        return [t for t in self.trials
                if t.model == str(model) and (intent is None or t.intent == intent)]

    def is_stale(self, preset: Preset, model_path: str | Path, build: str) -> bool:
        """True when a tuned preset no longer matches the model file or llama.cpp build.

        A preset is stale when it is tuned and either the llama.cpp build differs from the
        one its measurement recorded, or the model file's modification time differs from the
        one stored with the trial that produced the measurement. What cannot be compared
        (an unknown build, no matching trial, a missing model file) does not count.

        Args:
            preset: The preset to check.
            model_path: Model file the preset belongs to.
            build: The llama-server build string now ("" when unknown).

        Returns:
            Whether the preset should be re-tuned.
        """
        m = preset.measured
        if preset.source != TUNED or m is None:
            return False
        if build and m.build and build != m.build:
            return True
        trial = next((t for t in reversed(self.trials_for(model_path, preset.intent))
                      if t.measurement.at == m.at), None)
        if trial is None or not trial.mtime:
            return False
        try:
            return Path(model_path).stat().st_mtime != trial.mtime
        except OSError:
            return False
