"""Plain-data summaries of a model for the Models view and the intent picker.

Pure functions over the planner's output and the config, so the table cells and the detail rows
are unit-tested without Textual.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import planner
from .calibration import TUNED, Calibration
from .config import ModelEntry, Preset
from .gguf import ModelInfo
from .hardware import Host
from .intents import INTENTS, by_key
from .planner import (VERDICT_CPU, VERDICT_GPU, VERDICT_NO, VERDICT_SPILL, Plan,
                      verdict_label)

NO_PRESET = "New"            # shown for a model with no preset at all
MAX_BEST = 4                 # intents named in the "Best for" cell before "+N"
_RANK = {VERDICT_GPU: 0, VERDICT_SPILL: 1, VERDICT_CPU: 2, VERDICT_NO: 3}


@dataclass(frozen=True)
class BestFor:
    """The intents a model suits best.

    Attributes:
        verdict: The best verdict any intent got.
        keys: Intent keys that got it, in table order.
    """

    verdict: str
    keys: tuple[str, ...]

    def text(self) -> str:
        """Compact text: the first few keys then ``+N``; ``-`` when nothing fits."""
        if self.verdict == VERDICT_NO or not self.keys:
            return "-"
        shown = " ".join(self.keys[:MAX_BEST])
        more = len(self.keys) - MAX_BEST
        return shown + (f" +{more}" if more > 0 else "")


def best_for(plans: dict[str, Plan]) -> BestFor:
    """Summarise per-intent plans as the intents with the best verdict.

    Args:
        plans: Plans by intent key (``planner.plan_all``).

    Returns:
        The best verdict and the intents that reach it (verdict "NO" with no keys when empty).
    """
    if not plans:
        return BestFor(VERDICT_NO, ())
    best = min((p.verdict for p in plans.values()), key=lambda v: _RANK.get(v, 9))
    return BestFor(best, tuple(k for k, p in plans.items() if p.verdict == best))


@dataclass(frozen=True)
class PresetState:
    """The active preset of a model as the table words it.

    Attributes:
        name: Active preset name, "" when the model has none.
        source: "tuned", "estimated" or "edited"; "" when none.
        stale: A tuned preset whose model file or llama.cpp build changed.
        count: How many presets the model has.
        measured: Generation tok/s (prompt tok/s for embedding models) of a tuned active
            preset, None otherwise.
    """

    name: str = ""
    source: str = ""
    stale: bool = False
    count: int = 0
    measured: float | None = None

    @property
    def new(self) -> bool:
        """True for a model without presets."""
        return self.count == 0

    def text(self) -> str:
        """Plain text of the cell: ``New`` or ``agent tuned stale (+1)``."""
        if self.new:
            return NO_PRESET
        extra = f" (+{self.count - 1})" if self.count > 1 else ""
        return f"{self.name} {self.source}{' stale' if self.stale else ''}{extra}"


def speed_of(preset: Preset | None) -> float | None:
    """Measured speed of a tuned preset: generation tok/s, else prompt tok/s; None otherwise."""
    m = preset.measured if preset else None
    if preset is None or preset.source != TUNED or m is None or not m.ok:
        return None
    return m.gen_tps or m.prompt_tps or None


def preset_state(entry: ModelEntry | None, path: Path, calibration: Calibration,
                 build: str) -> PresetState:
    """The active preset's state for a model.

    Args:
        entry: The model's config entry, or None.
        path: Model file.
        calibration: Used to tell whether a tuned preset is stale.
        build: Current llama-server build string ("" when unknown).

    Returns:
        The state; ``new`` when the model has no presets.
    """
    if entry is None or not entry.presets:
        return PresetState()
    name = entry.active if entry.active in entry.presets else sorted(entry.presets)[0]
    p = entry.presets[name]
    return PresetState(name, p.source, calibration.is_stale(p, path, build), len(entry.presets),
                       speed_of(p))


@dataclass(frozen=True)
class IntentRow:
    """One applicable intent of a model, with its plan and preset.

    Attributes:
        key: Intent key.
        label: Intent label.
        description: What the intent is for.
        verdict: Plan verdict.
        ctx_per_slot: Planned context per slot.
        slots: Parallel slots.
        kv: KV cache type of the plan.
        needs: Bytes the plan needs.
        usable: Bytes of usable GPU memory (total minus headroom, over all devices).
        tps: Measured tok/s of a tuned preset for this intent, None otherwise.
        reason: The plan's top reason (context first), "" when it changed nothing.
        warnings: The plan's warnings.
        source: Source of the model's preset for this intent ("tuned" / "estimated" /
            "edited"), "" when there is none.
        active: That preset is the model's active one.
        stale: That preset is tuned and stale.
        pp_tg: For a MoE model whose preset has a measurement, "prompt / generation" tok/s
            to show instead of `tps`; "" otherwise.
    """

    key: str
    label: str
    description: str
    verdict: str
    ctx_per_slot: int
    slots: int
    kv: str
    needs: int
    usable: int
    tps: float | None
    reason: str
    warnings: tuple[str, ...]
    source: str
    active: bool
    stale: bool
    pp_tg: str = ""

    @property
    def has_preset(self) -> bool:
        """True when the model has a preset for this intent."""
        return bool(self.source)


def top_reason(plan: Plan) -> str:
    """The most telling line of a plan: why the context is what it is, else the first reason."""
    return plan.reasons.get("ctx_size") or next(iter(plan.reasons.values()), "")


def intent_rows(plans: dict[str, Plan], info: ModelInfo, entry: ModelEntry | None,
                path: Path, host: Host, headroom_pct: float, calibration: Calibration,
                build: str) -> list[IntentRow]:
    """One row per planned intent, in intent table order.

    Args:
        plans: Plans by intent key.
        info: Model metadata (for the MoE verdict label).
        entry: The model's config entry, or None.
        path: Model file.
        host: Host snapshot (for the usable memory).
        headroom_pct: Headroom percentage.
        calibration: For the stale flag.
        build: Current llama-server build string.

    Returns:
        The rows.
    """
    usable = planner.gpu_limit(host, headroom_pct)
    rows = []
    for intent in INTENTS:
        plan = plans.get(intent.key)
        if plan is None:
            continue
        preset = entry.presets.get(intent.key) if entry else None
        m = preset.measured if preset else None
        pp_tg = (f"{m.prompt_tps:.0f} / {m.gen_tps:.1f}"
                 if info.expert_count > 0 and m is not None and m.ok else "")
        rows.append(IntentRow(
            intent.key, intent.label, intent.description,
            verdict_label(plan.verdict, plan.profile, info), plan.ctx_per_slot,
            intent.slots, plan.profile.cache_type_k, plan.budget.total(), usable,
            speed_of(preset), top_reason(plan), plan.warnings, preset.source if preset else "",
            bool(entry and preset and entry.active == intent.key),
            bool(preset and calibration.is_stale(preset, path, build)), pp_tg))
    return rows


def default_intent_key(entry: ModelEntry | None, plans: dict[str, Plan], default: str) -> str:
    """The intent a model's actions apply to when the user has not picked one.

    The active preset's intent when it is one of the planned intents, else the configured
    default intent when planned, else the first planned intent.

    Args:
        entry: The model's config entry, or None.
        plans: Plans by intent key.
        default: Configured default intent key.

    Returns:
        An intent key ("" when there are no plans).
    """
    if entry is not None and entry.active in plans:
        return entry.active
    if default in plans and default in by_key:
        return default
    return next(iter(plans), "")
