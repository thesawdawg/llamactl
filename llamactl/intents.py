"""Intents: what a model is being served for, as declarative data.

An intent never contains planning logic; ``planner.plan`` reads these fields. Context sizes
are per slot, in tokens. A value of 0 for ``ctx_min`` / ``ctx_target`` / ``ctx_max`` means
"the model's trained context" and is resolved by the planner against ``ModelInfo.ctx_train``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .gguf import ModelInfo

TRAINED = 0   # ctx value meaning "the model's trained context"

KV_NEVER = "never"
KV_IF_NEEDED = "if_needed"
KV_PREFER = "prefer"
SPILL_AVOID = "avoid"
SPILL_ALLOW = "allow"


def _generative(info: ModelInfo) -> bool:
    """True for models that generate text (no pooling type declared)."""
    return not info.is_embedding


def _embedding(info: ModelInfo) -> bool:
    """True for embedding/reranker models (the GGUF declares a pooling type)."""
    return info.is_embedding


@dataclass(frozen=True)
class Intent:
    """A goal the planner turns into settings.

    Attributes:
        key: Short identifier, also the preset name ("chat", "agent", ...).
        label: Human readable name.
        description: One line on what the intent is for.
        ctx_min: Smallest acceptable context per slot (0 = trained).
        ctx_target: Context per slot the planner aims for (0 = trained).
        ctx_max: Largest useful context per slot (0 = trained).
        slots: Parallel slots; the total ``-c`` is ``ctx_per_slot * slots``.
        kv_quant: "never", "if_needed" or "prefer" q8_0 K/V cache.
        spill: "avoid" or "allow" moving weights off the GPU to reach ``ctx_target``.
        extra: Fixed Profile settings the intent applies (keys are schema keys).
        applies: Predicate telling whether the intent suits a model.
    """

    key: str
    label: str
    description: str
    ctx_min: int
    ctx_target: int
    ctx_max: int
    slots: int
    kv_quant: str
    spill: str
    extra: dict[str, Any] = field(default_factory=dict, hash=False)
    applies: Callable[[ModelInfo], bool] = _generative


INTENTS: tuple[Intent, ...] = (
    Intent("chat", "Chat", "interactive chat frontends",
           8192, 16384, 32768, 1, KV_IF_NEEDED, SPILL_AVOID),
    Intent("agent", "Agent", "coding agents and tool harnesses",
           32768, 65536, 131072, 1, KV_PREFER, SPILL_ALLOW, {"cache_reuse": 256}),
    Intent("context", "Long context", "largest usable window",
           32768, TRAINED, TRAINED, 1, KV_PREFER, SPILL_ALLOW),
    Intent("speed", "Speed", "highest tokens per second",
           4096, 8192, 16384, 1, KV_NEVER, SPILL_AVOID),
    Intent("api", "API", "several concurrent clients",
           4096, 8192, 16384, 4, KV_IF_NEEDED, SPILL_AVOID, {"metrics": True}),
    Intent("embed", "Embeddings", "embedding models",
           TRAINED, TRAINED, TRAINED, 4, KV_NEVER, SPILL_AVOID, {"embedding": True}, _embedding),
)

by_key: dict[str, Intent] = {i.key: i for i in INTENTS}


def applicable(info: ModelInfo) -> tuple[Intent, ...]:
    """Intents that suit a model: ``embed`` for embedding models, the rest otherwise.

    Args:
        info: Model metadata.

    Returns:
        The applicable intents in table order (never empty).
    """
    return tuple(i for i in INTENTS if i.applies(info))


def resolve(key: str, info: ModelInfo) -> Intent:
    """The intent named `key` when it applies to the model, else the first applicable one.

    Args:
        key: Preferred intent key (e.g. the configured default intent).
        info: Model metadata.

    Returns:
        An intent that applies to the model.
    """
    options = applicable(info)
    return next((i for i in options if i.key == key), options[0])
