"""Persistent per-preset goals, independent of concrete server launch settings."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .intents import Intent


@dataclass(frozen=True)
class TuningGoals:
    """Speed constraints and context ceiling for a preset.

    Attributes:
        min_effective_tps: Output tokens divided by whole request time; default 20.
        min_prompt_tps: Optional input-processing floor; zero disables it.
        min_gen_tps: Optional output-generation floor; zero disables it.
        max_context: Maximum tokens per slot; zero inherits the purpose ceiling.
        benchmark: Quick, normal or full workload suite.
    """

    min_effective_tps: float = 20.0
    min_prompt_tps: float = 0.0
    min_gen_tps: float = 0.0
    max_context: int = 0
    benchmark: str = "normal"

    def __post_init__(self) -> None:
        """Reject invalid goals before a search or configuration save.

        Raises:
            ValueError: If a speed is negative/nonfinite or the context/mode is invalid.
        """
        for value in (self.min_effective_tps, self.min_prompt_tps, self.min_gen_tps):
            if not math.isfinite(value) or value < 0:
                raise ValueError("speed floors must be finite nonnegative numbers")
        if self.max_context < 0 or int(self.max_context) != self.max_context:
            raise ValueError("maximum context must be a nonnegative integer")
        if self.benchmark not in ("quick", "normal", "full"):
            raise ValueError("benchmark must be quick, normal or full")

    @classmethod
    def from_dict(cls, data: Any) -> TuningGoals:
        """Read goals, falling back to defaults for malformed legacy configuration.

        Args:
            data: Optional JSON object.

        Returns:
            Valid goals with unknown keys ignored.
        """
        if not isinstance(data, dict):
            return cls()
        try:
            raw_context = data.get("max_context", 0)
            context = int(raw_context)
            if isinstance(raw_context, bool) or float(raw_context) != context:
                raise ValueError("maximum context must be a whole number")
            return cls(float(data.get("min_effective_tps", 20)),
                       float(data.get("min_prompt_tps", 0)),
                       float(data.get("min_gen_tps", 0)),
                       context, str(data.get("benchmark", "normal")))
        except (TypeError, ValueError, OverflowError):
            return cls()

    def resolve_intent(self, intent: Intent) -> Intent:
        """Apply the preset context ceiling consistently to planning and tuning.

        Args:
            intent: Inherited purpose definition.

        Returns:
            A purpose with this preset's context goal, or the inherited purpose.
        """
        if not self.max_context:
            return intent
        return replace(intent, ctx_target=self.max_context, ctx_max=self.max_context,
                       ctx_min=min(intent.ctx_min or self.max_context, self.max_context))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible goals.

        Returns:
            All goal values.
        """
        return asdict(self)
