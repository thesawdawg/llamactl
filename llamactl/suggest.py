"""The "next steps" the Home view offers, computed from the current state.

Pure: the same inputs always give the same list, so every rule is unit-tested without Textual.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .calibration import TUNED, Calibration
from .config import ESTIMATED, Config
from .gateway import GatewayStatus
from .gguf import ModelInfo
from .library import aliases_for

# What choosing a suggestion does (the app maps these to a view switch or a flow).
ACT_SETTINGS, ACT_DISCOVER, ACT_MODELS = "settings", "discover", "models"
ACT_TUNE, ACT_START_GATEWAY = "tune", "start_gateway"

MAX_PER_RULE = 3      # at most this many suggestions per model rule


@dataclass(frozen=True)
class Suggestion:
    """One next step.

    Attributes:
        key: Stable identifier (``binary``, ``no-models``, ``no-presets``,
            ``tune-moe:<path>:<preset>``, ``tune:<path>:<preset>``, ``retune:<path>:<preset>``,
            ``start-gateway``).
        text: One line for the user.
        action: What choosing it does (``ACT_*``).
        model: Model path the action is about, "" when none.
        preset: Preset name (an intent key or "custom") the action is about, "" when none.
    """

    key: str
    text: str
    action: str
    model: str = ""
    preset: str = ""


def binary_found(cfg: Config) -> bool:
    """True when ``llama-server`` exists in the configured llama.cpp folder."""
    return (Path(cfg.bin_dir).expanduser() / "llama-server").is_file()


def next_steps(cfg: Config, models: list[Path], infos: dict[str, ModelInfo],
               calibration: Calibration, status: GatewayStatus | None,
               build: str) -> list[Suggestion]:
    """What the user should probably do next, most urgent first.

    Rules, in this order: llama-server missing (Settings); no model files (Discover); models
    without any preset (Models); estimated active presets of a MoE model with experts in
    RAM (tune the split); other active presets that are only estimated, never tuned (tune);
    tuned presets gone stale because the model file or the llama.cpp build changed (re-tune);
    the gateway stopped while presets exist (start it). Tuning and starting are not suggested
    while llama-server is missing. Models whose header could not be read are ignored by the
    tuning rules.

    Args:
        cfg: Config holding the presets.
        models: Model files found on disk.
        infos: Parsed headers by model path (a missing key means unreadable or not read yet).
        calibration: Measured trials, used to tell whether a tuned preset is stale.
        status: Gateway status, or None before the first poll.
        build: The llama-server version string now, "" when unknown.

    Returns:
        The suggestions; empty when there is nothing to do.
    """
    out: list[Suggestion] = []
    found = binary_found(cfg)
    if not found:
        out.append(Suggestion(
            "binary", f"llama-server not found in {cfg.bin_dir}: set the llama.cpp folder in "
                      "Settings", ACT_SETTINGS))
    if not models:
        dirs = ", ".join(cfg.model_dirs) or "(no model folders set)"
        out.append(Suggestion("no-models", f"No models found in {dirs}: download one in Discover",
                              ACT_DISCOVER))
        return out
    aliases = aliases_for(models, cfg)
    bare = [m for m in models if not cfg.has_profile(str(m))]
    for m in bare[:MAX_PER_RULE]:
        out.append(Suggestion(f"no-presets:{m}", f"{aliases[m]} has no preset yet: pick what it "
                              "is for in Models (Enter)", ACT_MODELS, str(m)))
    if found:
        out += _tuning_steps(cfg, models, infos, calibration, build, aliases)
    configured = [m for m in models if cfg.has_profile(str(m))]
    if found and configured and status is not None and not status.running:
        n = len(configured)
        out.append(Suggestion(
            "start-gateway", f"Start the gateway to serve {n} model{'s' * (n != 1)} at "
                             f"{status.url}", ACT_START_GATEWAY))
    return out


def _tuning_steps(cfg: Config, models: list[Path], infos: dict[str, ModelInfo],
                  calibration: Calibration, build: str, aliases: dict[Path, str]) -> list[Suggestion]:
    """Tune suggestions: MoE expert splits first, then other estimated active presets, then
    re-tune ones for stale tuned presets."""
    moe: list[Suggestion] = []
    untuned: list[Suggestion] = []
    stale: list[Suggestion] = []
    for m in models:
        ent = cfg.models.get(str(m))
        if ent is None or str(m) not in infos:
            continue
        info = infos[str(m)]
        active = ent.presets.get(ent.active)
        if active is not None and active.source == ESTIMATED:
            if info.expert_count > 0 and active.profile.n_cpu_moe > 0:
                moe.append(Suggestion(
                    f"tune-moe:{m}:{ent.active}",
                    f"{aliases[m]} is a MoE model with experts in RAM: tune it to find the "
                    "fastest split", ACT_TUNE, str(m), ent.active))
            else:
                untuned.append(Suggestion(
                    f"tune:{m}:{ent.active}", f"{aliases[m]} ({ent.active}) is only an estimate: "
                    "tune it to measure what fits", ACT_TUNE, str(m), ent.active))
        for name, preset in sorted(ent.presets.items()):
            if preset.source == TUNED and calibration.is_stale(preset, m, build):
                stale.append(Suggestion(
                    f"retune:{m}:{name}", f"{aliases[m]} ({name}) was tuned for a different "
                    "model file or llama.cpp build: tune it again", ACT_TUNE, str(m), name))
    return moe[:MAX_PER_RULE] + untuned[:MAX_PER_RULE] + stale[:MAX_PER_RULE]
