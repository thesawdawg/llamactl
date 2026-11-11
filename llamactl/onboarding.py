"""First-run setup: what each step shows and checks, and what finishing does.

Pure logic with no Textual imports; ``screens/onboarding.py`` is a thin shell over it. The flow
works on a `Draft` (copies of the settings it may change) so that walking away half way, or
skipping, leaves the real config alone; `finish` is the one place that applies the draft.

Steps: (1) find llama.cpp, (2) look at the hardware, (3) confirm the model folders and scan
them, (4) choose what the models are for and optionally a first model, (5) apply and show the
endpoint. Every step can be passed over with Next; "Skip setup" ends the flow at any point.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import actions, planner
from .calibration import Calibration
from .config import Config
from .fmt import human
from .gguf import ModelInfo
from .hardware import Host
from .intents import INTENTS, Intent, resolve
from .library import aliases_for

BINARY = "llama-server"
SRC_CONFIG, SRC_PATH, SRC_NONE = "config", "PATH", "none"
LLAMA_HINT = ("Get llama.cpp from github.com/ggml-org/llama.cpp (prebuilt releases, or build it with "
              "cmake), then enter the folder that holds llama-server.")
TYPICAL_KV_PER_TOKEN = 128 * 1024        # f16 K+V of a typical 8B model: 32 layers x 8 kv heads x 128 x 2 x 2 bytes
TRAINED_CTX_ASSUMED = 32768              # context assumed for an intent that aims for the trained one

STEP_LLAMA, STEP_HARDWARE, STEP_MODELS, STEP_PURPOSE, STEP_FINISH = (
    "llama", "hardware", "models", "purpose", "finish")
STEPS = (STEP_LLAMA, STEP_HARDWARE, STEP_MODELS, STEP_PURPOSE, STEP_FINISH)
TITLES = {STEP_LLAMA: "llama.cpp", STEP_HARDWARE: "Hardware", STEP_MODELS: "Models",
          STEP_PURPOSE: "Purpose", STEP_FINISH: "Finish"}

Which = Callable[[str], "str | None"]
BuildOf = Callable[[str], str]


# --------------------------------------------------------------------------- step 1: llama.cpp

@dataclass(frozen=True)
class LlamaLocation:
    """Where llama-server was found.

    Attributes:
        bin_dir: Folder holding it, "" when not found.
        source: ``SRC_CONFIG`` (the configured folder), ``SRC_PATH`` or ``SRC_NONE``.
    """

    bin_dir: str = ""
    source: str = SRC_NONE

    @property
    def found(self) -> bool:
        """True when a folder was found."""
        return bool(self.bin_dir)

    @property
    def binary(self) -> str:
        """Full path of llama-server, "" when not found."""
        return str(Path(self.bin_dir) / BINARY) if self.bin_dir else ""


def check_bin_dir(value: str) -> str:
    """Validate a llama.cpp folder.

    Args:
        value: What the user typed.

    Returns:
        "" when the folder holds a ``llama-server`` file, else a one-line reason.
    """
    text = value.strip()
    if not text:
        return "enter the folder that holds llama-server"
    folder = Path(text).expanduser()
    if not folder.is_dir():
        return f"{text} is not a folder"
    if not (folder / BINARY).is_file():
        return f"no {BINARY} inside {text}"
    return ""


def locate_llama(bin_dir: str, which: Which = shutil.which) -> LlamaLocation:
    """Look for llama-server in the configured folder, then on ``PATH``.

    Args:
        bin_dir: The configured llama.cpp folder.
        which: ``shutil.which`` compatible lookup.

    Returns:
        The first place that has it, or a location with ``found`` False.
    """
    if bin_dir.strip() and not check_bin_dir(bin_dir):
        return LlamaLocation(str(Path(bin_dir).expanduser()), SRC_CONFIG)
    on_path = which(BINARY)
    if on_path:
        return LlamaLocation(str(Path(on_path).parent), SRC_PATH)
    return LlamaLocation()


def llama_text(loc: LlamaLocation, build: str) -> str:
    """What step 1 says about a location.

    Args:
        loc: Where llama-server was found.
        build: Its version string, "" when unknown or not read yet.

    Returns:
        One or two lines.
    """
    if not loc.found:
        return f"llama-server was not found in the configured folder or on PATH.\n{LLAMA_HINT}"
    where = "the configured folder" if loc.source == SRC_CONFIG else "PATH"
    return f"Found {loc.binary} (from {where}).\nBuild: {build or 'reading the version...'}"


# --------------------------------------------------------------------------- step 2: hardware

def _gb(n: float) -> str:
    """Bytes as ``12.3`` (GB)."""
    return f"{n / 1e9:.1f}"


def intent_ctx(intent: Intent) -> int:
    """The context per slot an intent aims for, with the trained one assumed to be 32k.

    Args:
        intent: The intent.

    Returns:
        Tokens.
    """
    return intent.ctx_target or TRAINED_CTX_ASSUMED


def fit_bytes(host: Host, intent: Intent, headroom_pct: float) -> int:
    """Largest model file that runs fully on the GPU (or in RAM without one) for an intent.

    The KV cache is assumed to be a typical 8B model's (f16); the real figure depends on the
    model, which the Models view shows once one is chosen.

    Args:
        host: Host snapshot.
        intent: The intent whose context target is reserved.
        headroom_pct: Percentage of memory kept free.

    Returns:
        Bytes of weights; 0 when even a tiny model would not fit.
    """
    reserve = planner.OVERHEAD + planner.COMPUTE_MIN + TYPICAL_KV_PER_TOKEN * intent_ctx(intent)
    pool = (planner.gpu_limit(host, headroom_pct) if host.gpus else
            max(0, host.ram_total - planner.headroom_bytes(host.ram_total, headroom_pct)))
    return max(0, pool - reserve)


def hardware_sentence(host: Host, intent: Intent, headroom_pct: float = 10) -> str:
    """What the detected hardware means, in one plain sentence.

    Args:
        host: Host snapshot.
        intent: The default intent (its context target sets the reserve for the KV cache).
        headroom_pct: Percentage of memory kept free.

    Returns:
        The sentence; a "still detecting" note while the host is empty.
    """
    if not host.gpus and not host.ram_total:
        return "Still detecting the hardware..."
    ctx = f"the {intent.label} intent's {intent_ctx(intent) // 1024}k context"
    room = fit_bytes(host, intent, headroom_pct)
    ram = f"{_gb(host.ram_total)} GB of RAM"
    if not host.gpus:
        if room <= 0:
            return f"No GPU was found and {ram} is too little for even a small model at {ctx}."
        return (f"No GPU was found, so models run on the CPU: a model file up to about {_gb(room)} GB "
                f"fits in {ram} at {ctx}, at a few tokens per second.")
    n = len(host.gpus)
    what = (host.gpus[0].name if n == 1 else f"{n} GPUs")
    mem = f"{_gb(host.vram_total)} GB {'of unified memory' if host.unified else 'VRAM'}"
    if room <= 0:
        return (f"{what} ({mem}) is too small to hold even a small model at {ctx}; models will "
                "run mostly on the CPU.")
    tail = ("." if host.unified else
            f"; larger ones spill into your {ram} and run slower.")
    return (f"With {what} ({mem}), a model file up to about {_gb(room)} GB runs fully on the GPU "
            f"at {ctx}{tail}")


# --------------------------------------------------------------------------- step 3: models

@dataclass(frozen=True)
class ScanSummary:
    """The models a scan found.

    Attributes:
        count: Model files found.
        total: Their combined size in bytes.
        unreadable: Files whose GGUF header could not be read.
    """

    count: int = 0
    total: int = 0
    unreadable: int = 0

    def text(self, dirs: list[str]) -> str:
        """One line for the screen.

        Args:
            dirs: The folders that were scanned.

        Returns:
            E.g. ``3 models, 41.2 GB in total`` or the no-models hint.
        """
        if not dirs:
            return "No model folders yet: add one below."
        if not self.count:
            return "No models found in these folders."
        bad = f" ({self.unreadable} unreadable)" if self.unreadable else ""
        return f"{self.count} model{'s' * (self.count != 1)}, {human(self.total)} in total{bad}"


def scan_summary(models: list[Path], infos: dict[str, ModelInfo]) -> ScanSummary:
    """Count and size the scanned models.

    Args:
        models: Model files found.
        infos: Parsed headers by path; a missing one counts as unreadable (its size from disk).

    Returns:
        The summary.
    """
    total = bad = 0
    for p in models:
        info = infos.get(str(p))
        if info is not None:
            total += info.size
            continue
        bad += 1
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return ScanSummary(len(models), total, bad)


# --------------------------------------------------------------------------- step 4: purpose

def intent_blurb(intent: Intent) -> str:
    """A card line for an intent without any per-model numbers.

    Args:
        intent: The intent.

    Returns:
        E.g. ``Chat: interactive chat frontends (16k context, 1 slot)``.
    """
    ctx = ("the model's trained context" if not intent.ctx_target
           else f"about {intent.ctx_target // 1024}k context")
    slots = f"{intent.slots} slot{'s' * (intent.slots != 1)}"
    return f"{intent.label}: {intent.description} ({ctx}, {slots})"


@dataclass(frozen=True)
class ModelChoice:
    """A model offered in step 4, with the verdict for the chosen intent.

    Attributes:
        path: Model file.
        alias: Name clients will use.
        size: File size in bytes.
        verdict: Planner verdict ("GPU", "GPU+CPU", "CPU", "NO").
        intent: Key of the intent the verdict is for (the default one, or the first that
            applies when the model is an embedding model).
        ctx: Planned context per slot.
    """

    path: str
    alias: str
    size: int
    verdict: str
    intent: str
    ctx: int


def model_choices(models: list[Path], infos: dict[str, ModelInfo], host: Host, cfg: Config,
                  calibration: Calibration | None, intent_key: str) -> list[ModelChoice]:
    """Each readable model with its verdict for an intent.

    Args:
        models: Model files found.
        infos: Parsed headers by path; unreadable models are left out.
        host: Host snapshot.
        cfg: Config supplying the base profile, headroom and aliases.
        calibration: Measured data for the plans, or None.
        intent_key: The intent to judge by.

    Returns:
        Choices in model order.
    """
    aliases = aliases_for(models, cfg)
    out = []
    for p in models:
        info = infos.get(str(p))
        if info is None:
            continue
        intent = resolve(intent_key, info)
        plan = actions.plan_for(cfg, info, host, intent, calibration)
        out.append(ModelChoice(str(p), aliases[p], info.size, plan.verdict, intent.key,
                               plan.ctx_per_slot))
    return out


# --------------------------------------------------------------------------- the flow

@dataclass
class Draft:
    """The choices made so far; nothing here touches the config until `finish`.

    Attributes:
        bin_dir: llama.cpp folder.
        model_dirs: Model folders.
        default_intent: Intent key made the default.
        model: Path of the first model to prepare, "" for none.
        tune: Open the tune screen when the flow ends.
        start_gateway: Start the gateway when the flow ends.
    """

    bin_dir: str = ""
    model_dirs: list[str] = field(default_factory=list)
    default_intent: str = "chat"
    model: str = ""
    tune: bool = False
    start_gateway: bool = True


class Flow:
    """Where the user is in the five steps, plus the draft.

    Args:
        cfg: The current config; the draft starts as a copy of its relevant settings.
    """

    def __init__(self, cfg: Config) -> None:
        self.index = 0
        keys = [i.key for i in INTENTS]
        self.draft = Draft(cfg.bin_dir, list(cfg.model_dirs),
                           cfg.default_intent if cfg.default_intent in keys else "chat")

    @property
    def step(self) -> str:
        """Id of the current step (``STEP_*``)."""
        return STEPS[self.index]

    @property
    def first(self) -> bool:
        """True on the first step (Back is unavailable)."""
        return self.index == 0

    @property
    def last(self) -> bool:
        """True on the last step (Next becomes Finish)."""
        return self.index == len(STEPS) - 1

    def progress(self) -> str:
        """``Step 2 of 5: Hardware``."""
        return f"Step {self.index + 1} of {len(STEPS)}: {TITLES[self.step]}"

    def next(self) -> bool:
        """Go to the next step.

        Returns:
            False on the last step.
        """
        if self.last:
            return False
        self.index += 1
        return True

    def back(self) -> bool:
        """Go to the previous step.

        Returns:
            False on the first step.
        """
        if self.first:
            return False
        self.index -= 1
        return True

    def llama_ok(self) -> bool:
        """True when the drafted llama.cpp folder holds llama-server."""
        return not check_bin_dir(self.draft.bin_dir)


@dataclass(frozen=True)
class FinishResult:
    """What finishing did, for the last screen.

    Attributes:
        model: Model path prepared, "" when none was chosen.
        intent: Intent key of its preset, "".
        preset: Preset name, "".
        created: The preset was planned just now (it did not exist).
        served: Name clients request.
        url: OpenAI-style base URL (``http://host:port/v1``).
        snippet: A ``curl`` line, "" without a model.
        start_gateway: The caller should start the gateway.
        tune: The caller should open the tune screen.
        bin_changed: The llama.cpp folder differs from before (re-read the build, host).
        dirs_changed: The model folders differ from before (rescan).
    """

    model: str = ""
    intent: str = ""
    preset: str = ""
    created: bool = False
    served: str = ""
    url: str = ""
    snippet: str = ""
    start_gateway: bool = False
    tune: bool = False
    bin_changed: bool = False
    dirs_changed: bool = False


def finish(cfg: Config, draft: Draft, infos: dict[str, ModelInfo], host: Host,
           calibration: Calibration | None, base_url: str) -> FinishResult:
    """Apply the draft: settings, the first model's preset, ``onboarded``; save the config.

    Must run on the thread that owns the config. For a chosen model an ``estimated`` preset for
    the default intent is created when missing (it becomes the active preset only when the
    model had none); an existing preset is never replaced.

    Args:
        cfg: Config to update and save.
        draft: The choices.
        infos: Parsed headers by path (the chosen model's is needed).
        host: Host snapshot for planning.
        calibration: Measured data for the plan, or None.
        base_url: Gateway URL without ``/v1``.

    Returns:
        What was done.

    Raises:
        LookupError: When the chosen model has no parsed header.
        OSError: When the config cannot be written.
    """
    bin_changed = draft.bin_dir.strip() != cfg.bin_dir
    dirs_changed = draft.model_dirs != cfg.model_dirs
    info = infos.get(draft.model) if draft.model else None
    if draft.model and info is None:
        raise LookupError("the chosen model's header has not been read")
    if draft.bin_dir.strip():
        cfg.bin_dir = draft.bin_dir.strip()
    cfg.model_dirs = list(draft.model_dirs)
    cfg.default_intent = draft.default_intent
    cfg.onboarded = True
    url = base_url.rstrip("/")
    out = FinishResult(url=f"{url}/v1", bin_changed=bin_changed, dirs_changed=dirs_changed)
    if info is not None:
        intent = resolve(draft.default_intent, info)
        chosen = actions.pick_preset(cfg, info, host, calibration, intent.key,
                                     activate=cfg.preset(draft.model) is None, save=False)
        served = actions.served_name(cfg, draft.model, chosen.name)
        runnable = not check_bin_dir(cfg.bin_dir)
        out = FinishResult(
            model=draft.model, intent=intent.key, preset=chosen.name, created=chosen.created,
            served=served, url=out.url,
            snippet=actions.curl_snippet(url, served, api_key=bool(cfg.gateway.api_key),
                                         embedding=info.is_embedding),
            start_gateway=draft.start_gateway and runnable, tune=draft.tune and runnable,
            bin_changed=bin_changed, dirs_changed=dirs_changed)
    cfg.save()
    return out


def skip(cfg: Config) -> bool:
    """Leave the flow without applying the draft; the user will not be asked again.

    Args:
        cfg: Config to mark as onboarded and save when it was not.

    Returns:
        True when the config was changed (and saved).

    Raises:
        OSError: When the config cannot be written.
    """
    if cfg.onboarded:
        return False
    cfg.onboarded = True
    cfg.save()
    return True
