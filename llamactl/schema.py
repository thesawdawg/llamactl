"""Settings registry: single source of truth for profile fields and CLI flags."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .config import Profile
    from .gguf import ModelInfo
    from .hostinfo import Host


class Kind(StrEnum):
    """Widget/validation type of a setting."""

    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    CHOICE = "choice"
    STR = "str"
    PATH = "path"


class Group(StrEnum):
    """Editor section a setting belongs to."""

    MEMORY = "memory"
    SAMPLING = "sampling"
    SERVER = "server"
    GENERAL = "general"


Impact = Callable[["Profile", "ModelInfo | None", "Host"], str]

CACHE_TYPES = ("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1")
FLASH_ATTN_MODES = ("on", "off", "auto")
LOAD_MODES = ("auto", "none", "mmap", "mlock", "mmap+mlock", "dio")


@dataclass(frozen=True)
class Setting:
    """One launch setting: profile field, llama.cpp flag, UI metadata.

    Attributes:
        key: Profile attribute and config key.
        flag: llama.cpp flag, e.g. "-c"; "" when handled specially.
        kind: Widget/validation type.
        label: Short UI label.
        hint: One line shown under the field.
        help: Multi-paragraph text for the help panel (plain ASCII).
        group: Editor section.
        default: Value meaning "let llama.cpp decide" where possible.
        modes: Launch modes the setting applies to.
        choices: Allowed values for CHOICE.
        minimum: Inclusive minimum for INT/FLOAT.
        maximum: Inclusive maximum for INT/FLOAT.
        step: UI stepper increment.
        unset: Value that means "omit the flag" (None = always emitted).
        impact: Optional live impact line computed from profile/model/host.
    """

    key: str
    flag: str
    kind: Kind
    label: str
    hint: str
    help: str
    group: Group
    default: Any
    modes: frozenset[str] = frozenset({"server", "cli"})
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    unset: Any = None
    impact: Impact | None = None


def _gb(n: float) -> str:
    """Format bytes as GB for impact lines."""
    return f"{n / 1e9:.1f} GB"


def _budget_impact(prof: Profile, info: ModelInfo | None, host: Host) -> str:
    """One-line memory impact of the current profile against this host.

    Args:
        prof: Profile being edited.
        info: Parsed model metadata, or None to describe the host only.
        host: Host resource snapshot.

    Returns:
        A single line of text shown under the field.
    """
    from .recommend import estimate_budget

    if info is None:
        return f"Host: {_gb(host.vram_free)} VRAM free, {_gb(host.ram_free)} RAM free"
    b = estimate_budget(info, prof, host)
    return (f"Needs ~{_gb(b.total())} ({_gb(b.kv)} KV) - {_gb(host.vram_free)} VRAM free, "
            f"{_gb(host.ram_free)} RAM free")


def _slots_impact(prof: Profile, info: ModelInfo | None, host: Host) -> str:
    """One-line impact of the parallel slot count on per-request context.

    Args:
        prof: Profile being edited.
        info: Parsed model metadata, or None when editing the default profile.
        host: Host snapshot (unused; -c is a pool, not multiplied by slots).

    Returns:
        A single line of text shown under the field.
    """
    if info is None:
        return "The context pool is split evenly between slots."
    ctx = prof.ctx_size or info.ctx_train
    if not prof.parallel:
        return f"auto slots share the {ctx} ctx pool"
    return f"{prof.parallel} slots -> each request gets {ctx // prof.parallel} of {ctx} ctx"


def validate(setting: Setting, raw: str) -> Any:
    """Parse and validate a raw text value for a setting.

    Args:
        setting: The setting the value belongs to.
        raw: Text as typed into the field.

    Returns:
        The parsed value in the setting's Python type.

    Raises:
        ValueError: With a user-readable message when the value is invalid.
    """
    raw = raw.strip()
    if setting.key == "gpu_layers":
        if raw in ("auto", "all"):
            return raw
        try:
            v = int(raw)
        except ValueError:
            raise ValueError("gpu_layers must be 'auto', 'all' or an integer")
        if v < 0:
            raise ValueError("gpu_layers must be >= 0")
        return str(v)
    if setting.kind == Kind.INT:
        try:
            v = int(raw)
        except ValueError:
            raise ValueError(f"{setting.key} must be an integer")
    elif setting.kind == Kind.FLOAT:
        try:
            v = float(raw)
        except ValueError:
            raise ValueError(f"{setting.key} must be a number")
    elif setting.kind == Kind.BOOL:
        low = raw.lower()
        if low in ("1", "true", "on", "yes"):
            return True
        if low in ("0", "false", "off", "no"):
            return False
        raise ValueError(f"{setting.key} must be true or false")
    elif setting.kind == Kind.CHOICE:
        if raw not in setting.choices:
            raise ValueError(f"{setting.key} must be one of {', '.join(setting.choices)}")
        return raw
    else:
        return raw
    if setting.minimum is not None and v < setting.minimum:
        raise ValueError(f"{setting.key} must be >= {setting.minimum:g}")
    if setting.maximum is not None and v > setting.maximum:
        raise ValueError(f"{setting.key} must be <= {setting.maximum:g}")
    return v


# Registry order is the order the editor shows settings.
SETTINGS: tuple[Setting, ...] = (
    Setting(
        key="ctx_size", flag="-c", kind=Kind.INT, label="Context size", group=Group.MEMORY,
        hint="Max tokens the model remembers (prompt + reply). 0 = model's trained max.",
        help=("Size of the prompt context. Larger context lets the model see more text but the\n"
              "KV cache grows linearly with it, using VRAM (or RAM if layers are not offloaded).\n\n"
              "0 loads the model's trained maximum, which can be very large on modern models.\n"
              "Typical values: 4096 for short chats, 8192-32768 for long documents.\n\n"
              "Flag: -c, --ctx-size N"),
        default=0, unset=0, step=1024, minimum=0, impact=_budget_impact),
    Setting(
        key="gpu_layers", flag="-ngl", kind=Kind.STR, label="GPU layers", group=Group.MEMORY,
        hint="Layers kept in VRAM: 'all', 'auto' or a number. Higher = faster, needs more VRAM.",
        help=("Max number of model layers stored in VRAM. More layers on the GPU means faster\n"
              "generation but more VRAM. 'all' offloads everything, 'auto' lets llama.cpp\n"
              "decide, a number offloads exactly that many layers.\n\n"
              "Lower it if you run out of memory; the rest runs on the CPU.\n\n"
              "Flag: -ngl, --gpu-layers N"),
        default="all", unset="", impact=_budget_impact),
    Setting(
        key="n_cpu_moe", flag="--n-cpu-moe", kind=Kind.INT, label="CPU MoE layers", group=Group.MEMORY,
        hint="Keep expert weights of the first N layers in RAM. MoE models only. 0 = off.",
        help=("Keeps the Mixture-of-Experts weights of the first N layers on the CPU while the\n"
              "rest runs on the GPU. Only the few experts actually used per token are read, so\n"
              "a big MoE model can run almost at GPU speed with little VRAM.\n\n"
              "Only meaningful for MoE models (expert_count > 0). 0 disables.\n\n"
              "Flag: --n-cpu-moe N"),
        default=0, unset=0, minimum=0, impact=_budget_impact),
    Setting(
        key="cache_type_k", flag="-ctk", kind=Kind.CHOICE, label="K cache type", group=Group.MEMORY,
        hint="KV cache data type for K. f16 default; q8_0 halves memory.",
        help=("Data type of the K (key) cache. f16 is the default and lossless.\n"
              "q8_0 roughly halves KV memory with a small quality loss; q4_0 is more\n"
              "aggressive. Quantised K works even without flash attention.\n\n"
              f"Allowed: {', '.join(CACHE_TYPES)}.\n\n"
              "Flag: -ctk, --cache-type-k TYPE"),
        default="f16", unset="f16", choices=CACHE_TYPES, impact=_budget_impact),
    Setting(
        key="cache_type_v", flag="-ctv", kind=Kind.CHOICE, label="V cache type", group=Group.MEMORY,
        hint="KV cache data type for V. Quantised V requires flash_attn != off.",
        help=("Data type of the V (value) cache. Same choices as cache_type_k, but a\n"
              "quantised V cache requires flash attention to be on or auto.\n\n"
              f"Allowed: {', '.join(CACHE_TYPES)}.\n\n"
              "Flag: -ctv, --cache-type-v TYPE"),
        default="f16", unset="f16", choices=CACHE_TYPES, impact=_budget_impact),
    Setting(
        key="flash_attn", flag="-fa", kind=Kind.CHOICE, label="Flash attention", group=Group.MEMORY,
        hint="on / off / auto. Faster attention and lower memory on supported GPUs.",
        help=("Enables the flash attention kernel. 'on' is faster and uses less KV memory on\n"
              "supported GPUs and is required for quantised V cache. 'auto' lets llama.cpp\n"
              "enable it when safe. 'off' forces the old code path.\n\n"
              "Flag: -fa, --flash-attn [on|off|auto]"),
        default="auto", choices=FLASH_ATTN_MODES),
    Setting(
        key="batch_size", flag="-b", kind=Kind.INT, label="Batch size", group=Group.MEMORY,
        hint="Logical max batch size (prompt processing). 0 = llama.cpp default (2048).",
        help=("Logical maximum batch size: how many tokens are processed per step during\n"
              "prompt ingestion. Larger batches speed up long prompts but need a bigger\n"
              "compute buffer. 0 uses the llama.cpp default of 2048.\n\n"
              "Flag: -b, --batch-size N"),
        default=0, unset=0, step=256, minimum=0, impact=_budget_impact),
    Setting(
        key="ubatch_size", flag="-ub", kind=Kind.INT, label="Micro batch", group=Group.MEMORY,
        hint="Physical max batch size, must be <= batch size. 0 = default (512).",
        help=("Physical maximum batch size: the chunk the logical batch is split into for the\n"
              "actual compute. Must be <= batch_size when both are set. Smaller values reduce\n"
              "the compute buffer at the cost of slower prompt processing.\n\n"
              "Flag: -ub, --ubatch-size N"),
        default=0, unset=0, step=128, minimum=0, impact=_budget_impact),
    Setting(
        key="load_mode", flag="--load-mode", kind=Kind.CHOICE, label="Load mode", group=Group.MEMORY,
        hint="How weights are read: auto / none / mmap / mlock / mmap+mlock / dio.",
        help=("Model loading mode. auto uses mmap unless a device does not support it.\n"
              "mmap memory-maps the file; mlock pins it in RAM so it cannot be swapped;\n"
              "dio uses DirectIO when available. The default is almost always right.\n\n"
              "Flag: --load-mode MODE"),
        default="auto", unset="auto", choices=LOAD_MODES),
    Setting(
        key="tensor_split", flag="-ts", kind=Kind.STR, label="Tensor split", group=Group.MEMORY,
        hint="Comma-separated VRAM fractions per GPU, e.g. 3,1. Only useful with >1 GPU.",
        help=("Fraction of the model to offload to each GPU, comma-separated proportions,\n"
              "e.g. '3,1' gives the first GPU three quarters. Only shown when more than\n"
              "one GPU is present. Empty lets llama.cpp split evenly.\n\n"
              "Flag: -ts, --tensor-split N0,N1,..."),
        default="", unset=""),
    Setting(
        key="threads", flag="-t", kind=Kind.INT, label="CPU threads", group=Group.MEMORY,
        hint="Threads for CPU work. Usually the number of physical cores. 0 = auto.",
        help=("Number of CPU threads used during generation. On a CPU-only or partially\n"
              "offloaded model this is the main speed knob: the number of physical cores\n"
              "(not hyperthreads) is usually best. 0 lets llama.cpp choose.\n\n"
              "Flag: -t, --threads N"),
        default=0, unset=0, minimum=0),
    Setting(
        key="temp", flag="--temp", kind=Kind.FLOAT, label="Temperature", group=Group.SAMPLING,
        hint="Randomness of the output. -1 = llama.cpp default (0.8). Range 0-2.",
        help=("Sampling temperature. Lower is more deterministic, higher more creative.\n"
              "0 is nearly greedy; most models work best around 0.6-1.0.\n"
              "-1 uses the llama.cpp default of 0.8.\n\n"
              "Flag: --temp, --temperature N"),
        default=-1.0, unset=-1, minimum=-1, maximum=2),
    Setting(
        key="top_k", flag="--top-k", kind=Kind.INT, label="Top K", group=Group.SAMPLING,
        hint="Keep the K most likely tokens. -1 = default (40); 0 disables.",
        help=("Top-k sampling: only the K most likely next tokens are considered.\n"
              "Lower values make output more predictable. -1 uses the default of 40;\n"
              "0 disables top-k entirely.\n\n"
              "Flag: --top-k N"),
        default=-1, unset=-1, minimum=-1),
    Setting(
        key="top_p", flag="--top-p", kind=Kind.FLOAT, label="Top P", group=Group.SAMPLING,
        hint="Nucleus sampling: keep tokens up to cumulative probability P. -1 = default (0.95).",
        help=("Top-p (nucleus) sampling: only tokens making up the top P of probability\n"
              "mass are kept. 1.0 disables. -1 uses the llama.cpp default of 0.95.\n\n"
              "Flag: --top-p N"),
        default=-1.0, unset=-1, minimum=-1, maximum=1),
    Setting(
        key="min_p", flag="--min-p", kind=Kind.FLOAT, label="Min P", group=Group.SAMPLING,
        hint="Drop tokens below P * top-token probability. -1 = default (0.05).",
        help=("Min-p sampling: tokens with probability below P times the top token's\n"
              "probability are dropped. A good low-temperature alternative to top-p.\n"
              "0.0 disables; -1 uses the llama.cpp default of 0.05.\n\n"
              "Flag: --min-p N"),
        default=-1.0, unset=-1, minimum=-1, maximum=1),
    Setting(
        key="repeat_penalty", flag="--repeat-penalty", kind=Kind.FLOAT, label="Repeat penalty", group=Group.SAMPLING,
        hint="Penalise repeated tokens. -1 = default (1.0 = off).",
        help=("Repeat penalty: divides the logits of recently seen tokens, discouraging\n"
              "loops. 1.0 disables; 1.05-1.2 is a typical range. -1 uses the llama.cpp\n"
              "default (1.0, disabled).\n\n"
              "Flag: --repeat-penalty N"),
        default=-1.0, unset=-1, minimum=-1),
    Setting(
        key="seed", flag="-s", kind=Kind.INT, label="Seed", group=Group.SAMPLING,
        hint="RNG seed. -1 = random each run.",
        help=("Random seed for sampling. Set a fixed value for reproducible outputs.\n"
              "-1 uses a random seed each run.\n\n"
              "Flag: -s, --seed SEED"),
        default=-1, unset=-1, minimum=-1),
    Setting(
        key="host", flag="--host", kind=Kind.STR, label="Host", group=Group.SERVER,
        hint="Listen address. 127.0.0.1 = this machine only; 0.0.0.0 = your network.",
        help=("IP address the server listens on. 127.0.0.1 keeps it private to this\n"
              "machine. 0.0.0.0 exposes it to your local network; combine with an API\n"
              "key if you do that, there is no auth by default.\n\n"
              "Flag: --host HOST"),
        default="127.0.0.1", modes=frozenset({"server"})),
    Setting(
        key="port", flag="--port", kind=Kind.INT, label="Port", group=Group.SERVER,
        hint="Listen port. 0 = first free port from 8080, resolved at launch.",
        help=("TCP port the server listens on (default 8080). 0 lets llamactl pick the\n"
              "first free port from 8080 so several models can run side by side.\n\n"
              "Flag: --port PORT"),
        default=0, modes=frozenset({"server"})),
    Setting(
        key="api_key", flag="--api-key", kind=Kind.STR, label="API key", group=Group.SERVER,
        hint="Require this key in the Authorization header. Empty = no auth.",
        help=("API key clients must send as a bearer token. Without it the server accepts\n"
              "any request - set one if you bind to 0.0.0.0. Keys are stored in the\n"
              "config file in plain text.\n\n"
              "Flag: --api-key KEY"),
        default="", unset="", modes=frozenset({"server"})),
    Setting(
        key="parallel", flag="-np", kind=Kind.INT, label="Slots", group=Group.SERVER,
        hint="Requests served at once. Context is split between slots. 0 = auto.",
        help=("Number of server slots: how many requests are decoded in parallel.\n"
              "The context is divided between slots, so more slots means less\n"
              "context each. 0 lets llama.cpp choose.\n\n"
              "Flag: -np, --parallel N"),
        default=0, unset=0, minimum=0, modes=frozenset({"server"}), impact=_slots_impact),
    Setting(
        key="cont_batching", flag="-cb", kind=Kind.BOOL, label="Continuous batching", group=Group.SERVER,
        hint="Let slots pick up new requests while others are still decoding. On is right.",
        help=("Continuous batching: finished slots start serving the next request\n"
              "immediately instead of waiting for the whole batch. Keep it on unless a\n"
              "model misbehaves; disabling emits -nocb.\n\n"
              "Flag: -cb, --cont-batching / -nocb, --no-cont-batching"),
        default=True, unset=True, modes=frozenset({"server"})),
    Setting(
        key="metrics", flag="--metrics", kind=Kind.BOOL, label="Metrics endpoint", group=Group.SERVER,
        hint="Expose a Prometheus-compatible /metrics endpoint.",
        help=("Enables the Prometheus-compatible /metrics endpoint with per-slot usage\n"
              "and performance counters. Off by default.\n\n"
              "Flag: --metrics"),
        default=False, unset=False, modes=frozenset({"server"})),
    Setting(
        key="alias", flag="--alias", kind=Kind.STR, label="Alias", group=Group.SERVER,
        hint="Model name reported by the API instead of the file name.",
        help=("Model name exposed through the API (e.g. in /v1/models) instead of the\n"
              "GGUF file name. Useful when a client expects a specific model id.\n\n"
              "Flag: --alias STRING"),
        default="", unset="", modes=frozenset({"server"})),
    Setting(
        key="draft_model", flag="-md", kind=Kind.PATH, label="Draft model", group=Group.SERVER,
        hint="Small draft model for speculative decoding. Empty = off.",
        help=("Path to a small draft model used for speculative decoding with the same\n"
              "vocabulary: the draft proposes tokens, the big model verifies them, which\n"
              "can nearly double generation speed when it works.\n\n"
              "Flag: -md, --spec-draft-model FNAME"),
        default="", unset="", modes=frozenset({"server"})),
    Setting(
        key="mmproj", flag="--mmproj", kind=Kind.PATH, label="mmproj", group=Group.SERVER,
        hint="Vision projector file for multimodal models.",
        help=("Path to a multimodal projector (mmproj) GGUF that pairs with a vision\n"
              "model so it can accept images. Only needed for vision-capable models.\n\n"
              "Flag: --mmproj FILE"),
        default="", unset="", modes=frozenset({"server"})),
    Setting(
        key="extra_args", flag="", kind=Kind.STR, label="Extra args", group=Group.GENERAL,
        hint="Raw llama.cpp flags appended last, e.g. --jinja --n-cpu-moe 24 --temp 0.7.",
        help=("Raw flags appended to the command line after all settings above. Use it\n"
              "for options llamactl does not expose yet; run llama-server --help for\n"
              "the full list. Split with normal shell quoting rules."),
        default="", unset=""),
)

by_key: dict[str, Setting] = {s.key: s for s in SETTINGS}


def for_group(group: Group) -> list[Setting]:
    """Settings in one editor section, registry order.

    Args:
        group: The section to filter by.

    Returns:
        Settings belonging to `group`.
    """
    return [s for s in SETTINGS if s.group == group]


def for_mode(mode: str) -> list[Setting]:
    """Settings valid for a launch mode ("server" or "cli"), registry order.

    Args:
        mode: The launch mode.

    Returns:
        Settings emitted for that mode.
    """
    return [s for s in SETTINGS if mode in s.modes]
