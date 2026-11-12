"""Settings registry: single source of truth for profile fields and CLI flags."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .config import Profile
    from .gguf import ModelInfo
    from .hardware import Host


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
FIT_MODES = ("on", "off")
POOLING_TYPES = ("", "none", "mean", "cls", "last", "rank")  # "" = model default


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
        choices: Allowed values for CHOICE.
        minimum: Inclusive minimum for INT/FLOAT.
        maximum: Inclusive maximum for INT/FLOAT.
        step: UI stepper increment.
        unset: Value that means "omit the flag" (None = always emitted).
        impact: Optional live impact line computed from profile/model/host.
        preset_key: Long flag name without leading dashes, as used in a router preset INI
            ("" when the setting has no flag).
        router_owned: True when the router overrides the setting (host, port, api_key, alias),
            so gateway presets omit it.
        negate_flag: Flag emitted when a BOOL whose unset value is True is False ("-nocb").
    """

    key: str
    flag: str
    kind: Kind
    label: str
    hint: str
    help: str
    group: Group
    default: Any
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    unset: Any = None
    impact: Impact | None = None
    preset_key: str = ""
    router_owned: bool = False
    negate_flag: str = ""


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
    from .planner import estimate_budget

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
    if setting.key == "override_tensor" and raw:
        # Python's re is close to, not identical with, llama.cpp's C++ regex.
        from .placement import parse_overrides
        parse_overrides(raw)
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
        key="ctx_size", flag="-c", preset_key="ctx-size", kind=Kind.INT, label="Context size", group=Group.MEMORY,
        hint="Max tokens the model remembers (prompt + reply). 0 = model's trained max.",
        help=("Size of the prompt context. Larger context lets the model see more text but the\n"
              "KV cache grows linearly with it, using VRAM (or RAM if layers are not offloaded).\n\n"
              "0 loads the model's trained maximum, which can be very large on modern models.\n"
              "Typical values: 4096 for short chats, 8192-32768 for long documents.\n\n"
              "Flag: -c, --ctx-size N"),
        default=0, unset=0, step=1024, minimum=0, impact=_budget_impact),
    Setting(
        key="gpu_layers", flag="-ngl", preset_key="n-gpu-layers", kind=Kind.STR, label="GPU layers", group=Group.MEMORY,
        hint="Layers kept in VRAM: 'all', 'auto' or a number. Higher = faster, needs more VRAM.",
        help=("Max number of model layers stored in VRAM. More layers on the GPU means faster\n"
              "generation but more VRAM. 'all' offloads everything, 'auto' lets llama.cpp\n"
              "decide, a number offloads exactly that many layers.\n\n"
              "Lower it if you run out of memory; the rest runs on the CPU.\n\n"
              "Flag: -ngl, --gpu-layers N"),
        default="all", unset="", impact=_budget_impact),
    Setting(
        key="override_tensor", flag="-ot", preset_key="override-tensor", kind=Kind.STR,
        label="Tensor overrides", group=Group.MEMORY,
        hint=("Place tensors by regex: PATTERN=DEVICE,... (MoE models). Turns auto-fit off. "
              "Checked with Python's re: close to, not identical with, llama.cpp's C++ regex."),
        help=("Place tensors by name pattern: -ot PATTERN=BUFFER[,PATTERN=BUFFER...]. The\n"
              "pattern is a regex searched in each tensor name (for example\n"
              "blk.12.ffn_up_exps.weight); the buffer is usually CPU or a device name.\n\n"
              "The first matching pattern wins, so these overrides are emitted before\n"
              "--n-cpu-moe and take precedence over it. Examples:\n"
              "  blk\\.(2[0-9]|3[0-9])\\.ffn_.*_exps\\.=CPU   experts of layers 20-39 in RAM\n"
              "  ffn_down_exps=CPU                          only down-projection experts in RAM\n\n"
              "Meant for MoE models. llamactl turns auto-fit off while it is set, and tune\n"
              "refuses presets that set it (the search places experts itself).\n\n"
              "Flag: -ot, --override-tensor <pattern>=<buffer type>,..."),
        default="", unset=""),
    Setting(
        key="n_cpu_moe", flag="--n-cpu-moe", preset_key="n-cpu-moe", kind=Kind.INT, label="CPU MoE layers", group=Group.MEMORY,
        hint="Keep expert weights of the first N layers in RAM. MoE models only. 0 = off.",
        help=("Keeps the Mixture-of-Experts weights of the first N layers on the CPU while the\n"
              "rest runs on the GPU. Only the few experts actually used per token are read, so\n"
              "a big MoE model can run almost at GPU speed with little VRAM.\n\n"
              "Only meaningful for MoE models (expert_count > 0). 0 disables.\n\n"
              "Flag: --n-cpu-moe N"),
        default=0, unset=0, minimum=0, impact=_budget_impact),
    Setting(
        key="fit", flag="-fit", preset_key="fit", kind=Kind.CHOICE, label="Auto-fit", group=Group.MEMORY,
        hint="on / off. Let llama.cpp lower unset settings (layers, context) until the model fits.",
        help=("When on, llama.cpp checks the free memory of each device at load time and lowers\n"
              "arguments you left unset (GPU layers 'auto', context size 0) until the model\n"
              "fits. Values you set explicitly are never changed.\n\n"
              "Turn it off for exact, repeatable placement (for example 'all' layers or a\n"
              "hand-made tensor split); loading then fails instead of adjusting when memory\n"
              "is short. 'on' is the llama.cpp default and is not emitted.\n\n"
              "Flag: -fit, --fit [on|off]"),
        default="on", unset="on", choices=FIT_MODES),
    Setting(
        key="fit_target", flag="-fitt", preset_key="fit-target", kind=Kind.INT, label="Fit margin (MiB)",
        group=Group.MEMORY,
        hint="Free memory in MiB that auto-fit keeps per device. 0 = llama.cpp default (1024).",
        help=("Target margin per device for --fit: how much memory, in MiB, auto-fit leaves free\n"
              "so the desktop, other programs and compute buffers still have room. A single\n"
              "value applies to every device. 0 leaves the flag out and uses the llama.cpp\n"
              "default of 1024 MiB. Only has an effect while auto-fit is on.\n\n"
              "Flag: -fitt, --fit-target MiB0,MiB1,..."),
        default=0, unset=0, step=256, minimum=0),
    Setting(
        key="cache_type_k", flag="-ctk", preset_key="cache-type-k", kind=Kind.CHOICE, label="K cache type", group=Group.MEMORY,
        hint="KV cache data type for K. f16 default; q8_0 halves memory.",
        help=("Data type of the K (key) cache. f16 is the default and lossless.\n"
              "q8_0 roughly halves KV memory with a small quality loss; q4_0 is more\n"
              "aggressive. Quantised K works even without flash attention.\n\n"
              f"Allowed: {', '.join(CACHE_TYPES)}.\n\n"
              "Flag: -ctk, --cache-type-k TYPE"),
        default="f16", unset="f16", choices=CACHE_TYPES, impact=_budget_impact),
    Setting(
        key="cache_type_v", flag="-ctv", preset_key="cache-type-v", kind=Kind.CHOICE, label="V cache type", group=Group.MEMORY,
        hint="KV cache data type for V. Quantised V requires flash_attn != off.",
        help=("Data type of the V (value) cache. Same choices as cache_type_k, but a\n"
              "quantised V cache requires flash attention to be on or auto.\n\n"
              f"Allowed: {', '.join(CACHE_TYPES)}.\n\n"
              "Flag: -ctv, --cache-type-v TYPE"),
        default="f16", unset="f16", choices=CACHE_TYPES, impact=_budget_impact),
    Setting(
        key="flash_attn", flag="-fa", preset_key="flash-attn", kind=Kind.CHOICE, label="Flash attention", group=Group.MEMORY,
        hint="on / off / auto. Faster attention and lower memory on supported GPUs.",
        help=("Enables the flash attention kernel. 'on' is faster and uses less KV memory on\n"
              "supported GPUs and is required for quantised V cache. 'auto' lets llama.cpp\n"
              "enable it when safe. 'off' forces the old code path.\n\n"
              "Flag: -fa, --flash-attn [on|off|auto]"),
        default="auto", choices=FLASH_ATTN_MODES),
    Setting(
        key="batch_size", flag="-b", preset_key="batch-size", kind=Kind.INT, label="Batch size", group=Group.MEMORY,
        hint="Logical max batch size (prompt processing). 0 = llama.cpp default (2048).",
        help=("Logical maximum batch size: how many tokens are processed per step during\n"
              "prompt ingestion. Larger batches speed up long prompts but need a bigger\n"
              "compute buffer. 0 uses the llama.cpp default of 2048.\n\n"
              "Flag: -b, --batch-size N"),
        default=0, unset=0, step=256, minimum=0, impact=_budget_impact),
    Setting(
        key="ubatch_size", flag="-ub", preset_key="ubatch-size", kind=Kind.INT, label="Micro batch", group=Group.MEMORY,
        hint="Physical max batch size, must be <= batch size. 0 = default (512).",
        help=("Physical maximum batch size: the chunk the logical batch is split into for the\n"
              "actual compute. Must be <= batch_size when both are set. Smaller values reduce\n"
              "the compute buffer at the cost of slower prompt processing.\n\n"
              "Flag: -ub, --ubatch-size N"),
        default=0, unset=0, step=128, minimum=0, impact=_budget_impact),
    Setting(
        key="load_mode", flag="--load-mode", preset_key="load-mode", kind=Kind.CHOICE, label="Load mode", group=Group.MEMORY,
        hint="How weights are read: auto / none / mmap / mlock / mmap+mlock / dio.",
        help=("Model loading mode. auto uses mmap unless a device does not support it.\n"
              "mmap memory-maps the file; mlock pins it in RAM so it cannot be swapped;\n"
              "dio uses DirectIO when available. The default is almost always right.\n\n"
              "Flag: --load-mode MODE"),
        default="auto", unset="auto", choices=LOAD_MODES),
    Setting(
        key="tensor_split", flag="-ts", preset_key="tensor-split", kind=Kind.STR, label="Tensor split", group=Group.MEMORY,
        hint="Comma-separated VRAM fractions per GPU, e.g. 3,1. Only useful with >1 GPU.",
        help=("Fraction of the model to offload to each GPU, comma-separated proportions,\n"
              "e.g. '3,1' gives the first GPU three quarters. Only shown when more than\n"
              "one GPU is present. Empty lets llama.cpp split evenly.\n\n"
              "Flag: -ts, --tensor-split N0,N1,..."),
        default="", unset=""),
    Setting(
        key="device", flag="-dev", preset_key="device", kind=Kind.STR, label="Devices", group=Group.MEMORY,
        hint="Comma-separated device ids to offload to, e.g. CUDA0,CUDA1. Empty = all. none = CPU.",
        help=("Restricts offloading to the listed devices, using the ids that\n"
              "'llama-server --list-devices' prints (CUDA0, Vulkan1, ...). Empty uses every\n"
              "device llama.cpp finds. 'none' keeps everything on the CPU.\n\n"
              "Useful to leave a display GPU free or to pick the fast card on a mixed system.\n\n"
              "Flag: -dev, --device <dev1,dev2,..>"),
        default="", unset=""),
    Setting(
        key="threads", flag="-t", preset_key="threads", kind=Kind.INT, label="CPU threads", group=Group.MEMORY,
        hint="Threads for CPU work. Usually the number of physical cores. 0 = auto.",
        help=("Number of CPU threads used during generation. On a CPU-only or partially\n"
              "offloaded model this is the main speed knob: the number of physical cores\n"
              "(not hyperthreads) is usually best. 0 lets llama.cpp choose.\n\n"
              "Flag: -t, --threads N"),
        default=0, unset=0, minimum=0),
    Setting(
        key="threads_batch", flag="-tb", preset_key="threads-batch", kind=Kind.INT, label="Batch threads",
        group=Group.MEMORY,
        hint="Threads for prompt/batch processing. 0 = same as CPU threads.",
        help=("Number of CPU threads used while ingesting the prompt (batch processing).\n"
              "Prompt processing is compute bound, so it can use more threads than generation,\n"
              "which is memory bound. Matters mostly when layers run on the CPU. 0 leaves the\n"
              "flag out and llama.cpp uses the same value as the CPU threads setting.\n\n"
              "Flag: -tb, --threads-batch N"),
        default=0, unset=0, minimum=0),
    Setting(
        key="temp", flag="--temp", preset_key="temp", kind=Kind.FLOAT, label="Temperature", group=Group.SAMPLING,
        hint="Randomness of the output. -1 = llama.cpp default (0.8). Range 0-2.",
        help=("Sampling temperature. Lower is more deterministic, higher more creative.\n"
              "0 is nearly greedy; most models work best around 0.6-1.0.\n"
              "-1 uses the llama.cpp default of 0.8.\n\n"
              "Flag: --temp, --temperature N"),
        default=-1.0, unset=-1, minimum=-1, maximum=2),
    Setting(
        key="top_k", flag="--top-k", preset_key="top-k", kind=Kind.INT, label="Top K", group=Group.SAMPLING,
        hint="Keep the K most likely tokens. -1 = default (40); 0 disables.",
        help=("Top-k sampling: only the K most likely next tokens are considered.\n"
              "Lower values make output more predictable. -1 uses the default of 40;\n"
              "0 disables top-k entirely.\n\n"
              "Flag: --top-k N"),
        default=-1, unset=-1, minimum=-1),
    Setting(
        key="top_p", flag="--top-p", preset_key="top-p", kind=Kind.FLOAT, label="Top P", group=Group.SAMPLING,
        hint="Nucleus sampling: keep tokens up to cumulative probability P. -1 = default (0.95).",
        help=("Top-p (nucleus) sampling: only tokens making up the top P of probability\n"
              "mass are kept. 1.0 disables. -1 uses the llama.cpp default of 0.95.\n\n"
              "Flag: --top-p N"),
        default=-1.0, unset=-1, minimum=-1, maximum=1),
    Setting(
        key="min_p", flag="--min-p", preset_key="min-p", kind=Kind.FLOAT, label="Min P", group=Group.SAMPLING,
        hint="Drop tokens below P * top-token probability. -1 = default (0.05).",
        help=("Min-p sampling: tokens with probability below P times the top token's\n"
              "probability are dropped. A good low-temperature alternative to top-p.\n"
              "0.0 disables; -1 uses the llama.cpp default of 0.05.\n\n"
              "Flag: --min-p N"),
        default=-1.0, unset=-1, minimum=-1, maximum=1),
    Setting(
        key="repeat_penalty", flag="--repeat-penalty", preset_key="repeat-penalty", kind=Kind.FLOAT, label="Repeat penalty", group=Group.SAMPLING,
        hint="Penalise repeated tokens. -1 = default (1.0 = off).",
        help=("Repeat penalty: divides the logits of recently seen tokens, discouraging\n"
              "loops. 1.0 disables; 1.05-1.2 is a typical range. -1 uses the llama.cpp\n"
              "default (1.0, disabled).\n\n"
              "Flag: --repeat-penalty N"),
        default=-1.0, unset=-1, minimum=-1),
    Setting(
        key="seed", flag="-s", preset_key="seed", kind=Kind.INT, label="Seed", group=Group.SAMPLING,
        hint="RNG seed. -1 = random each run.",
        help=("Random seed for sampling. Set a fixed value for reproducible outputs.\n"
              "-1 uses a random seed each run.\n\n"
              "Flag: -s, --seed SEED"),
        default=-1, unset=-1, minimum=-1),
    Setting(
        key="host", flag="--host", preset_key="host", router_owned=True, kind=Kind.STR, label="Host", group=Group.SERVER,
        hint="Listen address. 127.0.0.1 = this machine only; 0.0.0.0 = your network.",
        help=("IP address the server listens on. 127.0.0.1 keeps it private to this\n"
              "machine. 0.0.0.0 exposes it to your local network; combine with an API\n"
              "key if you do that, there is no auth by default.\n\n"
              "Flag: --host HOST"),
        default="127.0.0.1"),
    Setting(
        key="port", flag="--port", preset_key="port", router_owned=True, kind=Kind.INT, label="Port", group=Group.SERVER,
        hint="Listen port. 0 = first free port from 8080, resolved at launch.",
        help=("TCP port the server listens on (default 8080). 0 lets llamactl pick the\n"
              "first free port from 8080 so several models can run side by side.\n\n"
              "Flag: --port PORT"),
        default=0),
    Setting(
        key="api_key", flag="--api-key", preset_key="api-key", router_owned=True, kind=Kind.STR, label="API key", group=Group.SERVER,
        hint="Require this key in the Authorization header. Empty = no auth.",
        help=("API key clients must send as a bearer token. Without it the server accepts\n"
              "any request - set one if you bind to 0.0.0.0. Keys are stored in the\n"
              "config file in plain text.\n\n"
              "Flag: --api-key KEY"),
        default="", unset=""),
    Setting(
        key="parallel", flag="-np", preset_key="parallel", kind=Kind.INT, label="Slots", group=Group.SERVER,
        hint="Requests served at once. Context is split between slots. 0 = auto.",
        help=("Number of server slots: how many requests are decoded in parallel.\n"
              "The context is divided between slots, so more slots means less\n"
              "context each. 0 lets llama.cpp choose.\n\n"
              "Flag: -np, --parallel N"),
        default=0, unset=0, minimum=0, impact=_slots_impact),
    Setting(
        key="cache_reuse", flag="--cache-reuse", preset_key="cache-reuse", kind=Kind.INT,
        label="Cache reuse", group=Group.SERVER,
        hint="Min chunk size reused from the prompt cache via KV shifting. 0 = off. Helps agents.",
        help=("When a new prompt shares chunks of at least this many tokens with the cached\n"
              "one, the server shifts the cached KV entries instead of recomputing them. Agents\n"
              "and tools that resend a long, slightly edited prompt benefit most; 256 is a\n"
              "good start. Needs prompt caching (on by default). 0 disables.\n\n"
              "Flag: --cache-reuse N"),
        default=0, unset=0, step=64, minimum=0),
    Setting(
        key="cont_batching", flag="-cb", preset_key="cont-batching", kind=Kind.BOOL, label="Continuous batching", group=Group.SERVER,
        hint="Let slots pick up new requests while others are still decoding. On is right.",
        help=("Continuous batching: finished slots start serving the next request\n"
              "immediately instead of waiting for the whole batch. Keep it on unless a\n"
              "model misbehaves; disabling emits -nocb.\n\n"
              "Flag: -cb, --cont-batching / -nocb, --no-cont-batching"),
        default=True, unset=True, negate_flag="-nocb"),
    Setting(
        key="metrics", flag="--metrics", preset_key="metrics", kind=Kind.BOOL, label="Metrics endpoint", group=Group.SERVER,
        hint="Expose a Prometheus-compatible /metrics endpoint.",
        help=("Enables the Prometheus-compatible /metrics endpoint with per-slot usage\n"
              "and performance counters. Off by default.\n\n"
              "Flag: --metrics"),
        default=False, unset=False),
    Setting(
        key="embedding", flag="--embedding", preset_key="embedding", kind=Kind.BOOL,
        label="Embeddings only", group=Group.SERVER,
        hint="Serve embeddings only. Use with dedicated embedding models.",
        help=("Restricts the server to the embedding use case: /v1/embeddings works, text\n"
              "generation does not. Use it only with dedicated embedding models, which\n"
              "llamactl detects by their pooling type. Off by default.\n\n"
              "Flag: --embedding"),
        default=False, unset=False),
    Setting(
        key="pooling", flag="--pooling", preset_key="pooling", kind=Kind.CHOICE, label="Pooling",
        group=Group.SERVER,
        hint="How token embeddings become one vector: none / mean / cls / last / rank. Empty = model.",
        help=("Pooling type for embeddings. Leave it empty to use the model's own setting,\n"
              "which is right for nearly every embedding model. 'rank' is for reranker models,\n"
              "'none' returns one vector per token.\n\n"
              "Allowed: none, mean, cls, last, rank (empty = model default).\n\n"
              "Flag: --pooling {none,mean,cls,last,rank}"),
        default="", unset="", choices=POOLING_TYPES),
    Setting(
        key="sleep_idle", flag="--sleep-idle-seconds", preset_key="sleep-idle-seconds", kind=Kind.INT,
        label="Sleep when idle (s)", group=Group.SERVER,
        hint="Unload the model after N idle seconds; the next request reloads it. 0 = never.",
        help=("Number of seconds without a request after which the server unloads the model to\n"
              "free VRAM and RAM. The next request loads it again, which costs a few seconds\n"
              "to minutes depending on the model. 0 leaves the flag out (llama.cpp default:\n"
              "never sleep).\n\n"
              "Flag: --sleep-idle-seconds SECONDS"),
        default=0, unset=0, step=60, minimum=0),
    Setting(
        key="alias", flag="--alias", preset_key="alias", router_owned=True, kind=Kind.STR, label="Alias", group=Group.SERVER,
        hint="Model name reported by the API instead of the file name.",
        help=("Model name exposed through the API (e.g. in /v1/models) instead of the\n"
              "GGUF file name. Useful when a client expects a specific model id.\n\n"
              "Flag: --alias STRING"),
        default="", unset=""),
    Setting(
        key="draft_model", flag="-md", preset_key="spec-draft-model", kind=Kind.PATH, label="Draft model", group=Group.SERVER,
        hint="Small draft model for speculative decoding. Empty = off.",
        help=("Path to a small draft model used for speculative decoding with the same\n"
              "vocabulary: the draft proposes tokens, the big model verifies them, which\n"
              "can nearly double generation speed when it works.\n\n"
              "Flag: -md, --spec-draft-model FNAME"),
        default="", unset=""),
    Setting(
        key="mmproj", flag="--mmproj", preset_key="mmproj", kind=Kind.PATH, label="mmproj", group=Group.SERVER,
        hint="Vision projector file for multimodal models.",
        help=("Path to a multimodal projector (mmproj) GGUF that pairs with a vision\n"
              "model so it can accept images. Only needed for vision-capable models.\n\n"
              "Flag: --mmproj FILE"),
        default="", unset=""),
    Setting(
        key="extra_args", flag="", preset_key="", kind=Kind.STR, label="Extra args", group=Group.GENERAL,
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
