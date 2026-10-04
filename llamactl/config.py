"""Config, profiles and model discovery for llamactl."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from . import schema

CONFIG_DIR = Path.home() / ".config" / "llamactl"
CONFIG_FILE = CONFIG_DIR / "config.json"
LOG_DIR = CONFIG_DIR / "logs"
DEFAULT_BIN_DIR = Path.home() / "llama.cpp" / "build" / "bin"
DEFAULT_MODEL_DIRS = [str(Path.home() / "models")]
MIN_MODEL_BYTES = 50 * 1024 * 1024
CONFIG_VERSION = 2


def _coerce(key: str, value: Any) -> Any:
    """Coerce a config value to the type its schema setting expects.

    Args:
        key: Setting key.
        value: Raw value from JSON.

    Returns:
        The coerced value; legacy int gpu_layers (99 = all) is mapped to str.
    """
    s = schema.by_key[key]
    if key == "gpu_layers" and isinstance(value, int):
        return "all" if value == 99 else str(value)
    try:
        if s.kind == schema.Kind.INT:
            return int(value)
        if s.kind == schema.Kind.FLOAT:
            return float(value)
        if s.kind == schema.Kind.BOOL:
            return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "on", "yes")
        return str(value)
    except (TypeError, ValueError):
        return s.default


@dataclass
class Profile:
    """Launch settings for one model. Field for every key in schema.SETTINGS.

    Unset values (0, -1, "", "auto"/"f16" defaults) mean "let llama.cpp decide".
    """

    ctx_size: int = 0
    gpu_layers: str = "all"
    n_cpu_moe: int = 0
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    flash_attn: str = "auto"
    batch_size: int = 0
    ubatch_size: int = 0
    load_mode: str = "auto"
    tensor_split: str = ""
    threads: int = 0
    temp: float = -1.0
    top_k: int = -1
    top_p: float = -1.0
    min_p: float = -1.0
    repeat_penalty: float = -1.0
    seed: int = -1
    host: str = "127.0.0.1"
    port: int = 0
    api_key: str = ""
    parallel: int = 0
    cont_batching: bool = True
    metrics: bool = False
    alias: str = ""
    draft_model: str = ""
    mmproj: str = ""
    extra_args: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "Profile":
        """Build a profile, ignoring unknown keys and coercing types.

        Args:
            data: Raw dict from the config file.

        Returns:
            A Profile; legacy int gpu_layers (99) becomes "all".
        """
        return cls(**{k: _coerce(k, v) for k, v in data.items() if k in schema.by_key})

    def to_dict(self) -> dict:
        """Serialise to a JSON-able dict."""
        return asdict(self)

    def is_set(self, key: str) -> bool:
        """True when the field would emit a flag (value != the schema unset value).

        Args:
            key: Setting key.
        """
        return getattr(self, key) != schema.by_key[key].unset

    def merged(self, **overrides: Any) -> "Profile":
        """Copy with the given fields replaced."""
        return replace(self, **overrides)


@dataclass
class Config:
    """Global settings plus per-model profile overrides keyed by model path."""

    bin_dir: str = str(DEFAULT_BIN_DIR)
    model_dirs: list[str] = field(default_factory=lambda: list(DEFAULT_MODEL_DIRS))
    default_profile: Profile = field(default_factory=Profile)
    profiles: dict[str, Profile] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Config":
        """Load config from disk, falling back to defaults."""
        if not CONFIG_FILE.exists():
            return cls()
        raw = json.loads(CONFIG_FILE.read_text())
        return cls(
            bin_dir=raw.get("bin_dir", str(DEFAULT_BIN_DIR)),
            model_dirs=raw.get("model_dirs", list(DEFAULT_MODEL_DIRS)),
            default_profile=Profile.from_dict(raw.get("default_profile", {})),
            profiles={k: Profile.from_dict(v) for k, v in raw.get("profiles", {}).items()},
        )

    def save(self) -> None:
        """Write config to disk with the current format version."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps({"version": CONFIG_VERSION, **asdict(self)}, indent=2))

    def profile_for(self, model: str) -> Profile:
        """Return the model's override profile, or the default one."""
        return self.profiles.get(model, self.default_profile)

    def has_profile(self, model: str) -> bool:
        """True when the model has a custom profile (drives the New badge)."""
        return model in self.profiles


def discover_models(dirs: list[str]) -> list[Path]:
    """Find GGUF files in the given dirs, skipping vocab stubs and non-first shards."""
    found: dict[Path, None] = {}
    for d in dirs:
        root = Path(d).expanduser()
        if not root.is_dir():
            continue
        for p in root.rglob("*.gguf"):
            if p.stat().st_size < MIN_MODEL_BYTES or "mmproj" in p.name:
                continue
            if "-of-" in p.stem and "-00001-of-" not in p.stem:
                continue
            found[p] = None
    return sorted(found)
