"""Config v3: settings profiles, per-model presets, gateway settings and v2 migration."""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping

from . import schema
from .measure import Measurement

STATE_ENV = "LLAMACTL_HOME"


def state_dir(env: Mapping[str, str] | None = None, home: Path | None = None) -> Path:
    """The one place that decides where llamactl keeps its state.

    ``LLAMACTL_HOME`` wins, then ``$XDG_CONFIG_HOME/llamactl``, then ``~/.config/llamactl``.
    Empty variables count as unset; a leading ``~`` is expanded.

    Args:
        env: Environment to read (default: ``os.environ``).
        home: Home directory for the last fallback (default: the user's).

    Returns:
        The directory (it may not exist yet).
    """
    env = os.environ if env is None else env
    if explicit := env.get(STATE_ENV, "").strip():
        return Path(explicit).expanduser()
    if xdg := env.get("XDG_CONFIG_HOME", "").strip():
        return Path(xdg).expanduser() / "llamactl"
    return (home or Path.home()) / ".config" / "llamactl"


CONFIG_DIR = state_dir()
CONFIG_FILE = CONFIG_DIR / "config.json"
LOG_DIR = CONFIG_DIR / "logs"
DEFAULT_BIN_DIR = Path.home() / "llama.cpp" / "build" / "bin"
DEFAULT_MODEL_DIRS = [str(Path.home() / "models")]
CONFIG_VERSION = 3
BACKUP_NAME = "config.v2.bak.json"
CUSTOM = "custom"        # preset name/intent used for hand-edited settings
ESTIMATED = "estimated"  # Preset.source of a preset that came from the planner, unmeasured
PRESET_SEP = "@"         # joins alias and preset name: "qwen3-8b@agent" (":" is rewritten by llama.cpp)
SHARD_SUFFIX_RE = re.compile(r"-\d{5}-of-\d{5}$")


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
    fit: str = "on"
    fit_target: int = 0
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    flash_attn: str = "auto"
    batch_size: int = 0
    ubatch_size: int = 0
    load_mode: str = "auto"
    tensor_split: str = ""
    device: str = ""
    threads: int = 0
    threads_batch: int = 0
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
    cache_reuse: int = 0
    cont_batching: bool = True
    metrics: bool = False
    embedding: bool = False
    pooling: str = ""
    sleep_idle: int = 0
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

    def non_default(self) -> dict:
        """Fields differing from the schema default, for display summaries."""
        return {k: getattr(self, k) for k in schema.by_key
                if getattr(self, k) != schema.by_key[k].default}

    def merged(self, **overrides: Any) -> "Profile":
        """Copy with the given fields replaced."""
        return replace(self, **overrides)


def _parse_bool(value: Any, default: bool) -> bool:
    """Parse a JSON bool that may have been written as text.

    Args:
        value: Raw JSON value.
        default: Returned when `value` is not recognisable.

    Returns:
        The boolean.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("1", "true", "on", "yes", "0", "false", "off", "no"):
        return value.lower() in ("1", "true", "on", "yes")
    return default


def derive_alias(model: str | Path, taken: set[str]) -> str:
    """Derive a unique, URL-safe alias from a model file name.

    The alias is the lower-cased file stem (a split-model "-00001-of-00003" suffix removed)
    with every character outside ``[a-z0-9._-]`` replaced by "-". When it is already in
    `taken`, a numeric suffix is appended: "name-2", "name-3", ...

    Args:
        model: Path of the GGUF file.
        taken: Aliases already in use.

    Returns:
        An alias not in `taken`.
    """
    stem = SHARD_SUFFIX_RE.sub("", Path(model).stem.lower())
    base = re.sub(r"[^a-z0-9._-]", "-", stem) or "model"
    alias, n = base, 1
    while alias in taken:
        n += 1
        alias = f"{base}-{n}"
    return alias


@dataclass
class Preset:
    """One named set of launch settings for a model.

    Attributes:
        intent: An intent key, or "custom".
        profile: The concrete settings.
        source: "estimated", "tuned" or "edited".
        measured: Measurement of the tuned run, or None.
    """

    intent: str = CUSTOM
    profile: Profile = field(default_factory=Profile)
    source: str = "edited"
    measured: Measurement | None = None

    @classmethod
    def from_dict(cls, data: dict, name: str = CUSTOM) -> "Preset":
        """Build a preset from its JSON form, ignoring unknown keys.

        Args:
            data: Raw dict from the config file.
            name: Preset name, used as the intent when the dict has none.

        Returns:
            The preset.
        """
        raw = data.get("measured")
        measured = Measurement.from_dict(raw) if isinstance(raw, dict) else None
        return cls(intent=str(data.get("intent") or name),
                   profile=Profile.from_dict(data.get("profile") or {}),
                   source=str(data.get("source") or "edited"), measured=measured)

    def to_dict(self) -> dict:
        """Serialise to a JSON-able dict."""
        return {"intent": self.intent, "profile": self.profile.to_dict(), "source": self.source,
                "measured": self.measured.to_dict() if self.measured else None}


@dataclass
class ModelEntry:
    """Everything the config knows about one model file.

    Attributes:
        alias: Name clients use for the model; unique across models, editable.
        active: Name of the preset served by default ("" when there is none).
        presets: Presets by name.
    """

    alias: str = ""
    active: str = ""
    presets: dict[str, Preset] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelEntry":
        """Build an entry from its JSON form, ignoring unknown keys.

        Args:
            data: Raw dict from the config file.

        Returns:
            The entry.
        """
        raw = data.get("presets")
        presets = {str(k): Preset.from_dict(v, str(k)) for k, v in raw.items()
                   if isinstance(v, dict)} if isinstance(raw, dict) else {}
        return cls(alias=str(data.get("alias") or ""), active=str(data.get("active") or ""),
                   presets=presets)

    def to_dict(self) -> dict:
        """Serialise to a JSON-able dict."""
        return {"alias": self.alias, "active": self.active,
                "presets": {k: v.to_dict() for k, v in self.presets.items()}}


@dataclass
class GatewaySettings:
    """Settings of the router-mode gateway server.

    Attributes:
        host: Listen address.
        port: Listen port.
        api_key: Bearer key clients must send; "" for none.
        models_max: Models kept loaded at once.
        sleep_idle: Idle seconds before an unused model unloads; 0 = never.
        autostart: Start the gateway when the TUI opens.
    """

    host: str = "127.0.0.1"
    port: int = 8080
    api_key: str = ""
    models_max: int = 1
    sleep_idle: int = 600
    autostart: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "GatewaySettings":
        """Build settings from JSON, ignoring unknown keys and falling back on bad values.

        Args:
            data: Raw dict from the config file.

        Returns:
            The settings.
        """
        base = cls()
        out: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            want, v = type(getattr(base, f.name)), data[f.name]
            try:
                out[f.name] = (_parse_bool(v, getattr(base, f.name)) if want is bool
                               else want(v))
            except (TypeError, ValueError):
                pass
        return cls(**out)

    def to_dict(self) -> dict:
        """Serialise to a JSON-able dict."""
        return asdict(self)


@dataclass
class Config:
    """Global settings, the default profile and per-model entries keyed by model path."""

    bin_dir: str = str(DEFAULT_BIN_DIR)
    model_dirs: list[str] = field(default_factory=lambda: list(DEFAULT_MODEL_DIRS))
    default_profile: Profile = field(default_factory=Profile)
    models: dict[str, ModelEntry] = field(default_factory=dict)
    gateway: GatewaySettings = field(default_factory=GatewaySettings)
    default_intent: str = "chat"
    gauge_interval: float = 3.0
    headroom_pct: int = 10
    onboarded: bool = False

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        """Build a config from parsed JSON of any version, migrating v2 on the way.

        A v2 (or unversioned) file carries ``profiles[path]``; each becomes the model's
        "custom" preset (source "edited") and the active one. Such a file counts as onboarded
        (in memory only) when it has model presets or a non-default ``bin_dir`` /
        ``model_dirs``. Unknown keys are ignored.

        Args:
            raw: Parsed config.json.

        Returns:
            The config, in v3 shape.
        """
        version = raw.get("version", 0)
        cfg = cls(
            bin_dir=raw.get("bin_dir", str(DEFAULT_BIN_DIR)),
            model_dirs=list(raw.get("model_dirs", DEFAULT_MODEL_DIRS)),
            default_profile=Profile.from_dict(raw.get("default_profile", {})),
            gateway=GatewaySettings.from_dict(raw.get("gateway") or {}),
            default_intent=str(raw.get("default_intent", "chat")),
            gauge_interval=float(raw.get("gauge_interval", 3.0)),
            headroom_pct=int(raw.get("headroom_pct", 10)),
            onboarded=_parse_bool(raw.get("onboarded"), False),
        )
        if isinstance(version, int) and version >= 3:
            for path, data in (raw.get("models") or {}).items():
                if isinstance(data, dict):
                    cfg.models[path] = ModelEntry.from_dict(data)
        else:
            for path, data in (raw.get("profiles") or {}).items():
                cfg.models[path] = ModelEntry(
                    active=CUSTOM,
                    presets={CUSTOM: Preset(CUSTOM, Profile.from_dict(data), "edited")})
        for path, ent in cfg.models.items():
            if not ent.alias:
                ent.alias = derive_alias(path, {e.alias for e in cfg.models.values()})
        if not (isinstance(version, int) and version >= 3) and not cfg.onboarded:
            # A v2 user already set llamactl up: do not walk them through first-run setup.
            # Only in memory; the file is rewritten (as v3) when something else saves it.
            cfg.onboarded = bool(cfg.models) or cfg.bin_dir != str(DEFAULT_BIN_DIR) or \
                cfg.model_dirs != list(DEFAULT_MODEL_DIRS)
        return cfg

    @classmethod
    def load(cls) -> "Config":
        """Load config from disk, falling back to defaults."""
        if not CONFIG_FILE.exists():
            return cls()
        return cls.from_dict(json.loads(CONFIG_FILE.read_text()))

    def to_dict(self) -> dict:
        """Serialise to the v3 JSON shape."""
        return {
            "version": CONFIG_VERSION,
            "bin_dir": self.bin_dir,
            "model_dirs": self.model_dirs,
            "headroom_pct": self.headroom_pct,
            "gauge_interval": self.gauge_interval,
            "default_profile": self.default_profile.to_dict(),
            "default_intent": self.default_intent,
            "models": {k: v.to_dict() for k, v in self.models.items()},
            "gateway": self.gateway.to_dict(),
            "onboarded": self.onboarded,
        }

    @staticmethod
    def _backup_v2() -> None:
        """Copy a pre-v3 config file to config.v2.bak.json, once.

        Does nothing when the file is missing, already v3 or newer, or a backup exists.
        """
        backup = CONFIG_FILE.with_name(BACKUP_NAME)
        if not CONFIG_FILE.exists() or backup.exists():
            return
        try:
            version = json.loads(CONFIG_FILE.read_text()).get("version", 0)
        except (ValueError, AttributeError):
            version = 0
        if not isinstance(version, int) or version < CONFIG_VERSION:
            shutil.copy2(CONFIG_FILE, backup)

    def save(self) -> None:
        """Write config to disk as v3, backing up a v2 file first (once)."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self._backup_v2()
        CONFIG_FILE.write_text(json.dumps(self.to_dict(), indent=2))

    def entry(self, model: str) -> ModelEntry:
        """The model's entry, created (with a unique derived alias) when missing.

        Args:
            model: Model path.

        Returns:
            The entry; a new one is stored in `models` but not saved to disk.
        """
        ent = self.models.get(model)
        if ent is None:
            ent = self.models[model] = ModelEntry(
                alias=derive_alias(model, {e.alias for e in self.models.values()}))
        return ent

    def preset(self, model: str, name: str | None = None) -> Preset | None:
        """A preset of the model.

        Args:
            model: Model path.
            name: Preset name; None selects the active preset.

        Returns:
            The preset, or None when the model or preset does not exist.
        """
        ent = self.models.get(model)
        if ent is None:
            return None
        return ent.presets.get(ent.active if name is None else name)

    def set_preset(self, model: str, preset: Preset, activate: bool = True) -> None:
        """Store a preset under its intent name, replacing any with that name.

        Args:
            model: Model path.
            preset: The preset; its `intent` is the name it is stored under.
            activate: Make it the model's active preset.
        """
        ent = self.entry(model)
        ent.presets[preset.intent] = preset
        if activate or not ent.active:
            ent.active = preset.intent

    def delete_preset(self, model: str, name: str) -> bool:
        """Remove a preset; when it was active, the first remaining one becomes active.

        Args:
            model: Model path.
            name: Preset name.

        Returns:
            True when a preset was removed.
        """
        ent = self.models.get(model)
        if ent is None or ent.presets.pop(name, None) is None:
            return False
        if ent.active == name:
            ent.active = next(iter(ent.presets), "")
        return True

    def profile_for(self, model: str) -> Profile:
        """The active preset's profile, or the default profile when there is none."""
        p = self.preset(model)
        return p.profile if p else self.default_profile

    def has_profile(self, model: str) -> bool:
        """True when the model has any preset (a model without one is "new")."""
        ent = self.models.get(model)
        return bool(ent and ent.presets)
