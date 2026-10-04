"""Config, model discovery and command building for llamactl."""
from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "llamactl"
CONFIG_FILE = CONFIG_DIR / "config.json"
LOG_DIR = CONFIG_DIR / "logs"
DEFAULT_BIN_DIR = Path.home() / "llama.cpp" / "build" / "bin"
DEFAULT_MODEL_DIRS = [str(Path.home() / "models")]
MIN_MODEL_BYTES = 50 * 1024 * 1024


@dataclass
class Profile:
    """Launch settings for one model. Empty/zero values mean 'use llama.cpp default'."""

    ctx_size: int = 0
    gpu_layers: int = 99
    threads: int = 0
    parallel: int = 0
    flash_attn: str = "auto"
    port: int = 0
    host: str = "127.0.0.1"
    extra_args: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "Profile":
        """Build a profile, ignoring unknown keys."""
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in known})


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
        """Write config to disk."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(asdict(self), indent=2))

    def profile_for(self, model: str) -> Profile:
        """Return the model's override profile, or the default one."""
        return self.profiles.get(model, self.default_profile)


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


def build_command(cfg: Config, model: str, prof: Profile, mode: str, port: int) -> list[str]:
    """Build the llama-server or llama-cli argv for a model + profile."""
    cmd = [str(Path(cfg.bin_dir) / f"llama-{mode}"), "-m", model, "-ngl", str(prof.gpu_layers), "-fa", prof.flash_attn]
    if prof.ctx_size:
        cmd += ["-c", str(prof.ctx_size)]
    if prof.threads:
        cmd += ["-t", str(prof.threads)]
    if mode == "server":
        cmd += ["--host", prof.host, "--port", str(port)]
        if prof.parallel:
            cmd += ["-np", str(prof.parallel)]
    return cmd + shlex.split(prof.extra_args)
