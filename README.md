# llamactl

A terminal UI for launching, monitoring and chatting with local llama.cpp
servers. It scans your model directories, parses GGUF headers, estimates memory
budgets, recommends launch profiles and manages running servers.

## Requirements

- Python 3.11+
- A llama.cpp build (llama-server / llama-cli on disk)
- NVIDIA GPU optional (CPU-only hosts work; recommendations adapt)

## Install and run

Install the published package in an isolated environment with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install llama-tui
llamactl
```

Or use pip in a virtual environment: `python -m pip install llama-tui`.
llama.cpp binaries and model files are not bundled; configure their paths in Settings.
The PyPI distribution is named `llama-tui`; the command and Python module remain
`llamactl`. The published installation commands become available after the first
PyPI release.

### Develop from source

```bash
uv sync --extra dev --locked
uv run --locked python -m llamactl
uv run --locked pytest -q
uv build
```

Version: **0.1.3** (development release). See [CHANGELOG.md](CHANGELOG.md)
and [release instructions](docs/RELEASING.md) for Semantic Versioning and PyPI setup.

## Keyboard cheat-sheet

Global: `1-4` switch view, `r` rescan models, `q` quit. Enter on the sidebar
also activates a view.

### Models

| Key | Action |
|-----|--------|
| Enter / l | Launch llama-server |
| c | Chat in llama-cli |
| e | Edit profile |
| d | Edit default profile |
| F2 / R | Recommended profile |
| Delete | Delete the model's profile |

### Sessions

| Key | Action |
|-----|--------|
| t | Chat (llama-cli --url) |
| s | Stop / forget |
| a | Attach a running server by URL |
| o | Open the full log |
| y | Copy URL |

### Hugging Face

| Key | Action |
|-----|--------|
| / | Focus search |
| Enter | Search / list files / download |
| k | Set access token |
| x | Cancel download |

### Profile editor

`ctrl+s` save, `Esc` cancel, `F1` help for the focused field,
`F2` apply recommended, `ctrl+r` reset field, `+`/`-` step values.

## How recommendations work

llamactl parses the GGUF header (architecture, layers, head dims, trained ctx,
MoE expert count, weights size) and builds a budget:

```
budget = weights + KV + compute + overhead
KV     = kv_per_token(ctx_type) * ctx_size      # shared across all slots
```

It keeps `headroom_pct`% of total VRAM free (min 512 MiB), picks the largest
ctx tier that fits, upgrades to q8_0 KV cache when it unlocks a bigger ctx,
offloads MoE experts or dense layers as needed, and reports a verdict:
GPU / GPU+CPU / CPU / NO. See docs/SPEC.md section 7 for the full algorithm.

## Files

All state lives in `~/.config/llamactl/`:

- `config.json` - settings, default profile, per-model profiles
- `sessions.json` - managed and attached sessions
- `gguf_cache.json` - parsed GGUF headers keyed by path+mtime
- `seen.json` - models already offered a recommendation
- `hf_token` - Hugging Face access token
- `logs/` - per-session llama-server logs

Design spec: [docs/SPEC.md](docs/SPEC.md)
