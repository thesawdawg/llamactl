# llamactl

A server manager for [llama.cpp](https://github.com/ggml-org/llama.cpp), with a terminal UI
and a headless command line over the same core. You pick what a model is for, llamactl works
out settings for your hardware, and the model is served from one OpenAI-compatible endpoint
that chat frontends and agent harnesses connect to.

llamactl does not include a chat client. It starts and manages servers; use any client that
speaks the OpenAI API to talk to them.

How it works:

1. **Plan.** You choose an intent (chat, agent, context, speed, api, embed). The planner
   estimates context size, KV cache type, GPU offload and threads from the model's GGUF header
   and the devices llama.cpp reports (`llama-server --list-devices`, so any backend it
   supports). Each setting comes with a one-line reason.
2. **Tune (optional).** Guided tuning tests candidates against goals saved for that model and
   purpose. It compares the best candidate with the saved preset before offering to save it;
   saved settings apply on the next load. Measurements also correct later estimates on this
   machine.
3. **Serve.** Models are served by one `llama-server` in router mode (the gateway). Each model
   loads on demand. Standalone servers on their own port are also available.

## Requirements

- Python 3.11 or newer (CI covers 3.11 to 3.14; development uses 3.14).
- A llama.cpp build whose `llama-server` supports router mode (`--models-preset`,
  `--models-max`) and `--list-devices`. llamactl is checked against 0.5.0-dev (commit
  `5fc4f3c`). llamactl does not download llama.cpp; `llamactl update` pulls and rebuilds
  a checkout you already have.
- Linux is the main target. macOS is best effort.

## Install

The PyPI package is named `llama-tui`; the command and the Python module are `llamactl`.

```bash
uv tool install llama-tui
llamactl
```

Or with pip in a virtual environment: `python -m pip install llama-tui`.

To run from a source checkout:

```bash
uv sync --extra dev
.venv/bin/python -m llamactl            # the TUI
.venv/bin/python -m llamactl ls         # any subcommand
.venv/bin/pytest -q                     # tests
```

## First run

`llamactl` with no arguments opens the TUI. On the first run a five-step setup opens:

1. **llama.cpp.** It looks in the configured folder (default `~/llama.cpp/build/bin`), then
   for `llama-server` on `PATH`, and shows the build. You can enter another folder.
2. **Hardware.** Devices, RAM, CPU, and a sentence on the largest model file that fits.
3. **Models.** The folders to scan for `.gguf` files (default `~/models`). With no models
   found, "Find a model" opens Discover.
4. **Purpose.** The intent you will use most (it becomes the default) and, optionally, a first
   model.
5. **Finish.** Optionally tune the model now, start the gateway, and make it reachable from
   other machines (off by default; see [Serving on your network](#serving-on-your-network)).
   The endpoint URL, model name and a `curl` command are shown at the end.

Every step can be skipped. "Run setup again" in Settings repeats it.

## Intents

An intent states what the model is for. Context sizes are per slot.

| Intent | For | Context (min / target / max) | Slots | Notes |
|--------|-----|------------------------------|-------|-------|
| `chat` | interactive chat frontends | 8k / 16k / 32k | 1 | q8_0 KV cache only if needed; stays on the GPU |
| `agent` | coding agents and tool harnesses | 32k / 64k / 128k | 1 | prefers q8_0 KV cache; may spill to CPU; prompt cache reuse (256) |
| `context` | the largest usable window | 32k / trained / trained | 1 | prefers q8_0 KV cache; may spill to CPU |
| `speed` | highest tokens per second | 4k / 8k / 16k | 1 | f16 KV cache only; stays on the GPU |
| `api` | several concurrent clients | 4k / 8k / 16k | 4 | q8_0 KV cache only if needed; stays on the GPU; metrics on |
| `embed` | embedding models | trained | 4 | embeddings-only server; f16 KV cache only |

`embed` applies only to models whose GGUF declares a pooling type; the other intents apply
only to models that do not. A model can hold several presets (one per intent, plus edited
ones); one is active.

## Guided tuning

Each model/purpose preset can have its own context ceiling and throughput goals. The default
minimum is 20 effective output tokens/s; optional input-processing and output-generation
floors can be set as well. Effective output speed is measured per turn: the generated tokens
over the time a real turn takes — its new input processing plus generation, with the
conversation prefix served from the KV cache like a real client's. A warmup request runs
before measuring, and a workload whose result lands near a floor is sampled three times; the
median counts.

Tuning searches for the largest context that meets the enabled goals. It increases context;
a failing trial bounds the search — memory failures first try more modest memory layouts at
the same context, then the context is bisected against the largest proven pass until the gap
is within ~5%. Placement at each probed context uses the VRAM the earlier trials actually
measured. Every required workload must pass; if the trial budget ends first, the result is
reported as the best tested candidate, not a proven maximum. Benchmark levels are:

- **Quick:** a ~2k cold turn.
- **Normal:** Quick plus a turn at ~25k occupancy: a cached ~23k prefix plus ~2k of new
  input. If the context is too small, the reduced workload is labelled in the results.
- **Full:** 5%, 25%, 50%, 75% and 95% context occupancy, with the final 5% reserved for
  harness compaction. An occupancy above ~2k is measured as a cached prefix plus a ~2k
  turn delta.

Results show the saved preset beside the candidate. Saving queues the candidate for the next
load. A running model keeps its current launch settings and is not interrupted; the UI reports
the current running snapshot separately from saved changes pending the next load. The gateway
polls router state to detect when a model has stopped before applying pending settings. An
external on-demand load racing that poll can still use the old definition.

## Command line

Models are named by alias, file stem or unique substring. The alias is the lower-cased file
stem. Commands that print data accept `--json`. Exit codes: 0 success, 1 runtime failure,
2 usage error or unknown/ambiguous model.

| Command | Does |
|---------|------|
| `llamactl hw` | devices, RAM, CPU, calibration and the llama.cpp build |
| `llamactl ls` | models with the plan per intent and their presets |
| `llamactl plan MODEL [--for INTENT] [--all] [--save [--force]]` | show the plan and reasons; changes nothing unless `--save` |
| `llamactl tune MODEL [--for INTENT] [--trials N] [--force] [--min-effective-tps TPS] [--min-prompt-tps TPS] [--min-gen-tps TPS] [--max-context TOKENS] [--benchmark {quick,normal,full}] [--yes]` | measure and compare a candidate; save it after acceptance |
| `llamactl serve MODEL [--for INTENT] [--standalone [--port P]] [--lan \| --host ADDR] [--no-wait] [--no-activate]` | create a preset if missing, then serve it |
| `llamactl up [--lan \| --host ADDR]` / `llamactl down` | start / stop the gateway |
| `llamactl config [KEY=VALUE ...] [--new-api-key]` | show or change the gateway and planning settings |
| `llamactl ps` | gateway models and standalone servers with state, CPU, RAM and VRAM |
| `llamactl load NAME` / `llamactl unload NAME` | load / unload a model in the gateway |
| `llamactl stop [NAME \| --all]` | stop standalone servers (never the gateway) |
| `llamactl logs NAME [-f] [-n LINES]` | print or follow a log; `NAME` is `gateway` or a server |
| `llamactl update [--check] [-y] [--only {llamactl,llama.cpp}]` | report, then pull and rebuild llamactl and llama.cpp |

Examples:

```bash
llamactl hw                           # what llama.cpp can offload to
llamactl ls                           # what fits, per intent
llamactl plan qwen3-8b --for agent    # settings and the reason for each
llamactl plan qwen3-8b --all          # compare every applicable intent
llamactl tune qwen3-8b --for agent    # measure and compare a candidate
llamactl tune qwen3-8b --for agent --max-context 65536 --benchmark normal
llamactl tune qwen3-8b --for agent --yes  # accept the candidate for the next load
llamactl serve qwen3-8b --for agent   # start the gateway if needed, load the model
llamactl serve qwen3-8b --for speed --no-activate   # also served as qwen3-8b@speed
llamactl serve qwen3-8b --standalone --port 8090    # own server, not the gateway
llamactl up --lan                     # gateway on the local network, this start only
llamactl config gateway.host=lan --new-api-key      # make that permanent, with a key
llamactl ps
llamactl logs gateway -f
llamactl down
```

`hw`, `ls`, `plan` (without `--save`), `ps`, `logs` and `config` without arguments never write
state. `ps` samples CPU over half a second and shows each server's RAM and VRAM; a `~` marks
memory the GPU driver does not report per process (a tuned measurement or an estimate instead).
`tune` runs up to 32 trials by default, prints one line per trial to stderr, and compares the
candidate with the saved preset before saving. In a noninteractive run it does not save unless
`--yes` is supplied. Ctrl-C stops the running trial and keeps the trials so far.

## Connecting a client

The gateway listens on `127.0.0.1:8080` by default. Change host, port and the API key in
Settings.

- **Base URL:** `http://127.0.0.1:8080/v1`
- **Model name:** the alias (`qwen3-8b`) for a model's active preset, or `alias@intent`
  (`qwen3-8b@agent`) for another preset. `llamactl ps` lists the names.
- **API key:** none by default. If one is set in Settings, send `Authorization: Bearer KEY`.
- A model that is not loaded is loaded when a request names it. With the default
  `models_max` of 1, loading a second model unloads the first. Idle models go to sleep (and
  free their memory) after 600 seconds; Settings changes this.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -d '{"model": "qwen3-8b", "messages": [{"role": "user", "content": "Hello"}]}'
```

Leave out the `Authorization` header when no key is set. For an `embed` preset, call
`/v1/embeddings` with `{"model": ..., "input": ...}`. Home and Serving show a ready-made
`curl` command (`c` copies it).

A standalone server (`serve --standalone`) has its own URL, shown by `llamactl ps`, and
serves one model.

## Serving on your network

Everything listens on `127.0.0.1` (this machine only) until you change it. To serve chat
frontends or agents on other machines, the gateway has to listen on the network:

```bash
llamactl config gateway.host=lan --new-api-key   # save "listen on every interface" and a key
llamactl up                                      # (or: llamactl up --lan, for this start only)
```

`gateway.host` takes `local` (`127.0.0.1`), `lan` (`0.0.0.0`) or any other address. In the TUI
it is the "Listen on" choice in Settings (This machine, Local network, Custom address); under
it a line shows the URLs clients will use, a warning when there is no API key, and the Generate
button makes and copies a key. The setup's last step has a "Reachable from other machines"
checkbox that does the same. A standalone server's host has the same three choices in the
preset editor.

- **Client URLs.** A server listening on `0.0.0.0` is shown as one URL per address: loopback
  and each of this machine's LAN addresses (container bridges such as `docker0` and link-local
  `169.254.x.x` addresses are left out), for example `http://192.168.1.20:8080/v1`. Home,
  Serving, `up`, `serve`, `ps` (`urls` in `--json`) and the `curl` snippet show them; the snippet
  uses the first LAN URL. `0.0.0.0` is never shown as an address.
- **API key.** With no key, anyone who can reach the port can use the server and the models on
  it. Starting an exposed gateway or standalone server without a key prints a warning (and
  shows one in the TUI) but still starts. `llamactl config --new-api-key` makes a key, saves it
  and prints it once; `llamactl config` and the TUI never show it again. Clients send
  `Authorization: Bearer KEY`.
- **`--lan` and `--host ADDR`** on `up` and `serve` (mutually exclusive) apply to that start
  only and are never saved. If the gateway is already running with another bind they fail and
  tell you to run `llamactl down` first. `serve --standalone` takes the same flags for that
  server.
- **Changing the bind of a running gateway** needs `llamactl down` and `llamactl up`; saving
  the new host does not move a process that is already listening.
- **WSL2.** With the default NAT networking other machines cannot reach a server in WSL at all;
  use `networkingMode=mirrored` in `.wslconfig` or forward the port on Windows with
  `netsh interface portproxy`. With mirrored networking the Hyper-V firewall may still block
  inbound connections until a rule allows the port (`New-NetFirewallHyperVRule`, run in
  PowerShell as administrator). llamactl detects WSL and prints these notes next to an exposed
  bind in Settings, `up` and `config`.
- **Other firewalls.** llamactl does not touch the host firewall; open the port yourself if
  one is running.

`llamactl config` shows and changes `gateway.host`, `gateway.port`, `gateway.api_key`,
`gateway.models_max`, `gateway.sleep_idle`, `gateway.autostart`, `default_intent` and
`headroom_pct`. All `KEY=VALUE` pairs are checked before any is saved. Gateway settings except
`autostart` reach a running gateway only after `down` and `up`.

## The TUI

Five views in a sidebar. `1` to `5` switch views, `r` rescans models, `q` quits.

| View | Shows | Keys |
|------|-------|------|
| 1 Home | devices, gateway state and endpoint URLs, what is loaded, next steps | `u` start gateway, `d` stop it, `y` copy base URL, `c` copy `curl` |
| 2 Models | models with size, quant, best intent, active preset, measurements; per-intent detail | Enter opens model workspace, `v` serve, `s` standalone, `t` tune, `p` plan, `e` edit preset, `a` set active, Delete remove preset, `D` default profile |
| 3 Serving | gateway models and standalone or attached servers, with a log tail | `l` load, `x` unload, `s` stop standalone, `u`/`d` gateway up/down, `a` attach a URL, `o` full log, `y` copy URL, `c` copy `curl` |
| 4 Discover | Hugging Face search and download; GGUF files marked by whether the weights fit | `/` search, Enter search / list files / download, `k` set token, `x` cancel download |
| 5 Settings | llama.cpp folder, model folders, default intent, gateway, headroom, Hugging Face token | Ctrl+S save |

In the preset editor: Ctrl+S saves, Esc cancels, F1 explains the field, F2 applies the plan,
Ctrl+R resets the field.

Selecting a model opens its workspace and does not load it into the gateway. Choose a purpose
there to review or edit its preset, set tuning goals, tune, or serve. Tuning shows the saved
preset and candidate side by side; accepting saves for the next load, while keeping the current
choice leaves the preset unchanged. A running instance retains its launch settings. Its running
snapshot and any saved changes pending the next load are shown separately. New gateways use
a detached request supervisor to apply saved definitions before an external on-demand load,
even after the TUI closes. Running models retain their existing settings. Gateways started
before this update keep their direct-router behavior; stop and start the gateway once to
enable supervised loading. After a download, choose a purpose and optionally tune or serve.

## Where state lives

All state is in one directory, chosen in this order: `$LLAMACTL_HOME`, then
`$XDG_CONFIG_HOME/llamactl`, then `~/.config/llamactl`. Set `LLAMACTL_HOME` to a scratch
directory to try llamactl without touching your real setup.

| File | Holds |
|------|-------|
| `config.json` | settings, default profile, per-model aliases and presets, gateway settings |
| `gateway.ini` | router presets, generated from the config |
| `gateway.json` | pid, URL and listen address of a running gateway |
| `sessions.json` | standalone and attached servers |
| `tuning.json` | every measured trial, used to correct estimates |
| `gguf_cache.json` | parsed GGUF headers |
| `hf_token` | Hugging Face access token (the `HF_TOKEN` environment variable also works) |
| `logs/` | server logs |

The gateway and standalone servers keep running when the TUI closes.

## Upgrading from 0.1.x

The first run of 0.2 reads a 0.1.x `config.json` (config version 2) and saves it as version 3:

- Each model's saved profile becomes a preset named `custom`, marked as edited and active.
- The old file is copied to `config.v2.bak.json` once, before the first save.
- Folders and settings carry over. `seen.json` is no longer used.
- Commands that only read (`hw`, `ls`, `plan`, `ps`, `logs`) do not migrate or write.

Use `plan`, `tune` or the Models view to add intent presets next to `custom`.

## Removed in 0.2

- The built-in chat screen. Connect a frontend to the gateway instead.
- The `llama-cli` launch mode. Only `llama-server` is run.
- Launch modes, and the per-model recommendation prompts. The planner replaces them: it
  starts from an intent and shows its reasons.

## Development

```bash
uv sync --extra dev --locked
.venv/bin/pytest -q
uv build
```

The tests never start llama.cpp and never touch your real state directory. Design:
[docs/SPEC.md](docs/SPEC.md). Changes: [CHANGELOG.md](CHANGELOG.md). Release steps:
[docs/RELEASING.md](docs/RELEASING.md).
