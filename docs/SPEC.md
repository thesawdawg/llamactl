# llamactl v2: goal-first serving

Status: approved direction (2026-11-09), implementation in phases (section 11).
Progress is tracked in [TODO.md](TODO.md). The previous design is archived at
[archive/SPEC-v1.md](archive/SPEC-v1.md).

Target: Python >= 3.11, Textual >= 8.2, Linux first (macOS best effort). No new runtime
dependencies (`textual`, `httpx` only). Verified against llama.cpp `0.5.0-dev` commit `5fc4f3c`.

## 1. Vision

v1 is a flag editor with a recommender attached: you pick a model, look at ~30 settings, and
launch one server per model on its own port.

v2 turns that around. You say **what you want the model for**; llamactl works out the settings
for **this machine**, proves them by **measuring**, and serves everything behind **one stable
endpoint** that other frontends and agent harnesses connect to.

```
  what you have            what you want          what llamactl does
  -------------            -------------          ------------------
  hardware (any backend)   intent: chat / agent   plan  -> estimate settings, explain why
  GGUF models              / context / speed      tune  -> launch, measure VRAM + tok/s, keep the best
                           / api / embed          serve -> one gateway endpoint, models load on demand
```

Decisions taken with the owner:

| Topic | Decision |
|---|---|
| Tuning | Instant estimate, plus an on-demand measured tune that also calibrates later estimates |
| Serving | One gateway endpoint (llama-server router mode) first; standalone per-model servers stay available |
| Profiles | Intent presets; a model can hold several; any preset can still be hand-edited |
| Hardware | Backend-agnostic detection through llama.cpp's own `--list-devices` |
| Surfaces | TUI and a headless CLI over the same core |
| Onboarding | Guided first run |
| Chat | **Removed.** llamactl is a server-only tool. No chat screen, no `llama-cli` mode |

Non-goals: chat UI, web UI, building or updating llama.cpp, a hard-coded "best models" list,
remote/multi-host management.

## 2. Architecture

The core is plain Python with no Textual imports. The TUI and the CLI are thin layers over it.

```
llamactl/
  hardware.py     Device/Host detection (replaces hostinfo.py)
  gguf.py         GGUF header reader (kept; gains pooling/embedding detection)
  schema.py       Settings registry (kept; server-only, new flags)
  config.py       Config v3: presets per model, gateway settings, migration from v2
  command.py      argv builder + preset-INI key/value builder from the same registry
  intents.py      Intent definitions
  planner.py      Budget model + plan(intent, model, host) (replaces recommend.py)
  calibration.py  Measured results store + estimate correction factors
  tuner.py        Measured search over candidate plans
  gateway.py      Router-mode process, gateway.ini generation, router API client
  sessions.py     Standalone and attached servers (kept, slimmed)
  library.py      Model discovery + name resolution shared by TUI and CLI
  cli.py          argparse entry point; no args -> TUI
  app.py, views/, screens/, widgets/   TUI
```

Removed: `hostinfo.py`, `recommend.py`, `screens/chat.py`, the `cli` launch mode.

Rules: nothing slower than ~50 ms on the UI thread; every subprocess and HTTP call is injectable
so tests never start llama.cpp; code and tests never touch `~/.config/llamactl` or `~/llama.cpp`
(tests redirect all paths through the `isolated` fixture).

## 3. Hardware (`hardware.py`)

```python
@dataclass(frozen=True)
class Device:
    id: str          # "CUDA0", "Vulkan1", "Metal" ... as printed by llama.cpp
    backend: str     # id with trailing digits stripped, lower-cased: "cuda", "vulkan", "metal"
    name: str
    total: int       # bytes
    free: int        # bytes

@dataclass
class Host:
    gpus: list[Device]; ram_total: int; ram_free: int
    cpu_name: str; cpu_threads: int; cpu_cores: int
    unified: bool    # GPU memory is system RAM (Apple Silicon / integrated)
    # vram_total, vram_free, summary() as in v1
```

- `list_devices(bin_dir) -> list[Device]` runs `llama-server --list-devices` and parses lines of the
  form `  CUDA0: NVIDIA GeForce RTX 4060 Ti (16379 MiB, 15223 MiB free)`. This is the source of
  truth for what llama.cpp can actually offload to, whatever the vendor.
- Fallback when the binary is missing or prints no devices: `nvidia-smi` as in v1; otherwise no GPU.
- `detect(bin_dir, runner=subprocess.run) -> Host`. The device list (ids, names, totals) is cached
  per process; free memory is refreshed by `refresh(host)`: `nvidia-smi` when every device is CUDA
  (cheap), otherwise `--list-devices` again, no more than once every 15 s.
- RAM/CPU: `/proc` on Linux; `sysctl` on macOS; zeros elsewhere. `unified=True` when backend is
  `metal`.
- More than one device: budgets are computed per device and `tensor_split` is planned in
  proportion to usable memory (section 5.3).

## 4. Settings, presets and config

### 4.1 Schema changes

`Setting.modes` and the `cli` mode are dropped; every setting is a server setting. New entries:

| key | flag | kind | default / unset | notes |
|---|---|---|---|---|
| fit | -fit | CHOICE on/off | "on" / "on" | let llama.cpp place layers for unset args |
| fit_target | -fitt | INT (MiB) | 0 / 0 | free memory margin per device for `--fit` |
| device | -dev | STR | "" / "" | comma-separated device ids; empty = all |
| cache_reuse | --cache-reuse | INT | 0 / 0 | KV shifting chunk size; agents benefit |
| sleep_idle | --sleep-idle-seconds | INT | 0 / 0 | unload after N idle seconds |
| embedding | --embedding | BOOL | False | embeddings-only server |
| pooling | --pooling | CHOICE | "" / "" | none, mean, cls, last, rank |
| threads_batch | -tb | INT | 0 / 0 | prompt/batch threads |

Each setting also gains `preset_key: str` (the flag without leading dashes) and
`router_owned: bool` (host, port, api_key, alias: omitted from gateway presets because the router
overrides them).

### 4.2 Presets

```python
@dataclass
class Preset:
    intent: str                 # an intent key, or "custom"
    profile: Profile            # concrete settings
    source: str                 # "estimated" | "tuned" | "edited"
    measured: Measurement | None  # see section 6
```

### 4.3 Config v3 (`config.json`)

```json
{
  "version": 3,
  "bin_dir": "...", "model_dirs": ["..."],
  "headroom_pct": 10, "gauge_interval": 3.0,
  "default_profile": { },
  "default_intent": "chat",
  "models": {
    "/abs/model.gguf": {
      "alias": "qwen3-8b",
      "active": "agent",
      "presets": { "agent": { "intent": "agent", "profile": { }, "source": "tuned", "measured": { } } }
    }
  },
  "gateway": { "host": "127.0.0.1", "port": 8080, "api_key": "", "models_max": 1,
               "sleep_idle": 600, "autostart": false },
  "onboarded": true
}
```

- Migration: a v2 `profiles[path]` becomes `models[path].presets["custom"]` with
  `source: "edited"` and `active: "custom"`. The v2 file is copied to `config.v2.bak.json` once
  before the first v3 save. Unknown keys are ignored, as in v1.
- `default_profile` remains the base every plan starts from (sampling and server fields are
  never changed by the planner).
- Aliases are derived from the file stem (lower-case, quant suffix kept), unique across models,
  and editable.
- `seen.json` is retired; "new" means the model has no presets.

### 4.4 State directory

Every state file (`config.json`, `gateway.ini`, `gateway.json`, `sessions.json`, `gguf_cache.json`,
`tuning.json`, `hf_token`, `logs/`) lives in one directory, resolved once by
`config.state_dir()` when the package is imported: `$LLAMACTL_HOME` if set, else
`$XDG_CONFIG_HOME/llamactl` if set, else `~/.config/llamactl`. Empty variables count as unset.
Each module derives its path constants from `config.CONFIG_DIR`; the tests redirect those
constants in the `isolated` fixture, and `tests/conftest.py` additionally points
`LLAMACTL_HOME` at a temp dir for the whole session before anything is imported, so a test that
forgets `isolated` still cannot reach the real directory.

## 5. Intents and the planner

### 5.1 Intents (`intents.py`)

An intent is declarative data, so adding one never touches the planner.

```python
@dataclass(frozen=True)
class Intent:
    key: str; label: str; description: str
    ctx_min: int; ctx_target: int; ctx_max: int   # ctx is per slot; 0 ctx_max = trained ctx
    slots: int
    kv_quant: str            # "never" | "if_needed" | "prefer"   (q8_0 K/V)
    spill: str               # "avoid" | "allow": may weights leave the GPU to reach ctx_target?
    extra: dict[str, Any]    # fixed settings, e.g. {"cache_reuse": 256}
    applies: Callable[[ModelInfo], bool]   # e.g. embed only for embedding models
```

| key | for | ctx per slot (min / target / max) | slots | KV q8 | spill | extra |
|---|---|---|---|---|---|---|
| chat | interactive chat frontends | 8k / 16k / 32k | 1 | if_needed | avoid | |
| agent | coding agents and tool harnesses | 32k / 64k / 128k | 1 | prefer | allow | cache_reuse 256 |
| context | largest usable window | 32k / trained / trained | 1 | prefer | allow | |
| speed | highest tokens per second | 4k / 8k / 16k | 1 | never | avoid | |
| api | several concurrent clients | 4k / 8k / 16k | 4 | if_needed | avoid | metrics on |
| embed | embedding models | trained | 4 | never | avoid | embedding on |

`embed` applies only when the GGUF declares a pooling type; the others apply only when it does
not. Total `-c` is `ctx_per_slot * slots` and `-np` is emitted explicitly.

### 5.2 Budget (`planner.py`)

`Budget`, `kv_per_token`, `estimate_budget` and `verdict` move here unchanged in meaning, with two
additions: the total is multiplied by the calibration factor for the backend (section 6.3,
1.0 when nothing has been measured), and on a `unified` host VRAM and RAM are one pool.

### 5.3 Planning

```python
@dataclass(frozen=True)
class Plan:
    intent: str
    profile: Profile
    reasons: dict[str, str]     # changed key -> one line with the numbers used
    budget: Budget
    verdict: str                # "GPU" | "GPU+CPU" | "CPU" | "NO"
    ctx_per_slot: int
    warnings: tuple[str, ...]   # e.g. "ctx below this intent's minimum"

def plan(intent: Intent, info: ModelInfo, host: Host, base: Profile,
         headroom_pct: float = 10, calibration: Calibration | None = None) -> Plan
def plan_all(info, host, base, ...) -> dict[str, Plan]   # every applicable intent
```

Algorithm (usable VRAM = total minus headroom, per device):

1. Fixed: `threads` = physical cores, `flash_attn` on with a GPU, `parallel` = slots, the
   intent's `extra`.
2. Whole model on GPU: choose the largest ctx tier between `ctx_min` and `ctx_target` that fits
   with f16 KV. If `kv_quant` is `prefer`, or `if_needed` and the target was not reached, retry
   with q8_0 K/V. On success: `gpu_layers = "all"`, `fit = "off"`, verdict `GPU`.
3. Target not reached and `spill == "allow"`: keep the highest tier that fits in VRAM + free RAM,
   leave `gpu_layers = "auto"`, set `fit = "on"` and `fit_target` = headroom in MiB, so llama.cpp
   places the layers. For MoE models set `n_cpu_moe` as in v1 instead. Verdict `GPU+CPU`.
4. Target not reached and `spill == "avoid"`: keep the largest tier that fits on GPU when it is
   at least `ctx_min`. When no GPU-only tier reaches `ctx_min` (including when the weights alone
   exceed usable VRAM), spill instead (the step 3 mechanism) but aim only for `ctx_min`, never
   `ctx_target`: verdict `GPU+CPU` with a warning that part of the model runs on the CPU to keep
   a usable context. If even spilling cannot reach `ctx_min` (no spare RAM, or unified memory),
   keep the largest GPU tier, or the largest tier spilling allows, and warn that it is below the
   minimum. A spill-"avoid" intent therefore never plans more than `ctx_min` by spilling, so more
   VRAM never gives a smaller context for the same intent and model.
5. No GPU: `gpu_layers = "0"`, ctx against free RAM, verdict `CPU`. Nothing fits: verdict `NO`
   with best-effort values.
6. Multi-GPU: `tensor_split` proportional to each device's usable memory.

llamactl owns the trade-offs that depend on intent (context, slots, KV precision); llama.cpp's
`--fit` owns exact layer placement when the model does not fit; the tuner checks both.

## 6. Measured tuning

### 6.1 Measurement

```python
@dataclass(frozen=True)          # llamactl/measure.py (no imports from config/calibration)
class Measurement:
    ok: bool; error: str             # "<kind>: detail (log: path)"; kinds: out of memory, timeout,
                                     # unknown argument, load failed, benchmark failed, cancelled
    load_s: float
    vram_used: int; ram_used: int    # bytes, delta against the pre-launch snapshot
    prompt_tps: float; gen_tps: float  # from the /completion `timings` object
    at: str                          # ISO timestamp
    build: str                       # llama-server --version string (e.g. "0.5.0-dev (build 1, commit 5fc4f3c)")
```

`vram_used` is the larger of the deltas after load and after the benchmark request.
`resource_failure` (out of memory or timeout) is the only kind the search answers by stepping down;
an unknown argument or architecture will not improve with a smaller context.

### 6.2 Tuner (`tuner.py`)

```python
class Runner(Protocol):                       # the only part that touches processes / HTTP
    def measure(self, model: Path, profile: Profile,
                cancel: Callable[[], bool] | None = None) -> Measurement: ...

def tune(intent, info, host, base, runner, *, headroom_pct=10, calibration=None, max_trials=4,
         progress: Callable[[TrialEvent], None] | None = None,
         cancel: Callable[[], bool] | None = None) -> TuneResult
def save_result(cfg, model, result, calibration, *, activate=False) -> None
def preflight(plan, host) -> str | None       # refusal text when free memory is clearly short
```

- `ServerRunner` launches `llama-server` on a free port bound to 127.0.0.1 (host, port and API key
  of the profile are overridden, idle sleep off), logs to a temp file in the log directory, waits
  for `/health` (timeout 60 s + 20 s per GB of model file, at most 30 min), snapshots device
  memory and RAM, posts a fixed prompt to `/completion` (`n_predict` 128, `cache_prompt` false,
  seed 42, temperature 0), reads `timings`, then always stops the process group (SIGTERM, SIGKILL
  after 15 s) and waits for it to be gone. Embedding profiles post a fixed text to
  `/v1/embeddings` instead and report `usage.prompt_tokens` over wall time as `prompt_tps`.
  `preflight` refuses to start when free memory is below 90% of the plan's budget; offering to
  unload gateway models is the caller's job.
- `cancel` is checked before the launch, on every health poll and before the benchmark request;
  when it fires the server is stopped and the result is a failed `Measurement` whose error starts
  with `cancelled` (`Measurement.cancelled`). The search treats it as cancellation: it is neither a
  load failure to step down from nor a recorded trial (it is still reported to `progress`).
- Search starts from `plan()` and is bounded by `max_trials`; a plan with verdict `NO` is not tried:
  - load failed (memory or timeout): step down - next lower ctx tier (not below `ctx_min`); then
    q8_0 KV if the intent allows; then let llama.cpp place the layers (`gpu_layers` auto, `fit` on);
  - loaded fully on the GPU, below `ctx_target`, and the measured slack (free VRAM before the
    launch, minus the model's `vram_used`, minus headroom) exceeds the extra KV of the next tier:
    step up one tier. A failed step up ends the climb and keeps the last good trial;
  - `GPU+CPU` candidates: one extra trial with `threads` = all logical cores, keep the faster;
  - an identical profile is never launched twice.
- Score: among trials that loaded and reach `ctx_min`, the highest ctx up to `ctx_target`, then the
  highest `gen_tps` (`prompt_tps` for embedding models); `speed` scores by `gen_tps` first. When
  nothing reaches `ctx_min` the best loaded trial still wins and the message says it is below the
  minimum. Nothing loaded: `ok` is false, the trials are still recorded.
- `save_result` stores the winner as the model's preset with `source: "tuned"` and its
  `Measurement` (it becomes the active preset only when asked or when the model has none), and
  records every trial in `tuning.json`.

### 6.3 Calibration (`calibration.py`)

`tuning.json` stores every trial (model path, mtime, intent, profile, estimated budget,
measurement, backend). `Trial.estimated_total` is `Budget.raw_total()` - the uncalibrated sum - so
the factor never compounds. `Calibration.factor(backend)` is the median of
`measured vram / estimated total` over successful all-GPU trials, clamped to 0.8..1.5. The
planner applies it, so estimates on this machine improve with every tune. `record`/`save` write
the file atomically (temp file, then replace); `trials_for(model, intent)` lists trials;
`is_stale(preset, model_path, build)` is true for a tuned preset whose model mtime (kept in the
trial that produced the measurement) or llama.cpp build (kept in the `Measurement`) changed; the
UI flags those "stale".

## 7. Serving

### 7.1 Gateway (`gateway.py`)

One `llama-server` in router mode is the default way to serve:

```
llama-server --models-preset ~/.config/llamactl/gateway.ini --models-max N \
             --host H --port P [--api-key K]
```

- `render_presets(cfg, models=None) -> str` builds the INI from the same registry as the argv
  builder (`command.preset_items`). Output starts with `version = 1`, then `[*]`, then models
  sorted by alias and presets by name, so it is stable and diffable. One section per preset: the
  model's active preset is `[alias]`; every other preset is `[alias@name]`, so a client asks for
  `qwen3-8b` or `qwen3-8b@agent`. The separator is `@`, not `:`: llama.cpp treats the text after
  the last `:` of a section name as a tag and rewrites it (`qwen3-8b:agent` is served as
  `qwen3-8b:AGENT`, `qwen3-8b:my-preset` as `qwen3-8b:PRESET`), which would break lookup by
  name. Each section carries `model = /abs/path` plus the preset's non-unset, non-router-owned
  settings; models whose file is gone are skipped. `[*]` carries `sleep-idle-seconds` from the
  gateway settings when above 0.
- INI details verified against llama.cpp (`common/preset.cpp`): keys are long flags without
  dashes; booleans are `true` / `false`, and llama.cpp maps `cont-batching = false` to
  `--no-cont-batching` itself (so llamactl never writes `no-` keys for registry settings); a
  value ends at the first `;` or `#` and cannot hold a line break, so a preset containing one is
  skipped with a `; skipped [...]` comment and reported; an option the running llama.cpp does
  not know (for instance a misspelt one in `extra_args`) makes the whole router refuse to
  start, which `start()` reports from the log. `extra_args` is parsed into `--flag value` /
  `--flag` pairs; a bare token cannot be expressed and is an error.
- `Gateway` class: `start()`, `stop()`, `status()`, `models()` (`GET /models`),
  `load(name)`, `unload(name)`, `reload()` (rewrite INI, `GET /models?reload=1`). State (pid, url,
  log) persists in `gateway.json` so the gateway outlives the TUI, like sessions do. Model
  states are `unloaded`, `loading`, `loaded`, `sleeping` and `failed` (the router reports a
  crashed model as `unloaded` with `failed` and an `exit_code`). The API key is sent as a
  Bearer header; `/health` needs none.
- Saving or tuning a preset regenerates the INI and reloads a running gateway.
- `models_max` defaults to 1 per GPU-sized model; the Home view explains what will be unloaded
  when a second model is requested.

### 7.2 Standalone and attached servers (`sessions.py`)

Kept for pinned servers on their own port (`serve --standalone`) and for attaching external
URLs. `launch_server(cfg, model, preset_name)` takes the preset explicitly.

## 8. CLI (`cli.py`)

`llamactl` with no arguments opens the TUI. Subcommands (all accept `--json` where they print
data; models are named by alias, file stem or unique substring):

| Command | Does |
|---|---|
| `llamactl hw` | devices, RAM, CPU, calibration factor |
| `llamactl ls` | models with size, quant, verdict per intent, presets |
| `llamactl plan MODEL [--for INTENT]` | print the plan, reasons and exact command; changes nothing |
| `llamactl tune MODEL [--for INTENT] [--trials N]` | measured tune, saves the preset |
| `llamactl serve MODEL [--for INTENT] [--standalone] [--port P]` | ensure a preset exists (plan if not), then load it in the gateway (starting it if needed) or start a standalone server |
| `llamactl up` / `llamactl down` | start / stop the gateway |
| `llamactl ps` | gateway models and standalone sessions with state |
| `llamactl load NAME` / `unload NAME` | gateway model control |
| `llamactl stop NAME\|--all` | stop standalone servers |
| `llamactl logs NAME [-f]` | print or follow a log |

Exit codes: 0 ok, 1 runtime failure, 2 usage / unknown or ambiguous model.

As built:

- `cli.py` only parses arguments and runs commands. Each command is `cmd_x(args, services)`;
  `Services` holds the config, lazy host detection, model list, GGUF cache, calibration, `Gateway`,
  `SessionStore`, the tune `Runner` factory, clocks and the output streams, and tests replace it
  wholesale (`main(argv, services)`). `report.py` turns core objects into plain dicts (also what
  `--json` prints) and `render.py` turns those into aligned text; colour only on a TTY with
  `NO_COLOR` unset. Importing `cli` never imports Textual; no arguments launch the TUI.
- The multi-step logic the TUI needs too lives in `actions.py`: `choose_intent`, `plan_for`,
  `store_plan`, `pick_preset` (plan and store an `estimated` preset only when it is missing),
  `served_name` (`alias`, or `alias@preset` for a non-active preset), `ensure_gateway`,
  `serve_gateway`, `wait_loaded`, `wait_unloaded`, `wait_server_ready`, `run_tune` (preflight,
  search, save, reload the gateway; built from `search_tune`, `apply_tune` and `reload_after_tune`,
  and `save=False` runs only the search) and `curl_snippet`.
- `hw`, `ls`, `plan` (without `--save`), `ps` and `logs` never write the config or a state file
  (`Services(read_only=True)`: no GGUF cache writes, `sessions.json` and `gateway.json` untouched).
  Commands that save the config perform the v2 -> v3 migration and its backup.
- `plan --save` stores an `estimated` preset and refuses to replace a tuned or edited one without
  `--force`. `serve` uses an existing preset as is; `--for` makes the preset active (so it is served
  as plain `alias`) unless `--no-activate`, which serves `alias@intent`. `serve` waits for the model
  (progress dots on stderr, timeout as the tuner's health timeout, `--no-wait` to skip) and prints
  the base URL, the model name and a `curl` line. `tune` streams one line per trial to stderr;
  Ctrl-C stops the running trial's server and keeps the trials so far (exit 1).

## 9. TUI

Five views in a sidebar shell (shell, gauges and status bar carry over from v1):

1. **Home** - hardware card (devices with bars), gateway card (state, endpoint URL, `y` copies
   base URL, a ready-made `curl`/OpenAI snippet), what is loaded now, and "next step" prompts
   (no models, untuned presets, stale presets).
2. **Models** - table `Model | Size | Quant | Best for | Preset | Measured`. The detail pane
   shows one row per applicable intent with verdict, ctx and tok/s (estimated or measured).
   `Enter` serve (intent picker when several), `t` tune, `e` edit preset, `p` plan details,
   `Delete` remove preset.
3. **Serving** - gateway models (loaded / sleeping / unloaded / failed) and standalone or
   attached servers in one table; log tail; load, unload, stop, attach, copy URL.
4. **Discover** - Hugging Face search and download (carried over). File lists mark the largest
   quant that fully fits the GPU for the default intent; after a download the intent picker opens.
5. **Settings** - llama.cpp path, model dirs, gateway settings, headroom, HF token, default
   intent.

Screens: `IntentPicker` (cards with verdict/ctx/why per intent), `PlanScreen` (was
RecommendationScreen: verdict, budget, Setting/Base/Planned/Why table; save as preset, edit, tune, serve), `TuneScreen` (live trial table with
progress, cancel), `PresetEditor` (was ProfileEditor, unchanged mechanics), `Onboarding`.

As built (Phase 6; Discover's fit-aware lists, the post-download picker and Onboarding are
Phase 7):

- **Services and threading.** `LlamaCtl` owns `cfg`, `host`, `models` + `infos` (GGUF headers),
  `calibration`, `gateway`, `store`, the llama-server `build` string and the last
  `ServingSnapshot`; every one is injectable (`LlamaCtl(cfg=, gateway=, store=, runner_factory=,
  build=, health=)`), which is how the tests run without llama.cpp. Views never poll: the host
  poll (`hardware.detect/refresh`), the model scan, and one serving poll (`serving.gather`:
  gateway status + models + session health, every 2 s, every 6 s while nothing runs) run in thread
  workers through `run_bg`, and the serving poll posts one `ServingChanged` message that Home and
  Serving both consume. Workers hand results back with `LlamaCtl.ui`, which never blocks and drops
  the call when the app is closing or the target widget is gone (a plain `call_from_thread`
  raised `NoMatches` into the worker during shutdown: the old intermittent `WorkerFailed`).
  Failures (`GatewayError`, `ActionError`, `LookupError`, `ValueError`, `OSError`, `GGUFError`)
  become a status-bar message plus a notification, never a traceback. `presets_changed()` is the
  one helper after any preset change: save the config, rewrite and reload the gateway presets in
  a worker, refresh the rows.
- **Keys.** `1`-`5` switch views; `r` rescans models; `q` quits.
  Home: `u` start gateway, `d` stop (confirms), `y` copy base URL (`.../v1`), `c` copy curl, Enter
  on a next step. Models: Enter serve (IntentPicker when the model has no preset or several),
  `s` standalone, `t` tune, `p` plan, `e` edit preset, `a` set active, `Delete` remove preset,
  `D` default profile. Serving: `l` load, `x` unload, `s` stop standalone (confirms), `u`/`d`
  gateway up/down, `a` attach, `o` full log, `y` copy URL, `c` copy curl. Settings: `Ctrl+S`.
  Inside editors the F-key / ctrl convention applies (`Ctrl+S`, `F1`, `F2`, `Ctrl+R`).
- **Serve rules.** Enter on a model with exactly one preset serves it. Picking an intent in the
  picker plans and stores an `estimated` preset when missing; it only becomes active when the
  model had none, so an existing model's pick is served as `alias@intent` and its active preset
  does not change (`a` changes it).
- **Next steps** are `suggest.next_steps` (pure): llama-server missing, no models, models without
  presets, estimated active presets (tune), stale tuned presets (re-tune), gateway stopped while
  presets exist (start). At most three per rule.
- **TuneScreen** runs `actions.run_tune(save=False)` (search only) in a worker that outlives the
  screen; the result is applied on the UI thread (`actions.apply_tune`: trials, preset, saves) and the
  gateway reload (`actions.reload_after_tune`) runs in a second worker under the gateway lock. When
  the app is already quitting the worker applies the result itself. Cancel and closing
  set the cancel flag, which the tuner polls between trials and `ServerRunner.measure` polls
  while it waits for `/health` and before the benchmark: the trial in flight is stopped (its
  server too), later ones do not start, and the trials that finished are recorded.
- **Settings** saves in place into `cfg.gateway`, rewrites the presets and reloads a running
  gateway; host, port, API key and models-max are command line options of the router, so
  changing them tells the user to restart it.

Onboarding runs when `onboarded` is false: (1) locate llama.cpp (`bin_dir`, then `PATH`) and show
its version; (2) show detected hardware; (3) confirm model directories and scan; (4) no models:
go to Discover; otherwise pick a model and an intent; (5) plan, optional tune; (6) start the
gateway and show the endpoint. Every step can be skipped.

## 10. Testing

- Pure core (`hardware` parsing, `planner`, `intents`, `calibration`, `tuner` with a fake
  `Runner`, `gateway.render_presets`, `command`, config migration, `cli` with injected services)
  is covered by pytest without starting llama.cpp.
- Textual `Pilot` smoke tests for each view and screen.
- `pytest -q` is green at every commit.
- Manual check per serving phase on the reference machine (RTX 4060 Ti 16 GB): generated
  `gateway.ini` is accepted by the real router, and a tune of a local model completes.

## 11. Phases

Each phase lands as Conventional Commits on the `redesign-v2` branch. No pushing.

1. **Foundations** - remove chat and the cli mode; `hardware.py`; schema additions; config v3
   with presets and migration; `library.py`.
2. **Planner** - `intents.py`, `planner.py` (replacing `recommend.py`), calibration read path.
3. **Gateway** - preset INI rendering, router process and API client, sessions take presets.
4. **Tuner** - `Measurement`, `ServerRunner`, search, calibration store.
5. **CLI** - all subcommands over the core.
6. **TUI** - Home, Models, Serving, IntentPicker, PlanScreen, TuneScreen, PresetEditor.
7. **Onboarding and Discover** - first-run flow, fit-aware file lists.
8. **Release prep** - README, CHANGELOG, AGENTS.md, version 0.2.0 (no tag, no publish).
