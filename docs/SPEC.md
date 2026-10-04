# llamactl redesign specification

Status: approved design, implementation in phases (see section 9).
Target: Textual >= 8.2, Python >= 3.11, Linux. No new runtime dependencies.
Verified against llama.cpp build 5fc4f3c (2026-10-01) at `~/llama.cpp/build/bin`.

## 1. Goals

1. Responsive UI: nothing slower than ~50 ms runs on the UI thread.
2. Intuitive controls: sidebar navigation, typed widgets, visible state, no silent failures.
3. Guided settings: every setting has a short hint, a long help text, and (where it applies) a live computed impact.
4. More tunables as first-class settings (memory/offload, sampling, server/API). `extra_args` stays as the escape hatch.
5. GGUF-aware recommendations proposed on download / first sight, confirmed by the user before saving.
6. Testable core: schema, GGUF reader, recommender and command builder are pure Python with pytest coverage.

Non-goals: web UI, multiple profiles per model, named presets, chat screen redesign (deferred).

## 2. Module layout

```
llamactl/
  app.py              App shell: sidebar + ContentSwitcher, global bindings, host polling
  views/
    models.py         ModelsView: table + detail pane
    sessions.py       SessionsView: table + log/props pane
    hf.py             HFView (moved from hf_screen.py, same behaviour + recommendation hand-off)
    settings.py       SettingsView: bin_dir, model_dirs, HF token
  screens/
    profile.py        ProfileEditor screen (full screen)
    recommend.py      RecommendationScreen (propose -> confirm)
    help.py           HelpPanel modal for one setting
  widgets/
    gauges.py         ResourceGauges (VRAM / RAM bars) for the header area
    budget.py         BudgetPanel (weights / KV / overhead vs VRAM / RAM)
    field.py          SettingField: label + typed input + hint + impact line
  schema.py           Setting registry (single source of truth)
  config.py           Config / Profile (profile is schema-driven), discovery
  command.py          build_command from schema (moved out of config.py)
  gguf.py             GGUF header reader
  recommend.py        Recommender
  hostinfo.py         unchanged API, detection made cacheable
  sessions.py         unchanged API
  hf.py               unchanged
  forms.py            removed once SettingsView and HelpPanel replace FormScreen
tests/
  test_schema.py  test_gguf.py  test_recommend.py  test_command.py  test_config.py  test_ui.py
docs/SPEC.md          this file
```

All classes carry type hints and docstrings (Args / Returns / Raises). Keep functions short; prefer
composition over inheritance except where behaviour is genuinely shared (views share a `BaseView`).

## 3. Settings schema (`schema.py`)

### 3.1 Data shape

```python
class Kind(StrEnum): INT = "int"; FLOAT = "float"; BOOL = "bool"; CHOICE = "choice"; STR = "str"; PATH = "path"

@dataclass(frozen=True)
class Setting:
    key: str                      # Profile attribute and config key
    flag: str                     # llama.cpp flag, e.g. "-c"; "" when handled specially
    kind: Kind
    label: str                    # short UI label
    hint: str                     # one line shown under the field
    help: str                     # multi-paragraph text for the help panel
    group: Group                  # MEMORY, SAMPLING, SERVER, GENERAL
    default: Any                  # value meaning "let llama.cpp decide" where possible
    modes: frozenset[str] = frozenset({"server", "cli"})
    choices: tuple[str, ...] = () # for CHOICE
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None     # UI stepper increment
    unset: Any = None             # value that means "omit the flag" (e.g. 0 for ctx_size, "" for strings)
    impact: Callable[[Profile, ModelInfo | None, Host], str] | None = None
```

`SETTINGS: tuple[Setting, ...]` is ordered as the editor shows them. Helpers:
`by_key: dict[str, Setting]`, `for_group(group)`, `for_mode(mode)`, `validate(setting, raw: str) -> value`
(raises `ValueError` with a user-readable message).

### 3.2 Registry

| key | flag | kind | default | unset | group | notes |
|---|---|---|---|---|---|---|
| ctx_size | -c | INT | 0 | 0 | MEMORY | 0 = model's trained max; step 1024 |
| gpu_layers | -ngl | STR | "all" | "" | MEMORY | accepts "auto", "all" or an int; validate accordingly |
| n_cpu_moe | --n-cpu-moe | INT | 0 | 0 | MEMORY | server+cli; only meaningful if `ModelInfo.expert_count > 0` |
| cache_type_k | -ctk | CHOICE | "f16" | "f16" | MEMORY | f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1 |
| cache_type_v | -ctv | CHOICE | "f16" | "f16" | MEMORY | same choices; quantised V requires flash_attn != off |
| flash_attn | -fa | CHOICE | "auto" | - | MEMORY | on, off, auto; always emitted |
| batch_size | -b | INT | 0 | 0 | MEMORY | step 256 |
| ubatch_size | -ub | INT | 0 | 0 | MEMORY | step 128; must be <= batch_size when both set |
| load_mode | --load-mode | CHOICE | "auto" | "auto" | MEMORY | auto, none, mmap, mlock, mmap+mlock, dio |
| tensor_split | -ts | STR | "" | "" | MEMORY | comma-separated fractions; only shown when >1 GPU |
| threads | -t | INT | 0 | 0 | MEMORY | 0 = auto |
| temp | --temp | FLOAT | -1 | -1 | SAMPLING | -1 = llama.cpp default (0.8); 0..2 |
| top_k | --top-k | INT | -1 | -1 | SAMPLING | 0 disables |
| top_p | --top-p | FLOAT | -1 | -1 | SAMPLING | 0..1 |
| min_p | --min-p | FLOAT | -1 | -1 | SAMPLING | 0..1 |
| repeat_penalty | --repeat-penalty | FLOAT | -1 | -1 | SAMPLING | 1.0 disables |
| seed | -s | INT | -1 | -1 | SAMPLING | -1 random |
| host | --host | STR | "127.0.0.1" | - | SERVER | server only |
| port | --port | INT | 0 | - | SERVER | 0 = next free from 8080; resolved at launch, always emitted |
| api_key | --api-key | STR | "" | "" | SERVER | masked input |
| parallel | -np | INT | 0 | 0 | SERVER | server slots |
| cont_batching | -cb / -nocb | BOOL | True | True | SERVER | emit `-nocb` only when False |
| metrics | --metrics | BOOL | False | False | SERVER | |
| alias | --alias | STR | "" | "" | SERVER | model name exposed by API |
| draft_model | -md | PATH | "" | "" | SERVER | speculative decoding draft model |
| mmproj | --mmproj | PATH | "" | "" | SERVER | vision projector |
| extra_args | - | STR | "" | "" | GENERAL | appended last via shlex.split |

Sampling fields use -1 as "unset" so that legitimate 0 values (top_k 0, min_p 0) can be expressed.
`--jinja` is default-on in current llama.cpp and is therefore not a setting.

### 3.3 Help text

Hand-written in `schema.py`, sourced from `llama-server --help` and `tools/server/README.md`.
Each `help` covers: what it does, when to change it, typical values, memory/speed consequences, and the raw flag.
Plain ASCII.

## 4. Profile and config (`config.py`)

- `Profile` becomes a dataclass generated from `SETTINGS` (fields with defaults), plus:
  `from_dict(data)` (ignores unknown keys, coerces types), `to_dict()`, `is_set(key)`,
  `merged(**overrides)`.
- Legacy configs: `gpu_layers` stored as int 99 maps to `"all"`; other legacy fields load unchanged.
  `Config.load()` adds `version: 2` on first save.
- `Config.profile_for(model)` unchanged. New: `Config.has_profile(model) -> bool` (drives the *New* badge).
- `discover_models` unchanged, but callers run it in a worker.

## 5. Command builder (`command.py`)

`build_command(cfg, model, prof, mode, port) -> list[str]` iterates `for_mode(mode)` in registry order:

- skip if `value == setting.unset`
- BOOL: `cont_batching` emits `-nocb` when False; other BOOLs emit the flag when True
- INT/FLOAT/CHOICE/STR/PATH: `[flag, str(value)]`
- `port` uses the resolved port argument, `extra_args` is split with shlex and appended last.

Also exposes `command_string(argv) -> str` (shlex.join) for the preview pane.

## 6. GGUF reader (`gguf.py`)

```python
@dataclass(frozen=True)
class ModelInfo:
    path: Path; size: int                  # size = sum of all shards
    architecture: str; name: str
    n_layer: int; n_embd: int; n_head: int; n_head_kv: int   # n_head_kv = mean over layers if stored per layer
    head_dim_k: int; head_dim_v: int       # attention.key_length / value_length, fallback n_embd // n_head
    ctx_train: int; expert_count: int; file_type: int
    sharded: int                           # split.count or 1

def read_model_info(path: Path) -> ModelInfo   # raises GGUFError
```

Parsing: magic `GGUF`, version 2 or 3 (little endian), tensor_count u64, kv_count u64, then KV pairs.
Value types 0..12 per the GGUF spec; arrays are skipped by reading element lengths (strings) or seeking
(fixed-size types) so the tokenizer vocab is never materialised. Only keys with prefix `general.`,
`{arch}.`, `split.` are decoded. Stop reading once kv_count is exhausted; never read tensor data.
Shards: parse the `-00001-of-` file; size = sum of sibling shards present on disk.

Caching: `ModelInfoCache` keyed by `(path, mtime)`, persisted to `~/.config/llamactl/gguf_cache.json`
so restarts are instant. Reads run in a Textual worker; the UI shows metadata as it arrives.

## 7. Recommender (`recommend.py`)

### 7.1 Budget model

```python
@dataclass(frozen=True)
class Budget:
    weights: int        # ModelInfo.size
    kv: int             # kv_per_token(info, ctk, ctv) * ctx  (ctx is the total pool; slots share it, each gets ctx // np)
    compute: int        # see below
    overhead: int       # 512 MiB fixed
    def total(self) -> int
    def fits_vram(self, host) -> bool; def fits_ram(self, host) -> bool

BYTES_PER_ELEM = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34/32, "q5_1": 24/32, "q5_0": 22/32,
                  "q4_1": 20/32, "q4_0": 18/32, "iq4_nl": 18/32}

def kv_per_token(info, ctk, ctv) -> int:
    return info.n_layer * info.n_head_kv * (info.head_dim_k * BYTES_PER_ELEM[ctk] + info.head_dim_v * BYTES_PER_ELEM[ctv])

compute = max(256 MiB, n_embd * ubatch(512 default) * 4 bytes * 8)   # activation / scratch heuristic
```

`estimate_budget(info, prof, host) -> Budget` is also what the Models view and the editor's live impact use,
so recommendations and displayed numbers can never disagree.

### 7.2 Recommendation

```python
@dataclass(frozen=True)
class Recommendation:
    profile: Profile
    reasons: dict[str, str]    # key -> one-line reason, only for keys the recommender changed
    budget: Budget
    verdict: str               # "GPU", "GPU+CPU", "CPU", "NO"

def recommend(info: ModelInfo, host: Host, base: Profile) -> Recommendation
```

Algorithm (headroom = 10% of VRAM, minimum 512 MiB):

1. `threads` = physical cores (parse `cpu cores` x physical ids from `/proc/cpuinfo`; fallback `os.cpu_count() // 2`).
2. `flash_attn` = "on" when a GPU is present, else "auto".
3. Context: candidates `[4096, 8192, 16384, 32768, 65536, 131072]` filtered to `<= ctx_train`.
   Pick the largest whose `Budget` (with weights on GPU) fits VRAM minus headroom using f16 KV.
   If none fits at 4096 go to step 5.
4. If the largest candidate did not reach the next tier but would with `q8_0` K/V, set both to `q8_0`
   and take that tier (requires flash_attn on). Never recommend below q8_0.
5. Offload when weights alone do not fit VRAM minus headroom:
   - MoE (`expert_count > 0`): `gpu_layers = "all"`, `n_cpu_moe = ceil(n_layer * (1 - vram_avail / weights))`
     clamped to `[1, n_layer]`. Reason explains experts live in RAM.
   - Dense: `gpu_layers = floor(n_layer * vram_avail / weights)` and ctx recomputed with the remaining VRAM.
   If weights do not fit VRAM + RAM free: verdict "NO", still return best-effort values.
6. No GPU: `gpu_layers = "0"`, ctx chosen against RAM free, verdict "CPU".
7. Sampling and server fields are never changed by the recommender.

Every changed key gets a reason string with the numbers used, e.g.
`"65536 ctx -> 6.1 GB KV (f16); total 14.2 GB fits 16.0 GB VRAM with 1.6 GB headroom"`.

### 7.3 Tests (`tests/test_recommend.py`)

Fixtures: synthetic `ModelInfo` for an 8B dense (32 layers, 8 kv heads, head_dim 128, ctx 131072, 8.5 GB),
a 35B MoE (48 layers, 4 kv heads, 128, 262144, 20 GB, 128 experts), and a 1B (16 layers) model; hosts of
16 GB VRAM / 24 GB RAM and no GPU / 32 GB RAM. Assert the chosen tier, offload decision, verdict and that
`reasons` keys equal the changed keys. `kv_per_token` checked against a hand-computed value.

## 8. UI

### 8.1 Shell (`app.py`)

```
+------------+------------------------------------------------------+
| llamactl   | VRAM [########....] 10.2/16.0 G   RAM [###.....] 5/23 G |
|------------+------------------------------------------------------|
| > Models   |                                                      |
|   Sessions |                 active view                          |
|   HF       |                                                      |
|   Settings |                                                      |
|------------+------------------------------------------------------|
| status bar: last action result (persists until next action)       |
| footer: key bindings of the active view                           |
+-------------------------------------------------------------------+
```

- Sidebar is a `ListView`; `ContentSwitcher` holds the four views. Keys `1-4` and `Tab` move between views;
  each view defines its own `BINDINGS` so the footer is contextual.
- `ResourceGauges` refresh from a `@work(thread=True, exclusive=True)` host poller every 3 s. `hostinfo.detect()`
  never runs on the UI thread. Gauges turn yellow above 80%, red above 95%.
- Status bar is written only by user actions; health/progress updates go to the Sessions view and `notify()`.
- All screens: Esc returns, `F1` (also `?` outside text inputs) opens context help when a setting field is focused.
  Printable keys are swallowed by focused `Input`s, so every editor action also has an F-key or ctrl binding:
  `F1` help, `F2` apply recommended, `ctrl+r` reset field, `ctrl+s` save.

### 8.2 Models view

- Left (3fr): `DataTable` columns `Model | Size | Params | Quant | Fits | Ctx | Profile`.
  `Fits` is colored (`GPU` green, `GPU+CPU` yellow, `CPU` yellow, `NO` red, `...` while metadata loads).
  `Profile` shows `custom`, `default`, or `New` (no profile and never recommended).
- Right (2fr) detail pane for the highlighted row:
  - metadata (arch, layers, kv heads, trained ctx, experts, file type, shards)
  - `BudgetPanel`: weights / KV / compute / overhead stacked bar vs VRAM and RAM bars, with the numbers
  - profile summary (only non-default settings) and the exact command line
  - buttons: `Launch server (l)`, `CLI chat (c)`, `Edit profile (e)`, `Recommend (R)`, `Delete profile`
- `r` rescans in a worker; the table shows a "scanning" placeholder, never freezes.
- Highlighting a `New` model shows a one-line prompt in the pane: "No profile yet. Press R for a recommendation."

### 8.3 Profile editor (`screens/profile.py`)

Full screen. Left (3fr) scrollable `Collapsible` sections: Memory & offload, Sampling, Server & API, Extra.
Right (2fr) sticky `BudgetPanel` + command preview, recomputed on every change (debounced 150 ms).

`SettingField` widget per registry entry:

- INT/FLOAT: `Input` with `Integer`/`Number` validator and min/max; `+`/`-` keys step by `setting.step`.
- CHOICE: `Select`. BOOL: `Switch`. STR/PATH: `Input` (PATH validates existence on save, warning not error).
- `gpu_layers`: `Input` accepting `auto`, `all` or int, validated as such.
- Fields irrelevant to the model are disabled with a reason (e.g. `n_cpu_moe` on a dense model,
  `tensor_split` with one GPU).
- Below the input: the hint; below that, the `impact` line when the setting defines one
  (ctx_size, cache_type_k/v, gpu_layers, n_cpu_moe, parallel, batch sizes).
- Invalid fields are outlined red with the message; `Save` is disabled while any field is invalid.

Actions: `Save (ctrl+s)`, `Cancel (esc)`, `Apply recommended (R)` (fills fields, shows reasons inline as a
highlighted impact line, nothing saved), `Reset field to default (ctrl+r)`, `Help (?)`.
The same screen edits the default profile (title "Default profile", no model metadata, impact lines use the
host only).

### 8.4 Recommendation screen (`screens/recommend.py`)

Pushed after an HF download finishes and from `Recommend` on a model. Shows:
verdict + `BudgetPanel`, then a table `Setting | Current | Recommended | Why` for the changed keys.
Buttons: `Accept and save`, `Open in editor` (opens ProfileEditor pre-filled), `Skip`.
Skip marks the model as "seen" in `~/.config/llamactl/seen.json` so the *New* badge clears.

### 8.5 Sessions view

- Table columns `Name | State | URL | Port | Slots | Uptime | Kind`. State badge styles:
  `starting` dim, `loading` yellow spinner, `ready` green, `down` grey, `crashed` red.
- Detail pane: last 60 log lines for managed sessions, `/props` JSON for external, both refreshed only on change
  (existing diff logic kept). Slots in use are read from `/slots` when `metrics`/slots endpoint is reachable;
  otherwise `-`.
- Actions: `Chat (t)`, `Stop (s)` with confirmation for managed sessions, `Attach URL (a)`, `Open log (o)`
  (shows full log in a scrollable modal), `Copy URL (y)`.
- Table rows are updated in place by row key; no clear/rebuild per tick.

### 8.6 HF view

Behaviour unchanged except: host detection off the UI thread; `Fits` uses `recommend.estimate_budget`
with a size-only `ModelInfo` placeholder (KV unknown -> shown as "weights only"); after a successful download
the file is parsed and `RecommendationScreen` is pushed.

### 8.7 Settings view

Replaces the two `FormScreen` uses: bin dir (validated: `llama-server` must exist), model dirs (list editor:
add/remove rows), HF token (masked, same storage as today), gauge refresh interval, default headroom %.

### 8.8 Responsiveness rules

- Workers for: host detection, model discovery, GGUF parsing, HF requests, health probes, process spawn.
- `DataTable` updates by row key (`update_cell`/`add_row(key=)`), never `clear()` on a timer.
- Any computation in an `impact` callback must be O(1) arithmetic on cached `ModelInfo`.

## 9. Phases

Each phase lands as one or more Conventional Commits on `main` (repo initialised in phase 1). No pushing.

1. **Core** - `schema.py`, schema-driven `Profile`, `command.py`, `gguf.py`, `recommend.py`, `hostinfo`
   physical-core detection, pytest suite, `AGENTS.md` with run/test commands. Existing UI keeps working
   through the new builder (behaviour-preserving).
2. **Profile editor** - `SettingField`, `BudgetPanel`, `ProfileEditor`, `HelpPanel`; wire `e`/`d` to it.
3. **Shell + Models view** - sidebar, gauges, background refresh, detail pane; remove the tab layout.
4. **Sessions view** - states, in-place updates, log modal, confirm on stop.
5. **HF + recommendation flow** - `RecommendationScreen`, post-download hand-off, *New* badge, Settings view,
   delete `forms.py`.
6. **Chat** - deferred; not part of this spec.

## 10. Verification

- `pytest -q` green for every phase.
- `python -m llamactl` starts, each view reachable by key, no tracebacks in `textual console`.
- Textual `Pilot` smoke tests: open editor, type an invalid ctx, assert Save disabled; apply recommended;
  save and reload config round-trips.
- Manual: launch the Q8 9B model with a recommended profile on the 16 GB GPU and confirm `ready`.
