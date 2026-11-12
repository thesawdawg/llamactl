# llamactl v2 progress

Tracks [SPEC.md](SPEC.md) section 11. Tick an item in the same commit that completes it.

## Phase 0 - Plan

- [x] Agree direction with the owner (tuning, serving, presets, scope)
- [x] Write `docs/SPEC.md` (v2) and archive v1
- [x] Create this todo list

## Phase 1 - Foundations

- [x] Remove `screens/chat.py`, the `cli` launch mode and their bindings/tests
- [x] `hardware.py`: `Device`, `Host`, `--list-devices` parser, nvidia-smi fallback, refresh; delete `hostinfo.py`
- [x] Schema: drop `modes`; add fit, fit_target, device, cache_reuse, sleep_idle, embedding, pooling, threads_batch; `preset_key`, `router_owned`
- [x] GGUF: detect pooling type (embedding models)
- [x] Config v3: `Preset`, per-model presets/alias/active, gateway settings, v2 migration with backup
- [x] `library.py`: discovery + alias/substring model resolution

## Phase 2 - Planner

- [x] `intents.py`: six intents as data, `applies` rules
- [x] `planner.py`: budget, `plan`, `plan_all`, warnings, multi-GPU split, unified memory
- [x] Calibration factor applied to estimates (read path)
- [x] Delete `recommend.py`; port its tests

## Phase 3 - Gateway

- [x] `command.py`: preset key/value rendering from the registry
- [x] `gateway.render_presets` (alias and alias@preset sections)
- [x] `Gateway` process control + router API client, state in `gateway.json`
- [x] `sessions.launch_server` takes a preset
- [x] Manual: real router accepts the generated INI

## Phase 4 - Tuner

- [x] `Measurement`, `Runner` protocol, `ServerRunner`
- [x] Search (step down / step up / threads trial) and scoring
- [x] `calibration.py` store, stale detection
- [x] Manual: tune a local model end to end

## Phase 5 - CLI

- [x] `cli.py` dispatcher; no args opens the TUI
- [x] `hw`, `ls`, `plan` (with `--json`)
- [x] `tune`, `serve`, `up`, `down`, `ps`, `load`, `unload`, `stop`, `logs`

## Phase 6 - TUI

- [x] Shell with five views; host polling through `hardware`
- [x] Home view
- [x] Models view with per-intent detail
- [x] IntentPicker, PlanScreen, PresetEditor
- [x] TuneScreen
- [x] Serving view (gateway + standalone + attached)
- [x] Settings view: gateway and default intent

## Phase 7 - Onboarding and Discover

- [x] Onboarding flow
- [x] Discover: fit-aware file lists, post-download intent picker

## Phase 8 - Release prep

- [x] README rewrite, CHANGELOG, AGENTS.md
- [x] Version 0.2.0 (no tag, no publish)

## Phase 9 - Network access

- [x] `network.py`: bind modes, LAN addresses, client URLs, API key, WSL mode and notes
- [x] Client URLs everywhere an endpoint is shown (Home, Serving, onboarding, `up`, `serve`, `ps`, curl)
- [x] Exposed-without-key warning when starting the gateway or a standalone server
- [x] Settings: "Listen on" select, URL line, Generate key; PresetEditor host choices
- [x] Onboarding: "Reachable from other machines" option
- [x] CLI: `--lan` / `--host` on `up` and `serve`; `llamactl config`
- [x] README, CHANGELOG (0.2.0) and AGENTS.md updated
- [x] Manual: gateway on `0.0.0.0` answers on the LAN address

## Phase 10 - Resource usage

- [x] `usage.py`: /proc and `ps` process probing, `nvidia-smi` per-process memory, VRAM
  fallback order process -> tuned -> estimate, `~` markers, sleeping models never estimated
- [x] `ps` (text and `--json`) shows CPU/RAM/VRAM per gateway model and server
- [x] Serving view CPU/RAM/VRAM columns, CPU deltas between polls

## Phase 11 - Tuner: largest full-GPU context

- [x] All-GPU trial when the plan split a model whose weights fit VRAM whole
- [x] Headroom guard: a loaded GPU trial only counts when it keeps the headroom free
  (Windows/WSL driver sysmem fallback), and it stops the fill
- [x] Context fill probes the largest full-GPU context from the measured bytes-per-token
  slope, in 4096 steps up to `ctx_max`
- [x] Scoring: context beyond the target wins unless it costs more than 5% speed
- [x] Realistic benchmark prompt (~2k tokens), `--trials` default 6

## Phase 12 - MoE safety (SPEC 12.2)

- [x] `command.py`: force `-fit off` in argv and INI when `n_cpu_moe > 0` or `override_tensor` is set
- [x] Tuner headroom guard for every `gpu_layers == "all"` trial, not only verdict `GPU`
- [x] Spill check: `gen_tps` below a third of the family's best fails `fits` (`spilled to system RAM?`)
- [x] UI: PresetEditor Auto-fit conflict warning; preview shows the forced `-fit off`
- [x] UI: TuneScreen outcome names guarded trials (spilled / headroom) with numbers
- [ ] Manual: CPU MoE layers 5 with Auto-fit on shows the warning and `-fit off` in the preview

## Phase 13 - MoE placement in the planner (SPEC 12.3)

- [ ] `gguf.py`: tensor table across shards, sizes from offsets and alignment; `expert_bytes`, `dense_bytes`, `expert_used`; cache gate
- [ ] Planner: GPU part from `dense_bytes`; `_moe_layers` from `expert_bytes`; RAM check and swap warning; `Plan.expert_ram`, `Plan.layer_gain`; fallback to the average
- [ ] UI: BudgetPanel splits GPU part and experts in RAM (never red for a valid MoE plan)
- [ ] UI: Models meta MoE line and `GPU+experts` verdict cell
- [ ] UI: PlanScreen "Expert placement" block; `plan` text and `--json` fields
- [ ] Manual: Qwen3.6-35B-A3B plan loads first time with headroom free; numbers readable

## Phase 14 - MoE search in the tuner (SPEC 12.4)

- [ ] `down` raises `n_cpu_moe` from the overflow; falls through at `n_layer`
- [ ] `moe_fill` predicts the lowest fitting `n_cpu_moe` from measured slack
- [ ] Scoring drops trials >30% slower on prompt; threads trial sets `threads_batch`; +3 trials for MoE
- [ ] UI: TuneScreen `CPU MoE` column, plain change labels, summary explains the winner
- [ ] UI: Home next step for estimated MoE presets with experts in RAM; Models `pp / tg` for MoE
- [ ] Manual: tune Qwen3.6-35B-A3B (agent); `CPU MoE` drops over trials, summary explains the pick

## Phase 15 - Manual placement and host hints (SPEC 12.5)

- [ ] Schema `override_tensor` (`-ot`) emitted before `--n-cpu-moe`; tune refuses presets that set it
- [ ] Config `wsl_windows_ram`; start-up modal when WSL is detected and it was never asked; onboarding Hardware field; Settings field
- [ ] `hardware.wsl_advice` (suggested `memory=`, gain, threshold)
- [ ] UI: PresetEditor `-ot` examples, validation and tensor match preview
- [ ] UI: WSL advice with Copy snippet on Home and onboarding; `llamactl hw` prints it
- [ ] UI: Help page "Running MoE models on limited VRAM"
- [ ] Manual: first start under WSL asks for Windows RAM once; 32 GiB gives `memory=28GB`
