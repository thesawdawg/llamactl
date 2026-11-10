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
- [ ] Models view with per-intent detail
- [ ] IntentPicker, PlanScreen, PresetEditor
- [ ] TuneScreen
- [x] Serving view (gateway + standalone + attached)
- [ ] Settings view: gateway and default intent

## Phase 7 - Onboarding and Discover

- [ ] Onboarding flow
- [ ] Discover: fit-aware file lists, post-download intent picker

## Phase 8 - Release prep

- [ ] README rewrite, CHANGELOG, AGENTS.md
- [ ] Version 0.2.0 (no tag, no publish)
