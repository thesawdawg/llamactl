# TUI experience review

Assessment date: 2026-11-12. Branch: `redesign-v2`.
Scope: review existing implementation and interview the owner before changing the UI.
No application behavior changes are included.

## App model

llamactl is a terminal-only server manager. Its shared core supports a CLI and five
Textual views:

| View | Current role |
| --- | --- |
| Home | Hardware, gateway endpoint, loaded models, suggested next steps |
| Models | Library, per-purpose plans and presets, edit/tune/serve/default selection |
| Serving | Gateway models, standalone and attached servers, resources, logs, connection snippets |
| Discover | Hugging Face search, file fit estimates, downloads, post-download setup |
| Settings | Paths, planning defaults, gateway configuration, display, credentials, setup |

The main model workflow is choose purpose → estimate a plan → optionally edit or
measure settings → save a preset → serve through the gateway or a standalone server.
A model can have several presets. Its active preset gets the plain client alias;
other presets get `alias@preset`. Active therefore describes client naming/default
selection, independently of whether a model is currently loaded.

Settings exist at several scopes: application/planner defaults, a model's saved
preset, gateway settings, and the settings captured by an existing running process.
The UI needs to explain these distinctions at the point of action.

## Findings and priorities

### 1. Models combines values from different configurations

**Priority:** high. **Evidence:** `modelrows.intent_rows` reads context, slots, cache,
fit and memory from a fresh plan but source, active state and measurements from a
saved preset. `ModelsView._show_intent` displays the plan budget and warnings while
using the saved preset for its command preview when one exists.

An edited/tuned preset can consequently have different settings from the figures
displayed beside it. This is a verified source mismatch, with the visible discrepancy
depending on the saved settings.

**Proposal:** show saved settings as the primary model/preset summary. Label proposed
values separately and identify measured values with their configuration and freshness.
**Effort:** medium; establish shared configuration summaries before rearranging views.

### 2. Tuning commits before the final choice

**Priority:** high. **Evidence:** `TuneScreen._searched` calls `actions.apply_tune`,
then reloads a running gateway, before presenting final buttons. “Keep as is” only
closes the screen; “Use (make active)” changes the default client alias. Successful
tuning replaces the purpose's preset; cancelling can still retain a successful
winner from completed trials. The search begins from the default profile and purpose,
rather than treating all current preset edits as constraints (`actions.search_tune`).

**Proposal:** explicitly describe the starting configuration, what tuning may change,
and what it saves. Prefer reviewing a candidate against the saved preset before
replacement, subject to the owner's preference. Retain measurements independently.
**Effort:** medium/high; affects result persistence as well as button text.

### 3. Saving can affect running gateway models

**Priority:** high. **Evidence:** editor callbacks call `LlamaCtl.presets_changed`,
which rewrites presets and reloads the gateway. `Gateway.reload` documents that the
router unloads running models whose presets changed or vanished. Activation and
deletion use this same path. Standalone processes retain their launch settings.

**Proposal:** before applying a change, show target model/preset, changed fields,
affected served names/processes, and whether interruption or restart is involved.
Distinguish save, make default for clients, and apply to a running process.
**Effort:** medium/high; needs a reliable action-impact summary and runtime comparison.
**Limit:** actual llama.cpp unload/request behavior was not exercised in this review.

### 4. Navigation also performs operational actions

**Priority:** medium. **Evidence:** Enter on a model serves it; with exactly one
preset it does so immediately. With multiple presets it opens a picker. Enter on a
purpose row serves that purpose. Edit/tune/plan target the highlighted purpose,
whereas standalone serving uses the active preset when presets already exist.
The same `s` key means standalone in Models and stop in Serving.

**Proposal:** Enter opens a model workspace with an explicit selected preset and
visible actions. Keep keyboard shortcuts, but distinguish navigation from launch.
**Effort:** medium; owner preference needed before changing established navigation.

### 5. Change previews lack the most useful baseline

**Priority:** medium. **Evidence:** PlanScreen compares planned values with the global
default profile, not the existing preset. PresetEditor provides live impacts, reasons,
memory budget and command preview, but no consolidated saved → draft comparison.
Router-owned fields remain editable without serving-mode-specific scope labels.

**Proposal:** a review panel with Setting / Saved / Proposed / Why, estimated resource
changes, and scope/timing labels. Show gateway-controlled network settings in their
appropriate context; label standalone-only fields explicitly.
**Effort:** medium; reuse the existing schema, reasons and budget widgets.

### 6. Settings mixes several application timings

**Priority:** medium. **Evidence:** Settings saves update planning/display/configuration
together, reload gateway presets, and report certain restart requirements afterward.
The gauge interval explicitly takes effect on the next app start. Global default-profile
editing recomputes plans without replacing saved model presets. Changing gateway bind,
port, API key or capacity leaves the running gateway on its old values until restart.

**Proposal:** group by scope and show “future plans”, “saved preset”, “next server
start”, or “restart gateway” before saving. Keep pending-restart status visible.
**Effort:** small/medium once scope metadata exists.

## Confirmed direction from owner interview

The owner confirmed these requirements:

- Selecting a model opens its controls; selection must not load it into the gateway.
- Use a guided flow with optional advanced views for low-level and infrequent settings.
- Present tuning results against the saved configuration before any configuration change.
  Acceptance must explain exactly what changes; keeping the current configuration must
  actually preserve it.
- Allow per-model context-window profile defaults: a fixed global purpose target may
  be too large for one model and too small for another.
- Store tuning goals separately for each model/purpose preset: minimum tokens per
  second and maximum context per request. Both can be set together.
- Increase context toward the preset's maximum until measured throughput falls
  below its minimum, then reduce that failing context by 5% and measure again.
  Repeat until the candidate meets the floor, or no acceptable candidate remains.
- Offer prompt-processing and generation throughput controls and display both.
  Use effective output throughput (generated tokens / whole request wall time,
  including input processing) as the main metric, with a default minimum of
  20 tokens/s per preset. Individual input/output floors are optional.
- Offer three benchmark levels per preset: Quick (~2k input tokens), Normal (Quick
  plus ~25k expected harness starting input), and Full (5%, 25%, 50%, 75%, 95% context
  occupancy). The final 5% is reserved for the owner's harness compaction workflow.
- Accepting a candidate saves it for later. A running instance keeps its launch
  configuration; accepted settings take effect automatically on its next load
  after it stops/unloads. Saving must not force a model restart/unload.

The two reported friction examples were accepting versus keeping a tuning result
without understanding the changes, and clicking a model expecting configuration
but triggering a gateway load.

### Proposed guided flow

1. **Open model:** current saved preset, running state, client name and purpose.
2. **Choose goal:** purpose, minimum tokens/s and maximum context per request; offer
   inherited defaults and show the model's reported context limit. Advanced controls
   remain available.
3. **Review plan / run tuning:** explain what will be searched, constraints and
   resource impact; tuning collects candidates without replacing the saved preset.
4. **Compare:** saved versus candidate context per request, parallel requests,
   GPU/RAM placement, cache settings and measured performance. Label estimates and
   missing measurements. Explain why a candidate is recommended and any tradeoff.
5. **Accept or keep current:** explicit actions to save a candidate or leave the
   preset unchanged. Default client selection and serving are separately labelled
   decisions. Acceptance queues the saved settings for the next load without
   interrupting a running instance. Show a persistent “Saved; applies on next load”
   status and allow comparison with the running configuration.

### Context goals: verified implementation gap

`intents.INTENTS` defines global minimum/target/maximum context values by purpose;
zero means the model's trained context. `planner.ctx_bounds` resolves the target
against that limit, and `planner.ctx_ceiling` resolves the maximum. Tuning can
already step down and probe larger contexts, but its bounds and scoring derive
from those global purpose definitions. `actions.search_tune` starts from the
default profile, so editing a saved preset's context does not redefine its tuning
goal. Config has no per-model purpose context-goal override.

Proposed addition: persistent context-goal overrides distinct from the concrete
launch context. Resolve them consistently across planning, tuning, comparisons and
CLI operations. Preserve existing behavior when overrides are absent. Context values
must be labelled per request/slot, alongside total allocation for parallel slots.

The owner chose per-preset overrides with two simultaneous constraints: a throughput
floor and a context ceiling. The objective is the largest verified context within
both constraints, subject to memory/headroom and model limits. If the ceiling is
reached while speed is acceptable, tuning should finish without forcing a threshold
failure. If no candidate meets the speed floor, report that the goal was not met and
leave the saved configuration intact. If the trial budget runs out before the search
finishes, label the result as the best tested candidate, not a proven maximum.

The 5% context backoff must be distinct from the existing 5% speed tolerance in
`tuner.choose_best`: the new minimum throughput is a constraint, not an allowance
to select a result 5% below the requested speed.

### Benchmark and deferred-application implications

Current generation trials use an approximately 2k-token benchmark prompt capped by
half the context, and generate 128 output tokens (`bench_prompt`,
`ServerRunner._complete`). Allocating a large context is different from measuring
throughput while using most of that context. The comparison must state prompt size,
output size and speed metric. Full-window tests need room for generated output as
well as input. The owner has chosen the following levels:

| Level | Requested test workloads |
| --- | --- |
| Quick | Approximately 2k input tokens |
| Normal | Quick plus approximately 25k input tokens, representing harness startup |
| Full | 5%, 25%, 50%, 75%, 95% of each candidate's context window |

Every requested workload must be tied to the candidate context, not the originally
requested maximum, when testing a reduced candidate. Proposed rules: a candidate
passes only when all required workloads meet every enabled throughput floor and
memory/headroom checks; show each workload's result and the limiting workload. Do
not conceal a slow high-occupancy test inside an average across occupancy levels.
The owner requested an average TPS metric for a request, not an average that hides
individual failing workload results.

Use actual tokenizer counts for occupancy, including special tokens, and leave room
for benchmark output. Preserve the 5% compaction reserve in the Full test's highest
occupancy workload; input plus planned output must fit within that boundary. This
reserve models the owner's harness policy; llamactl does not perform compaction.
Do not silently truncate a 25k Normal workload for a small context and label it as
25k-tested: identify a reduced/unavailable workload explicitly in the result.

Benchmark output length, caching policy, actual token counts and elapsed durations
must be recorded and shown with results. Comparing throughput requires a consistent
output length and declared caching behavior. Existing trials disable prompt caching
and generate up to 128 tokens; short output budgets materially affect a combined
request metric when prompt processing is included. Neither Quick nor Normal should
be described as validating speed throughout a much larger context window.

### Confirmed throughput terminology

Verified against upstream llama.cpp `server_slot_stats::to_json`:
[server-common.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-common.cpp).
It exposes `prompt_n`, `prompt_ms`, `prompt_per_second`, `predicted_n`,
`predicted_ms`, and `predicted_per_second`, plus cached prompt counts. Its timing
object does not provide a single combined user-experience throughput value.

Proposed UI terminology:

- **Input processing:** llama.cpp `prompt_per_second`.
- **Output generation:** llama.cpp `predicted_per_second`.
- **Effective output speed:** generated tokens / measured request wall time, including
  prompt processing; this is a derived llamactl metric, not a native llama.cpp field.
- **Total processing throughput:** (processed input tokens + generated tokens) /
  ((prompt_ms + predicted_ms) / 1000); this is also derived and can mask slow output
  or a long initial wait when prompt throughput is high.

The owner selected **Effective output speed** as the main metric, with a default
20 TPS minimum per preset. Measure request wall time from sending the benchmark
request until its complete response is received. Include prompt processing and
request overhead; exclude model startup and separate tokenizer/setup requests.
Divide the actual generated token count by that duration, not the requested output
budget. Individual input-processing and output-generation floors remain available
as optional advanced constraints. Total processing throughput is not the primary
metric and need not be exposed in the guided flow.

Do not compute the arithmetic mean of input and output TPS. For repeated measurements
of the same workload, calculate aggregate effective output TPS as total generated
tokens / total request duration. Keep distinct workload results separate and require
each to meet enabled floors; do not average across workloads to hide failures.
Record the definition with stored results, and avoid comparing incompatible benchmark
metrics. Show request duration and, where measured, time to first output as supporting
data. The current non-streaming runner does not measure time to first output.

Current `Gateway.reload` can unload changed running presets, so removing the reload
only from the accept button is insufficient: a later unrelated reload could still
apply pending settings. The implementation needs separate saved and effective
runtime configurations. Preserve the old gateway definition for each running preset
while it has pending accepted changes, then promote those changes after that preset
is unloaded, before its next load (including router on-demand loads). Standalone
servers keep their captured launch settings until stopped and launched again.
Pending state must survive app restart; concurrent loads, reloads and unloads need
consistent handling. Sleeping is not automatically equivalent to unloaded.

## Suggested navigation structure

Retain the five top-level destinations initially. Let Models open a persistent model
workspace with Overview, Presets/Settings, Optimize and Serving sections. Always show
model, selected preset, default client name and runtime state. Keep advanced llama.cpp
controls and command preview available behind secondary sections.

A review panel should answer: what changes, why, which model/preset/client is affected,
when it takes effect, and whether running work is interrupted. Routine browsing should
remain quick; disruptive changes deserve explicit review.

Implementation order after interview:

1. Correct mixed plan/preset summaries and establish shared scope/configuration data.
2. Add saved/draft/running comparisons and action-impact summaries.
3. Implement the agreed navigation and tuning acceptance behavior.
4. Validate keyboard navigation and small terminal layouts using existing Pilot fixtures.

## Owner interview

Initial questions submitted in the conversation:

1. Which confusion dominates: action scope, navigation, or optimization tradeoffs?
2. What are the two most frequent real workflows?
3. Prefer a model workspace, clearer existing tables, or a guided sequence?

The owner answered the earlier questions: reduce failing context by 5% and retest;
offer input/output speed plus a main combined TPS with default floor 20; offer Quick,
Normal and Full benchmark workloads as described above.

The owner selected effective output TPS (generated tokens / whole request duration)
as the main 20 TPS floor. The tuning metric and benchmark-level decisions are resolved.

Terminal dimensions, keyboard/mouse usage and gateway/standalone frequency can be
resolved later when they affect concrete layout or interaction choices.

## Coverage

Reviewed shell/navigation, all five views, planner/editor/tuning screens, shared
actions, preset summaries, gateway reload path, specification and relevant UI tests.
The model-row tests pass independently: 9 passed in 0.11 seconds.
The combined focused test run produced ten passing test markers, then stopped
making progress in the UI tests and was interrupted. The UI test run is incomplete;
it does not establish a UI regression or a passing UI suite.
No manual live TUI session, model benchmark, network request to a real server, or
llama.cpp restart was performed. Visual preferences and real-world workflow frequency
remain owner questions; proposed usability benefits are hypotheses until validated.
