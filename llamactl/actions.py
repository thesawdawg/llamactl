"""Multi-step operations shared by the CLI and the TUI.

Everything here sits between the pure core (planner, tuner, gateway) and a front end: choosing
or creating the preset to serve, getting a model loaded in the gateway, waiting for it, and
running a tune end to end. No Textual, no printing; waiting takes injectable `clock` / `sleep`
functions and a `tick` callback so a front end can show progress and tests never sleep.
"""
from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import intents, network, planner, tuner
from .calibration import Calibration
from .config import ESTIMATED, PRESET_SEP, Config, Preset
from .gateway import (LOADED, LOADING, SLEEPING, Gateway, GatewayError, GatewayModel,
                      GatewayState, log_reason)
from .gguf import ModelInfo
from .goals import TuningGoals
from .hardware import Host
from .intents import Intent
from .planner import Plan
from .sessions import Session
from .support import ArchSupport

GATEWAY_START_TIMEOUT = 30.0     # seconds for a started router to answer /health
POLL_INTERVAL = 1.0              # seconds between checks while waiting
UNLOADED_GRACE = 10.0            # a model the router still lists as unloaded this long after load failed

Clock = Callable[[], float]
Sleep = Callable[[float], None]
Tick = Callable[[], None]


class ActionError(RuntimeError):
    """An operation was refused or failed in a way the user can act on (message is for them)."""


# --------------------------------------------------------------------------- intents and presets

def choose_intent(key: str | None, default: str, info: ModelInfo) -> Intent:
    """The intent to use for a model.

    Args:
        key: Intent the user asked for, or None for "whatever fits the default".
        default: The configured default intent key.
        info: Model metadata.

    Returns:
        With a `key`, exactly that intent; without one, `intents.resolve(default, info)`.

    Raises:
        LookupError: When `key` names no intent, or an intent that does not apply to the model
            (for instance ``embed`` for a text model).
    """
    if key is None:
        return intents.resolve(default, info)
    intent = intents.by_key.get(key)
    options = intents.applicable(info)
    if intent is None or intent not in options:
        what = "no such intent" if intent is None else "does not apply to this model"
        raise LookupError(f"intent '{key}': {what} (use one of: "
                          f"{', '.join(i.key for i in options)})")
    return intent


def plan_for(cfg: Config, info: ModelInfo, host: Host, intent: Intent,
             calibration: Calibration | None = None,
             support: ArchSupport | None = None, *, goals: TuningGoals | None = None) -> Plan:
    """Plan an intent with the config's default profile, headroom and the measured calibration.

    Args:
        cfg: Config supplying the base profile and headroom.
        info: Model metadata.
        host: Host snapshot.
        intent: What the model is for.
        calibration: Measured data, or None.
        support: The architecture support check of the model; a blocking one gives
            ``VERDICT_UNSUPPORTED``.
        goals: Unsaved goals to preview; otherwise use the preset's stored goals.

    Returns:
        The plan.
    """
    resolved = tuning_goals(cfg, str(info.path), intent.key, goals)
    return planner.plan(resolved.resolve_intent(intent), info, host, cfg.default_profile, cfg.headroom_pct, calibration,
                        support)


def tuning_goals(cfg: Config, model: str, key: str,
                 goals: TuningGoals | None = None) -> TuningGoals:
    """Resolve draft, saved, or inherited tuning constraints.

    Args:
        cfg: Configuration holding model presets.
        model: Model file path.
        key: Preset name.
        goals: Explicit draft goals, if provided.

    Returns:
        Validated goals without mutating configuration.
    """
    preset = cfg.preset(model, key)
    return goals if goals is not None else preset.goals if preset else TuningGoals()


def store_plan(cfg: Config, model: str, plan: Plan, *, activate: bool = False,
               overwrite: bool = False, save: bool = True) -> Preset:
    """Store a plan as the model's ``estimated`` preset named after its intent.

    A preset of that name that is tuned or hand-edited is kept unless `overwrite` is set; an
    existing estimated one is simply replaced.

    Args:
        cfg: Config to update.
        model: Model path (the key of ``cfg.models``).
        plan: The plan to store.
        activate: Make it the model's active preset (it becomes active anyway when the model
            has none).
        overwrite: Replace a tuned or edited preset too.
        save: Write the config to disk.

    Returns:
        The stored preset.

    Raises:
        ActionError: When a tuned or edited preset of that name exists and `overwrite` is off.
        OSError: When the config cannot be written.
    """
    old = cfg.preset(model, plan.intent)
    if old is not None and old.source != ESTIMATED and not overwrite:
        raise ActionError(f"the '{plan.intent}' preset is {old.source}; it would be replaced "
                          "by an estimate (use --force to do that)")
    preset = Preset(plan.intent, plan.profile, ESTIMATED,
                    goals=old.goals if old else TuningGoals())
    if save:
        save_preset(cfg, model, preset, activate=activate)
    else:
        cfg.set_preset(model, preset, activate=activate)
    return preset


def save_preset(cfg: Config, model: str, preset: Preset, *, activate: bool = False) -> None:
    """Persist an accepted preset, rolling back memory if the write fails.

    Args:
        cfg: Configuration to update.
        model: Model file path.
        preset: Explicitly accepted settings and goals.
        activate: Also select it as the default for clients.

    Raises:
        OSError: If the configuration cannot be saved.
    """
    previous = deepcopy(cfg.models.get(model))
    cfg.set_preset(model, preset, activate=activate)
    try:
        cfg.save()
    except OSError:
        if previous is None:
            cfg.models.pop(model, None)
        else:
            cfg.models[model] = previous
        raise


@dataclass(frozen=True)
class Chosen:
    """The preset a serve will use.

    Attributes:
        name: Preset name (the key in ``ModelEntry.presets``).
        preset: The preset.
        created: It did not exist and was planned just now (stored as ``estimated``).
    """

    name: str
    preset: Preset
    created: bool


def pick_preset(cfg: Config, info: ModelInfo, host: Host, calibration: Calibration | None,
                intent_key: str | None = None, *, activate: bool = True,
                save: bool = True) -> Chosen:
    """The preset to serve, planning and storing an estimated one when it is missing.

    With an explicit `intent_key` that intent's preset is used (made active when `activate`).
    Without one, the model's active preset is used as it is; a model with none gets a preset
    for the default intent, which becomes its active preset. An existing preset is never
    re-planned.

    Args:
        cfg: Config to update.
        info: Model metadata; ``info.path`` is the model.
        host: Host snapshot.
        calibration: Measured data for the plan, or None.
        intent_key: Intent asked for explicitly, or None.
        activate: With an explicit intent: make its preset the active one.
        save: Write the config when something changed.

    Returns:
        The chosen preset.

    Raises:
        LookupError: When `intent_key` is unknown or does not apply to the model.
        OSError: When the config cannot be written.
    """
    model = str(info.path)
    if intent_key is None:
        active = cfg.preset(model)
        if active is not None:
            return Chosen(cfg.models[model].active, active, False)
    intent = choose_intent(intent_key, cfg.default_intent, info)
    keep_active = intent_key is not None and not activate
    existing = cfg.preset(model, intent.key)
    if existing is not None:
        if not keep_active and cfg.models[model].active != intent.key:
            cfg.models[model].active = intent.key
            if save:
                cfg.save()
        return Chosen(intent.key, existing, False)
    preset = store_plan(cfg, model, plan_for(cfg, info, host, intent, calibration),
                        activate=not keep_active, save=save)
    return Chosen(intent.key, preset, True)


def served_name(cfg: Config, model: str, preset_name: str) -> str:
    """The model id clients request for a preset: ``alias`` for the active one, else ``alias@name``.

    Args:
        cfg: Config holding the entry.
        model: Model path.
        preset_name: Preset name.

    Returns:
        The id as the gateway lists it.
    """
    ent = cfg.entry(model)
    return ent.alias if preset_name == ent.active else f"{ent.alias}{PRESET_SEP}{preset_name}"


# --------------------------------------------------------------------------- waiting

def _wait_until(done: Callable[[], bool], timeout: float, *, interval: float, clock: Clock,
                sleep: Sleep, tick: Tick | None) -> bool:
    """Poll `done` until it is true or `timeout` seconds passed.

    Args:
        done: Check; may raise to abort the wait.
        timeout: Seconds to wait.
        interval: Seconds between checks.
        clock: Monotonic clock.
        sleep: Sleep function.
        tick: Called once per pause (progress dots), or None.

    Returns:
        True when `done` returned true, False on timeout.
    """
    deadline = clock() + timeout
    while not done():
        if clock() >= deadline:
            return False
        if tick is not None:
            tick()
        sleep(interval)
    return True


def gateway_urls(gw: Gateway, host: str | None = None) -> list[str]:
    """The base URLs a client can use for the gateway, running or about to start.

    Args:
        gw: The gateway.
        host: Listen address it would start with, when that is not the configured one.

    Returns:
        `network.client_urls_for` the running gateway's address, else `host` or the
        configured one: loopback first, then one URL per LAN address for a wildcard bind.
    """
    st = gw.state
    if st is not None:
        return network.client_urls_for(st.url, st.host or gw.cfg.gateway.host)
    return network.client_urls_for(gw.listen_url, host or gw.cfg.gateway.host)


def bind_conflict(gw: Gateway, host: str) -> str:
    """Why a running gateway cannot be asked to listen on another address.

    Args:
        gw: The gateway.
        host: The listen address wanted.

    Returns:
        A message with the fix, or "" when the gateway is not running or listens there already.
    """
    st = gw.state
    if st is None:
        return ""
    have = st.host or gw.cfg.gateway.host
    if network.normalise(have) == network.normalise(host):
        return ""
    return (f"the gateway is already running and listens on {have}, not {host}; "
            "run `llamactl down` first, then start it again")


def ensure_gateway(gw: Gateway, *, timeout: float = GATEWAY_START_TIMEOUT,
                   clock: Clock = time.monotonic, sleep: Sleep = time.sleep,
                   tick: Tick | None = None, host: str | None = None,
                   learn: Callable[[Path], None] | None = None) -> tuple[GatewayState, bool]:
    """Make sure the gateway runs and answers ``/health``, starting it when needed.

    Args:
        gw: The gateway.
        timeout: Seconds a started router gets to become healthy.
        clock: Monotonic clock.
        sleep: Sleep function.
        tick: Progress callback, called once per pause.
        host: Listen address for this start only (never saved); None uses the configured one.
        learn: Called with the gateway's log file when it exits, so an
            ``unknown model architecture`` line can be remembered (or None).

    Returns:
        (state, started): `started` is False when it was already running.

    Raises:
        GatewayError: When it cannot be started, exits, or stays unhealthy past `timeout`, or
            when it already runs on a different address than `host`.
    """
    if host is not None:
        if problem := bind_conflict(gw, host):
            raise GatewayError(problem)
        gw.host_override = host
    started = gw.state is None
    st = gw.start()

    def healthy() -> bool:
        if not gw.running:
            if learn is not None:
                learn(Path(st.log))
            raise GatewayError(f"the gateway exited: {log_reason(Path(st.log))} (log: {st.log})")
        return gw.status().healthy

    if not _wait_until(healthy, timeout, interval=0.5, clock=clock, sleep=sleep, tick=tick):
        raise GatewayError(f"the gateway at {st.url} did not answer within {timeout:.0f} s "
                           f"(log: {st.log})")
    return st, started


def serve_gateway(cfg: Config, gw: Gateway, model: str, preset_name: str, *,
                  clock: Clock = time.monotonic, sleep: Sleep = time.sleep,
                  tick: Tick | None = None, host: str | None = None,
                  learn: Callable[[Path], None] | None = None) -> str:
    """Get a preset loading in the gateway: write the presets, start it if needed, reload, load.

    Returns as soon as the router accepted the load request; use `wait_loaded` to wait for it.
    A model the router already runs (loaded, loading or sleeping) is not loaded again.

    Args:
        cfg: Config holding the preset (already stored).
        gw: The gateway.
        model: Model path.
        preset_name: Preset to serve.
        clock: Monotonic clock.
        sleep: Sleep function.
        tick: Progress callback while the gateway starts.
        host: Listen address for this start only (see `ensure_gateway`).
        learn: Called with the gateway's log file when it exits (see `ensure_gateway`).

    Returns:
        The model id to request (``alias`` or ``alias@preset``).

    Raises:
        GatewayError: When the gateway cannot start, runs on another address than `host`, does
            not know the preset (the message lists presets that could not be written), or
            refuses the load.
    """
    name = served_name(cfg, model, preset_name)
    ensure_gateway(gw, clock=clock, sleep=sleep, tick=tick, host=host, learn=learn)
    models = gw.reload()
    target = next((m for m in models if m.name == name), None)
    if target is None:
        extra = f" ({'; '.join(gw.problems)})" if gw.problems else ""
        raise GatewayError(f"the gateway does not list '{name}' after reloading its presets"
                           f"{extra}")
    if target.state not in (LOADED, LOADING, SLEEPING):
        gw.load(name)
    return name


def wait_loaded(gw: Gateway, name: str, timeout: float, *, clock: Clock = time.monotonic,
                sleep: Sleep = time.sleep, tick: Tick | None = None,
                interval: float = POLL_INTERVAL,
                learn: Callable[[Path], None] | None = None) -> GatewayModel:
    """Wait until the router reports a model as loaded (or sleeping, which wakes on a request).

    Args:
        gw: The gateway.
        name: Model id.
        timeout: Seconds to wait.
        clock: Monotonic clock.
        sleep: Sleep function.
        tick: Progress callback, called once per pause.
        interval: Seconds between checks.
        learn: Called with the gateway's log file when the model fails, so an
            ``unknown model architecture`` line can be remembered (or None).

    Returns:
        The model as the router last reported it.

    Raises:
        GatewayError: When the model failed (the message has the exit code and the last error
            line of the gateway log), vanished, stayed unloaded, or `timeout` passed.
    """
    seen: list[GatewayModel] = []
    first = clock()

    def ready() -> bool:
        found = next((m for m in gw.models() if m.name == name), None)
        if found is None:
            raise GatewayError(f"the gateway no longer lists '{name}'")
        seen[:] = [found]
        if found.state == "failed":
            st = gw.state
            log = st.log if st else ""
            if learn is not None and log:
                learn(Path(log))
            code = f" (exit code {found.exit_code})" if found.exit_code is not None else ""
            why = log_reason(Path(log)) if log else "no log"
            raise GatewayError(f"'{name}' failed to load{code}: {why}"
                               + (f" (log: {log})" if log else ""))
        if found.state == "unloaded" and clock() - first >= UNLOADED_GRACE:
            raise GatewayError(f"the gateway reports '{name}' as unloaded after the load "
                               "request; see the gateway log")
        return found.state in (LOADED, SLEEPING)

    if not _wait_until(ready, timeout, interval=interval, clock=clock, sleep=sleep, tick=tick):
        raise GatewayError(f"'{name}' is still {seen[0].state if seen else 'loading'} after "
                           f"{timeout:.0f} s; it may yet come up (llamactl logs gateway -f)")
    return seen[0]


def wait_unloaded(gw: Gateway, name: str, timeout: float, *, clock: Clock = time.monotonic,
                  sleep: Sleep = time.sleep, tick: Tick | None = None,
                  interval: float = 0.5) -> bool:
    """Wait until the router no longer reports a model as loaded, loading or sleeping.

    The router answers an unload request before the model is gone.

    Args:
        gw: The gateway.
        name: Model id.
        timeout: Seconds to wait.
        clock: Monotonic clock.
        sleep: Sleep function.
        tick: Progress callback, called once per pause.
        interval: Seconds between checks.

    Returns:
        True when the model is down, False when `timeout` passed first.

    Raises:
        GatewayError: When the gateway cannot be asked.
    """
    def gone() -> bool:
        found = next((m for m in gw.models() if m.name == name), None)
        return found is None or found.state not in (LOADED, LOADING, SLEEPING)

    return _wait_until(gone, timeout, interval=interval, clock=clock, sleep=sleep, tick=tick)


def wait_server_ready(session: Session, health: Callable[[str], str], timeout: float, *,
                      clock: Clock = time.monotonic, sleep: Sleep = time.sleep,
                      tick: Tick | None = None, interval: float = 0.5,
                      learn: Callable[[Path], None] | None = None) -> None:
    """Wait until a standalone server answers ``/health`` with 200.

    Args:
        session: The server just launched.
        health: Returns "ready", "loading" or "down" for a base URL.
        timeout: Seconds to wait.
        clock: Monotonic clock.
        sleep: Sleep function.
        tick: Progress callback, called once per pause.
        interval: Seconds between checks.
        learn: Called with the server's log file when it exits, so an
            ``unknown model architecture`` line can be remembered (or None).

    Raises:
        ActionError: When the process exits (with the last error line of its log) or
            `timeout` passes.
    """
    def ready() -> bool:
        if not session.alive:
            if learn is not None:
                learn(Path(session.log))
            raise ActionError(f"the server exited: {log_reason(Path(session.log))} "
                              f"(log: {session.log})")
        return health(session.url) == "ready"

    if not _wait_until(ready, timeout, interval=interval, clock=clock, sleep=sleep, tick=tick):
        raise ActionError(f"{session.url} did not become ready within {timeout:.0f} s "
                          f"(log: {session.log})")


# --------------------------------------------------------------------------- tuning

@dataclass(frozen=True)
class TuneOutcome:
    """What `run_tune` did.

    Attributes:
        result: The tuner's result (trials are recorded and the winner saved already).
        reloaded: A running gateway was told to re-read the presets.
        reload_error: Why that failed, "" when it did not.
    """

    result: tuner.TuneResult
    reloaded: bool = False
    reload_error: str = ""


def tune_refusal(cfg: Config, model: str, intent_key: str) -> str | None:
    """Why a tune must not run for this preset: it places tensors by hand.

    The search places experts itself, so it refuses when the default profile or the
    model's existing preset for this intent sets ``override_tensor``. Unlike the
    free-memory preflight this cannot be forced past.

    Args:
        cfg: Config holding the default profile and presets.
        model: Model path.
        intent_key: Preset/intent to tune.

    Returns:
        The refusal text, or None.
    """
    ent = cfg.models.get(model)
    preset = ent.presets.get(intent_key) if ent else None
    if cfg.default_profile.override_tensor or (preset and preset.profile.override_tensor):
        return "remove the tensor overrides to tune; the search places experts itself"
    return None


def search_tune(cfg: Config, info: ModelInfo, host: Host, intent: Intent, runner: tuner.Runner,
                calibration: Calibration, *, trials: int = 6, force: bool = False,
                progress: Callable[[tuner.TrialEvent], None] | None = None,
                cancel: Callable[[], bool] | None = None,
                goals: TuningGoals | None = None) -> tuner.TuneResult:
    """Preflight and search: the part of a tune that needs no write access to the config.

    The saved preset supplies the base settings when present. The search returns
    candidates without changing that preset.

    Args:
        cfg: Config supplying the base profile and headroom.
        info: Model metadata.
        host: Host snapshot with current free memory.
        intent: What to tune for.
        runner: Launches and measures candidates.
        calibration: Corrects the estimates (read only here).
        trials: Most launches to make.
        force: Skip the preflight memory check.
        progress: Called with every trial event.
        cancel: Polled between trials and by the runner during one; True stops the search.
        goals: Draft goals or None to use the saved preset's constraints.

    Returns:
        The tuner's result (nothing is saved).

    Raises:
        ActionError: When the preflight refuses (and `force` is off).
    """
    resolved = tuning_goals(cfg, str(info.path), intent.key, goals)
    preset = cfg.preset(str(info.path), intent.key)
    base = preset.profile if preset else cfg.default_profile
    plan = planner.plan(resolved.resolve_intent(intent), info, host, base,
                        cfg.headroom_pct, calibration)
    if not force and (refusal := tuner.preflight(plan, host)):
        raise ActionError(refusal)
    return tuner.tune(intent, info, host, base, runner,
                      headroom_pct=cfg.headroom_pct, calibration=calibration,
                      max_trials=trials, progress=progress, cancel=cancel, goals=resolved)


def apply_tune(cfg: Config, info: ModelInfo, result: tuner.TuneResult,
               calibration: Calibration, *, goals: TuningGoals | None = None,
               record: bool = True) -> None:
    """Record a tune's trials and store its winner as the model's tuned preset; save both files.

    This is the step that writes ``tuning.json``, mutates the config and saves it. A front end
    with a UI thread runs it there, never on the worker that searched.

    Args:
        cfg: Config to update and save.
        info: Model metadata.
        result: What `search_tune` returned.
        calibration: Store receiving the trials.
        goals: Accepted draft goals or None to preserve stored goals.
        record: Record trials unless the review screen already recorded them.

    Raises:
        OSError: When tuning.json or the config cannot be written.
    """
    if record:
        record_tune(result, calibration)
    if result.ok and not result.cancelled and result.best is not None and result.measurement is not None:
        accepted = tuning_goals(cfg, str(info.path), result.intent, goals)
        save_preset(cfg, str(info.path), Preset(result.intent, result.best.profile, "tuned",
                                               result.measurement, accepted))


def record_tune(result: tuner.TuneResult, calibration: Calibration) -> None:
    """Retain benchmark history without accepting a candidate configuration.

    Args:
        result: Completed or cancelled search.
        calibration: Trial history to update and persist.

    Raises:
        OSError: If benchmark history cannot be saved.
    """
    if result.trials:
        for trial in result.trials:
            calibration.record(trial)
        calibration.save()


def reload_after_tune(gw: Gateway, result: tuner.TuneResult) -> TuneOutcome:
    """Tell a running gateway to re-read the presets after a successful tune.

    Args:
        gw: The gateway.
        result: The tune's result.

    Returns:
        The outcome; ``reloaded`` or ``reload_error`` is set when a reload was attempted.
    """
    if not result.ok or not gw.running:
        return TuneOutcome(result)
    try:
        gw.reload()
    except GatewayError as e:
        return TuneOutcome(result, False, str(e))
    return TuneOutcome(result, True)


def run_tune(cfg: Config, info: ModelInfo, host: Host, intent: Intent, runner: tuner.Runner,
             calibration: Calibration, gw: Gateway, *, trials: int = 6, force: bool = False,
             progress: Callable[[tuner.TrialEvent], None] | None = None,
             cancel: Callable[[], bool] | None = None, save: bool = True,
             goals: TuningGoals | None = None) -> TuneOutcome:
    """Tune a model for an intent end to end: preflight, search, save, reload the gateway.

    Trials are recorded in ``tuning.json`` even when the tune failed or was cancelled; a
    successful winner is stored as the model's tuned preset; a running gateway then re-reads
    the presets. With ``save=False`` only the search runs (`search_tune`): the caller applies
    the result itself, on the right thread, with `apply_tune` and `reload_after_tune`.

    Args:
        cfg: Config to update and save.
        info: Model metadata.
        host: Host snapshot with current free memory.
        intent: What to tune for.
        runner: Launches and measures candidates.
        calibration: Store receiving the trials; also corrects the estimates.
        gw: The gateway, reloaded when running.
        trials: Most launches to make.
        force: Skip the preflight memory check.
        progress: Called with every trial event.
        cancel: Polled between trials and by the runner during one; True stops the search.
        save: Save the result and reload the gateway (the CLI); off for the TUI worker.
        goals: Draft constraints or None for the stored preset's goals.

    Returns:
        The outcome.

    Raises:
        ActionError: When the preflight refuses (and `force` is off).
        OSError: When tuning.json or the config cannot be written (only with `save`).
    """
    result = search_tune(cfg, info, host, intent, runner, calibration, trials=trials,
                         force=force, progress=progress, cancel=cancel, goals=goals)
    if not save:
        return TuneOutcome(result)
    apply_tune(cfg, info, result, calibration, goals=goals)
    return reload_after_tune(gw, result)


# --------------------------------------------------------------------------- hints

def count_presets(ini: Path) -> int:
    """Number of model sections in a rendered preset file (the ``[*]`` section not counted).

    Args:
        ini: The ``gateway.ini`` path.

    Returns:
        The count; 0 when the file cannot be read.
    """
    try:
        lines = ini.read_text().splitlines()
    except OSError:
        return 0
    return sum(1 for ln in lines if ln.startswith("[") and ln != "[*]")


def curl_snippet(base_url: str, model: str | None, *, api_key: bool = False,
                 embedding: bool = False) -> str:
    """A one-line ``curl`` call against an OpenAI-compatible endpoint.

    Args:
        base_url: Server URL without ``/v1``.
        model: Model id to put in the request, or None to leave it out (standalone servers).
        api_key: Add an Authorization header with a ``YOUR_API_KEY`` placeholder.
        embedding: Call ``/v1/embeddings`` instead of chat completions.

    Returns:
        The command line.
    """
    body: dict = {"input": "Hello"} if embedding else {
        "messages": [{"role": "user", "content": "Hello"}]}
    if model:
        body = {"model": model, **body}
    path = "/v1/embeddings" if embedding else "/v1/chat/completions"
    auth = " -H 'Authorization: Bearer YOUR_API_KEY'" if api_key else ""
    return (f"curl {base_url.rstrip('/')}{path} -H 'Content-Type: application/json'{auth} "
            f"-d '{json.dumps(body)}'")
