"""Multi-step operations shared by the CLI and the TUI.

Everything here sits between the pure core (planner, tuner, gateway) and a front end: choosing
or creating the preset to serve, getting a model loaded in the gateway, waiting for it, and
running a tune end to end. No Textual, no printing; waiting takes injectable `clock` / `sleep`
functions and a `tick` callback so a front end can show progress and tests never sleep.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import intents, planner, tuner
from .calibration import Calibration
from .config import ESTIMATED, PRESET_SEP, Config, Preset
from .gateway import (LOADED, LOADING, SLEEPING, Gateway, GatewayError, GatewayModel,
                      GatewayState, log_reason)
from .gguf import ModelInfo
from .hardware import Host
from .intents import Intent
from .planner import Plan
from .sessions import Session

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
             calibration: Calibration | None = None) -> Plan:
    """Plan an intent with the config's default profile, headroom and the measured calibration.

    Args:
        cfg: Config supplying the base profile and headroom.
        info: Model metadata.
        host: Host snapshot.
        intent: What the model is for.
        calibration: Measured data, or None.

    Returns:
        The plan.
    """
    return planner.plan(intent, info, host, cfg.default_profile, cfg.headroom_pct, calibration)


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
    preset = Preset(plan.intent, plan.profile, ESTIMATED)
    cfg.set_preset(model, preset, activate=activate)
    if save:
        cfg.save()
    return preset


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


def ensure_gateway(gw: Gateway, *, timeout: float = GATEWAY_START_TIMEOUT,
                   clock: Clock = time.monotonic, sleep: Sleep = time.sleep,
                   tick: Tick | None = None) -> tuple[GatewayState, bool]:
    """Make sure the gateway runs and answers ``/health``, starting it when needed.

    Args:
        gw: The gateway.
        timeout: Seconds a started router gets to become healthy.
        clock: Monotonic clock.
        sleep: Sleep function.
        tick: Progress callback, called once per pause.

    Returns:
        (state, started): `started` is False when it was already running.

    Raises:
        GatewayError: When it cannot be started, exits, or stays unhealthy past `timeout`.
    """
    started = gw.state is None
    st = gw.start()

    def healthy() -> bool:
        if not gw.running:
            raise GatewayError(f"the gateway exited: {log_reason(Path(st.log))} (log: {st.log})")
        return gw.status().healthy

    if not _wait_until(healthy, timeout, interval=0.5, clock=clock, sleep=sleep, tick=tick):
        raise GatewayError(f"the gateway at {st.url} did not answer within {timeout:.0f} s "
                           f"(log: {st.log})")
    return st, started


def serve_gateway(cfg: Config, gw: Gateway, model: str, preset_name: str, *,
                  clock: Clock = time.monotonic, sleep: Sleep = time.sleep,
                  tick: Tick | None = None) -> str:
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

    Returns:
        The model id to request (``alias`` or ``alias@preset``).

    Raises:
        GatewayError: When the gateway cannot start, does not know the preset (the message
            lists presets that could not be written), or refuses the load.
    """
    name = served_name(cfg, model, preset_name)
    ensure_gateway(gw, clock=clock, sleep=sleep, tick=tick)
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
                interval: float = POLL_INTERVAL) -> GatewayModel:
    """Wait until the router reports a model as loaded (or sleeping, which wakes on a request).

    Args:
        gw: The gateway.
        name: Model id.
        timeout: Seconds to wait.
        clock: Monotonic clock.
        sleep: Sleep function.
        tick: Progress callback, called once per pause.
        interval: Seconds between checks.

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
                      tick: Tick | None = None, interval: float = 0.5) -> None:
    """Wait until a standalone server answers ``/health`` with 200.

    Args:
        session: The server just launched.
        health: Returns "ready", "loading" or "down" for a base URL.
        timeout: Seconds to wait.
        clock: Monotonic clock.
        sleep: Sleep function.
        tick: Progress callback, called once per pause.
        interval: Seconds between checks.

    Raises:
        ActionError: When the process exits (with the last error line of its log) or
            `timeout` passes.
    """
    def ready() -> bool:
        if not session.alive:
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


def run_tune(cfg: Config, info: ModelInfo, host: Host, intent: Intent, runner: tuner.Runner,
             calibration: Calibration, gw: Gateway, *, trials: int = 4, force: bool = False,
             progress: Callable[[tuner.TrialEvent], None] | None = None,
             cancel: Callable[[], bool] | None = None) -> TuneOutcome:
    """Tune a model for an intent end to end: preflight, search, save, reload the gateway.

    Trials are recorded in ``tuning.json`` even when the tune failed or was cancelled; a
    successful winner is stored as the model's tuned preset; a running gateway then re-reads
    the presets.

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
        cancel: Polled between trials; True stops the search.

    Returns:
        The outcome.

    Raises:
        ActionError: When the preflight refuses (and `force` is off).
        OSError: When tuning.json or the config cannot be written.
    """
    plan = plan_for(cfg, info, host, intent, calibration)
    if not force and (refusal := tuner.preflight(plan, host)):
        raise ActionError(refusal)
    result = tuner.tune(intent, info, host, cfg.default_profile, runner,
                        headroom_pct=cfg.headroom_pct, calibration=calibration,
                        max_trials=trials, progress=progress, cancel=cancel)
    if result.trials or result.ok:
        tuner.save_result(cfg, info.path, result, calibration)
    if not result.ok or not gw.running:
        return TuneOutcome(result)
    try:
        gw.reload()
    except GatewayError as e:
        return TuneOutcome(result, False, str(e))
    return TuneOutcome(result, True)


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
