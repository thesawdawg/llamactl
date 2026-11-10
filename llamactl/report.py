"""Plain-data views of the core objects, for ``--json`` output and for text rendering.

Every function returns JSON-able dicts and lists (numbers in bytes, seconds, tokens), so a
front end can print them as JSON or format them itself. Nothing here prints or does I/O except
reading a model file's modification time through `Calibration.is_stale`.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import hf, planner
from .calibration import Calibration
from .command import build_command, command_string
from .config import Config, ModelEntry
from .fmt import params_str
from .gateway import GatewayModel, GatewayStatus
from .gguf import ModelInfo
from .hardware import Host
from .intents import Intent, applicable
from .measure import Measurement
from .planner import Plan
from .sessions import BASE_PORT, Session
from .tuner import TrialEvent, TuneResult


def hw_data(host: Host, calibration: Calibration, bin_dir: str, found: bool, build: str) -> dict:
    """Hardware summary.

    Args:
        host: Host snapshot.
        calibration: Measured data; supplies a factor per backend present.
        bin_dir: Configured llama.cpp binary folder.
        found: ``llama-server`` exists in `bin_dir`.
        build: llama-server version string, "" when unknown.

    Returns:
        Dict with devices, ram, cpu, unified, calibration and llama_cpp.
    """
    backends = list(dict.fromkeys(g.backend for g in host.gpus))
    return {
        "devices": [asdict(g) for g in host.gpus],
        "ram": {"total": host.ram_total, "free": host.ram_free},
        "cpu": {"name": host.cpu_name, "threads": host.cpu_threads, "cores": host.cpu_cores},
        "unified": host.unified,
        "calibration": {b: {"factor": round(calibration.factor(b), 4),
                            "trials": sum(1 for t in calibration.trials
                                          if t.backend == b and t.ratio() is not None)}
                        for b in backends},
        "llama_cpp": {"bin_dir": bin_dir, "found": found, "build": build},
    }


def quant_of(path: Path) -> str:
    """Quantisation tag guessed from the file name, as the TUI shows it ("" when none)."""
    return hf.quant_of(path.name)


def preset_rows(entry: ModelEntry | None, path: Path, calibration: Calibration,
                build: str) -> list[dict]:
    """The model's presets, active one first.

    Args:
        entry: The model's config entry, or None.
        path: Model file.
        calibration: Used to tell whether a tuned preset is stale.
        build: Current llama-server build string ("" when unknown).

    Returns:
        One dict per preset: name, intent, source, active, stale and the measured speed.
    """
    if entry is None:
        return []
    out = []
    for name in sorted(entry.presets, key=lambda n: (n != entry.active, n)):
        p = entry.presets[name]
        m = p.measured
        out.append({"name": name, "intent": p.intent, "source": p.source,
                    "active": name == entry.active,
                    "stale": calibration.is_stale(p, path, build),
                    "gen_tps": m.gen_tps if m and m.ok else None})
    return out


def plan_summary(plan: Plan, slots: int) -> dict:
    """One intent's plan in compact form (for the ``ls`` column and ``plan --all``).

    Args:
        plan: The plan.
        slots: Parallel slots of the intent.

    Returns:
        Dict with intent, verdict, ctx_per_slot, slots, cache type, total bytes and warnings.
    """
    return {"intent": plan.intent, "verdict": plan.verdict, "ctx_per_slot": plan.ctx_per_slot,
            "slots": slots, "cache_type": plan.profile.cache_type_k,
            "needs": plan.budget.total(), "warnings": list(plan.warnings)}


def model_row(path: Path, alias: str, info: ModelInfo | None, error: str, cfg: Config,
              host: Host, calibration: Calibration, build: str) -> dict:
    """One model of ``llamactl ls``.

    Args:
        path: Model file.
        alias: Name clients use for it.
        info: Parsed GGUF header, or None when it could not be read.
        error: Why `info` is None ("" otherwise).
        cfg: Config (default profile, headroom, presets).
        host: Host snapshot.
        calibration: Measured data.
        build: Current llama-server build string.

    Returns:
        Dict with alias, path, size, quant, params, error, fits (one summary per applicable
        intent) and presets.
    """
    size = info.size if info else _size(path)
    fits: list[dict] = []
    if info is not None:
        for intent, plan in _plans(cfg, info, host, calibration):
            fits.append(plan_summary(plan, intent.slots))
    return {"alias": alias, "path": str(path), "size": size, "quant": quant_of(path),
            "params": params_str(info.parameter_count) if info else "", "error": error,
            "fits": fits,
            "presets": preset_rows(cfg.models.get(str(path)), path, calibration, build)}


def _size(path: Path) -> int:
    """File size, 0 when unreadable."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _plans(cfg: Config, info: ModelInfo, host: Host,
           calibration: Calibration) -> list[tuple[Intent, Plan]]:
    """Plans for every intent that applies to the model."""
    return [(i, planner.plan(i, info, host, cfg.default_profile, cfg.headroom_pct, calibration))
            for i in applicable(info)]


def plan_all_data(cfg: Config, path: Path, alias: str, info: ModelInfo, host: Host,
                  calibration: Calibration) -> dict:
    """Every applicable intent's plan side by side.

    Args:
        cfg: Config.
        path: Model file.
        alias: Name clients use for it.
        info: Model metadata.
        host: Host snapshot.
        calibration: Measured data.

    Returns:
        Dict with the model and a list of `plan_summary` dicts under "plans".
    """
    return {"model": _model(path, alias, info),
            "plans": [plan_summary(p, i.slots) for i, p in _plans(cfg, info, host, calibration)]}


def _model(path: Path, alias: str, info: ModelInfo) -> dict:
    """Identity block of a model."""
    return {"alias": alias, "path": str(path), "size": info.size, "quant": quant_of(path),
            "params": params_str(info.parameter_count)}


def plan_data(cfg: Config, path: Path, alias: str, info: ModelInfo, host: Host, plan: Plan,
              intent: Intent) -> dict:
    """Everything ``llamactl plan`` prints for one intent.

    Args:
        cfg: Config (headroom, binary folder, existing presets).
        path: Model file.
        alias: Name clients use for it.
        info: Model metadata.
        host: Host snapshot.
        plan: The plan.
        intent: The intent it was made for.

    Returns:
        Dict with model, intent, verdict, ctx, budget, memory, reasons, warnings, profile,
        existing preset source and the exact command.
    """
    b = plan.budget
    pct = cfg.headroom_pct
    existing = cfg.preset(str(path), plan.intent)
    return {
        "model": _model(path, alias, info),
        "intent": {"key": intent.key, "label": intent.label, "description": intent.description,
                   "ctx_min": planner.ctx_bounds(intent, info)[2], "slots": intent.slots},
        "verdict": plan.verdict,
        "ctx_per_slot": plan.ctx_per_slot,
        "slots": intent.slots,
        "ctx_total": plan.profile.ctx_size,
        "budget": {"weights": b.weights, "kv": b.kv, "compute": b.compute,
                   "overhead": b.overhead, "scale": b.scale, "total": b.total()},
        "memory": {"vram_total": host.vram_total, "vram_free": host.vram_free,
                   "vram_usable": planner.gpu_limit(host, pct),
                   "ram_total": host.ram_total, "ram_free": host.ram_free,
                   "unified": host.unified, "headroom_pct": pct,
                   "devices": len(host.gpus)},
        "reasons": dict(plan.reasons),
        "warnings": list(plan.warnings),
        "profile": plan.profile.to_dict(),
        "existing_preset": existing.source if existing else None,
        "command": command_string(build_command(cfg, str(path), plan.profile,
                                                plan.profile.port or BASE_PORT)),
    }


def measurement_data(m: Measurement | None) -> dict | None:
    """A measurement as a dict, or None."""
    return m.to_dict() if m else None


def tune_data(alias: str, result: TuneResult, reloaded: bool, factor: float) -> dict:
    """Result of ``llamactl tune``.

    Args:
        alias: Model alias.
        result: What the tuner returned.
        reloaded: The running gateway was told to re-read the presets.
        factor: Calibration factor of the host backend after recording the trials.

    Returns:
        Dict with intent, ok, cancelled, message, the winner and every trial.
    """
    best = result.best
    return {
        "model": alias, "intent": result.intent, "ok": result.ok, "cancelled": result.cancelled,
        "message": result.message,
        "best": None if best is None else {
            "how": best.how, "ctx_per_slot": best.ctx_per_slot, "verdict": best.verdict,
            "cache_type": best.profile.cache_type_k, "profile": best.profile.to_dict(),
            "measurement": measurement_data(result.measurement)},
        "trials": [{"verdict": t.verdict, "ctx_size": t.profile.get("ctx_size"),
                    "cache_type": t.profile.get("cache_type_k"),
                    "measurement": t.measurement.to_dict()} for t in result.trials],
        "gateway_reloaded": reloaded, "calibration_factor": round(factor, 4),
    }


def event_line(e: TrialEvent) -> str:
    """One progress line for a trial event.

    Args:
        e: The event.

    Returns:
        E.g. ``[2/4] start  ctx down: 8192 ctx, f16 KV, GPU``.
    """
    head = f"[{e.index}/{e.total}]"
    if e.phase == "start" or e.measurement is None:
        return f"{head} trying  {e.summary}"
    m = e.measurement
    if not m.ok:
        return f"{head} failed  {e.summary}: {m.error}"
    speed = f"{m.gen_tps:.1f} tok/s" if m.gen_tps else f"{m.prompt_tps:.1f} prompt tok/s"
    return f"{head} ok      {e.summary}: {m.vram_used / 1e9:.1f} GB VRAM, {speed}, load {m.load_s:.0f} s"


def ps_data(status: GatewayStatus, models: list[GatewayModel] | None, error: str,
            sessions: list[Session], states: dict[str, str], now: float,
            started: float = 0.0) -> dict:
    """Gateway and standalone/attached servers.

    Args:
        status: Gateway process status.
        models: The router's models, or None when they could not be fetched.
        error: Why `models` is None ("" otherwise).
        sessions: Standalone and attached sessions.
        states: Health state by session id ("ready", "loading", "down").
        now: Current Unix time.
        started: Gateway start time (Unix), 0 when unknown.

    Returns:
        Dict with "gateway" and "sessions".
    """
    return {
        "gateway": {"running": status.running, "url": status.url, "healthy": status.healthy,
                    "pid": status.pid, "uptime_s": int(now - started) if started else None,
                    "error": error,
                    "models": None if models is None else [
                        {"name": m.name, "state": m.state, "exit_code": m.exit_code}
                        for m in models]},
        "sessions": [{"id": s.id, "name": s.name, "url": s.url, "pid": s.pid,
                      "external": s.external, "log": s.log,
                      "state": states.get(s.id, "down"),
                      "uptime_s": int(now - s.started) if s.started else None}
                     for s in sessions],
    }
