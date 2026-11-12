"""Plain-text rendering of the dicts built by `report`: aligned tables, optional ANSI colour.

No Rich or Textual here. A `Style` made with ``enabled=False`` emits no escape codes at all.
Cells are strings, or ``(text, colour)`` pairs; widths are measured on the plain text, so colour
never disturbs alignment.
"""
from __future__ import annotations

import os
from typing import Sequence, TextIO

from .usage import SRC_ESTIMATE, SRC_PROCESS, SRC_TUNED, format_cpu, format_mem

GB = 1e9

_CODES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "cyan": "36"}
_VERDICT_COLOUR = {"GPU": "green", "GPU+CPU": "yellow", "GPU+experts": "cyan",
                   "CPU": "yellow", "NO": "red"}
_STATE_COLOUR = {"loaded": "green", "ready": "green", "sleeping": "cyan", "loading": "yellow",
                 "unloaded": "dim", "down": "red", "failed": "red"}


class Style:
    """Wraps text in ANSI colour codes, or leaves it alone when disabled.

    Args:
        enabled: Emit escape codes.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def paint(self, text: str, colour: str = "") -> str:
        """Colour `text`.

        Args:
            text: Plain text.
            colour: One of bold, dim, red, green, yellow, cyan; "" or unknown for none.

        Returns:
            The text, wrapped in escape codes when enabled and the colour is known.
        """
        code = _CODES.get(colour)
        return f"\x1b[{code}m{text}\x1b[0m" if self.enabled and code else text

    def verdict(self, v: str) -> str:
        """A plan verdict in its colour."""
        return self.paint(v, _VERDICT_COLOUR.get(v, ""))

    def state(self, s: str) -> str:
        """A server or model state in its colour."""
        return self.paint(s, _STATE_COLOUR.get(s, ""))


def use_colour(stream: TextIO, environ: "os._Environ[str] | dict[str, str] | None" = None) -> bool:
    """Whether to colour output on `stream`: a TTY, with ``NO_COLOR`` unset and ``TERM`` not dumb.

    Args:
        stream: The output stream.
        environ: Environment to look at; defaults to ``os.environ``.

    Returns:
        True when ANSI codes are welcome.
    """
    env = os.environ if environ is None else environ
    if env.get("NO_COLOR") or env.get("TERM") == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def _plain(cell: "str | tuple[str, str]") -> str:
    """Text of a cell."""
    return cell if isinstance(cell, str) else cell[0]


def table(headers: Sequence[str], rows: Sequence[Sequence["str | tuple[str, str]"]],
          style: Style, right: Sequence[int] = (), indent: str = "") -> str:
    """Render rows as an aligned text table with a bold header.

    Args:
        headers: Column titles.
        rows: Cells per row; a cell is a string or a ``(text, colour)`` pair.
        style: Colouring.
        right: Indexes of right-aligned columns.
        indent: Prefix for every line.

    Returns:
        The table, one line per row, without a trailing newline.
    """
    widths = [max([len(h)] + [len(_plain(r[i])) for r in rows]) for i, h in enumerate(headers)]

    def line(cells: Sequence["str | tuple[str, str]"], head: bool = False) -> str:
        out = []
        for i, c in enumerate(cells):
            text = _plain(c)
            pad = " " * (widths[i] - len(text))
            colour = "bold" if head else ("" if isinstance(c, str) else c[1])
            painted = c.painted if isinstance(c, _Painted) else style.paint(text, colour)
            out.append(pad + painted if i in right else painted + (pad if i < len(cells) - 1 else ""))
        return (indent + "  ".join(out)).rstrip()

    return "\n".join([line(headers, True)] + [line(r) for r in rows])


def gb(n: float) -> str:
    """Bytes as decimal gigabytes with one decimal ("14.4 GB")."""
    return f"{n / GB:.1f} GB"


def ctx_str(n: int) -> str:
    """A token count the short way: 16384 -> "16k", 131072 -> "128k", 40960 -> "40k", 512 -> "512"."""
    return f"{n // 1024}k" if n >= 1024 and n % 1024 == 0 else f"{n / 1000:.0f}k" if n >= 1000 else str(n)


def duration(seconds: float | None) -> str:
    """Seconds as "45s", "12m", "3h05m" or "2d04h"; "-" when unknown."""
    if seconds is None:
        return "-"
    s = int(seconds)
    if s < 90:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h{s % 3600 // 60:02d}m"
    return f"{s // 86400}d{s % 86400 // 3600:02d}h"


def heading(text: str, style: Style) -> str:
    """A bold section title."""
    return style.paint(text, "bold")


def kv(pairs: Sequence[tuple[str, str]], indent: str = "") -> str:
    """Aligned ``label  value`` lines.

    Args:
        pairs: (label, value) rows.
        indent: Prefix for every line.

    Returns:
        The lines joined by newlines.
    """
    width = max((len(k) for k, _ in pairs), default=0)
    return "\n".join(f"{indent}{k.ljust(width)}  {v}".rstrip() for k, v in pairs)


# --------------------------------------------------------------------------- commands

def render_hw(d: dict, style: Style) -> str:
    """Text for ``llamactl hw``."""
    lines = [heading("Devices", style)]
    if d["devices"]:
        lines.append(table(
            ["ID", "BACKEND", "NAME", "TOTAL", "FREE"],
            [[g["id"], g["backend"], g["name"], gb(g["total"]), gb(g["free"])]
             for g in d["devices"]], style, right=(3, 4), indent="  "))
    else:
        lines.append("  none (CPU only)")
    cpu, ram = d["cpu"], d["ram"]
    lines += ["", heading("System", style), kv([
        ("RAM", f"{gb(ram['total'])} total, {gb(ram['free'])} free"),
        ("CPU", f"{cpu['name'] or 'unknown'} ({cpu['cores']} cores, {cpu['threads']} threads)"),
        ("Unified memory", "yes (GPU memory is system RAM)" if d["unified"] else "no"),
    ], "  ")]
    cal = d["calibration"]
    lines += ["", heading("Estimate calibration", style)]
    if cal:
        lines.append(kv([(b, f"x{c['factor']:.2f} from {c['trials']} measured trial"
                             f"{'s' * (c['trials'] != 1)}" if c["trials"] else "x1.00 (nothing measured yet)")
                         for b, c in cal.items()], "  "))
    else:
        lines.append("  not applicable without a GPU")
    ll = d["llama_cpp"]
    state = (ll["build"] or "found (version unknown)") if ll["found"] else "llama-server not found"
    lines += ["", heading("llama.cpp", style), kv([("Folder", ll["bin_dir"]), ("Build", state)], "  ")]
    if (w := d.get("wsl")) is not None:
        lines += ["", heading("WSL", style)]
        if advice := w["advice"]:
            lines.append(f"  {advice['text']}")
            lines += [f"  {ln}" for ln in advice["snippet"].rstrip("\n").split("\n")]
            lines.append("  then run `wsl --shutdown` in Windows")
        else:
            lines.append("  Windows RAM unknown: set it in Settings to see whether WSL can "
                         "use more")
    return "\n".join(lines)


def _fit_cell(f: dict, style: Style) -> str:
    """``chat GPU 16k`` for one intent summary (the verdict coloured)."""
    return f"{f['intent']} {style.verdict(f['verdict'])} {ctx_str(f['ctx_per_slot'])}"


def _fit_plain(f: dict) -> str:
    """Uncoloured `_fit_cell`, for width measuring."""
    return f"{f['intent']} {f['verdict']} {ctx_str(f['ctx_per_slot'])}"


def _preset_text(p: dict) -> str:
    """``*chat (tuned, stale)`` for one preset row."""
    notes = ", ".join([p["source"]] + (["stale"] if p["stale"] else []))
    return f"{'*' if p['active'] else ''}{p['name']} ({notes})"


def render_ls(rows: list[dict], style: Style) -> str:
    """Text for ``llamactl ls``: one row per model; unreadable models carry their error."""
    if not rows:
        return "no models found; add a folder to model_dirs in the config"
    body: list[list] = []
    for r in rows:
        size = gb(r["size"]) if r["size"] else "?"
        if r["error"]:
            body.append([r["alias"], size, r["quant"], "", (f"unreadable: {r['error']}", "red"), ""])
            continue
        fits = "  ".join(_fit_plain(f) for f in r["fits"])
        cell: "str | tuple[str, str]" = fits
        if style.enabled:
            # colour the verdicts inside the joined cell; padding uses the plain width
            cell = _Painted(fits, "  ".join(_fit_cell(f, style) for f in r["fits"]))
        body.append([r["alias"], size, r["quant"], r["params"], cell,
                     ", ".join(_preset_text(p) for p in r["presets"]) or "-"])
    return table(["MODEL", "SIZE", "QUANT", "PARAMS", "FITS (plan per intent)", "PRESETS"],
                 body, style, right=(1,))


class _Painted(str):
    """A string that prints as `painted` but measures as itself (for pre-coloured cells)."""

    def __new__(cls, plain: str, painted: str) -> "_Painted":
        obj = super().__new__(cls, plain)
        obj.painted = painted
        return obj


def render_plan(d: dict, style: Style) -> str:
    """Text for ``llamactl plan`` (one intent)."""
    m, it, b, mem = d["model"], d["intent"], d["budget"], d["memory"]
    lines = [
        f"{heading(m['alias'], style)} for {it['key']} ({it['description']})",
        f"  verdict  {style.verdict(d['verdict_label'])}",
        f"  context  {d['ctx_per_slot']} tokens per slot x {d['slots']} slot"
        f"{'s' * (d['slots'] != 1)} = {d['ctx_total']} total (-c)",
        "",
        heading("Memory", style),
    ]
    scale = f"  x{b['scale']:.2f} calibration" if b["scale"] != 1.0 else ""
    lines.append(kv([
        ("weights", gb(b["weights"])), ("KV cache", gb(b["kv"])),
        ("compute", gb(b["compute"])), ("overhead", gb(b["overhead"])),
        ("needs", gb(b["total"]) + scale)], "  "))
    if mem["devices"]:
        pool = "unified memory" if mem["unified"] else "VRAM"
        lines.append(f"  against usable {pool} {gb(mem['vram_usable'])} "
                     f"({gb(mem['vram_total'])} total, {mem['headroom_pct']}% headroom; "
                     f"{gb(mem['vram_free'])} free now) and {gb(mem['ram_free'])} free RAM")
    else:
        lines.append(f"  against {gb(mem['ram_free'])} free RAM (no GPU)")
    if d["expert_ram"]:
        lines += ["", heading("Expert placement", style),
                  f"  experts of {d['profile']['n_cpu_moe']}/{m['layers']} layers in RAM "
                  f"({gb(d['expert_ram'])}); GPU part {gb(b['total'] - d['expert_ram'])}",
                  f"  one more layer on the GPU needs {gb(d['layer_gain'])} more VRAM; "
                  "fewer layers in RAM is faster"]
    lines += ["", heading("Why", style)]
    if d["reasons"]:
        lines.append(table(["SETTING", "REASON"], [[k, v] for k, v in d["reasons"].items()],
                           style, indent="  "))
    else:
        lines.append("  the default profile already fits")
    if d["warnings"]:
        lines += ["", style.paint("Warnings", "yellow")]
        lines += [f"  - {w}" for w in d["warnings"]]
    if d["existing_preset"]:
        lines += ["", f"The model already has a '{it['key']}' preset ({d['existing_preset']}); "
                      "this plan does not change it."]
    lines += ["", heading("Command", style), f"  {d['command']}"]
    return "\n".join(lines)


def render_plan_all(d: dict, style: Style) -> str:
    """Text for ``llamactl plan --all``: one row per applicable intent."""
    rows = [[p["intent"], (p["verdict_label"], _VERDICT_COLOUR.get(p["verdict_label"], "")),
             str(p["ctx_per_slot"]), str(p["slots"]), p["cache_type"], gb(p["needs"]),
             "; ".join(p["warnings"])] for p in d["plans"]]
    m = d["model"]
    return (f"{heading(m['alias'], style)} ({gb(m['size'])})\n"
            + table(["INTENT", "VERDICT", "CTX/SLOT", "SLOTS", "KV", "NEEDS", "WARNINGS"],
                    rows, style, right=(2, 3, 5)))


def render_tune(d: dict, style: Style) -> str:
    """Text for the result of ``llamactl tune``."""
    head = "cancelled" if d["cancelled"] else ("tuned" if d["ok"] else "tune failed")
    colour = "green" if d["ok"] and not d["cancelled"] else "red" if not d["ok"] else "yellow"
    said = d["message"].startswith(head)          # "cancelled; ..." already says it
    lines = [f"{'' if said else style.paint(head, colour) + ': '}{d['model']} for "
             f"{d['intent']}: {d['message']}"]
    best = d["best"]
    if best:
        m = best["measurement"]
        lines.append(kv([
            ("context", f"{best['ctx_per_slot']} per slot, {best['cache_type']} KV, {best['verdict']}"),
            ("speed", f"{m['gen_tps']:.1f} tok/s generation, {m['prompt_tps']:.1f} tok/s prompt"),
            ("memory", f"{gb(m['vram_used'])} VRAM, {gb(m['ram_used'])} RAM"),
        ], "  "))
        if d["why"]:
            lines.append(f"  {d['why']}")
        lines.append(f"  saved as the '{d['intent']}' preset (tuned)")
    lines.append(f"  calibration factor now x{d['calibration_factor']:.2f}")
    if d["gateway_reloaded"]:
        lines.append("  gateway reloaded")
    return "\n".join(lines)


def _usages(d: dict) -> list[dict]:
    """Every non-null usage dict in `ps_data` output."""
    rows = [m.get("usage") for m in d["gateway"].get("models") or ()]
    rows += [s.get("usage") for s in d["sessions"]]
    return [u for u in rows if u]


def _usage_cells(u: dict | None) -> list[str]:
    """CPU/RAM/VRAM cells of one `usage` dict; ``-`` each when there is none."""
    if not u:
        return ["-", "-", "-"]
    return [format_cpu(u["cpu_pct"]), format_mem(u["rss"]),
            format_mem(u["vram"], u["vram_source"])]


def render_ps(d: dict, style: Style) -> str:
    """Text for ``llamactl ps``."""
    g = d["gateway"]
    if g["running"]:
        health = (style.paint("healthy", "green") if g["healthy"]
                  else style.paint("not answering", "red"))
        up = f", up {duration(g['uptime_s'])}" if g["uptime_s"] is not None else ""
        lines = [f"{heading('Gateway', style)}  running at {', '.join(g['urls'])} (pid {g['pid']}{up}), "
                 f"{health}"]
        if g["models"] is None:
            lines.append(f"  models unavailable: {g['error']}")
        elif g["models"]:
            lines.append(table(["MODEL", "STATE", "CPU", "RAM", "VRAM"],
                               [[m["name"], (m["state"], _STATE_COLOUR.get(m["state"], "")),
                                 *_usage_cells(m.get("usage"))]
                                for m in g["models"]], style, indent="  "))
        else:
            lines.append("  no models configured")
    else:
        lines = [f"{heading('Gateway', style)}  not running (llamactl up starts it on {', '.join(g['urls'])})"]
    lines += ["", heading("Servers", style)]
    if d["sessions"]:
        lines.append(table(
            ["NAME", "URL", "STATE", "UP", "KIND", "CPU", "RAM", "VRAM"],
            [[s["name"], ", ".join(s["urls"]), (s["state"], _STATE_COLOUR.get(s["state"], "")),
              duration(s["uptime_s"]), "attached" if s["external"] else "standalone",
              *_usage_cells(s.get("usage"))]
             for s in d["sessions"]], style, indent="  "))
    else:
        lines.append("  no standalone or attached servers")
    sources = {u["vram_source"] for u in _usages(d) if u["vram_source"]}
    if sources - {SRC_PROCESS}:
        note = ("~ estimated (the GPU driver does not report per-process memory)"
                if sources == {SRC_ESTIMATE} else
                "~ from a tuned measurement or estimate (the GPU driver does not report "
                "per-process memory)")
        lines.append(style.paint(f"  {note}", "dim"))
    return "\n".join(lines)


def render_config(d: dict, style: Style) -> str:
    """Text for ``llamactl config``: the settings, where clients connect, and any warning.

    Args:
        d: Result of `cfgedit.show`.
        style: Colour settings.

    Returns:
        The lines; the API key is already masked in `d`.
    """
    g = d["gateway"]
    key = f"{g['api_key']} (set)" if g["api_key"] else "none"
    rows = [("gateway.host", f"{g['host']}  ({g['mode']})"), ("gateway.port", str(g["port"])),
            ("gateway.api_key", key), ("gateway.models_max", str(g["models_max"])),
            ("gateway.sleep_idle", str(g["sleep_idle"])),
            ("gateway.autostart", "true" if g["autostart"] else "false"),
            ("default_intent", d["default_intent"]), ("headroom_pct", str(d["headroom_pct"]))]
    lines = [kv(rows), "", heading("Clients connect to", style)]
    lines += [f"  {u}/v1" for u in d["urls"]]
    lines += [style.paint(f"! {w}", "yellow") for w in d["warnings"]]
    lines += [f"note: {n}" for n in d["notes"]]
    return "\n".join(lines)
