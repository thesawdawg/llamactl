"""The settings ``llamactl config`` can show and change, with their validation.

Pure: each key has a parser that turns text into a typed value (or explains what is expected),
and `apply` validates every pair before it changes anything, so a bad value leaves the config
alone. Saving is the caller's job.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from . import intents, network
from .config import Config

TRUE, FALSE = ("1", "true", "on", "yes"), ("0", "false", "off", "no")


class ConfigError(ValueError):
    """A key is unknown or a value is not acceptable (the message says what to type)."""


def _int(low: int, high: int | None = None) -> Callable[[str], int]:
    """A parser for a whole number in a range.

    Args:
        low: Smallest accepted value.
        high: Largest accepted value, None for no limit.

    Returns:
        The parser; it raises `ValueError` with the accepted range.
    """
    want = f"a whole number from {low} to {high}" if high is not None else \
        f"a whole number, {low} or more"

    def parse(text: str) -> int:
        try:
            n = int(text.strip())
        except ValueError:
            raise ValueError(f"expected {want}") from None
        if n < low or (high is not None and n > high):
            raise ValueError(f"expected {want}")
        return n

    return parse


def _bool(text: str) -> bool:
    """Parse true/false, yes/no, on/off or 1/0.

    Raises:
        ValueError: For anything else.
    """
    t = text.strip().lower()
    if t in TRUE:
        return True
    if t in FALSE:
        return False
    raise ValueError("expected true or false")


def _host(text: str) -> str:
    """Parse a listen address; ``local`` and ``lan`` stand for 127.0.0.1 and 0.0.0.0.

    Raises:
        ValueError: When it is empty or holds spaces.
    """
    return network.check_host(text)


def _intent(text: str) -> str:
    """Parse an intent key.

    Raises:
        ValueError: When no intent has that key.
    """
    t = text.strip()
    if t not in intents.by_key:
        raise ValueError(f"expected one of {', '.join(intents.by_key)}")
    return t


def _any(text: str) -> str:
    """Take the text as it is (an empty string clears the setting)."""
    return text


@dataclass(frozen=True)
class Key:
    """One setting of ``llamactl config``.

    Attributes:
        name: The key as typed (``gateway.port``).
        parse: Text to value; raises `ValueError`.
        get: Reads the value from a config.
        put: Writes the value to a config.
        restart: Changing it does not reach a running gateway until it is restarted.
    """

    name: str
    parse: Callable[[str], Any]
    get: Callable[[Config], Any]
    put: Callable[[Config, Any], None]
    restart: bool = False


def _gateway(attr: str, parse: Callable[[str], Any], restart: bool = True) -> Key:
    """A key backed by a field of ``cfg.gateway``."""
    return Key(f"gateway.{attr}", parse, lambda c: getattr(c.gateway, attr),
               lambda c, v: setattr(c.gateway, attr, v), restart)


KEYS: dict[str, Key] = {k.name: k for k in (
    _gateway("host", _host),
    _gateway("port", _int(1, 65535)),
    _gateway("api_key", _any),
    _gateway("models_max", _int(1)),
    _gateway("sleep_idle", _int(0)),
    _gateway("autostart", _bool, restart=False),
    Key("default_intent", _intent, lambda c: c.default_intent,
        lambda c, v: setattr(c, "default_intent", v)),
    Key("headroom_pct", _int(0, 50), lambda c: c.headroom_pct,
        lambda c, v: setattr(c, "headroom_pct", v)),
)}


def split_pair(text: str) -> tuple[str, str]:
    """Split ``KEY=VALUE``.

    Args:
        text: One argument.

    Returns:
        (key, value); the value may be empty.

    Raises:
        ConfigError: When there is no ``=``.
    """
    key, eq, value = text.partition("=")
    if not eq or not key.strip():
        raise ConfigError(f"'{text}' is not KEY=VALUE (keys: {', '.join(KEYS)})")
    return key.strip(), value


def apply(cfg: Config, pairs: list[tuple[str, str]]) -> list[Key]:
    """Validate every pair, then set the ones that change something.

    Args:
        cfg: Config to update in place.
        pairs: (key, text) as typed.

    Returns:
        The keys whose value changed, in the order given.

    Raises:
        ConfigError: For an unknown key or a bad value; `cfg` is untouched then.
    """
    parsed: list[tuple[Key, Any]] = []
    for name, text in pairs:
        key = KEYS.get(name)
        if key is None:
            raise ConfigError(f"unknown setting '{name}' (settings: {', '.join(KEYS)})")
        try:
            parsed.append((key, key.parse(text)))
        except ValueError as e:
            raise ConfigError(f"{name}: {e}") from None
    changed: list[Key] = []
    for key, value in parsed:
        if key.get(cfg) != value:
            key.put(cfg, value)
            if key not in changed:
                changed.append(key)
    return changed


def show(cfg: Config) -> dict:
    """The settings ``config`` prints, with the API key masked.

    Args:
        cfg: The config.

    Returns:
        A JSON-able dict: ``gateway`` (host, mode, port, api_key, models_max, sleep_idle,
        autostart), ``default_intent``, ``headroom_pct``, ``urls`` (what clients connect to),
        ``exposed``, and ``warnings`` / ``notes`` for an exposed bind.
    """
    g = cfg.gateway
    warning = network.key_warning(g.host, g.api_key)
    return {
        "gateway": {"host": g.host, "mode": network.bind_mode(g.host), "port": g.port,
                    "api_key": network.mask_key(g.api_key), "models_max": g.models_max,
                    "sleep_idle": g.sleep_idle, "autostart": g.autostart},
        "default_intent": cfg.default_intent,
        "headroom_pct": cfg.headroom_pct,
        "urls": network.client_urls(g.host, g.port),
        "exposed": network.is_exposed(g.host),
        "warnings": [warning] if warning else [],
        "notes": network.exposure_notes(g.host),
    }
