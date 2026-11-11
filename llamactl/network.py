"""Network access: bind modes, this machine's LAN addresses, client URLs, API keys, WSL notes.

Everything here is pure or takes its inputs as arguments. The three things that touch the
machine (``ip -o -4 addr show``, a UDP ``connect`` to find the default route, and the WSL
files) are small module-level functions that every caller can replace, and the results of
`lan_addresses` and `wsl_mode` are cached per process (`reset_cache` forgets them), so the
tests never run ``ip`` or read ``/proc``.

Loopback is the default everywhere. ``local`` is ``127.0.0.1``, ``lan`` is ``0.0.0.0`` and
``custom`` is any other address.
"""
from __future__ import annotations

import ipaddress
import re
import secrets
import socket
import subprocess
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

LOCAL, LAN, CUSTOM = "local", "lan", "custom"
LOOPBACK, WILDCARD = "127.0.0.1", "0.0.0.0"
HOST_MODES = ((LOCAL, "This machine"), (LAN, "Local network"), (CUSTOM, "Custom"))
WILDCARDS = frozenset({"", "0.0.0.0", "::"})       # compared after `normalise`
IGNORED_INTERFACES = ("docker", "br-", "veth", "virbr", "cni", "flannel")
OSRELEASE = Path("/proc/sys/kernel/osrelease")
ROUTE_PROBE = ("192.0.2.1", 9)           # TEST-NET-1: nothing is sent, connect only picks a route
KEY_BYTES = 32
MASK = "********"
NAT_PRIVATE = ipaddress.ip_network("172.16.0.0/12")
LINK_LOCAL = ipaddress.ip_network("169.254.0.0/16")

_IP_LINE = re.compile(r"^\d+:\s+(\S+)\s+inet\s+(\d+\.\d+\.\d+\.\d+)(?:/\d+)?")

Interfaces = list[tuple[str, str]]


# --------------------------------------------------------------------------- bind modes

def normalise(host: str) -> str:
    """A listen address without padding, IPv6 brackets or case.

    Args:
        host: Address or name as typed or configured.

    Returns:
        ``::1`` for ``[::1]``, ``localhost`` for `` LocalHost ``, and so on.
    """
    h = host.strip().lower()
    return h[1:-1] if h.startswith("[") and h.endswith("]") else h


def is_wildcard(host: str) -> bool:
    """True for an address that listens on every interface (``0.0.0.0``, ``::``, ``[::]``, "").

    Args:
        host: Listen address.

    Returns:
        Whether it is a wildcard.
    """
    return normalise(host) in WILDCARDS


def bind_mode(host: str) -> str:
    """The mode a listen address belongs to.

    Args:
        host: Listen address.

    Returns:
        ``local`` for ``127.0.0.1``, ``lan`` for ``0.0.0.0``, else ``custom`` (that includes
        ``localhost``, ``::`` and a specific interface address).
    """
    h = host.strip()
    return LOCAL if h == LOOPBACK else LAN if h == WILDCARD else CUSTOM


def host_for(mode: str, custom: str = "") -> str:
    """The listen address a mode stands for.

    Args:
        mode: ``local``, ``lan`` or ``custom``.
        custom: The address used for ``custom``.

    Returns:
        ``127.0.0.1``, ``0.0.0.0`` or the stripped custom address.

    Raises:
        ValueError: When `mode` is unknown.
    """
    if mode == LOCAL:
        return LOOPBACK
    if mode == LAN:
        return WILDCARD
    if mode == CUSTOM:
        return custom.strip()
    raise ValueError(f"unknown bind mode '{mode}'")


def is_exposed(host: str) -> bool:
    """True when other machines can reach a server listening on `host`.

    Only loopback addresses (``127.0.0.0/8``, ``::1``) and ``localhost`` are private; wildcards,
    interface addresses and any host name are exposed.

    Args:
        host: Listen address.

    Returns:
        Whether the bind is reachable from the network.
    """
    h = normalise(host)
    if h == "localhost":
        return False
    try:
        return not ipaddress.ip_address(h).is_loopback
    except ValueError:
        return True


def check_host(host: str) -> str:
    """Validate a listen address typed by a person.

    Args:
        host: Address, host name, ``local`` or ``lan``.

    Returns:
        The address to store (``local`` and ``lan`` are expanded).

    Raises:
        ValueError: When it is empty or holds whitespace.
    """
    h = host.strip()
    if h.lower() in (LOCAL, LAN):
        return host_for(h.lower())
    if not h or re.search(r"\s", h):
        raise ValueError("expected a host name or address (or 'local' / 'lan')")
    return h


def new_api_key() -> str:
    """A fresh random API key.

    Returns:
        A URL-safe token of 32 random bytes.
    """
    return secrets.token_urlsafe(KEY_BYTES)


def mask_key(key: str) -> str:
    """What to show of an API key.

    Args:
        key: The key, possibly empty.

    Returns:
        "" for no key, else a fixed mask that reveals nothing.
    """
    return MASK if key else ""


def key_warning(host: str, api_key: str) -> str:
    """The warning for a server that anyone on the network can use.

    Args:
        host: Listen address.
        api_key: The key clients must send, "" for none.

    Returns:
        A one-line warning, or "" when the bind is private or a key is set.
    """
    if api_key or not is_exposed(host):
        return ""
    return (f"listening on {host.strip() or WILDCARD} without an API key: anyone on your "
            "network can use this server (set one with `llamactl config --new-api-key`)")


# --------------------------------------------------------------------------- addresses

def parse_ip_addr(text: str) -> Interfaces:
    """Interface and IPv4 address pairs from ``ip -o -4 addr show``.

    Args:
        text: Command output, one ``2: eth0    inet 192.168.1.5/24 ...`` line per address.

    Returns:
        ``(interface, address)`` in output order; unparsable lines are skipped and a
        ``name@parent`` suffix is dropped from the interface.
    """
    out: Interfaces = []
    for line in text.splitlines():
        if m := _IP_LINE.match(line.strip()):
            out.append((m.group(1).split("@", 1)[0], m.group(2)))
    return out


def usable(iface: str, address: str) -> bool:
    """True for an address another machine could plausibly connect to.

    Args:
        iface: Interface name.
        address: IPv4 address.

    Returns:
        False for loopback, link-local ``169.254/16`` and container or virtual bridges
        (``docker*``, ``br-*``, ``veth*``, ``virbr*``, ``cni*``, ``flannel*``).
    """
    if iface == "lo" or iface.startswith(IGNORED_INTERFACES):
        return False
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return not (ip.is_loopback or ip.is_unspecified or ip in LINK_LOCAL)


def read_ip_output() -> str:
    """Run ``ip -o -4 addr show``.

    Returns:
        Its output, "" when ``ip`` is missing or fails.
    """
    try:
        done = subprocess.run(["ip", "-o", "-4", "addr", "show"], capture_output=True,
                              text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout if done.returncode == 0 else ""


def route_address() -> str | None:
    """The address of the interface that holds the default route.

    A UDP socket is connected (which only chooses a route; no packet is sent) and its own
    address is read back.

    Returns:
        The IPv4 address, or None when there is no route.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(ROUTE_PROBE)
            return str(s.getsockname()[0])
    except OSError:
        return None


_lan_cache: list[str] | None = None
_wsl_cache: list[str | None] = []


def reset_cache() -> None:
    """Forget the cached LAN addresses and WSL mode (they are looked up again on next use)."""
    global _lan_cache
    _lan_cache = None
    _wsl_cache.clear()


def lan_addresses(ip_output: Callable[[], str] | None = None,
                  route: Callable[[], str | None] | None = None) -> list[str]:
    """This machine's non-loopback IPv4 addresses that other machines could use.

    Reads ``ip -o -4 addr show``, skipping loopback, link-local and container bridges; when
    that yields nothing (no ``ip``, or only filtered interfaces) it falls back to the address
    of the default route. With no arguments the answer is cached for the process.

    Args:
        ip_output: Returns the ``ip`` output (default `read_ip_output`).
        route: Returns the default-route address (default `route_address`).

    Returns:
        Addresses in the order ``ip`` lists them, without repeats; empty when none is found.
    """
    global _lan_cache
    cached = ip_output is None and route is None
    if cached and _lan_cache is not None:
        return list(_lan_cache)
    found: list[str] = []
    for iface, addr in parse_ip_addr((ip_output or read_ip_output)()):
        if usable(iface, addr) and addr not in found:
            found.append(addr)
    if not found:
        addr = (route or route_address)()
        if addr and usable("", addr):
            found.append(addr)
    if cached:
        _lan_cache = list(found)
    return found


# --------------------------------------------------------------------------- client URLs

def url_host(host: str) -> str:
    """A host as it appears in a URL (IPv6 gets brackets).

    Args:
        host: Address or name.

    Returns:
        ``::1`` becomes ``[::1]``; everything else is unchanged.
    """
    h = host.strip()
    return f"[{normalise(h)}]" if ":" in h and not h.startswith("[") else h


def probe_host(host: str) -> str:
    """The address llamactl itself connects to for a listen address.

    Args:
        host: Listen address.

    Returns:
        Loopback for a wildcard, else the address.
    """
    return LOOPBACK if is_wildcard(host) else url_host(host)


def client_urls(host: str, port: int, addresses: list[str] | None = None) -> list[str]:
    """The base URLs a person can give to a client for a server listening on `host`.

    Args:
        host: Listen address.
        port: Listen port.
        addresses: This machine's LAN addresses (default `lan_addresses()`); only used for a
            wildcard bind.

    Returns:
        For a wildcard: the loopback URL, then one URL per LAN address. Otherwise one URL with
        the bind address itself. Never a URL with ``0.0.0.0``.
    """
    if not is_wildcard(host):
        return [f"http://{url_host(host)}:{port}"]
    lan = lan_addresses() if addresses is None else addresses
    return [f"http://{LOOPBACK}:{port}", *(f"http://{a}:{port}" for a in lan)]


def client_urls_for(base_url: str, host: str = "", addresses: list[str] | None = None) -> list[str]:
    """`client_urls` for a server known by its base URL.

    Args:
        base_url: ``http://host:port`` as llamactl stores it.
        host: The listen address when it differs from the URL's host (a wildcard bind is
            stored as a loopback URL); "" means the URL's own host.
        addresses: This machine's LAN addresses (default `lan_addresses()`).

    Returns:
        The client URLs; `base_url` alone when it cannot be parsed.
    """
    try:
        parts = urlsplit(base_url)
        port = parts.port
    except ValueError:
        return [base_url.rstrip("/")]
    if parts.hostname is None or port is None:
        return [base_url.rstrip("/")]
    return client_urls(host or parts.hostname, port, addresses)


def primary_url(urls: list[str]) -> str:
    """The URL to hand to a client on another machine.

    Args:
        urls: Result of `client_urls`.

    Returns:
        The first LAN URL when there is one, else the only (loopback or bind) URL.
    """
    return urls[1] if len(urls) > 1 else urls[0]


# --------------------------------------------------------------------------- WSL

def read_osrelease() -> str:
    """The kernel release string.

    Returns:
        The content of ``/proc/sys/kernel/osrelease``, "" when unreadable.
    """
    try:
        return OSRELEASE.read_text()
    except OSError:
        return ""


def read_wslinfo() -> str | None:
    """The WSL networking mode as ``wslinfo --networking-mode`` prints it.

    Returns:
        Its output, or None when ``wslinfo`` is missing or fails.
    """
    try:
        done = subprocess.run(["wslinfo", "--networking-mode"], capture_output=True, text=True,
                              timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def wsl_mode(osrelease: Callable[[], str] | None = None,
             wslinfo: Callable[[], str | None] | None = None,
             ip_output: Callable[[], str] | None = None) -> str | None:
    """How WSL is networked, or None when this is not WSL.

    WSL is recognised by "microsoft" in the kernel release. The mode comes from
    ``wslinfo --networking-mode`` when that exists, else from whether ``eth0`` holds a
    ``172.16/12`` address (NAT). With no arguments the answer is cached for the process.

    Args:
        osrelease: Returns the kernel release (default `read_osrelease`).
        wslinfo: Returns the ``wslinfo`` output or None (default `read_wslinfo`).
        ip_output: Returns the ``ip`` output (default `read_ip_output`).

    Returns:
        None off WSL; ``mirrored`` or ``nat``; ``unknown`` when it cannot be told.
    """
    cached = osrelease is None and wslinfo is None and ip_output is None
    if cached and _wsl_cache:
        return _wsl_cache[0]
    mode = _wsl_mode((osrelease or read_osrelease)(), wslinfo or read_wslinfo,
                     ip_output or read_ip_output)
    if cached:
        _wsl_cache[:] = [mode]
    return mode


def _wsl_mode(release: str, wslinfo: Callable[[], str | None],
              ip_output: Callable[[], str]) -> str | None:
    """The uncached part of `wsl_mode`."""
    if "microsoft" not in release.lower():
        return None
    said = (wslinfo() or "").strip().lower()
    if said in ("mirrored", "nat"):
        return said
    if said:
        return "unknown"
    for iface, addr in parse_ip_addr(ip_output()):
        try:
            if iface == "eth0" and ipaddress.ip_address(addr) in NAT_PRIVATE:
                return "nat"
        except ValueError:
            continue
    return "unknown"


def exposure_notes(host: str, wsl: Callable[[], str | None] | None = None) -> list[str]:
    """What an exposed bind needs on this machine.

    Args:
        host: Listen address.
        wsl: Returns the WSL mode (default `wsl_mode`).

    Returns:
        Nothing for a private bind or off WSL. On WSL with NAT networking: other machines
        cannot reach it without mirrored networking or a ``netsh`` port proxy. With mirrored
        networking: the Hyper-V firewall may block inbound connections until a rule allows
        the port. When the mode is unknown: a pointer to both.
    """
    if not is_exposed(host):
        return []
    mode = (wsl or wsl_mode)()
    if mode == "nat":
        return ["WSL uses NAT networking: other machines cannot reach this server. Enable "
                "networkingMode=mirrored in .wslconfig, or forward the port on Windows with "
                "`netsh interface portproxy`."]
    if mode == "mirrored":
        return ["WSL uses mirrored networking: the Hyper-V firewall may block inbound "
                "connections until a rule allows this port (New-NetFirewallHyperVRule with "
                "VMCreatorId {40E0AC32-46A5-438A-A0B2-2B479E8F2E90}, run in PowerShell as "
                "administrator)."]
    if mode == "unknown":
        return ["WSL networking could not be detected: if other machines cannot connect, use "
                "mirrored networking or `netsh interface portproxy`, and check the Hyper-V "
                "firewall."]
    return []
