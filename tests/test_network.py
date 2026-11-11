"""network.py: bind modes, interface filtering, client URLs, WSL mode and notes. All injected."""
from __future__ import annotations

import pytest

from llamactl import network

# The autouse fixture replaces these with fakes; the tests of the real ones need the originals.
REAL_ROUTE, REAL_IP, REAL_WSLINFO = network.route_address, network.read_ip_output, network.read_wslinfo

IP_OUTPUT = """\
1: lo    inet 127.0.0.1/8 scope host lo\\       valid_lft forever preferred_lft forever
2: eth0    inet 172.20.1.2/20 brd 172.20.15.255 scope global eth0\\       valid_lft forever
3: eth1    inet 192.168.50.7/24 brd 192.168.50.255 scope global dynamic eth1\\       valid_lft 86000sec
4: docker0    inet 172.17.0.1/16 brd 172.17.255.255 scope global docker0\\       valid_lft forever
5: br-1a2b3c4d5e6f    inet 172.18.0.1/16 scope global br-1a2b3c4d5e6f\\       valid_lft forever
6: veth12ab@if5    inet 10.1.1.1/24 scope global veth12ab\\       valid_lft forever
7: virbr0    inet 192.168.122.1/24 scope global virbr0\\       valid_lft forever
8: cni0    inet 10.244.0.1/24 scope global cni0\\       valid_lft forever
9: flannel.1    inet 10.244.0.0/32 scope global flannel.1\\       valid_lft forever
10: wlan0    inet 169.254.7.7/16 scope link wlan0\\       valid_lft forever
"""


# --------------------------------------------------------------------------- bind modes

@pytest.mark.parametrize("host, mode", [
    ("127.0.0.1", "local"), (" 127.0.0.1 ", "local"), ("0.0.0.0", "lan"),
    ("localhost", "custom"), ("::", "custom"), ("[::]", "custom"), ("::1", "custom"),
    ("192.168.50.7", "custom"), ("myhost.lan", "custom"), ("", "custom")])
def test_bind_mode(host: str, mode: str) -> None:
    assert network.bind_mode(host) == mode


@pytest.mark.parametrize("host, exposed", [
    ("127.0.0.1", False), ("127.0.1.1", False), ("localhost", False), ("LocalHost", False),
    ("::1", False), ("[::1]", False), ("0.0.0.0", True), ("::", True), ("[::]", True),
    ("", True), ("192.168.50.7", True), ("myhost.lan", True)])
def test_is_exposed(host: str, exposed: bool) -> None:
    assert network.is_exposed(host) is exposed


def test_host_for_and_check_host() -> None:
    assert network.host_for("local") == "127.0.0.1"
    assert network.host_for("lan") == "0.0.0.0"
    assert network.host_for("custom", " 10.0.0.5 ") == "10.0.0.5"
    with pytest.raises(ValueError):
        network.host_for("nope")
    assert network.check_host("LAN") == "0.0.0.0" and network.check_host("local") == "127.0.0.1"
    assert network.check_host(" 10.0.0.5 ") == "10.0.0.5"
    for bad in ("", "  ", "a b"):
        with pytest.raises(ValueError):
            network.check_host(bad)


def test_keys_and_warning() -> None:
    a, b = network.new_api_key(), network.new_api_key()
    assert a != b and len(a) >= 43
    assert network.mask_key("") == "" and network.mask_key(a) == network.MASK and a not in network.MASK
    assert "without an API key" in network.key_warning("0.0.0.0", "")
    assert network.key_warning("0.0.0.0", "k") == ""
    assert network.key_warning("127.0.0.1", "") == ""
    assert network.key_warning("::1", "") == ""


# --------------------------------------------------------------------------- addresses

def test_parse_ip_addr() -> None:
    got = network.parse_ip_addr(IP_OUTPUT)
    assert got[0] == ("lo", "127.0.0.1") and ("eth1", "192.168.50.7") in got
    assert ("veth12ab", "10.1.1.1") in got           # the @if5 suffix is dropped
    assert len(got) == 10
    assert network.parse_ip_addr("") == [] and network.parse_ip_addr("garbage\n2: x inet6 ::1/128") == []


@pytest.mark.parametrize("iface, addr, ok", [
    ("eth1", "192.168.50.7", True), ("wlan0", "10.0.0.8", True), ("eth0", "169.254.1.1", False),
    ("lo", "127.0.0.1", False), ("eth0", "127.0.0.2", False), ("docker0", "172.17.0.1", False),
    ("br-1a2b", "172.18.0.1", False), ("veth9", "10.1.1.1", False), ("virbr0", "192.168.122.1", False),
    ("cni0", "10.244.0.1", False), ("flannel.1", "10.244.0.0", False), ("eth0", "0.0.0.0", False),
    ("eth0", "999.1.1.1", False)])
def test_usable(iface: str, addr: str, ok: bool) -> None:
    assert network.usable(iface, addr) is ok


def test_lan_addresses_filters_the_ip_output() -> None:
    def no_route() -> str | None:
        raise AssertionError("the route fallback must not run when ip found addresses")

    assert network.lan_addresses(lambda: IP_OUTPUT, no_route) == ["172.20.1.2", "192.168.50.7"]


def test_lan_addresses_removes_repeats() -> None:
    text = "2: eth0    inet 10.0.0.2/24 scope global eth0\n3: eth0    inet 10.0.0.2/24 scope global eth0\n"
    assert network.lan_addresses(lambda: text, lambda: None) == ["10.0.0.2"]


def test_lan_addresses_falls_back_to_the_default_route() -> None:
    only_filtered = "1: lo    inet 127.0.0.1/8 scope host lo\n3: docker0    inet 172.17.0.1/16 scope global docker0\n"
    assert network.lan_addresses(lambda: only_filtered, lambda: "10.9.8.7") == ["10.9.8.7"]
    assert network.lan_addresses(lambda: "", lambda: "10.9.8.7") == ["10.9.8.7"]
    # the fallback address is filtered like the others
    assert network.lan_addresses(lambda: "", lambda: "127.0.0.1") == []
    assert network.lan_addresses(lambda: "", lambda: "169.254.3.3") == []
    assert network.lan_addresses(lambda: "", lambda: None) == []


def test_lan_addresses_is_cached_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def ip() -> str:
        calls.append(1)
        return "2: eth1    inet 10.1.2.3/24 scope global eth1\n"

    monkeypatch.setattr(network, "read_ip_output", ip)
    network.reset_cache()
    assert network.lan_addresses() == ["10.1.2.3"] and network.lan_addresses() == ["10.1.2.3"]
    assert len(calls) == 1
    network.reset_cache()
    network.lan_addresses()
    assert len(calls) == 2
    # explicit probes bypass the cache and never fill it
    network.reset_cache()
    assert network.lan_addresses(lambda: "", lambda: "10.5.5.5") == ["10.5.5.5"]
    assert network.lan_addresses() == ["10.1.2.3"] and len(calls) == 3


def test_route_address_uses_a_udp_socket_that_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list = []

    class FakeSocket:
        def __init__(self, family: int, kind: int) -> None:
            seen.append(("open", family, kind))

        def __enter__(self) -> "FakeSocket":
            return self

        def __exit__(self, *exc: object) -> None:
            pass

        def connect(self, addr: tuple) -> None:
            seen.append(("connect", addr))

        def getsockname(self) -> tuple[str, int]:
            return ("10.7.7.7", 51000)

        def sendto(self, *a: object) -> None:
            raise AssertionError("nothing may be sent")

    monkeypatch.setattr(network.socket, "socket", FakeSocket)
    assert REAL_ROUTE() == "10.7.7.7"
    assert seen[0][2] == network.socket.SOCK_DGRAM and seen[1] == ("connect", network.ROUTE_PROBE)

    class NoRoute(FakeSocket):
        def connect(self, addr: tuple) -> None:
            raise OSError("Network is unreachable")

    monkeypatch.setattr(network.socket, "socket", NoRoute)
    assert REAL_ROUTE() is None


def test_read_ip_output_survives_a_missing_command(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*a: object, **k: object) -> None:
        raise FileNotFoundError("ip")

    monkeypatch.setattr(network.subprocess, "run", missing)
    assert REAL_IP() == "" and REAL_WSLINFO() is None


# --------------------------------------------------------------------------- client URLs

def test_client_urls_for_loopback_and_a_specific_address() -> None:
    assert network.client_urls("127.0.0.1", 8080, ["10.0.0.1"]) == ["http://127.0.0.1:8080"]
    assert network.client_urls("localhost", 9, ["10.0.0.1"]) == ["http://localhost:9"]
    assert network.client_urls("::1", 8080, []) == ["http://[::1]:8080"]
    assert network.client_urls("192.168.50.7", 8080, ["10.0.0.1"]) == ["http://192.168.50.7:8080"]


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "[::]", ""])
def test_client_urls_for_a_wildcard(host: str) -> None:
    urls = network.client_urls(host, 8080, ["192.168.50.7", "10.0.0.2"])
    assert urls == ["http://127.0.0.1:8080", "http://192.168.50.7:8080", "http://10.0.0.2:8080"]
    assert not any("0.0.0.0" in u for u in urls)


def test_client_urls_for_a_wildcard_without_addresses_is_loopback_only() -> None:
    assert network.client_urls("0.0.0.0", 8080, []) == ["http://127.0.0.1:8080"]


def test_client_urls_use_the_cached_lan_addresses_by_default() -> None:
    from conftest import FAKE_LAN

    assert network.client_urls("0.0.0.0", 1) == ["http://127.0.0.1:1", f"http://{FAKE_LAN}:1"]


def test_client_urls_for_a_stored_base_url() -> None:
    lan = ["192.168.50.7"]
    assert network.client_urls_for("http://127.0.0.1:8080", "0.0.0.0", lan) == [
        "http://127.0.0.1:8080", "http://192.168.50.7:8080"]
    assert network.client_urls_for("http://127.0.0.1:8080", "", lan) == ["http://127.0.0.1:8080"]
    assert network.client_urls_for("http://10.0.0.9:9000", "", lan) == ["http://10.0.0.9:9000"]
    assert network.client_urls_for("not a url", "", lan) == ["not a url"]
    assert network.client_urls_for("http://x", "", lan) == ["http://x"]       # no port


def test_primary_url_prefers_the_first_lan_url() -> None:
    assert network.primary_url(["http://127.0.0.1:1"]) == "http://127.0.0.1:1"
    assert network.primary_url(["http://127.0.0.1:1", "http://10.0.0.1:1", "http://10.0.0.2:1"]) \
        == "http://10.0.0.1:1"


def test_probe_host_is_loopback_for_wildcards() -> None:
    assert [network.probe_host(h) for h in ("0.0.0.0", "::", "[::]", "", "10.0.0.5", "::1")] == [
        "127.0.0.1", "127.0.0.1", "127.0.0.1", "127.0.0.1", "10.0.0.5", "[::1]"]


# --------------------------------------------------------------------------- WSL

def wsl(release: str = "5.15.153.1-microsoft-standard-WSL2", info: str | None = None,
        ip: str = "") -> str | None:
    return network.wsl_mode(lambda: release, lambda: info, lambda: ip)


def test_wsl_mode_is_none_off_wsl() -> None:
    assert wsl("6.8.0-45-generic", "mirrored") is None


@pytest.mark.parametrize("info, mode", [("mirrored\n", "mirrored"), ("nat", "nat"), ("NAT\n", "nat"),
                                        ("virtioproxy\n", "unknown")])
def test_wsl_mode_from_wslinfo(info: str, mode: str) -> None:
    assert wsl(info=info) == mode


def test_wsl_mode_falls_back_on_the_eth0_address() -> None:
    nat = "2: eth0    inet 172.20.1.2/20 scope global eth0\n"
    mirrored = "2: eth0    inet 192.168.50.7/24 scope global eth0\n3: eth1    inet 10.0.0.2/24 scope global eth1\n"
    assert wsl(ip=nat) == "nat"
    assert wsl(ip="2: eth1    inet 172.20.1.2/20 scope global eth1\n") == "unknown"   # not eth0
    assert wsl(ip=mirrored) == "unknown" and wsl(ip="") == "unknown"
    assert wsl(release="5.15-MICROSOFT-standard", ip=nat) == "nat"


def test_wsl_mode_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(network, "read_osrelease", lambda: calls.append(1) or "5.15-microsoft")
    monkeypatch.setattr(network, "read_wslinfo", lambda: "mirrored")
    network.reset_cache()
    assert network.wsl_mode() == "mirrored" and network.wsl_mode() == "mirrored"
    assert len(calls) == 1


def test_exposure_notes() -> None:
    assert network.exposure_notes("127.0.0.1", lambda: "nat") == []
    assert network.exposure_notes("::1", lambda: "mirrored") == []
    assert network.exposure_notes("0.0.0.0", lambda: None) == []
    nat = network.exposure_notes("0.0.0.0", lambda: "nat")
    assert len(nat) == 1 and "mirrored" in nat[0] and "portproxy" in nat[0]
    mirrored = network.exposure_notes("0.0.0.0", lambda: "mirrored")
    assert len(mirrored) == 1 and "Hyper-V firewall" in mirrored[0] and "port" in mirrored[0]
    unknown = network.exposure_notes("192.168.50.7", lambda: "unknown")
    assert len(unknown) == 1 and "could not be detected" in unknown[0]


def test_exposure_notes_default_to_the_detected_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(network, "read_osrelease", lambda: "5.15-microsoft-standard-WSL2")
    monkeypatch.setattr(network, "read_wslinfo", lambda: "nat")
    network.reset_cache()
    assert "NAT" in network.exposure_notes("0.0.0.0")[0]
