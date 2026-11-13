"""Measurement: JSON round trip, tolerant loading, failure kinds."""
from __future__ import annotations

import json

import pytest

from llamactl.measure import ERR_ARGS, ERR_OOM, ERR_TIMEOUT, Measurement

FULL = Measurement(ok=True, error="", load_s=12.5, vram_used=9_500_000_000, ram_used=400_000_000,
                   prompt_tps=1800.5, gen_tps=61.25, at="2026-11-09T10:00:00+00:00",
                   build="0.5.0-dev (build 1, commit 5fc4f3c)")


def test_round_trip_through_json() -> None:
    assert Measurement.from_dict(json.loads(json.dumps(FULL.to_dict()))) == FULL
    assert list(FULL.to_dict()) == ["ok", "error", "load_s", "vram_used", "ram_used",
                                    "prompt_tps", "gen_tps", "at", "build", "effective_tps",
                                    "request_s", "prompt_tokens", "generated_tokens",
                                    "cached_tokens", "workloads"]


def test_unknown_keys_and_missing_keys() -> None:
    m = Measurement.from_dict({"ok": True, "gen_tps": 5, "future": [1], "vram_used": "7"})
    assert m == Measurement(ok=True, gen_tps=5.0, vram_used=7)
    assert Measurement.from_dict({}) == Measurement()


def test_bad_values_take_defaults() -> None:
    m = Measurement.from_dict({"ok": "yes", "error": 3, "load_s": "fast", "vram_used": None,
                               "gen_tps": True, "at": [], "build": {}})
    assert m == Measurement()


@pytest.mark.parametrize("junk", [None, [], "x", 3])
def test_non_dict_is_rejected(junk) -> None:
    with pytest.raises(ValueError):
        Measurement.from_dict(junk)


def test_resource_failure_kinds() -> None:
    assert Measurement(error=f"{ERR_OOM}: cudaMalloc failed").resource_failure
    assert Measurement(error=f"{ERR_TIMEOUT}: no /health after 90 s").resource_failure
    assert not Measurement(error=f"{ERR_ARGS}: --nope").resource_failure
    assert not Measurement(error="load failed: unknown model architecture").resource_failure
    assert not Measurement(ok=True, error=ERR_OOM).resource_failure
    assert not Measurement().resource_failure
