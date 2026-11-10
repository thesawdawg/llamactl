"""Registry invariants and validate() behaviour."""
from __future__ import annotations

import dataclasses
import re

import pytest

from llamactl import schema
from llamactl.config import Profile


def test_keys_unique() -> None:
    keys = [s.key for s in schema.SETTINGS]
    assert len(keys) == len(set(keys))


def test_profile_fields_match_schema() -> None:
    assert {f.name for f in dataclasses.fields(Profile)} == set(schema.by_key)


def test_profile_defaults_match_schema() -> None:
    for f in dataclasses.fields(Profile):
        assert f.default == schema.by_key[f.name].default


@pytest.mark.parametrize("raw,want", [("4096", 4096), ("0", 0), ("131072", 131072)])
def test_validate_int(raw: str, want: int) -> None:
    assert schema.validate(schema.by_key["ctx_size"], raw) == want


@pytest.mark.parametrize("raw", ["abc", "4.5", ""])
def test_validate_int_bad(raw: str) -> None:
    with pytest.raises(ValueError):
        schema.validate(schema.by_key["ctx_size"], raw)


@pytest.mark.parametrize("raw,want", [("all", "all"), ("auto", "auto"), ("24", "24"), ("0", "0")])
def test_validate_gpu_layers(raw: str, want: str) -> None:
    assert schema.validate(schema.by_key["gpu_layers"], raw) == want


@pytest.mark.parametrize("raw", ["many", "-1", "1.5"])
def test_validate_gpu_layers_bad(raw: str) -> None:
    with pytest.raises(ValueError):
        schema.validate(schema.by_key["gpu_layers"], raw)


def test_validate_choice() -> None:
    assert schema.validate(schema.by_key["cache_type_k"], "q8_0") == "q8_0"
    with pytest.raises(ValueError):
        schema.validate(schema.by_key["cache_type_k"], "q3_k")


def test_validate_bool() -> None:
    assert schema.validate(schema.by_key["metrics"], "true") is True
    assert schema.validate(schema.by_key["metrics"], "0") is False
    with pytest.raises(ValueError):
        schema.validate(schema.by_key["metrics"], "maybe")


def test_validate_min_max() -> None:
    with pytest.raises(ValueError):
        schema.validate(schema.by_key["temp"], "3")
    with pytest.raises(ValueError):
        schema.validate(schema.by_key["top_p"], "1.5")
    assert schema.validate(schema.by_key["top_p"], "0") == 0


def test_no_launch_modes() -> None:
    assert not hasattr(schema, "for_mode")
    assert not any(hasattr(s, "modes") for s in schema.SETTINGS)


NEW_KEYS = ("fit", "fit_target", "device", "cache_reuse", "sleep_idle", "embedding", "pooling",
            "threads_batch")


def test_new_settings_present() -> None:
    for key in NEW_KEYS:
        s = schema.by_key[key]
        assert s.flag and s.hint and s.help and f"Flag: {s.flag}" in s.help, key


def test_preset_key_for_every_flagged_setting() -> None:
    seen: set[str] = set()
    for s in schema.SETTINGS:
        if not s.flag:
            assert s.preset_key == "", s.key
            continue
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", s.preset_key), s.key
        assert s.preset_key not in seen, s.key
        seen.add(s.preset_key)
    assert schema.by_key["ctx_size"].preset_key == "ctx-size"
    assert schema.by_key["gpu_layers"].preset_key == "n-gpu-layers"
    assert schema.by_key["sleep_idle"].preset_key == "sleep-idle-seconds"


def test_router_owned() -> None:
    assert {s.key for s in schema.SETTINGS if s.router_owned} == {"host", "port", "api_key", "alias"}


def test_default_on_bools_have_negation() -> None:
    for s in schema.SETTINGS:
        if s.kind == schema.Kind.BOOL and s.default:
            assert s.negate_flag, s.key


def test_pooling_choices_allow_unset() -> None:
    s = schema.by_key["pooling"]
    assert s.default == "" and s.unset == ""
    assert schema.validate(s, "") == "" and schema.validate(s, "rank") == "rank"
    with pytest.raises(ValueError):
        schema.validate(s, "max")
    assert schema.validate(schema.by_key["fit"], "off") == "off"
    assert schema.validate(schema.by_key["embedding"], "true") is True
