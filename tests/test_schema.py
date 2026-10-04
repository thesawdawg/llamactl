"""Registry invariants and validate() behaviour."""
from __future__ import annotations

import dataclasses

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


def test_modes() -> None:
    assert {s.key for s in schema.for_mode("server")} == set(schema.by_key)
    cli = {s.key for s in schema.for_mode("cli")}
    assert "port" not in cli and "parallel" not in cli and "ctx_size" in cli
