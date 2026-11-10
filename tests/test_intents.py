"""Intent table values and applicability."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from llamactl import intents, schema
from llamactl.gguf import ModelInfo


def _info(pooling: int = -1) -> ModelInfo:
    return ModelInfo(path=Path("m.gguf"), size=1, architecture="x", name="m", n_layer=4,
                     n_embd=64, n_head=4, n_head_kv=4, head_dim_k=16, head_dim_v=16,
                     ctx_train=8192, expert_count=0, file_type=0, sharded=1,
                     parameter_count=0, pooling_type=pooling)


# key: (ctx_min, ctx_target, ctx_max, slots, kv_quant, spill, extra); 0 = trained
TABLE = {
    "chat": (8192, 16384, 32768, 1, "if_needed", "avoid", {}),
    "agent": (32768, 65536, 131072, 1, "prefer", "allow", {"cache_reuse": 256}),
    "context": (32768, 0, 0, 1, "prefer", "allow", {}),
    "speed": (4096, 8192, 16384, 1, "never", "avoid", {}),
    "api": (4096, 8192, 16384, 4, "if_needed", "avoid", {"metrics": True}),
    "embed": (0, 0, 0, 4, "never", "avoid", {"embedding": True}),
}


def test_six_intents_in_order() -> None:
    assert [i.key for i in intents.INTENTS] == list(TABLE)
    assert set(intents.by_key) == set(TABLE)


@pytest.mark.parametrize("key", list(TABLE))
def test_table_values(key: str) -> None:
    i = intents.by_key[key]
    got = (i.ctx_min, i.ctx_target, i.ctx_max, i.slots, i.kv_quant, i.spill, i.extra)
    assert got == TABLE[key]
    assert i.label and i.description


def test_extras_are_schema_keys_and_intents_are_frozen() -> None:
    for i in intents.INTENTS:
        assert set(i.extra) <= set(schema.by_key)
        with pytest.raises(dataclasses.FrozenInstanceError):
            i.slots = 9  # type: ignore[misc]
        hash(i)


def test_applicability() -> None:
    gen = [i.key for i in intents.applicable(_info())]
    assert gen == ["chat", "agent", "context", "speed", "api"]
    assert [i.key for i in intents.applicable(_info(pooling=1))] == ["embed"]
    assert [i.key for i in intents.applicable(_info(pooling=0))] == gen  # pooling "none"


def test_resolve_prefers_key_then_first() -> None:
    assert intents.resolve("api", _info()).key == "api"
    assert intents.resolve("embed", _info()).key == "chat"
    assert intents.resolve("chat", _info(pooling=2)).key == "embed"
    assert intents.resolve("nope", _info()).key == "chat"
