"""build_command argv construction."""
from __future__ import annotations

import pytest

from llamactl.command import build_command, command_string, extra_args_items, preset_items
from llamactl.config import Config, Profile

MODEL = "/models/m.gguf"


def argv(prof: Profile, port: int = 8080) -> list[str]:
    return build_command(Config(bin_dir="/bin"), MODEL, prof, port)


def test_default_server_shape() -> None:
    assert argv(Profile()) == ["/bin/llama-server", "-m", MODEL, "-ngl", "all", "-fa", "auto",
                             "--host", "127.0.0.1", "--port", "8080"]


def test_unset_omitted() -> None:
    cmd = argv(Profile())
    for flag in ("-c", "-t", "-b", "-ub", "-np", "-ctk", "-ctv", "--temp", "-s", "--api-key"):
        assert flag not in cmd


def test_set_emitted() -> None:
    cmd = argv(Profile(ctx_size=8192, threads=8, parallel=4, temp=0.7, top_k=0, seed=1,
                       cache_type_k="q8_0", extra_args="--jinja"))
    pairs = dict(zip(cmd, cmd[1:]))
    assert pairs["-c"] == "8192" and pairs["-t"] == "8" and pairs["-np"] == "4"
    assert pairs["--temp"] == "0.7" and pairs["-s"] == "1" and pairs["-ctk"] == "q8_0"
    assert pairs["--top-k"] == "0"  # 0 is a real value, -1 is unset


def test_nocb() -> None:
    cmd = argv(Profile(cont_batching=False))
    assert "-nocb" in cmd and "-cb" not in cmd
    assert "-nocb" not in argv(Profile())


def test_bool_flag() -> None:
    assert "--metrics" in argv(Profile(metrics=True))


def test_extra_args_last() -> None:
    cmd = argv(Profile(ctx_size=4096, extra_args="--jinja --temp 0.9"))
    assert cmd[-3:] == ["--jinja", "--temp", "0.9"]


def test_always_server_binary() -> None:
    cmd = argv(Profile(parallel=4, port=1234), port=1234)
    assert cmd[0] == "/bin/llama-server"
    assert "--host" in cmd and cmd[cmd.index("--port") + 1] == "1234" and "-np" in cmd


def test_command_string() -> None:
    assert command_string(["a b", "c"]) == "'a b' c"


def test_new_flags_emitted() -> None:
    cmd = argv(Profile(fit="off", fit_target=512, device="CUDA0,CUDA1", cache_reuse=256,
                       sleep_idle=600, pooling="mean", threads_batch=12))
    pairs = dict(zip(cmd, cmd[1:]))
    assert pairs["-fit"] == "off" and pairs["-fitt"] == "512" and pairs["-dev"] == "CUDA0,CUDA1"
    assert pairs["--cache-reuse"] == "256" and pairs["--sleep-idle-seconds"] == "600"
    assert pairs["--pooling"] == "mean" and pairs["-tb"] == "12"


def test_new_flags_omitted_by_default() -> None:
    cmd = argv(Profile())
    for flag in ("-fit", "-fitt", "-dev", "--cache-reuse", "--sleep-idle-seconds", "--pooling",
                 "-tb", "--embedding"):
        assert flag not in cmd


def test_embedding_bool_flag() -> None:
    cmd = argv(Profile(embedding=True, metrics=True))
    assert "--embedding" in cmd and "--metrics" in cmd
    assert "--embedding" not in argv(Profile(embedding=False))


def test_bool_handling_is_generic() -> None:
    """Default-off BOOLs emit their flag when True; default-on BOOLs emit the negation when False."""
    from llamactl import schema
    for s in schema.SETTINGS:
        if s.kind != schema.Kind.BOOL:
            continue
        flipped = not s.default
        cmd = argv(Profile(**{s.key: flipped}))
        want = s.negate_flag if s.default else s.flag
        assert want and want in cmd
        assert (s.flag if s.default else s.negate_flag) not in cmd
        assert want not in argv(Profile())


# --------------------------------------------------------------------------- preset_items

def items(prof: Profile) -> dict[str, str]:
    return dict(preset_items(prof))


def test_preset_items_default_profile() -> None:
    """Only non-unset settings: gpu_layers 'all' and flash_attn 'auto' are not unset; host/port are owned."""
    assert preset_items(Profile()) == [("n-gpu-layers", "all"), ("flash-attn", "auto")]


def test_preset_items_skip_unset_and_router_owned() -> None:
    d = items(Profile(ctx_size=8192, parallel=4, host="0.0.0.0", port=9999, api_key="k",
                      alias="x", threads=8))
    assert d["ctx-size"] == "8192" and d["parallel"] == "4" and d["threads"] == "8"
    for owned in ("host", "port", "api-key", "alias"):
        assert owned not in d
    for unset in ("batch-size", "temp", "seed", "device", "mmproj", "pooling", "metrics"):
        assert unset not in d


def test_preset_items_zero_is_a_value_only_where_unset_is_not_zero() -> None:
    d = items(Profile(top_k=0, seed=0, temp=0.0))
    assert d["top-k"] == "0" and d["seed"] == "0" and d["temp"] == "0.0"


def test_preset_items_booleans_both_ways() -> None:
    """True writes 'true'; the default-on cont_batching writes 'false' (llama.cpp adds --no-)."""
    d = items(Profile(metrics=True, embedding=True, cont_batching=False))
    assert d["metrics"] == "true" and d["embedding"] == "true"
    assert d["cont-batching"] == "false" and "no-cont-batching" not in d
    assert "cont-batching" not in items(Profile())     # default on: nothing to write
    assert "metrics" not in items(Profile(metrics=False))


def test_preset_items_paths_and_choices() -> None:
    d = items(Profile(mmproj="/m/proj.gguf", draft_model="/m/draft.gguf", fit="off",
                      fit_target=512, device="CUDA0", pooling="mean", sleep_idle=60))
    assert d["mmproj"] == "/m/proj.gguf" and d["spec-draft-model"] == "/m/draft.gguf"
    assert (d["fit"], d["fit-target"], d["device"]) == ("off", "512", "CUDA0")
    assert (d["pooling"], d["sleep-idle-seconds"]) == ("mean", "60")


def test_preset_items_keys_come_from_the_registry() -> None:
    from llamactl import schema
    every = Profile(**{s.key: (not s.default if s.kind == schema.Kind.BOOL else
                               (s.choices[-1] if s.choices else
                                ("x" if s.kind in (schema.Kind.STR, schema.Kind.PATH) else 7)))
                       for s in schema.SETTINGS if s.key != "extra_args"})
    keys = [k for k, _ in preset_items(every)]
    want = [s.preset_key for s in schema.SETTINGS if s.preset_key and not s.router_owned]
    assert keys == want and len(set(keys)) == len(keys)


def test_extra_args_mapping() -> None:
    p = Profile(extra_args="--jinja --chat-template-kwargs '{\"a\": 1}' --no-mmap -ngl=20 "
                           "--temp -1 --ctx-checkpoints 4")
    assert extra_args_items(p.extra_args) == [
        ("jinja", "true"), ("chat-template-kwargs", '{"a": 1}'), ("no-mmap", "true"),
        ("ngl", "20"), ("temp", "-1"), ("ctx-checkpoints", "4")]
    d = items(p)
    assert d["jinja"] == "true" and d["chat-template-kwargs"] == '{"a": 1}'
    assert d["no-mmap"] == "true" and d["ctx-checkpoints"] == "4" and d["temp"] == "-1"


def test_extra_args_replace_registry_values_for_the_same_option() -> None:
    out = preset_items(Profile(temp=0.7, ctx_size=4096, extra_args="--temp 0.9 -c 8192"))
    d = dict(out)
    assert d["temp"] == "0.9" and d["ctx-size"] == "8192"
    assert [k for k, _ in out].count("temp") == 1 and [k for k, _ in out].count("ctx-size") == 1


@pytest.mark.parametrize("extra,needle", [
    ("stray --jinja", "'stray'"),
    ("--jinja 3 4", "'4'"),
    ("--temp 1 --temp 2", "twice"),
    ("--bad=1;2", "';'"),
    ("--k 'a # b'", "'#'"),
    ("'unterminated", "extra_args"),
    ("-- x", "not a valid option name"),
])
def test_extra_args_unmappable(extra: str, needle: str) -> None:
    with pytest.raises(ValueError, match=needle):
        preset_items(Profile(extra_args=extra))


def test_value_with_comment_character_is_refused() -> None:
    with pytest.raises(ValueError, match="mmproj"):
        preset_items(Profile(mmproj="/m/a#1.gguf"))
