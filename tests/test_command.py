"""build_command argv construction."""
from __future__ import annotations

from llamactl.command import build_command, command_string
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
