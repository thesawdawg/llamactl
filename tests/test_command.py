"""build_command argv construction."""
from __future__ import annotations

from llamactl.command import build_command, command_string
from llamactl.config import Config, Profile

MODEL = "/models/m.gguf"


def argv(prof: Profile, mode: str = "server", port: int = 8080) -> list[str]:
    return build_command(Config(bin_dir="/bin"), MODEL, prof, mode, port)


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


def test_cli_mode() -> None:
    cmd = argv(Profile(parallel=4, port=1234), mode="cli")
    assert cmd[0] == "/bin/llama-cli"
    assert "--host" not in cmd and "--port" not in cmd and "-np" not in cmd


def test_command_string() -> None:
    assert command_string(["a b", "c"]) == "'a b' c"
