"""Discover's weights-only fit verdicts and the best-fit marker."""
from __future__ import annotations

from fakes import make_host
from llamactl import discover
from llamactl.hardware import Host


def f(name: str, gb: float, parts: int = 1, quant: str = "") -> dict:
    return {"path": name, "size": int(gb * 1e9), "parts": parts, "quant": quant}


def by_path(fits: list[discover.FileFit]) -> dict[str, discover.FileFit]:
    return {x.path: x for x in fits}


def test_the_largest_quant_that_fits_the_gpu_is_marked_and_sorted_first_among_fits() -> None:
    files = [f("m-Q4_K_M.gguf", 5, quant="Q4_K_M"), f("m-Q8_0.gguf", 9, quant="Q8_0"),
             f("m-BF16.gguf", 17, quant="BF16"), f("m-Q6_K.gguf", 7, quant="Q6_K"),
             f("m-Q5_K_M.gguf", 6, quant="Q5_K_M")]
    got = discover.rank_files(files, make_host(16e9))
    assert [x.path for x in got] == ["m-BF16.gguf", "m-Q8_0.gguf", "m-Q6_K.gguf",
                                     "m-Q5_K_M.gguf", "m-Q4_K_M.gguf"]       # largest first
    assert [x.verdict for x in got] == ["GPU+CPU", "GPU", "GPU", "GPU", "GPU"]
    assert [x.path for x in got if x.best] == ["m-Q8_0.gguf"]
    assert got[1].label == "GPU (weights only)" and got[0].label == "GPU+CPU (weights only)"


def test_nothing_fitting_the_gpu_marks_nothing() -> None:
    got = discover.rank_files([f("a.gguf", 30), f("b.gguf", 80)], make_host(16e9))
    assert not any(x.best for x in got)
    assert by_path(got)["b.gguf"].verdict == "NO" and by_path(got)["b.gguf"].label == "NO"
    assert by_path(got)["a.gguf"].verdict == "GPU+CPU"


def test_the_headroom_decides_what_fits() -> None:
    files = [f("big.gguf", 13.5)]
    assert discover.rank_files(files, make_host(16e9), 10)[0].best          # 14.4 GB usable
    assert not discover.rank_files(files, make_host(16e9), 25)[0].best      # 12 GB usable


def test_without_a_gpu_sizes_are_judged_against_ram_and_nothing_is_marked() -> None:
    host = Host(ram_total=int(16e9), ram_free=int(12e9), cpu_threads=8, cpu_cores=4)
    got = discover.rank_files([f("s.gguf", 4), f("l.gguf", 20)], host)
    assert [(x.path, x.verdict) for x in got] == [("l.gguf", "NO"), ("s.gguf", "CPU")]
    assert got[1].label == "CPU (weights only)" and not any(x.best for x in got)
    assert "No GPU found" in discover.legend(host, "Chat")


def test_sharded_models_are_one_entry_with_the_summed_size() -> None:
    # hf.list_gguf already folds shards into the first shard with the total size and part count
    files = [f("big/m-Q8_0-00001-of-00003.gguf", 15, parts=3, quant="Q8_0"),
             f("m-Q4_K_M.gguf", 6, quant="Q4_K_M")]
    got = discover.rank_files(files, make_host(16e9))
    assert [(x.path, x.parts, x.verdict, x.best) for x in got] == [
        ("big/m-Q8_0-00001-of-00003.gguf", 3, "GPU+CPU", False),
        ("m-Q4_K_M.gguf", 1, "GPU", True)]


def test_ties_zero_sizes_and_empty_lists() -> None:
    got = discover.rank_files([f("b.gguf", 5), f("a.gguf", 5), f("z.gguf", 0)], make_host(16e9))
    assert [x.path for x in got] == ["a.gguf", "b.gguf", "z.gguf"]
    assert [x.best for x in got] == [True, False, False]                    # a size of 0 is never best
    assert discover.rank_files([], make_host(16e9)) == []


def test_before_the_host_is_known_nothing_is_marked() -> None:
    host = Host()
    assert not discover.host_ready(host)
    assert not any(x.best for x in discover.rank_files([f("a.gguf", 1)], host))
    assert "Detecting" in discover.legend(host, "Chat")


def test_legend_names_the_marker_and_the_intent() -> None:
    text = discover.legend(make_host(16e9), "Agent")
    assert "best fit" in text and "Agent" in text and "KV cache" in text
