"""Manual tensor placement: parse ``-ot`` override lists and simulate their CPU bytes.

llama.cpp splits ``--override-tensor`` on commas into PATTERN=BUFFER rules; each tensor is
placed by the first rule whose pattern is found in its name (regex search), and
``--n-cpu-moe N`` is just N generated rules for the first N layers' expert tensors
(common/common.h). Simulating the same order over a GGUF tensor table tells the user what
their patterns actually move before they launch anything.
"""
from __future__ import annotations

import re


def parse_overrides(text: str) -> list[tuple[str, str]]:
    """Parse an ``-ot`` value into (pattern, buffer) rules.

    Split on "," like llama.cpp, then at the last "=" so a pattern may contain "=".
    Empty text means no overrides.

    Args:
        text: The raw ``override_tensor`` value.

    Returns:
        The rules, in order.

    Raises:
        ValueError: On an empty pattern or buffer, or a pattern Python's ``re`` cannot
            compile (llama.cpp uses its own C++ regex; this check is close, not identical).
    """
    rules = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        pattern, sep, buffer = item.rpartition("=")
        if not sep or not pattern.strip():
            raise ValueError(f"'{item}' has no pattern: write PATTERN=BUFFER")
        if not buffer.strip():
            raise ValueError(f"'{item}' has no buffer: write PATTERN=BUFFER (for example =CPU)")
        try:
            re.compile(pattern.strip())
        except re.error as e:
            raise ValueError(f"'{pattern.strip()}' is not a valid pattern: {e}") from e
        rules.append((pattern.strip(), buffer.strip()))
    return rules


def cpu_bytes(tensors: list[tuple[str, int]], overrides: list[tuple[str, str]],
              n_cpu_moe: int) -> tuple[int, int]:
    """Bytes the user's placement puts on the CPU, simulating llama.cpp's rule order.

    Rules are the ``-ot`` overrides first (their first match wins), then for each
    ``i < n_cpu_moe`` the expert-tensor pattern llama.cpp generates for layer i. A buffer
    named ``cpu`` (any case) counts as RAM; the first matching rule decides a tensor.

    Args:
        tensors: (name, size) pairs of every tensor (all shards).
        overrides: Parsed `-ot` rules (`parse_overrides`).
        n_cpu_moe: ``--n-cpu-moe`` count.

    Returns:
        (tensors matched by the user overrides, total bytes placed on the CPU by any rule).
    """
    rules = [(re.compile(p), b) for p, b in overrides]
    rules += [(re.compile(rf"blk\.{i}\.ffn_(up|down|gate|gate_up)_(ch|)exps"), "CPU")
              for i in range(n_cpu_moe)]
    matched = total = 0
    for name, size in tensors:
        for j, (pattern, buffer) in enumerate(rules):
            if pattern.search(name):
                if j < len(overrides):
                    matched += 1
                if buffer.lower() == "cpu":
                    total += size
                break
    return matched, total
