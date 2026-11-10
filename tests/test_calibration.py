"""Calibration read path: trial records, factor maths, clamping, bad files."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llamactl import calibration
from llamactl.calibration import Calibration, Trial


def _trial(vram: float, est: int = 10_000, backend: str = "cuda", verdict: str = "GPU",
           ok: bool = True) -> Trial:
    return Trial(model="/m.gguf", mtime=1.0, intent="chat", profile={"ctx_size": 8192},
                 estimated_total=est, verdict=verdict, backend=backend,
                 measurement={"ok": ok, "vram_used": vram, "gen_tps": 50.0})


def test_trial_round_trip() -> None:
    t = _trial(11_000)
    assert Trial.from_dict(json.loads(json.dumps(t.to_dict()))) == t
    assert set(t.to_dict()) == {"model", "mtime", "intent", "profile", "estimated_total",
                                "verdict", "backend", "measurement"}


@pytest.mark.parametrize("bad", [None, [], "x", {}, {"model": "m", "profile": []},
                                 {"model": "m", "estimated_total": "lots"}])
def test_trial_from_dict_rejects_junk(bad) -> None:
    with pytest.raises(ValueError):
        Trial.from_dict(bad)


def test_no_data_is_one() -> None:
    assert Calibration().factor("cuda") == 1.0


def test_factor_is_median_of_ratios() -> None:
    cal = Calibration([_trial(11_000), _trial(12_000), _trial(14_000)])
    assert cal.factor("cuda") == pytest.approx(1.2)
    cal = Calibration([_trial(11_000), _trial(12_000), _trial(13_000), _trial(14_000)])
    assert cal.factor("cuda") == pytest.approx(1.25)   # even count: mean of the middle two


def test_clamped_to_range() -> None:
    assert Calibration([_trial(30_000)]).factor("cuda") == calibration.FACTOR_MAX
    assert Calibration([_trial(1_000)]).factor("cuda") == calibration.FACTOR_MIN
    assert Calibration([_trial(8_000)]).factor("cuda") == pytest.approx(0.8)


def test_per_backend_separation() -> None:
    cal = Calibration([_trial(12_000, backend="cuda"), _trial(9_000, backend="vulkan")])
    assert cal.factor("cuda") == pytest.approx(1.2)
    assert cal.factor("vulkan") == pytest.approx(0.9)
    assert cal.factor("metal") == 1.0


def test_only_successful_all_gpu_trials_count() -> None:
    cal = Calibration([_trial(12_000),
                       _trial(15_000, ok=False),             # failed load
                       _trial(15_000, verdict="GPU+CPU"),    # layers on CPU: not comparable
                       _trial(0.0),                           # nothing measured
                       _trial(15_000, est=0)])                # no estimate
    assert cal.factor("cuda") == pytest.approx(1.2)


def test_load_missing_file(tmp_path: Path) -> None:
    assert Calibration.load(tmp_path / "nope.json").factor("cuda") == 1.0


@pytest.mark.parametrize("text", ["", "{not json", "[]", '"x"', '{"trials": 3}', "\x00\x01"])
def test_load_corrupt_file(tmp_path: Path, text: str) -> None:
    p = tmp_path / "tuning.json"
    p.write_text(text)
    cal = Calibration.load(p)
    assert cal.trials == () and cal.factor("cuda") == 1.0


def test_load_skips_bad_records(tmp_path: Path) -> None:
    p = tmp_path / "tuning.json"
    p.write_text(json.dumps({"version": 1, "trials": [
        _trial(12_000).to_dict(), "junk", {"model": "x", "profile": 3}, _trial(12_000).to_dict()]}))
    cal = Calibration.load(p)
    assert len(cal.trials) == 2 and cal.factor("cuda") == pytest.approx(1.2)


def test_default_path_follows_isolated_fixture(isolated: SimpleNamespace) -> None:
    assert calibration.TUNING_FILE == isolated.cfg_dir / "tuning.json"
    isolated.cfg_dir.mkdir()
    (isolated.cfg_dir / "tuning.json").write_text(
        json.dumps({"trials": [_trial(13_000).to_dict()]}))
    assert Calibration.load().factor("cuda") == pytest.approx(1.3)
