"""Calibration read path: trial records, factor maths, clamping, bad files."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from llamactl import calibration
from llamactl.calibration import Calibration, Trial
from llamactl.config import Preset, Profile
from llamactl.measure import Measurement


def _trial(vram: float, est: int = 10_000, backend: str = "cuda", verdict: str = "GPU",
           ok: bool = True) -> Trial:
    return Trial(model="/m.gguf", mtime=1.0, intent="chat", profile={"ctx_size": 8192},
                 estimated_total=est, verdict=verdict, backend=backend,
                 measurement=Measurement(ok=ok, vram_used=int(vram), gen_tps=50.0))


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


# --------------------------------------------------------------------------- write path

def test_record_save_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "tuning.json"
    cal = Calibration()
    cal.record(_trial(12_000))
    cal.record(_trial(14_000, backend="vulkan"))
    cal.save(path)
    raw = json.loads(path.read_text())
    assert raw["version"] == 1 and len(raw["trials"]) == 2
    back = Calibration.load(path)
    assert back.trials == cal.trials
    assert back.factor("cuda") == pytest.approx(1.2) and back.factor("vulkan") == pytest.approx(1.4)


def test_save_defaults_to_the_tuning_file(isolated: SimpleNamespace) -> None:
    cal = Calibration([_trial(13_000)])
    cal.save()
    assert (isolated.cfg_dir / "tuning.json").is_file()
    assert Calibration.load().factor("cuda") == pytest.approx(1.3)


def test_save_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed write leaves the old file untouched and no temp file behind."""
    path = tmp_path / "tuning.json"
    Calibration([_trial(12_000)]).save(path)
    before = path.read_text()

    def boom(src: str, dst: str) -> None:
        raise OSError("disk gone")

    monkeypatch.setattr(calibration.os, "replace", boom)
    with pytest.raises(OSError):
        Calibration([_trial(12_000), _trial(14_000)]).save(path)
    assert path.read_text() == before
    assert [p.name for p in tmp_path.iterdir()] == ["tuning.json"]


def test_save_goes_through_a_temp_file_in_the_same_folder(tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[Path, Path]] = []
    real = os.replace
    monkeypatch.setattr(calibration.os, "replace",
                        lambda a, b: (seen.append((Path(a), Path(b))), real(a, b))[1])
    Calibration([_trial(12_000)]).save(tmp_path / "tuning.json")
    (src, dst), = seen
    assert src.parent == dst.parent == tmp_path and src != dst


def test_trials_for_filters_by_model_and_intent() -> None:
    a, b = _trial(12_000), _trial(13_000)
    b = Trial(**{**b.__dict__, "model": "/other.gguf", "intent": "agent"})
    cal = Calibration([a, b])
    assert cal.trials_for("/m.gguf") == [a]
    assert cal.trials_for("/other.gguf", "agent") == [b]
    assert cal.trials_for("/other.gguf", "chat") == [] and cal.trials_for("/none") == []


def _tuned(build: str = "b1", at: str = "t1", source: str = "tuned") -> Preset:
    return Preset("chat", Profile(), source, Measurement(ok=True, build=build, at=at))


def _stale_setup(tmp_path: Path, mtime: float | None = None) -> tuple[Calibration, Path]:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    t = _trial(12_000)
    t = Trial(**{**t.__dict__, "model": str(model), "mtime": mtime or model.stat().st_mtime,
                 "measurement": Measurement(ok=True, vram_used=12_000, build="b1", at="t1")})
    return Calibration([t]), model


def test_fresh_preset_is_not_stale(tmp_path: Path) -> None:
    cal, model = _stale_setup(tmp_path)
    assert cal.is_stale(_tuned(), model, "b1") is False


def test_new_build_makes_a_preset_stale(tmp_path: Path) -> None:
    cal, model = _stale_setup(tmp_path)
    assert cal.is_stale(_tuned(), model, "b2") is True
    assert cal.is_stale(_tuned(), model, "") is False          # unknown build: cannot tell
    assert cal.is_stale(_tuned(build=""), model, "b2") is False


def test_changed_model_file_makes_a_preset_stale(tmp_path: Path) -> None:
    cal, model = _stale_setup(tmp_path)
    os.utime(model, (1_000_000, 1_000_000))
    assert cal.is_stale(_tuned(), model, "b1") is True


def test_untuned_or_unmeasured_presets_are_never_stale(tmp_path: Path) -> None:
    cal, model = _stale_setup(tmp_path)
    assert cal.is_stale(_tuned(source="estimated"), model, "b2") is False
    assert cal.is_stale(Preset("chat", Profile(), "tuned", None), model, "b2") is False


def test_stale_check_tolerates_missing_trial_or_file(tmp_path: Path) -> None:
    cal, model = _stale_setup(tmp_path)
    assert cal.is_stale(_tuned(at="other"), model, "b1") is False     # no matching trial
    assert Calibration().is_stale(_tuned(), model, "b1") is False
    model.unlink()
    assert cal.is_stale(_tuned(), model, "b1") is False
