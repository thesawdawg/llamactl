"""The Home view's next-step suggestions: one test per rule, plus ordering."""
from __future__ import annotations

from pathlib import Path

import pytest

from fakes import make_info
from llamactl import suggest
from llamactl.calibration import Calibration, Trial
from llamactl.config import ESTIMATED, Config, ModelEntry, Preset, Profile
from llamactl.gateway import GatewayStatus
from llamactl.measure import Measurement
from llamactl.suggest import Suggestion, next_steps

STOPPED = GatewayStatus(False, "http://127.0.0.1:8080")
RUNNING = GatewayStatus(True, "http://127.0.0.1:8080", True, 7)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """A config whose llama.cpp folder holds a llama-server."""
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "llama-server").touch()
    return Config(bin_dir=str(tmp_path / "bin"), model_dirs=[str(tmp_path / "models")])


def _model(tmp_path: Path, name: str = "qwen3.gguf") -> Path:
    p = tmp_path / name
    p.write_bytes(b"x")
    return p


def _add(cfg: Config, model: Path, source: str = ESTIMATED, name: str = "chat",
         measured: Measurement | None = None) -> None:
    cfg.set_preset(str(model), Preset(name, Profile(), source, measured))


def _keys(steps: list[Suggestion]) -> list[str]:
    return [s.key.split(":")[0] for s in steps]


def test_nothing_to_do(cfg: Config, tmp_path: Path) -> None:
    m = _model(tmp_path)
    _add(cfg, m, "edited")
    assert next_steps(cfg, [m], {str(m): make_info(m)}, Calibration(), RUNNING, "") == []


def test_missing_binary_points_to_settings(cfg: Config, tmp_path: Path) -> None:
    cfg.bin_dir = str(tmp_path / "nope")
    m = _model(tmp_path)
    _add(cfg, m)
    steps = next_steps(cfg, [m], {str(m): make_info(m)}, Calibration(), STOPPED, "")
    assert steps[0].action == suggest.ACT_SETTINGS and "llama-server not found" in steps[0].text
    # tuning and starting need the binary: not suggested
    assert _keys(steps) == ["binary"]


def test_no_models_points_to_discover(cfg: Config) -> None:
    steps = next_steps(cfg, [], {}, Calibration(), STOPPED, "")
    assert _keys(steps) == ["no-models"] and steps[0].action == suggest.ACT_DISCOVER
    assert cfg.model_dirs[0] in steps[0].text


def test_no_models_and_no_binary_lists_both_in_order(cfg: Config, tmp_path: Path) -> None:
    cfg.bin_dir = str(tmp_path / "nope")
    assert _keys(next_steps(cfg, [], {}, Calibration(), None, "")) == ["binary", "no-models"]


def test_models_without_presets_point_to_models(cfg: Config, tmp_path: Path) -> None:
    a, b = _model(tmp_path, "a.gguf"), _model(tmp_path, "b.gguf")
    _add(cfg, b, "edited")
    steps = next_steps(cfg, [a, b], {}, Calibration(), RUNNING, "")
    assert _keys(steps) == ["no-presets"]
    assert steps[0].action == suggest.ACT_MODELS and steps[0].model == str(a)
    assert "a" in steps[0].text


def test_estimated_active_preset_suggests_tune(cfg: Config, tmp_path: Path) -> None:
    m = _model(tmp_path)
    _add(cfg, m)
    steps = next_steps(cfg, [m], {str(m): make_info(m)}, Calibration(), RUNNING, "")
    assert _keys(steps) == ["tune"]
    s = steps[0]
    assert (s.action, s.model, s.preset) == (suggest.ACT_TUNE, str(m), "chat")


def test_only_the_active_estimate_counts_and_unreadable_models_are_skipped(
        cfg: Config, tmp_path: Path) -> None:
    m, n = _model(tmp_path, "a.gguf"), _model(tmp_path, "b.gguf")
    _add(cfg, m, "edited", "chat")
    _add(cfg, m, ESTIMATED, "agent")      # estimated but not active: not worth a nag
    cfg.models[str(m)].active = "chat"
    _add(cfg, n)
    assert next_steps(cfg, [m, n], {str(m): make_info(m)}, Calibration(), RUNNING, "") == []


def test_stale_tuned_preset_suggests_retune(cfg: Config, tmp_path: Path) -> None:
    m = _model(tmp_path)
    meas = Measurement(ok=True, gen_tps=30.0, at="t1", build="old-build")
    _add(cfg, m, "tuned", "agent", meas)
    steps = next_steps(cfg, [m], {str(m): make_info(m)}, Calibration(), RUNNING, "new-build")
    assert _keys(steps) == ["retune"]
    assert steps[0].preset == "agent" and steps[0].action == suggest.ACT_TUNE
    # same build, no trial to compare the file time with: not stale
    assert next_steps(cfg, [m], {str(m): make_info(m)}, Calibration(), RUNNING, "old-build") == []


def test_stale_by_changed_model_file(cfg: Config, tmp_path: Path) -> None:
    m = _model(tmp_path)
    meas = Measurement(ok=True, gen_tps=30.0, at="t1", build="b")
    _add(cfg, m, "tuned", "chat", meas)
    cal = Calibration([Trial(str(m), m.stat().st_mtime - 100, "chat", {}, 1, "GPU", "cuda", meas)])
    steps = next_steps(cfg, [m], {str(m): make_info(m)}, cal, RUNNING, "b")
    assert _keys(steps) == ["retune"]


def test_gateway_stopped_with_presets_suggests_start(cfg: Config, tmp_path: Path) -> None:
    m = _model(tmp_path)
    _add(cfg, m, "edited")
    steps = next_steps(cfg, [m], {str(m): make_info(m)}, Calibration(), STOPPED, "")
    assert _keys(steps) == ["start-gateway"] and steps[0].action == suggest.ACT_START_GATEWAY
    assert "http://127.0.0.1:8080" in steps[0].text
    # unknown status (before the first poll) and a running gateway: no suggestion
    assert next_steps(cfg, [m], {str(m): make_info(m)}, Calibration(), None, "") == []
    assert next_steps(cfg, [m], {str(m): make_info(m)}, Calibration(), RUNNING, "") == []


def test_no_presets_means_nothing_to_start(cfg: Config, tmp_path: Path) -> None:
    m = _model(tmp_path)
    assert _keys(next_steps(cfg, [m], {}, Calibration(), STOPPED, "")) == ["no-presets"]


def test_order_and_per_rule_cap(cfg: Config, tmp_path: Path) -> None:
    models = [_model(tmp_path, f"m{i}.gguf") for i in range(6)]
    for m in models[:4]:
        _add(cfg, m)                      # four estimated actives
    infos = {str(m): make_info(m) for m in models}
    steps = next_steps(cfg, models, infos, Calibration(), STOPPED, "")
    assert _keys(steps) == ["no-presets"] * 2 + ["tune"] * 3 + ["start-gateway"]
    assert len({s.key for s in steps}) == len(steps)
