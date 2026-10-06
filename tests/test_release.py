"""Regression tests for release metadata and tag validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_release import check_release


@pytest.mark.parametrize("version", ["0.1.0", "1.0.0", "10.20.30"])
def test_valid_release(tmp_path: Path, version: str) -> None:
    """Accept matching stable tags and dated changelog entries.

    Args:
        tmp_path: Isolated repository directory.
        version: Stable Semantic Version to validate.

    Returns:
        None.
    """
    (tmp_path / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n')
    (tmp_path / "CHANGELOG.md").write_text(f"## [{version}] - 2026-10-03\n")
    assert check_release(tmp_path, f"v{version}") == version


@pytest.mark.parametrize("version,tag,heading", [
    ("0.1.0", "v0.2.0", "## [0.1.0] - 2026-10-03"),
    ("0.1.0", "0.1.0", "## [0.1.0] - 2026-10-03"),
    ("0.1.0", "v0.1.0", "## [0.2.0] - 2026-10-03"),
    ("0.1.0", "v0.1.0", "## [0.1.0] - TBD"),
    ("01.1.0", "v01.1.0", "## [01.1.0] - 2026-10-03"),
    ("0.1.0.post1", "v0.1.0.post1", "## [0.1.0.post1] - 2026-10-03"),
    ("0.1.0rc1", "v0.1.0rc1", "## [0.1.0rc1] - 2026-10-03"),
])
def test_invalid_release(tmp_path: Path, version: str, tag: str, heading: str) -> None:
    """Reject mismatched tags, missing notes and unsupported version formats.

    Args:
        tmp_path: Isolated repository directory.
        version: Package metadata version.
        tag: Candidate release tag.
        heading: Candidate changelog heading.

    Returns:
        None.
    """
    (tmp_path / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n')
    (tmp_path / "CHANGELOG.md").write_text(heading + "\n")
    with pytest.raises(ValueError):
        check_release(tmp_path, tag)


def test_repository_release() -> None:
    """Verify that the initial release metadata and changelog agree.

    Args:
        None.

    Returns:
        None.
    """
    assert check_release(Path(__file__).resolve().parents[1], "v0.1.0") == "0.1.0"
